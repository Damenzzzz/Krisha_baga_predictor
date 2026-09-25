"""Контракт данных объявлений.

Парсер может отдавать колонки как угодно — к одной схеме всё приводится здесь.
Остальной код (поиск, модель цены) знает только эту схему. Когда придут реальные
данные, правится RENAME, остальное должно заработать само.
"""
import re
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import ART_DIR, LISTINGS_PATH, LISTINGS_RAW

REQUIRED = ["listing_id", "price", "area", "rooms", "city"]
NUMERIC = ["price", "area", "rooms", "floor", "floors_total", "year_built",
           "kitchen_area", "ceiling_height", "lat", "lon"]
CATEGORICAL = ["city", "district", "building_type", "condition", "bathroom", "complex_name"]
TEXT = ["title", "address", "description", "url"]

# "как колонка называется у парсера": "как у нас". Сейчас — схема KrishaParser (krisha.db).
RENAME = {"id": "listing_id", "price_kzt": "price", "area_total": "area",
          "area_kitchen": "kitchen_area", "renovation_text": "condition"}

# Санитарные границы цены за м², отдельно для продажи и аренды (аренда — ₸/м² в месяц).
# Аренда: по krisha.db медиана 6 000, 99.9% объявлений в 1 200–24 500.
# Всё, что за границами, — опечатки и «35 000 ₸ за 80 м²», а не рынок.
PRICE_PER_M2_BOUNDS = {"sale": (50_000, 5_000_000), "rent": (1_000, 30_000)}
AREA_MIN, AREA_MAX = 8, 1000
CURRENT_YEAR = 2026

# Парсер пишет латиницей, а запросы (и LLM) будут на русском — приводим к русскому.
CITY_NAMES = {"almaty": "алматы", "kaskelen": "каскелен", "astana": "астана"}
DISTRICT_NAMES = {"bostandyk": "бостандыкский", "alatau": "алатауский", "almalin": "алмалинский",
                  "medeu": "медеуский", "auezov": "ауэзовский", "nauryzbay": "наурызбайский",
                  "turksib": "турксибский", "zhetysu": "жетысуский", "esil": "есильский",
                  "saryarka": "сарыаркинский", "baikonur": "байконурский", "nura": "нура"}


def _read_any(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Неизвестный формат: {path}")


def _to_number(s: pd.Series) -> pd.Series:
    """«45 000 000 〒» -> 45000000, «65,5 м²» -> 65.5. Числа не трогаем."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = (s.astype("string").str.replace(",", ".", regex=False)
               .str.replace(r"[^\d.\-]", "", regex=True))
    return pd.to_numeric(cleaned, errors="coerce")


def _district_ru(v):
    """«bostandykskiy_r-n» -> «бостандыкский р-н». Русские названия не трогаем."""
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return pd.NA
    for key, ru in DISTRICT_NAMES.items():
        if key in v:
            return f"{ru} р-н"
    return v


DISTRICT_RE = re.compile(r"([А-Яа-яЁё\-]+)\s+р-н")


def district_from_address(addr) -> object:
    """«Медеуский р-н, мкр Самал-2, ...» -> «медеуский р-н». В колонке district район
    заполнен у 39% строк, в адресе — у 55%, адрес есть у всех."""
    if not isinstance(addr, str):
        return pd.NA
    m = DISTRICT_RE.search(addr)
    return f"{m.group(1).lower()} р-н" if m else pd.NA


def normalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.rename(columns=RENAME).copy()
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"Нет обязательных колонок {missing}. В файле есть: {list(raw.columns)}. "
                         f"Допишите соответствие в listings.RENAME.")

    df["listing_id"] = df["listing_id"].astype(str).str.strip()
    for c in NUMERIC:
        if c in df:
            df[c] = _to_number(df[c])
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("string").str.strip().str.lower().replace("", pd.NA)
    df["city"] = df["city"].replace(CITY_NAMES)
    if "district" in df:
        df["district"] = df["district"].map(_district_ru).astype("string")
    if "address" in df:   # добираем район из адреса там, где колонка пустая
        from_addr = df["address"].map(district_from_address).astype("string")
        df["district"] = df["district"].fillna(from_addr) if "district" in df else from_addr
    if "district" in df:
        df["district"] = df["district"].fillna("не указан")
    if "deal_type" not in df:
        df["deal_type"] = "sale"

    report = {"строк на входе": len(df)}

    def drop(mask, why):
        report[why] = int(mask.sum())
        return df[~mask]

    if "rent_period" in df:   # посуточная аренда — другой рынок, в модель не берём
        df = drop(df.rent_period.fillna("month") != "month", "посуточная аренда")
    df = drop(df.duplicated("listing_id"), "дубли listing_id")
    df = drop(df.price.isna() | (df.price <= 0), "нет цены")
    df = drop(df.area.isna() | (df.area < AREA_MIN) | (df.area > AREA_MAX), "площадь вне границ")
    df = drop(df.rooms.isna() | (df.rooms < 0) | (df.rooms > 20), "комнаты вне границ")
    ppm = df.price / df.area
    lo = df.deal_type.map(lambda d: PRICE_PER_M2_BOUNDS.get(d, PRICE_PER_M2_BOUNDS["sale"])[0])
    hi = df.deal_type.map(lambda d: PRICE_PER_M2_BOUNDS.get(d, PRICE_PER_M2_BOUNDS["sale"])[1])
    df = drop((ppm < lo) | (ppm > hi), "цена за м² нереальная")

    df = df.copy()
    df["rooms"] = df["rooms"].astype(int)
    df["price_per_m2"] = df.price / df.area
    if {"floor", "floors_total"} <= set(df):
        df["floor_rel"] = df.floor / df.floors_total
        df["is_first_floor"] = (df.floor == 1).astype(float).where(df.floor.notna())
        df["is_last_floor"] = (df.floor == df.floors_total).astype(float).where(df.floor.notna())
    if "year_built" in df:
        df["building_age"] = CURRENT_YEAR - df.year_built

    report["строк на выходе"] = len(df)
    return df.reset_index(drop=True), report


def prepare(raw_path=LISTINGS_RAW) -> pd.DataFrame:
    df, report = normalize(_read_any(raw_path))
    ART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LISTINGS_PATH, index=False)

    for k, v in report.items():
        print(f"  {k:28s} {v}")
    optional = [c for c in NUMERIC + CATEGORICAL + TEXT if c in df and c not in REQUIRED]
    print("\nзаполненность необязательных колонок:")
    print(df[optional].notna().mean().sort_values().round(2).to_string())
    absent = [c for c in NUMERIC + CATEGORICAL + TEXT if c not in df]
    if absent:
        print(f"\nколонок нет вовсе: {absent}")
    return df


def load_listings() -> pd.DataFrame:
    return pd.read_parquet(LISTINGS_PATH)


# ---------------------------------------------------------------- фильтры

class Filters(BaseModel):
    """Жёсткие условия запроса. Сейчас заполняются из CLI, потом ту же схему заполнит LLM
    из текста пользователя («двушка в Алматы до 40 млн» -> rooms=[2], city, price_max)."""
    city: Optional[str] = None
    districts: Optional[list[str]] = Field(None, description="подстрока названия района")
    rooms: Optional[list[int]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    area_min: Optional[float] = None
    area_max: Optional[float] = None
    year_min: Optional[int] = None
    building_types: Optional[list[str]] = None
    not_first_floor: bool = False
    not_last_floor: bool = False


def _b(s: pd.Series) -> pd.Series:
    """Сравнения с пропусками дают NA — считаем их «не подходит»."""
    return s.fillna(False).astype(bool)


def _contains_any(col: pd.Series, needles) -> pd.Series:
    col = col.astype("string").str.lower()
    m = pd.Series(False, index=col.index)
    for n in needles:
        m |= _b(col.str.contains(n.lower(), regex=False))
    return m


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if f.city:
        m &= _b(df.city == f.city.lower())
    if f.districts and "district" in df:
        m &= _contains_any(df.district, f.districts)
    if f.rooms:
        m &= df.rooms.isin(f.rooms)
    for col, lo, hi in [("price", f.price_min, f.price_max), ("area", f.area_min, f.area_max)]:
        if lo is not None:
            m &= _b(df[col] >= lo)
        if hi is not None:
            m &= _b(df[col] <= hi)
    if f.year_min is not None and "year_built" in df:
        m &= _b(df.year_built >= f.year_min)
    if f.building_types and "building_type" in df:
        m &= _contains_any(df.building_type, f.building_types)
    if f.not_first_floor and "is_first_floor" in df:
        m &= ~_b(df.is_first_floor == 1)
    if f.not_last_floor and "is_last_floor" in df:
        m &= ~_b(df.is_last_floor == 1)
    return df[m]
