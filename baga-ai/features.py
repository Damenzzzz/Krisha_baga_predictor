"""Признаки для модели цены.

Главное правило: в признаки не должна попасть цена. У 60% объявлений она написана
прямо в описании («… 750 000 ₸ за месяц …»), и в 57.6% случаев это ровно наш таргет.
Модель с таким текстом покажет отличную метрику и развалится на новых данных,
поэтому clean_description() вырезает цену ДО всего остального.

Второй источник признаков — адрес: он заполнен у 100% объявлений и содержит район,
микрорайон, улицу и иногда название ЖК. Для аренды локация важнее всего.
"""
import re

import numpy as np
import pandas as pd

# Цену пишут как угодно: «750 000 ₸», «200000+ком.услуги», «190000 тысяч», «350000 тнг».
# Поэтому маскируем ЛЮБОЕ число от 10 000: аренда всегда такая, а площади, этажи,
# годы постройки и высота потолков — меньше.
MONEY_MIN = 10_000
NUM_SEQ_RE = re.compile(r"\d[\d\s\u00a0.,]*\d|\d")
TAIL_RE = re.compile(r"(Хозяин недвижимости|В Избранное|Сохранить в подборку).*$", re.S)

FLAGS = {   # признак -> что искать в описании
    "furnished":     r"меблирован",
    "no_furniture":  r"без мебели",
    "appliances":    r"(?:стиральн|холодильник|посудомо)",
    "internet":      r"интернет",
    "deposit":       r"(?:депозит|предоплат)",
    "kids_pets_ok":  r"(?:можно с детьми|с животными)",
    "new_building":  r"(?:новостройк|жк\s)",
    "renovation":    r"ремонт",
    "parking":       r"(?:парковк|паркинг)",
    "security":      r"(?:охран|консьерж|видеонаблюден)",
    "separate_bath": r"санузел\s*раздельн",
    "combined_bath": r"санузел\s*совмещ",
}

NUMS = {    # признак -> regex с одной группой
    "kitchen_area_txt": r"кухня\s*(\d+[.,]?\d*)\s*м",
    "living_area_txt":  r"жил\.?\s*площадь\s*(\d+[.,]?\d*)",
    "ceiling":          r"потолки\s*(\d[.,]\d)",
    "year_txt":         r"(\d{4})\s*г\.?\s*п",
}

ADDR_HOUSE_RE = re.compile(r"\s+\d+[а-я]?(/\d+)?$", re.I)


def _mask_money(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return " цена " if digits and len(digits) >= 5 and int(digits) >= MONEY_MIN else m.group(0)


def clean_description(text) -> str:
    """Убирает цену (в любом виде) и хвост сайта («Хозяин недвижимости … В Избранное …»)."""
    if not isinstance(text, str):
        return ""
    text = TAIL_RE.sub(" ", text)
    text = NUM_SEQ_RE.sub(_mask_money, text)
    return re.sub(r"\s+", " ", text).strip()


def assert_no_price_leak(texts: pd.Series, prices: pd.Series) -> None:
    """Страховка: если цена всё же осталась в тексте, лучше упасть, чем обучиться на утечке."""
    left = [str(int(p)) in t.replace(" ", "") for t, p in zip(texts.fillna(""), prices.fillna(0))]
    share = float(np.mean(left))
    if share > 0.01:
        raise AssertionError(f"цена осталась в тексте у {share:.1%} объявлений — проверьте clean_description")


def parse_address(addr) -> dict:
    """«Медеуский р-н, мкр Самал-2, Аль-Фараби 17 — Достык»
    -> {mkr: 'мкр самал-2', street: 'аль-фараби', cross_street: 'достык'}
    Часть после тире — это пересечение улиц, а не название ЖК."""
    out = {"mkr": pd.NA, "street": pd.NA, "cross_street": pd.NA}
    if not isinstance(addr, str):
        return out
    head, _, tail = addr.partition("—")
    if tail.strip():
        out["cross_street"] = tail.strip().lower()
    parts = [p.strip().lower() for p in head.split(",") if p.strip()]
    for p in parts:
        if p.startswith(("мкр", "мк-р", "микрорайон")):
            out["mkr"] = p
    street = [p for p in parts if not p.startswith(("мкр", "мк-р", "микрорайон")) and "р-н" not in p]
    if street:
        out["street"] = ADDR_HOUSE_RE.sub("", street[-1]).strip()
    return out


NUM_COLS = ["area", "rooms", "floor", "floors_total", "floor_rel", "is_first_floor", "is_last_floor",
            "kitchen_area", "area_per_room", "photos_count", "desc_len", "lat", "lon",
            "kitchen_area_txt", "living_area_txt", "ceiling", "year_txt"]
CAT_COLS = ["city", "district", "mkr", "street", "cross_street", "condition", "bathroom"]
FLAG_COLS = list(FLAGS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков в том же порядке строк, что и df."""
    f = pd.DataFrame(index=df.index)
    desc = df.get("description", pd.Series("", index=df.index)).map(clean_description)
    assert_no_price_leak(desc, df.price)

    for c in ["area", "rooms", "floor", "floors_total", "floor_rel", "is_first_floor",
              "is_last_floor", "kitchen_area", "photos_count", "lat", "lon"]:
        f[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    f["area_per_room"] = f.area / f.rooms.replace(0, np.nan)
    f["desc_len"] = desc.str.len()

    for name, pat in NUMS.items():
        f[name] = pd.to_numeric(desc.str.extract(pat, flags=re.I)[0].str.replace(",", ".", regex=False),
                                errors="coerce")
    for name, pat in FLAGS.items():
        f[name] = desc.str.contains(pat, case=False, regex=True).astype(int)

    addr = pd.DataFrame([parse_address(a) for a in df.get("address", pd.Series(index=df.index))], index=df.index)
    for c in CAT_COLS:
        src = addr[c] if c in addr else df.get(c)
        f[c] = (pd.Series(src, index=df.index).astype("string").fillna("не указан")
                if src is not None else "не указан")
    return f[NUM_COLS + FLAG_COLS + CAT_COLS]
