"""Оценка цены одного объявления: интервал, вердикт, аналоги.

Это точка входа для агента, MCP-сервера и сайта. Модель обучается отдельно
(price_model.py / notebooks/price_model.ipynb) и лежит в artifacts/price_model.cbm.

Вердикт формулируется от интервала и в терминах объявлений, а не сделок:
«цена выше диапазона похожих объявлений». Модель видела цены публикаций, а не
реальных сделок, и говорить «вы переплачиваете» она права не имеет.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import ART_DIR
from features import build_features
from listings import load_listings

MODEL_PATH = ART_DIR / "price_model.cbm"
OOF_PATH = ART_DIR / "price_oof.parquet"
META_PATH = ART_DIR / "price_model.json"
MIN_COMPARABLES = 10        # меньше — предупреждаем, что оценка ненадёжна


class PriceEstimate(BaseModel):
    """Ответ оценщика. Эта же схема отдаётся наружу MCP-инструментом estimate_price."""
    listing_id: str | None = None
    price: float | None = Field(None, description="цена в объявлении, ₸/мес")
    p10: float = Field(..., description="нижняя граница диапазона похожих")
    p50: float = Field(..., description="медиана похожих")
    p90: float = Field(..., description="верхняя граница")
    verdict: str
    diff_pct: float | None = Field(None, description="на сколько % цена выше/ниже медианы")
    reliable: bool = Field(..., description="достаточно ли похожих объявлений")
    source: str = Field("модель", description="out-of-fold для объявлений из базы, модель — для новых")
    n_similar: int
    comparables: list[dict] = []
    note: str = "оценка по ценам объявлений, а не по ценам сделок"


@lru_cache(maxsize=1)
def _model():
    from catboost import CatBoostRegressor
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"нет {MODEL_PATH} — обучите модель (price_model.py)")
    m = CatBoostRegressor()
    m.load_model(str(MODEL_PATH))
    return m, json.loads(META_PATH.read_text())


@lru_cache(maxsize=1)
def _data() -> pd.DataFrame:
    return load_listings()


@lru_cache(maxsize=1)
def _oof() -> pd.DataFrame:
    """Честные предсказания для объявлений, которые есть в базе: их делала модель,
    НЕ видевшая это объявление. Обученная на всех данных модель их запомнила
    и показала бы неправдоподобно точную оценку."""
    if not OOF_PATH.exists():
        return pd.DataFrame(columns=["listing_id", "p10", "p50", "p90"]).set_index("listing_id")
    return pd.read_parquet(OOF_PATH).set_index("listing_id")


def predict_interval(rows: pd.DataFrame) -> np.ndarray:
    """(n, 3): p10, p50, p90 в тенге. Конформная поправка уже применена."""
    model, meta = _model()
    X = build_features(rows)
    cols, cats = meta["features"], meta["cat_features"]
    X = X[cols].copy()
    for c in cats:
        X[c] = X[c].astype(str)
    p = np.sort(np.exp(model.predict(X)), axis=1)
    q = float(meta.get("conformal_q", 0.0))
    return np.stack([p[:, 0] * np.exp(-q), p[:, 1], p[:, 2] * np.exp(q)], axis=1)


def comparables(row, k=5) -> tuple[pd.DataFrame, str, int]:
    """Похожие объявления со смягчением условий: для редких квартир точных аналогов нет,
    и пустой список рядом с вердиктом выглядит как решение без обоснования."""
    df = _data()
    same = df.listing_id != str(row.get("listing_id", ""))
    steps = [
        ("район + комнаты + площадь ±20%",
         (df.district == row["district"]) & (df.rooms == row["rooms"])
         & df.area.between(row["area"] * .8, row["area"] * 1.2)),
        ("район + комнаты", (df.district == row["district"]) & (df.rooms == row["rooms"])),
        ("город + комнаты + площадь ±30%",
         (df.city == row["city"]) & (df.rooms == row["rooms"])
         & df.area.between(row["area"] * .7, row["area"] * 1.3)),
        ("город + площадь ±30%", (df.city == row["city"]) & df.area.between(row["area"] * .7, row["area"] * 1.3)),
    ]
    for label, m in steps:
        m = m & same
        if m.sum() >= 3:
            out = df[m].assign(d=(df[m].area - row["area"]).abs()).nsmallest(k, "d")
            return out, label, int(m.sum())
    pool = df[same & (df.city == row["city"])]
    return pool.assign(d=(pool.area - row["area"]).abs()).nsmallest(k, "d"), "похожих нет, ближайшие по площади", 0


def estimate(listing: dict | pd.Series, with_comparables=True, use_oof=True) -> PriceEstimate:
    """listing — строка из таблицы или словарь с полями area, rooms, city, district, floor…"""
    row = pd.Series(listing) if isinstance(listing, dict) else listing
    lid = str(row.get("listing_id")) if row.get("listing_id") is not None else None

    oof = _oof()
    if use_oof and lid in oof.index:
        p10, p50, p90 = oof.loc[lid, ["p10", "p50", "p90"]]
        source = "out-of-fold"
    else:
        p10, p50, p90 = predict_interval(pd.DataFrame([row]))[0]
        source = "модель"

    comps, how, n = (comparables(row) if with_comparables
                     else (pd.DataFrame(), "не запрашивались", MIN_COMPARABLES))
    reliable = n >= MIN_COMPARABLES
    price = float(row["price"]) if pd.notna(row.get("price")) else None

    if price is None:
        verdict = "цена не указана, показана оценка диапазона"
    elif price > p90:
        verdict = "цена выше диапазона похожих объявлений"
    elif price < p10:
        verdict = "цена ниже диапазона похожих объявлений"
    else:
        verdict = "цена в диапазоне похожих объявлений"
    if not reliable:
        verdict += " (похожих мало, оценка ненадёжна)"

    return PriceEstimate(
        listing_id=lid, source=source,
        price=price, p10=float(p10), p50=float(p50), p90=float(p90), verdict=verdict,
        diff_pct=round((price - p50) / p50 * 100, 1) if price else None,
        reliable=reliable, n_similar=n,
        comparables=[{"listing_id": c.listing_id, "price": float(c.price), "area": float(c.area),
                      "rooms": int(c.rooms), "district": str(c.district), "matched_by": how}
                     for c in comps.itertuples()] if len(comps) else [],
    )


def estimate_by_id(listing_id: str) -> PriceEstimate:
    df = _data()
    hit = df[df.listing_id == str(listing_id)]
    if hit.empty:
        raise KeyError(f"объявление {listing_id} не найдено")
    return estimate(hit.iloc[0])


if __name__ == "__main__":
    import sys
    lid = sys.argv[1] if len(sys.argv) > 1 else _data().listing_id.iloc[0]
    print(estimate_by_id(lid).model_dump_json(indent=2, exclude_none=True))
