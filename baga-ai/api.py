"""Публичный слой функций Baga AI.

Всё, что нужно агенту, MCP-серверу и сайту, лежит здесь. Функции возвращают обычные
словари (JSON-совместимые), ничего не печатают и не требуют состояния снаружи.

Обернуть в MCP-инструменты:
    search_listings   — поиск квартир по условиям и стилю
    estimate_price    — справедливый диапазон цены и вердикт
    get_comparables   — похожие объявления, на которых построен вердикт
    get_listing       — карточка объявления по id

Тяжёлые объекты (индексы, модель) грузятся один раз при первом вызове.
"""
from typing import Any, Optional

import pandas as pd

from config import ART_DIR
from listings import Filters, load_listings


def _df() -> pd.DataFrame:
    """Таблица объявлений. Намеренно НЕ через поисковый движок: оценка цены не должна
    тянуть за собой матрицу эмбеддингов и подключение к Qdrant."""
    from functools import lru_cache

    from listings import load_listings

    @lru_cache(maxsize=1)
    def _cached():
        return load_listings().set_index("listing_id", drop=False)

    return _cached()


def search_listings(query: Optional[str] = None, city: Optional[str] = None,
                    districts: Optional[list[str]] = None, rooms: Optional[list[int]] = None,
                    price_min: Optional[float] = None, price_max: Optional[float] = None,
                    area_min: Optional[float] = None, area_max: Optional[float] = None,
                    photo_rooms: Optional[list[str]] = None, image: Optional[str] = None,
                    images: Optional[list[str]] = None,
                    k: int = 10, with_price_verdict: bool = True,
                    with_answer: bool = False) -> dict[str, Any]:
    """Найти квартиры.

    query — свободное описание желаемого стиля («светлая кухня в скандинавском стиле»);
        по нему ищутся похожие ФОТОГРАФИИ и описания. Числа и условия сюда писать не нужно.
    city / districts / rooms / price_min / price_max / area_min / area_max — жёсткие условия.
        districts принимает часть названия: «бостандык» найдёт «бостандыкский р-н».
    photo_rooms — ограничить поиск по фото типом комнаты: kitchen, living_room, bedroom,
        bathroom, hallway, balcony.
    image — путь к картинке вместо текста (в том числе сгенерированной ИИ).
    images — несколько картинок сразу («хочу такую кухню И такую детскую»): каждая ищется
        отдельно со своим типом комнаты, выше встают объявления, закрывшие больше фото.
    with_price_verdict — добавить к каждому результату оценку цены.
    with_answer — сгенерировать связный ответ по найденному (шаг генерации в RAG).

    Возвращает: n_candidates (сколько прошло фильтры), channels_used (по чему ранжировали),
    results — список объявлений со скором, карточкой, фото и вердиктом по цене.
    """
    from search import get_search
    from tracing import span
    f = Filters(city=city, districts=districts, rooms=rooms, price_min=price_min,
                price_max=price_max, area_min=area_min, area_max=area_max)
    with span("search_listings", kind="RETRIEVER", **{"input.value": query or image or f"{len(images or [])} фото",
                                                      "filters": f.model_dump_json()}) as sp:
        res = get_search().search(text=query, filters=f, image=image, images=images,
                                  rooms=photo_rooms, k=k)
        sp.set(**{"n_candidates": res["n_candidates"], "channels": ",".join(res["channels_used"]),
                  "n_results": len(res["results"])})
    import rerank
    if rerank.enabled() and query and len(res["results"]) > 1:
        with span("rerank", kind="RETRIEVER", **{"input.value": query}) as sp:
            res["results"], rep = rerank.rerank(query, res["results"], url_of=_photo_url)
            sp.set(output=[r["listing_id"] for r in res["results"]], **rep)
    if with_price_verdict:
        for r in res["results"]:
            try:
                r["price_check"] = estimate_price(listing_id=r["listing_id"], with_comparables=False)
            except Exception as e:
                r["price_check"] = {"error": str(e)}
    if with_answer and query:
        from explain import answer_search
        with span("answer_search", kind="LLM", **{"input.value": query}):
            try:
                res["answer"] = answer_search(query, res["results"])
            except Exception as e:
                res["answer"] = f"(ответ недоступен: {e})"
    return res


def estimate_price(listing_id: Optional[str] = None, listing: Optional[dict] = None,
                   with_comparables: bool = True) -> dict[str, Any]:
    """Справедливый диапазон месячной аренды и вердикт.

    Передать либо listing_id объявления из базы, либо listing — словарь с полями
    area, rooms, city, district, floor, floors_total, address, description, price.

    Возвращает p10/p50/p90, verdict, diff_pct, reliable (достаточно ли похожих),
    comparables и source: out-of-fold для объявлений из базы, модель — для новых.

    Важно: оценка построена на ценах ОБЪЯВЛЕНИЙ, а не сделок. Формулировка вердикта —
    «выше диапазона похожих объявлений», а не «вы переплачиваете».
    """
    from predict import estimate, estimate_by_id
    from tracing import span
    with span("estimate_price", kind="TOOL", listing_id=str(listing_id or "")):
        return _estimate(listing_id, listing, with_comparables)


def _estimate(listing_id, listing, with_comparables):
    from predict import estimate, estimate_by_id
    if listing_id is not None:
        est = estimate_by_id(str(listing_id)) if with_comparables else \
            estimate(_df().loc[str(listing_id)], with_comparables=False)
    elif listing is not None:
        est = estimate(listing, with_comparables=with_comparables)
    else:
        raise ValueError("нужен listing_id или listing")
    return est.model_dump(exclude_none=True)


def get_comparables(listing_id: str, k: int = 5) -> dict[str, Any]:
    """Похожие объявления, на которых построен вердикт по цене.

    Условия смягчаются по шагам: район+комнаты+площадь → район+комнаты → город+комнаты →
    город+площадь. matched_by показывает, на каком шаге нашлось, n_similar — сколько всего.
    """
    from predict import comparables
    df = _df()
    if str(listing_id) not in df.index:
        raise KeyError(f"объявление {listing_id} не найдено")
    comps, how, n = comparables(df.loc[str(listing_id)], k=k)
    return {"listing_id": str(listing_id), "matched_by": how, "n_similar": n,
            "reliable": n >= 10,
            "comparables": [{"listing_id": c.listing_id, "price": float(c.price),
                             "area": float(c.area), "rooms": int(c.rooms),
                             "district": str(c.district), "floor": _num(c.floor),
                             "url": getattr(c, "url", None)} for c in comps.itertuples()]}


def get_listing(listing_id: str) -> dict[str, Any]:
    """Карточка объявления: характеристики, адрес, описание, ссылка."""
    df = _df()
    if str(listing_id) not in df.index:
        raise KeyError(f"объявление {listing_id} не найдено")
    row = df.loc[str(listing_id)]
    fields = ["listing_id", "price", "area", "rooms", "floor", "floors_total", "city", "district",
              "address", "condition", "description", "url", "photos_count"]
    return {c: _plain(row[c]) for c in fields if c in row.index}


def _num(v):
    return None if pd.isna(v) else float(v)


def _plain(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


TOOLS = [search_listings, estimate_price, get_comparables, get_listing]


_urls = None


def _photo_url(path: str | None, listing_id: str | None = None) -> str | None:
    """CDN-ссылка на фото: для реранкера там, где файлов фото нет (Docker, Spaces)."""
    global _urls
    if _urls is None:
        try:
            u = pd.read_parquet(ART_DIR / "photo_urls.parquet")
            _urls = (dict(zip(u.path, u.url)),
                     u.sort_values(["listing_id", "photo_n"]).groupby("listing_id").url.first().to_dict())
        except Exception:
            _urls = ({}, {})
    return _urls[0].get(path) or _urls[1].get(str(listing_id))
