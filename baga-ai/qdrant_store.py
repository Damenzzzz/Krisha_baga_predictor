"""Векторное хранилище на Qdrant: фото (SigLIP 2) и описания (ALEM text-1024).

Почему Qdrant, а не матрица в numpy:
  - фильтры по характеристикам (город, комнаты, цена, площадь) выполняются ВНУТРИ базы
    вместе с поиском по вектору, а не отдельным проходом по pandas;
  - группировка по объявлению делается запросом query_points_groups — у квартиры 10-20 фото,
    и без группировки одна квартира занимала бы весь топ;
  - состояние переживает перезапуск процесса, и тем же индексом пользуются агент и MCP-сервер.

Два режима, переключаются переменной QDRANT_URL:
  - сервер (docker compose up -d qdrant, как в репозитории парсера) — QDRANT_URL=http://localhost:6333
  - встроенный, без сервера: данные в artifacts/qdrant (по умолчанию)
"""
import os

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from config import (ART_DIR, DESC_EMB_PATH, DESC_IDS_PATH, EMB_PATH, RENDER_THR, ROOMS_PATH,
                    VALID_PATH)
from embed import load_embeddings
from listings import Filters, load_listings

PHOTOS_COLLECTION = os.getenv("QDRANT_PHOTOS", "listing_photos_siglip")
DESC_COLLECTION = os.getenv("QDRANT_DESC", "listing_descriptions")
BATCH = 512

_client = None


def close():
    """Встроенный Qdrant держит файл открытым; без явного закрытия интерпретатор
    ругается при выходе."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        _client = QdrantClient(url=url) if url else QdrantClient(path=str(ART_DIR / "qdrant"))
        print(f"qdrant: {'сервер ' + url if url else 'встроенный ' + str(ART_DIR / 'qdrant')}")
        import atexit
        atexit.register(close)
    return _client


def _recreate(name: str, dim: int, indexed: dict):
    cl = get_client()
    if cl.collection_exists(name):
        cl.delete_collection(name)
    cl.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    for field, schema in indexed.items():   # без индексов фильтрация по payload медленная
        cl.create_payload_index(name, field, schema)


def ingest_photos():
    """Векторы фото + характеристики объявления в payload (чтобы фильтровать внутри базы)."""
    idx, E, valid = load_embeddings()
    rooms = pd.read_parquet(ROOMS_PATH) if ROOMS_PATH.exists() else None
    lst = load_listings().set_index("listing_id")

    _recreate(PHOTOS_COLLECTION, E.shape[1], {
        "listing_id": models.PayloadSchemaType.KEYWORD,
        "city": models.PayloadSchemaType.KEYWORD,
        "district": models.PayloadSchemaType.KEYWORD,
        "room_type": models.PayloadSchemaType.KEYWORD,
        "rooms": models.PayloadSchemaType.INTEGER,
        "price": models.PayloadSchemaType.FLOAT,
        "area": models.PayloadSchemaType.FLOAT,
        "is_render": models.PayloadSchemaType.BOOL,
    })

    cl, n = get_client(), 0
    for start in tqdm(range(0, len(idx), BATCH), desc="фото"):
        sl = slice(start, start + BATCH)
        points = []
        for i in range(start, min(start + BATCH, len(idx))):
            if not valid[i]:
                continue
            lid = idx.listing_id.iat[i]
            if lid not in lst.index:            # фото есть, а объявление отсеяно чисткой
                continue
            row = lst.loc[lid]
            payload = {
                "listing_id": lid, "photo_n": int(idx.photo_n.iat[i]), "path": idx.path.iat[i],
                "room_type": str(rooms.room_type.iat[i]) if rooms is not None else "unknown",
                "is_render": bool(rooms.render_prob.iat[i] > RENDER_THR) if rooms is not None else False,
                "city": str(row.city), "district": str(row.district), "rooms": int(row.rooms),
                "price": float(row.price), "area": float(row.area),
                "dup_group": str(row.get("duplicate_group_id", lid)),
            }
            points.append(models.PointStruct(id=i, vector=E[i].tolist(), payload=payload))
        if points:
            cl.upsert(PHOTOS_COLLECTION, points=points)
            n += len(points)
    print(f"в коллекции {PHOTOS_COLLECTION}: {n} векторов фото")


def ingest_descriptions():
    if not DESC_EMB_PATH.exists():
        print("нет desc_emb.npy — пропускаю описания")
        return
    E = np.load(DESC_EMB_PATH).astype(np.float32)
    ids = np.load(DESC_IDS_PATH)
    lst = load_listings().set_index("listing_id")
    _recreate(DESC_COLLECTION, E.shape[1], {
        "listing_id": models.PayloadSchemaType.KEYWORD,
        "city": models.PayloadSchemaType.KEYWORD,
        "district": models.PayloadSchemaType.KEYWORD,
        "rooms": models.PayloadSchemaType.INTEGER,
        "price": models.PayloadSchemaType.FLOAT,
        "area": models.PayloadSchemaType.FLOAT,
    })
    cl, points = get_client(), []
    for i, lid in enumerate(ids):
        if lid not in lst.index:
            continue
        row = lst.loc[lid]
        points.append(models.PointStruct(id=i, vector=E[i].tolist(), payload={
            "listing_id": str(lid), "city": str(row.city), "district": str(row.district),
            "rooms": int(row.rooms), "price": float(row.price), "area": float(row.area)}))
        if len(points) >= BATCH:
            cl.upsert(DESC_COLLECTION, points=points); points = []
    if points:
        cl.upsert(DESC_COLLECTION, points=points)
    print(f"в коллекции {DESC_COLLECTION}: {len(ids)} векторов описаний")


def build_filter(f: Filters | None, rooms: list[str] | None = None,
                 allow_renders=False, districts_exact: list[str] | None = None) -> models.Filter | None:
    """pydantic-фильтры пользователя -> условия Qdrant. Район приходит уже развёрнутым
    в точные названия (LLM говорит «бостандык», в данных «бостандыкский р-н»)."""
    must: list = []
    f = f or Filters()
    if f.city:
        must.append(models.FieldCondition(key="city", match=models.MatchValue(value=f.city.lower())))
    if districts_exact:
        must.append(models.FieldCondition(key="district", match=models.MatchAny(any=districts_exact)))
    if f.rooms:
        must.append(models.FieldCondition(key="rooms", match=models.MatchAny(any=[int(r) for r in f.rooms])))
    if f.price_min is not None or f.price_max is not None:
        must.append(models.FieldCondition(key="price", range=models.Range(gte=f.price_min, lte=f.price_max)))
    if f.area_min is not None or f.area_max is not None:
        must.append(models.FieldCondition(key="area", range=models.Range(gte=f.area_min, lte=f.area_max)))
    if rooms:
        must.append(models.FieldCondition(key="room_type", match=models.MatchAny(any=list(rooms))))
    if not allow_renders:
        must.append(models.FieldCondition(key="is_render", match=models.MatchValue(value=False)))
    return models.Filter(must=must) if must else None


def search_photos(vector, f: Filters | None = None, rooms=None, districts_exact=None,
                  allow_renders=False, k=10, photos_per_listing=3) -> list[dict]:
    """Поиск с группировкой по объявлению средствами Qdrant: скор объявления = лучшее фото."""
    res = get_client().query_points_groups(
        PHOTOS_COLLECTION, query=list(map(float, vector)), group_by="listing_id",
        limit=k, group_size=photos_per_listing, with_payload=True,
        query_filter=build_filter(f, rooms, allow_renders, districts_exact))
    return [{
        "listing_id": str(g.id),
        "score": float(g.hits[0].score),
        "photos": [{"path": h.payload["path"], "room_type": h.payload["room_type"],
                    "score": float(h.score)} for h in g.hits],
    } for g in res.groups]


def search_descriptions(vector, f: Filters | None = None, districts_exact=None, k=50) -> list[tuple[str, float]]:
    res = get_client().query_points(
        DESC_COLLECTION, query=list(map(float, vector)), limit=k, with_payload=True,
        query_filter=build_filter(f, None, True, districts_exact))
    return [(p.payload["listing_id"], float(p.score)) for p in res.points]


def has_collection(name: str) -> bool:
    try:
        return get_client().collection_exists(name)
    except Exception:
        return False


def stats():
    cl = get_client()
    for name in (PHOTOS_COLLECTION, DESC_COLLECTION):
        if cl.collection_exists(name):
            info = cl.get_collection(name)
            print(f"{name:28s} точек: {info.points_count}, размерность: {info.config.params.vectors.size}")
        else:
            print(f"{name:28s} нет")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Загрузка векторов в Qdrant")
    p.add_argument("cmd", choices=["ingest", "ingest-photos", "ingest-desc", "stats"])
    a = p.parse_args()
    if a.cmd in ("ingest", "ingest-photos"):
        ingest_photos()
    if a.cmd in ("ingest", "ingest-desc"):
        ingest_descriptions()
    stats()
