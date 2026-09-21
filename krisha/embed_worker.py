"""Phase B: embed listing photos with open_clip and upsert into Qdrant.

Heavy deps (torch/open_clip/qdrant-client) are imported lazily so the parser
core runs without them installed."""
from __future__ import annotations

import os
import uuid

from . import config
from .db import Database
from .logging_setup import get_logger

log = get_logger("embed")

_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000c11b")  # stable ns for clip
COLLECTION = os.getenv("QDRANT_COLLECTION", "listing_photos")


def _load_model():
    import open_clip  # type: ignore
    import torch  # type: ignore

    # ViT-B-32-quickgelu + openai: the openai weights were trained with the
    # QuickGELU activation; pairing them with plain "ViT-B-32" (GELU) triggers a
    # runtime mismatch warning and degrades embedding quality. Keep them paired.
    model_name = os.getenv("CLIP_MODEL", "ViT-B-32-quickgelu")
    pretrained = os.getenv("CLIP_PRETRAINED", "openai")
    device = os.getenv("CLIP_DEVICE", "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device)
    model.eval()
    dim = model.visual.output_dim
    return model, preprocess, device, dim, torch


def _client(dim: int):
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType  # type: ignore

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field, ftype in [("city", PayloadSchemaType.KEYWORD),
                             ("district", PayloadSchemaType.KEYWORD),
                             ("rooms", PayloadSchemaType.INTEGER),
                             ("rent_period", PayloadSchemaType.KEYWORD),
                             ("price_kzt", PayloadSchemaType.INTEGER),
                             ("duplicate_group_id", PayloadSchemaType.KEYWORD)]:
            client.create_payload_index(COLLECTION, field, ftype)
    return client


def run(db: Database, batch_size: int = 16, limit: int | None = None) -> dict:
    from PIL import Image  # type: ignore
    from qdrant_client.models import PointStruct  # type: ignore

    model, preprocess, device, dim, torch = _load_model()
    client = _client(dim)

    pending = db.photos_pending_embed(limit=limit)
    log.info("embed: %d photos pending", len(pending))
    stats = {"embedded": 0, "skipped": 0}

    batch: list = []
    metas: list = []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            tensor = torch.stack(batch).to(device)
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        points = []
        for vec, meta in zip(feats.cpu().numpy().tolist(), metas):
            pid = str(uuid.uuid5(_NAMESPACE, meta["sha256"]))
            points.append(PointStruct(id=pid, vector=vec, payload=meta["payload"]))
        client.upsert(COLLECTION, points=points)
        for meta in metas:
            db.upsert_photo(meta["listing_id"], meta["idx"], embedded_at=_now())
            stats["embedded"] += 1
        batch.clear()
        metas.clear()

    for row in pending:
        listing = db.get_listing(row["listing_id"])
        if listing is None:
            stats["skipped"] += 1
            continue
        try:
            img = Image.open(row["local_path"]).convert("RGB")
        except Exception as e:  # noqa: BLE001
            log.warning("cannot open %s: %s", row["local_path"], e)
            stats["skipped"] += 1
            continue
        batch.append(preprocess(img))
        price = listing["price_kzt"]
        area = listing["area_total"]
        metas.append({
            "sha256": row["sha256"],
            "listing_id": row["listing_id"],
            "idx": row["idx"],
            "payload": {
                "listing_id": row["listing_id"],
                "photo_idx": row["idx"],
                "city": listing["city"],
                "district": listing["district"],
                "rooms": listing["rooms"],
                "area_total": area,
                "floor": listing["floor"],
                "price_kzt": price,
                "price_per_m2": (price / area) if price and area else None,
                "rent_period": listing["rent_period"],
                "lat": listing["lat"],
                "lon": listing["lon"],
                "duplicate_group_id": listing["duplicate_group_id"],
            },
        })
        if len(batch) >= batch_size:
            flush()
    flush()
    log.info("embed done: %s", stats)
    print(f"embed: {stats}")
    return stats


def search_by_image(db: Database, image_path: str, city: str | None = None,
                    limit: int = 10) -> list[dict]:
    """Phase B smoke query: find listings whose photos look like ``image_path``.

    Embeds the query image with the same CLIP model used for indexing and runs a
    cosine search over the ``listing_photos`` collection, optionally filtered to a
    single ``city``. Returns a ranked list of ``{score, payload, url}`` dicts."""
    from PIL import Image  # type: ignore
    from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

    model, preprocess, device, dim, torch = _load_model()
    client = _client(dim)

    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        tensor = preprocess(img).unsqueeze(0).to(device)
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    vector = feats.cpu().numpy()[0].tolist()

    qfilter = None
    if city:
        qfilter = Filter(must=[FieldCondition(key="city", match=MatchValue(value=city))])

    hits = client.query_points(COLLECTION, query=vector, query_filter=qfilter,
                               limit=limit, with_payload=True).points
    results = []
    for h in hits:
        payload = h.payload or {}
        listing = db.get_listing(payload.get("listing_id"))
        results.append({
            "score": h.score,
            "payload": payload,
            "url": listing["url"] if listing is not None else None,
        })
    return results


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
