"""Migrate the `listing_photos` collection from one Qdrant to another
(local Docker -> Qdrant Cloud) without re-encoding any photos.

Copies vectors + payloads point-for-point via scroll/upsert, recreating the
collection and its payload indexes on the destination. Point ids are stable
(uuid5 of sha256), so this is idempotent — re-running upserts the same ids.

Source defaults to local Docker; destination defaults to the env (QDRANT_URL +
QDRANT_API_KEY), i.e. the cloud cluster you configured in .env.

Usage:
    python -m scripts.migrate_qdrant [--source-url URL] [--dest-url URL] [--batch N]
"""
from __future__ import annotations

import argparse
import os
import sys

from krisha.embed_worker import COLLECTION, make_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-url", default="http://localhost:6333",
                    help="source Qdrant (default: local Docker)")
    ap.add_argument("--source-api-key", default="", help="source api key (default: none)")
    ap.add_argument("--dest-url", default=os.getenv("QDRANT_URL"),
                    help="destination Qdrant (default: QDRANT_URL from .env)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from qdrant_client.models import (Distance, VectorParams,  # type: ignore
                                      PayloadSchemaType, PointStruct)

    src = make_client(url=args.source_url, api_key=args.source_api_key)
    dst = make_client(url=args.dest_url)  # api_key from env
    if args.source_url == (args.dest_url or ""):
        print("source and destination are identical; nothing to do")
        return 1

    src_cols = {c.name for c in src.get_collections().collections}
    if COLLECTION not in src_cols:
        print(f"source has no collection {COLLECTION!r}")
        return 1
    info = src.get_collection(COLLECTION)
    total = info.points_count
    vsize = info.config.params.vectors.size
    vdist = info.config.params.vectors.distance
    print(f"source: {total} points, dim={vsize}, distance={vdist}")
    print(f"dest  : {args.dest_url}")
    if args.dry_run:
        return 0

    # (re)create destination collection + payload indexes to match embed_worker
    dst_cols = {c.name for c in dst.get_collections().collections}
    if COLLECTION not in dst_cols:
        dst.create_collection(
            COLLECTION, vectors_config=VectorParams(size=vsize, distance=vdist))
        for field, ftype in [("city", PayloadSchemaType.KEYWORD),
                             ("district", PayloadSchemaType.KEYWORD),
                             ("rooms", PayloadSchemaType.INTEGER),
                             ("rent_period", PayloadSchemaType.KEYWORD),
                             ("price_kzt", PayloadSchemaType.INTEGER),
                             ("duplicate_group_id", PayloadSchemaType.KEYWORD)]:
            dst.create_payload_index(COLLECTION, field, ftype)
        print("created destination collection + payload indexes")

    copied, offset = 0, None
    while True:
        points, offset = src.scroll(
            COLLECTION, limit=args.batch, offset=offset,
            with_payload=True, with_vectors=True)
        if not points:
            break
        dst.upsert(COLLECTION, points=[
            PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
        ], wait=True)
        copied += len(points)
        print(f"  copied {copied}/{total}", end="\r", flush=True)
        if offset is None:
            break
    print(f"\ndone: copied {copied} points to {args.dest_url}")

    dst_count = dst.get_collection(COLLECTION).points_count
    print(f"destination now holds {dst_count} points")
    return 0 if dst_count >= total else 1


if __name__ == "__main__":
    sys.exit(main())
