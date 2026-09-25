"""Resync mutable payload fields of embedded Qdrant points from the SQLite DB.

Some payload fields are set at embed time but change later:
  - `city` was split by casing until canonicalized (normalize_city);
  - `duplicate_group_id` is assigned by `dedup`, which usually runs AFTER embed,
    so freshly embedded points carry a stale/NULL group.

This scrolls every point, reads its `listing_id` from the payload, looks up the
current DB values, and patches all mutable listing fields, including price,
coordinates and dedup groups. Idempotent; re-run after details/embed/dedup.

Usage (targets QDRANT_URL from .env — local Docker or cloud):
    python -m scripts.resync_qdrant_payload [--dry-run]
"""
from __future__ import annotations

import argparse
import math
import sys

from krisha.db import Database
from krisha.embed_worker import COLLECTION, make_client
from krisha.payload import listing_payload


def payload_matches(payload: dict, expected: dict) -> bool:
    """Ignore JSON floating-point roundoff, but retain meaningful field changes."""
    for key, value in expected.items():
        actual = payload.get(key)
        if actual == value:
            continue
        if (isinstance(value, float) and isinstance(actual, (int, float))
                and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-9)):
            continue
        return False
    return True


def sync_listings(client, db: Database, listing_ids: set[int]) -> None:
    """Incremental sync by indexed owner listing_id; shared-photo owners stay intact."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SetPayload, SetPayloadOperation
    operations = []
    for lid in sorted(listing_ids):
        row = db.get_listing(lid)
        if row is None:
            continue
        operations.append(SetPayloadOperation(set_payload=SetPayload(
            payload=listing_payload(row),
            filter=Filter(must=[FieldCondition(key="listing_id", match=MatchValue(value=lid))]))))
        if len(operations) >= 100:
            client.batch_update_points(COLLECTION, operations, wait=True)
            operations = []
    if operations:
        client.batch_update_points(COLLECTION, operations, wait=True)


def resync(client, db: Database, dry_run: bool = False) -> dict:
    from qdrant_client.models import SetPayload, SetPayloadOperation

    truth = {r["id"]: listing_payload(r) for r in db.all_listings()}
    scanned = missing = 0
    stale: dict[int, list] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for point in points:
            scanned += 1
            payload = point.payload or {}
            lid = payload.get("listing_id")
            if lid not in truth:
                missing += 1
                continue
            if not payload_matches(payload, truth[lid]):
                stale.setdefault(lid, []).append(point.id)
        if offset is None:
            break
    count = sum(len(ids) for ids in stale.values())
    print(f"scanned {scanned} points; stale {count} across {len(stale)} listings; "
          f"{missing} points had no matching DB listing", flush=True)
    fixed = 0
    operations = []
    queued = 0
    if not dry_run:
        for lid, ids in stale.items():
            for start in range(0, len(ids), 1000):
                chunk = ids[start:start + 1000]
                operations.append(SetPayloadOperation(set_payload=SetPayload(
                    payload=truth[lid], points=chunk)))
                queued += len(chunk)
                if len(operations) >= 100:
                    client.batch_update_points(COLLECTION, operations, wait=True)
                    fixed += queued
                    operations, queued = [], 0
                    print(f"confirmed {fixed}/{count} points updated", flush=True)
        if operations:
            client.batch_update_points(COLLECTION, operations, wait=True)
            fixed += queued
    return {"scanned": scanned, "stale": count, "fixed": fixed, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    client = make_client()
    cols = {c.name for c in client.get_collections().collections}
    if COLLECTION not in cols:
        print(f"collection {COLLECTION!r} not found; run `embed` first")
        return 1

    try:
        with Database() as db:
            print(resync(client, db, dry_run=args.dry_run))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
