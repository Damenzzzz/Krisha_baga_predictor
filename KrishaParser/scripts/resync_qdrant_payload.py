"""Resync mutable payload fields of embedded Qdrant points from the SQLite DB.

Some payload fields are set at embed time but change later:
  - `city` was split by casing until canonicalized (normalize_city);
  - `duplicate_group_id` is assigned by `dedup`, which usually runs AFTER embed,
    so freshly embedded points carry a stale/NULL group.

This scrolls every point, reads its `listing_id` from the payload, looks up the
current DB values, and patches only the points whose `city` or
`duplicate_group_id` differ. Idempotent; safe to re-run after any crawl/embed/dedup.

Usage (targets QDRANT_URL from .env — local Docker or cloud):
    python -m scripts.resync_qdrant_payload [--dry-run]
"""
from __future__ import annotations

import argparse
import sys

from krisha.db import Database
from krisha.embed_worker import COLLECTION, make_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    client = make_client()
    cols = {c.name for c in client.get_collections().collections}
    if COLLECTION not in cols:
        print(f"collection {COLLECTION!r} not found; run `embed` first")
        return 1

    db = Database()
    # listing_id -> (city, duplicate_group_id) authoritative values
    truth = {r["id"]: (r["city"], r["duplicate_group_id"])
             for r in db.conn.execute(
                 "SELECT id, city, duplicate_group_id FROM listings")}

    # Group stale point ids by listing so all a listing's photos are patched in
    # one call (a listing's points share city + duplicate_group_id).
    scanned = missing = 0
    stale: dict[int, list] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION, limit=1000, offset=offset,
            with_payload=["listing_id", "city", "duplicate_group_id"],
            with_vectors=False)
        for p in points:
            scanned += 1
            pl = p.payload or {}
            lid = pl.get("listing_id")
            if lid not in truth:
                missing += 1
                continue
            city, dgid = truth[lid]
            if pl.get("city") != city or pl.get("duplicate_group_id") != dgid:
                stale.setdefault(lid, []).append(p.id)
        if offset is None:
            break

    points_fixed = sum(len(v) for v in stale.values())
    verb = "would fix" if args.dry_run else "fixed"
    print(f"scanned {scanned} points; {verb} {points_fixed} points across "
          f"{len(stale)} listings; {missing} points had no matching DB listing")
    if args.dry_run:
        return 0

    for lid, ids in stale.items():
        city, dgid = truth[lid]
        for i in range(0, len(ids), 1000):
            client.set_payload(COLLECTION,
                               payload={"city": city, "duplicate_group_id": dgid},
                               points=ids[i:i + 1000], wait=False)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
