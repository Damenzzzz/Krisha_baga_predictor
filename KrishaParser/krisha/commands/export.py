"""`export --csv`: dump monthly listings with detail for ML.

One row per listing; one representative per duplicate_group_id is kept (the
lowest id) so a single flat cannot leak across train/test."""
from __future__ import annotations

import csv
from datetime import datetime

from .. import config
from ..db import Database
from ..logging_setup import get_logger

log = get_logger("export")

COLUMNS = [
    "id", "url", "rent_period", "price_kzt", "rooms", "area_total", "area_living",
    "area_kitchen", "floor", "floors_total", "year_built", "building_type",
    "complex_name", "city", "district", "address", "lat", "lon", "furniture",
    "bathroom", "balcony", "renovation_text", "photos_count", "published_at",
    "duplicate_group_id", "description",
]


def run(db: Database, out_path: str | None = None, dedup: bool = True) -> str:
    rows = db.all_listings()
    seen_groups: set[str] = set()
    out_rows = []
    for r in rows:
        if r["rent_period"] not in (None, "month"):
            continue
        # Keep rows usable for ML: either detail-enriched, or with the core
        # card fields (price + area). Detail pages may be unavailable (anti-bot),
        # in which case card-level data still forms a valid dataset.
        has_detail = r["detail_fetched_at"] is not None
        has_core = r["price_kzt"] is not None and r["area_total"] is not None
        if not (has_detail or has_core):
            continue
        gid = r["duplicate_group_id"]
        if dedup and gid:
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
        out_rows.append({c: r[c] for c in COLUMNS})

    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(config.EXPORTS_DIR / f"listings_{ts}.csv")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    log.info("exported %d rows -> %s", len(out_rows), out_path)
    print(f"exported {len(out_rows)} listings -> {out_path}")
    return out_path
