"""`photos`: download pending listing photos (httpx) with sha256 dedup."""
from __future__ import annotations

from ..db import Database
from ..logging_setup import get_logger
from ..photos import download_pending

log = get_logger("photos-cmd")


def run(db: Database, limit: int | None = None) -> dict:
    stats = download_pending(db, limit=limit)
    print(f"photos: {stats}")
    return stats
