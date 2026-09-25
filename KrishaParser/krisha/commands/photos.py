"""`photos`: download pending listing photos (httpx) with sha256 dedup."""
from __future__ import annotations

from ..db import Database
from ..fetcher.http_engine import HttpEngine
from ..logging_setup import get_logger
from ..photos import download_pending, expand_photos

log = get_logger("photos-cmd")


def run(db: Database, limit: int | None = None, expand: bool = False) -> dict:
    engine = HttpEngine()
    try:
        if expand:
            exp = expand_photos(db, engine=engine)
            print(f"expand: {exp}")
        stats = download_pending(db, engine=engine, limit=limit)
    finally:
        engine.close()
    print(f"photos: {stats}")
    return stats
