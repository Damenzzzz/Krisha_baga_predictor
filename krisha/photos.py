"""Download listing photos to data/photos/{listing_id}/{idx}.jpg with sha256
dedup. If the same image bytes already exist for another listing, we reuse the
existing file path instead of writing a copy."""
from __future__ import annotations

import hashlib

from . import config
from .db import Database
from .fetcher.http_engine import HttpEngine
from .logging_setup import get_logger

log = get_logger("photos")


def download_pending(db: Database, engine: HttpEngine | None = None,
                     limit: int | None = None) -> dict[str, int]:
    own_engine = engine is None
    engine = engine or HttpEngine()
    stats = {"downloaded": 0, "deduped": 0, "failed": 0}
    try:
        pending = db.photos_pending_download(limit=limit)
        for row in pending:
            lid, idx, url = row["listing_id"], row["idx"], row["url"]
            status, content = engine.get_bytes(url, referer=config.detail_url(lid))
            if not content or (status and status != 200):
                stats["failed"] += 1
                log.warning("photo failed %s (status=%s)", url, status)
                continue
            sha = hashlib.sha256(content).hexdigest()
            existing = db.sha_exists(sha)
            if existing:
                db.upsert_photo(lid, idx, sha256=sha,
                                local_path=existing["local_path"],
                                downloaded_at=_now())
                stats["deduped"] += 1
                continue
            dest_dir = config.PHOTOS_DIR / str(lid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{idx}.jpg"
            dest.write_bytes(content)
            db.upsert_photo(lid, idx, sha256=sha, local_path=str(dest),
                            downloaded_at=_now())
            stats["downloaded"] += 1
        return stats
    finally:
        if own_engine:
            engine.close()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
