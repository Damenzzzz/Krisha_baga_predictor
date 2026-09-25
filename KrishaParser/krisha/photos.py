"""Download listing photos to data/photos/{listing_id}/{idx}.jpg with sha256
dedup. If the same image bytes already exist for another listing, we reuse the
existing file path instead of writing a copy.

Networking here targets the public image CDN (krisha-photos.kcdn.online), a
static asset host separate from the krisha.kz anti-bot site. Modest parallelism
is polite and normal for a CDN, so photo I/O uses a small thread pool
(CDN_CONCURRENCY); the krisha.kz max-concurrency=2 rule applies to the anti-bot
site, not this CDN. Network fetches run in worker threads; all SQLite writes are
serialized in the calling thread (single-writer)."""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from . import config
from .db import Database
from .fetcher.http_engine import HttpEngine
from .logging_setup import get_logger

log = get_logger("photos")

CDN_CONCURRENCY = 8

# All photos of a listing share one CDN stem, indexed 1..N:
#   https://krisha-photos.kcdn.online/webp/xx/<uuid>/<n>-full.jpg
_STEM_RE = re.compile(r"^(https://krisha-photos\.kcdn\.online/[a-z]+/[0-9a-f]{2}/[0-9a-f-]{36})/\d+-\w+\.jpg")


def _walk_stem(engine: HttpEngine, stem: str, cap: int) -> list[str]:
    """HEAD-walk a listing's album until the first missing index."""
    urls: list[str] = []
    for i in range(1, cap + 1):
        u = f"{stem}/{i}-full.jpg"
        if engine.head(u) != 200:
            break
        urls.append(u)
    return urls


def expand_photos(db: Database, engine: HttpEngine | None = None, cap: int = 40,
                  limit: int | None = None,
                  concurrency: int = CDN_CONCURRENCY) -> dict[str, int]:
    """Discover the FULL photo set of each listing by walking CDN indices.

    The search card only exposes photo #1; the detail page (which enumerates the
    album) is anti-bot blocked. Because every photo lives under the same public
    CDN stem as photo #1, we walk `{stem}/{i}-full.jpg` until a 404 and register
    every photo — no detail page, no evasion. Idempotent + resumable."""
    own = engine is None
    engine = engine or HttpEngine()
    stats = {"listings": 0, "photos_added": 0}
    try:
        q = ("SELECT l.id AS lid, p.url AS first_url FROM listings l "
             "JOIN photos p ON p.listing_id=l.id AND p.idx=0 "
             "WHERE p.url LIKE '%kcdn%' AND (l.photos_count IS NULL OR l.photos_count<=2)")
        if limit:
            q += f" LIMIT {int(limit)}"
        targets = []
        for row in db.conn.execute(q).fetchall():
            m = _STEM_RE.match(row["first_url"])
            if m:
                targets.append((row["lid"], m.group(1)))

        def work(t):
            lid, stem = t
            return lid, _walk_stem(engine, stem, cap)

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for lid, urls in ex.map(work, targets):  # DB writes here (main thread)
                if not urls:
                    continue
                for i, u in enumerate(urls):
                    db.upsert_photo(lid, i, url=u)
                db.conn.execute(
                    "UPDATE listings SET photo_urls=?, photos_count=? WHERE id=?",
                    (json.dumps(urls, ensure_ascii=False), len(urls), lid),
                )
                db.commit()
                stats["listings"] += 1
                stats["photos_added"] += max(0, len(urls) - 1)
        log.info("expand_photos: %s", stats)
        return stats
    finally:
        if own:
            engine.close()


def download_pending(db: Database, engine: HttpEngine | None = None,
                     limit: int | None = None,
                     concurrency: int = CDN_CONCURRENCY) -> dict[str, int]:
    own_engine = engine is None
    engine = engine or HttpEngine()
    stats = {"downloaded": 0, "deduped": 0, "failed": 0}
    try:
        pending = db.photos_pending_download(limit=limit)

        def fetch(row):
            status, content = engine.get_bytes(
                row["url"], referer=config.detail_url(row["listing_id"]))
            return row, status, content

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for row, status, content in ex.map(fetch, pending):  # writes: main thread
                lid, idx = row["listing_id"], row["idx"]
                if not content or (status and status != 200):
                    stats["failed"] += 1
                    log.warning("photo failed %s (status=%s)", row["url"], status)
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
        log.info("download_pending: %s", stats)
        return stats
    finally:
        if own_engine:
            engine.close()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
