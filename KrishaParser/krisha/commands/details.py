"""`details`: fetch each listing's detail page (Playwright) and parse fields.

Detail pages return an anti-bot 468 over plain HTTP, so this always uses the
Playwright engine. Idempotent: only listings with detail_fetched_at IS NULL are
fetched, so a re-run resumes where it stopped."""
from __future__ import annotations

from .. import config
from ..controller import Controller
from ..db import Database
from ..fetcher import cache
from ..fetcher.base import FetchResult
from ..logging_setup import get_logger
from ..parsers import parse_detail
from ..parsers.listing_detail import PARSER_VERSION

log = get_logger("details")


def run(db: Database, ctrl: Controller, city: str | None = None,
        limit: int | None = None, refresh_missing: bool = False,
        cached_only: bool = False, listing_ids: list[int] | None = None) -> dict:
    ids = (listing_ids if listing_ids is not None else
           db.ids_without_detail(city=city, limit=limit, refresh_missing=refresh_missing))
    stats = {"parsed": 0, "empty": 0, "photos": 0, "cache_miss": 0}
    log.info("details: %d listings to fetch", len(ids))
    referer = config.city_url(city) if city else config.BASE_URL

    for lid in ids:
        url = config.detail_url(lid)
        if cached_only:
            html = cache.get(url)
            if html is None:
                stats["cache_miss"] += 1
                continue
            result = FetchResult(url, 200, html, "cache", 0, from_cache=True)
            delay = 0
        else:
            result, delay = ctrl.fetch(url, engine="playwright", referer=referer)
        if not result.ok:
            ctrl.record(result, result.classify(), delay)
            continue

        detail = parse_detail(result.text, url=url)
        if not detail.parsed:
            ctrl.record(result, "empty_parse", delay)
            stats["empty"] += 1
            continue

        cache.put(url, result.text)
        save_detail(db, lid, detail)
        stats["photos"] += len(detail.photos)
        ctrl.record(result, "ok", delay)
        stats["parsed"] += 1
        if stats["parsed"] % 25 == 0:
            log.info("details progress: %d/%d parsed", stats["parsed"], len(ids))

    log.info("details done: %s", stats)
    return stats


def save_detail(db: Database, lid: int, detail) -> None:
    """Merge gallery by URL so detail order cannot overwrite downloaded photo IDs."""
    existing = db.photos_for(lid)
    by_url = {p["url"]: p["idx"] for p in existing}
    next_idx = max((p["idx"] for p in existing), default=-1) + 1
    urls = [p["url"] for p in existing if p["url"]]
    for ph in detail.photos:
        idx = by_url.get(ph["url"])
        if idx is None:
            idx = next_idx
            next_idx += 1
            by_url[ph["url"]] = idx
            urls.append(ph["url"])
        db.upsert_photo(lid, idx, url=ph["url"],
                        width=ph.get("width"), height=ph.get("height"))
    fields = dict(detail.fields)
    if urls:
        fields.update(photo_urls=urls, photos_count=len(urls))
    db.update_detail(lid, fields)
    db.conn.execute("UPDATE listings SET detail_parser_version=? WHERE id=?",
                    (PARSER_VERSION, lid))
    db.commit()
