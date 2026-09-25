"""`details`: fetch each listing's detail page (Playwright) and parse fields.

Detail pages return an anti-bot 468 over plain HTTP, so this always uses the
Playwright engine. Idempotent: only listings with detail_fetched_at IS NULL are
fetched, so a re-run resumes where it stopped."""
from __future__ import annotations

from .. import config
from ..controller import Controller
from ..db import Database
from ..logging_setup import get_logger
from ..parsers import parse_detail

log = get_logger("details")


def run(db: Database, ctrl: Controller, city: str | None = None,
        limit: int | None = None) -> dict:
    ids = db.ids_without_detail(city=city, limit=limit)
    stats = {"parsed": 0, "empty": 0, "photos": 0}
    log.info("details: %d listings to fetch", len(ids))
    referer = config.city_url(city) if city else config.BASE_URL

    for lid in ids:
        url = config.detail_url(lid)
        result, delay = ctrl.fetch(url, engine="playwright", referer=referer)
        if not result.ok:
            ctrl.record(result, result.classify(), delay)
            continue

        detail = parse_detail(result.text, url=url)
        if not detail.parsed:
            ctrl.record(result, "empty_parse", delay)
            stats["empty"] += 1
            continue

        db.update_detail(lid, detail.fields)
        for ph in detail.photos:
            db.upsert_photo(lid, ph["idx"], url=ph["url"],
                            width=ph.get("width"), height=ph.get("height"))
            stats["photos"] += 1
        ctrl.record(result, "ok", delay)
        stats["parsed"] += 1

    log.info("details done: %s", stats)
    return stats
