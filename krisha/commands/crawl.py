"""`crawl`: walk search pages for a city and store listing id stubs.

List pages are served fine over httpx. Pagination is driven by
``window.data.search`` (currentPage / nbTotal). Re-running is cheap: seen ids
are just refreshed (last_seen_at)."""
from __future__ import annotations

from .. import config
from ..controller import Controller, LayoutChangeStop
from ..db import Database
from ..logging_setup import get_logger
from ..parsers import parse_list

log = get_logger("crawl")


def run(db: Database, ctrl: Controller, city: str, max_pages: int | None = None) -> dict:
    if city not in config.CITY_PATHS:
        raise ValueError(f"unknown city {city!r}; choose from {list(config.CITY_PATHS)}")

    stats = {"pages": 0, "new": 0, "seen": 0}
    page = 1
    total_pages: int | None = max_pages
    referer = config.BASE_URL

    while total_pages is None or page <= total_pages:
        if max_pages and page > max_pages:
            break
        url = config.city_url(city, page)
        result, delay = ctrl.fetch(url, engine="http", referer=referer)
        if not result.ok:
            ctrl.record(result, result.classify(), delay)
            log.warning("stopping crawl at page %d (status=%s)", page, result.status_code)
            break

        parsed = parse_list(result.text)
        if not parsed.ids:
            ctrl.record(result, "empty_parse", delay)
            log.info("no ids on page %d; stopping", page)
            break
        ctrl.record(result, "ok", delay)

        for cid in parsed.ids:
            card = parsed.cards.get(cid)
            card_data = None
            if card:
                card_data = {"rooms": card.rooms, "area_total": card.area_total,
                             "price_kzt": card.price_kzt, "address": card.address}
            is_new = db.upsert_stub(cid, config.detail_url(cid), city, card_data)
            stats["new" if is_new else "seen"] += 1

        stats["pages"] += 1
        # derive last page from nbTotal on the first page
        if total_pages is None:
            total_pages = parsed.page_count
            log.info("city=%s nb_total=%d -> %d pages", city, parsed.nb_total, total_pages)
        referer = url
        page += 1

    log.info("crawl done: %s", stats)
    return stats
