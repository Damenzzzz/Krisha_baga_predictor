"""Parse a search/list page: canonical ids + pagination + light card fields.

Pagination uses ``window.data.search`` (currentPage, nbTotal, ids) which is the
authoritative result set for the page (20 items), avoiding the promoted "hot"
cards that also render in the HTML.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .. import config
from . import dom, normalize as N
from .selectors import CARD, CARD_DESCR, CARD_PRICE, CARD_SUBTITLE, CARD_TITLE
from .window_data import extract_window_data

_SHOW_RE = re.compile(r"/a/show/(\d+)")


@dataclass
class Card:
    id: int
    url: str
    price_kzt: int | None = None
    rooms: int | None = None
    area_total: float | None = None
    address: str | None = None


@dataclass
class ListPage:
    ids: list[int] = field(default_factory=list)
    current_page: int = 1
    nb_total: int = 0
    cards: dict[int, Card] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        if self.nb_total <= 0:
            return self.current_page
        return math.ceil(self.nb_total / config.LISTINGS_PER_PAGE)


def parse_list(html: str) -> ListPage:
    page = ListPage()
    data = extract_window_data(html)
    if data and isinstance(data.get("search"), dict):
        s = data["search"]
        page.current_page = N.to_int(s.get("currentPage")) or 1
        page.nb_total = N.to_int(s.get("nbTotal")) or 0
        ids = s.get("ids") or []
        page.ids = [int(i) for i in ids if str(i).isdigit()]

    # Card details (best-effort; detail pages remain the source of truth).
    tree = dom.parse(html)
    for card in tree.css(CARD):
        data_id = card.attr("data-id")
        if not data_id or not data_id.isdigit():
            continue
        cid = int(data_id)
        title_node = card.css_first(CARD_TITLE)
        title = title_node.text() if title_node else None
        price_node = card.css_first(CARD_PRICE)
        sub_node = card.css_first(CARD_SUBTITLE)
        areas = N.parse_areas(title)
        page.cards[cid] = Card(
            id=cid,
            url=config.detail_url(cid),
            price_kzt=N.to_int(price_node.text()) if price_node else None,
            rooms=N.parse_rooms(title),
            area_total=areas["area_total"],
            address=N.clean(sub_node.text()) if sub_node else None,
        )

    # Fallback: if window.data had no ids, take unique ids from card links.
    if not page.ids:
        seen: list[int] = []
        for m in _SHOW_RE.finditer(html):
            v = int(m.group(1))
            if v not in seen:
                seen.append(v)
        page.ids = seen
    return page
