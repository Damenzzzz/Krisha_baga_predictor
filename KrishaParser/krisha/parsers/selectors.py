"""Centralized selectors and the offer parameter map.

Kept in one place so a versta (layout) change is a single-file fix, and so the
tests document exactly what the parser depends on.
"""
from __future__ import annotations

import re

# window.data = {...}; — embedded on both list and detail pages.
WINDOW_DATA_RE = re.compile(r"window\.data\s*=\s*(\{.*?\});", re.S)

# --- list / search page ---
CARD = "div.a-card[data-id]"
CARD_TITLE = ".a-card__title"          # contains the /a/show/{id} link + title
CARD_PRICE = ".a-card__price"
CARD_SUBTITLE = ".a-card__subtitle"    # address line
CARD_DESCR = ".a-card__descr"

# --- detail page ---
INFO_ITEM = "div.offer__info-item"
INFO_TITLE = ".offer__info-title"
INFO_VALUE = ".offer__advert-short-info"
DESCRIPTION = ".offer__description"
TITLE_H1 = "h1"

# data-name of an offer__info-item -> our listing column (or a virtual key
# handled specially in listing_detail.py).
PARAM_MAP: dict[str, str] = {
    "flat.building": "building_type",
    "house.year": "year_built",
    "flat.floor": "_floor",          # "1 из 5" -> floor / floors_total
    "live.square": "_square",        # "74 м², Площадь кухни — 9 м²"
    "flat.renovation": "renovation_text",
    "flat.rent_renovation": "_renovation_extra",
    "flat.toilet": "bathroom",
    "flat.bath": "bathroom",
    "flat.balcony": "balcony",
    "live.furniture": "furniture",
    "flat.furniture": "furniture",
    "map.complex": "complex_name",
    "flat.priedomovaya_territoriya": "_ignore",
}
