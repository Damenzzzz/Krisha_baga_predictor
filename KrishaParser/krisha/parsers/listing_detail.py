"""Parse a detail page (/a/show/{id}).

Primary source is the embedded ``window.data.advert`` JSON (price, rooms,
square, geo, address, photos). Characteristics not in the JSON (building type,
year, kitchen area, renovation, bathroom, balcony, furniture) come from the
``offer__info-item[data-name]`` blocks. Every field is optional.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import dom, normalize as N
from .selectors import (DESCRIPTION, INFO_ITEM, INFO_TITLE, INFO_VALUE, PARAM_MAP,
                        PARAMETER_ROW, PARAMETER_NAME, PARAMETER_VALUE, TITLE_H1)
from .window_data import extract_window_data

_CREATED_RE = re.compile(r'"createdAt"\s*:\s*"([^"]+)"')
PARSER_VERSION = 3


@dataclass
class DetailResult:
    fields: dict[str, Any] = field(default_factory=dict)
    photos: list[dict[str, Any]] = field(default_factory=list)
    parsed: bool = False  # True if we recognized the page structure


def extract_characteristics(html: str) -> list[dict[str, Any]]:
    """Read both the summary cards and the description's definition lists."""
    return _characteristics(dom.parse(html))


def _characteristics(tree) -> list[dict[str, Any]]:
    rows = []
    for selector, name_selector, value_selector in (
        (INFO_ITEM, None, INFO_VALUE),
        (PARAMETER_ROW, PARAMETER_NAME, PARAMETER_VALUE),
    ):
        for item in tree.css(selector):
            name_node = item.css_first(name_selector) if name_selector else item
            if name_node is None:
                continue
            name = name_node.attr("data-name")
            title = name_node if name_selector else item.css_first(INFO_TITLE)
            values = [N.clean(v.text()) for v in item.css(value_selector)]
            rows.append({"data-name": name, "title": title.text() if title else None,
                         "value": ", ".join(v for v in values if v) or None,
                         "column": PARAM_MAP.get(name), "source": selector})
    return rows


def parse_detail(html: str, url: str | None = None) -> DetailResult:
    res = DetailResult()
    data = extract_window_data(html)
    advert = (data or {}).get("advert") if isinstance(data, dict) else None
    tree = dom.parse(html)

    f: dict[str, Any] = {"deal_type": "rent", "rent_period": "month"}
    if url:
        f["url"] = url

    if isinstance(advert, dict):
        res.parsed = True
        f["raw_json"] = json.dumps(advert, ensure_ascii=False)
        f["price_kzt"] = N.to_int(advert.get("price"))
        f["rooms"] = N.to_int(advert.get("rooms"))
        f["area_total"] = N.to_float(advert.get("square"))
        _map = advert.get("map") or {}
        f["lat"] = N.to_float(_map.get("lat"))
        f["lon"] = N.to_float(_map.get("lon"))
        addr = advert.get("address") or {}
        f["city"] = N.clean(addr.get("city"))
        f["district"] = N.clean(addr.get("district"))
        f["address"] = N.clean(advert.get("addressTitle")) or _join_address(addr)
        # rent period guard: daily listings live in a different section.
        if advert.get("sectionAlias") and advert.get("sectionAlias") != "arenda":
            f["rent_period"] = "day"
        # photos
        for i, ph in enumerate(advert.get("photos") or []):
            src = ph.get("src")
            if src:
                res.photos.append({"idx": i, "url": src,
                                   "width": N.to_int(ph.get("w")),
                                   "height": N.to_int(ph.get("h"))})
        f["photo_urls"] = [p["url"] for p in res.photos]
        f["photos_count"] = len(res.photos)

    # Title (rooms / floor fallback)
    h1 = tree.css_first(TITLE_H1)
    title = h1.text() if h1 else (advert.get("title") if isinstance(advert, dict) else None)
    if not f.get("rooms"):
        f["rooms"] = N.parse_rooms(title)
    floor, floors_total = N.parse_floor(title)

    # Characteristics from offer__info-item blocks
    reno_parts: list[str] = []
    balcony_parts: list[str] = []
    for characteristic in _characteristics(tree):
        col = characteristic["column"]
        value = characteristic["value"]
        if not value:
            continue
        if col in (None, "_ignore"):
            continue
        if col == "_floor":
            fl, ft = N.parse_floor(value)
            floor = fl or floor
            floors_total = ft or floors_total
        elif col == "_square":
            areas = N.parse_areas(value)
            f.setdefault("area_total", None)
            if not f.get("area_total"):
                f["area_total"] = areas["area_total"]
            f["area_kitchen"] = areas["area_kitchen"]
            f["area_living"] = areas["area_living"]
        elif col == "renovation_text":
            if value:
                reno_parts.append(value)
        elif col == "_renovation_extra":
            if value:
                reno_parts.append(value)
        elif col == "year_built":
            f[col] = N.to_int(value)
            res.parsed = True
        elif col in ("_balcony_count", "_loggia_count"):
            label = "балкон" if col == "_balcony_count" else "лоджия"
            balcony_parts.append(f"{label}: {value}")
            res.parsed = True
        else:
            f[col] = value
            res.parsed = True

    if floor is not None:
        f["floor"] = floor
    if floors_total is not None:
        f["floors_total"] = floors_total
    if reno_parts:
        f["renovation_text"] = " — ".join(reno_parts)
    if balcony_parts:
        f["balcony"] = "; ".join(balcony_parts)

    # Description
    desc = tree.css_first(DESCRIPTION)
    if desc:
        text = desc.text()
        # strip a leading "О квартире" heading if present
        text = re.sub(r"^О квартире\s*", "", text).strip()
        f["description"] = N.clean(text)

    # Published date
    m = _CREATED_RE.search(html)
    if m:
        f["published_at"] = m.group(1)

    res.fields = {k: v for k, v in f.items() if v is not None}
    return res


def _join_address(addr: dict) -> str | None:
    parts = [addr.get("city"), addr.get("district"), addr.get("street"),
             addr.get("house_num")]
    parts = [str(p) for p in parts if p]
    return ", ".join(parts) if parts else None
