"""Small, defensive text/number normalizers. Every function returns None on
failure rather than raising, so the parser never crashes on odd input."""
from __future__ import annotations

import re

_DIGITS = re.compile(r"\d[\d\s ]*")


def clean(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    return t or None


# Canonical (lowercase, ASCII) city keys — match config.CITY_PATHS and the CLI
# --city argument. The crawl path stores the CLI arg ("almaty"); the detail path
# stores window.data's addressTitle ("Almaty"/"Алматы"), which split the same
# city across values and broke city filtering (incl. the Qdrant payload index).
_CITY_ALIASES: dict[str, str] = {
    "almaty": "almaty",
    "алматы": "almaty",
    "kaskelen": "kaskelen",
    "каскелен": "kaskelen",
}


def normalize_city(value: str | None) -> str | None:
    """Canonicalize a city name to a single lowercase key so crawl (CLI arg)
    and detail (JSON) paths never split the same city. Unknown cities are just
    lowercased/cleaned so nothing is dropped."""
    c = clean(value)
    if not c:
        return None
    return _CITY_ALIASES.get(c.lower(), c.lower())


def to_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = _DIGITS.search(str(value))
    if not m:
        return None
    try:
        return int(re.sub(r"[\s ]", "", m.group(0)))
    except ValueError:
        return None


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(?:[.,]\d+)?", str(value).replace(" ", " "))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_rooms(title: str | None) -> int | None:
    """'3-комнатная квартира ...' -> 3."""
    if not title:
        return None
    m = re.search(r"(\d+)\s*-?\s*комнат", title)
    return int(m.group(1)) if m else None


def parse_floor(text: str | None) -> tuple[int | None, int | None]:
    """'1 из 5' or '1/5 этаж' -> (1, 5)."""
    if not text:
        return None, None
    m = re.search(r"(\d+)\s*(?:из|/)\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\b(\d+)\b", text)
    return (int(m.group(1)), None) if m else (None, None)


def parse_areas(text: str | None) -> dict[str, float | None]:
    """'74 м², Площадь кухни — 9 м²' -> total/kitchen/living where present."""
    out: dict[str, float | None] = {"area_total": None, "area_kitchen": None,
                                     "area_living": None}
    if not text:
        return out
    total = re.search(r"(\d+(?:[.,]\d+)?)\s*м", text)
    if total:
        out["area_total"] = to_float(total.group(1))
    kit = re.search(r"кухн[а-я]*\D*(\d+(?:[.,]\d+)?)", text, re.I)
    if kit:
        out["area_kitchen"] = to_float(kit.group(1))
    liv = re.search(r"жил[а-я]*\D*(\d+(?:[.,]\d+)?)", text, re.I)
    if liv:
        out["area_living"] = to_float(liv.group(1))
    return out
