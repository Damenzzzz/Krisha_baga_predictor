"""Disk cache of raw responses keyed by sha1(url).

Lets a re-run skip already-downloaded pages (checkpoint/resume at the HTTP
layer, complementing the DB-level dedup).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .. import config


def _key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def path_for(url: str) -> Path:
    return config.CACHE_DIR / f"{_key(url)}.html"


def get(url: str) -> str | None:
    p = path_for(url)
    if p.exists() and p.stat().st_size > 0:
        return p.read_text(encoding="utf-8", errors="replace")
    return None


def put(url: str, text: str) -> None:
    path_for(url).write_text(text, encoding="utf-8", errors="replace")
