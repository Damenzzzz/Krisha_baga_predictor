"""Extract the embedded ``window.data`` JSON object from a page."""
from __future__ import annotations

import json

from .selectors import WINDOW_DATA_RE


def extract_window_data(html: str) -> dict | None:
    m = WINDOW_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
