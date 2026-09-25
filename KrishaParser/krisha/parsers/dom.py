"""Tiny DOM adapter: selectolax fast-path, BeautifulSoup fallback.

Exposes a uniform Node with .css(sel), .text(), .attr(name), .html so parsers
don't care which backend is installed.
"""
from __future__ import annotations

from typing import List, Optional

try:
    from selectolax.parser import HTMLParser as _SLX  # type: ignore
    _BACKEND = "selectolax"
except Exception:  # noqa: BLE001
    _SLX = None
    _BACKEND = "bs4"

if _BACKEND == "bs4":
    from bs4 import BeautifulSoup  # type: ignore


class Node:
    def __init__(self, raw, backend: str):
        self._raw = raw
        self._b = backend

    def css(self, selector: str) -> List["Node"]:
        if self._b == "selectolax":
            return [Node(n, self._b) for n in self._raw.css(selector)]
        return [Node(n, self._b) for n in self._raw.select(selector)]

    def css_first(self, selector: str) -> Optional["Node"]:
        nodes = self.css(selector)
        return nodes[0] if nodes else None

    def text(self, strip: bool = True) -> str:
        if self._b == "selectolax":
            t = self._raw.text(separator=" ")
        else:
            t = self._raw.get_text(separator=" ")
        return " ".join(t.split()) if strip else t

    def attr(self, name: str) -> Optional[str]:
        if self._b == "selectolax":
            return self._raw.attributes.get(name)
        return self._raw.get(name)


def parse(html: str) -> Node:
    if _BACKEND == "selectolax":
        return Node(_SLX(html), "selectolax")
    return Node(BeautifulSoup(html, "lxml"), "bs4")
