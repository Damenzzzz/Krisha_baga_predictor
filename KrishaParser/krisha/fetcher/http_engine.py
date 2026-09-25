"""Primary engine: httpx.Client with a real cookie session.

Works for list/search pages and images. Detail pages (/a/show/) are served an
anti-bot 468 to non-browser clients, so those go through the Playwright engine.
"""
from __future__ import annotations

import time

import httpx

from .. import config
from ..logging_setup import get_logger
from .base import Fetcher, FetchResult, parse_retry_after

log = get_logger("http")


class HttpEngine(Fetcher):
    name = "http"

    def __init__(self) -> None:
        self._client = httpx.Client(
            headers=config.default_headers(),
            timeout=httpx.Timeout(config.REQUEST_TIMEOUT_S),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=config.MAX_CONCURRENCY),
        )
        self._warmed = False

    def _warmup(self) -> None:
        """Hit the homepage once to obtain krssid/krishauid cookies."""
        if self._warmed:
            return
        try:
            self._client.get(config.BASE_URL)
        except httpx.HTTPError as e:  # pragma: no cover
            log.warning("warmup failed: %s", e)
        self._warmed = True

    def fetch(self, url: str, referer: str | None = None) -> FetchResult:
        self._warmup()
        headers = config.default_headers(referer)
        t0 = time.monotonic()
        try:
            r = self._client.get(url, headers=headers)
            latency = int((time.monotonic() - t0) * 1000)
            retry_after = parse_retry_after(r.headers.get("Retry-After"))
            return FetchResult(url, r.status_code, r.text, self.name, latency,
                               retry_after=retry_after)
        except httpx.HTTPError as e:
            latency = int((time.monotonic() - t0) * 1000)
            log.warning("http error %s: %s", url, e)
            return FetchResult(url, None, "", self.name, latency)

    def head(self, url: str) -> int | None:
        """Cheap existence check (used to discover a listing's full photo set)."""
        self._warmup()
        try:
            r = self._client.head(url, headers=config.default_headers())
            return r.status_code
        except httpx.HTTPError:
            return None

    def get_bytes(self, url: str, referer: str | None = None) -> tuple[int | None, bytes]:
        """Download binary content (photos)."""
        self._warmup()
        try:
            r = self._client.get(url, headers=config.default_headers(referer))
            return r.status_code, r.content
        except httpx.HTTPError as e:
            log.warning("image fetch error %s: %s", url, e)
            return None, b""

    def close(self) -> None:
        self._client.close()
