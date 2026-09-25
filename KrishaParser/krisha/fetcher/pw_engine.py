"""Fallback engine: real Chromium via Playwright (sync API).

Required for detail pages (/a/show/) which return an anti-bot 468 to plain HTTP
clients. Browser + context are launched lazily and reused across requests.
"""
from __future__ import annotations

import time

from .. import config
from ..logging_setup import get_logger
from .base import Fetcher, FetchResult

log = get_logger("playwright")


class PlaywrightEngine(Fetcher):
    name = "playwright"

    def __init__(self, headless: bool | None = None) -> None:
        self._headless = config.PW_HEADLESS if headless is None else headless
        self._pw = None
        self._browser = None
        self._ctx = None
        self._warmed = False

    def _ensure(self) -> None:
        if self._ctx is not None:
            return
        from playwright.sync_api import sync_playwright  # lazy import

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._ctx = self._browser.new_context(
            user_agent=config.USER_AGENT,
            locale="ru-RU",
            viewport={"width": 1366, "height": 900},
        )

    def _warmup(self) -> None:
        """Visit the homepage once so the context looks like a real session
        (cookies, referer chain) before requesting protected detail pages."""
        if self._warmed:
            return
        self._warmed = True
        page = self._ctx.new_page()
        try:
            page.goto(config.BASE_URL, wait_until="domcontentloaded",
                     timeout=int(config.REQUEST_TIMEOUT_S * 1000))
        except Exception as e:  # noqa: BLE001
            log.debug("pw warmup failed: %s", e)
        finally:
            page.close()

    def fetch(self, url: str, referer: str | None = None) -> FetchResult:
        self._ensure()
        self._warmup()
        page = self._ctx.new_page()
        t0 = time.monotonic()
        status: int | None = None
        text = ""
        try:
            resp = page.goto(url, wait_until="domcontentloaded",
                             timeout=int(config.REQUEST_TIMEOUT_S * 1000),
                             referer=referer)
            status = resp.status if resp else None
            text = page.content()
        except Exception as e:  # noqa: BLE001 - navigation/timeout
            log.warning("playwright error %s: %s", url, e)
            status = status or None
            if self._headless:
                self._debug_screenshot(page, url)
        finally:
            latency = int((time.monotonic() - t0) * 1000)
            page.close()
        return FetchResult(url, status, text, self.name, latency)

    def _debug_screenshot(self, page, url: str) -> None:
        try:
            fname = config.LOGS_DIR / f"debug_stop_{int(time.time())}.png"
            page.screenshot(path=str(fname))
            log.info("saved debug screenshot %s for %s", fname, url)
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            if self._ctx:
                self._ctx.close()
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        self._ctx = self._browser = self._pw = None
