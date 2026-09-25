"""Fallback engine: real Chromium via Playwright (sync API).

Required for detail pages (/a/show/) which return an anti-bot 468 to plain HTTP
clients. Browser + context are launched lazily and reused across requests.
"""
from __future__ import annotations

import time
import os

from .. import config
from ..logging_setup import get_logger
from .base import Fetcher, FetchResult, parse_retry_after
from .proxy import load_proxies
from . import warp

log = get_logger("playwright")


class PlaywrightEngine(Fetcher):
    name = "playwright"

    def __init__(self, headless: bool | None = None) -> None:
        self._headless = config.PW_HEADLESS if headless is None else headless
        self._pw = None
        self._browser = None
        self._ctx = None
        self._warmed = False
        self._proxies = load_proxies()
        try:
            self._proxy_index = int(os.getenv("KRISHA_PROXY_START_INDEX", "0"))
            if not 0 <= self._proxy_index < max(1, len(self._proxies)):
                raise ValueError
        except ValueError:
            raise ValueError("KRISHA_PROXY_START_INDEX is outside the configured proxy list") from None
        self._requests = 0
        try:
            self._rotate_every = max(0, int(os.getenv("KRISHA_WARP_ROTATE_EVERY", "0")))
        except ValueError:
            raise ValueError("KRISHA_WARP_ROTATE_EVERY must be an integer") from None

    def _ensure(self) -> None:
        if self._ctx is not None:
            return
        from playwright.sync_api import sync_playwright  # lazy import

        self._pw = sync_playwright().start()
        options = {"headless": self._headless}
        if self._proxies:
            options["proxy"] = self._proxies[self._proxy_index]
        try:
            self._browser = self._pw.chromium.launch(**options)
        except Exception:
            self.close()
            raise RuntimeError("Browser launch failed; check Playwright installation and proxy configuration") from None
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
            log.debug("pw warmup failed: %s", type(e).__name__)
        finally:
            page.close()

    def fetch(self, url: str, referer: str | None = None) -> FetchResult:
        if (self._rotate_every and self._requests >= self._rotate_every
                and self._proxies and warp.enabled(self._proxies[self._proxy_index])):
            self.rotate_proxy()
        self._ensure()
        self._warmup()
        page = self._ctx.new_page()
        t0 = time.monotonic()
        status: int | None = None
        text = ""
        retry_after = None
        try:
            resp = page.goto(url, wait_until="domcontentloaded",
                             timeout=int(config.REQUEST_TIMEOUT_S * 1000),
                             referer=referer)
            status = resp.status if resp else None
            retry_after = parse_retry_after(resp.header_value("retry-after")) if resp else None
            text = page.content()
        except Exception as e:  # noqa: BLE001 - navigation/timeout
            log.warning("playwright error %s: %s", url, type(e).__name__)
            status = status or None
            if self._headless:
                self._debug_screenshot(page, url)
        finally:
            latency = int((time.monotonic() - t0) * 1000)
            page.close()
        self._requests += 1
        return FetchResult(url, status, text, self.name, latency, retry_after=retry_after)

    def rotate_proxy(self) -> bool:
        """Advance through configured routes once; never silently fall back direct."""
        if self._proxy_index + 1 >= len(self._proxies):
            if self._proxies and warp.enabled(self._proxies[self._proxy_index]):
                self.close()
                warp.reconnect()
                self._requests = 0
                log.info("reconnected WARP local proxy; browser session reset")
                return True
            return False
        self.close()
        self._proxy_index += 1
        self._requests = 0
        log.info("using configured proxy %d/%d", self._proxy_index + 1, len(self._proxies))
        return True

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
        self._warmed = False
