"""Engine interface shared by the httpx and Playwright fetchers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .. import config


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


@dataclass
class FetchResult:
    url: str
    status_code: int | None
    text: str
    engine: str
    latency_ms: int
    from_cache: bool = False
    retry_after: float | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and bool(self.text)

    @property
    def is_block(self) -> bool:
        if self.status_code in config.BLOCK_STATUS_CODES:
            return True
        return self.is_captcha

    @property
    def is_captcha(self) -> bool:
        low = self.text[:5000]
        return any(m in low for m in config.CAPTCHA_MARKERS)

    def classify(self) -> str:
        """Map to a scrape_events outcome (parse-level outcomes set by caller)."""
        if self.status_code == 429:
            return "http_429"
        if self.status_code in (403, 468):
            return "http_403"
        if self.is_captcha:
            return "captcha"
        if self.status_code is None:
            return "timeout"
        if self.ok:
            return "ok"
        return "parse_error"


class Fetcher(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, url: str, referer: str | None = None) -> FetchResult:
        ...

    def close(self) -> None:  # pragma: no cover - optional
        pass
