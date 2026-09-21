"""Central configuration: paths, hard scraping limits, URL builders, headers.

All tunables come from environment (.env) with safe defaults. The politeness
limits here are HARD RULES from the project brief and must not be relaxed for
speed: max concurrency 2, realistic UA, no proxies, no captcha solving.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../KrishaParser
load_dotenv(PROJECT_ROOT / ".env")

# DATA_DIR defaults to <repo>/data but can be pointed elsewhere via KRISHA_DATA_DIR.
# This lets a git worktree (whose own data/ is gitignored and empty) run against
# the primary checkout's DB + photos without copying gigabytes around.
DATA_DIR = Path(os.getenv("KRISHA_DATA_DIR", str(PROJECT_ROOT / "data")))
CACHE_DIR = DATA_DIR / "cache"
PHOTOS_DIR = DATA_DIR / "photos"
EXPORTS_DIR = DATA_DIR / "exports"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

DB_PATH = DATA_DIR / "krisha.db"
LOG_FILE = LOGS_DIR / "krisha.log"
RUNTIME_PROFILE_PATH = CONFIG_DIR / "runtime_profile.json"

for _d in (DATA_DIR, CACHE_DIR, PHOTOS_DIR, EXPORTS_DIR, LOGS_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# --- Hard scraping limits (do not raise) -----------------------------------
MAX_CONCURRENCY = min(2, _int("KRISHA_MAX_CONCURRENCY", 2))  # capped at 2, always
MIN_DELAY_MS = _int("KRISHA_MIN_DELAY_MS", 1500)
MAX_DELAY_MS = _int("KRISHA_MAX_DELAY_MS", 6000)
START_DELAY_MS = _int("KRISHA_START_DELAY_MS", 2500)

REQUEST_TIMEOUT_S = 40.0
MAX_RETRIES = 2
COOLDOWN_S = 90.0            # after a ban/captcha
EMPTY_PARSE_STOP_STREAK = 8  # sustained empty parses => probable layout change

USER_AGENT = os.getenv(
    "KRISHA_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)
PW_HEADLESS = os.getenv("KRISHA_PW_HEADLESS", "1") != "0"

# --- Site constants --------------------------------------------------------
BASE_URL = "https://krisha.kz"
# Monthly rent of apartments. Daily ("посуточно") is a separate section we do
# NOT crawl. robots.txt disallows the `raion=` district param, so we crawl the
# whole-city path and never filter by district via query string.
CITY_PATHS = {
    "kaskelen": "/arenda/kvartiry/kaskelen/",
    "almaty": "/arenda/kvartiry/almaty/",
}
LISTINGS_PER_PAGE = 20  # krisha default page size (window.data.search.ids)

# krisha's custom anti-bot status codes / markers observed during recon.
BLOCK_STATUS_CODES = {403, 429, 468}
CAPTCHA_MARKERS = ("/captcha", "g-recaptcha", "captcha__", "Доступ ограничен")


def default_headers(referer: str | None = None) -> dict[str, str]:
    """Browser-like headers for the httpx engine (list pages / images)."""
    h = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin" if referer else "none",
    }
    if referer:
        h["Referer"] = referer
    return h


def city_url(city: str, page: int = 1) -> str:
    path = CITY_PATHS[city]
    url = BASE_URL + path
    if page > 1:
        url += f"?page={page}"
    return url


def detail_url(listing_id: int | str) -> str:
    return f"{BASE_URL}/a/show/{listing_id}"
