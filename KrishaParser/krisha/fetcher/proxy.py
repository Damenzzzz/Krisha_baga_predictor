"""Explicit, private Playwright proxy configuration (never logged)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .. import config


def load_proxies(filename: str | None = None) -> list[dict[str, str]]:
    filename = filename or os.getenv("KRISHA_PROXIES_FILE")
    if not filename:
        return []
    path = Path(filename)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    try:
        proxies = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(proxies, list) or not proxies:
            raise ValueError
        for proxy in proxies:
            if not isinstance(proxy, dict) or set(proxy) - {"server", "username", "password"}:
                raise ValueError
            if not all(isinstance(v, str) for v in proxy.values()):
                raise ValueError
            url = urlsplit(proxy.get("server", ""))
            if url.scheme not in {"http", "https", "socks5"} or not url.hostname or not url.port:
                raise ValueError
            if url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
                raise ValueError
            if url.scheme == "socks5" and (proxy.get("username") or proxy.get("password")):
                raise ValueError
        return proxies
    except (OSError, ValueError, TypeError):
        raise ValueError("Invalid KRISHA_PROXIES_FILE: expected a nonempty JSON list of "
                         "{server, username?, password?}; credentials must be separate; "
                         "SOCKS5 authentication is unsupported") from None
