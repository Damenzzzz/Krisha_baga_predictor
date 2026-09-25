"""Check a bounded sample from ProxyScrape's official public HTTPS proxy list.

Checks only an HTTPS connectivity endpoint with normal certificate validation.
Passing connectivity does NOT establish that Krisha will accept the proxy.
No application credentials or local data are sent through these routes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import json
from pathlib import Path
import time

import httpx

SOURCE = "https://cdn.jsdelivr.net/gh/proxyscrape/free-proxy-list@main/proxies/protocols/https/data.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/public-proxies.json"))
    args = ap.parse_args()
    if not 1 <= args.limit <= 50:
        ap.error("--limit must be between 1 and 50")
    response = httpx.get(SOURCE, timeout=30)
    response.raise_for_status()
    rows = sorted(response.json(), key=lambda p: (-(p.get("uptime_percent") or 0),
                                                  p.get("latency_ms") or 99999))
    candidates = []
    seen = set()
    for row in rows:
        try:
            ip = ipaddress.ip_address(row["ip"])
            port = int(row["port"])
            if not ip.is_global or ip.is_multicast or ip.version != 4 or not 1 <= port <= 65535:
                continue
            server = f"http://{ip}:{port}"
            if row.get("protocol") != "http" or server in seen:
                continue
            seen.add(server)
            candidates.append((server, row.get("country_code")))
        except (KeyError, ValueError, TypeError):
            continue
        if len(candidates) >= args.limit:
            break

    def check(candidate):
        server, country = candidate
        started = time.monotonic()
        try:
            with httpx.Client(proxy=server, timeout=8, trust_env=False) as client:
                result = client.get("https://www.cloudflare.com/cdn-cgi/trace")
                ok = result.status_code == 200 and "ip=" in result.text
        except httpx.HTTPError:
            ok = False
        print(f"{country}: {'HTTPS reachable' if ok else 'unavailable'}", flush=True)
        return (time.monotonic() - started, {"server": server}) if ok else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        timings = [result for result in pool.map(check, candidates) if result]
    working = [proxy for _, proxy in sorted(timings, key=lambda item: item[0])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(working, indent=2), encoding="utf-8")
    print(f"{len(working)}/{len(candidates)} passed HTTPS connectivity; saved {args.out}")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
