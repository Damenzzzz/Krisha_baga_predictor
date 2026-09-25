"""Capture a live detail page and its characteristics for parser diagnosis."""
from __future__ import annotations

import argparse
import json

from krisha import config
from krisha.controller import BlockedStop, CaptchaStop, Controller, LayoutChangeStop, NetworkStop
from krisha.db import Database
from krisha.fetcher import cache
from krisha.logging_setup import setup_logging
from krisha.parsers import parse_detail
from krisha.parsers.listing_detail import extract_characteristics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("listing_id", type=int)
    args = ap.parse_args()
    setup_logging()
    with Database() as db:
        ctrl = Controller(db)
        try:
            result, delay = ctrl.fetch(config.detail_url(args.listing_id), "playwright")
            detail = parse_detail(result.text, url=result.url)
            ctrl.record(result, "ok" if result.ok and detail.parsed else "empty_parse", delay)
            if not result.ok or not detail.parsed:
                print(f"No detail page: HTTP {result.status_code}")
                return 1
            out = config.DATA_DIR / "detail_diagnostics"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{args.listing_id}.html").write_text(result.text, encoding="utf-8")
            cache.put(result.url, result.text)
            rows = extract_characteristics(result.text)
            report = {"url": result.url, "status": result.status_code,
                      "characteristics": rows, "parsed_fields": detail.fields}
            (out / f"{args.listing_id}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(rows, ensure_ascii=True, indent=2))
            print(f"Saved HTML and JSON under {out}")
            return 0
        except (BlockedStop, CaptchaStop, LayoutChangeStop, NetworkStop) as error:
            print(f"STOP: {type(error).__name__}: {error}")
            return 2
        finally:
            ctrl.close()


if __name__ == "__main__":
    raise SystemExit(main())
