"""CLI entrypoint: ``python -m krisha <command> [options]``."""
from __future__ import annotations

import argparse
import sys

from . import config
from .controller import (BlockedStop, CaptchaStop, Controller, LayoutChangeStop)
from .db import Database
from .logging_setup import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="krisha", description="Baga AI krisha.kz rent parser")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("crawl", help="collect listing ids from search pages")
    c.add_argument("--city", choices=list(config.CITY_PATHS), required=True)
    c.add_argument("--deal", choices=["rent"], default="rent")
    c.add_argument("--max-pages", type=int, default=None)

    d = sub.add_parser("details", help="fetch + parse detail pages (Playwright)")
    d.add_argument("--city", choices=list(config.CITY_PATHS), default=None)
    d.add_argument("--limit", type=int, default=None)

    ph = sub.add_parser("photos", help="download pending photos (httpx)")
    ph.add_argument("--limit", type=int, default=None)
    ph.add_argument("--expand", action="store_true",
                    help="discover a listing's FULL photo set via CDN index walk")

    sub.add_parser("dedup", help="assign near-duplicate group ids")

    e = sub.add_parser("embed", help="embed photos into Qdrant (Phase B)")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--batch-size", type=int, default=16)

    si = sub.add_parser("similar", help="find listings similar to one photo (Phase B)")
    si.add_argument("--image", required=True, help="path to a query photo")
    si.add_argument("--city", default=None, help="filter to a city (e.g. almaty)")
    si.add_argument("--limit", type=int, default=10)

    sub.add_parser("stats", help="show scrape events + current profile")

    ex = sub.add_parser("export", help="export listings CSV for ML")
    ex.add_argument("--csv", action="store_true")
    ex.add_argument("--out", default=None)
    ex.add_argument("--no-dedup", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    db = Database()

    needs_ctrl = args.command in ("crawl", "details")
    ctrl = Controller(db) if needs_ctrl else None
    try:
        if args.command == "crawl":
            from .commands import crawl
            crawl.run(db, ctrl, args.city, args.max_pages)
        elif args.command == "details":
            from .commands import details
            details.run(db, ctrl, args.city, args.limit)
        elif args.command == "photos":
            from .commands import photos
            photos.run(db, args.limit, expand=args.expand)
        elif args.command == "dedup":
            from .dedup import run_dedup
            run_dedup(db)
        elif args.command == "embed":
            from .commands import embed
            embed.run(db, batch_size=args.batch_size, limit=args.limit)
        elif args.command == "similar":
            from .commands import similar
            similar.run(db, args.image, city=args.city, limit=args.limit)
        elif args.command == "stats":
            from .commands import stats
            stats.run(db)
        elif args.command == "export":
            from .commands import export
            export.run(db, out_path=args.out, dedup=not args.no_dedup)
    except (CaptchaStop, BlockedStop, LayoutChangeStop) as e:
        print(f"STOP: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    finally:
        if ctrl:
            ctrl.close()
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
