"""Unattended bounded-proxy backfill with SQLite, embeddings, Qdrant and CSV checkpoints.

Each route is tried with the controller's normal retry/cooldown budget. Exhausted
routes are not recycled. CAPTCHA/layout errors stop the worker. A stop-file is
checked between batches. State and logs survive the interactive agent session.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

from krisha import config
from krisha.commands import details, export, photos
from krisha.controller import BlockedStop, CaptchaStop, Controller, LayoutChangeStop, NetworkStop
from krisha.db import Database
from krisha.dedup import run_dedup
from krisha import embed_worker
from krisha.fetcher.proxy import load_proxies
from krisha.logging_setup import setup_logging
from scripts.resync_qdrant_payload import resync, sync_listings

STATE = config.DATA_DIR / "auto_backfill_status.json"
STOP = config.DATA_DIR / "STOP_AUTO_BACKFILL"


def write_state(db, phase, **extra):
    counts = dict(db.conn.execute(
        "SELECT COUNT(*) total, COUNT(detail_fetched_at) with_detail, "
        "SUM(detail_fetched_at IS NULL) pending_detail, COUNT(year_built) year_built, "
        "COUNT(building_type) building_type, COUNT(bathroom) bathroom, COUNT(balcony) balcony "
        "FROM listings").fetchone())
    state = {"pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(),
             "phase": phase, **counts,
             "pending_refresh": len(db.ids_without_detail(refresh_missing=True)), **extra}
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)
    print(json.dumps(state), flush=True)


def finalize_batch(db, client, changed: set[int], full=False):
    old_groups = {r["id"]: r["duplicate_group_id"] for r in db.all_listings()}
    write_state(db, "photos")
    if db.photos_pending_download():
        photos.run(db)
    write_state(db, "embedding")
    pending = db.photos_pending_embed()
    changed.update(r["listing_id"] for r in pending)
    if pending:
        embed_worker.run(db, batch_size=16)
    write_state(db, "dedup")
    run_dedup(db)
    changed.update(r["id"] for r in db.all_listings()
                   if old_groups.get(r["id"]) != r["duplicate_group_id"])
    write_state(db, "qdrant_sync")
    if full:
        resync(client, db)
    else:
        sync_listings(client, db, changed)
    temporary = config.EXPORTS_DIR / "listings_latest.tmp.csv"
    export.run(db, out_path=str(temporary))
    temporary.replace(config.EXPORTS_DIR / "listings_latest.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--fallback-proxies", default=None)
    args = ap.parse_args()
    if not 1 <= args.batch_size <= 100:
        ap.error("--batch-size must be 1..100")
    setup_logging()
    # OS releases this exclusive lock even if the worker exits unexpectedly.
    lock = (config.DATA_DIR / "auto_backfill.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another auto_backfill worker already holds the lock", flush=True)
        lock.close()
        return 1

    db = Database()
    client = None
    try:
        pool = load_proxies()
        if args.fallback_proxies:
            pool += load_proxies(args.fallback_proxies)
        if pool:
            pool_path = config.DATA_DIR / "auto_backfill_proxies.json"
            pool_path.write_text(json.dumps(pool), encoding="utf-8")
            os.environ["KRISHA_PROXIES_FILE"] = str(pool_path)
        route = 0
        unavailable: set[int] = set()
        client = embed_worker.make_client()
        from qdrant_client.models import PayloadSchemaType
        collection = client.get_collection(embed_worker.COLLECTION)
        if "listing_id" not in (collection.payload_schema or {}):
            client.create_payload_index(embed_worker.COLLECTION, "listing_id",
                                        PayloadSchemaType.INTEGER, wait=True)
        # Publish progress already made interactively before further scraping.
        finalize_batch(db, client, set(), full=True)
        terminal = "complete"
        while True:
            pending = [lid for lid in db.ids_without_detail(refresh_missing=True)
                       if lid not in unavailable]
            if not pending:
                terminal = "partial_unavailable" if unavailable else "complete"
                break
            if STOP.exists():
                terminal = "stopped_by_file"
                break
            if route >= max(1, len(pool)):
                terminal = "routes_exhausted"
                break
            os.environ["KRISHA_PROXY_START_INDEX"] = str(route)
            selected = pending[:args.batch_size]
            write_state(db, "fetching", proxy_index=route, proxy_count=len(pool),
                        batch_size=len(selected))
            ctrl = Controller(db)
            failure = None
            try:
                details.run(db, ctrl, refresh_missing=True, listing_ids=selected)
            except (BlockedStop, NetworkStop, CaptchaStop, LayoutChangeStop, RuntimeError) as error:
                failure = type(error).__name__
                print(f"Batch stopped: {failure}", flush=True)
            finally:
                engine = ctrl._engines.get("playwright")
                route = getattr(engine, "_proxy_index", route)
                ctrl.close()
            remaining = set(db.ids_without_detail(refresh_missing=True))
            changed = set(selected) - remaining
            if changed:
                finalize_batch(db, client, changed)
            if failure in {"CaptchaStop", "LayoutChangeStop"}:
                terminal = failure
                break
            if failure:
                route += 1
                # No immediate retry after the last attempt of an exhausted route.
                write_state(db, "route_cooldown", next_proxy_index=route, failure=failure)
                time.sleep(config.COOLDOWN_S)
            else:
                unavailable.update(set(selected) & remaining)
        write_state(db, "final_audit", terminal=terminal)
        audit = resync(client, db, dry_run=True)
        if audit["stale"]:
            resync(client, db)
        smoke = subprocess.run([sys.executable, "-m", "scripts.smoke_similar",
                                "--city", "almaty_oblast"], check=False).returncode
        write_state(db, terminal, unavailable_ids=sorted(unavailable), smoke_exit_code=smoke)
        return 0 if terminal == "complete" and smoke == 0 else 2
    except Exception as error:
        write_state(db, "error", error_type=type(error).__name__)
        raise
    finally:
        if client:
            client.close()
        db.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
