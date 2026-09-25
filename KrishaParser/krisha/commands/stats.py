"""`stats`: event summary (last run + all-time), profile, ban history."""
from __future__ import annotations

from ..controller import RuntimeProfile
from ..db import Database


def run(db: Database) -> None:
    last = db.last_run_id()
    print("=== Krisha parser stats ===")
    print(f"listings total        : {db.count_listings()}")
    print(f"listings with detail  : {db.count_listings(with_detail=True)}")
    print(f"listings with photos  : {db.count_listings(with_photos=True)}")

    prof = RuntimeProfile.load()
    print(f"\ncurrent profile       : engine={prof.engine} "
          f"delay_ms={prof.delay_ms} concurrency={prof.concurrency}")

    print(f"\nlast run ({last}):")
    for outcome, n in sorted(db.event_summary(last).items()):
        print(f"  {outcome:14s}: {n}")

    print("\nall-time events:")
    for outcome, n in sorted(db.event_summary().items()):
        print(f"  {outcome:14s}: {n}")

    bans = db.bans()
    print(f"\nbans/blocks: {len(bans)}")
    for b in bans[:10]:
        print(f"  {b['ts']}  {b['outcome']} (status={b['status_code']})  {b['url']}")
