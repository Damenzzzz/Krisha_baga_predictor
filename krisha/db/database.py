"""SQLite access layer: listings/photos upserts, checkpoints, event log.

Upserts are idempotent so re-running never loses data and acts as the resume
checkpoint (an already-seen listing is not re-fetched).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config
from ..logging_setup import get_logger

log = get_logger("db")
_SCHEMA = Path(__file__).with_name("schema.sql")

# Columns that a detail-parse may fill (all nullable). Kept explicit so we can
# build UPSERTs dynamically and so adding a field is a one-line change here.
LISTING_FIELDS: tuple[str, ...] = (
    "url", "deal_type", "rent_period", "price_kzt", "rooms", "area_total",
    "area_living", "area_kitchen", "floor", "floors_total", "year_built",
    "building_type", "complex_name", "city", "district", "address", "lat",
    "lon", "description", "furniture", "bathroom", "balcony", "renovation_text",
    "published_at", "photo_urls", "photos_count", "raw_json",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str = config.DB_PATH):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        # Let concurrent writers (e.g. `embed` and `details` running at once) wait
        # for the lock instead of failing immediately with "database is locked".
        self.conn.execute("PRAGMA busy_timeout=30000;")
        self.init_db()

    def init_db(self) -> None:
        self.conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- listings ---------------------------------------------------------
    def upsert_stub(self, listing_id: int, url: str, city: str,
                    card: dict[str, Any] | None = None) -> bool:
        """Insert a listing id discovered on a search page. Returns True if new."""
        now = _now()
        card = card or {}
        cur = self.conn.execute("SELECT id FROM listings WHERE id=?", (listing_id,))
        exists = cur.fetchone() is not None
        if exists:
            # Backfill card-level fields when still empty (e.g. rows created
            # before card parsing was enriched).
            self.conn.execute(
                "UPDATE listings SET last_seen_at=?, url=COALESCE(url,?), "
                "rooms=COALESCE(rooms,?), area_total=COALESCE(area_total,?), "
                "floor=COALESCE(floor,?), floors_total=COALESCE(floors_total,?), "
                "price_kzt=COALESCE(price_kzt,?), address=COALESCE(address,?), "
                "description=COALESCE(description,?) WHERE id=?",
                (now, url, card.get("rooms"), card.get("area_total"),
                 card.get("floor"), card.get("floors_total"), card.get("price_kzt"),
                 card.get("address"), card.get("description"), listing_id),
            )
        else:
            self.conn.execute(
                "INSERT INTO listings (id, url, deal_type, rent_period, city, "
                "rooms, area_total, floor, floors_total, price_kzt, address, "
                "description, first_seen_at, last_seen_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (listing_id, url, "rent", "month", city,
                 card.get("rooms"), card.get("area_total"), card.get("floor"),
                 card.get("floors_total"), card.get("price_kzt"),
                 card.get("address"), card.get("description"), now, now),
            )
        self.conn.commit()
        return not exists

    def update_detail(self, listing_id: int, fields: dict[str, Any]) -> None:
        """Fill parsed detail fields on an existing listing row."""
        cols = [c for c in LISTING_FIELDS if c in fields]
        if not cols:
            return
        for c in ("photo_urls", "raw_json"):
            if c in fields and not isinstance(fields[c], (str, type(None))):
                fields[c] = json.dumps(fields[c], ensure_ascii=False)
        assignments = ", ".join(f"{c}=?" for c in cols)
        values = [fields[c] for c in cols]
        values += [_now(), listing_id]
        self.conn.execute(
            f"UPDATE listings SET {assignments}, detail_fetched_at=?, "
            f"last_seen_at=detail_fetched_at WHERE id=?",
            values,
        )
        self.conn.commit()

    def add_card_photos(self, listing_id: int, urls: list[str]) -> None:
        """Attach photo urls discovered on a search card (fills photos when the
        detail page is unavailable). Does not set detail_fetched_at."""
        if not urls:
            return
        self.conn.execute(
            "UPDATE listings SET photo_urls=?, photos_count=? WHERE id=? "
            "AND (photos_count IS NULL OR photos_count=0)",
            (json.dumps(urls, ensure_ascii=False), len(urls), listing_id),
        )
        for idx, url in enumerate(urls):
            self.upsert_photo(listing_id, idx, url=url)
        self.conn.commit()

    def set_duplicate_group(self, listing_id: int, group_id: str) -> None:
        self.conn.execute(
            "UPDATE listings SET duplicate_group_id=? WHERE id=?",
            (group_id, listing_id),
        )

    def commit(self) -> None:
        self.conn.commit()

    def ids_without_detail(self, city: str | None = None,
                           limit: int | None = None) -> list[int]:
        q = "SELECT id FROM listings WHERE detail_fetched_at IS NULL"
        params: list[Any] = []
        if city:
            q += " AND city=?"
            params.append(city)
        q += " ORDER BY id DESC"
        if limit:
            q += " LIMIT ?"
            params.append(limit)
        return [r["id"] for r in self.conn.execute(q, params)]

    def get_listing(self, listing_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM listings WHERE id=?", (listing_id,)
        ).fetchone()

    def all_listings(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM listings"))

    def count_listings(self, with_detail: bool = False,
                       with_photos: bool = False) -> int:
        q = "SELECT COUNT(*) c FROM listings WHERE 1=1"
        if with_detail:
            q += " AND detail_fetched_at IS NOT NULL"
        if with_photos:
            q += " AND photos_count > 0"
        return self.conn.execute(q).fetchone()["c"]

    # --- photos -----------------------------------------------------------
    def upsert_photo(self, listing_id: int, idx: int, **fields: Any) -> None:
        row = self.conn.execute(
            "SELECT 1 FROM photos WHERE listing_id=? AND idx=?", (listing_id, idx)
        ).fetchone()
        if row:
            cols = ", ".join(f"{k}=?" for k in fields)
            self.conn.execute(
                f"UPDATE photos SET {cols} WHERE listing_id=? AND idx=?",
                [*fields.values(), listing_id, idx],
            )
        else:
            keys = ["listing_id", "idx", *fields.keys()]
            ph = ",".join("?" * len(keys))
            self.conn.execute(
                f"INSERT INTO photos ({','.join(keys)}) VALUES ({ph})",
                [listing_id, idx, *fields.values()],
            )
        self.conn.commit()

    def photos_for(self, listing_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM photos WHERE listing_id=? ORDER BY idx", (listing_id,)
        ))

    def sha_exists(self, sha256: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT listing_id, idx, local_path FROM photos WHERE sha256=? "
            "AND local_path IS NOT NULL LIMIT 1", (sha256,)
        ).fetchone()

    def photos_pending_download(self, limit: int | None = None) -> list[sqlite3.Row]:
        q = ("SELECT * FROM photos WHERE (local_path IS NULL OR downloaded_at IS NULL) "
             "AND url IS NOT NULL ORDER BY listing_id, idx")
        if limit:
            q += f" LIMIT {int(limit)}"
        return list(self.conn.execute(q))

    def photos_pending_embed(self, limit: int | None = None) -> list[sqlite3.Row]:
        q = ("SELECT * FROM photos WHERE embedded_at IS NULL AND local_path IS NOT NULL "
             "ORDER BY listing_id, idx")
        if limit:
            q += f" LIMIT {int(limit)}"
        return list(self.conn.execute(q))

    # --- events -----------------------------------------------------------
    def log_event(self, run_id: str, url: str, engine: str, status_code: int | None,
                  latency_ms: int, delay_ms: int, concurrency: int,
                  outcome: str) -> None:
        self.conn.execute(
            "INSERT INTO scrape_events (ts, run_id, url, engine, status_code, "
            "latency_ms, delay_ms, concurrency, outcome) VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), run_id, url, engine, status_code, latency_ms, delay_ms,
             concurrency, outcome),
        )
        self.conn.commit()

    def event_summary(self, run_id: str | None = None) -> dict[str, int]:
        q = "SELECT outcome, COUNT(*) c FROM scrape_events"
        params: Sequence[Any] = ()
        if run_id:
            q += " WHERE run_id=?"
            params = (run_id,)
        q += " GROUP BY outcome"
        return {r["outcome"]: r["c"] for r in self.conn.execute(q, params)}

    def last_run_id(self) -> str | None:
        r = self.conn.execute(
            "SELECT run_id FROM scrape_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return r["run_id"] if r else None

    def bans(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT ts, url, status_code, outcome FROM scrape_events "
            "WHERE outcome IN ('http_429','http_403','captcha') ORDER BY id DESC"
        ))
