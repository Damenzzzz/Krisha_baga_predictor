"""Listing deduplication.

Exact duplicates are already prevented by the ``id`` primary key. This module
groups *near*-duplicates (the same physical flat re-posted) so one apartment
never lands in both the train and test split.

A near-duplicate group is formed by union-find over two signals:
  1. same structural key: (rooms, round(area_total), floor, address|complex);
  2. any shared photo sha256.
Each connected component gets a stable ``duplicate_group_id`` = "g{min_id}".
"""
from __future__ import annotations

from .db import Database
from .logging_setup import get_logger

log = get_logger("dedup")


class _UF:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _struct_key(row) -> tuple | None:
    rooms = row["rooms"]
    area = row["area_total"]
    floor = row["floor"]
    loc = row["address"] or row["complex_name"]
    if rooms is None or area is None or not loc:
        return None
    return (rooms, round(float(area)), floor, loc.strip().lower())


def run_dedup(db: Database) -> int:
    listings = db.all_listings()
    ids = [r["id"] for r in listings]
    if not ids:
        return 0
    uf = _UF(ids)

    # 1) structural key
    by_key: dict[tuple, int] = {}
    for r in listings:
        key = _struct_key(r)
        if key is None:
            continue
        if key in by_key:
            uf.union(by_key[key], r["id"])
        else:
            by_key[key] = r["id"]

    # 2) shared photo sha256
    by_sha: dict[str, int] = {}
    for r in listings:
        for ph in db.photos_for(r["id"]):
            sha = ph["sha256"]
            if not sha:
                continue
            if sha in by_sha:
                uf.union(by_sha[sha], r["id"])
            else:
                by_sha[sha] = r["id"]

    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(uf.find(i), []).append(i)

    n_multi = 0
    for root, members in groups.items():
        gid = f"g{root}"
        if len(members) > 1:
            n_multi += 1
        for m in members:
            db.set_duplicate_group(m, gid)
    db.commit()
    log.info("dedup: %d listings -> %d groups (%d with >1 member)",
             len(ids), len(groups), n_multi)
    return n_multi
