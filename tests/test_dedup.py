import tempfile
from pathlib import Path

from krisha.db import Database
from krisha.dedup import run_dedup


def _make_db() -> Database:
    tmp = Path(tempfile.mkdtemp()) / "t.db"
    return Database(tmp)


def test_dedup_groups_same_flat_by_structure():
    db = _make_db()
    db.upsert_stub(1, "u1", "kaskelen")
    db.upsert_stub(2, "u2", "kaskelen")
    db.upsert_stub(3, "u3", "kaskelen")
    common = dict(rooms=2, area_total=50.0, floor=3, address="Абая 10")
    db.update_detail(1, common)
    db.update_detail(2, common)  # same flat re-posted
    db.update_detail(3, dict(rooms=1, area_total=30.0, floor=1, address="Иная 5"))

    run_dedup(db)
    g1 = db.get_listing(1)["duplicate_group_id"]
    g2 = db.get_listing(2)["duplicate_group_id"]
    g3 = db.get_listing(3)["duplicate_group_id"]
    assert g1 == g2 != g3
    db.close()


def test_dedup_groups_by_shared_photo_sha():
    db = _make_db()
    db.upsert_stub(10, "u10", "almaty")
    db.upsert_stub(11, "u11", "almaty")
    db.update_detail(10, dict(rooms=2, area_total=60.0, floor=4, address="A 1"))
    db.update_detail(11, dict(rooms=3, area_total=99.0, floor=9, address="B 2"))
    db.upsert_photo(10, 0, sha256="deadbeef", local_path="/x/0.jpg")
    db.upsert_photo(11, 0, sha256="deadbeef", local_path="/y/0.jpg")

    run_dedup(db)
    assert db.get_listing(10)["duplicate_group_id"] == \
        db.get_listing(11)["duplicate_group_id"]
    db.close()
