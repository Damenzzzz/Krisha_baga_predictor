import json

from krisha.commands.details import save_detail
from krisha.db import Database
from krisha.parsers.listing_detail import DetailResult, PARSER_VERSION


def test_refresh_checkpoint_and_gallery_preservation(tmp_path):
    with Database(tmp_path / "test.db") as db:
        db.upsert_stub(1, "https://krisha.kz/a/show/1", "almaty_oblast")
        db.upsert_photo(1, 0, url="https://cdn/a.jpg", local_path="photo.jpg",
                        sha256="abc", embedded_at="done")
        db.upsert_photo(1, 1, url="https://cdn/b.jpg", local_path="b.jpg")
        db.update_detail(1, {"rooms": 2})
        assert db.ids_without_detail() == []
        assert db.ids_without_detail(refresh_missing=True) == [1]
        detail = DetailResult(fields={"city": "Talgar", "bathroom": "раздельный"},
                              photos=[{"idx": 0, "url": "https://cdn/b.jpg"},
                                      {"idx": 1, "url": "https://cdn/c.jpg"}], parsed=True)
        save_detail(db, 1, detail)
        listing = db.get_listing(1)
        assert listing["city"] == "almaty_oblast"
        assert listing["detail_parser_version"] == PARSER_VERSION
        assert listing["last_seen_at"] == listing["detail_fetched_at"]
        assert db.ids_without_detail(refresh_missing=True) == []
        assert listing["year_built"] is None  # absent at source, not endlessly retried
        assert json.loads(listing["photo_urls"]) == ["https://cdn/a.jpg", "https://cdn/b.jpg", "https://cdn/c.jpg"]
        photos = db.photos_for(1)
        assert photos[0]["sha256"] == "abc"
        assert photos[0]["embedded_at"] == "done"
        assert photos[1]["local_path"] == "b.jpg"
        assert photos[2]["embedded_at"] is None
        save_detail(db, 1, detail)
        assert len(db.photos_for(1)) == 3


def test_pending_filters(tmp_path):
    with Database(tmp_path / "test.db") as db:
        db.upsert_stub(1, "url", "almaty")
        db.upsert_stub(2, "url", "kaskelen")
        assert db.ids_without_detail(city="almaty", refresh_missing=True) == [1]
        assert db.ids_without_detail(limit=0) == []


def test_legacy_schema_migration(tmp_path):
    import sqlite3
    from pathlib import Path
    path = tmp_path / "old.db"
    schema = Path("krisha/db/schema.sql").read_text(encoding="utf-8")
    with sqlite3.connect(path) as conn:
        conn.executescript(schema.replace("    detail_parser_version INTEGER,\n", ""))
        conn.execute("INSERT INTO listings (id, city) VALUES (1, 'almaty')")
    with Database(path) as db:
        assert db.get_listing(1)["detail_parser_version"] is None
        assert db.get_listing(1)["city"] == "almaty"
