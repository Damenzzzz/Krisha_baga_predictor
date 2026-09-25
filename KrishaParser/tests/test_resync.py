from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from krisha.db import Database
from krisha.embed_worker import COLLECTION
from scripts.resync_qdrant_payload import payload_matches, resync, sync_listings


def test_remote_json_roundoff_does_not_trigger_endless_writes():
    assert payload_matches({"price_per_m2": 7236.842105263157},
                           {"price_per_m2": 7236.8421052631575})
    assert not payload_matches({"price_per_m2": 7236.84},
                               {"price_per_m2": 7236.85})
    assert not payload_matches({"lat": 43.2}, {"lat": 43.20001})
    assert not payload_matches({"price_kzt": 200000}, {"price_kzt": 200001})


def test_resync_full_metadata_is_idempotent(tmp_path):
    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert(COLLECTION, points=[PointStruct(id=1, vector=[1.0, 0.0], payload={
        "listing_id": 42, "photo_idx": 3, "city": "wrong", "price_kzt": 1,
        "lat": 0.0, "unrelated": "preserved",
    })])
    with Database(tmp_path / "test.db") as db:
        db.upsert_stub(42, "url", "almaty_oblast")
        db.update_detail(42, {"price_kzt": 200000, "area_total": 50, "lat": 43.2})
        db.set_duplicate_group(42, "g42")
        db.commit()
        assert resync(client, db, dry_run=True)["stale"] == 1
        assert client.retrieve(COLLECTION, [1])[0].payload["city"] == "wrong"
        assert resync(client, db)["fixed"] == 1
        point = client.retrieve(COLLECTION, [1], with_vectors=True)[0]
        assert point.payload["city"] == "almaty_oblast"
        assert point.payload["price_per_m2"] == 4000
        assert point.payload["lat"] == 43.2
        assert point.payload["duplicate_group_id"] == "g42"
        assert point.payload["photo_idx"] == 3
        assert point.payload["unrelated"] == "preserved"
        assert point.vector == [1.0, 0.0]
        assert resync(client, db)["stale"] == 0
        db.update_detail(42, {"price_kzt": 225000, "lat": 43.3})
        sync_listings(client, db, {42})
        point = client.retrieve(COLLECTION, [1], with_vectors=True)[0]
        assert point.payload["price_kzt"] == 225000
        assert point.payload["price_per_m2"] == 4500
        assert point.payload["lat"] == 43.3
        assert point.payload["listing_id"] == 42
        assert point.payload["photo_idx"] == 3
        assert point.payload["unrelated"] == "preserved"
        assert point.vector == [1.0, 0.0]
    client.close()
