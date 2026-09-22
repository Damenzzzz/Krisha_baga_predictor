"""Phase B smoke test: find similar listings by a single photo, filtered by city.

Encodes one local photo with the same open_clip model used by `embed`, then runs
two Qdrant searches — unfiltered and filtered to the query photo's city — to prove
the vector index + the `city` KEYWORD payload filter work end to end.

Usage (Qdrant up, photos embedded):
    python -m scripts.smoke_similar [--image PATH] [--city CITY] [--topk N]
"""
from __future__ import annotations

import argparse
import os
import sys

from krisha.db import Database
from krisha.embed_worker import COLLECTION, _load_model


def _pick_query(db: Database, city: str | None):
    """Return (local_path, listing_id, city) for a downloaded+embedded photo."""
    where = "p.local_path IS NOT NULL AND p.embedded_at IS NOT NULL AND l.city IS NOT NULL"
    params: list = []
    if city:
        where += " AND l.city=?"
        params.append(city)
    row = db.conn.execute(
        f"SELECT p.local_path, p.listing_id, l.city FROM photos p "
        f"JOIN listings l ON l.id=p.listing_id WHERE {where} "
        f"ORDER BY p.listing_id LIMIT 1", params).fetchone()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="query photo (default: pick one from DB)")
    ap.add_argument("--city", default=None, help="restrict query pick + filter to this city")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    from PIL import Image  # type: ignore
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

    db = Database()
    if args.image:
        image_path, q_listing, q_city = args.image, None, args.city
    else:
        row = _pick_query(db, args.city)
        if not row:
            print("no embedded photo found to use as a query")
            return 1
        image_path, q_listing, q_city = row["local_path"], row["listing_id"], row["city"]
    print(f"query image: {image_path}\n  listing_id={q_listing} city={q_city!r}\n")

    model, preprocess, device, dim, torch = _load_model()
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        vec = model.encode_image(preprocess(img).unsqueeze(0).to(device))
        vec = (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy()[0].tolist()

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)

    def show(title, flt):
        hits = client.query_points(COLLECTION, query=vec, limit=args.topk,
                                    query_filter=flt, with_payload=True).points
        print(title)
        for h in hits:
            p = h.payload or {}
            print(f"  score={h.score:.3f} listing={p.get('listing_id')} "
                  f"city={p.get('city')!r} rooms={p.get('rooms')} "
                  f"price={p.get('price_kzt')} idx={p.get('photo_idx')}")
        cities = {(h.payload or {}).get("city") for h in hits}
        print(f"  -> distinct cities in results: {cities}\n")
        return hits

    show(f"[1] UNFILTERED top-{args.topk}:", None)
    if q_city:
        city_filter = Filter(must=[FieldCondition(key="city",
                                                  match=MatchValue(value=q_city))])
        hits = show(f"[2] FILTERED city={q_city!r} top-{args.topk}:", city_filter)
        bad = [h for h in hits if (h.payload or {}).get("city") != q_city]
        if bad:
            print(f"FAIL: {len(bad)} results leaked outside city={q_city!r}")
            return 1
        print("PASS: city filter is airtight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
