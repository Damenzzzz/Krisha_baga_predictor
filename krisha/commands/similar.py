"""`similar`: Phase B smoke query — find listings similar to one photo.

Filters by city (optional) and prints the top matches with price/rooms/url so a
human can eyeball whether the visual similarity search works end to end."""
from __future__ import annotations

from ..db import Database


def run(db: Database, image_path: str, city: str | None = None,
        limit: int = 10) -> None:
    try:
        from ..embed_worker import search_by_image
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Embedding deps missing. Install Phase B extras: "
            "pip install open-clip-torch torch pillow qdrant-client\n"
            f"({e})"
        )

    results = search_by_image(db, image_path, city=city, limit=limit)
    if not results:
        print(f"no matches (city={city!r}) — is the collection populated?")
        return

    print(f"top {len(results)} similar to {image_path}"
          + (f" in {city}" if city else "") + ":")
    for i, r in enumerate(results, 1):
        p = r["payload"]
        price = p.get("price_kzt")
        price_s = f"{price:,} kzt".replace(",", " ") if price else "—"
        print(f"{i:2d}. score={r['score']:.3f}  "
              f"listing={p.get('listing_id')} idx={p.get('photo_idx')}  "
              f"{p.get('city') or '—'}/{p.get('district') or '—'}  "
              f"rooms={p.get('rooms') or '—'} area={p.get('area_total') or '—'} "
              f"price={price_s}")
        if r["url"]:
            print(f"      {r['url']}")
