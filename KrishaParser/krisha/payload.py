"""Mutable listing metadata shared by embedding and payload reconciliation."""
from __future__ import annotations

PAYLOAD_FIELDS = (
    "city", "district", "rooms", "area_total", "floor", "price_kzt",
    "rent_period", "lat", "lon", "duplicate_group_id",
)


def listing_payload(listing) -> dict:
    payload = {field: listing[field] for field in PAYLOAD_FIELDS}
    price, area = listing["price_kzt"], listing["area_total"]
    payload["price_per_m2"] = price / area if price and area else None
    return payload
