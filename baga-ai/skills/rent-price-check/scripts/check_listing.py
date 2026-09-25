"""CLI навыка rent-price-check: тонкая обёртка над api.py.

Работает и с id, и со ссылкой krisha.kz. Всё, что печатается, — JSON,
чтобы вызывающая модель могла разобрать ответ без парсинга текста.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # корень проекта baga-ai


def listing_id(value: str) -> str:
    m = re.search(r"/a/show/(\d+)", value or "")
    return m.group(1) if m else str(value).strip()


def main():
    p = argparse.ArgumentParser(description="Baga AI: проверка цены аренды и поиск по ремонту")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("price", help="оценить цену объявления или произвольной квартиры")
    pr.add_argument("listing", nargs="?", help="id объявления или ссылка krisha.kz")
    pr.add_argument("--area", type=float); pr.add_argument("--rooms", type=int)
    pr.add_argument("--city"); pr.add_argument("--district"); pr.add_argument("--address", default="")
    pr.add_argument("--floor", type=float); pr.add_argument("--floors-total", type=float)
    pr.add_argument("--price", type=float); pr.add_argument("--description", default="")
    pr.add_argument("--no-explain", action="store_true")

    se = sub.add_parser("search", help="найти квартиры по описанию ремонта")
    se.add_argument("query")
    se.add_argument("--city"); se.add_argument("--district")
    se.add_argument("--rooms", type=int, nargs="*")
    se.add_argument("--price-min", type=float); se.add_argument("--price-max", type=float)
    se.add_argument("--k", type=int, default=5)

    a = p.parse_args()
    import api

    if a.cmd == "price":
        if a.listing:
            out = api.estimate_price(listing_id=listing_id(a.listing))
            card = api.get_listing(out["listing_id"])
        else:
            card = {"price": a.price, "area": a.area, "rooms": a.rooms, "city": a.city,
                    "district": a.district, "address": a.address, "floor": a.floor,
                    "floors_total": a.floors_total, "description": a.description}
            out = api.estimate_price(listing=card)
        if not a.no_explain:
            from explain import explain_price
            try:
                out["explanation"] = explain_price(card, out)
            except Exception as e:
                out["explanation"] = f"объяснение недоступно: {e}"
        out["listing"] = card
    else:
        out = api.search_listings(query=a.query, city=a.city,
                                  districts=[a.district] if a.district else None,
                                  rooms=a.rooms or None, price_min=a.price_min,
                                  price_max=a.price_max, k=a.k)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
