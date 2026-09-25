"""A/B: температура генерации объяснений цены.

Что меряем. Для объяснения важна не «красота», а достоверность: модель не должна
выдумывать цифры и ссылаться на несуществующие объявления. Обе вещи проверяются
автоматически, без судьи:

  придуманные суммы — все числа от 1000 в тексте должны встречаться в исходных фактах
                      (цена, границы диапазона, цены аналогов);
  выдуманные id     — всё, что в квадратных скобках, должно быть среди аналогов.

Плюс длина ответа и время. Запуск: python ab_temperature.py
"""
import re
import time

import numpy as np
import pandas as pd

from explain import explain_price
from listings import load_listings
from predict import estimate_by_id

TEMPERATURES = (0.0, 0.3, 0.7)
N_LISTINGS = 12
TOLERANCE = 0.02      # «около 260 тысяч» вместо 262 439 — не выдумка, а округление


def allowed_numbers(listing: dict, est: dict) -> list[float]:
    nums = [est["p10"], est["p50"], est["p90"]]
    if listing.get("price"):
        nums.append(float(listing["price"]))
    for c in est.get("comparables", []):
        nums += [float(c["price"]), float(c["area"])]
    return nums


RENT_MIN, RENT_MAX = 10_000, 50_000_000     # правдоподобные суммы аренды в тенге
NBSP = "\u00a0"


def check(text: str, listing: dict, est: dict) -> dict:
    allowed = allowed_numbers(listing, est)
    ids_in_text = set(re.findall(r"\[(\d+)\]", text))
    known_ids = {str(c["listing_id"]) for c in est.get("comparables", [])}

    # id объявлений — девятизначные числа в скобках. Если их не вырезать, метрика
    # посчитает их «придуманными суммами» (эта ошибка уже один раз испортила замер).
    body = re.sub(r"\[[\d,\s]+\]", " ", text)
    found = [float(n.replace(" ", "").replace(NBSP, ""))
             for n in re.findall(r"\d[\d\s" + NBSP + r"]{3,}", body)]
    found = [n for n in found if RENT_MIN <= n <= RENT_MAX]
    invented = [n for n in found
                if not any(abs(n - a) <= max(TOLERANCE * a, 1000) for a in allowed)]

    return {"чисел в тексте": len(found), "придуманных сумм": len(invented),
            "_invented": invented, "выдуманных id": len(ids_in_text - known_ids),
            "длина, симв.": len(text)}


def run(n=N_LISTINGS, seed=0):
    df = load_listings()
    sample = df.sample(n, random_state=seed)
    rows = []
    for t in TEMPERATURES:
        for r in sample.itertuples():
            est = estimate_by_id(r.listing_id).model_dump()
            listing = {"price": r.price, "area": r.area, "rooms": r.rooms, "district": r.district,
                       "floor": r.floor, "floors_total": r.floors_total, "condition": r.condition}
            t0 = time.time()
            try:
                text = explain_price(listing, est, temperature=t)
            except Exception as e:
                print("ошибка:", e)
                continue
            c = check(text, listing, est)
            if c["_invented"] and len(rows) < 3:      # показать примеры для проверки самой метрики
                print(f"  пример: числа не из фактов {c['_invented']}\n  текст: {text[:300]}\n")
            rows.append({"температура": t, "секунд": round(time.time() - t0, 1),
                         **{k: v for k, v in c.items() if not k.startswith("_")}})
        print(f"температура {t} готова")

    res = pd.DataFrame(rows)
    table = res.groupby("температура").agg(
        объяснений=("длина, симв.", "size"),
        придуманных_сумм=("придуманных сумм", "mean"),
        выдуманных_id=("выдуманных id", "mean"),
        длина=("длина, симв.", "mean"),
        секунд=("секунд", "mean")).round(2)
    print("\nA/B по температуре (в среднем на одно объяснение):")
    print(table.to_string())
    print("\nвыбираем минимальную температуру с наименьшим числом придуманных чисел:")
    print(" ->", table.придуманных_сумм.idxmin())
    return table


if __name__ == "__main__":
    run()
