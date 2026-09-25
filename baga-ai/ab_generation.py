"""A/B параметров генерации: temperature, top_p, max_tokens.

Что меряем. Для наших ответов важна не «красота», а достоверность и полнота:
  придуманные суммы — числа-суммы в тексте, которых нет в переданных фактах;
  выдуманные id     — [id] в тексте, которых нет среди переданных объявлений;
  обрезано          — finish_reason == "length": ответ упёрся в max_tokens и оборван;
  отказ фильтра     — доля ответов, которые выходной guardrail заменил бы шаблоном.
Плюс длина ответа, токены и время. Проверки те же, что в проде (guardrails.check_output):
эксперимент меряет ровно то, что потом отсекает фильтр.

Эксперименты:
  temperature  0.0 / 0.3 / 0.7                         (top_p 1.0, max_tokens 400)
  top_p        0.5 / 0.9 / 1.0 при temperature 0.7     — при temperature 0 декодирование
               жадное и top_p ни на что не влияет, поэтому сравнивать его имеет смысл
               только там, где модель действительно сэмплирует.
  max_tokens   100 / 200 / 400 / 800 при temperature 0 — на ОБЕИХ задачах: объяснение
               цены (2–4 предложения) и ответ по выдаче (3–5 предложений, длиннее).
               Лимит общий (CHAT_MAX_TOKENS), значит выбирать его надо по длинной задаче.

Запуск: python ab_generation.py [--only temperature|top_p|max_tokens] [--n 12]
Результаты: artifacts/ab_generation.json
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import ART_DIR
from explain import SEARCH_SYSTEM, _complete, _search_context, explain_price_raw, price_facts
from guardrails import check_output
from listings import load_listings
from predict import estimate, estimate_by_id

N_LISTINGS = 12
SEARCH_QUERIES = ["светлая кухня в скандинавском стиле", "свежий ремонт, новая мебель",
                  "квартира без мебели", "панорамные окна и вид на горы"]

EXPERIMENTS = {
    "temperature": [{"temperature": t, "top_p": 1.0, "max_tokens": 400} for t in (0.0, 0.3, 0.7)],
    "top_p": [{"temperature": 0.7, "top_p": p, "max_tokens": 400} for p in (0.5, 0.9, 1.0)],
    "max_tokens": [{"temperature": 0.0, "top_p": 1.0, "max_tokens": m} for m in (100, 200, 400, 800)],
}


def allowed_numbers(listing: dict, est: dict) -> list[float]:
    """Оставлено для совместимости: те же факты, что видит выходной фильтр."""
    return price_facts(listing, est)[0]


def check(text: str, listing: dict, est: dict) -> dict:
    nums, ids = price_facts(listing, est)
    r = check_output(text, nums, ids)
    return {"чисел в тексте": r["numbers"], "придуманных сумм": len(r["invented_numbers"]),
            "_invented": r["invented_numbers"], "выдуманных id": len(r["invented_ids"]),
            "длина, симв.": len(text)}


# --------------------------------------------------------------- задачи

def price_tasks(df: pd.DataFrame, n: int, seed=0) -> list[dict]:
    out = []
    for r in df.sample(n, random_state=seed).itertuples():
        est = estimate_by_id(r.listing_id).model_dump()
        listing = {"price": r.price, "area": r.area, "rooms": r.rooms, "district": r.district,
                   "floor": r.floor, "floors_total": r.floors_total, "condition": r.condition}
        out.append({"task": "объяснение цены", "listing": listing, "est": est})
    return out


def search_tasks(df: pd.DataFrame, n: int, seed=0) -> list[dict]:
    """Ответ по выдаче: 6 объявлений одного района, как их вернул бы поиск.
    Выдачу собираем из базы, а не из поиска, — меряем генерацию, а не ретривер."""
    out = []
    districts = df.district.value_counts().index[:n]
    for i, d in enumerate(districts):
        rows = df[df.district == d].sample(6, random_state=seed + i)
        results = []
        for r in rows.itertuples():
            pc = estimate(df.loc[df.listing_id == r.listing_id].iloc[0], with_comparables=False).model_dump()
            results.append({"listing_id": r.listing_id, "photos": [], "price_check": pc,
                            "listing": {"price": r.price, "rooms": r.rooms, "area": r.area,
                                        "district": r.district, "city": r.city,
                                        "description": r.description}})
        out.append({"task": "ответ по выдаче", "query": SEARCH_QUERIES[i % len(SEARCH_QUERIES)],
                    "results": results})
    return out


def run_one(task: dict, params: dict) -> dict:
    t0 = time.time()
    try:
        if task["task"] == "объяснение цены":
            out = explain_price_raw(task["listing"], task["est"], **params)
            nums, ids = price_facts(task["listing"], task["est"])
        else:
            out = _complete(SEARCH_SYSTEM, _search_context(task["query"], task["results"]), **params)
            nums = [x for r in task["results"] for x in
                    (r["listing"]["price"], r["price_check"]["p10"], r["price_check"]["p50"], r["price_check"]["p90"])]
            ids = [str(r["listing_id"]) for r in task["results"]]
    except Exception as e:
        return {"task": task["task"], **params, "error": str(e)}
    rep = check_output(out["text"], nums, ids)
    return {"task": task["task"], **params, "секунд": round(time.time() - t0, 2),
            "обрезано": out["finish_reason"] == "length",
            "придуманных сумм": len(rep["invented_numbers"]), "выдуманных id": len(rep["invented_ids"]),
            "отказ фильтра": not rep["ok"], "длина": len(out["text"]),
            "токенов": out["completion_tokens"], "_text": out["text"]}


def run(only=None, n=N_LISTINGS, workers=6):
    df = load_listings()
    tasks_price = price_tasks(df, n)
    tasks_search = search_tasks(df, max(4, n // 2))
    report = {}
    for name, grid in EXPERIMENTS.items():
        if only and name != only:
            continue
        tasks = tasks_price + (tasks_search if name == "max_tokens" else [])
        jobs = [(t, p) for p in grid for t in tasks]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(lambda j: run_one(*j), jobs))
        res = pd.DataFrame(rows)
        errors = res["error"].notna().sum() if "error" in res else 0
        res = res[res.get("error").isna()] if "error" in res else res
        key = {"temperature": "temperature", "top_p": "top_p", "max_tokens": "max_tokens"}[name]
        table = res.groupby(["task", key]).agg(
            ответов=("длина", "size"),
            обрезано_pct=("обрезано", lambda s: round(100 * s.mean(), 1)),
            придуманных_сумм=("придуманных сумм", "mean"),
            выдуманных_id=("выдуманных id", "mean"),
            отказ_фильтра_pct=("отказ фильтра", lambda s: round(100 * s.mean(), 1)),
            длина=("длина", "mean"),
            токенов=("токенов", "mean"),
            секунд=("секунд", "median")).round(2)
        print(f"\n=== {name} (ошибок вызова: {errors})\n{table.to_string()}")
        report[name] = {"table": table.reset_index().to_dict(orient="records"),
                        "examples": res.drop_duplicates(["task", key])[["task", key, "_text"]]
                                       .to_dict(orient="records")}
    ART_DIR.mkdir(exist_ok=True)
    path = ART_DIR / "ab_generation.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(report)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str))
    print(f"\nсохранено: {path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(EXPERIMENTS))
    ap.add_argument("--n", type=int, default=N_LISTINGS)
    a = ap.parse_args()
    run(a.only, a.n)
