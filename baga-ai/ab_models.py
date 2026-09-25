"""A/B: ALEM alemllm против Gemini 3.6 Flash (Vertex AI) на всех задачах проекта.

Гипотеза: Gemini точнее разбирает запрос и реже выдумывает цифры в объяснениях, но
стоит денег и может быть медленнее. Решение принимается по задачам, а не «в целом»:
модель, которая лучше разбирает запрос, не обязательно лучше пишет объяснения.

Задачи и метрики (одинаковый промпт, temperature 0, без кэша и без guardrails —
меряем саму модель):

1. parse — разбор запроса на фильтры. Golden set PARSE_GOLDEN: 36 размеченных запросов,
   из них 8 ловушек, где правильный ответ — «ничего не ставить» («недорогая квартира»,
   «в центре», «для семьи с ребёнком»).
     field_acc     — доля верных полей (10 полей × 36 запросов);
     exact         — доля запросов, где верны ВСЕ поля;
     hallucinated  — доля запросов, где модель поставила условие, которого не было.
                     Самая дорогая ошибка: фильтр молча режет выдачу;
     json_ok       — ответ разобрался как JSON.
2. explain — объяснение цены по аналогам, 20 объявлений:
     придуманные суммы / id (проверка guardrails.check_output), обрезано по max_tokens,
     покрытие — упомянут диапазон и хотя бы один аналог.
3. answer — ответ по выдаче, 6 выдач по 6 объявлений, те же метрики.
4. judge — модель как судья релевантности: 12 запросов × (5 описаний, где признак есть,
   + 5, где его нет, по регулярке). Согласие с разметкой и каппа Коэна.

Для Gemini отдельно перебирается thinking_level (minimal / low / medium): рассуждения
улучшают разбор, но добавляют латентность и оплачиваемые токены.

Запуск:  python ab_models.py [--only parse|explain|answer|judge] [--quick]
Результат: artifacts/ab_models.json, таблицы — в EVALS.md, раздел 6.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config import ART_DIR

CONFIGS = {
    "alem": {"providers": ["alem"], "thinking": None},
    "gemini·minimal": {"providers": ["gemini"], "thinking": "minimal"},
    "gemini·low": {"providers": ["gemini"], "thinking": "low"},
    "gemini·medium": {"providers": ["gemini"], "thinking": "medium"},
}
FIELDS = ["city", "districts", "rooms", "price_min", "price_max", "area_min", "area_max",
          "not_first_floor", "not_last_floor", "photo_rooms"]

# Ожидаемые поля; не указанные — null/False. alt — допустимые варианты там, где запрос
# честно неоднозначен («студия» — 1 комната или не указано).
PARSE_GOLDEN = [
    {"q": "двушка в Алматы до 400 тысяч, светлая кухня, не первый этаж",
     "e": {"city": "алматы", "rooms": [2], "price_max": 400000, "not_first_floor": True, "photo_rooms": ["kitchen"]}},
    {"q": "однушка в Бостандыке до 250к", "e": {"districts": ["бостандык"], "rooms": [1], "price_max": 250000},
     "alt": {"city": ["алматы"]}},
    {"q": "3-комнатная квартира от 300 до 500 тысяч в Медеуском районе",
     "e": {"rooms": [3], "price_min": 300000, "price_max": 500000, "districts": ["медеу"]}, "alt": {"city": ["алматы"]}},
    {"q": "студия в Каскелене", "e": {"city": "каскелен"}, "alt": {"rooms": [[1]]}},
    {"q": "квартира с панорамными окнами и видом на горы", "e": {}},
    {"q": "лофт с кирпичной стеной", "e": {}},
    {"q": "не первый и не последний этаж, трешка в Алматы",
     "e": {"city": "алматы", "rooms": [3], "not_first_floor": True, "not_last_floor": True}},
    {"q": "до 200 тысяч", "e": {"price_max": 200000}},
    {"q": "от 150к, светлый ремонт", "e": {"price_min": 150000}},
    {"q": "площадь от 60 квадратов, 2 комнаты", "e": {"area_min": 60, "rooms": [2]}},
    {"q": "квартира 40-50 м2 в Ауэзовском районе", "e": {"area_min": 40, "area_max": 50, "districts": ["ауэзов"]},
     "alt": {"city": ["алматы"]}},
    {"q": "1 или 2 комнаты в Алмалинском до 300000 тенге",
     "e": {"rooms": [1, 2], "districts": ["алмалин"], "price_max": 300000}, "alt": {"city": ["алматы"]}},
    {"q": "уютная спальня в бежевых тонах", "e": {"photo_rooms": ["bedroom"]}},
    {"q": "ванная с тёмной плиткой, двушка", "e": {"rooms": [2], "photo_rooms": ["bathroom"]}},
    {"q": "квартира в Астане до 300 тысяч", "e": {"price_max": 300000}},
    {"q": "купить квартиру в Алматы", "e": {"city": "алматы"}},
    {"q": "2 bedroom apartment in Almaty under 400k", "e": {"city": "алматы", "price_max": 400000},
     "alt": {"rooms": [[2], [3]], "photo_rooms": [["bedroom"]]}},
    {"q": "двушка за 350 тыщ в Турксибе", "e": {"rooms": [2], "price_max": 350000, "districts": ["турксиб"]},
     "alt": {"city": ["алматы"]}},
    {"q": "трёшка, 80+ метров, Бостандыкский район, кухня-гостиная",
     "e": {"rooms": [3], "area_min": 80, "districts": ["бостандык"], "photo_rooms": ["kitchen", "living_room"]},
     "alt": {"city": ["алматы"], "photo_rooms": [["kitchen"], ["living_room"]]}},
    {"q": "квартира без мебели в Наурызбайском районе", "e": {"districts": ["наурызбай"]}, "alt": {"city": ["алматы"]}},
    {"q": "однокомнатная до полумиллиона", "e": {"rooms": [1], "price_max": 500000}},
    {"q": "до 1.2 млн, 4 комнаты", "e": {"rooms": [4], "price_max": 1200000}},
    {"q": "Алматы, Жетысуский район, 2-3 комнаты, от 200 до 350 тыс",
     "e": {"city": "алматы", "districts": ["жетысу"], "rooms": [2, 3], "price_min": 200000, "price_max": 350000}},
    {"q": "светлая квартира", "e": {}},
    {"q": "квартира с балконом до 280к", "e": {"price_max": 280000, "photo_rooms": ["balcony"]}},
    # ловушки: правильный ответ — ничего не додумывать
    {"q": "нужна квартира для семьи с ребёнком", "e": {}, "trap": True},
    {"q": "недорогая квартира", "e": {}, "trap": True},
    {"q": "квартира в центре", "e": {}, "trap": True},
    {"q": "большая квартира", "e": {}, "trap": True},
    {"q": "квартира как у бабушки, советский ремонт", "e": {}, "trap": True},
    {"q": "хочу жить у гор", "e": {}, "trap": True},
    {"q": "квартира для студента", "e": {}, "trap": True},
    {"q": "элитная квартира с дизайнерским ремонтом", "e": {}, "trap": True},
    {"q": "двушка не на первом этаже, Алатауский район",
     "e": {"rooms": [2], "not_first_floor": True, "districts": ["алатау"]}, "alt": {"city": ["алматы"]}},
    {"q": "Каскелен, однушка, от 100 до 150 тысяч",
     "e": {"city": "каскелен", "rooms": [1], "price_min": 100000, "price_max": 150000}},
    {"q": "двушку или трешку, санузел раздельный, до 600 000",
     "e": {"rooms": [2, 3], "price_max": 600000, "photo_rooms": ["bathroom"]}},
]


def _norm_field(field, v):
    if v in (None, [], "", False):
        return None
    if field == "districts":
        return tuple(sorted({re.sub(r"[^а-я]", "", str(x).lower().replace("ё", "е"))[:5] for x in v}))
    if field in ("rooms",):
        return tuple(sorted(int(x) for x in v))
    if field == "photo_rooms":
        return tuple(sorted(v))
    if field == "city":
        return str(v).lower().replace("ё", "е")
    if field in ("not_first_floor", "not_last_floor"):
        return bool(v)
    return float(v)


def score_parse(item, got: dict | None) -> dict:
    exp, alt = item["e"], item.get("alt", {})
    if got is None:
        return {"json_ok": False, "correct": 0, "exact": False, "hallucinated": True, "wrong": FIELDS}
    ok, wrong, halluc = 0, [], False
    for f in FIELDS:
        g = _norm_field(f, got.get(f))
        allowed = {_norm_field(f, exp.get(f))} | {_norm_field(f, a) for a in alt.get(f, [])}
        if g in allowed:
            ok += 1
            continue
        wrong.append(f)
        if g is not None and _norm_field(f, exp.get(f)) is None:
            halluc = True
    return {"json_ok": True, "correct": ok, "exact": not wrong, "hallucinated": halluc, "wrong": wrong}


# --------------------------------------------------------------- задачи

def run_parse(cfg_names, workers=6):
    from query_parser import llm_parse

    def one(job):
        name, item = job
        c = CONFIGS[name]
        t0 = time.time()
        try:
            d, r = llm_parse(item["q"], providers=c["providers"], thinking=c["thinking"])
            meta = {"cost": r.cost_usd, "latency": r.latency_s, "provider": r.provider,
                    "out_tokens": r.completion_tokens + r.thinking_tokens}
        except Exception as e:
            d, meta = None, {"cost": 0.0, "latency": time.time() - t0, "error": str(e)[:200]}
        return {"config": name, "q": item["q"], "trap": item.get("trap", False), **score_parse(item, d),
                **meta, "_got": d}

    jobs = [(n, it) for n in cfg_names for it in PARSE_GOLDEN]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    table = df.groupby("config", sort=False).agg(
        запросов=("q", "size"),
        field_acc=("correct", lambda s: round(100 * s.sum() / (len(s) * len(FIELDS)), 1)),
        exact=("exact", lambda s: round(100 * s.mean(), 1)),
        hallucinated=("hallucinated", lambda s: round(100 * s.mean(), 1)),
        json_ok=("json_ok", lambda s: round(100 * s.mean(), 1)),
        p50_s=("latency", lambda s: round(s.median(), 2)),
        p95_s=("latency", lambda s: round(s.quantile(0.95), 2)),
        usd_per_1k=("cost", lambda s: round(1000 * s.mean(), 3)))
    traps = df[df.trap].groupby("config", sort=False).hallucinated.mean().mul(100).round(1)
    table["ловушки_провалено"] = traps
    errors = df[~df.exact][["config", "q", "wrong", "_got"]].to_dict(orient="records")
    return table, errors


def _gen_rows(tasks, cfg_names, workers=6):
    from explain import SEARCH_SYSTEM, _complete, _search_context, explain_price_raw, price_facts
    from guardrails import check_output

    def one(job):
        name, t = job
        c = CONFIGS[name]
        kw = {"providers": c["providers"], "thinking": c["thinking"], "use_cache": False}
        try:
            if t["task"] == "explain":
                out = explain_price_raw(t["listing"], t["est"], **kw)
                nums, ids = price_facts(t["listing"], t["est"])
                rng = [t["est"]["p10"], t["est"]["p90"], t["est"]["p50"]]
            else:
                out = _complete(SEARCH_SYSTEM, _search_context(t["query"], t["results"]), task="answer_search", **kw)
                nums = [x for r in t["results"] for x in (r["listing"]["price"], r["price_check"]["p10"],
                                                            r["price_check"]["p50"], r["price_check"]["p90"])]
                ids, rng = [str(r["listing_id"]) for r in t["results"]], []
        except Exception as e:
            return {"config": name, "task": t["task"], "error": str(e)[:200]}
        rep = check_output(out["text"], nums, ids)
        cited = set(re.findall(r"\[(\d{6,})\]", out["text"])) & set(ids)
        mentions_range = (not rng) or any(_money_in(out["text"], v) for v in rng)
        return {"config": name, "task": t["task"], "обрезано": out["finish_reason"] == "length",
                "придуманных_сумм": len(rep["invented_numbers"]), "выдуманных_id": len(rep["invented_ids"]),
                "отказ_фильтра": not rep["ok"], "покрытие": bool(cited) and mentions_range,
                "длина": len(out["text"]), "latency": out["latency_s"], "cost": out["cost_usd"],
                "_text": out["text"]}

    jobs = [(n, t) for n in cfg_names for t in tasks]
    with ThreadPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(one, jobs)))


def _money_in(text: str, v) -> bool:
    """Сумма упомянута в любом из форматов: 380 000, 380000, 380 тыс."""
    if v is None:
        return False
    v = int(round(v))
    t = text.replace(" ", " ")
    k = v // 1000
    return any(s in t for s in (f"{v:,}".replace(",", " "), str(v), f"{k} тыс", f"{k}к", f"{k} к"))


def gen_table(df):
    ok = df[df.get("error").isna()] if "error" in df else df
    t = ok.groupby("config", sort=False).agg(
        ответов=("длина", "size"),
        отказ_фильтра_pct=("отказ_фильтра", lambda s: round(100 * s.mean(), 1)),
        придуманных_сумм=("придуманных_сумм", lambda s: round(s.mean(), 2)),
        выдуманных_id=("выдуманных_id", lambda s: round(s.mean(), 2)),
        обрезано_pct=("обрезано", lambda s: round(100 * s.mean(), 1)),
        покрытие_pct=("покрытие", lambda s: round(100 * s.mean(), 1)),
        длина=("длина", lambda s: int(s.mean())),
        p50_s=("latency", lambda s: round(s.median(), 2)),
        usd_per_1k=("cost", lambda s: round(1000 * s.mean(), 3)))
    if "error" in df:
        t["ошибок"] = df.groupby("config", sort=False).error.apply(lambda s: int(s.notna().sum()))
    return t


def run_judge(cfg_names, per_side=5, workers=8, seed=0):
    from evals import GOLDEN, judge_one
    from listings import load_listings
    desc = load_listings().set_index("listing_id").description.fillna("")
    desc = desc[desc.str.len() >= 40]
    pairs = []
    for g in [g for g in GOLDEN if g["pattern"]]:
        hit = desc[desc.str.contains(g["pattern"], regex=True)]
        miss = desc[~desc.str.contains(g["pattern"], regex=True)]
        for d in hit.sample(min(per_side, len(hit)), random_state=seed):
            pairs.append((g["ru"], d, 1))
        for d in miss.sample(per_side, random_state=seed):
            pairs.append((g["ru"], d, 0))
    rows = []
    for name in cfg_names:
        c = CONFIGS[name]
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            votes = list(ex.map(lambda p: judge_one((p[0], p[1]), providers=c["providers"]), pairs))
        y = np.array([p[2] for p, v in zip(pairs, votes) if v is not None])
        v = np.array([x for x in votes if x is not None])
        po = float((y == v).mean()) if len(v) else float("nan")
        pe = float(y.mean() * v.mean() + (1 - y.mean()) * (1 - v.mean())) if len(v) else float("nan")
        rows.append({"config": name, "пар": len(pairs), "ответил": len(v),
                     "согласие_pct": round(100 * po, 1), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
                     "полнота_pct": round(100 * float(v[y == 1].mean()), 1),
                     "ложные_да_pct": round(100 * float(v[y == 0].mean()), 1),
                     "сек_на_пару": round((time.time() - t0) / max(len(pairs), 1) * workers, 2)})
    return pd.DataFrame(rows).set_index("config")


def run(only=None, quick=False):
    from ab_generation import price_tasks, search_tasks
    from listings import load_listings
    report = {"when": time.strftime("%Y-%m-%d %H:%M")}
    gen_cfgs = ["alem", "gemini·minimal", "gemini·low"]

    if only in (None, "parse"):
        table, errors = run_parse(["alem", "gemini·minimal", "gemini·low"] if quick else list(CONFIGS))
        print(f"\n=== 1. Разбор запроса ({len(PARSE_GOLDEN)} запросов)\n{table.to_string()}")
        report["parse"] = {"table": table.reset_index().to_dict(orient="records"),
                           "errors": json.loads(json.dumps(errors, ensure_ascii=False, default=str))}
    if only in (None, "explain", "answer"):
        df = load_listings()
        n = 6 if quick else 20
        if only in (None, "explain"):
            tasks = [{**t, "task": "explain"} for t in price_tasks(df, n)]
            res = _gen_rows(tasks, gen_cfgs)
            t = gen_table(res)
            print(f"\n=== 2. Объяснение цены ({n} объявлений)\n{t.to_string()}")
            report["explain"] = {"table": t.reset_index().to_dict(orient="records"),
                                 "examples": res.drop_duplicates("config")[["config", "_text"]].to_dict("records")}
        if only in (None, "answer"):
            tasks = [{**t, "task": "answer"} for t in search_tasks(df, 3 if quick else 6)]
            res = _gen_rows(tasks, gen_cfgs)
            t = gen_table(res)
            print(f"\n=== 3. Ответ по выдаче\n{t.to_string()}")
            report["answer"] = {"table": t.reset_index().to_dict(orient="records"),
                                "examples": res.drop_duplicates("config")[["config", "_text"]].to_dict("records")}
    if only in (None, "judge"):
        t = run_judge(["alem", "gemini·minimal"], per_side=3 if quick else 5)
        print(f"\n=== 4. Судья релевантности\n{t.to_string()}")
        report["judge"] = {"table": t.reset_index().to_dict(orient="records")}

    path = ART_DIR / "ab_models.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(report)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str))
    print(f"\nсохранено: {path}")
    return report


def gate(report, config="gemini·minimal", min_exact=90.0, max_halluc=0.0) -> bool:
    """Порог для CI: регрессия разбора запроса (правка промпта, смена модели) роняет сборку."""
    rows = {r["config"]: r for r in report.get("parse", {}).get("table", [])}
    r = rows.get(config)
    if r is None:
        print(f"gate: нет результатов для {config}")
        return False
    ok = r["exact"] >= min_exact and r["hallucinated"] <= max_halluc
    print(f"gate {config}: exact {r['exact']}% (порог {min_exact}), выдумано {r['hallucinated']}% "
          f"(порог {max_halluc}) -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["parse", "explain", "answer", "judge"])
    ap.add_argument("--quick", action="store_true", help="меньше примеров — для CI")
    ap.add_argument("--gate", action="store_true", help="упасть, если разбор запроса хуже порога")
    a = ap.parse_args()
    rep = run(a.only, a.quick)
    if a.gate and not gate(rep):
        sys.exit(1)
