"""Визуальный судья: Gemini смотрит на ФОТО найденного и решает, подходит ли оно запросу.

Текстовый судья (evals.py, метрика 3) читает описание объявления, а стиль ремонта там почти
не описан — поэтому он давал 11–16% «релевантных» при хорошей выдаче на глаз. Поиск идёт по
фото, значит и судить надо по фото.

Для каждого из 32 golden-запросов и каждой конфигурации поиска: топ-5 объявлений, у каждого
лучшее под запрос фото, одно обращение к Gemini с пятью картинками -> «да/нет» по каждой.

Проверка самого судьи: на 12 запросах с автоматической разметкой считается согласие
вердиктов судьи с разметкой по описанию (грубая, но независимая от картинки правда).

Оговорка: реранкер (rerank.py) — та же модель. Сравнение «с реранкером / без» этим судьёй
смещено в пользу реранкера; честная метрика для него — автоматическая разметка.

Запуск: python eval_visual_judge.py [--rerank]  ->  artifacts/visual_judge.json
"""
import argparse
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import ART_DIR
from evals import GOLDEN, Bench

K = 5
PROMPT = """Пользователь ищет квартиру и описал, что хочет увидеть: «{q}».
Ниже {n} фото из разных объявлений с номерами. Для каждого ответь, видно ли на фото то,
что просит пользователь (да — если явно видно или очень похоже, нет — иначе).
Верни ТОЛЬКО JSON: {{"v": [{{"i": 1, "ok": true}}, ...]}}"""


def best_photo(bench, lid, q, rooms):
    m = (bench.listing == lid) & bench.valid
    if rooms:
        m2 = m & np.isin(bench.room, rooms)
        m = m2 if m2.any() else m
    i = np.flatnonzero(m)
    return bench.idx.path.iloc[i[np.argmax(bench.E[i] @ q)]] if len(i) else None


def judge(query: str, paths: list[str | None]) -> list[int | None]:
    import llm
    from google.genai import types
    from rerank import _image_bytes
    parts, pos = [PROMPT.format(q=query, n=len(paths))], []
    for i, p in enumerate(paths, 1):
        img = _image_bytes(p, None) if p else None
        if img:
            parts += [f"[{i}]", types.Part.from_bytes(data=img, mime_type="image/jpeg")]
            pos.append(i)
    if not pos:
        return [None] * len(paths)
    try:
        r = llm.complete("", parts, task="visual_judge", temperature=0, max_tokens=300, json_mode=True,
                         thinking="minimal", providers=["gemini"], use_cache=False)
        v = {int(x["i"]): bool(x["ok"]) for x in json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))["v"]}
    except Exception:
        return [None] * len(paths)
    return [int(v[i]) if i in v else None for i in range(1, len(paths) + 1)]


def run(with_rerank=False, workers=6):
    from embed import embed_texts
    from text_embed import embed_texts as embed_desc
    bench = Bench()
    jobs = []
    for g in GOLDEN:
        qru, qen = embed_texts([g["ru"]])[0], embed_texts([g["en"]])[0]
        qmix = (qru + qen) / np.linalg.norm(qru + qen)
        configs = {"фото (ру)": bench.photo(qru, g["room"]), "фото (англ)": bench.photo(qen, g["room"]),
                   "фото (ру+англ)": bench.photo(qmix, g["room"])}
        configs["фото+описания (RRF)"] = bench.rrf(configs["фото (ру)"], bench.desc_search(embed_desc([g["ru"]])[0]))
        if with_rerank:
            from rerank import rerank
            cands = [{"listing_id": l, "photos": [{"path": best_photo(bench, l, qru, g["room"])}]}
                     for l in bench.photo(qru, g["room"], k=12)]
            configs["фото (ру) + реранкер"] = [c["listing_id"] for c in rerank(g["ru"], cands)[0]][:K]
        for name, ids in configs.items():
            jobs.append((g, name, ids[:K], [best_photo(bench, l, qru, g["room"]) for l in ids[:K]]))
        print(f"  {g['id']:22s} кандидаты готовы")
    with ThreadPoolExecutor(workers) as ex:
        verdicts = list(ex.map(lambda j: judge(j[0]["ru"], j[3]), jobs))
    rows = []
    for (g, name, ids, _), v in zip(jobs, verdicts):
        vv = [x for x in v if x is not None]
        rows.append({"id": g["id"], "config": name, "p@5": float(np.mean(vv)) if vv else np.nan,
                     "pattern": g["pattern"], "ids": ids, "v": v})
    table = {}
    for name in dict.fromkeys(r["config"] for r in rows):
        vals = [r["p@5"] for r in rows if r["config"] == name and not np.isnan(r["p@5"])]
        table[name] = round(100 * float(np.mean(vals)), 1)
    # согласие судьи с автоматической разметкой
    agree = []
    for r in rows:
        if r["pattern"]:
            for lid, v in zip(r["ids"], r["v"]):
                if v is not None:
                    agree.append((int(bool(re.search(r["pattern"], bench.desc.get(lid, "")))), v))
    y, v = np.array([a for a, _ in agree]), np.array([b for _, b in agree])
    report = {"visual_precision@5": table,
              "судья vs разметка": {"пар": len(agree), "согласие_pct": round(100 * float((y == v).mean()), 1),
                                    "полнота_pct": round(100 * float(v[y == 1].mean()), 1) if (y == 1).any() else None,
                                    "доля_да_где_разметка_нет_pct": round(100 * float(v[y == 0].mean()), 1)},
              "по запросам": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "по запросам"}, ensure_ascii=False, indent=1))
    (ART_DIR / "visual_judge.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    run(ap.parse_args().rerank)
