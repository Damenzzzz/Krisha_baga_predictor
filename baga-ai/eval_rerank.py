"""Нужен ли мультимодальный реранкер? precision@5 до и после на golden-запросах.

Кандидаты — топ-12 объявлений канала «по фото» (SigLIP 2, ру-запрос, фильтр по типу комнаты),
разметка — та же автоматическая, что в evals.py (признак упомянут в описании объявления).
Реранкер видит только фото — описание ему не показывается, поэтому разметка по описанию
остаётся независимой.

Запуск: python eval_rerank.py   ->  artifacts/rerank_results.json
"""
import json
import time

import numpy as np

from config import ART_DIR
from evals import GOLDEN, Bench, precision_auto


def run(k_cand=12, k=5):
    from embed import embed_texts
    from rerank import rerank
    bench = Bench()
    lid_best_path = {}
    rows = []
    for g in [g for g in GOLDEN if g["pattern"]]:
        q = embed_texts([g["ru"]])[0]
        ids = bench.photo(q, g["room"], k=k_cand)
        # лучшая фотография каждого кандидата под этот запрос
        cands = []
        for lid in ids:
            m = (bench.listing == lid) & bench.valid
            if g["room"]:
                m &= np.isin(bench.room, g["room"])
            i = np.flatnonzero(m)
            best = i[np.argmax(bench.E[i] @ q)] if len(i) else None
            cands.append({"listing_id": lid, "photos": [{"path": bench.idx.path.iloc[best]}] if best is not None else []})
        t0 = time.time()
        out, rep = rerank(g["ru"], cands, n=k_cand)
        after = [c["listing_id"] for c in out]
        rows.append({"id": g["id"], "до": precision_auto(bench, ids[:k], g["pattern"]),
                     "после": precision_auto(bench, after[:k], g["pattern"]),
                     "секунд": round(time.time() - t0, 1), "usd": rep.get("cost_usd", 0), "ok": rep["ok"]})
        print(f"  {g['id']:18s} {rows[-1]['до']:.1f} -> {rows[-1]['после']:.1f}  ({rows[-1]['секунд']} с)")
    before = np.mean([r["до"] for r in rows]) * 100
    after = np.mean([r["после"] for r in rows]) * 100
    report = {"precision@5 до": round(before, 1), "precision@5 после": round(after, 1),
              "запросов": len(rows), "успешно": sum(r["ok"] for r in rows),
              "секунд на запрос (медиана)": float(np.median([r["секунд"] for r in rows])),
              "$ на запрос": round(float(np.mean([r["usd"] for r in rows])), 5), "по запросам": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "по запросам"}, ensure_ascii=False))
    (ART_DIR / "rerank_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    run()
