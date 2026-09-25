"""Мультимодальный реранкер: Gemini смотрит на фото кандидатов и переставляет их.

SigLIP сравнивает запрос и фото одним косинусом: «кухня с посудомоечной машиной» для него
почти то же, что «кухня». Реранкер получает запрос и по одной лучшей фотографии каждого из
топ-N кандидатов и ставит оценку 0–10 — он видит мелкие детали, которые вектор усредняет.

Стоимость: одно обращение на запрос, 12 картинок по 384 px (≈ 260 токенов каждая) —
≈ $0.003 и +3–5 с. Поэтому выключен по умолчанию (RERANK=1) и решение о включении —
по evals (eval_rerank.py, EVALS.md раздел 9). Любая ошибка — исходный порядок.
"""
import io
import json
import os
import re

from PIL import Image

RERANK_N = int(os.getenv("RERANK_N", 12))
PROMPT = """Пользователь ищет квартиру в аренду и описал, как она должна выглядеть:
«{query}»

Ниже {n} фотографий из разных объявлений, по одной на объявление, с номерами.
Оцени каждую от 0 до 10: насколько на фото видно то, что просит пользователь.
10 — точно оно, 0 — совсем не то. Оценивай только то, что видно на фото.
Верни ТОЛЬКО JSON: {{"scores": [{{"i": 1, "s": 7}}, ...]}}"""


def enabled() -> bool:
    return os.getenv("RERANK", "0") == "1"


def _image_bytes(path: str | None, url: str | None, side: int = 384) -> bytes | None:
    from config import PHOTOS_DIR
    raw = None
    if path and (PHOTOS_DIR / path).exists():
        raw = (PHOTOS_DIR / path).read_bytes()
    elif url:
        import httpx
        try:
            raw = httpx.get(url, timeout=10).content
        except Exception:
            return None
    if not raw:
        return None
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def rerank(query: str, candidates: list[dict], url_of=None, n: int = RERANK_N) -> tuple[list[dict], dict]:
    """candidates — [{listing_id, photos: [{path}], ...}] в порядке ретривера.
    Возвращает (переставленный список, отчёт). Хвост после n не трогается."""
    import llm
    from google.genai import types
    head, tail = candidates[:n], candidates[n:]
    parts, idx = [PROMPT.format(query=query, n=len(head))], []
    for i, c in enumerate(head, 1):
        ph = (c.get("photos") or [{}])[0]
        img = _image_bytes(ph.get("path"), url_of(ph.get("path"), c["listing_id"]) if url_of else None)
        if img is None:
            continue
        parts += [f"[{i}]", types.Part.from_bytes(data=img, mime_type="image/jpeg")]
        idx.append(i)
    if len(idx) < 2:
        return candidates, {"ok": False, "reason": "нет фото"}
    try:
        r = llm.complete("", parts, task="rerank", temperature=0, max_tokens=400, json_mode=True,
                         thinking="minimal", providers=["gemini"], use_cache=False)
        m = re.search(r"\{.*\}", r.text, re.S)
        scores = {int(x["i"]): float(x["s"]) for x in json.loads(m.group(0))["scores"]}
    except Exception as e:
        return candidates, {"ok": False, "reason": str(e)[:200]}
    order = sorted(range(1, len(head) + 1), key=lambda i: (-scores.get(i, -1), i))
    out = [head[i - 1] | {"rerank_score": scores.get(i)} for i in order]
    return out + tail, {"ok": True, "cost_usd": r.cost_usd, "latency_s": r.latency_s, "n": len(idx)}
