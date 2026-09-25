"""Смысловой кэш: перефразированный запрос не идёт в LLM второй раз.

«двушка в алматы до 400к, светлая кухня» и «2-комнатная Алматы до 400 тысяч, кухня
светлая» разбираются одинаково, а LLM-разбор стоит ~1–2 с и деньги. Кэш ищет ранее
разобранный запрос с близким эмбеддингом и отдаёт его разбор.

Главная опасность смыслового кэша — «похоже, но не то»: «до 400 тысяч» и «до 500 тысяч»
почти совпадают по эмбеддингу, а фильтры у них разные. Поэтому попадание требует ДВУХ
условий:
  1. совпадает жёсткая подпись запроса — числа, комнаты, город, районы, этажи, «от/до»,
     комнаты на фото, покупка/аренда; её считает детерминированный код, не модель;
  2. косинус эмбеддингов ≥ SEMANTIC_CACHE_THRESHOLD (подобран в eval_semantic_cache.py).
Подпись — ещё и индекс: сравниваем вектор только с запросами той же подписи.

Эмбеддинги — ALEM text-1024 (та же модель, что для описаний объявлений). Недоступна —
кэш просто промахивается, разбор идёт в LLM как обычно.
"""
import json
import re
import time

import numpy as np

import store
from config import SEMANTIC_CACHE_THRESHOLD

_NUM = re.compile(r"(?<![а-яa-z\d.,])(\d+(?:[\s ]\d{3})*(?:[.,]\d+)?)\s*(млн|миллион\w*|тыс\w*|тр\b|к\b|k\b)?")
_WORDS = {
    "r1": r"однушк|однокомнат|(?<![\d.,])1[\s-]*комн|(?<![\d.,])1[\s-]*к\b|(?<![\d.,])1-?х?\s*комн",
    "r2": r"двушк|двухкомнат|(?<![\d.,])2[\s-]*комн|(?<![\d.,])2[\s-]*к\b|(?<![\d.,])2-?х\s*комн",
    "r3": r"трешк|трехкомнат|(?<![\d.,])3[\s-]*комн|(?<![\d.,])3[\s-]*к\b|(?<![\d.,])3-?х\s*комн",
    "r4": r"четырехкомнат|(?<![\d.,])4[\s-]*комн|(?<![\d.,])4-?х\s*комн",
    "studio": r"студи",
    "almaty": r"алмат", "kaskelen": r"каскел", "other_city": r"астан|шымкент|караганд|актоб|атырау|павлодар",
    "not_first": r"не\s+перв", "not_last": r"не\s+последн", "first": r"(?<!не\s)перв\w*\s+этаж",
    "from": r"\bот\b|больше|дороже|выше", "to": r"\bдо\b|меньше|дешевле|не\s+дороже|максимум|ниже",
    "area": r"м2|м²|кв\.?\s*м|квадрат|метр",
    "kitchen": r"кухн", "bedroom": r"спальн", "bath": r"ванн|санузел|туалет|душ", "living": r"гостин|зал\b",
    "balcony": r"балкон|лодж", "hall": r"прихож|коридор", "kids": r"детск",
    "sale": r"куп[ил]|продаж|продает|ипотек",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


# «2-комнатная», «3х комн», «2к» — число комнат уже в словах подписи (r2), а не цена
_ROOMS_NUM = re.compile(r"(?<![\d.,])\d\s*-?\s*х?\s*(?:комн\w*|к\b)")


def _numbers(q: str) -> list[float]:
    out = []
    for m in _NUM.finditer(_ROOMS_NUM.sub(" ", q)):
        v = float(re.sub(r"[\s ]", "", m.group(1)).replace(",", "."))
        unit = (m.group(2) or "").lower()
        if unit.startswith(("млн", "миллион")):
            v *= 1e6
        elif unit.startswith(("тыс", "тр", "к", "k")):
            v *= 1e3
        out.append(round(v))
    return sorted(out)


_district_stems = None


def _districts() -> list[str]:
    """Корни названий районов: «бостандыкский р-н» -> «бостан». Корень, а не слово
    целиком: люди пишут «в Бостандыке», «бостандыкский», «Бостандык»."""
    global _district_stems
    if _district_stems is None:
        try:
            from listings import load_listings
            names = [n for n in load_listings().district.dropna().unique() if "р-н" in n or "район" in n]
            _district_stems = sorted({_norm(n).split()[0][:6] for n in names})
        except Exception:
            _district_stems = []
    return _district_stems


def signature(text: str) -> str:
    """Всё, что меняет фильтры. Два запроса с разной подписью никогда не делят кэш."""
    q = _norm(text)
    words = set(re.findall(r"[а-яa-z]+", q))
    sig = {
        "n": _numbers(q),
        "w": sorted(k for k, rx in _WORDS.items() if re.search(rx, q)),
        "d": sorted(s for s in _districts() if any(w.startswith(s) for w in words)),
    }
    return json.dumps(sig, ensure_ascii=False, sort_keys=True)


def _embed_alem(text: str) -> np.ndarray:
    from text_embed import embed_texts
    return embed_texts([text])[0].astype(np.float32)


class SemanticCache:
    def __init__(self, namespace: str, threshold: float = SEMANTIC_CACHE_THRESHOLD, embed=None,
                 ttl_s: float = 30 * 86400):
        self.namespace, self.threshold, self.ttl_s = namespace, threshold, ttl_s
        self.embed = embed or _embed_alem

    def _vec(self, text):
        try:
            v = np.asarray(self.embed(_norm(text)), dtype=np.float32)
            return v / (np.linalg.norm(v) + 1e-12)
        except Exception:
            return None                      # эмбеддер недоступен — работаем без кэша

    def lookup(self, text: str) -> tuple[dict | None, dict]:
        """(значение или None, отчёт: score, matched, signature)."""
        sig = signature(text)
        info = {"signature": sig, "score": None, "matched": None}
        rows = store.query("SELECT id, text, vec, value, created FROM semantic_cache "
                           "WHERE namespace = ? AND signature = ?", (self.namespace, sig))
        rows = [r for r in rows if time.time() - r["created"] < self.ttl_s]
        if not rows:
            return None, info
        v = self._vec(text)
        if v is None:
            return None, info
        M = np.stack([np.frombuffer(r["vec"], dtype=np.float16).astype(np.float32) for r in rows])
        s = M @ v
        i = int(np.argmax(s))
        info.update(score=round(float(s[i]), 4), matched=rows[i]["text"])
        if s[i] < self.threshold:
            return None, info
        store.execute("UPDATE semantic_cache SET hits = hits + 1 WHERE id = ?", (rows[i]["id"],))
        return json.loads(rows[i]["value"]), info

    def put(self, text: str, value: dict):
        v = self._vec(text)
        if v is None:
            return
        store.execute("INSERT INTO semantic_cache(namespace, signature, text, vec, value, created) "
                      "VALUES (?,?,?,?,?,?)", (self.namespace, signature(text), _norm(text),
                                               v.astype(np.float16).tobytes(),
                                               json.dumps(value, ensure_ascii=False), time.time()))

    def stats(self) -> dict:
        r = store.query("SELECT COUNT(*) AS n, COALESCE(SUM(hits), 0) AS hits FROM semantic_cache "
                        "WHERE namespace = ?", (self.namespace,))[0]
        return {"entries": r["n"], "hits": r["hits"]}
