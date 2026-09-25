"""Оценка качества поиска: golden dataset, две метрики, A/B эксперименты.

Метрика 1 — precision@5 по автоматической разметке.
    Берём признаки, которые ВИДНО на фото и которые авторы упоминают в описании
    (гардеробная, посудомоечная машина, балкон, панорамные окна…). Ищем ТОЛЬКО по
    фотографиям, описания в поиске не участвуют. Релевантным считается объявление,
    в описании которого этот признак упомянут. Разметка независимая: описание писал
    человек, модель смотрела на картинку.
    Это нижняя оценка: если балкон есть, но в описании не написан, мы засчитаем промах.
    Зато сравнение конфигураций честное, и разметка не стоит ни часа ручной работы.
    База для сравнения — доля таких объявлений в корпусе (сколько дал бы случайный выбор).

Метрика 2 — LLM-as-judge.
    Для стилевых запросов («уютная кухня в тёплых тонах») формального признака нет.
    Судья (шлюз llm.py: Gemini 3.6 Flash → ALEM, temperature=0) видит запрос и описание найденного объявления
    и отвечает, подходит ли. Считаем долю «да» в топ-5.

Эксперименты: каналы (фото / описания / слияние RRF), язык запроса (ру / англ / оба),
температура генерации объяснений.

Запуск: python evals.py [--quick]
Работает по локальным матрицам, не занимая Qdrant, — сайт может работать параллельно.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config import ART_DIR, RENDER_THR, RRF_K

# --------------------------------------------------------------- golden dataset

# pattern — регулярка для автоматической разметки по описанию (если признак виден на фото)
GOLDEN = [
    # --- проверяемые автоматически
    {"id": "wardrobe", "ru": "гардеробная комната со стеллажами", "en": "walk-in closet with shelves",
     "room": None, "pattern": r"гардеробн"},
    {"id": "dishwasher", "ru": "кухня с посудомоечной машиной", "en": "kitchen with a dishwasher",
     "room": ["kitchen"], "pattern": r"посудомо"},
    {"id": "balcony", "ru": "застеклённый балкон", "en": "glazed balcony",
     "room": ["balcony"], "pattern": r"(балкон|лоджи)"},
    {"id": "panoramic", "ru": "панорамные окна во всю стену", "en": "floor to ceiling panoramic windows",
     "room": None, "pattern": r"панорамн"},
    {"id": "studio", "ru": "квартира студия, кухня и кровать в одной комнате",
     "en": "studio apartment, kitchen and bed in one room", "room": None, "pattern": r"студи"},
    {"id": "kitchen_living", "ru": "кухня-гостиная с островом", "en": "open plan kitchen living room with island",
     "room": ["kitchen", "living_room"], "pattern": r"кухня-гостиная"},
    {"id": "aircon", "ru": "комната с кондиционером на стене", "en": "room with an air conditioner on the wall",
     "room": None, "pattern": r"кондиционер"},
    {"id": "designer", "ru": "дизайнерский ремонт премиум класса", "en": "premium designer interior",
     "room": None, "pattern": r"дизайнерск"},
    {"id": "new_building", "ru": "квартира в новом жилом комплексе", "en": "apartment in a new residential complex",
     "room": None, "pattern": r"(новостройк|жилой комплекс|жк )"},
    {"id": "unfurnished", "ru": "пустая квартира без мебели", "en": "empty apartment without furniture",
     "room": None, "pattern": r"без мебели"},
    {"id": "mountain_view", "ru": "вид на горы из окна", "en": "mountain view from the window",
     "room": ["window_view"], "pattern": r"(вид на горы|видовая)"},
    {"id": "sauna", "ru": "сауна в квартире", "en": "sauna inside the apartment", "room": None, "pattern": r"саун"},

    # --- стилевые: формального признака нет, судит LLM
    {"id": "scandi_kitchen", "ru": "светлая кухня в скандинавском стиле, белые фасады",
     "en": "bright scandinavian kitchen with white cabinets", "room": ["kitchen"], "pattern": None},
    {"id": "soviet", "ru": "старый советский ремонт, ковёр, обои",
     "en": "old soviet style room with carpet and wallpaper", "room": None, "pattern": None},
    {"id": "loft", "ru": "лофт, кирпичная стена, бетон", "en": "loft interior with brick wall and concrete",
     "room": None, "pattern": None},
    {"id": "minimal_bedroom", "ru": "минималистичная спальня в светлых тонах",
     "en": "minimalist bedroom in light colors", "room": ["bedroom"], "pattern": None},
    {"id": "dark_bath", "ru": "ванная с тёмной плиткой под мрамор", "en": "bathroom with dark marble tiles",
     "room": ["bathroom"], "pattern": None},
    {"id": "wood_kitchen", "ru": "кухня с деревянной столешницей", "en": "kitchen with wooden countertop",
     "room": ["kitchen"], "pattern": None},
    {"id": "cozy_warm", "ru": "уютная комната в тёплых бежевых тонах", "en": "cozy room in warm beige tones",
     "room": None, "pattern": None},
    {"id": "empty_repair", "ru": "квартира требует ремонта, голые стены",
     "en": "apartment needs renovation, bare walls", "room": None, "pattern": None},
    {"id": "classic", "ru": "классический интерьер с лепниной и люстрой",
     "en": "classic interior with moldings and a chandelier", "room": None, "pattern": None},
    {"id": "green_wall", "ru": "комната с зелёными стенами", "en": "room with green walls", "room": None, "pattern": None},
    {"id": "big_sofa", "ru": "гостиная с большим угловым диваном", "en": "living room with a large corner sofa",
     "room": ["living_room"], "pattern": None},
    {"id": "tiny_kitchen", "ru": "маленькая кухня в хрущёвке", "en": "tiny old soviet kitchen",
     "room": ["kitchen"], "pattern": None},
    {"id": "white_bath", "ru": "белая ванная с душевой кабиной", "en": "white bathroom with a shower cabin",
     "room": ["bathroom"], "pattern": None},
    {"id": "parquet", "ru": "паркет ёлочкой на полу", "en": "herringbone parquet floor", "room": None, "pattern": None},
    {"id": "kids_room", "ru": "детская комната", "en": "children room", "room": ["bedroom"], "pattern": None},
    {"id": "led_ceiling", "ru": "натяжной потолок с подсветкой", "en": "stretch ceiling with led lighting",
     "room": None, "pattern": None},
    {"id": "corridor_closet", "ru": "прихожая со встроенным шкафом", "en": "hallway with a built-in wardrobe",
     "room": ["hallway"], "pattern": None},
    {"id": "granite_kitchen", "ru": "тёмная кухня с каменной столешницей", "en": "dark kitchen with stone countertop",
     "room": ["kitchen"], "pattern": None},
    {"id": "old_bathroom", "ru": "старая ванная с советской плиткой", "en": "old bathroom with soviet tiles",
     "room": ["bathroom"], "pattern": None},
    {"id": "large_windows_living", "ru": "просторная гостиная с большими окнами",
     "en": "spacious living room with large windows", "room": ["living_room"], "pattern": None},
]

K = 5

# --------------------------------------------------------------- поисковые конфигурации


class Bench:
    """Локальный поиск по матрицам: не занимает Qdrant, чтобы сайт мог работать параллельно."""

    def __init__(self):
        from embed import load_embeddings
        from listings import load_listings
        from rooms import load_rooms
        self.idx, self.E, self.valid = load_embeddings()
        r = load_rooms()
        self.room, self.render = r.room_type.to_numpy(), r.render_prob.to_numpy()
        self.listing = self.idx.listing_id.to_numpy()
        self.df = load_listings().set_index("listing_id")
        self.desc = self.df.description.fillna("").str.lower()

        from config import DESC_EMB_PATH, DESC_IDS_PATH
        self.De = np.load(DESC_EMB_PATH).astype(np.float32)
        self.Dids = np.load(DESC_IDS_PATH)

    def photo(self, text_vec, rooms=None, k=K):
        s = self.E @ text_vec
        mask = self.valid & (self.render < RENDER_THR)
        if rooms:
            mask &= np.isin(self.room, rooms)
        s = np.where(mask, s, -np.inf)
        out, seen = [], set()
        for i in np.argsort(-s):
            lid = self.listing[i]
            if lid in seen or not np.isfinite(s[i]):
                continue
            seen.add(lid)
            out.append(lid)
            if len(out) == k:
                break
        return out

    def desc_search(self, vec, k=K):
        s = self.De @ vec
        return [self.Dids[i] for i in np.argsort(-s)[:k]]

    @staticmethod
    def rrf(*rankings, k=K):
        score = {}
        for lst in rankings:
            for rank, lid in enumerate(lst, 1):
                score[lid] = score.get(lid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(score, key=score.get, reverse=True)[:k]


# --------------------------------------------------------------- метрики

def precision_auto(bench: Bench, listing_ids, pattern: str) -> float:
    """Доля найденных, у которых признак упомянут в описании."""
    hits = [bool(re.search(pattern, bench.desc.get(lid, ""))) for lid in listing_ids]
    return float(np.mean(hits)) if hits else 0.0


def base_rate(bench: Bench, pattern: str) -> float:
    return float(bench.desc.str.contains(pattern, regex=True).mean())


JUDGE_SYSTEM = """Ты оцениваешь релевантность квартиры поисковому запросу.
Тебе дают запрос пользователя о желаемом интерьере и описание найденной квартиры.
Ответь ОДНИМ словом: "да" — если квартира правдоподобно подходит под запрос, "нет" — если нет.
Если данных в описании недостаточно, отвечай "нет"."""


def judge_one(args, providers=None) -> int | None:
    """1 — релевантно, 0 — нет, None — судья не ответил (не считаем за «нет»,
    иначе падение модели выглядело бы как плохой поиск)."""
    query, description = args
    import llm
    from config import JUDGE_CHAIN
    try:
        r = llm.complete(JUDGE_SYSTEM, f"Запрос: {query}\n\nОписание квартиры: {description[:700]}",
                         task="judge", temperature=0, max_tokens=10, thinking="minimal",
                         providers=providers or JUDGE_CHAIN)
        return int("да" in r.text.strip().lower())
    except Exception:
        return None


def judge_precision(bench: Bench, query: str, listing_ids, workers=8) -> float:
    pairs = [(query, bench.desc.get(lid, "")) for lid in listing_ids]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        votes = [v for v in ex.map(judge_one, pairs) if v is not None]
    return float(np.mean(votes)) if votes else float("nan")


# --------------------------------------------------------------- прогон

def run(quick=False, judge=True):
    from embed import embed_texts
    from text_embed import embed_texts as embed_desc
    bench = Bench()
    golden = GOLDEN[:8] if quick else GOLDEN
    rows = []

    for g in golden:
        qru = embed_texts([g["ru"]])[0]
        qen = embed_texts([g["en"]])[0]
        qmix = (qru + qen) / np.linalg.norm(qru + qen)
        photo_ru = bench.photo(qru, g["room"])
        photo_en = bench.photo(qen, g["room"])
        photo_mix = bench.photo(qmix, g["room"])
        d_ru = bench.desc_search(embed_desc([g["ru"]])[0])
        fused = bench.rrf(photo_ru, d_ru)

        row = {"id": g["id"], "запрос": g["ru"]}
        configs = {"фото (ру)": photo_ru, "фото (англ)": photo_en, "фото (ру+англ)": photo_mix,
                   "описания": d_ru, "фото+описания (RRF)": fused}
        if g["pattern"]:
            row["база (случайно)"] = base_rate(bench, g["pattern"])
            for name, ids in configs.items():
                row[f"auto:{name}"] = precision_auto(bench, ids, g["pattern"])
        if judge:
            for name in ("фото (ру)", "фото (англ)", "фото+описания (RRF)"):
                row[f"judge:{name}"] = judge_precision(bench, g["ru"], configs[name])
        rows.append(row)
        print(f"  {g['id']:22s} ok")

    res = pd.DataFrame(rows)
    ART_DIR.mkdir(exist_ok=True)
    res.to_json(ART_DIR / "evals_results.json", orient="records", force_ascii=False, indent=1)

    print(f"\n{'=' * 70}\nЗапросов в наборе: {len(res)}\n")
    auto_cols = [c for c in res.columns if c.startswith("auto:")]
    if auto_cols:
        auto = res.dropna(subset=auto_cols)
        print(f"МЕТРИКА 1 — precision@{K} по автоматической разметке ({len(auto)} запросов):")
        summary = auto[auto_cols].mean().rename(lambda c: c.replace("auto:", "")).sort_values(ascending=False)
        print((summary * 100).round(1).to_string())
        print(f"{'случайный выбор':22s} {auto['база (случайно)'].mean() * 100:.1f}")
    judge_cols = [c for c in res.columns if c.startswith("judge:")]
    if judge_cols:
        print(f"\nМЕТРИКА 2 — LLM-as-judge, доля релевантных в топ-{K} ({len(res)} запросов):")
        print((res[judge_cols].mean().rename(lambda c: c.replace("judge:", "")) * 100).round(1).to_string())
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="только первые 8 запросов")
    p.add_argument("--no-judge", action="store_true", help="без LLM-судьи")
    a = p.parse_args()
    run(quick=a.quick, judge=not a.no_judge)
