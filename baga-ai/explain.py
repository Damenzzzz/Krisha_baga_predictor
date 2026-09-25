"""Объяснение цены через LLM по найденным аналогам (RAG).

Контекст модели — не «весь интернет», а конкретные похожие объявления из нашей базы.
Поэтому объяснение проверяемо: в тексте стоят id, по которым можно открыть исходные
объявления и убедиться.

Формулировки осторожные: мы знаем цены объявлений, а не сделок.
"""
from functools import lru_cache

from openai import OpenAI

from config import CHAT_KEY, CHAT_MAX_TOKENS, CHAT_MODEL, CHAT_TEMPERATURE, CHAT_URL

SYSTEM = """Ты помощник по аренде квартир в Казахстане. Отвечай кратко, 2-4 предложения, по-русски.

Правила:
- Опирайся ТОЛЬКО на приведённые данные: характеристики квартиры, диапазон модели и аналоги.
- Ничего не выдумывай: нет данных о ремонте или ЖК — не упоминай их.
- Говори «дороже/дешевле похожих объявлений», а не «переплата»: мы знаем цены объявлений, а не сделок.
- Ссылайся на аналоги в квадратных скобках по id, например [1015807119].
- Про надёжность пиши только то, что сказано в строке «надёжность»: не добавляй
  оговорку «аналогов мало», если там написано, что их достаточно."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not (CHAT_KEY and CHAT_URL):
        raise RuntimeError("нет ALEM_API_KEY / ALEM_URL в .env")
    import tracing
    return tracing.wrap_llm_client(OpenAI(api_key=CHAT_KEY, base_url=CHAT_URL))


def _clean(v):
    """pd.NA в условиях бросает «boolean value of NA is ambiguous» — приводим к None."""
    try:
        import pandas as pd
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
    except Exception:
        pass
    return v


def _facts(listing: dict, est: dict) -> str:
    listing = {k: _clean(v) for k, v in listing.items()}

    def money(x):
        return f"{int(x):,} ₸".replace(",", " ")

    lines = [
        "КВАРТИРА:",
        f"  {listing.get('rooms')}-комнатная, {listing.get('area')} м², {listing.get('district')}, "
        f"этаж {listing.get('floor')}/{listing.get('floors_total')}",
        f"  цена в объявлении: {money(listing['price'])} в месяц" if listing.get("price") else "  цена не указана",
        f"  состояние: {listing.get('condition') or 'не указано'}",
        "",
        "ОЦЕНКА МОДЕЛИ (по ценам объявлений, не сделок):",
        f"  типичный диапазон похожих: {money(est['p10'])} – {money(est['p90'])}, медиана {money(est['p50'])}",
        f"  вывод модели: {est['verdict']}",
        f"  похожих объявлений найдено: {est.get('n_similar', 0)}",
        "  надёжность: аналогов достаточно, оценке можно доверять" if est.get("reliable")
        else "  надёжность: аналогов МАЛО, оценка ненадёжна — обязательно предупреди об этом",
        "",
        "АНАЛОГИ:",
    ]
    for c in est.get("comparables", [])[:5]:
        lines.append(f"  [{c['listing_id']}] {money(c['price'])} · {c['area']} м² · "
                     f"{c['rooms']}-комн · {c['district']}")
    return "\n".join(lines)


def explain_price(listing: dict, est: dict, temperature: float | None = None) -> str:
    """Короткое объяснение вердикта. listing — карточка, est — ответ estimate_price."""
    r = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": _facts(listing, est) + "\n\nОбъясни вывод."}],
        temperature=CHAT_TEMPERATURE if temperature is None else temperature,
        max_tokens=CHAT_MAX_TOKENS)
    return r.choices[0].message.content.strip()


def explain_by_id(listing_id: str, temperature: float | None = None) -> str:
    import api
    return explain_price(api.get_listing(listing_id), api.estimate_price(listing_id=listing_id), temperature)


# --------------------------------------------------------------- ответ по результатам поиска

SEARCH_SYSTEM = """Ты помощник по аренде квартир. Пользователь описал желаемый ремонт,
система нашла объявления по фотографиям и описаниям. Напиши короткий ответ, 3-5 предложений.

Правила:
- Опирайся ТОЛЬКО на переданные объявления. Ничего не добавляй от себя.
- Ссылайся на объявления по id в квадратных скобках, например [1015807119].
- Скажи, что общего у находок и чем они отличаются (цена, район, площадь).
- Если у какого-то объявления цена выше диапазона похожих — предупреди об этом.
- Не обещай, что ремонт именно такой: мы сопоставили фотографии, а не проверяли квартиру.
- Не пиши «вы переплачиваете»: мы знаем цены объявлений, а не сделок."""


def _search_context(query: str, results: list[dict]) -> str:
    lines = [f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}", "", "НАЙДЕННЫЕ ОБЪЯВЛЕНИЯ:"]
    for r in results:
        l, pc = r.get("listing", {}), r.get("price_check", {}) or {}
        rooms = ", ".join(sorted({p["room_type"] for p in r.get("photos", [])})) or "—"
        price = f"{int(l['price']):,} ₸".replace(",", " ") if l.get("price") else "цена не указана"
        line = (f"[{r['listing_id']}] {price} · {l.get('rooms')}-комн · {l.get('area')} м² · "
                f"{l.get('district') or l.get('city')} · совпали фото: {rooms}")
        if pc.get("verdict"):
            line += f" · оценка: {pc['verdict']}"
        lines.append(line)
        desc = (l.get("description") or "")[:200]
        if desc:
            lines.append(f"      описание: {desc}")
    return "\n".join(lines)


def answer_search(query: str, results: list[dict], temperature: float | None = None) -> str:
    """Генерация ответа по результатам векторного поиска — это и есть шаг G в RAG:
    контекст собран ретривером (фото + описания), модель только пересказывает найденное."""
    if not results:
        return "По таким условиям ничего не нашлось."
    r = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SEARCH_SYSTEM},
                  {"role": "user", "content": _search_context(query, results)}],
        temperature=CHAT_TEMPERATURE if temperature is None else temperature,
        max_tokens=CHAT_MAX_TOKENS)
    return r.choices[0].message.content.strip()
