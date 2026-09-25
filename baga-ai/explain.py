"""Объяснение цены через LLM по найденным аналогам (RAG).

Контекст модели — не «весь интернет», а конкретные похожие объявления из нашей базы.
Поэтому объяснение проверяемо: в тексте стоят id, по которым можно открыть исходные
объявления и убедиться.

Формулировки осторожные: мы знаем цены объявлений, а не сделок.
"""
import llm
from guardrails import check_input, guard_output, mask_pii

SYSTEM = """Ты помощник по аренде квартир в Казахстане. Отвечай кратко, 2-4 предложения, по-русски.

Правила:
- Опирайся ТОЛЬКО на приведённые данные: характеристики квартиры, диапазон модели и аналоги.
- Ничего не выдумывай: нет данных о ремонте или ЖК — не упоминай их.
- Говори «дороже/дешевле похожих объявлений», а не «переплата»: мы знаем цены объявлений, а не сделок.
- Ссылайся на аналоги в квадратных скобках по id, например [1015807119].
- Про надёжность пиши только то, что сказано в строке «надёжность»: не добавляй
  оговорку «аналогов мало», если там написано, что их достаточно."""


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


def _complete(system: str, user: str, temperature=None, top_p=None, max_tokens=None, *,
              task: str = "generate", providers=None, thinking=None, use_cache=None) -> dict:
    """Один вызов модели через шлюз llm.py (Gemini → ALEM). Отдаёт и текст, и finish_reason:
    "length" значит, что ответ обрезан по max_tokens — одна из метрик A/B."""
    r = llm.complete(system, user, task=task, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                     providers=providers, thinking=thinking, use_cache=use_cache)
    return {"text": r.text, "finish_reason": r.finish_reason, "completion_tokens": r.completion_tokens,
            "provider": r.provider, "model": r.model, "latency_s": r.latency_s, "cost_usd": r.cost_usd,
            "thinking_tokens": r.thinking_tokens, "prompt_tokens": r.prompt_tokens, "cached": r.cached,
            "trace_id": r.trace_id}


def _money(x) -> str:
    return f"{int(x):,} ₸".replace(",", " ")


def price_facts(listing: dict, est: dict) -> tuple[list[float], list[str]]:
    """Числа и id, которые модели разрешено упоминать: всё, что есть в её контексте."""
    nums = [est.get("p10"), est.get("p50"), est.get("p90"), _clean(listing.get("price"))]
    comps = est.get("comparables", [])[:5]
    nums += [c["price"] for c in comps]
    return nums, [str(c["listing_id"]) for c in comps]


def template_explanation(listing: dict, est: dict) -> str:
    """Запасной ответ без LLM — из тех же фактов. Подставляется, если модель выдумала
    сумму или id, и когда LLM недоступна."""
    price = _clean(listing.get("price"))
    head = f"В объявлении {_money(price)} в месяц. " if price else ""
    text = (f"{head}Похожие объявления сдают за {_money(est['p10'])} – {_money(est['p90'])}, "
            f"медиана {_money(est['p50'])}: {est['verdict']}.")
    comps = est.get("comparables", [])[:3]
    if comps:
        text += " Аналоги: " + ", ".join(f"[{c['listing_id']}] {_money(c['price'])}" for c in comps) + "."
    if not est.get("reliable"):
        text += " Похожих объявлений мало, оценка ненадёжна."
    return text


def explain_price_raw(listing: dict, est: dict, temperature=None, top_p=None, max_tokens=None, **kw) -> dict:
    """Ответ модели БЕЗ выходного фильтра — для экспериментов, где меряется сама модель."""
    return _complete(SYSTEM, _facts(listing, est) + "\n\nОбъясни вывод.", temperature, top_p, max_tokens,
                     task="explain_price", **kw)


def explain_price(listing: dict, est: dict, temperature: float | None = None) -> str:
    """Короткое объяснение вердикта. listing — карточка, est — ответ estimate_price.
    Прошедшее через выходной фильтр: выдуманная сумма или id -> шаблонный ответ.
    Все модели недоступны -> тот же шаблон: вердикт не зависит от LLM, только формулировка."""
    try:
        raw = explain_price_raw(listing, est, temperature)["text"]
    except llm.LLMUnavailable as e:
        _log_fallback("explain_price", e)
        return template_explanation(listing, est)
    nums, ids = price_facts(listing, est)
    text, report = guard_output(raw, nums, ids, fallback=template_explanation(listing, est))
    _log_guard("explain_price", report)
    return text


def _log_fallback(where: str, err: Exception):
    import sys
    from tracing import span
    print(f"[llm] {where}: модели недоступны, ответ собран шаблоном — {err}", file=sys.stderr)
    with span("llm_unavailable_template", kind="GUARDRAIL", where=where) as sp:
        sp.set(output=str(err)[:300])


def _log_guard(where: str, report: dict):
    if not report["ok"]:
        import sys
        from tracing import span
        print(f"[guardrails] {where}: ответ модели заменён шаблоном {report}", file=sys.stderr)
        with span("guardrail_output_blocked", kind="GUARDRAIL", where=where, report=str(report)):
            pass


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
        desc = mask_pii((l.get("description") or "")[:200])[0]     # телефоны риелторов — не в LLM
        if desc:
            lines.append(f"      описание: {desc}")
    return "\n".join(lines)


def answer_search(query: str, results: list[dict], temperature: float | None = None) -> str:
    """Генерация ответа по результатам векторного поиска — это и есть шаг G в RAG:
    контекст собран ретривером (фото + описания), модель только пересказывает найденное."""
    if not results:
        return "По таким условиям ничего не нашлось."
    guard = check_input(query)
    # инъекцию в LLM не передаём: модель пересказывает выдачу, не видя текста запроса
    shown_query = "(текст запроса скрыт фильтром безопасности)" if guard.blocked else guard.text
    try:
        raw = _complete(SEARCH_SYSTEM, _search_context(shown_query, results), temperature,
                        task="answer_search")["text"]
    except llm.LLMUnavailable as e:
        _log_fallback("answer_search", e)
        return _template_search(results)

    nums, ids = [], []
    for r in results:
        pc = r.get("price_check") or {}
        nums += [r.get("listing", {}).get("price"), pc.get("p10"), pc.get("p50"), pc.get("p90")]
        ids.append(str(r["listing_id"]))
    text, report = guard_output(raw, nums, ids, fallback=_template_search(results))
    _log_guard("answer_search", report)
    return text


def _template_search(results: list[dict]) -> str:
    parts = []
    for r in results[:3]:
        l, pc = r.get("listing", {}), r.get("price_check") or {}
        price = _money(l["price"]) if l.get("price") else "цена не указана"
        verdict = f", {pc['verdict']}" if pc.get("verdict") else ""
        parts.append(f"[{r['listing_id']}] {price}, {l.get('rooms')}-комн, "
                     f"{l.get('district') or l.get('city')}{verdict}")
    return "Ближе всего к запросу по фотографиям и описаниям: " + "; ".join(parts) + "."
