"""Разбор запроса пользователя на жёсткие условия и стилевую часть.

«Двушка в Алматы до 400 тысяч, светлая кухня в скандинавском стиле, не первый этаж»
  -> Filters(city="алматы", rooms=[2], price_max=400000, not_first_floor=True)
  -> style = "светлая кухня в скандинавском стиле"

Почему разделяем: «до 400 тысяч» эмбеддингом невыразимо (вектор не умеет в сравнения),
а «скандинавский стиль» невыразим фильтром. Числа идут в Qdrant, стиль — в поиск по фото.

Главный риск — модель придумывает условия, которых человек не называл: поставит
price_max, и половина вариантов молча отвалится. Поэтому в промпте жёстко: не сказано — null,
а всё, что модель всё же добавила от себя, помечается и выносится на подтверждение
пользователю (assumptions).
"""
import hashlib
import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from guardrails import check_input
from listings import Filters, load_listings

SYSTEM = """Ты разбираешь запрос об аренде квартиры на структурированные условия и описание стиля.

Верни ТОЛЬКО JSON без пояснений:
{"city": null|"алматы"|"каскелен",
 "districts": null|["часть названия района"],
 "rooms": null|[числа],
 "price_min": null|число, "price_max": null|число,
 "area_min": null|число, "area_max": null|число,
 "not_first_floor": true|false, "not_last_floor": true|false,
 "photo_rooms": null|["kitchen","living_room","bedroom","bathroom","hallway","balcony"],
 "style": "часть запроса про внешний вид и ремонт, пустая строка если её нет"}

Правила:
- Чего пользователь НЕ говорил — null. Не додумывай цену, район или площадь.
- «двушка» = rooms [2], «однушка» = [1], «трёшка» = [3]. «до 400 тысяч» = price_max 400000,
  «400к» = 400000, «от 200 до 300 тысяч» = price_min 200000, price_max 300000.
- Цены в тенге за месяц.
- districts — часть названия: «в Бостандыке» -> ["бостандык"].
- photo_rooms ставь, только если человек явно назвал комнату («кухня», «санузел»).
- style — только про внешний вид: «светлая кухня в скандинавском стиле». Без цен и районов."""


class ParsedQuery(BaseModel):
    filters: Filters
    style: str = ""
    photo_rooms: Optional[list[str]] = None
    assumptions: list[str] = Field(default_factory=list,
                                   description="условия, которых пользователь явно не называл (текстом)")
    assumed_fields: list[str] = Field(default_factory=list,
                                      description="имена этих полей — по ним их можно убрать при отказе")
    warnings: list[str] = Field(default_factory=list,
                                description="предупреждения входного фильтра: другой город, покупка, инъекция")
    source: str = Field("", description="кто разобрал: llm:<провайдер>, cache, rules")


# Версия промпта входит в ключ кэша: поменяли промпт — старые разборы не используются.
PROMPT_VERSION = hashlib.sha1(SYSTEM.encode()).hexdigest()[:8]
_cache = None


def _semantic_cache():
    global _cache
    if _cache is None:
        from semantic_cache import SemanticCache
        _cache = SemanticCache(f"parse_query:{PROMPT_VERSION}")
    return _cache


def _json_from(text: str) -> dict:
    """Модель иногда оборачивает JSON в ```json или добавляет фразу до него."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"в ответе нет JSON: {text[:200]}")
    return json.loads(m.group(0))


def _known_districts() -> list[str]:
    return sorted(load_listings().district.dropna().unique())


def _rule_based(query: str) -> ParsedQuery:
    """Запасной разбор без LLM: ловит числа и ключевые слова. Нужен, чтобы поиск
    работал, даже когда LLM недоступна."""
    q = query.lower()
    f = Filters()
    if "алматы" in q:
        f.city = "алматы"
    elif "каскелен" in q:
        f.city = "каскелен"
    for word, n in [("однушк", 1), ("1-комн", 1), ("двушк", 2), ("2-комн", 2),
                    ("трёшк", 3), ("трешк", 3), ("3-комн", 3)]:
        if word in q:
            f.rooms = [n]
    m = re.search(r"до\s*(\d[\d\s]*)\s*(тыс|к\b|000)?", q)
    if m:
        val = float(re.sub(r"\D", "", m.group(1)))
        f.price_max = val * 1000 if val < 10_000 else val
    return ParsedQuery(filters=f, style=query, assumptions=["разбор без LLM (запасной режим)"])


def parse(query: str, use_llm: bool = True, use_cache: bool = True) -> ParsedQuery:
    guard = check_input(query)
    query = guard.text                     # PII замаскированы — в LLM и трейсы они не уходят
    if guard.blocked:
        # инъекцию в модель не отправляем; регулярки её «не услышат», а стиль всё равно
        # пойдёт в поиск по фото — пользователь получит выдачу, модель не получит команд
        p = _rule_based(query)
        p.assumptions = []
        p.warnings = guard.warnings + ["запрос содержит инструкции для модели — разобран без LLM"]
        return p
    if not use_llm:
        p = _rule_based(query)
        p.warnings, p.source = guard.warnings, "rules"
        return p
    cached, _ = _semantic_cache().lookup(query) if use_cache else (None, {})
    if cached is not None:
        p = ParsedQuery(**cached)
        p.warnings, p.source = guard.warnings, "cache"
        return p
    try:
        d, r = llm_parse(query)
    except Exception:
        p = _rule_based(query)             # фолбэк: пользователь всё равно получит выдачу
        p.warnings = guard.warnings
        return p
    p = _from_llm(query, d)
    p.warnings, p.source = guard.warnings, f"llm:{r.provider}"
    if use_cache:
        _semantic_cache().put(query, p.model_dump(exclude={"warnings", "source"}))
    return p


def llm_parse(query: str, providers=None, thinking=None):
    """Сырой разбор моделью: (dict, LLMResult). Отдельно — чтобы A/B сравнивал модели
    на одном и том же промпте без кэша и guardrails."""
    import llm
    user = f"Известные районы: {', '.join(_known_districts())}\n\nЗапрос: {query}"
    return llm.complete_json(SYSTEM, user, task="parse_query", temperature=0, max_tokens=400,
                             providers=providers, thinking=thinking, use_cache=False)


def _from_llm(query: str, d: dict) -> ParsedQuery:
    d = dict(d)

    style = (d.pop("style", "") or "").strip()
    photo_rooms = d.pop("photo_rooms", None)
    # null у булевых полей ломает валидацию — пустые значения просто отбрасываем
    f = Filters(**{k: v for k, v in d.items() if k in Filters.model_fields and v is not None})

    # что модель добавила от себя — на подтверждение пользователю
    assumptions, assumed_fields = [], []
    if f.price_max is not None and not re.search(r"\d", query):
        assumptions.append(f"цена до {int(f.price_max):,} ₸".replace(",", " "))
        assumed_fields.append("price_max")
    if f.city and f.city not in query.lower():
        assumptions.append(f"город: {f.city}")
        assumed_fields.append("city")
    if f.rooms and not re.search(r"\d|комн|однушк|двушк|трёшк|трешк", query.lower()):
        assumptions.append(f"комнат: {f.rooms}")
        assumed_fields.append("rooms")
    return ParsedQuery(filters=f, style=style or query, photo_rooms=photo_rooms,
                       assumptions=assumptions, assumed_fields=assumed_fields)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "двушка в Алматы до 400 тысяч, светлая кухня, не первый этаж"
    print(parse(q).model_dump_json(indent=2, exclude_none=True))
