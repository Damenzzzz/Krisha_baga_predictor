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
import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from config import CHAT_KEY, CHAT_MODEL, CHAT_URL
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


def _client():
    from openai import OpenAI
    import tracing
    cl = OpenAI(api_key=CHAT_KEY, base_url=CHAT_URL)
    return tracing.wrap_llm_client(cl)


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


def parse(query: str, use_llm: bool = True) -> ParsedQuery:
    if not use_llm:
        return _rule_based(query)
    try:
        r = _client().chat.completions.create(
            model=CHAT_MODEL, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Известные районы: {', '.join(_known_districts())}\n\n"
                                                  f"Запрос: {query}"}])
        d = _json_from(r.choices[0].message.content)
    except Exception:
        return _rule_based(query)          # фолбэк: пользователь всё равно получит выдачу

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
