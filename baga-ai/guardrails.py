"""Входная и выходная фильтрация вокруг LLM.

Своя реализация, а не Guardrails AI / NeMo: у нас всего три LLM-вызова с узкими задачами
(разбор запроса, объяснение цены, ответ по выдаче), и для каждого известно, что считать
нарушением. Фреймворк добавил бы зависимость и ещё один LLM-вызов на проверку, а здесь
проверки детерминированные: регулярки на входе и сверка с фактами на выходе. Работают
за микросекунды, не стоят токенов и одинаково ведут себя в evals и в проде.

ВХОД (check_input) — перед тем, как текст пользователя попадёт в промпт:
  prompt injection — «игнорируй инструкции», «ты теперь…», служебные теги ролей.
      Такой запрос НЕ отправляется в LLM: разбор идёт запасным путём на регулярках,
      поиск по стилю продолжает работать. Пользователь получает выдачу, модель — ничего.
  PII — телефоны, e-mail, ИИН, номера карт маскируются. Люди вставляют текст объявления
      целиком вместе с телефоном риелтора, а запрос уходит во внешний API и в трейсы.
  домен — другие города и покупка вместо аренды. Не блокируем, а предупреждаем:
      в базе только аренда в Алматы и Каскелене, молча выдать Алматы на запрос про
      Астану — хуже, чем сказать об этом.

ВЫХОД (check_output) — после ответа модели:
  придуманные суммы — каждое число-сумма в тексте должно совпасть с фактом из контекста;
  выдуманные id     — всё в [квадратных скобках] должно быть среди переданных объявлений;
  запрещённые формулировки — «вы переплачиваете»: мы знаем цены объявлений, а не сделок;
  чужой город      — в базе только Алматы и Каскелен. Найдено на прогоне агента: на запрос
                     «однушка в Астане» модель написала «квартиры в Астане», хотя выдача алматинская.
  Нарушение -> ответ модели заменяется шаблонным текстом из тех же фактов (guard_output).
"""
import re
from dataclasses import dataclass, field

MAX_QUERY_CHARS = 1000

# --------------------------------------------------------------- prompt injection

_INJECTION = [
    # «игнорируй / забудь предыдущие инструкции»
    r"(игнорир\w*|забуд\w*|отмен\w*|не\s+обращай\s+внимания\s+на|не\s+следуй)\s+(\w+\s+){0,3}"
    r"(инструкци|правил|указани|промпт|ограничени)",
    r"(ignore|disregard|forget|override)\s+(\w+\s+){0,3}(instructions?|rules|prompt|guidelines)",
    # смена роли
    r"\b(ты\s+теперь|отныне\s+ты|теперь\s+ты|представь,?\s+что\s+ты|притворись)\b",
    r"\b(you\s+are\s+now|from\s+now\s+on\s+you|pretend\s+(to\s+be|you\s+are)|act\s+as\s+(a|an)\b)",
    # вытащить системный промпт
    r"(покажи|выведи|напечатай|раскрой|повтори)\s+(\w+\s+){0,3}(системн\w*\s+)?(промпт|инструкци|prompt)",
    r"(reveal|print|show|repeat)\s+(\w+\s+){0,3}(system\s+)?(prompt|instructions)",
    r"(системн\w+\s+промпт|system\s+prompt)",
    # известные джейлбрейки и служебная разметка ролей
    r"\b(jailbreak|DAN\s+mode|developer\s+mode|режим\s+разработчика)\b",
    r"(<\s*/?\s*(system|assistant|user)\s*>|\[/?INST\]|<\|im_(start|end)\|>|^\s*(system|assistant)\s*:)",
]
_INJECTION_RE = [re.compile(p, re.I | re.M) for p in _INJECTION]

# --------------------------------------------------------------- PII

_PII = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    # казахстанские мобильные: +7 707 123 45 67, 8(701)1234567
    "телефон": r"(?<!\d)(\+7|8)[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)",
    # номер карты: 16 цифр группами. Раньше ИИН, иначе ИИН съест часть карты.
    "карта": r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
    # ИИН — ровно 12 цифр. id объявлений krisha — 8–10 цифр, их не задевает.
    "ИИН": r"(?<!\d)\d{12}(?!\d)",
}
_PII_RE = {k: re.compile(p) for k, p in _PII.items()}

# --------------------------------------------------------------- домен

_OTHER_CITIES = r"(астан|нур-султан|шымкент|караганд|актоб|тараз|павлодар|усть-каменогорск|семе[йя]|" \
                r"атырау|костана|кызылорд|уральск|петропавловск|актау|туркестан|талдыкорган|конаев|" \
                r"капчага|москв|бишкек|ташкент)"
_SALE = r"\b(купить|куплю|покупк\w*|продаж\w*|продаю|ипотек\w*|в\s+собственность)\b"


@dataclass
class InputCheck:
    text: str                                  # очищенный текст: можно отдавать в LLM
    blocked: bool = False                      # True — в LLM не отправлять
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)


def detect_injection(text: str) -> list[str]:
    return [m.group(0).strip() for r in _INJECTION_RE if (m := r.search(text or ""))]


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for kind, r in _PII_RE.items():
        text, n = r.subn(f"[{kind} скрыт]", text)
        found += [kind] * n
    return text, found


def domain_warnings(text: str) -> list[str]:
    q = (text or "").lower()
    out = []
    if m := re.search(_OTHER_CITIES, q):
        out.append(f"в базе только Алматы и Каскелен, а в запросе упомянут другой город ({m.group(0)}…)")
    if re.search(_SALE, q):
        out.append("в базе только помесячная аренда — продажу и ипотеку мы не оцениваем")
    return out


def check_input(text: str) -> InputCheck:
    text = (text or "")[:MAX_QUERY_CHARS]
    masked, pii = mask_pii(text)
    inj = detect_injection(masked)
    return InputCheck(text=masked, blocked=bool(inj),
                      reasons=[f"похоже на prompt injection: «{s}»" for s in inj],
                      warnings=domain_warnings(masked), pii=pii)


# --------------------------------------------------------------- выход

RENT_MIN, RENT_MAX = 10_000, 50_000_000     # правдоподобные суммы аренды в тенге
TOLERANCE = 0.02                            # «около 260 тысяч» вместо 262 439 — округление, не выдумка
NBSP = " "
FORBIDDEN = [r"переплачива", r"переплат[аиуы]"]


def _amounts(text: str) -> list[float]:
    # id объявлений — длинные числа в скобках. Если их не вырезать, они посчитаются
    # «придуманными суммами» (эта ошибка однажды уже испортила замер).
    body = re.sub(r"\[[\d,\s]+\]", " ", text)
    nums = [float(n.replace(" ", "").replace(NBSP, "").replace(" ", ""))
            for n in re.findall(r"\d[\d\s" + NBSP + " " + r"]{3,}", body)]
    return [n for n in nums if RENT_MIN <= n <= RENT_MAX]


def check_output(text: str, allowed_numbers, allowed_ids) -> dict:
    """Сверка ответа модели с фактами, которые ей передали."""
    allowed_numbers = [float(a) for a in allowed_numbers if a is not None]
    found = _amounts(text)
    invented = [n for n in found
                if not any(abs(n - a) <= max(TOLERANCE * a, 1000) for a in allowed_numbers)]
    ids = set(re.findall(r"\[(\d+)\]", text))
    bad_ids = sorted(ids - {str(i) for i in allowed_ids})
    phrases = [p for p in FORBIDDEN if re.search(p, text, re.I)]
    cities = sorted({m.group(0) for m in re.finditer(_OTHER_CITIES, text.lower())})
    return {"ok": not (invented or bad_ids or phrases or cities), "numbers": len(found),
            "invented_numbers": invented, "invented_ids": bad_ids, "forbidden": phrases,
            "other_cities": cities}


def guard_output(text: str, allowed_numbers, allowed_ids, fallback: str) -> tuple[str, dict]:
    """Ответ модели, если он прошёл проверку, иначе — шаблонный текст из тех же фактов."""
    report = check_output(text, allowed_numbers, allowed_ids)
    return (text if report["ok"] else fallback), report
