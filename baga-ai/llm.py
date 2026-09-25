"""Единый шлюз к LLM: Gemini 3.6 Flash (Vertex AI) → ALEM alemllm → шаблон без LLM.

Все генерации проекта (разбор запроса, объяснение цены, ответ по выдаче, судья в evals,
расшифровка голоса) идут через complete(). В одном месте:

ФОЛБЭК. Провайдеры перебираются по LLM_CHAIN. Упал основной (таймаут, 429, 5xx, пустой
  ответ) — отвечает следующий. Упали все — LLMUnavailable, и вызывающий код подставляет
  шаблонный ответ из фактов (explain.template_explanation, query_parser._rule_based):
  пользователь получает выдачу даже без единой работающей модели.
ПРЕДОХРАНИТЕЛЬ (circuit breaker). После 3 ошибок подряд провайдер выключается на 60 с:
  иначе каждый запрос ждал бы таймаут мёртвого сервиса, прежде чем уйти на фолбэк.
БЮДЖЕТ. Расход считается по токенам и прайсу. Превышен дневной LLM_DAILY_BUDGET_USD —
  платные провайдеры пропускаются, отвечает бесплатный ALEM («резко подорожало»).
КЭШ. Ответ при temperature=0 детерминирован, поэтому одинаковый запрос (тот же промпт и
  параметры) берётся из SQLite: «Почему цена?» по одному объявлению считается один раз.
  Смысловой кэш для перефразированных запросов — в semantic_cache.py.
ТРЕЙСИНГ. Каждая попытка — generation в Langfuse: модель, промпт, ответ, токены, цена,
  с какого провайдера случился фолбэк.

Почему свой шлюз, а не LiteLLM: провайдеров два, и у Gemini есть особенность, которую
обёртки не решают — thinking-токены съедают max_output_tokens (см. _GeminiProvider).
"""
import hashlib
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

import config as C

# USD за 1M токенов (вход, выход). Gemini 3.6 Flash: вводная цена Vertex AI до 31.12.2026,
# с 01.01.2027 — 1.50 / 7.50. Thinking-токены тарифицируются как выходные.
# ALEM — без оплаты по токенам.
PRICES = {
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (0.50, 3.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "alemllm": (0.0, 0.0),
}
# Сколько токенов добавить к лимиту под рассуждения: max_tokens в проекте означает длину
# ОТВЕТА (как у ALEM), а у Gemini лимит общий на мысли и ответ.
THINKING_RESERVE = {"minimal": 0, "low": 1024, "medium": 4096, "high": 8192}
CACHE_TTL_S = 7 * 86400
# Таймаут по задаче. У Gemini бывают хвостовые задержки: на замере разбор занял 14.7 с,
# а один вызов через 26 с упал с 499 CANCELLED. Разбор короткий (p50 1.4 с) — ждать
# дольше 10 с незачем, быстрее уйти на ALEM (2.4 с). Ответы длиннее — 20 с.
TASK_TIMEOUT_S = {"parse_query": 10, "answer_search": 20, "explain_price": 20, "judge": 15,
                  "stt": 30, "rerank": 30, "visual_judge": 60}
BREAKER_FAILS, BREAKER_COOLDOWN_S = 3, 60.0


class LLMUnavailable(RuntimeError):
    """Ни один провайдер не ответил — вызывающий код переходит на шаблон."""


class ProviderError(RuntimeError):
    def __init__(self, msg, transient=True):
        super().__init__(msg)
        self.transient = transient


@dataclass
class LLMResult:
    text: str
    finish_reason: str          # stop | length | safety | other
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    fallback_from: list[str] = field(default_factory=list)
    trace_id: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def cost(model: str, prompt_tokens: int, out_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * pin + out_tokens * pout) / 1e6


# --------------------------------------------------------------- провайдеры

class _GeminiProvider:
    name = "gemini"
    paid = True

    def __init__(self):
        self.model = C.GEMINI_MODEL
        self._client = None

    def available(self) -> bool:
        return bool(C.GEMINI_KEY)

    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(vertexai=True, api_key=C.GEMINI_KEY,
                                        http_options=types.HttpOptions(timeout=int(C.LLM_TIMEOUT_S * 1000)))
        return self._client

    def call(self, system, user, temperature, top_p, max_tokens, json_mode, thinking, timeout_s=None) -> LLMResult:
        from google.genai import errors, types
        level = thinking or C.GEMINI_THINKING
        cfg = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=int((timeout_s or C.LLM_TIMEOUT_S) * 1000)),
            system_instruction=system or None, temperature=temperature, top_p=top_p,
            max_output_tokens=max_tokens + THINKING_RESERVE.get(level, 0),
            thinking_config=types.ThinkingConfig(thinking_level=level),
            response_mime_type="application/json" if json_mode else None)
        contents = user if isinstance(user, (str, list)) else str(user)
        t0 = time.time()
        try:
            r = self.client().models.generate_content(model=self.model, contents=contents, config=cfg)
        except errors.APIError as e:
            code = getattr(e, "code", 0) or 0
            raise ProviderError(f"gemini {code}: {str(e)[:200]}", transient=code in (408, 429, 499) or code >= 500)
        except Exception as e:                       # таймаут, сеть
            # на таймаут не повторяем тот же провайдер — сразу фолбэк: повтор удвоил бы ожидание
            timeout = "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()
            raise ProviderError(f"gemini: {type(e).__name__}: {str(e)[:200]}", transient=not timeout)
        cand = (r.candidates or [None])[0]
        fin = str(getattr(cand, "finish_reason", "") or "").upper()
        text = (r.text or "").strip() if cand is not None else ""
        if not text:
            raise ProviderError(f"gemini: пустой ответ (finish={fin or 'нет кандидатов'})")
        u = r.usage_metadata
        pt, ct, tt = (u.prompt_token_count or 0), (u.candidates_token_count or 0), (u.thoughts_token_count or 0)
        return LLMResult(text=text, finish_reason=_finish(fin), provider=self.name, model=self.model,
                         prompt_tokens=pt, completion_tokens=ct, thinking_tokens=tt,
                         latency_s=round(time.time() - t0, 3), cost_usd=cost(self.model, pt, ct + tt))


class _AlemProvider:
    name = "alem"
    paid = False

    def __init__(self):
        self.model = C.CHAT_MODEL
        self._client = None

    def available(self) -> bool:
        return bool(C.CHAT_KEY and C.CHAT_URL)

    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=C.CHAT_KEY, base_url=C.CHAT_URL, timeout=C.LLM_TIMEOUT_S, max_retries=0)
        return self._client

    def call(self, system, user, temperature, top_p, max_tokens, json_mode, thinking, timeout_s=None) -> LLMResult:
        import openai
        if not isinstance(user, str):
            raise ProviderError("alem: только текст (аудио и картинки — не к этой модели)", transient=False)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        t0 = time.time()
        try:
            r = self.client().with_options(timeout=timeout_s or C.LLM_TIMEOUT_S).chat.completions.create(
                model=self.model, messages=msgs, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        except openai.APIStatusError as e:
            raise ProviderError(f"alem {e.status_code}: {str(e)[:200]}",
                                transient=e.status_code in (408, 429) or e.status_code >= 500)
        except openai.APITimeoutError as e:
            raise ProviderError(f"alem: таймаут {timeout_s or C.LLM_TIMEOUT_S} с", transient=False) from e
        except Exception as e:
            raise ProviderError(f"alem: {type(e).__name__}: {str(e)[:200]}")
        c = r.choices[0]
        text = (c.message.content or "").strip()
        if not text:
            raise ProviderError("alem: пустой ответ")
        pt = getattr(r.usage, "prompt_tokens", 0) or 0
        ct = getattr(r.usage, "completion_tokens", 0) or 0
        return LLMResult(text=text, finish_reason=_finish(c.finish_reason or ""), provider=self.name,
                         model=self.model, prompt_tokens=pt, completion_tokens=ct,
                         latency_s=round(time.time() - t0, 3), cost_usd=cost(self.model, pt, ct))


def _finish(raw: str) -> str:
    raw = raw.upper()
    if "MAX_TOKENS" in raw or raw == "LENGTH":
        return "length"
    if raw in ("STOP", "FINISHREASON.STOP") or raw.endswith(".STOP"):
        return "stop"
    if "SAFETY" in raw or "BLOCK" in raw or "CONTENT_FILTER" in raw:
        return "safety"
    return raw.lower() or "stop"


PROVIDERS = {"gemini": _GeminiProvider(), "alem": _AlemProvider()}


# --------------------------------------------------------------- предохранитель

class _Breaker:
    def __init__(self):
        self.fails, self.open_until, self.lock = {}, {}, threading.Lock()

    def is_open(self, name) -> bool:
        return time.time() < self.open_until.get(name, 0)

    def success(self, name):
        with self.lock:
            self.fails[name] = 0

    def failure(self, name):
        with self.lock:
            self.fails[name] = self.fails.get(name, 0) + 1
            if self.fails[name] >= BREAKER_FAILS:
                self.open_until[name] = time.time() + BREAKER_COOLDOWN_S
                self.fails[name] = 0
                print(f"[llm] {name}: {BREAKER_FAILS} ошибки подряд — выключен на {BREAKER_COOLDOWN_S:.0f} с",
                      file=sys.stderr)

    def reset(self):
        self.fails.clear()
        self.open_until.clear()


breaker = _Breaker()


def health() -> dict:
    """Состояние провайдеров — для админ-панели и /health."""
    import store
    return {"chain": C.LLM_CHAIN, "spent_today_usd": round(store.spent_today(), 4),
            "budget_usd": C.LLM_DAILY_BUDGET_USD,
            "providers": {n: {"model": p.model, "configured": p.available(), "breaker_open": breaker.is_open(n)}
                          for n, p in PROVIDERS.items()}}


# --------------------------------------------------------------- главный вызов

def _cache_key(system, user, params: dict, chain: list[str]) -> str:
    models = [f"{n}:{PROVIDERS[n].model}" for n in chain if n in PROVIDERS]
    raw = json.dumps({"s": system, "u": user, "p": params, "m": models}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def complete(system: str, user, *, task: str, temperature: float | None = None, top_p: float | None = None,
             max_tokens: int | None = None, json_mode: bool = False, thinking: str | None = None,
             providers: list[str] | None = None, use_cache: bool | None = None,
             user_id: str | None = None, timeout_s: float | None = None) -> LLMResult:
    """Один ответ модели с фолбэком. providers — принудительный список (для A/B),
    по умолчанию LLM_CHAIN. user — строка или список частей (текст + аудио) для Gemini."""
    import store
    import tracing

    temperature = C.CHAT_TEMPERATURE if temperature is None else temperature
    top_p = C.CHAT_TOP_P if top_p is None else top_p
    max_tokens = C.CHAT_MAX_TOKENS if max_tokens is None else max_tokens
    chain = providers or C.LLM_CHAIN
    timeout_s = timeout_s or TASK_TIMEOUT_S.get(task, C.LLM_TIMEOUT_S)
    params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "json_mode": json_mode,
              "thinking": thinking or C.GEMINI_THINKING}
    cacheable = (C.LLM_CACHE if use_cache is None else use_cache) and temperature == 0 and isinstance(user, str)

    key = _cache_key(system, user, params, chain) if cacheable else None
    if key and (hit := store.cache_get(key, CACHE_TTL_S)):
        res = LLMResult(**{**hit, "cached": True, "cost_usd": 0.0, "latency_s": 0.0, "fallback_from": []})
        store.log_call(task, res.provider, res.model, ok=True, cached=True, user_id=user_id)
        with tracing.span(f"{task}:cache_hit", kind="CHAIN", provider=res.provider) as sp:
            sp.set(output=res.text)
        res.trace_id = tracing.current_trace_id()
        return res

    over_budget = store.spent_today() >= C.LLM_DAILY_BUDGET_USD
    skipped, errors = [], []
    for name in chain:
        p = PROVIDERS.get(name)
        why = ("неизвестный провайдер" if p is None else "нет ключа" if not p.available()
               else "предохранитель" if breaker.is_open(name)
               else "бюджет исчерпан" if (p.paid and over_budget) else None)
        if why:
            skipped.append(f"{name} ({why})")
            continue
        for attempt in (1, 2):
            with tracing.generation(task, model=p.model, input=_trace_input(system, user),
                                    model_parameters={k: params[k] for k in ("temperature", "top_p", "max_tokens")}
                                    | ({"thinking": params["thinking"]} if name == "gemini" else {}),
                                    metadata={"provider": name, "attempt": attempt,
                                              "fallback_from": ", ".join(skipped) or None}) as obs:
                t_call = time.time()
                try:
                    res = p.call(system, user, temperature, top_p, max_tokens, json_mode, thinking, timeout_s)
                except ProviderError as e:
                    obs.update(level="ERROR", status_message=str(e))
                    errors.append(str(e))
                    store.log_call(task, name, p.model, ok=False, error=str(e), user_id=user_id,
                                   latency_s=round(time.time() - t_call, 3))
                    if e.transient and attempt == 1:
                        time.sleep(0.5)
                        continue
                    breaker.failure(name)
                    break
                obs.update(output=res.text,
                           usage_details={"input": res.prompt_tokens, "output": res.completion_tokens,
                                          "reasoning": res.thinking_tokens},
                           cost_details={"total": res.cost_usd},
                           metadata={"provider": name, "finish_reason": res.finish_reason})
                res.trace_id = tracing.current_trace_id()
            breaker.success(name)
            res.fallback_from = skipped
            store.log_call(task, name, p.model, ok=True, prompt_tokens=res.prompt_tokens,
                           completion_tokens=res.completion_tokens, thinking_tokens=res.thinking_tokens,
                           cost_usd=res.cost_usd, latency_s=res.latency_s, fallback_from=res.fallback_from,
                           user_id=user_id)
            if key and res.finish_reason == "stop":        # обрезанный ответ не кэшируем
                store.cache_put(key, {k: v for k, v in res.as_dict().items()
                                      if k not in ("cached", "fallback_from", "trace_id")})
            return res
        skipped.append(f"{name} (ошибка)")
    raise LLMUnavailable("ни одна модель не ответила: " + "; ".join(skipped + errors))


def _trace_input(system, user):
    if isinstance(user, str):
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    parts = [u if isinstance(u, str) else f"<{getattr(getattr(u, 'inline_data', None), 'mime_type', 'binary')}>"
             for u in user]
    return [{"role": "system", "content": system}, {"role": "user", "content": parts}]


def complete_json(system: str, user: str, *, task: str, **kw) -> tuple[dict, LLMResult]:
    """Ответ в JSON. Gemini отдаёт чистый JSON (response_mime_type), ALEM иногда
    оборачивает в ```json или добавляет фразу — вырезаем объект регуляркой."""
    import re
    res = complete(system, user, task=task, json_mode=True, **kw)
    m = re.search(r"\{.*\}", res.text, re.S)
    if not m:
        raise ValueError(f"в ответе нет JSON: {res.text[:200]}")
    return json.loads(m.group(0)), res
