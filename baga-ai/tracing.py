"""Трейсинг: Langfuse (основной) и Arize Phoenix (локальный, по желанию).

Langfuse — облачный дашборд с трейсами, стоимостью, оценками пользователей и сессиями.
Выбран основным, потому что:
  * вызовы Gemini идут не через клиент OpenAI, и автоинструментация их не видит —
    нужна ручная запись generation с моделью, токенами и ценой, а в Langfuse это штатно;
  * у LangGraph есть готовый CallbackHandler: узлы графа видны как вложенные спаны;
  * оценки 👍/👎 из интерфейса и Telegram пишутся как score на тот же трейс — видно,
    какие ответы людям не понравились, прямо рядом с промптом и ответом модели.

Phoenix остаётся для офлайн-работы без аккаунта (TRACING=1, дашборд на :6006).

Всё, что здесь есть, — наблюдение, а не функциональность: если Langfuse недоступен или
ключей нет (CI), функции молча ничего не делают, и поиск работает как обычно.
Вывод только в stderr: stdout MCP-сервера занят протоколом.
"""
import contextlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ключи Langfuse — в .env; config.py его тоже грузит, но тянет torch, а трейсинг
# импортируется и там, где config ещё не нужен
load_dotenv(Path(__file__).resolve().parent / ".env")

# типы observation в Langfuse совпадают по смыслу с openinference-видами Phoenix
_LF_TYPES = {"CHAIN": "chain", "RETRIEVER": "retriever", "TOOL": "tool", "AGENT": "agent",
             "GUARDRAIL": "guardrail", "LLM": "span", "EMBEDDING": "embedding", "EVALUATOR": "evaluator"}

_provider = None
_broken = False          # Phoenix не поднялся — работаем без него
_lf = None
_lf_broken = False


# --------------------------------------------------------------- Langfuse

def langfuse_enabled() -> bool:
    return (not _lf_broken and bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))
            and os.getenv("LANGFUSE_TRACING", "true").lower() != "false")


def langfuse():
    """Клиент Langfuse или None. Ключи и адрес берутся из окружения
    (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)."""
    global _lf, _lf_broken
    if not langfuse_enabled():
        return None
    if _lf is None:
        try:
            from langfuse import get_client
            _lf = get_client()
        except Exception as e:
            print(f"[tracing] Langfuse недоступен, работаю без него: {e}", file=sys.stderr)
            _lf_broken = True
            return None
    return _lf


@contextlib.contextmanager
def trace_context(user_id: str | None = None, session_id: str | None = None,
                  tags: list[str] | None = None, name: str | None = None, **metadata):
    """Привязывает все вложенные спаны к пользователю и сессии — в Langfuse по ним
    фильтруются трейсы и считается стоимость на пользователя."""
    lf = langfuse()
    if lf is None:
        yield
        return
    from langfuse import propagate_attributes
    meta = {k: str(v) for k, v in metadata.items() if v is not None}
    with propagate_attributes(user_id=user_id, session_id=session_id, tags=tags,
                              trace_name=name, metadata=meta or None):
        yield


class _NoopObs:
    id = None

    def update(self, **_):
        return self


@contextlib.contextmanager
def generation(name: str, model: str, input, model_parameters: dict | None = None, metadata: dict | None = None):
    """Один вызов LLM как generation: модель, промпт, ответ, токены, цена.
    Шлюз llm.py вызывает obs.update(output=..., usage_details=..., cost_details=...)."""
    lf = langfuse()
    if lf is None:
        yield _NoopObs()
        return
    params = {k: v for k, v in (model_parameters or {}).items() if v is not None}
    try:
        cm = lf.start_as_current_observation(name=name, as_type="generation", model=model, input=input,
                                             model_parameters=params, metadata=metadata)
        obs = cm.__enter__()
    except Exception as e:
        print(f"[tracing] generation не записан: {e}", file=sys.stderr)
        yield _NoopObs()
        return
    try:
        yield obs
    except BaseException as exc:
        obs.update(level="ERROR", status_message=str(exc)[:500])
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


def current_trace_id() -> str | None:
    lf = langfuse()
    if lf is None:
        return None
    try:
        return lf.get_current_trace_id()
    except Exception:
        return None


def score(trace_id: str | None, name: str, value, comment: str | None = None, data_type: str | None = None):
    """Оценка на трейс: 👍/👎 пользователя, срабатывание guardrail и т. п."""
    lf = langfuse()
    if lf is None or not trace_id:
        return
    try:
        kw = {"data_type": data_type} if data_type else {}
        lf.create_score(name=name, value=value, trace_id=trace_id, comment=comment, **kw)
    except Exception as e:
        print(f"[tracing] score не записан: {e}", file=sys.stderr)


def langchain_callbacks() -> list:
    """Колбэк для LangGraph: узлы parse → confirm → search → relax → answer становятся
    вложенными спанами того же трейса."""
    if langfuse() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []


def flush():
    if _lf is not None:
        try:
            _lf.flush()
        except Exception:
            pass


def trace_url(trace_id: str | None) -> str | None:
    lf = langfuse()
    if lf is None or not trace_id:
        return None
    try:
        return lf.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


# --------------------------------------------------------------- Phoenix (локально)

def enabled() -> bool:
    """Phoenix включён: TRACING=1 и пакет установлен."""
    return os.getenv("TRACING", "0") == "1" and not _broken


def setup(project_name="baga-ai", launch_ui=True):
    """Поднимает Phoenix и включает автотрассировку клиента OpenAI (вызовы ALEM).
    Повторные вызовы безопасны."""
    global _provider
    if _provider is not None:
        return _provider
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    if launch_ui and not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        import phoenix as px
        px.launch_app()

    _provider = register(project_name=project_name, auto_instrument=False)
    OpenAIInstrumentor().instrument(tracer_provider=_provider)
    return _provider


def _phoenix_tracer():
    global _broken
    if not enabled():
        return None
    try:
        return setup().get_tracer("baga-ai")
    except Exception as e:
        print(f"[tracing] Phoenix недоступен, работаю без него: {e}", file=sys.stderr)
        _broken = True
        return None


def _plain(v):
    return v if isinstance(v, (str, int, float, bool)) else str(v)


class span:
    """Спан своего шага (поиск, оценка цены, guardrail) в Langfuse и Phoenix сразу.

    Без трейсинга молча ничего не делает — спаны расставлены в коде один раз,
    без проверок вокруг каждого.
    """

    def __init__(self, name: str, kind: str = "CHAIN", **attrs):
        self.name, self.kind = name, kind
        self.attrs = {k: v for k, v in attrs.items() if v is not None}
        self._px_cm = self._px = self._lf_cm = self._lf = None

    def __enter__(self):
        lf = langfuse()
        if lf is not None:
            try:
                inp = self.attrs.get("input.value")
                meta = {k: _plain(v) for k, v in self.attrs.items() if k != "input.value"}
                self._lf_cm = lf.start_as_current_observation(
                    name=self.name, as_type=_LF_TYPES.get(self.kind, "span"), input=inp, metadata=meta or None)
                self._lf = self._lf_cm.__enter__()
            except Exception as e:
                print(f"[tracing] спан Langfuse не открыт: {e}", file=sys.stderr)
                self._lf_cm = None
        t = _phoenix_tracer()
        if t is not None:
            self._px_cm = t.start_as_current_span(self.name)
            self._px = self._px_cm.__enter__()
            self._px.set_attribute("openinference.span.kind", self.kind)
            for k, v in self.attrs.items():
                self._px.set_attribute(k, _plain(v))
        return self

    def set(self, output=None, **attrs):
        """output — что вернул шаг (видно в Langfuse), attrs — метаданные."""
        attrs = {k: v for k, v in attrs.items() if v is not None}
        if self._lf is not None:
            try:
                self._lf.update(output=output, metadata={k: _plain(v) for k, v in attrs.items()} or None)
            except Exception:
                pass
        if self._px is not None:
            if output is not None:
                self._px.set_attribute("output.value", _plain(output))
            for k, v in attrs.items():
                self._px.set_attribute(k, _plain(v))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lf_cm is not None:
            if exc is not None:
                try:
                    self._lf.update(level="ERROR", status_message=str(exc)[:500])
                except Exception:
                    pass
            self._lf_cm.__exit__(exc_type, exc, tb)
        if self._px_cm is not None:
            if exc is not None:
                self._px.record_exception(exc)
            self._px_cm.__exit__(exc_type, exc, tb)
        return False
