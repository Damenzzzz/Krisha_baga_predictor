"""Трейсинг всех вызовов LLM и эмбеддингов через Arize Phoenix.

Phoenix выбран вместо LangSmith по простой причине: он поднимается локально и не требует
ключей и аккаунта. Все наши обращения к ALEM (объяснения цены, эмбеддинги описаний, судья
в evals) идут через клиент OpenAI, который инструментируется автоматически — в трейсы
попадают промпт, ответ, модель, токены и время.

Свои шаги (поиск, оценка цены) добавляются вручную через span: иначе в дашборде будет
видно только «был вызов LLM», без контекста, ради чего он делался.

Включение: TRACING=1 в .env или переменной окружения. Дашборд: http://localhost:6006
"""
import os

_provider = None


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGSMITH_TRACING", "true") != "false"


def wrap_llm_client(client):
    """LangSmith видит вызовы LangChain/LangGraph сам, но наши обращения к ALEM идут
    через голый клиент OpenAI — его нужно обернуть явно, иначе в трейсе будет пусто."""
    if not langsmith_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai
        return wrap_openai(client)
    except Exception:
        return client


def enabled() -> bool:
    return os.getenv("TRACING", "0") == "1"


def setup(project_name="baga-ai", launch_ui=True):
    """Поднимает Phoenix и включает автотрассировку. Повторные вызовы безопасны."""
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


def tracer():
    return setup().get_tracer("baga-ai")


class span:
    """Контекст-менеджер, который молча ничего не делает, когда трейсинг выключен.

    Позволяет расставить спаны в коде один раз и не городить вокруг них проверки.
    """

    def __init__(self, name: str, kind: str = "CHAIN", **attrs):
        self.name, self.kind, self.attrs, self._cm, self._span = name, kind, attrs, None, None

    def __enter__(self):
        if not enabled():
            return self
        self._cm = tracer().start_as_current_span(self.name)
        self._span = self._cm.__enter__()
        self._span.set_attribute("openinference.span.kind", self.kind)
        for k, v in self.attrs.items():
            if v is not None:
                self._span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        return self

    def set(self, **attrs):
        if self._span is not None:
            for k, v in attrs.items():
                if v is not None:
                    self._span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._cm is not None:
            if exc is not None:
                self._span.record_exception(exc)
            self._cm.__exit__(exc_type, exc, tb)
        return False
