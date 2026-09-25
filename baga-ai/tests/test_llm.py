"""Шлюз llm.py без сети: провайдеры подменены фейками. Проверяем то, что ломается молча —
порядок фолбэка, предохранитель, бюджет, кэш и то, что обрезанный ответ не кэшируется."""
import pytest

import config as C
import llm
import store


class Fake:
    def __init__(self, name, paid=False, fail=0, transient=True, text="ok", finish="stop", cost=0.0):
        self.name, self.paid, self.model = name, paid, f"{name}-model"
        self.fail, self.transient, self.text, self.finish, self.cost_usd, self.calls = fail, transient, text, finish, cost, 0

    def available(self):
        return True

    def call(self, system, user, temperature, top_p, max_tokens, json_mode, thinking, timeout_s=None):
        self.calls += 1
        if self.calls <= self.fail:
            raise llm.ProviderError(f"{self.name} 503", transient=self.transient)
        return llm.LLMResult(text=self.text, finish_reason=self.finish, provider=self.name, model=self.model,
                             prompt_tokens=10, completion_tokens=5, cost_usd=self.cost_usd)


@pytest.fixture
def providers(monkeypatch):
    llm.breaker.reset()
    store.execute("DELETE FROM llm_cache")
    store.execute("DELETE FROM llm_calls")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def install(**fakes):
        monkeypatch.setattr(llm, "PROVIDERS", fakes)
        monkeypatch.setattr(C, "LLM_CHAIN", list(fakes))
        return fakes
    return install


def test_primary_answers(providers):
    p = providers(a=Fake("a"), b=Fake("b"))
    r = llm.complete("s", "u", task="t", use_cache=False)
    assert r.provider == "a" and p["b"].calls == 0 and r.fallback_from == []


def test_transient_error_retried_once_then_fallback(providers):
    p = providers(a=Fake("a", fail=5), b=Fake("b", text="from b"))
    r = llm.complete("s", "u", task="t", use_cache=False)
    assert r.provider == "b" and r.text == "from b"
    assert p["a"].calls == 2                        # одна повторная попытка на временную ошибку
    assert r.fallback_from == ["a (ошибка)"]


def test_permanent_error_not_retried(providers):
    p = providers(a=Fake("a", fail=5, transient=False), b=Fake("b"))
    llm.complete("s", "u", task="t", use_cache=False)
    assert p["a"].calls == 1


def test_all_down_raises_for_template_fallback(providers):
    providers(a=Fake("a", fail=9), b=Fake("b", fail=9))
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("s", "u", task="t", use_cache=False)


def test_breaker_opens_after_repeated_failures(providers):
    p = providers(a=Fake("a", fail=100, transient=False), b=Fake("b"))
    for _ in range(llm.BREAKER_FAILS):
        llm.complete("s", "u", task="t", use_cache=False)
    assert llm.breaker.is_open("a")
    before = p["a"].calls
    r = llm.complete("s", "u", task="t", use_cache=False)
    assert p["a"].calls == before                   # мёртвый провайдер больше не ждём
    assert "a (предохранитель)" in r.fallback_from


def test_budget_skips_paid_provider(providers, monkeypatch):
    p = providers(paid=Fake("paid", paid=True), free=Fake("free"))
    monkeypatch.setattr(C, "LLM_DAILY_BUDGET_USD", 0.001)
    store.log_call("t", "paid", "paid-model", ok=True, cost_usd=0.01)
    r = llm.complete("s", "u", task="t", use_cache=False)
    assert r.provider == "free" and p["paid"].calls == 0
    assert "paid (бюджет исчерпан)" in r.fallback_from


def test_cache_hit_on_same_prompt(providers):
    p = providers(a=Fake("a"))
    llm.complete("s", "u", task="t", temperature=0)
    r = llm.complete("s", "u", task="t", temperature=0)
    assert r.cached and p["a"].calls == 1 and r.cost_usd == 0


def test_no_cache_when_sampling(providers):
    p = providers(a=Fake("a"))
    llm.complete("s", "u", task="t", temperature=0.7)
    llm.complete("s", "u", task="t", temperature=0.7)
    assert p["a"].calls == 2                        # при сэмплировании ответ не детерминирован


def test_truncated_answer_not_cached(providers):
    p = providers(a=Fake("a", finish="length"))
    llm.complete("s", "u", task="t", temperature=0)
    llm.complete("s", "u", task="t", temperature=0)
    assert p["a"].calls == 2


def test_cost_and_finish_mapping():
    assert llm.cost("gemini-3.6-flash", 1_000_000, 0) == pytest.approx(0.75)
    assert llm.cost("gemini-3.6-flash", 0, 1_000_000) == pytest.approx(3.75)
    assert llm.cost("alemllm", 10**6, 10**6) == 0
    assert llm._finish("FinishReason.MAX_TOKENS") == "length"
    assert llm._finish("length") == "length"
    assert llm._finish("FinishReason.STOP") == "stop"


def test_explain_falls_back_to_template_when_llm_down(monkeypatch):
    import explain

    def down(*a, **k):
        raise llm.LLMUnavailable("нет")
    monkeypatch.setattr(llm, "complete", down)
    listing = {"price": 500000, "area": 60, "rooms": 2, "district": "медеуский р-н"}
    est = {"p10": 300000, "p50": 380000, "p90": 450000, "verdict": "выше диапазона похожих",
           "reliable": True, "comparables": [{"listing_id": "111111", "price": 390000, "area": 58,
                                               "rooms": 2, "district": "медеуский р-н"}]}
    text = explain.explain_price(listing, est)
    assert "300 000" in text and "[111111]" in text
