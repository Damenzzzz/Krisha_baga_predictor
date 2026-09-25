"""API сайта без моделей поиска: вход, роли, проверка цены, отзывы, админка.
Поиск и ассистент здесь не вызываются — им нужны SigLIP и Qdrant (гигабайты)."""
import os

os.environ["WARMUP"] = "0"

import pytest
from fastapi.testclient import TestClient

import auth
import server


@pytest.fixture(scope="module")
def admin_ready():
    auth.add_user("siteadmin", "admin-pass-1", "admin", overwrite=True)


def client():
    return TestClient(server.app)


def test_pages_and_static():
    c = client()
    for path in ("/", "/app", "/login", "/admin", "/static/css/base.css", "/static/js/app.js", "/healthz"):
        assert c.get(path).status_code == 200, path


def test_guest_permissions():
    c = client()
    me = c.get("/api/me").json()
    assert me["role"] == "guest" and "assistant" not in me["permissions"]
    r = c.post("/api/assistant", json={"query": "двушка"})
    assert r.status_code == 403 and "Войдите" in r.json()["error"]
    assert c.get("/api/admin/summary").status_code == 403


def test_register_login_logout():
    c = client()
    r = c.post("/api/auth/register", json={"username": "webuser", "password": "web-pass-1"})
    assert r.status_code == 200 and r.json()["role"] == "user"
    assert c.get("/api/me").json()["user"] == "webuser"
    c.post("/api/auth/logout")
    assert c.get("/api/me").json()["role"] == "guest"
    assert c.post("/api/auth/login", json={"username": "webuser", "password": "wrong-pass"}).status_code == 401
    assert c.post("/api/auth/login", json={"username": "webuser", "password": "web-pass-1"}).status_code == 200


def test_tampered_cookie_is_guest():
    c = client()
    c.cookies.set("baga", "forged.value.here")
    assert c.get("/api/me").json()["role"] == "guest"


def test_price_by_link_and_errors():
    c = client()
    r = c.post("/api/price", json={"link": "https://krisha.kz/a/show/10635793"})
    assert r.status_code == 200
    body = r.json()
    assert body["band"]["p10"] < body["band"]["p50"] < body["band"]["p90"]
    assert body["llm"] is False and body["text"]            # гость — шаблон без LLM
    assert c.post("/api/price", json={"link": "https://krisha.kz/a/show/999999999"}).status_code == 404
    assert c.post("/api/price", json={"link": "не ссылка"}).status_code == 400
    r = c.post("/api/price", json={"area": 60, "rooms": 2, "city": "алматы", "price": 550000})
    assert r.status_code == 200 and r.json()["source"] == "модель"


def test_feedback_requires_login_and_is_stored():
    c = client()
    assert c.post("/api/feedback", json={"kind": "assistant", "value": 1}).status_code == 403
    c.post("/api/auth/register", json={"username": "fbuser", "password": "fb-pass-12"})
    assert c.post("/api/feedback", json={"kind": "assistant", "value": 0, "query": "q"}).json()["ok"]
    assert c.post("/api/feedback", json={"kind": "hack", "value": 1}).status_code == 422


def test_admin_summary(admin_ready):
    c = client()
    c.post("/api/auth/login", json={"username": "siteadmin", "password": "admin-pass-1"})
    d = c.get("/api/admin/summary").json()
    assert {"health", "today", "usage", "feedback", "users"} <= set(d)
    assert d["health"]["chain"]
