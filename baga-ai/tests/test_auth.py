import pytest

import auth


def test_add_verify_roles():
    auth.add_user("alice", "correct horse", "user")
    assert auth.verify("alice", "correct horse") == "user"
    assert auth.verify("ALICE", "correct horse") == "user"          # логин без учёта регистра
    assert auth.verify("alice", "wrong password") is None
    assert auth.verify("nobody", "whatever1") is None


def test_password_not_stored_in_clear():
    auth.add_user("bob", "secret-pass-1", "admin")
    raw = auth.USERS_PATH.read_text()
    assert "secret-pass-1" not in raw and auth.verify("bob", "secret-pass-1") == "admin"


def test_validation():
    with pytest.raises(ValueError):
        auth.add_user("x", "longenough", "user")                   # короткий логин
    with pytest.raises(ValueError):
        auth.add_user("carol", "short", "user")                    # короткий пароль
    with pytest.raises(ValueError):
        auth.add_user("dave", "longenough", "guest")               # гость не регистрируется
    auth.add_user("erin", "longenough", "user")
    with pytest.raises(ValueError):
        auth.add_user("erin", "longenough", "user")                # дубликат


def test_permissions():
    assert not auth.can("guest", "llm") and auth.can("guest", "search")
    assert auth.can("user", "assistant") and not auth.can("user", "admin")
    assert auth.can("admin", "admin")


def test_rate_limit():
    lim = auth.RateLimiter()
    assert not lim.allow("g", "guest")                              # гостю LLM не положен
    n = auth.RATE_LIMITS["user"]
    assert all(lim.allow("u", "user") for _ in range(n))
    assert not lim.allow("u", "user")
    assert lim.remaining("u", "user") == 0
