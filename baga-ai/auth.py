"""Пользователи, роли и лимиты.

Роли выбраны по деньгам, а не «для галочки»: каждое действие с LLM стоит токенов,
поэтому доступ к ним и частота ограничены ролью.

  guest — без регистрации: поиск по описанию и по фото, вердикт по цене. БЕЗ вызовов LLM:
          объяснение цены и ответ по выдаче собираются шаблоном из фактов. Дёшево и
          безопасно выставить в интернет.
  user  — всё, что у гостя, плюс ассистент (LangGraph), объяснения LLM, голос.
          Лимит LLM-действий в час.
  admin — плюс админ-панель: расход и бюджет, здоровье провайдеров, кэш, отзывы.

Пароли — PBKDF2-SHA256 (hashlib, 200 000 итераций, соль на пользователя): стандартная
библиотека, без сторонних пакетов. Хранилище — JSON (var/users.json, не в git).
Первый админ создаётся из BAGA_ADMIN_USER / BAGA_ADMIN_PASSWORD при старте.

CLI:  python auth.py add <логин> [--role user|admin]   (пароль спросит)
      python auth.py list
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from config import USERS_PATH

ROLES = ("guest", "user", "admin")
PERMISSIONS = {
    "guest": {"search", "photo_search"},
    "user": {"search", "photo_search", "assistant", "llm", "voice", "feedback"},
    "admin": {"search", "photo_search", "assistant", "llm", "voice", "feedback", "admin"},
}
# LLM-действий в час на пользователя (разбор + ответ ассистента, «Почему цена?», голос)
RATE_LIMITS = {"guest": 0, "user": int(os.getenv("RATE_LIMIT_USER", 60)), "admin": 10_000}
ITERATIONS = 200_000
_lock = threading.Lock()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITERATIONS).hex()


def _load() -> dict:
    if not USERS_PATH.exists():
        return {}
    return json.loads(USERS_PATH.read_text())


def _save(users: dict):
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = USERS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, ensure_ascii=False, indent=1))
    tmp.replace(USERS_PATH)
    os.chmod(USERS_PATH, 0o600)


def _valid_login(username: str) -> bool:
    return 3 <= len(username) <= 32 and all(c.isalnum() or c in "._-@" for c in username)


def add_user(username: str, password: str, role: str = "user", overwrite: bool = False) -> None:
    username = username.strip().lower()
    if role not in ROLES or role == "guest":
        raise ValueError(f"роль должна быть user или admin, а не {role!r}")
    if not _valid_login(username):
        raise ValueError("логин: 3–32 символа, буквы, цифры и . _ - @")
    if len(password) < 8:
        raise ValueError("пароль не короче 8 символов")
    with _lock:
        users = _load()
        if username in users and not overwrite:
            raise ValueError("такой пользователь уже есть")
        salt = secrets.token_hex(16)
        users[username] = {"salt": salt, "hash": _hash(password, salt), "role": role, "created": time.time()}
        _save(users)


def verify(username: str, password: str) -> str | None:
    """Роль при верном пароле, иначе None. Сравнение за постоянное время."""
    u = _load().get(username.strip().lower())
    if not u:
        _hash(password, "timing-equalizer")     # одинаковое время ответа для несуществующих логинов
        return None
    return u["role"] if hmac.compare_digest(u["hash"], _hash(password, u["salt"])) else None


def list_users() -> list[dict]:
    return [{"username": k, "role": v["role"], "created": time.strftime("%Y-%m-%d", time.localtime(v["created"]))}
            for k, v in _load().items()]


def can(role: str, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, set())


def bootstrap_admin():
    """Админ из переменных окружения — чтобы контейнер поднимался с рабочим входом."""
    user, pwd = os.getenv("BAGA_ADMIN_USER"), os.getenv("BAGA_ADMIN_PASSWORD")
    if user and pwd and user.lower() not in _load():
        add_user(user, pwd, "admin")


# --------------------------------------------------------------- лимит частоты

class RateLimiter:
    """Скользящее окно в памяти процесса: N действий за window_s секунд."""

    def __init__(self, window_s: float = 3600):
        self.window_s, self.hits, self.lock = window_s, defaultdict(deque), threading.Lock()

    def allow(self, key: str, role: str) -> bool:
        limit = RATE_LIMITS.get(role, 0)
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def remaining(self, key: str, role: str) -> int:
        now = time.time()
        q = [t for t in self.hits.get(key, ()) if now - t <= self.window_s]
        return max(0, RATE_LIMITS.get(role, 0) - len(q))


limiter = RateLimiter()


if __name__ == "__main__":
    import argparse
    import getpass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--role", default="user", choices=["user", "admin"])
    a.add_argument("--password", help="лучше не передавать в командной строке — спросит сам")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "add":
        add_user(args.username, args.password or getpass.getpass("пароль: "), args.role, overwrite=True)
        print(f"ok: {args.username} ({args.role})")
    else:
        for u in list_users():
            print(u)
