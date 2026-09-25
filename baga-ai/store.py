"""Рабочее состояние сервиса в одном SQLite: учёт вызовов LLM, кэш ответов, отзывы.

Почему SQLite, а не Redis/Postgres: один процесс Streamlit + бот + MCP на одной машине,
нагрузка — единицы запросов в секунду. SQLite в режиме WAL держит несколько процессов-
читателей и одного писателя без отдельного сервера, а файл переживает перезапуск.
При переезде на несколько реплик меняется только этот модуль.
"""
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone

from config import DB_PATH

_lock = threading.RLock()        # conn() вызывается и под замком из execute/query
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    ts REAL, day TEXT, task TEXT, provider TEXT, model TEXT, ok INTEGER, cached INTEGER,
    prompt_tokens INTEGER, completion_tokens INTEGER, thinking_tokens INTEGER,
    cost_usd REAL, latency_s REAL, fallback_from TEXT, error TEXT, user_id TEXT);
CREATE INDEX IF NOT EXISTS llm_calls_day ON llm_calls(day);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT, created REAL, hits INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY, namespace TEXT, signature TEXT, text TEXT, vec BLOB, value TEXT, created REAL,
    hits INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS semantic_cache_ns ON semantic_cache(namespace, signature);
CREATE TABLE IF NOT EXISTS feedback (
    ts REAL, user_id TEXT, channel TEXT, kind TEXT, value INTEGER, comment TEXT,
    query TEXT, answer TEXT, trace_id TEXT);
"""


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30, isolation_level=None)
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA busy_timeout=30000")
                c.executescript(SCHEMA)
                _conn = c
    return _conn


def execute(sql: str, args=()):
    with _lock:
        return conn().execute(sql, args)


def query(sql: str, args=()) -> list[sqlite3.Row]:
    with _lock:
        cur = conn().execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------- учёт LLM

def log_call(task: str, provider: str, model: str, ok: bool, cached: bool = False, prompt_tokens=0,
             completion_tokens=0, thinking_tokens=0, cost_usd=0.0, latency_s=0.0, fallback_from=(),
             error: str | None = None, user_id: str | None = None):
    execute("INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), today(), task, provider, model, int(ok), int(cached), prompt_tokens or 0,
             completion_tokens or 0, thinking_tokens or 0, cost_usd or 0.0, latency_s or 0.0,
             ",".join(fallback_from), (error or "")[:500] or None, user_id))


def spent_today() -> float:
    row = query("SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE day = ?", (today(),))
    return float(row[0]["s"])


def usage_summary(days: int = 7) -> list[dict]:
    return query("""SELECT day, provider, task, COUNT(*) AS calls, SUM(ok) AS ok, SUM(cached) AS cached,
                           SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens + thinking_tokens) AS out_tokens,
                           ROUND(SUM(cost_usd), 4) AS cost_usd, ROUND(AVG(latency_s), 2) AS avg_latency_s
                    FROM llm_calls WHERE ts > ? GROUP BY day, provider, task ORDER BY day DESC, calls DESC""",
                 (time.time() - days * 86400,))


# --------------------------------------------------------------- точный кэш

def cache_get(key: str, ttl_s: float):
    rows = query("SELECT value, created FROM llm_cache WHERE key = ?", (key,))
    if not rows or time.time() - rows[0]["created"] > ttl_s:
        return None
    execute("UPDATE llm_cache SET hits = hits + 1 WHERE key = ?", (key,))
    return json.loads(rows[0]["value"])


def cache_put(key: str, value: dict):
    execute("INSERT OR REPLACE INTO llm_cache(key, value, created, hits) VALUES (?,?,?,0)",
            (key, json.dumps(value, ensure_ascii=False), time.time()))


# --------------------------------------------------------------- отзывы

def add_feedback(user_id, channel, kind, value, comment=None, query_text=None, answer=None, trace_id=None):
    execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), user_id, channel, kind, int(value), comment, query_text, (answer or "")[:2000], trace_id))


def feedback_summary() -> list[dict]:
    return query("""SELECT channel, kind, COUNT(*) AS n, SUM(value > 0) AS up, SUM(value <= 0) AS down
                    FROM feedback GROUP BY channel, kind""")


def recent_feedback(n: int = 50) -> list[dict]:
    return query("SELECT datetime(ts, 'unixepoch') AS time, user_id, channel, kind, value, comment, query "
                 "FROM feedback ORDER BY ts DESC LIMIT ?", (n,))
