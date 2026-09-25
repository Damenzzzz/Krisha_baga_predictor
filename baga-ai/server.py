"""Веб-сервер Baga AI: JSON API + статический фронтенд (web/).

    uvicorn server:app --port 8501        ->  http://localhost:8501

Страницы:  /  (главная)   /app  (поиск, ассистент, по фото, проверка цены)
           /login          /admin (панель админа)
API — /api/*. Та же логика, что у Telegram-бота и MCP-сервера: agent.invoke, api.*,
explain, voice. Роли и лимиты — auth.py, трейсы — tracing.py.

Почему FastAPI + свой фронтенд, а не Streamlit: главной странице нужны анимации и
вёрстка, которых Streamlit не даёт, а сценарию «ассистент остановился и спрашивает»
нужен настоящий запрос-ответ, а не перезапуск всего скрипта на каждый клик.
Эндпоинты синхронные: FastAPI выполняет их в пуле потоков, модели и индексы общие.
"""
import os
import secrets
from contextlib import asynccontextmanager
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel, Field

import auth
import config as C

WEB = Path(__file__).resolve().parent / "web"
SESSION_DAYS = 30

@asynccontextmanager
async def lifespan(_app):
    """Первый админ из окружения и прогрев индексов в фоне: первый посетитель не ждёт
    загрузку SigLIP и Qdrant (~20 с)."""
    auth.bootstrap_admin()
    if os.getenv("WARMUP", "1") == "1":
        import threading

        def warm():
            try:
                from search import get_search
                get_search()
                _demo()
            except Exception as e:
                print(f"[server] прогрев не удался: {e}")
        threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="Baga AI", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)


# --------------------------------------------------------------- сессии

def _secret() -> str:
    """Ключ подписи cookie: из BAGA_SECRET или сгенерированный и сохранённый в var/."""
    if s := os.getenv("BAGA_SECRET"):
        return s
    p = C.VAR_DIR / "secret.key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
        os.chmod(p, 0o600)
    return p.read_text().strip()


_signer = URLSafeTimedSerializer(_secret(), salt="baga-session")


class Viewer(BaseModel):
    user: Optional[str] = None
    role: str = "guest"
    sid: str

    @property
    def uid(self) -> str:
        return self.user or f"guest-{self.sid}"


def viewer(request: Request) -> Viewer:
    raw = request.cookies.get("baga")
    if raw:
        try:
            d = _signer.loads(raw, max_age=SESSION_DAYS * 86400)
            return Viewer(**d)
        except (BadSignature, TypeError, ValueError):
            pass
    return Viewer(sid=request.cookies.get("baga_sid") or secrets.token_hex(6))


def _set_session(resp: Response, v: Viewer):
    resp.set_cookie("baga", _signer.dumps(v.model_dump()), max_age=SESSION_DAYS * 86400,
                    httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE", "0") == "1")


@app.middleware("http")
async def guest_sid(request: Request, call_next):
    """У гостя тоже есть стабильный id — для лимитов, трейсов и отзывов."""
    resp = await call_next(request)
    if not request.cookies.get("baga_sid") and not request.cookies.get("baga"):
        resp.set_cookie("baga_sid", secrets.token_hex(6), max_age=SESSION_DAYS * 86400, httponly=True,
                        samesite="lax")
    return resp


def require(permission: str):
    def dep(v: Viewer = Depends(viewer)) -> Viewer:
        if not auth.can(v.role, permission):
            raise HTTPException(403, "Войдите, чтобы пользоваться этой функцией" if v.role == "guest"
                                else "Недостаточно прав")
        return v
    return dep


def llm_allowed(v: Viewer) -> bool:
    return auth.can(v.role, "llm") and auth.limiter.allow(f"web:{v.uid}", v.role)


# --------------------------------------------------------------- данные для карточек

@lru_cache(maxsize=1)
def _photo_maps():
    import pandas as pd
    p = C.ART_DIR / "photo_urls.parquet"
    if not p.exists():
        return {}, {}
    df = pd.read_parquet(p)
    first = df.sort_values(["listing_id", "photo_n"]).groupby("listing_id").url.first().to_dict()
    return dict(zip(df.path, df.url)), first


@lru_cache(maxsize=1)
def _trust() -> dict:
    try:
        import dedup
        return dedup.load_trust()
    except Exception:
        return {}


def _num(x):
    try:
        import math
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


ROOM_RU = {"kitchen": "кухня", "living_room": "гостиная", "bedroom": "спальня", "bathroom": "санузел",
           "hallway": "прихожая", "balcony": "балкон", "exterior": "дом снаружи", "other": "другое"}


def card(r: dict) -> dict:
    """Результат поиска -> то, что рисует фронтенд. Без путей к файлам и сырых скоров."""
    by_path, first = _photo_maps()
    l, pc = r.get("listing", {}), r.get("price_check") or {}
    photos = r.get("photos", [])
    photo = next((by_path[p["path"]] for p in photos if p.get("path") in by_path), None) \
        or first.get(str(r["listing_id"]))
    return {
        "id": str(r["listing_id"]), "url": l.get("url"), "photo": photo,
        "price": _num(l.get("price")), "rooms": l.get("rooms"), "area": _num(l.get("area")),
        "floor": _num(l.get("floor")), "floors_total": _num(l.get("floors_total")),
        "city": l.get("city"), "district": l.get("district"),
        "matched": sorted({ROOM_RU.get(p.get("room_type"), p.get("room_type")) for p in photos
                           if p.get("room_type")}),
        "by_description": "desc" in (r.get("rank_by") or {}),
        "coverage": r.get("coverage"),                     # «2 из 3» ваших фото закрыто
        "band": {k: _num(pc.get(k)) for k in ("p10", "p50", "p90")} if pc.get("p10") else None,
        "verdict": pc.get("verdict"), "diff_pct": _num(pc.get("diff_pct")), "reliable": pc.get("reliable"),
        "duplicates": int(_trust().get(str(r["listing_id"]), 0) or 0),
    }


# --------------------------------------------------------------- страницы

def _page(name: str):
    return FileResponse(WEB / name, headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
def page_home():
    return _page("index.html")


@app.get("/app", include_in_schema=False)
def page_app():
    return _page("app.html")


@app.get("/login", include_in_schema=False)
def page_login():
    return _page("login.html")


@app.get("/admin", include_in_schema=False)
def page_admin():
    return _page("admin.html")


@app.get("/healthz", include_in_schema=False)
@app.get("/_stcore/health", include_in_schema=False)       # совместимость со старым healthcheck
def health():
    return {"ok": True}


# --------------------------------------------------------------- авторизация

class Credentials(BaseModel):
    username: str
    password: str


@app.get("/api/me")
def me(v: Viewer = Depends(viewer)):
    return {"user": v.user, "role": v.role, "permissions": sorted(auth.PERMISSIONS.get(v.role, ())),
            "llm_left": auth.limiter.remaining(f"web:{v.uid}", v.role),
            "signup": os.getenv("ALLOW_SIGNUP", "1") == "1"}


@app.post("/api/auth/login")
def login(c: Credentials, response: Response, v: Viewer = Depends(viewer)):
    role = auth.verify(c.username, c.password)
    if not role:
        raise HTTPException(401, "Неверный логин или пароль")
    _set_session(response, Viewer(user=c.username.strip().lower(), role=role, sid=v.sid))
    return {"user": c.username.strip().lower(), "role": role}


@app.post("/api/auth/register")
def register(c: Credentials, response: Response, v: Viewer = Depends(viewer)):
    if os.getenv("ALLOW_SIGNUP", "1") != "1":
        raise HTTPException(403, "Регистрация закрыта")
    try:
        auth.add_user(c.username, c.password, "user")
    except ValueError as e:
        raise HTTPException(400, str(e))
    _set_session(response, Viewer(user=c.username.strip().lower(), role="user", sid=v.sid))
    return {"user": c.username.strip().lower(), "role": "user"}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie("baga")
    return {"ok": True}


# --------------------------------------------------------------- справочники и демо

@app.get("/api/meta")
@lru_cache(maxsize=1)
def meta():
    from listings import load_listings
    df = load_listings()
    return {"cities": sorted(df.city.dropna().unique().tolist()),
            "districts": sorted(d for d in df.district.dropna().unique().tolist() if d != "не указан"),
            "rooms": sorted(int(r) for r in df.rooms.dropna().unique()),
            "price_min": int(df.price.min()), "price_max": int(df.price.quantile(0.99)),
            "listings": int(len(df)), "photos": 72963}


DEMO_QUERY = "светлая кухня, белые фасады"


@app.get("/api/demo")
def demo():
    """Живое демо для главной: настоящие объявления из базы по стилевому запросу. Без LLM."""
    return _demo()


@lru_cache(maxsize=1)
def _demo():
    import api
    res = api.search_listings(query=DEMO_QUERY, city="алматы", rooms=[1, 2], price_max=450000,
                              photo_rooms=["kitchen"], k=6)
    return {"query": "двушка в Алматы до 450 тысяч, " + DEMO_QUERY,
            "results": [card(r) for r in res["results"]]}


# --------------------------------------------------------------- ассистент (LangGraph)

class AssistantIn(BaseModel):
    query: str = Field(min_length=2, max_length=500)


class ResumeIn(BaseModel):
    thread_id: str
    confirmed: bool


def _agent_out(state: dict, trace_id, thread_id: str) -> dict:
    if "__interrupt__" in state:
        p = state["__interrupt__"][0].value
        return {"status": "paused", "thread_id": thread_id, "question": p["question"],
                "assumptions": p["assumptions"], "trace_id": trace_id}
    return {"status": "done", "thread_id": thread_id, "answer": state.get("answer", ""),
            "filters": state.get("filters", {}), "style": state.get("style"),
            "log": state.get("log", []), "results": [card(r) for r in state.get("results") or []],
            "trace_id": trace_id}


@app.post("/api/assistant")
def assistant(body: AssistantIn, v: Viewer = Depends(require("assistant"))):
    import agent
    if not auth.limiter.allow(f"web:{v.uid}", v.role):
        raise HTTPException(429, "Лимит запросов к ассистенту на этот час исчерпан. Поиск по описанию работает.")
    thread = f"web-{v.sid}-{uuid.uuid4().hex[:8]}"
    state, trace_id = agent.invoke(body.query, thread_id=thread, user_id=v.uid, channel="web")
    return _agent_out(state, trace_id, thread)


@app.post("/api/assistant/resume")
def assistant_resume(body: ResumeIn, v: Viewer = Depends(require("assistant"))):
    import agent
    if not body.thread_id.startswith(f"web-{v.sid}-"):
        raise HTTPException(403, "Чужой диалог")
    state, trace_id = agent.invoke(resume=body.confirmed, thread_id=body.thread_id, user_id=v.uid, channel="web")
    return _agent_out(state, trace_id, body.thread_id)


# --------------------------------------------------------------- поиск

class SearchIn(BaseModel):
    query: Optional[str] = Field(None, max_length=500)
    city: Optional[str] = None
    districts: Optional[list[str]] = None
    rooms: Optional[list[int]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    area_min: Optional[float] = None
    area_max: Optional[float] = None
    photo_rooms: Optional[list[str]] = None
    k: int = Field(12, ge=3, le=30)


def _search(v: Viewer, images=None, **kw):
    import api
    import tracing
    with_answer = bool(kw.get("query")) and llm_allowed(v)
    with tracing.trace_context(user_id=v.uid, session_id=v.sid, tags=["web", "search"], name="search"):
        res = api.search_listings(images=images, with_answer=with_answer, **kw)
        trace_id = tracing.current_trace_id()
    tracing.flush()
    qr = res.get("query_room")
    return {"n_candidates": res["n_candidates"], "channels": res["channels_used"], "answer": res.get("answer"),
            "query_rooms": [ROOM_RU.get(q.get("room"), q.get("room")) for q in qr] if isinstance(qr, list)
            else ([ROOM_RU.get(qr.get("room"), qr.get("room"))] if isinstance(qr, dict) and qr.get("room") else []),
            "results": [card(r) for r in res["results"]], "trace_id": trace_id, "llm": with_answer}


@app.post("/api/search")
def search(body: SearchIn, v: Viewer = Depends(require("search"))):
    from guardrails import check_input
    warnings = []
    if body.query:
        g = check_input(body.query)
        body.query, warnings = g.text, g.warnings
    out = _search(v, **body.model_dump())
    out["warnings"] = warnings
    return out


@app.post("/api/search/photo")
def search_photo(images: list[UploadFile] = File(...), city: Optional[str] = Form(None),
                 price_max: Optional[float] = Form(None), rooms: Optional[str] = Form(None),
                 v: Viewer = Depends(require("photo_search"))):
    if len(images) > 6:
        raise HTTPException(400, "Не больше 6 фото за раз")
    paths = []
    for f in images:
        data = f.file.read()
        if len(data) > 12 * 1024 * 1024:
            raise HTTPException(400, f"{f.filename}: файл больше 12 МБ")
        p = Path(tempfile.gettempdir()) / f"baga_{uuid.uuid4().hex}{Path(f.filename or '.jpg').suffix[:5]}"
        p.write_bytes(data)
        paths.append(str(p))
    try:
        return _search(v, images=paths, city=city or None, price_max=price_max,
                       rooms=[int(x) for x in rooms.split(",") if x.strip()] if rooms else None, k=12)
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)


# --------------------------------------------------------------- цена

@app.get("/api/listing/{listing_id}/explain")
def explain_listing(listing_id: str, v: Viewer = Depends(viewer)):
    import api
    from explain import explain_price, template_explanation
    try:
        card_, est = api.get_listing(listing_id), api.estimate_price(listing_id=listing_id)
    except KeyError:
        raise HTTPException(404, "Объявления нет в базе")
    llm_used = llm_allowed(v)
    text = explain_price(card_, est) if llm_used else template_explanation(card_, est)
    return {"text": text, "llm": llm_used, "band": {k: est.get(k) for k in ("p10", "p50", "p90")},
            "verdict": est.get("verdict"), "reliable": est.get("reliable"),
            "comparables": [{"id": str(c["listing_id"]), "price": c["price"], "area": c.get("area"),
                             "rooms": c.get("rooms"), "district": c.get("district")}
                            for c in est.get("comparables", [])[:5]]}


class PriceIn(BaseModel):
    link: Optional[str] = None
    area: Optional[float] = None
    rooms: Optional[int] = None
    city: Optional[str] = None
    district: Optional[str] = None
    floor: Optional[float] = None
    floors_total: Optional[float] = None
    price: Optional[float] = None


@app.post("/api/price")
def price_check(body: PriceIn, v: Viewer = Depends(viewer)):
    import re

    import api
    from explain import explain_price, template_explanation
    if body.link:
        m = re.search(r"/a/show/(\d+)", body.link) or re.fullmatch(r"\s*(\d{6,})\s*", body.link)
        if not m:
            raise HTTPException(400, "Нужна ссылка вида krisha.kz/a/show/<номер> или номер объявления")
        try:
            listing, est = api.get_listing(m.group(1)), api.estimate_price(listing_id=m.group(1))
        except KeyError:
            raise HTTPException(404, "Этого объявления нет в базе (срез 21.09.2026). Введите характеристики вручную.")
    else:
        if not (body.area and body.rooms and body.city):
            raise HTTPException(400, "Укажите площадь, число комнат и город")
        listing = {"area": body.area, "rooms": body.rooms, "city": body.city.lower(),
                   "district": (body.district or "").lower(), "floor": body.floor,
                   "floors_total": body.floors_total, "price": body.price, "address": "", "description": ""}
        est = api.estimate_price(listing=listing)
    llm_used = llm_allowed(v)
    text = explain_price(listing, est) if llm_used else template_explanation(listing, est)
    return {"listing": {k: listing.get(k) for k in ("listing_id", "price", "area", "rooms", "district", "city", "url")},
            "band": {k: est.get(k) for k in ("p10", "p50", "p90")}, "verdict": est.get("verdict"),
            "diff_pct": est.get("diff_pct"), "reliable": est.get("reliable"), "source": est.get("source"),
            "n_similar": est.get("n_similar"), "text": text, "llm": llm_used,
            "comparables": [{"id": str(c["listing_id"]), "price": c["price"], "area": c.get("area"),
                             "rooms": c.get("rooms"), "district": c.get("district")}
                            for c in est.get("comparables", [])[:5]]}


# --------------------------------------------------------------- голос

@app.post("/api/voice/transcribe")
def voice_transcribe(audio: UploadFile = File(...), v: Viewer = Depends(require("voice"))):
    import voice
    if not auth.limiter.allow(f"web:{v.uid}", v.role):
        raise HTTPException(429, "Лимит запросов на этот час исчерпан")
    data = audio.file.read()
    mime = (audio.content_type or "audio/webm").split(";")[0]
    try:
        return {"text": voice.transcribe(data, mime, user_id=v.uid)}
    except voice.VoiceUnavailable as e:
        raise HTTPException(503, str(e))


class SpeakIn(BaseModel):
    text: str = Field(max_length=4000)


@app.post("/api/voice/speak")
def voice_speak(body: SpeakIn, v: Viewer = Depends(require("voice"))):
    import voice
    wav = voice.synthesize(body.text)
    if not wav:
        raise HTTPException(503, "Озвучка сейчас недоступна")
    return Response(wav, media_type="audio/wav")


# --------------------------------------------------------------- отзывы

class FeedbackIn(BaseModel):
    kind: str = Field(pattern="^(assistant|search|price)$")
    value: int = Field(ge=0, le=1)
    trace_id: Optional[str] = None
    query: Optional[str] = Field(None, max_length=500)
    answer: Optional[str] = Field(None, max_length=4000)
    comment: Optional[str] = Field(None, max_length=1000)


@app.post("/api/feedback")
def feedback(body: FeedbackIn, v: Viewer = Depends(require("feedback"))):
    import store
    import tracing
    store.add_feedback(v.uid, "web", body.kind, body.value, comment=body.comment, query_text=body.query,
                       answer=body.answer, trace_id=body.trace_id)
    tracing.score(body.trace_id, "user_feedback", body.value, comment=body.comment, data_type="BOOLEAN")
    tracing.flush()
    return {"ok": True}


# --------------------------------------------------------------- админка

@app.get("/api/admin/summary")
def admin_summary(days: int = 7, v: Viewer = Depends(require("admin"))):
    import llm
    import store
    from query_parser import PROMPT_VERSION
    from semantic_cache import SemanticCache
    today = store.query("SELECT COUNT(*) AS calls, COALESCE(SUM(cached),0) AS cached, "
                        "COALESCE(SUM(ok = 0),0) AS errors, ROUND(COALESCE(SUM(cost_usd),0), 4) AS cost "
                        "FROM llm_calls WHERE day = ?", (store.today(),))[0]
    by_day = store.query("SELECT day, ROUND(SUM(cost_usd), 4) AS cost, COUNT(*) AS calls FROM llm_calls "
                         "WHERE ts > ? GROUP BY day ORDER BY day", (time.time() - days * 86400,))
    return {"health": llm.health(), "today": today, "by_day": by_day, "usage": store.usage_summary(days),
            "cache": SemanticCache(f"parse_query:{PROMPT_VERSION}").stats(),
            "feedback": store.feedback_summary(), "recent_feedback": store.recent_feedback(30),
            "users": auth.list_users(), "langfuse": os.getenv("LANGFUSE_BASE_URL")}


# --------------------------------------------------------------- статика и ошибки

@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
