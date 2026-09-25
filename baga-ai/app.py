"""Baga AI — поиск квартир по стилю ремонта с проверкой цены.

Запуск:  streamlit run app.py

Поиск идёт по фотографиям (SigLIP 2) и описаниям (ALEM text-1024), жёсткие условия
фильтруются внутри Qdrant, цену проверяет CatBoost, дубли объявлений размечены
DINOv3 (notebooks/kaggle_dedup.ipynb). Картинки берутся с CDN krisha.

Роли (auth.py): гость — поиск и вердикт без LLM; пользователь — ассистент, объяснения
LLM, голос; админ — ещё и панель расходов, провайдеров, кэша и отзывов.
"""
import hashlib
import os
import secrets
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
import ui

st.set_page_config(page_title="Baga AI — аренда", page_icon="◆", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "1") == "1"
S = st.session_state


# --------------------------------------------------------------- вход

@st.cache_resource
def _bootstrap():
    auth.bootstrap_admin()
    return True


_bootstrap()
S.setdefault("sid", secrets.token_hex(6))


def login_panel():
    with st.sidebar:
        if S.get("user"):
            st.markdown(f"**{S['user']}** · {S['role']}")
            if st.button("Выйти", width="stretch"):
                for k in ("user", "role"):
                    S.pop(k, None)
                st.rerun()
            left = auth.limiter.remaining(f"web:{S['user']}", S["role"])
            st.caption(f"LLM-запросов осталось в этом часе: {left}")
            return
        tabs = st.tabs(["Вход", "Регистрация"] if ALLOW_SIGNUP else ["Вход"])
        with tabs[0].form("login"):
            u = st.text_input("Логин")
            p = st.text_input("Пароль", type="password")
            if st.form_submit_button("Войти", width="stretch"):
                if role := auth.verify(u, p):
                    S["user"], S["role"] = u.strip().lower(), role
                    st.rerun()
                st.error("Неверный логин или пароль")
        if ALLOW_SIGNUP:
            with tabs[1].form("signup"):
                u = st.text_input("Логин", key="su_u")
                p = st.text_input("Пароль (от 8 символов)", type="password", key="su_p")
                if st.form_submit_button("Создать аккаунт", width="stretch"):
                    try:
                        auth.add_user(u, p, "user")
                        S["user"], S["role"] = u.strip().lower(), "user"
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        st.caption("Без входа — гостевой режим: поиск и вердикт по цене, объяснения без LLM.")


login_panel()
ROLE = S.get("role", "guest")
UID = S.get("user") or f"guest-{S['sid']}"


def llm_allowed() -> bool:
    """Право на LLM по роли и лимит частоты. Гость и исчерпанный лимит получают шаблоны."""
    return auth.can(ROLE, "llm") and auth.limiter.allow(f"web:{UID}", ROLE)


# --------------------------------------------------------------- данные

@st.cache_resource(show_spinner="Загружаю индексы и модели…")
def boot():
    import tracing
    if tracing.enabled():
        try:
            tracing.setup()      # Phoenix: http://localhost:6006
        except Exception as e:   # без Phoenix сайт работает, просто без трейсов
            tracing._broken = True
            print(f"[tracing] Phoenix недоступен: {e}")
    import api
    from search import get_search
    return api, get_search()


@st.cache_data
def photo_urls() -> dict:
    p = Path("artifacts/photo_urls.parquet")
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df.path, df.url))


@st.cache_data
def listing_photos() -> dict:
    """Фото по id объявления: нужны, когда объявление нашлось только по описанию."""
    p = Path("artifacts/photo_urls.parquet")
    if not p.exists():
        return {}
    df = pd.read_parquet(p).sort_values(["listing_id", "photo_n"])
    return df.groupby("listing_id").url.first().to_dict()


@st.cache_data
def trust() -> dict:
    import dedup
    return dedup.load_trust()


@st.cache_data
def options():
    from listings import load_listings
    df = load_listings()
    return (sorted(df.city.dropna().unique()),
            sorted(df.district.dropna().unique()),
            sorted(int(r) for r in df.rooms.unique()),
            int(df.price.min()), int(df.price.quantile(0.99)))


def hero_photo(res: dict, urls: dict) -> str | None:
    for p in res.get("photos", [])[:1]:
        if u := urls.get(p["path"]):
            return u
    return listing_photos().get(res["listing_id"])


def explain_for(listing_id: str) -> str:
    from explain import explain_by_id, template_explanation
    if llm_allowed():
        return explain_by_id(listing_id)
    est = api.estimate_price(listing_id=listing_id)
    note = ("" if auth.can(ROLE, "llm")
            else "\n\n_Подробное объяснение — после входа._")
    return template_explanation(api.get_listing(listing_id), est) + note


def card(res: dict, urls: dict):
    st.markdown(ui.card_html(res, hero_photo(res, urls), trust().get(str(res["listing_id"]), 0)),
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if url := res["listing"].get("url"):
        b1.link_button("Открыть", url, width="stretch")
    # Streamlit перезапускает скрипт на каждое нажатие, поэтому ответ LLM кладём
    # в состояние сессии — иначе он исчезнет при следующем клике.
    wkey = f"why_{res['listing_id']}"
    if b2.button("Почему цена?", key=f"btn_{res['listing_id']}", width="stretch"):
        with st.spinner("Сравниваю с похожими…"):
            try:
                S[wkey] = explain_for(res["listing_id"])
            except Exception as e:
                S[wkey] = f"не удалось получить объяснение: {e}"
    if S.get(wkey):
        st.info(S[wkey])


def grid(results: list, urls: dict, per_row: int = 3):
    for i in range(0, len(results), per_row):
        row = results[i:i + per_row]
        for col, res in zip(st.columns(per_row, gap="medium"), row):
            with col:
                card(res, urls)


def feedback_widget(kind: str, query: str, answer: str, trace_id: str | None, key: str):
    """👍/👎 под ответом: в SQLite (админка) и оценкой на трейс в Langfuse."""
    if not auth.can(ROLE, "feedback"):
        return
    val = st.feedback("thumbs", key=f"fb_{key}")
    sent = f"fb_sent_{key}"
    if val is not None and not S.get(sent):
        import store
        import tracing
        store.add_feedback(UID, "web", kind, int(val), query_text=query, answer=answer, trace_id=trace_id)
        tracing.score(trace_id, "user_feedback", int(val), data_type="BOOLEAN")
        tracing.flush()
        S[sent] = True
        st.toast("Спасибо! Отзыв записан.")


# --------------------------------------------------------------- админка

def admin_page():
    import llm
    import store
    from semantic_cache import SemanticCache
    from query_parser import PROMPT_VERSION
    h = llm.health()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Расход LLM сегодня", f"${h['spent_today_usd']:.4f}", f"бюджет ${h['budget_usd']:.2f}",
              delta_color="off")
    calls = store.query("SELECT COUNT(*) AS n, SUM(cached) AS c, SUM(ok = 0) AS e FROM llm_calls "
                        "WHERE day = ?", (store.today(),))[0]
    c2.metric("Вызовов сегодня", calls["n"] or 0, f"из кэша {calls['c'] or 0}", delta_color="off")
    c3.metric("Ошибок провайдеров", calls["e"] or 0)
    sc = SemanticCache(f"parse_query:{PROMPT_VERSION}").stats()
    c4.metric("Смысловой кэш", f"{sc['entries']} запросов", f"попаданий {sc['hits']}", delta_color="off")

    st.markdown("#### Провайдеры (цепочка фолбэка)")
    st.dataframe(pd.DataFrame([{"провайдер": n, **v} for n, v in h["providers"].items()]),
                 hide_index=True, width="stretch")
    if url := os.getenv("LANGFUSE_BASE_URL"):
        st.link_button("Открыть трейсы в Langfuse", url)
    st.markdown("#### Расход по дням и задачам")
    st.dataframe(pd.DataFrame(store.usage_summary(7)), hide_index=True, width="stretch")
    st.markdown("#### Отзывы пользователей")
    fs = store.feedback_summary()
    if fs:
        st.dataframe(pd.DataFrame(fs), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(store.recent_feedback(50)), hide_index=True, width="stretch")
    else:
        st.caption("отзывов пока нет")
    st.markdown("#### Пользователи")
    st.dataframe(pd.DataFrame(auth.list_users()), hide_index=True, width="stretch")


# --------------------------------------------------------------- страница

api, engine = boot()
urls = photo_urls()
cities, districts, rooms_all, price_min, price_max = options()

st.markdown(ui.header(), unsafe_allow_html=True)

MODES = (["Ассистент"] if auth.can(ROLE, "assistant") else []) + ["По описанию", "По фото"] \
    + (["Админка"] if auth.can(ROLE, "admin") else [])
mode = st.segmented_control("Режим", MODES, default=MODES[0], label_visibility="collapsed")
mode = mode or MODES[0]

if mode == "Админка":
    admin_page()
    st.stop()

query, ups = "", []
if mode == "По фото":
    ups = st.file_uploader("Фото интерьера — можно несколько: кухню, детскую, санузел",
                           type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
else:
    query = st.text_input(
        "Запрос", key=f"q_{mode}", label_visibility="collapsed",
        placeholder=("двушка в Алматы до 400 тысяч, светлая кухня, не первый этаж"
                     if mode == "Ассистент" else "светлая кухня в скандинавском стиле, белые фасады"))

spoken = None
if mode == "Ассистент" and auth.can(ROLE, "voice"):
    audio = st.audio_input("Или скажите голосом", key="voice_in")
    if audio is not None:
        digest = hashlib.sha1(audio.getvalue()).hexdigest()
        if S.get("voice_digest") != digest:          # новая запись, а не перерисовка страницы
            S["voice_digest"] = digest
            import voice
            with st.spinner("Распознаю речь…"):
                try:
                    spoken = voice.transcribe(audio.getvalue(), audio.type or "audio/wav", user_id=UID)
                    S["voice_answer"] = True
                except voice.VoiceUnavailable as e:
                    st.warning(str(e))
            if spoken:
                st.caption(f"🎙 {spoken}")

left, right = st.columns([1, 5])
with left.popover("Фильтры", width="stretch"):
    city = st.selectbox("Город", ["любой", *cities])
    dsel = st.multiselect("Районы", districts)
    rsel = st.multiselect("Комнат", rooms_all)
    pmin, pmax = st.slider("Цена, ₸/мес", price_min, price_max,
                           (price_min, min(500_000, price_max)), step=10_000)
    amin, amax = st.slider("Площадь, м²", 10, 300, (10, 300), step=5)
    photo_rooms = st.multiselect("Искать по фото комнаты", list(ui.ROOM_RU)[:6],
                                 format_func=lambda r: ui.ROOM_RU[r])
    k = st.slider("Сколько показать", 3, 24, 9, step=3)
go = right.button("Найти", type="primary")

manual = [c for c in [
    None if city == "любой" else city,
    ", ".join(dsel) if dsel else None,
    ", ".join(f"{r} комн" for r in rsel) if rsel else None,
    f"{pmin // 1000}–{pmax // 1000} тыс ₸" if (pmin, pmax) != (price_min, min(500_000, price_max)) else None,
    f"{amin}–{amax} м²" if (amin, amax) != (10, 300) else None,
    ", ".join(ui.ROOM_RU[r] for r in photo_rooms) if photo_rooms else None,
] if c]
# Чипы рисуются здесь, но условия агента известны только после его запуска ниже,
# поэтому место резервируем, а заполняем в самом конце.
chips_slot = st.empty()

if ups:
    for col, f in zip(st.columns(min(len(ups), 6)), ups):
        col.image(f, width="stretch")
    if len(ups) > 1:
        st.caption("Каждое фото ищется отдельно со своим типом комнаты. Выше встанут "
                   "объявления, закрывшие больше ваших фото.")


def run_agent(q: str, resume=None):
    """Граф останавливается на подтверждении: сохраняем состояние и ждём кнопку."""
    from agent import invoke
    state, trace_id = invoke(q or None, resume=resume, thread_id=S.get("agent_thread", "web"),
                             user_id=UID, channel="web")
    paused = "__interrupt__" in state
    S["agent_pause"] = state["__interrupt__"][0].value if paused else None
    S["agent_state"] = None if paused else state
    S["agent_trace"] = trace_id or S.get("agent_trace")


agent_query = spoken or (query if go and mode == "Ассистент" else None)
if agent_query:
    if not auth.limiter.allow(f"web:{UID}", ROLE):
        st.warning("Лимит запросов к ассистенту на этот час исчерпан — пока доступен поиск по описанию.")
    else:
        S["agent_thread"] = f"web-{S['sid']}-{abs(hash(agent_query)) % 10**8}"
        S["agent_query"] = agent_query
        S["voice_answer"] = bool(spoken)
        with st.spinner("Агент работает…"):
            run_agent(agent_query)
elif go and (query or ups):
    image_paths = []
    for f in ups or []:
        tmp = Path(tempfile.gettempdir()) / f"baga_{f.name}"
        tmp.write_bytes(f.getvalue())
        image_paths.append(str(tmp))
    with st.spinner("Ищу…"):
        import tracing
        with tracing.trace_context(user_id=UID, session_id=S["sid"], tags=["web", "search"], name="search"):
            S["res"] = api.search_listings(
                query=query or None, images=image_paths or None,
                city=None if city == "любой" else city,
                districts=dsel or None, rooms=rsel or None,
                price_min=pmin, price_max=pmax, area_min=amin, area_max=amax,
                photo_rooms=photo_rooms or None, k=k, with_answer=bool(query) and llm_allowed())
        tracing.flush()

if pause := S.get("agent_pause"):
    st.warning(pause["question"])
    for a in pause["assumptions"]:
        st.markdown(f"- {a}")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Да, искать с ними", width="stretch"):
        with st.spinner("Ищу…"):
            run_agent("", resume=True)
        st.rerun()
    if c2.button("Нет, убрать их", width="stretch"):
        with st.spinner("Ищу…"):
            run_agent("", resume=False)
        st.rerun()

if (astate := S.get("agent_state")) and mode == "Ассистент":
    answer = astate.get("answer", "")
    st.markdown(f'<div class="summary">{answer}</div>', unsafe_allow_html=True)
    fcol, vcol, _ = st.columns([1, 1, 4])
    with fcol:
        feedback_widget("assistant", S.get("agent_query", ""), answer, S.get("agent_trace"),
                        key=S.get("agent_thread", "web"))
    if auth.can(ROLE, "voice"):
        akey = f"tts_{S.get('agent_thread')}"
        if vcol.button("🔊 Озвучить", width="stretch") or (S.pop("voice_answer", False)
                                                                    and akey not in S):
            import voice
            with st.spinner("Озвучиваю…"):
                S[akey] = voice.synthesize(answer)
        if S.get(akey):
            st.audio(S[akey], format="audio/wav", autoplay=True)
    with st.expander("Ход работы агента"):
        for line in astate.get("log", []):
            st.markdown(f"- {line}")
    grid(astate.get("results", []), urls)

if (res := S.get("res")) and mode not in ("Ассистент",):
    line = (f'Подошло по условиям: <b>{res["n_candidates"]}</b> объявлений · '
            f'ранжировано по: {", ".join(res["channels_used"]) or "цене за м²"}')
    qr = res.get("query_room")
    if isinstance(qr, list) and qr:
        line += " · на ваших фото: " + ", ".join(
            f"#{q['n'] + 1} {ui.ROOM_RU.get(q.get('room'), q.get('room'))}" for q in qr)
    elif isinstance(qr, dict) and qr.get("room"):
        line += f" · на фото: {ui.ROOM_RU.get(qr['room'], qr['room'])} ({qr['conf']:.2f})"
    st.markdown(f'<div class="summary">{line}</div>', unsafe_allow_html=True)
    if res.get("answer"):
        st.info(res["answer"])          # ответ собран по найденному: шаг генерации в RAG
    if not res["results"]:
        st.warning("По таким условиям ничего не нашлось — попробуйте расширить цену или район.")
    grid(res["results"], urls)

shown = manual
if mode == "Ассистент" and (a := S.get("agent_state")):
    shown = ui.filter_labels(a.get("filters", {})) or manual
chips_slot.markdown(ui.chips(shown), unsafe_allow_html=True)

st.caption(f"поиск: {engine.backend} · каналы: {', '.join(engine.channels) or 'только фильтры'} · "
           f"дублей размечено: {len(trust())} · роль: {ROLE}")
