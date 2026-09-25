"""Baga AI — поиск квартир по стилю ремонта с проверкой цены.

Запуск:  streamlit run app.py

Поиск идёт по фотографиям (SigLIP 2) и описаниям (ALEM text-1024), жёсткие условия
фильтруются внутри Qdrant, цену проверяет CatBoost, дубли объявлений размечены
DINOv3 (notebooks/kaggle_dedup.ipynb). Картинки берутся с CDN krisha.
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import ui

st.set_page_config(page_title="Baga AI — аренда", page_icon="◆", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

MODES = ["Ассистент", "По описанию", "По фото"]


@st.cache_resource(show_spinner="Загружаю индексы и модели…")
def boot():
    import tracing
    if tracing.enabled():
        tracing.setup()          # Phoenix: http://localhost:6006
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


def card(res: dict, urls: dict):
    st.markdown(ui.card_html(res, hero_photo(res, urls), trust().get(str(res["listing_id"]), 0)),
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if url := res["listing"].get("url"):
        b1.link_button("Открыть", url, use_container_width=True)
    # Streamlit перезапускает скрипт на каждое нажатие, поэтому ответ LLM кладём
    # в состояние сессии — иначе он исчезнет при следующем клике.
    wkey = f"why_{res['listing_id']}"
    if b2.button("Почему цена?", key=f"btn_{res['listing_id']}", use_container_width=True):
        with st.spinner("Сравниваю с похожими…"):
            from explain import explain_by_id
            try:
                st.session_state[wkey] = explain_by_id(res["listing_id"])
            except Exception as e:
                st.session_state[wkey] = f"не удалось получить объяснение: {e}"
    if st.session_state.get(wkey):
        st.info(st.session_state[wkey])


def grid(results: list, urls: dict, per_row: int = 3):
    for i in range(0, len(results), per_row):
        row = results[i:i + per_row]
        for col, res in zip(st.columns(per_row, gap="medium"), row):
            with col:
                card(res, urls)


api, engine = boot()
urls = photo_urls()
cities, districts, rooms_all, price_min, price_max = options()

st.markdown(ui.header(), unsafe_allow_html=True)

mode = st.segmented_control("Режим", MODES, default=MODES[0], label_visibility="collapsed")
mode = mode or MODES[0]

query, ups = "", []
if mode == "По фото":
    ups = st.file_uploader("Фото интерьера — можно несколько: кухню, детскую, санузел",
                           type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
else:
    query = st.text_input(
        "Запрос", key=f"q_{mode}", label_visibility="collapsed",
        placeholder=("двушка в Алматы до 400 тысяч, светлая кухня, не первый этаж"
                     if mode == "Ассистент" else "светлая кухня в скандинавском стиле, белые фасады"))

left, right = st.columns([1, 5])
with left.popover("Фильтры", use_container_width=True):
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
        col.image(f, use_container_width=True)
    if len(ups) > 1:
        st.caption("Каждое фото ищется отдельно со своим типом комнаты. Выше встанут "
                   "объявления, закрывшие больше ваших фото.")


def run_agent(q: str, resume=None):
    """Граф останавливается на подтверждении: сохраняем состояние и ждём кнопку."""
    from langgraph.types import Command
    from agent import graph
    cfg = {"configurable": {"thread_id": st.session_state.get("agent_thread", "web")}}
    state = graph().invoke(Command(resume=resume) if resume is not None else {"query": q}, cfg)
    paused = "__interrupt__" in state
    st.session_state["agent_pause"] = state["__interrupt__"][0].value if paused else None
    st.session_state["agent_state"] = None if paused else state


if go and mode == "Ассистент" and query:
    st.session_state["agent_thread"] = f"web-{abs(hash(query))}"
    with st.spinner("Агент работает…"):
        run_agent(query)
elif go and (query or ups):
    image_paths = []
    for f in ups or []:
        tmp = Path(tempfile.gettempdir()) / f"baga_{f.name}"
        tmp.write_bytes(f.getvalue())
        image_paths.append(str(tmp))
    with st.spinner("Ищу…"):
        st.session_state["res"] = api.search_listings(
            query=query or None, images=image_paths or None,
            city=None if city == "любой" else city,
            districts=dsel or None, rooms=rsel or None,
            price_min=pmin, price_max=pmax, area_min=amin, area_max=amax,
            photo_rooms=photo_rooms or None, k=k, with_answer=bool(query))

if pause := st.session_state.get("agent_pause"):
    st.warning(pause["question"])
    for a in pause["assumptions"]:
        st.markdown(f"- {a}")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Да, искать с ними", use_container_width=True):
        with st.spinner("Ищу…"):
            run_agent("", resume=True)
        st.rerun()
    if c2.button("Нет, убрать их", use_container_width=True):
        with st.spinner("Ищу…"):
            run_agent("", resume=False)
        st.rerun()

if astate := st.session_state.get("agent_state"):
    st.markdown(f'<div class="summary">{astate.get("answer", "")}</div>', unsafe_allow_html=True)
    with st.expander("Ход работы агента"):
        for line in astate.get("log", []):
            st.markdown(f"- {line}")
    grid(astate.get("results", []), urls)

if (res := st.session_state.get("res")) and mode != "Ассистент":
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
if mode == "Ассистент" and (a := st.session_state.get("agent_state")):
    shown = ui.filter_labels(a.get("filters", {})) or manual
chips_slot.markdown(ui.chips(shown), unsafe_allow_html=True)

st.caption(f"поиск: {engine.backend} · каналы: {', '.join(engine.channels) or 'только фильтры'} · "
           f"дублей размечено: {len(trust())}")
