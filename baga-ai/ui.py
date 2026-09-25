"""Оформление сайта: стили и разметка карточки.

Вынесено из app.py, потому что карточка рисуется одним куском HTML, а не
виджетами Streamlit: только так бейдж вердикта ложится поверх фотографии.
"""
import html
from pathlib import Path

ROOM_RU = {"kitchen": "кухня", "living_room": "гостиная", "bedroom": "спальня",
           "bathroom": "санузел", "hallway": "коридор", "balcony": "балкон",
           "exterior": "фасад", "entrance": "подъезд", "window_view": "вид из окна",
           "floor_plan": "планировка", "other": "другое", "unknown": "—"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', -apple-system, sans-serif; }
/* иконки Streamlit — лигатуры шрифта Material Symbols: без этого вместо стрелки видно «expand_more» */
[data-testid="stIconMaterial"], [class*="material-symbols"], span[translate="no"] {
  font-family: 'Material Symbols Rounded' !important; }
#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

.brand { display:flex; align-items:center; gap:10px; margin-bottom:.2rem; }
.brand .mark { width:30px; height:30px; border-radius:9px; background:#E8F0FE; color:#2563EB;
    display:flex; align-items:center; justify-content:center; font-size:16px; }
.brand .name { font-size:19px; font-weight:600; letter-spacing:-.01em; }
.brand .sub { font-size:13px; color:#6B7280; }

.chips { display:flex; flex-wrap:wrap; gap:6px; margin:.1rem 0 1rem; }
.chip { font-size:12px; padding:4px 11px; border-radius:99px; border:1px solid #E5E7EB;
    color:#4B5563; background:#fff; white-space:nowrap; }
.chip.none { border-style:dashed; color:#9CA3AF; }

.summary { font-size:13px; color:#4B5563; padding:10px 0 14px; border-bottom:1px solid #EEF0F2;
    margin-bottom:1.1rem; }

.bcard { border:1px solid #E7E9EC; border-radius:14px; overflow:hidden; background:#fff;
    margin-bottom:8px; }
.bphoto { height:172px; background-size:cover; background-position:center; background-color:#F1F2F4;
    position:relative; }
.bphoto .ph { display:flex; height:100%; align-items:center; justify-content:center;
    color:#9CA3AF; font-size:12px; }
.badge { position:absolute; top:9px; font-size:11px; font-weight:500; padding:4px 9px;
    border-radius:99px; backdrop-filter:saturate(1.4); }
.badge.v { left:9px; }
.badge.d { right:9px; background:#FEF3C7; color:#92400E; }
.v-low  { background:#DCFCE7; color:#15803D; }
.v-high { background:#FEE2E2; color:#B91C1C; }
.v-mid  { background:#FFFFFFE6; color:#4B5563; border:1px solid #E5E7EB; }
.v-none { background:#FFFFFFE6; color:#9CA3AF; border:1px solid #E5E7EB; }

.bbody { padding:11px 13px 13px; }
.bprice { font-size:18px; font-weight:600; letter-spacing:-.02em; color:#111827; }
.bprice span { font-size:12px; font-weight:400; color:#9CA3AF; }
.bmeta { font-size:12.5px; color:#4B5563; margin:3px 0 7px; }
.brange { font-size:11.5px; color:#6B7280; line-height:1.55; }
.bwhy { font-size:11.5px; color:#9CA3AF; line-height:1.55; margin-top:3px; }
.bcover { font-size:11.5px; color:#15803D; line-height:1.55; margin-top:3px; }

div[data-testid="stHorizontalBlock"] div.stButton > button,
div[data-testid="stHorizontalBlock"] a[data-testid="stBaseLinkButton-secondary"] {
    font-size:12px; padding:2px 8px; min-height:32px; border-radius:8px; }
</style>
"""


def _esc(v) -> str:
    return html.escape(str(v))


def _money(v) -> str:
    return f"{int(v):,}".replace(",", " ")


def verdict_badge(pc: dict) -> str:
    """Главное сообщение карточки: насколько цена отличается от похожих."""
    if not pc or "p10" not in pc:
        return '<span class="badge v v-none">цена не проверена</span>'
    d = pc.get("diff_pct")
    if d is None:
        return '<span class="badge v v-none">цена не указана</span>'
    if "выше" in pc["verdict"]:
        cls, txt = "v-high", f"выше рынка на {abs(d):.0f}%"
    elif "ниже" in pc["verdict"]:
        cls, txt = "v-low", f"ниже рынка на {abs(d):.0f}%"
    else:
        cls, txt = "v-mid", "в рынке"
    if not pc.get("reliable", True):
        txt += " ·  мало похожих"
    return f'<span class="badge v {cls}">{_esc(txt)}</span>'


def card_html(res: dict, photo: str | None, dupes: int) -> str:
    lst, pc = res["listing"], res.get("price_check") or {}

    if photo and str(photo).startswith("http"):
        hero = f'<div class="bphoto" style="background-image:url(\'{_esc(photo)}\')">'
    else:
        hero = '<div class="bphoto"><div class="ph">фото недоступно</div>'

    dupe = f'<span class="badge d">ещё {dupes} с этими фото</span>' if dupes else ""

    price = lst.get("price")
    head = f'{_money(price)} ₸<span>/мес</span>' if price else "цена не указана"

    area = lst.get("area")
    meta = " · ".join(filter(None, [
        f"{lst.get('rooms')} комн" if lst.get("rooms") else None,
        f"{area:.0f} м²" if area else None,
        lst.get("district") or lst.get("city"),
    ]))

    rng = (f'Похожие сдают за {_money(pc["p10"])} – {_money(pc["p90"])} ₸'
           if pc.get("p10") else "")

    why = []
    if any(ch.startswith("photo") for ch in res.get("rank_by", {})):
        rooms = {ROOM_RU.get(p["room_type"], p["room_type"]) for p in res.get("photos", [])}
        why.append("фото: " + ", ".join(sorted(rooms)) if rooms else "фото")
    if "desc" in res.get("rank_by", {}):
        why.append("описание")

    cover = (f'<div class="bcover">закрыто ваших фото: {_esc(res["coverage"])}</div>'
             if res.get("coverage") else "")

    body = (f'<div class="bbody"><div class="bprice">{head}</div>'
            f'<div class="bmeta">{_esc(meta)}</div>'
            f'<div class="brange">{_esc(rng)}</div>{cover}'
            + (f'<div class="bwhy">совпало: {_esc("; ".join(why))}</div>' if why else "")
            + '</div>')

    return f'<div class="bcard">{hero}{verdict_badge(pc)}{dupe}</div>{body}</div>'


def header() -> str:
    return ('<div class="brand"><div class="mark">◆</div>'
            '<div class="name">Baga AI</div>'
            '<div class="sub">аренда по стилю ремонта · проверка цены</div></div>')


def chips(items: list[str]) -> str:
    if not items:
        return '<div class="chips"><span class="chip none">условия не заданы</span></div>'
    return '<div class="chips">' + "".join(f'<span class="chip">{_esc(i)}</span>' for i in items) + "</div>"


def _thou(v) -> str:
    return f"{int(v) // 1000} тыс" if v >= 1000 else str(int(v))


def filter_labels(f: dict) -> list[str]:
    """Условия агента человеческим языком — те же чипы, что и у ручных фильтров."""
    out = []
    if f.get("city"):
        out.append(str(f["city"]).capitalize())
    if f.get("districts"):
        out.append(", ".join(d.capitalize() for d in f["districts"]))
    if f.get("rooms"):
        out.append(", ".join(f"{r} комн" for r in f["rooms"]))
    lo, hi = f.get("price_min"), f.get("price_max")
    if lo and hi:
        out.append(f"{_thou(lo)}–{_thou(hi)} ₸")
    elif hi:
        out.append(f"до {_thou(hi)} ₸")
    elif lo:
        out.append(f"от {_thou(lo)} ₸")
    if f.get("area_min") and f.get("area_max"):
        out.append(f'{f["area_min"]:.0f}–{f["area_max"]:.0f} м²')
    elif f.get("area_min"):
        out.append(f'от {f["area_min"]:.0f} м²')
    elif f.get("area_max"):
        out.append(f'до {f["area_max"]:.0f} м²')
    if f.get("year_min"):
        out.append(f'от {int(f["year_min"])} года')
    if f.get("building_types"):
        out.append(", ".join(f["building_types"]))
    if f.get("not_first_floor"):
        out.append("не первый этаж")
    if f.get("not_last_floor"):
        out.append("не последний этаж")
    return out
