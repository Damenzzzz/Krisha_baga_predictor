"""MCP-сервер Baga AI: поиск аренды по стилю ремонта и проверка цены.

Инструменты — тонкая обёртка над api.py, той же функцией пользуются сайт и агент.
Поэтому ответ через MCP и на сайте один и тот же: разные только интерфейсы.

    search_apartments  — поиск по условиям и описанию ремонта, с вердиктом по цене
    check_rent_price   — диапазон похожих p10/p50/p90 и вердикт: по id, ссылке krisha.kz
                         или по характеристикам квартиры, которой нет в базе
    get_comparables    — аналоги, на которых построен вердикт (чтобы его можно было проверить)
    get_listing        — карточка объявления

Почему MCP, а не просто HTTP API. Потребитель здесь — LLM-клиент (Claude Desktop,
Claude Code, Cursor): ему нужны не эндпоинты, а описанные инструменты со схемой
параметров, которые модель сама выбирает по ходу разговора. MCP даёт это без
отдельного клиента под каждое приложение: один сервер подключается к любому.

Запуск (stdio):  python mcp_server.py
Подключение в Claude Code:
    claude mcp add baga-ai -- /путь/к/python /путь/к/baga-ai/mcp_server.py
Проверка без клиента:  python mcp_server.py --selftest
"""
import contextlib
import re
import sys
from typing import Annotated, Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

mcp = FastMCP(
    "baga-ai",
    instructions=(
        "Аренда квартир в Алматы и Каскелене (krisha.kz, 7 476 объявлений, срез 21.09.2026). "
        "Оценка цены построена на ценах ОБЪЯВЛЕНИЙ, а не сделок: говори «дороже похожих "
        "объявлений», а не «вы переплачиваете». Если reliable=false — предупреди, что аналогов мало."
    ),
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                            openWorldHint=False)


@contextlib.contextmanager
def _quiet_stdout():
    """В stdio-режиме stdout — это канал протокола. Любой print при загрузке моделей
    или индексов сломал бы клиенту разбор JSON-RPC, поэтому вывод уходит в stderr."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _api():
    with _quiet_stdout():
        import api
    return api


def _listing_id(value: str) -> str:
    """Принимает id или ссылку вида https://krisha.kz/a/show/1014955701."""
    m = re.search(r"/a/show/(\d+)", value or "")
    lid = m.group(1) if m else str(value).strip()
    if not lid.isdigit():
        raise ValueError(f"не похоже на id или ссылку krisha.kz: {value!r}")
    return lid


def _photo_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not hasattr(_photo_url, "map"):
        try:
            import pandas as pd
            from config import ART_DIR
            df = pd.read_parquet(ART_DIR / "photo_urls.parquet")
            _photo_url.map = dict(zip(df.path, df.url))
        except Exception:
            _photo_url.map = {}
    return _photo_url.map.get(path)


def _compact(r: dict) -> dict:
    """Результат поиска без служебных полей: LLM-клиенту не нужны пути к файлам и сырые скоры фото."""
    l, pc = r.get("listing", {}), r.get("price_check") or {}
    photos = r.get("photos", [])
    return {
        "listing_id": r["listing_id"], "url": l.get("url"),
        "price": l.get("price"), "rooms": l.get("rooms"), "area": l.get("area"),
        "floor": l.get("floor"), "city": l.get("city"), "district": l.get("district"),
        "relevance": r.get("score"),
        "matched_rooms": sorted({p.get("room_type") for p in photos if p.get("room_type")}),
        "photo_url": _photo_url(photos[0]["path"]) if photos else None,
        "price_verdict": pc.get("verdict"), "fair_range": [pc.get("p10"), pc.get("p90")] if pc.get("p10") else None,
        "reliable": pc.get("reliable"),
    }


@mcp.tool(annotations=READ_ONLY)
def search_apartments(
    style: Annotated[Optional[str], Field(description="Как должна выглядеть квартира: «светлая кухня в "
                                                      "скандинавском стиле», «свежий ремонт без мебели». "
                                                      "Без цен и районов — их передавай отдельными полями.")] = None,
    city: Annotated[Optional[str], Field(description="«алматы» или «каскелен»")] = None,
    districts: Annotated[Optional[list[str]], Field(description="Часть названия района: «бостандык», «медеу»")] = None,
    rooms: Annotated[Optional[list[int]], Field(description="Число комнат, например [2] или [2, 3]")] = None,
    price_min: Annotated[Optional[float], Field(description="Минимальная цена, ₸ в месяц")] = None,
    price_max: Annotated[Optional[float], Field(description="Максимальная цена, ₸ в месяц")] = None,
    area_min: Annotated[Optional[float], Field(description="Площадь от, м²")] = None,
    area_max: Annotated[Optional[float], Field(description="Площадь до, м²")] = None,
    limit: Annotated[int, Field(ge=1, le=20, description="Сколько объявлений вернуть")] = 5,
) -> dict[str, Any]:
    """Найти квартиры в аренду по условиям и по внешнему виду ремонта.

    Стиль ищется по фотографиям (SigLIP 2) и описаниям объявлений, условия — фильтрами.
    У каждого результата есть вердикт по цене относительно похожих объявлений.
    """
    from guardrails import check_input
    warnings = []
    if style:
        g = check_input(style)
        style, warnings = g.text, g.warnings + g.reasons
    with _quiet_stdout():
        res = _api().search_listings(query=style or None, city=city, districts=districts, rooms=rooms,
                                     price_min=price_min, price_max=price_max,
                                     area_min=area_min, area_max=area_max, k=limit)
    return {"n_candidates": res["n_candidates"], "channels_used": res["channels_used"],
            "warnings": warnings, "results": [_compact(r) for r in res["results"]]}


@mcp.tool(annotations=READ_ONLY)
def check_rent_price(
    listing: Annotated[Optional[str], Field(description="id объявления или ссылка krisha.kz/a/show/<id>")] = None,
    area: Annotated[Optional[float], Field(description="Для квартиры не из базы: площадь, м²")] = None,
    rooms: Annotated[Optional[int], Field(description="Для квартиры не из базы: число комнат")] = None,
    city: Annotated[Optional[str], Field(description="Для квартиры не из базы: «алматы» или «каскелен»")] = None,
    district: Annotated[Optional[str], Field(description="Для квартиры не из базы: район")] = None,
    floor: Annotated[Optional[float], Field(description="Этаж")] = None,
    floors_total: Annotated[Optional[float], Field(description="Этажей в доме")] = None,
    price: Annotated[Optional[float], Field(description="Цена, которую просят, ₸ в месяц")] = None,
) -> dict[str, Any]:
    """Проверить, в рынке ли цена аренды: диапазон похожих объявлений (p10/p50/p90) и вердикт.

    Либо listing (объявление из базы), либо характеристики квартиры: area, rooms, city
    обязательны. Для объявлений из базы оценка out-of-fold — модель это объявление не видела.
    """
    with _quiet_stdout():
        if listing:
            return _api().estimate_price(listing_id=_listing_id(listing))
        if not (area and rooms and city):
            raise ValueError("нужен listing или area + rooms + city")
        return _api().estimate_price(listing={
            "area": area, "rooms": rooms, "city": city.lower(), "district": (district or "").lower(),
            "floor": floor, "floors_total": floors_total, "price": price,
            "address": "", "description": ""})


@mcp.tool(annotations=READ_ONLY)
def get_comparables(
    listing: Annotated[str, Field(description="id объявления или ссылка krisha.kz")],
    k: Annotated[int, Field(ge=1, le=20, description="Сколько аналогов вернуть")] = 5,
) -> dict[str, Any]:
    """Похожие объявления, на которых построен вердикт по цене, и по какому правилу они отобраны
    (matched_by: район+комнаты+площадь → … → город+площадь)."""
    with _quiet_stdout():
        return _api().get_comparables(_listing_id(listing), k=k)


@mcp.tool(annotations=READ_ONLY)
def get_listing(
    listing: Annotated[str, Field(description="id объявления или ссылка krisha.kz")],
) -> dict[str, Any]:
    """Карточка объявления: цена, площадь, комнаты, этаж, адрес, описание, ссылка."""
    with _quiet_stdout():
        return _api().get_listing(_listing_id(listing))


async def _selftest():
    """Подключается к серверу как настоящий MCP-клиент (в памяти) и вызывает инструменты."""
    import json

    from mcp.shared.memory import create_connected_server_and_client_session as connect

    async with connect(mcp._mcp_server) as client:
        tools = (await client.list_tools()).tools
        print("инструменты:", ", ".join(t.name for t in tools))
        with _quiet_stdout():
            lid = str(_api()._df().index[0])
        for name, args in [("check_rent_price", {"listing": f"https://krisha.kz/a/show/{lid}"}),
                           ("check_rent_price", {"area": 60, "rooms": 2, "city": "алматы",
                                                 "district": "бостандыкский р-н", "price": 550000}),
                           ("get_comparables", {"listing": lid, "k": 3}),
                           ("search_apartments", {"style": "светлая кухня", "city": "алматы",
                                                  "rooms": [2], "price_max": 400000, "limit": 3})]:
            r = await client.call_tool(name, args)
            body = r.structuredContent or (r.content[0].text if r.content else None)
            print(f"\n{name}({args}) -> {'ОШИБКА' if r.isError else 'ok'}")
            print(json.dumps(body, ensure_ascii=False, default=str)[:400])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import anyio
        anyio.run(_selftest)
    else:
        mcp.run()
