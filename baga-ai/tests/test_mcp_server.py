"""MCP-сервер через настоящий MCP-клиент (транспорт в памяти): схема инструментов и вызовы.

search_apartments здесь не вызывается: ему нужны SigLIP и индекс Qdrant (гигабайты),
которых нет в CI. Остальные инструменты работают на закоммиченной модели цены.
"""
import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session as connect

from mcp_server import mcp

LISTING = "10635793"


def call(name, args):
    async def go():
        async with connect(mcp._mcp_server) as client:
            if name is None:
                return await client.list_tools()
            return await client.call_tool(name, args)
    return anyio.run(go)


def test_tools_are_described():
    tools = {t.name: t for t in call(None, None).tools}
    assert set(tools) == {"search_apartments", "check_rent_price", "get_comparables", "get_listing"}
    for t in tools.values():
        assert t.description and t.annotations.readOnlyHint


def test_check_price_by_krisha_url():
    r = call("check_rent_price", {"listing": f"https://krisha.kz/a/show/{LISTING}"})
    assert not r.isError
    body = r.structuredContent
    assert body["source"] == "out-of-fold"
    assert body["p10"] < body["p50"] < body["p90"]


def test_check_price_for_apartment_not_in_base():
    r = call("check_rent_price", {"area": 60, "rooms": 2, "city": "алматы",
                                  "district": "бостандыкский р-н", "price": 550000})
    assert not r.isError
    assert r.structuredContent["source"] == "модель"


def test_comparables_are_real_listings():
    r = call("get_comparables", {"listing": LISTING, "k": 3})
    assert not r.isError and len(r.structuredContent["comparables"]) == 3


@pytest.mark.parametrize("args", [{"listing": "не id"}, {"area": 60}])
def test_bad_input_is_a_tool_error_not_a_crash(args):
    assert call("check_rent_price", args).isError
