"""Агент Baga AI на LangGraph.

Граф:

    parse ──► confirm? ──да──► [ПАУЗА: подтверждение пользователя] ──┐
      │                                                              │
      └──нет────────────────────────────────────────────────────────►├──► search
                                                                      │      │
                             ┌────────── relax ◄── пусто, попыток<3 ──┘      │
                             │            (ЦИКЛ)                             │
                             └────────────────────────────────────────► answer ──► END

Три вещи, которые требует задание, и зачем они здесь на самом деле:

ВЕТВЛЕНИЕ — подтверждение запрашивается не всегда, а только когда LLM додумала условие,
  которого пользователь не называл. Спрашивать каждый раз — раздражать, не спрашивать
  никогда — молча резать выдачу.
ЦИКЛ — если после фильтров ноль кандидатов, агент ослабляет самое жёсткое условие и ищет
  заново, до трёх попыток. Иначе пользователь получает пустой экран вместо «поднял цену
  на 30%, вот что есть».
ПОДТВЕРЖДЕНИЕ ЧЕЛОВЕКА — через interrupt() LangGraph: граф останавливается и ждёт ответа,
  состояние живёт в чекпоинтере.

Запуск:  python agent.py "двушка в Алматы до 400 тысяч, светлая кухня"
"""
from typing import Annotated, Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

MAX_RELAX = 3


class State(TypedDict, total=False):
    query: str
    filters: dict
    style: str
    photo_rooms: Optional[list[str]]
    assumptions: list[str]
    assumed_fields: list[str]
    confirmed: bool
    relaxed: list[str]
    attempts: int
    n_candidates: int
    results: list[dict]
    answer: str
    log: Annotated[list[str], lambda a, b: (a or []) + (b or [])]


# --------------------------------------------------------------- узлы

def parse_node(state: State) -> dict:
    from query_parser import parse
    p = parse(state["query"])
    return {"filters": p.filters.model_dump(exclude_none=True), "style": p.style,
            "photo_rooms": p.photo_rooms, "assumptions": p.assumptions,
            "assumed_fields": p.assumed_fields,
            "attempts": 0, "relaxed": [],
            "log": [f"разобрал запрос: условия={p.filters.model_dump(exclude_none=True)}, стиль='{p.style}'"]}


def confirm_node(state: State) -> dict:
    """Останавливает граф и ждёт человека. Возобновление — Command(resume=True/False)."""
    answer = interrupt({
        "question": "Я добавил условия, которых вы не называли. Искать с ними?",
        "assumptions": state.get("assumptions", []),
        "filters": state.get("filters", {}),
    })
    ok = bool(answer) if not isinstance(answer, dict) else bool(answer.get("confirmed"))
    if ok:
        return {"confirmed": True, "log": ["пользователь подтвердил домысленные условия"]}
    dropped = state.get("assumed_fields", [])
    keep = {k: v for k, v in state["filters"].items() if k not in dropped}
    return {"confirmed": True, "filters": keep,
            "log": [f"пользователь отклонил домысленные условия — убрал {dropped or 'ничего'}"]}


def search_node(state: State) -> dict:
    import api
    f = state.get("filters", {})
    res = api.search_listings(query=state.get("style") or state["query"],
                              photo_rooms=state.get("photo_rooms"), k=6,
                              **{k: f.get(k) for k in ("city", "districts", "rooms",
                                                       "price_min", "price_max", "area_min", "area_max")})
    return {"results": res["results"], "n_candidates": res["n_candidates"],
            "attempts": state.get("attempts", 0) + 1,
            "log": [f"поиск #{state.get('attempts', 0) + 1}: кандидатов {res['n_candidates']}, "
                    f"в выдаче {len(res['results'])}, каналы {res['channels_used']}"]}


RELAX_STEPS = [
    ("price_max", lambda v: v * 1.3, "поднял верхнюю цену на 30%"),
    ("districts", lambda v: None, "убрал ограничение по району"),
    ("rooms", lambda v: None, "убрал ограничение по числу комнат"),
    ("area_min", lambda v: None, "убрал нижнюю границу площади"),
]


def relax_node(state: State) -> dict:
    """Ослабляет ровно одно условие за проход — чтобы было видно, что именно изменилось."""
    f = dict(state.get("filters", {}))
    for key, fn, note in RELAX_STEPS:
        if f.get(key) is not None and note not in state.get("relaxed", []):
            new = fn(f[key])
            if new is None:
                f.pop(key, None)
            else:
                f[key] = new
            return {"filters": f, "relaxed": state.get("relaxed", []) + [note], "log": [f"ничего не нашлось — {note}"]}
    return {"log": ["ослаблять больше нечего"], "relaxed": state.get("relaxed", []) + ["предел"]}


def answer_node(state: State) -> dict:
    from explain import answer_search
    if not state.get("results"):
        return {"answer": "По таким условиям ничего не нашлось, даже после ослабления фильтров.",
                "log": ["ответ: пусто"]}
    text = answer_search(state["query"], state["results"])
    if state.get("relaxed"):
        text += "\n\nЧтобы что-то найти, пришлось ослабить условия: " + "; ".join(
            n for n in state["relaxed"] if n != "предел") + "."
    return {"answer": text, "log": ["сгенерировал ответ по найденному"]}


# --------------------------------------------------------------- ветвления

def need_confirmation(state: State) -> str:
    return "confirm" if state.get("assumptions") and not state.get("confirmed") else "search"


def after_search(state: State) -> str:
    if state.get("results"):
        return "answer"
    if state.get("attempts", 0) >= MAX_RELAX or "предел" in state.get("relaxed", []):
        return "answer"
    return "relax"


def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("parse", parse_node)
    g.add_node("confirm", confirm_node)
    g.add_node("search", search_node)
    g.add_node("relax", relax_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", need_confirmation, {"confirm": "confirm", "search": "search"})
    g.add_edge("confirm", "search")
    g.add_conditional_edges("search", after_search, {"relax": "relax", "answer": "answer"})
    g.add_edge("relax", "search")          # цикл
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(query: str, thread_id: str = "cli", on_confirm=None) -> dict[str, Any]:
    """on_confirm(payload) -> bool. Если не передан, домысленные условия принимаются."""
    cfg = {"configurable": {"thread_id": thread_id}}
    state = graph().invoke({"query": query}, cfg)
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        answer = on_confirm(payload) if on_confirm else True
        state = graph().invoke(Command(resume=answer), cfg)
    return state


if __name__ == "__main__":
    import sys

    def ask(payload):
        print("\n[пауза графа] " + payload["question"])
        for a in payload["assumptions"]:
            print("   •", a)
        return input("   искать с этими условиями? [Y/n] ").strip().lower() not in ("n", "no", "нет")

    q = sys.argv[1] if len(sys.argv) > 1 else "двушка в Алматы до 400 тысяч, светлая кухня"
    out = run(q, on_confirm=ask)
    print("\n--- ход работы графа")
    for line in out.get("log", []):
        print("  •", line)
    print("\n--- ответ\n" + out.get("answer", ""))
    for r in out.get("results", [])[:5]:
        l, pc = r["listing"], r.get("price_check", {})
        print(f"  [{r['listing_id']}] {int(l['price']):>8,} ₸ · {l.get('district')} · {pc.get('verdict', '')}"
              .replace(",", " "))
