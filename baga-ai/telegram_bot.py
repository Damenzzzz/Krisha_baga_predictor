"""Telegram-бот Baga AI: тот же агент, что на сайте, в мессенджере.

Зачем: квартиры ищут с телефона, а ссылки на объявления krisha.kz пересылают друг другу
в Telegram. Бот закрывает оба сценария там, где человек уже находится:
  * текст или ГОЛОСОВОЕ «двушка в Алматы до 400 тысяч, светлая кухня» — LangGraph-агент,
    пауза на подтверждение домысленных условий — inline-кнопки «Да / Нет»;
  * пересланная ссылка krisha.kz/a/show/<id> — проверка цены с объяснением по аналогам;
  * 👍/👎 под каждым ответом — отзыв в SQLite и оценка на трейс в Langfuse.

Long polling через httpx (уже стоит как зависимость openai): вебхук потребовал бы
публичный HTTPS-адрес, а polling работает с ноутбука и из docker compose одинаково.

Запуск:  TELEGRAM_BOT_TOKEN=... python telegram_bot.py
Токен — у @BotFather. Админы (без лимита): TELEGRAM_ADMIN_IDS=123,456
"""
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import httpx

import config  # noqa: F401 — загружает .env до чтения переменных ниже

TOKEN =os.getenv("TELEGRAM_BOT_TOKEN", "")
API = f"https://api.telegram.org/bot{TOKEN}"
ADMIN_IDS = {s.strip() for s in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if s.strip()}
LINK = re.compile(r"krisha\.kz/a/show/(\d+)")
HELP = ("Я ищу квартиры в аренду в Алматы и Каскелене по описанию ремонта и проверяю, "
        "в рынке ли цена.\n\n"
        "• Напишите или скажите голосом: «двушка в Бостандыке до 400 тысяч, светлая кухня»\n"
        "• Пришлите ссылку krisha.kz/a/show/… — скажу, дороже ли она похожих объявлений\n"
        "• /price <id или ссылка> — то же самое\n\n"
        "Цены сравниваю с ценами объявлений, а не сделок.")

http = httpx.Client(timeout=httpx.Timeout(70.0, connect=10.0))
_photo_urls = None


def log(*a):
    print("[bot]", *a, file=sys.stderr, flush=True)


def tg(method: str, **params):
    files = params.pop("files", None)
    data = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
            for k, v in params.items() if v is not None}
    r = http.post(f"{API}/{method}", data=data, files=files)
    body = r.json()
    if not body.get("ok"):
        log(method, "ошибка:", body.get("description"))
    return body.get("result")


def photo_url(listing_id: str) -> str | None:
    global _photo_urls
    if _photo_urls is None:
        try:
            import pandas as pd
            from config import ART_DIR
            df = pd.read_parquet(ART_DIR / "photo_urls.parquet").sort_values(["listing_id", "photo_n"])
            _photo_urls = df.groupby("listing_id").url.first().to_dict()
        except Exception:
            _photo_urls = {}
    return _photo_urls.get(str(listing_id))


def money(x) -> str:
    return f"{int(x):,} ₸".replace(",", " ") if x else "—"


def role_of(chat_id) -> str:
    return "admin" if str(chat_id) in ADMIN_IDS else "user"


def feedback_kb(kind: str, ref: str):
    return {"inline_keyboard": [[{"text": "👍", "callback_data": f"fb:{kind}:1:{ref}"},
                                 {"text": "👎", "callback_data": f"fb:{kind}:0:{ref}"}]]}


# ref → (запрос, ответ, trace_id) — чтобы отзыв знал, к чему он относится
_answers: dict[str, tuple[str, str, str | None]] = {}


def remember(query: str, answer: str, trace_id: str | None) -> str:
    ref = f"{int(time.time() * 1000) % 10**10}"
    _answers[ref] = (query, answer, trace_id)
    if len(_answers) > 2000:
        for k in list(_answers)[:500]:
            _answers.pop(k, None)
    return ref


# --------------------------------------------------------------- сценарии

def handle_query(chat_id, text: str, voice_reply: bool = False):
    import agent
    from auth import limiter
    if not limiter.allow(f"tg:{chat_id}", role_of(chat_id)):
        tg("sendMessage", chat_id=chat_id, text="Лимит запросов на этот час исчерпан, попробуйте позже.")
        return
    tg("sendChatAction", chat_id=chat_id, action="typing")
    thread = f"tg-{chat_id}-{int(time.time())}"
    state, trace_id = agent.invoke(text, thread_id=thread, user_id=f"tg:{chat_id}", channel="telegram")
    if "__interrupt__" in state:
        p = state["__interrupt__"][0].value
        _answers[f"q:{thread}"] = (text, "", trace_id)
        tg("sendMessage", chat_id=chat_id,
           text=p["question"] + "\n" + "\n".join(f"• {a}" for a in p["assumptions"]),
           reply_markup={"inline_keyboard": [[
               {"text": "Да, искать с ними", "callback_data": f"cf:1:{thread}:{int(voice_reply)}"},
               {"text": "Нет, убрать", "callback_data": f"cf:0:{thread}:{int(voice_reply)}"}]]})
        return
    send_results(chat_id, text, state, trace_id, voice_reply)


def resume(chat_id, thread: str, ok: bool, voice_reply: bool):
    import agent
    tg("sendChatAction", chat_id=chat_id, action="typing")
    state, trace_id = agent.invoke(resume=ok, thread_id=thread, user_id=f"tg:{chat_id}", channel="telegram")
    query = _answers.pop(f"q:{thread}", ("", "", None))[0]
    send_results(chat_id, query, state, trace_id, voice_reply)


def send_results(chat_id, query: str, state: dict, trace_id, voice_reply: bool):
    answer = state.get("answer") or "Ничего не нашлось."
    ref = remember(query, answer, trace_id)
    tg("sendMessage", chat_id=chat_id, text=answer, reply_markup=feedback_kb("assistant", ref),
       disable_web_page_preview=True)
    for r in (state.get("results") or [])[:3]:
        l, pc = r.get("listing", {}), r.get("price_check") or {}
        cap = (f"{money(l.get('price'))}/мес · {l.get('rooms')}-комн · {l.get('area')} м² · "
               f"{l.get('district') or l.get('city')}\n{pc.get('verdict', '')}\n{l.get('url', '')}")
        if url := photo_url(r["listing_id"]):
            tg("sendPhoto", chat_id=chat_id, photo=url, caption=cap[:1000])
        else:
            tg("sendMessage", chat_id=chat_id, text=cap, disable_web_page_preview=True)
    if voice_reply:
        send_voice(chat_id, answer)


def send_voice(chat_id, text: str):
    import voice
    wav = voice.synthesize(text)
    ogg = voice.wav_to_ogg_opus(wav) if wav else None
    if ogg:
        tg("sendVoice", chat_id=chat_id, files={"voice": ("answer.ogg", ogg, "audio/ogg")})


def handle_price(chat_id, listing_id: str):
    import api
    from explain import explain_price
    from auth import limiter
    try:
        est = api.estimate_price(listing_id=listing_id)
        card = api.get_listing(listing_id)
    except KeyError:
        tg("sendMessage", chat_id=chat_id,
           text=f"Объявления {listing_id} нет в нашей базе (срез 21.09.2026). Проверить можно по "
                "характеристикам на сайте: площадь, комнаты, район.")
        return
    if limiter.allow(f"tg:{chat_id}", role_of(chat_id)):
        text = explain_price(card, est)
    else:
        from explain import template_explanation
        text = template_explanation(card, est)
    ref = remember(listing_id, text, None)
    head = (f"{money(card.get('price'))}/мес · {card.get('rooms')}-комн · {card.get('area')} м² · "
            f"{card.get('district')}\nПохожие: {money(est['p10'])} – {money(est['p90'])}\n")
    tg("sendMessage", chat_id=chat_id, text=head + "\n" + text, reply_markup=feedback_kb("price", ref),
       disable_web_page_preview=True)


def handle_voice(chat_id, file_id: str, mime: str):
    import voice
    f = tg("getFile", file_id=file_id)
    if not f:
        return
    audio = http.get(f"https://api.telegram.org/file/bot{TOKEN}/{f['file_path']}").content
    try:
        text = voice.transcribe(audio, mime or "audio/ogg", user_id=f"tg:{chat_id}")
    except voice.VoiceUnavailable as e:
        tg("sendMessage", chat_id=chat_id, text=str(e))
        return
    if not text:
        tg("sendMessage", chat_id=chat_id, text="Не расслышал — повторите, пожалуйста.")
        return
    tg("sendMessage", chat_id=chat_id, text=f"🎙 {text}")
    route_text(chat_id, text, voice_reply=True)


def route_text(chat_id, text: str, voice_reply: bool = False):
    if m := LINK.search(text):
        handle_price(chat_id, m.group(1))
    elif text.startswith("/price"):
        arg = text.split(maxsplit=1)[1] if " " in text else ""
        lid = (LINK.search(arg) or re.search(r"(\d{6,})", arg))
        if lid:
            handle_price(chat_id, lid.group(1))
        else:
            tg("sendMessage", chat_id=chat_id, text="Формат: /price 1014955701 или ссылка krisha.kz")
    elif text.startswith(("/start", "/help")):
        tg("sendMessage", chat_id=chat_id, text=HELP)
    else:
        handle_query(chat_id, text, voice_reply)


def handle_callback(cb: dict):
    import store
    import tracing
    chat_id = cb["message"]["chat"]["id"]
    data = cb.get("data", "")
    tg("answerCallbackQuery", callback_query_id=cb["id"])
    if data.startswith("cf:"):
        _, ok, thread, vr = data.split(":", 3)
        tg("editMessageReplyMarkup", chat_id=chat_id, message_id=cb["message"]["message_id"],
           reply_markup={"inline_keyboard": []})
        resume(chat_id, thread, ok == "1", vr == "1")
    elif data.startswith("fb:"):
        _, kind, val, ref = data.split(":", 3)
        q, a, trace_id = _answers.get(ref, ("", "", None))
        store.add_feedback(f"tg:{chat_id}", "telegram", kind, int(val), query_text=q, answer=a, trace_id=trace_id)
        tracing.score(trace_id, "user_feedback", int(val), data_type="BOOLEAN")
        tracing.flush()
        tg("editMessageReplyMarkup", chat_id=chat_id, message_id=cb["message"]["message_id"],
           reply_markup={"inline_keyboard": [[{"text": "Спасибо за отзыв!", "callback_data": "noop"}]]})


def handle_update(u: dict):
    if cb := u.get("callback_query"):
        return handle_callback(cb)
    msg = u.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    if not chat_id:
        return
    if v := (msg.get("voice") or msg.get("audio")):
        return handle_voice(chat_id, v["file_id"], v.get("mime_type"))
    if text := (msg.get("text") or "").strip():
        route_text(chat_id, text)


def main():
    if not TOKEN:
        sys.exit("нужен TELEGRAM_BOT_TOKEN (выдаёт @BotFather)")
    me = tg("getMe")
    log(f"запущен как @{me and me.get('username')}")
    tg("setMyCommands", commands=[{"command": "help", "description": "что умеет бот"},
                                   {"command": "price", "description": "проверить цену объявления"}])
    offset = None
    while True:
        try:
            updates = tg("getUpdates", timeout=50, offset=offset,
                         allowed_updates=["message", "callback_query"]) or []
        except httpx.HTTPError as e:
            log("сеть:", e)
            time.sleep(3)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            try:
                handle_update(u)
            except Exception:
                log(traceback.format_exc())
                chat = (u.get("message") or u.get("callback_query", {}).get("message") or {}).get("chat", {})
                if chat.get("id"):
                    tg("sendMessage", chat_id=chat["id"], text="Что-то пошло не так, попробуйте ещё раз.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
