"""Голос: речь → текст → агент → текст → речь.

Сценарий, ради которого это сделано: человек смотрит квартиры с телефона на ходу и
говорит «двушка в Бостандыке до четырёхсот, чтобы кухня светлая» вместо набора
текста с фильтрами. Ответ можно прослушать, не читая.

Каскад, а не нативная audio-to-audio модель (Gemini Live): между ушами и голосом стоит
тот же LangGraph-агент с фильтрами, подтверждением и guardrails. Нативная голосовая
модель отвечала бы сама — без поиска по базе и без проверки цифр.

  STT — Gemini 3.6 Flash (аудио на вход): русский, казахские топонимы, числа
        «четыреста тысяч» -> «400 тысяч». Проверено: фраза с ценой и районом распознана
        дословно за 2.6 с.
  TTS — Gemini 2.5 Flash Lite TTS (голос Kore), ~4 с на ответ из 3–4 предложений.

ALEM аудио не принимает, поэтому фолбэк голоса — честный отказ «наберите текстом»,
а не тишина. Всё идёт через трейсинг: STT — как generation шлюза llm.py, TTS — отдельно.
"""
import io
import re
import time
import wave

import config as C

STT_PROMPT = ("Дословно расшифруй речь. Это запрос об аренде квартиры в Казахстане. "
              "Числа пиши цифрами («400 тысяч»), районы Алматы — как пишутся: Бостандыкский, "
              "Медеуский, Алмалинский, Ауэзовский, Турксибский, Жетысуский, Наурызбайский, Алатауский. "
              "Верни только текст, без пояснений. Если речи нет — верни пустую строку.")
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_TTS_CHARS = 700


class VoiceUnavailable(RuntimeError):
    pass


def transcribe(audio: bytes, mime: str = "audio/wav", user_id: str | None = None) -> str:
    import llm
    from google.genai import types
    if not audio:
        raise VoiceUnavailable("пустая запись")
    if len(audio) > MAX_AUDIO_BYTES:
        raise VoiceUnavailable("запись длиннее ~5 минут — сократите запрос")
    audio, mime = normalize_audio(audio, mime)
    try:
        r = llm.complete("", [STT_PROMPT, types.Part.from_bytes(data=audio, mime_type=mime)], task="stt",
                         temperature=0, max_tokens=300, thinking="minimal", providers=["gemini"],
                         use_cache=False, user_id=user_id)
    except llm.LLMUnavailable as e:
        raise VoiceUnavailable("распознавание речи сейчас недоступно — наберите запрос текстом") from e
    return r.text.strip().strip('"«»')


def speakable(text: str) -> str:
    """Для озвучки: без [id] объявлений и markdown, не длиннее MAX_TTS_CHARS."""
    t = re.sub(r"\[\d{6,}\]", "", text)
    t = re.sub(r"[*_`#>]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > MAX_TTS_CHARS:
        cut = t[:MAX_TTS_CHARS]
        t = cut[:cut.rfind(".") + 1] or cut
    return t


def synthesize(text: str) -> bytes | None:
    """WAV-байты или None, если TTS недоступен (ответ остаётся текстом)."""
    import tracing
    from google.genai import types
    import llm
    text = speakable(text)
    if not text or not C.GEMINI_KEY:
        return None
    with tracing.generation("tts", model=C.GEMINI_TTS_MODEL, input=text,
                            metadata={"voice": C.GEMINI_TTS_VOICE}) as obs:
        t0 = time.time()
        try:
            r = llm.PROVIDERS["gemini"].client().models.generate_content(
                model=C.GEMINI_TTS_MODEL, contents=f"Спокойно и дружелюбно прочитай: {text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=C.GEMINI_TTS_VOICE)))))
            part = r.candidates[0].content.parts[0].inline_data
        except Exception as e:
            obs.update(level="ERROR", status_message=str(e)[:300])
            return None
        u = r.usage_metadata
        obs.update(output=f"<audio {len(part.data)} байт, {time.time() - t0:.1f} с>",
                   usage_details={"input": u.prompt_token_count or 0, "output": u.candidates_token_count or 0})
    return pcm_to_wav(part.data, rate=_rate(part.mime_type))


def _rate(mime: str | None) -> int:
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else 24000


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


GEMINI_AUDIO = {"audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"}


def _ffmpeg() -> str | None:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                       # pip install imageio-ffmpeg — ffmpeg без brew/apt
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def normalize_audio(data: bytes, mime: str) -> tuple[bytes, str]:
    """Браузеры пишут webm (Chrome, Firefox) или mp4 (Safari), а Gemini их не принимает —
    перекодируем в WAV 16 кГц моно. Поддерживаемые форматы отдаём как есть."""
    import subprocess
    if mime in GEMINI_AUDIO:
        return data, mime
    exe = _ffmpeg()
    if not exe:
        raise VoiceUnavailable("на сервере нет ffmpeg для перекодирования записи")
    p = subprocess.run([exe, "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                       input=data, capture_output=True, timeout=60)
    if p.returncode != 0 or not p.stdout:
        raise VoiceUnavailable("не удалось прочитать запись — попробуйте ещё раз")
    return p.stdout, "audio/wav"


def wav_to_ogg_opus(wav: bytes) -> bytes | None:
    """Голосовое сообщение Telegram принимает только OGG/Opus. Нужен ffmpeg (есть в
    Docker-образе) или пакет imageio-ffmpeg; без них бот отвечает текстом."""
    import subprocess
    exe = _ffmpeg()
    if not exe:
        return None
    p = subprocess.run([exe, "-loglevel", "error", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k",
                        "-f", "ogg", "pipe:1"], input=wav, capture_output=True, timeout=60)
    return p.stdout if p.returncode == 0 and p.stdout else None
