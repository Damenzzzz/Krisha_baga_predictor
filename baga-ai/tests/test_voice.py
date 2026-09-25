import io
import wave

import voice


def test_speakable_strips_ids_and_markdown():
    t = voice.speakable("Похожие сдают за 380 000 ₸ [1014955701], **дороже** [1015807119].")
    assert "[" not in t and "*" not in t and "380 000" in t


def test_speakable_cuts_on_sentence():
    t = voice.speakable("Первое предложение. " * 100)
    assert len(t) <= voice.MAX_TTS_CHARS and t.endswith(".")


def test_pcm_to_wav_header():
    wav = voice.pcm_to_wav(b"\x00\x00" * 24000, rate=24000)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 24000 and w.getnchannels() == 1
