"""Смысловой кэш: подпись не даёт делить кэш запросам с разными фильтрами,
даже если эмбеддинги совпадают полностью."""
import numpy as np

from semantic_cache import SemanticCache, signature


def const_embed(text):                  # худший случай: все запросы «одинаковые» по смыслу
    return np.ones(8, dtype=np.float32)


def test_signature_separates_numbers_and_rooms():
    assert signature("двушка до 400 тысяч") != signature("двушка до 500 тысяч")
    assert signature("двушка до 400к") != signature("трешка до 400к")
    assert signature("от 400 тысяч") != signature("до 400 тысяч")
    assert signature("не первый этаж") != signature("")


def test_signature_equal_for_paraphrases():
    assert signature("двушка в Алматы до 400 тысяч") == signature("2-комнатная алматы до 400 тыс")
    assert signature("однушка до 12к") == signature("однокомнатная до 12 тысяч")
    assert signature("до 1.2 млн") == signature("до 1 200 000")


def test_room_count_is_not_a_price():
    assert "r2" in signature("2к до 400к") and '"n": [400000]' in signature("2к до 400к")
    assert '"n": [60]' in signature("60 м2")          # «м2» — не число 2


def test_cache_hit_and_miss_by_signature():
    c = SemanticCache("test:sig", threshold=0.9, embed=const_embed)
    c.put("двушка до 400 тысяч, светлая кухня", {"filters": {"price_max": 400000}})
    hit, info = c.lookup("2-комнатная до 400 тыс, кухня светлая")
    assert hit == {"filters": {"price_max": 400000}} and info["score"] > 0.99
    miss, _ = c.lookup("двушка до 500 тысяч, светлая кухня")
    assert miss is None


def test_threshold_blocks_different_style():
    vecs = {"светлая кухня": np.array([1, 0], np.float32), "темная кухня": np.array([0.8, 0.6], np.float32)}
    c = SemanticCache("test:thr", threshold=0.94, embed=lambda t: vecs[t])
    c.put("светлая кухня", {"style": "светлая"})
    assert c.lookup("темная кухня")[0] is None       # косинус 0.8 < порога


def test_embedder_down_means_miss_not_error():
    def broken(t):
        raise RuntimeError("нет сети")
    c = SemanticCache("test:down", embed=broken)
    c.put("x", {"a": 1})
    assert c.lookup("x")[0] is None
