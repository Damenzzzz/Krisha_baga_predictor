"""Zero-shot разметка фото: тип комнаты и «рендер или реальное фото».

Зачем: в объявлениях половина фото — не комнаты (фасад, подъезд, планировка),
а у новостроек стоят 3D-рендеры. Сгенерированная ИИ-картинка идеальна, рендер
тоже идеален — без фильтра они будут в топе выдачи вместо реальных квартир.

Несколько формулировок на класс усредняются (prompt ensembling): одна фраза
шумная, среднее по 2-3 стабильнее. Качество проверяется в validate.py rooms.
"""
import numpy as np
import pandas as pd

from config import ROOMS_PATH
from embed import embed_texts, l2, load_embeddings, load_model

ROOM_PROMPTS = {
    "kitchen":     ["фото кухни в квартире", "a photo of a kitchen in an apartment"],
    "living_room": ["фото гостиной комнаты", "a photo of a living room"],
    "bedroom":     ["фото спальни с кроватью", "a photo of a bedroom with a bed"],
    "bathroom":    ["фото ванной комнаты или санузла", "a photo of a bathroom with a toilet or bathtub"],
    "hallway":     ["фото коридора или прихожей в квартире", "a photo of an apartment hallway"],
    "balcony":     ["фото балкона или лоджии", "a photo of a balcony"],
    "exterior":    ["фасад жилого дома снаружи", "exterior of an apartment building"],
    "entrance":    ["подъезд или лестничная клетка", "a stairwell in an apartment building"],
    "floor_plan":  ["планировка квартиры, чертёж", "an apartment floor plan drawing"],
    "window_view": ["вид из окна на город или двор", "a view from the window"],
    "other":       ["документ или скриншот с текстом", "a screenshot with text"],
}
INTERIOR = {"kitchen", "living_room", "bedroom", "bathroom", "hallway", "balcony"}

RENDER_PROMPTS = {
    "render": ["3d визуализация интерьера, рендер", "a 3d render of an interior, cgi visualization"],
    "photo":  ["обычное фото квартиры на телефон", "an amateur phone photo of a real apartment"],
}

_class_cache = {}


def _class_matrix(prompts):
    key = tuple(prompts)
    if key not in _class_cache:
        _class_cache[key] = np.stack([l2(embed_texts(prompts[n]).mean(0)) for n in prompts])
    return list(prompts), _class_cache[key]


def classify(E, prompts):
    """(N, D) -> (имена классов, вероятности (N, C))."""
    model, _, _ = load_model()
    scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
    names, C = _class_matrix(prompts)
    logits = (E @ C.T) * scale
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    return names, P / P.sum(1, keepdims=True)


def room_of(q):
    """Тип комнаты для одной query-картинки -> (комната, уверенность)."""
    names, P = classify(q[None], ROOM_PROMPTS)
    return names[int(P[0].argmax())], float(P[0].max())


def classify_all():
    idx, E, valid = load_embeddings()
    names, P = classify(E, ROOM_PROMPTS)
    rnames, R = classify(E, RENDER_PROMPTS)

    df = idx[["listing_id", "photo_n", "path"]].copy()
    df["room_type"] = np.array(names)[P.argmax(1)]
    df["room_conf"] = P.max(1)
    df["render_prob"] = R[:, rnames.index("render")]
    df.loc[~valid, "room_type"] = "invalid"
    df.to_parquet(ROOMS_PATH, index=False)

    print(df["room_type"].value_counts().to_string())
    print(f"похоже на рендер: {(df.render_prob > 0.7).mean():.1%} фото")
    return df


def load_rooms() -> pd.DataFrame:
    return pd.read_parquet(ROOMS_PATH)
