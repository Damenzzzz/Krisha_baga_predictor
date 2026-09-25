"""Эмбеддинги описаний объявлений (если описание есть в данных).

Фото и текст ищутся разными моделями: SigLIP хорошо сопоставляет «светлая кухня»
с картинкой, но текст с текстом сравнивает слабо. Для описаний — обычная текстовая
модель; по умолчанию ALEM text-1024, как в медицинском RAG.
"""
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from config import DESC_EMB_PATH, DESC_IDS_PATH, TEXT_EMBED_KEY, TEXT_EMBED_MODEL, TEXT_EMBED_URL
from listings import load_listings

MIN_DESC_LEN = 20       # короче — не описание, а «продам срочно»
MAX_DESC_LEN = 2000


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _client():
    if not (TEXT_EMBED_KEY and TEXT_EMBED_URL):
        raise RuntimeError("Нет EMBED_API_KEY / ALEM_URL в krisha_search/.env")
    return OpenAI(api_key=TEXT_EMBED_KEY, base_url=TEXT_EMBED_URL)


def embed_texts(texts, batch=64, verbose=False):
    cl, out = _client(), []
    rng = range(0, len(texts), batch)
    for i in (tqdm(rng, desc="описания") if verbose else rng):
        r = cl.embeddings.create(model=TEXT_EMBED_MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in r.data)
    return _l2(np.asarray(out, dtype=np.float32))


def embed_descriptions():
    df = load_listings()
    if "description" not in df:
        print("в данных нет колонки description — канал «по описанию» будет выключен")
        return
    d = df[df.description.fillna("").str.len() >= MIN_DESC_LEN]
    if d.empty:
        print("описаний нет — канал «по описанию» будет выключен")
        return
    E = embed_texts(d.description.str.slice(0, MAX_DESC_LEN).tolist(), verbose=True)
    np.save(DESC_EMB_PATH, E.astype(np.float16))
    np.save(DESC_IDS_PATH, d.listing_id.to_numpy().astype(str))
    print(f"описаний: {len(d)} из {len(df)} объявлений, dim={E.shape[1]}")


class DescSearch:
    def __init__(self):
        self.E = np.load(DESC_EMB_PATH).astype(np.float32)
        self.ids = np.load(DESC_IDS_PATH)

    def rank(self, text, listing_ids=None, k=200):
        s = self.E @ embed_texts([text])[0]
        if listing_ids is not None:
            s = np.where(np.isin(self.ids, list(listing_ids)), s, -np.inf)
        top = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in top if np.isfinite(s[i])]
