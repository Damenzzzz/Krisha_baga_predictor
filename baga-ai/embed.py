"""Эмбеддинги фото и текстовых запросов через SigLIP 2.

Схема из museum_multires: картинка декодируется один раз, тяжёлый прогон
делается один раз и лежит на диске, дальше поиск, классификация комнат
и валидация работают по готовой матрице на CPU за секунды.

Отличие от museum: там искали ТУ ЖЕ картину (instance retrieval) -> DINOv3.
Здесь ищем ПОХОЖИЙ СТИЛЬ ремонта, в том числе по тексту -> нужна модель
с общим пространством картинка/текст, поэтому SigLIP 2.
"""
import hashlib
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from config import (BATCH_SIZE, CLIP_MODEL_ID, DEVICE, DTYPE, EMB_META, EMB_PATH,
                    NUM_WORKERS, SHARD_SIZE, SHARDS_DIR, USE_CLAHE, VALID_PATH, VIEWS)
from photos import load_photo_index

BICUBIC = T.InterpolationMode.BICUBIC


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def open_rgb(src, max_side=None):
    """EXIF-поворот обязателен: фото с телефона без него лежат на боку."""
    if isinstance(src, Image.Image):
        im = ImageOps.exif_transpose(src).convert("RGB")
    else:
        with Image.open(src) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
    if max_side:
        im.thumbnail((max_side, max_side))
    return im


# ---------------------------------------------------------------- препроцессинг (из museum-dorewka)

def apply_clahe(im, clip_limit=2.0, tile=(8, 8)):
    """CLAHE только по яркости (L в LAB) — цвета ремонта не трогаем."""
    lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))


def preprocess(im):
    """Одинаково для галереи и запросов: домены должны обрабатываться идентично."""
    return apply_clahe(im) if USE_CLAHE else im


class PadToSquare:
    """resize_and_pad из museum-dorewka как transform. Цвет полей = mean модели
    (там было ImageNet mean (124,116,104), у SigLIP mean 0.5 -> (128,128,128)):
    после Normalize поля становятся нулями и не добавляют в картинку лишнего."""

    def __init__(self, size, fill):
        self.size, self.fill = size, fill

    def __call__(self, im):
        w, h = im.size
        k = self.size / max(w, h)
        nw, nh = max(1, round(w * k)), max(1, round(h * k))
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(im.resize((nw, nh), Image.BICUBIC), ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


# ---------------------------------------------------------------- модель

_model = _proc = _views = None
_img_size = 224


def load_model():
    global _model, _proc, _views, _img_size
    if _model is None:
        _proc = AutoProcessor.from_pretrained(CLIP_MODEL_ID)
        _model = AutoModel.from_pretrained(CLIP_MODEL_ID).eval().to(DEVICE, dtype=DTYPE)
        _views, _img_size = make_views(_proc.image_processor)
    return _model, _proc, _views


def make_views(ip, views=VIEWS):
    size = ip.size if isinstance(ip.size, dict) else {}
    s = size.get("height") or size.get("shortest_edge") or 224
    norm = T.Normalize(ip.image_mean, ip.image_std)
    fill = tuple(round(m * 255) for m in ip.image_mean)
    specs = {
        # вписать с полями: интерьер 4:3 не сплющивается и не теряет края
        "pad": T.Compose([PadToSquare(s, fill), T.ToTensor(), norm]),
        # растяжение в квадрат — ровно то, что делает процессор SigLIP
        "full": T.Compose([T.Resize((s, s), interpolation=BICUBIC), T.ToTensor(), norm]),
        # пропорции целы, края отрезаны — второй вид из museum
        "center": T.Compose([T.Resize(s, interpolation=BICUBIC), T.CenterCrop(s), T.ToTensor(), norm]),
    }
    return {k: specs[k] for k in views}, s


def _as_tensor(out):
    # в разных версиях transformers get_*_features отдаёт тензор или ModelOutput
    return out if torch.is_tensor(out) else out.pooler_output


@torch.inference_mode()
def encode_views(model, xs, forward=None):
    """{вид: (B,3,s,s)} -> (B, D): нормировать каждый вид, сложить, снова нормировать
    (правило из museum). forward переопределяется для DINOv3 в dedup.py."""
    forward = forward or (lambda x: _as_tensor(model.get_image_features(pixel_values=x)))
    acc = None
    for x in xs.values():
        v = F.normalize(forward(x.to(DEVICE, DTYPE)).float(), dim=-1)
        acc = v if acc is None else acc + v
    return F.normalize(acc, dim=-1).cpu().numpy()


# ---------------------------------------------------------------- галерея

class PhotoDS(Dataset):
    """Один декод файла -> все виды (как MultiViewDS в museum)."""

    def __init__(self, paths, views, size):
        self.paths, self.views, self.size = paths, views, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = preprocess(open_rgb(self.paths[i]))
            return {k: tf(im) for k, tf in self.views.items()}, i, True
        except Exception:
            # битое фото не выкидываем, а помечаем: порядок строк должен совпадать с индексом
            return {k: torch.zeros(3, self.size, self.size) for k in self.views}, i, False


def embed_paths(paths, model=None, views=None, size=None, forward=None, verbose=True):
    """-> (E: (N, D) float32 L2-норм., ok: (N,) bool)."""
    if model is None:
        model, _, views = load_model()
        size = _img_size
    dl = DataLoader(PhotoDS(paths, views, size), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    E, ok = None, np.zeros(len(paths), dtype=bool)
    for xs, idx, good in (tqdm(dl, leave=False) if verbose else dl):
        v = encode_views(model, xs, forward)
        if E is None:
            E = np.zeros((len(paths), v.shape[1]), dtype=np.float32)
        ii = idx.numpy()
        E[ii], ok[ii] = v, good.numpy()
    E[~ok] = 0
    return E, ok


def _paths_hash(paths):
    return hashlib.sha1("\n".join(paths).encode()).hexdigest()


def _meta(paths):
    return {"model": CLIP_MODEL_ID, "views": list(VIEWS), "clahe": USE_CLAHE, "n": len(paths),
            "paths_hash": _paths_hash(paths)}


def embed_all(force=False):
    """Считает эмбеддинги всех фото шардами. Прервали — перезапуск продолжит с места."""
    idx = load_photo_index()
    paths, files = idx["path"].tolist(), idx["abs_path"].tolist()
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    meta, meta_f = _meta(paths), SHARDS_DIR / "meta.json"

    if force:
        for f in SHARDS_DIR.glob("shard_*.npz"):
            f.unlink()
    elif meta_f.exists() and json.loads(meta_f.read_text()) != meta:
        raise RuntimeError("Индекс фото, модель или виды поменялись с прошлого запуска — шарды "
                           "несовместимы. Удалите artifacts/emb_shards или запустите с --force.")
    meta_f.write_text(json.dumps(meta))

    n_shards = math.ceil(len(paths) / SHARD_SIZE)
    for s in tqdm(range(n_shards), desc="шарды"):
        f = SHARDS_DIR / f"shard_{s:05d}.npz"
        if f.exists():
            continue
        E, ok = embed_paths(files[s * SHARD_SIZE:(s + 1) * SHARD_SIZE])
        tmp = SHARDS_DIR / f"shard_{s:05d}.tmp.npz"
        np.savez(tmp, emb=E.astype(np.float16), valid=ok)
        os.replace(tmp, f)   # недописанный шард не будет принят за готовый

    parts = [np.load(SHARDS_DIR / f"shard_{s:05d}.npz") for s in range(n_shards)]
    E = np.concatenate([p["emb"] for p in parts])
    V = np.concatenate([p["valid"] for p in parts])
    np.save(EMB_PATH, E)
    np.save(VALID_PATH, V)
    EMB_META.write_text(json.dumps(meta))
    print(f"эмбеддинги: {E.shape}, битых фото: {(~V).sum()}")


def load_embeddings():
    """-> (index, E float32, valid). Падает, если матрица посчитана для другого индекса
    или другой модели: иначе поиск молча выдавал бы мусор."""
    idx = load_photo_index()
    meta = json.loads(EMB_META.read_text())
    if meta["paths_hash"] != _paths_hash(idx["path"].tolist()):
        raise RuntimeError("photo_emb.npy посчитан для другого индекса фото — перезапустите embed")
    now = _meta(idx["path"].tolist())
    for key in ("model", "views", "clahe"):
        if meta.get(key) != now[key]:
            raise RuntimeError(f"Галерея посчитана с {key}={meta.get(key)}, а запросы будут с {now[key]} — "
                               f"векторы несравнимы. Верните настройку или перезапустите embed --force.")
    return idx, np.load(EMB_PATH).astype(np.float32), np.load(VALID_PATH)


# ---------------------------------------------------------------- запросы

def embed_images(images):
    """Список путей или PIL -> (B, D). Те же виды, что у галереи: домены обрабатываются одинаково."""
    model, _, views = load_model()
    ims = [preprocess(open_rgb(x)) for x in images]
    xs = {k: torch.stack([tf(im) for im in ims]) for k, tf in views.items()}
    return encode_views(model, xs)


@torch.inference_mode()
def embed_texts(texts):
    """Список строк -> (B, D). SigLIP обучали на тексте в нижнем регистре с паддингом до 64."""
    model, proc, _ = load_model()
    inp = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64,
               truncation=True, return_tensors="pt").to(DEVICE)
    v = _as_tensor(model.get_text_features(**inp))
    return F.normalize(v.float(), dim=-1).cpu().numpy()
