"""Пути и параметры. Всё, что зависит от машины, — через переменные окружения или .env."""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- данные (от парсера)
DATA_DIR = Path(os.getenv("KRISHA_DATA", ROOT / "data"))
LISTINGS_RAW = Path(os.getenv("LISTINGS_RAW", DATA_DIR / "listings_raw.parquet"))  # csv/parquet/json/jsonl
PHOTOS_DIR = DATA_DIR / "photos"                                                     # photos/<listing_id>/*.jpg

# ---------------------------------------------------------------- артефакты (наши)
ART_DIR = Path(os.getenv("KRISHA_ARTIFACTS", ROOT / "artifacts"))
LISTINGS_PATH = ART_DIR / "listings.parquet"    # после listings.prepare — единая схема
PHOTOS_INDEX = ART_DIR / "photos.parquet"       # listing_id, photo_n, path — порядок = строки эмбеддингов
SHARDS_DIR = ART_DIR / "emb_shards"
EMB_PATH = ART_DIR / "photo_emb.npy"            # (N, D) float16, L2-нормированы
VALID_PATH = ART_DIR / "photo_valid.npy"        # (N,) bool — битые фото = False
EMB_META = ART_DIR / "photo_emb.json"           # модель, виды, хэш индекса — защита от рассинхрона
ROOMS_PATH = ART_DIR / "photo_rooms.parquet"
DESC_EMB_PATH = ART_DIR / "desc_emb.npy"
DESC_IDS_PATH = ART_DIR / "desc_ids.npy"

# ---------------------------------------------------------------- фото: SigLIP 2
# Общее пространство картинка/текст, понимает русский -> «светлая кухня» ищется прямо по фото.
# Для качества: google/siglip2-so400m-patch14-384 (медленнее в ~4-5 раз).
CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "google/siglip2-base-patch16-224")

# Виды картинки, усредняются (l2 -> mean -> l2, как в museum):
#   pad    — вписать в квадрат с полями (resize_and_pad из museum-dorewka): пропорции и края целы
#   full   — растянуть в квадрат (так делает стандартный процессор SigLIP)
#   center — центральный кроп
# Мультискейл 336/512 из museum сюда не переносится: SigLIP обучена на фиксированном
# разрешении, на других размерах качество падает (DINOv3 так умел, SigLIP — нет).
VIEWS = tuple(os.getenv("VIEWS", "pad").split(","))

# CLAHE из museum-dorewka. Может сблизить тёмные фото с телефона и «глянцевые» ИИ-картинки,
# а может стереть разницу «светлый/тёмный ремонт», которую мы как раз ищем.
# Включать (USE_CLAHE=1), только если validate robust покажет прирост на dark/polish.
USE_CLAHE = os.getenv("USE_CLAHE", "0") == "1"

SHARD_SIZE = 4096        # после каждого шарда результат на диске: прервали — продолжит с места
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", min(8, os.cpu_count() or 1)))

RENDER_THR = 0.7         # выше — считаем фото 3D-рендером застройщика
ROOM_AUTO_CONF = 0.5     # ниже — не доверяем авто-определению комнаты у query-картинки

# ---------------------------------------------------------------- описания: текстовые эмбеддинги
# Та же ALEM-модель, что в медицинском RAG. Ключи — в .env (EMBED_API_KEY, ALEM_URL).
TEXT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-1024")
TEXT_EMBED_URL = os.getenv("ALEM_URL")
TEXT_EMBED_KEY = os.getenv("EMBED_API_KEY")

# ---------------------------------------------------------------- LLM (объяснения, разбор запроса)
# ALEM alemllm: единственная доступная нам модель с русским языком и без оплаты по токенам.
CHAT_MODEL = os.getenv("ALEM_CHAT_MODEL", "alemllm")
CHAT_URL = os.getenv("ALEM_URL")
CHAT_KEY = os.getenv("ALEM_API_KEY")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.2))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 400))

# ---------------------------------------------------------------- поиск
RRF_K = 60               # слияние рейтингов «по фото» и «по описанию», как в медицинском RAG
SEED = 0


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _device()
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
