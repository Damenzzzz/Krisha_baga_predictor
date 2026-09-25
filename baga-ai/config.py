"""Пути и параметры. Всё, что зависит от машины, — через переменные окружения или .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- данные (от парсера)
DATA_DIR = Path(os.getenv("KRISHA_DATA", ROOT / "data"))
LISTINGS_RAW = Path(os.getenv("LISTINGS_RAW", DATA_DIR / "listings_raw.parquet"))  # csv/parquet/json/jsonl
PHOTOS_DIR = Path(os.getenv("PHOTOS_DIR", DATA_DIR / "photos"))                     # photos/<listing_id>/*.jpg

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
# Параметры генерации подобраны экспериментом (ab_generation.py, EVALS.md раздел 3):
#   temperature 0 — достоверность на 0 / 0.3 / 0.7 одинаковая, а 0 даёт воспроизводимость;
#   top_p 1.0    — при temperature 0 декодирование жадное и top_p ни на что не влияет;
#   max_tokens 800 — при 400 обрывалась половина ответов по выдаче (русский текст у alemllm
#                  ~1.7 символа на токен), при 800 — ни одного. Это потолок, а не расход:
#                  средний ответ 190–290 токенов.
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.0))
CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", 1.0))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 800))

# ---------------------------------------------------------------- шлюз LLM (llm.py)
# Цепочка провайдеров: первый доступный отвечает, остальные — фолбэк. Порядок выбран
# A/B-экспериментом ALEM против Gemini (ab_models.py, EVALS.md раздел 6).
LLM_CHAIN = [p.strip() for p in os.getenv("LLM_CHAIN", "gemini,alem").split(",") if p.strip()]
GEMINI_KEY = os.getenv("GEMINI_API_KEY")            # ключ Vertex AI (express mode)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# У Gemini 3.x рассуждения (thinking) включены по умолчанию и съедают max_output_tokens:
# при лимите 400 модель тратила 382 токена на мысли и обрывала ответ. Уровень подобран
# экспериментом (ab_models.py, EVALS.md раздел 6): minimal — 100% точных разборов,
# low не лучше, но в 2.5–4 раза дороже и медленнее, medium хуже (додумывает «центр»).
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "minimal")
# Судья в evals: согласие с разметкой у ALEM и Gemini одинаковое (каппа 0.63), ALEM бесплатный.
JUDGE_CHAIN = [p.strip() for p in os.getenv("JUDGE_CHAIN", "alem,gemini").split(",") if p.strip()]
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-lite-preview-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", 30))
# Дневной бюджет на платные модели. Превышен — шлюз переходит на бесплатный ALEM.
LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", 3.0))
LLM_CACHE = os.getenv("LLM_CACHE", "1") == "1"
# 0.94 — минимальный порог без ложных попаданий на «светлая/тёмная кухня» (eval_semantic_cache.py)
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.94))

# ---------------------------------------------------------------- рабочие данные сервиса
# Кэш, учёт расходов, пользователи, отзывы. Не в git: это состояние, а не код.
VAR_DIR = Path(os.getenv("BAGA_VAR", ROOT / "var"))
VAR_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = VAR_DIR / "baga.sqlite"
USERS_PATH = Path(os.getenv("BAGA_USERS", VAR_DIR / "users.json"))

# ---------------------------------------------------------------- поиск
RRF_K = 60               # слияние рейтингов «по фото» и «по описанию», как в медицинском RAG
# Вес канала описаний в RRF. При 1.0 визуальный судья видел нужное на фото лишь в 59.5% топ-5,
# при 0.5 — в 81.9% (авторазметка 36.7 -> 35.0%, в пределах шума). EVALS.md, раздел 8.
RRF_DESC_WEIGHT = float(os.getenv("RRF_DESC_WEIGHT", 0.5))
SEED = 0


# torch нужен только для векторизации (embed.py, rooms.py). Оценке цены, MCP-серверу и CI
# он не нужен, а весит ~800 МБ — поэтому импорт необязательный.
try:
    import torch
except ImportError:
    torch = None


def _device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _device()
DTYPE = None if torch is None else (torch.float16 if DEVICE == "cuda" else torch.float32)
