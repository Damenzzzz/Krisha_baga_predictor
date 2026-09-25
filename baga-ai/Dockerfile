# Baga AI: сайт (Streamlit), Telegram-бот и MCP-сервер — один образ, разные команды.
#
#   docker compose up -d            — Qdrant + загрузка индекса + сайт на :8501
#   docker compose --profile telegram up -d   — ещё и бот
#
# Веса SigLIP 2 скачиваются при сборке: контейнер стартует без интернета до Hugging Face
# и без минутной загрузки модели на первый запрос.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    BAGA_VAR=/data/var \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ffmpeg — голосовые ответы в Telegram (WAV -> OGG/Opus); libgomp — CatBoost и torch на CPU
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch для CPU отдельно: колесо с CUDA весит ~2.5 ГБ, а GPU сервису не нужен —
# на CPU кодируется только текст запроса или одна картинка (~50 мс)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG CLIP_MODEL_ID=google/siglip2-base-patch16-224
RUN python -c "from transformers import AutoModel, AutoProcessor; \
AutoModel.from_pretrained('${CLIP_MODEL_ID}'); AutoProcessor.from_pretrained('${CLIP_MODEL_ID}')"

COPY . .

RUN useradd --create-home --uid 1000 baga \
    && mkdir -p /data/var \
    && chown -R baga:baga /app /data
USER baga

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
