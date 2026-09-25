#!/bin/sh
# Старт сайта (FastAPI + фронтенд web/) в одном контейнере (Hugging Face Spaces, docker run без compose):
# индекс Qdrant строится при первом запуске (встроенный режим, если QDRANT_URL пуст),
# при повторном — только проверка. В docker compose это уже сделал сервис init.
set -e
python qdrant_store.py ensure
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8501}" --proxy-headers --forwarded-allow-ips="*"
