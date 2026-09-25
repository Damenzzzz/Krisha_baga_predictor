#!/bin/sh
# Старт сайта в одном контейнере (Hugging Face Spaces, docker run без compose):
# индекс Qdrant строится при первом запуске (встроенный режим, если QDRANT_URL пуст),
# при повторном — только проверка. В docker compose это уже сделал сервис init.
set -e
python qdrant_store.py ensure
exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0
