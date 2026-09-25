#!/bin/sh
# Старт сайта (FastAPI + фронтенд web/) в одном контейнере: Hugging Face Spaces, docker run.
# Индекс Qdrant строится при первом запуске (встроенный режим, если QDRANT_URL пуст),
# при повторном — только проверка. В docker compose это уже сделал сервис init.
# RUN_BOT=1 и TELEGRAM_BOT_TOKEN — рядом с сайтом стартует Telegram-бот (в compose у бота
# свой сервис, поэтому там RUN_BOT не задан и второй экземпляр не запускается).
set -e
python qdrant_store.py ensure
if [ "${RUN_BOT:-0}" = "1" ] && [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
  python telegram_bot.py &
fi
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8501}" --proxy-headers --forwarded-allow-ips="*"
