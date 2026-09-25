# Публичный деплой: Hugging Face Spaces (Docker)

Почему Spaces: бесплатный CPU-тариф даёт 16 ГБ RAM и 2 vCPU — хватает на SigLIP 2 (кодирование
запроса ~50 мс на CPU), встроенный Qdrant на 73 тыс. векторов и CatBoost. На бесплатных
Render / Railway (512 МБ) модель не помещается.

## Шаги (≈10 минут)

1. https://huggingface.co/new-space → SDK **Docker**, шаблон Blank, железо CPU basic.
2. В Settings → Variables and secrets добавить секреты из `.env.example`:
   `GEMINI_API_KEY`, `ALEM_URL`, `ALEM_API_KEY`, `EMBED_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `BAGA_ADMIN_USER`, `BAGA_ADMIN_PASSWORD`.
   `QDRANT_URL` не задавать — будет встроенный Qdrant.
3. Залить содержимое `baga-ai/` в репозиторий Space и добавить в начало `README.md` Space:

   ```yaml
   ---
   title: Baga AI
   sdk: docker
   app_port: 8501
   ---
   ```

   ```bash
   git clone https://huggingface.co/spaces/<user>/baga-ai hf-space
   rsync -a --exclude .env --exclude var --exclude data --exclude 'artifacts/qdrant' baga-ai/ hf-space/
   cd hf-space && git lfs track "*.npy" "*.parquet" "*.cbm" && git add -A && git commit -m deploy && git push
   ```

Первый старт ~5 минут: сборка образа и загрузка векторов в Qdrant (`start.sh`). Хранилище Space
эфемерное — пользователи и отзывы живут до перезапуска; для постоянного хранения включить
Persistent storage и задать `BAGA_VAR=/data/var`.

## Быстрая публичная ссылка на время демо

С ноутбука, без аккаунта: `brew install cloudflared && cloudflared tunnel --url http://localhost:8501`
— выдаст адрес `https://….trycloudflare.com`, живёт, пока запущен.
