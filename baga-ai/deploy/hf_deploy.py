"""Деплой бэкенда на Hugging Face Spaces (Docker): сайт, API, Telegram-бот.

С сентября 2026 Docker-Spaces на бесплатном CPU требуют подписку HF PRO (ответ 402).
Сейчас бэкенд работает на нашем Mac за Cloudflare-туннелем (deploy/public_up.sh);
этот скрипт — готовый переезд, если появится PRO.

    python deploy/hf_deploy.py            # нужен HF_TOKEN (Write) в .env

Почему Spaces, а не Vercel, для бэкенда: бесплатный CPU-тариф даёт 16 ГБ памяти и постоянный
процесс. Нам нужны оба: SigLIP 2 и torch весят ~2 ГБ (лимит функции Vercel для Python —
500 МБ), а пауза ассистента на подтверждение живёт в памяти процесса между двумя запросами.
Vercel отдаёт фронтенд и проксирует /api сюда.

Ключи уходят в секреты Space через API и в репозиторий Space не попадают. Скрипт
идемпотентный: повторный запуск обновляет код и секреты.
"""
import io
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values, set_key
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "baga-ai"
SECRET_KEYS = ["GEMINI_API_KEY", "ALEM_URL", "ALEM_API_KEY", "EMBED_API_KEY", "LANGFUSE_PUBLIC_KEY",
               "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "BAGA_ADMIN_USER", "BAGA_ADMIN_PASSWORD",
               "TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_IDS", "BAGA_SECRET"]
VARIABLES = {"COOKIE_SECURE": "1", "TRACING": "0", "RUN_BOT": "1", "LLM_DAILY_BUDGET_USD": "3",
             "GEMINI_MODEL": "gemini-3.6-flash", "GEMINI_THINKING": "minimal"}
IGNORE = [".env", ".env.*", "var/**", "data/**", ".pytest_cache/**", "**/__pycache__/**", "*.pyc", ".DS_Store",
          "artifacts/qdrant/**", "artifacts/qdrant_server/**", "artifacts/emb_shards/**", "artifacts/photo_emb.npy",
          "artifacts/dino_emb.npy", "notebooks/**", "review/**", "mew.ipynb", ".claude/**", "README.md"]

SPACE_README = """---
title: Baga — аренда в Алматы по виду ремонта
emoji: 🏔
colorFrom: gray
colorTo: red
sdk: docker
app_port: 8501
pinned: true
short_description: Поиск аренды по фото ремонта и проверка цены
---

Бэкенд и сайт проекта Baga (Исламбек Султанбек и Нурдаулет, nFactorial 2026).
Код и документация: https://github.com/Damenzzzz/Krisha_baga_predictor
"""


def main():
    env_path = ROOT / ".env"
    env = dotenv_values(env_path)
    token = env.get("HF_TOKEN")
    if not token:
        sys.exit("нет HF_TOKEN в .env — создайте токен с правом Write: huggingface.co/settings/tokens")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    print(f"Space: https://huggingface.co/spaces/{repo_id}")

    if not env.get("BAGA_SECRET"):                  # стабильная подпись cookie между перезапусками
        env["BAGA_SECRET"] = secrets.token_hex(32)
        set_key(str(env_path), "BAGA_SECRET", env["BAGA_SECRET"])
    for k in SECRET_KEYS:
        if env.get(k):
            api.add_space_secret(repo_id, k, env[k])
    for k, v in VARIABLES.items():
        api.add_space_variable(repo_id, k, v)
    print(f"секретов: {sum(bool(env.get(k)) for k in SECRET_KEYS)}, переменных: {len(VARIABLES)}")

    api.upload_file(path_or_fileobj=io.BytesIO(SPACE_README.encode()), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", commit_message="README Space")
    api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="деплой baga-ai")
    host = f"https://{user.lower().replace('_', '-')}-{SPACE_NAME}.hf.space"
    print(f"загружено; сборка идёт несколько минут. Адрес: {host}")
    print(f"HF_SPACE_URL={host}")


if __name__ == "__main__":
    main()
