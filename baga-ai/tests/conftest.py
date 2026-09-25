import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # модули лежат в корне проекта

# Тесты не трогают рабочую базу var/ и не шлют трейсы: всё состояние — во временной папке.
# Ставится ДО импорта config: он читает эти переменные при загрузке.
_tmp = tempfile.mkdtemp(prefix="baga-test-")
os.environ["BAGA_VAR"] = _tmp
os.environ["BAGA_USERS"] = str(Path(_tmp) / "users.json")
os.environ["LANGFUSE_TRACING"] = "false"
os.environ["TRACING"] = "0"
os.environ["COOKIE_SECURE"] = "0"          # тестовый клиент ходит по http — secure-cookie он не отправит
