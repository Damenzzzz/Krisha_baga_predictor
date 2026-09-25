"""Индекс фотографий: какая картинка к какому объявлению относится.

Порядок строк в photos.parquet = порядок строк в матрице эмбеддингов.
После того как посчитан embed, индекс не перестраивать (embed.py это проверяет).

path хранится ОТНОСИТЕЛЬНО PHOTOS_DIR («<listing_id>/<n>.jpg»): индекс, посчитанный
на Kaggle, работает локально без правок. Абсолютный путь — колонка abs_path.
"""
from pathlib import Path

import pandas as pd

from config import ART_DIR, PHOTOS_DIR, PHOTOS_INDEX

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def build_photo_index(photos_dir=PHOTOS_DIR) -> pd.DataFrame:
    """Ожидает photos/<listing_id>/*.jpg. Если у вас другая раскладка
    (одна папка, listing_id в имени файла, CSV со ссылками) — меняется только эта функция."""
    rows = []
    for d in sorted(p for p in Path(photos_dir).iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        for n, f in enumerate(files):
            rows.append({"listing_id": d.name, "photo_n": n, "path": f"{d.name}/{f.name}"})
    if not rows:
        raise FileNotFoundError(f"В {photos_dir} не найдено ни одного фото")

    df = pd.DataFrame(rows)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PHOTOS_INDEX, index=False)
    print(f"фото: {len(df)}, объявлений: {df.listing_id.nunique()}")
    return df


def load_photo_index() -> pd.DataFrame:
    df = pd.read_parquet(PHOTOS_INDEX)
    df["listing_id"] = df["listing_id"].astype(str)
    df["abs_path"] = [str(PHOTOS_DIR / p) for p in df["path"]]
    return df
