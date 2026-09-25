"""Каркас поиска квартир. Шаги по порядку (каждый можно перезапускать):

  Данные
    python run_pipeline.py mock [--photos]        # синтетика, пока нет реальных данных
    python run_pipeline.py prepare                # сырые объявления -> единая схема (listings.py)
    python run_pipeline.py status                 # что уже посчитано, какие каналы поиска включатся

  Векторизация
    python run_pipeline.py index                  # photos/<listing_id>/*.jpg -> photos.parquet
    python run_pipeline.py embed                  # SigLIP 2 по всем фото (шардами, можно прерывать)
    python run_pipeline.py rooms                  # тип комнаты + «рендер или фото»
    python run_pipeline.py embed-text             # описания -> ALEM text-1024 (если описания есть)

  Поиск
    python run_pipeline.py search --text "светлая кухня, скандинавский стиль" --city алматы --rooms 2 --price-max 45000000
    python run_pipeline.py search --image gen.png --district бостандык

  Проверки
    python run_pipeline.py validate robust        # чувствительность к качеству/свету/ракурсу/«глянцу»
    python run_pipeline.py validate sample-rooms  # -> review/rooms_todo.csv для ручной разметки
    python run_pipeline.py validate rooms review/rooms_todo.csv
    python run_pipeline.py validate review queries.example.csv
    python run_pipeline.py validate score review/review.csv
"""
import argparse
import json


def _ints(s):
    return [int(x) for x in s.split(",")] if s else None


def _strs(s):
    return [x.strip() for x in s.split(",")] if s else None


def status():
    import numpy as np
    import pandas as pd
    from config import (DESC_EMB_PATH, DESC_IDS_PATH, EMB_PATH, LISTINGS_PATH, LISTINGS_RAW,
                        PHOTOS_DIR, PHOTOS_INDEX, ROOMS_PATH, VALID_PATH)

    def line(ok, what, extra=""):
        print(f"  [{'x' if ok else ' '}] {what:34s} {extra}")

    line(LISTINGS_RAW.exists(), "сырые объявления", str(LISTINGS_RAW))
    lids = set()
    if LISTINGS_PATH.exists():
        lids = set(pd.read_parquet(LISTINGS_PATH, columns=["listing_id"]).listing_id)
    line(bool(lids), "listings.parquet (prepare)", f"{len(lids)} объявлений" if lids else "")
    n_dirs = sum(1 for p in PHOTOS_DIR.iterdir() if p.is_dir()) if PHOTOS_DIR.exists() else 0
    line(n_dirs > 0, "папки с фото", f"{n_dirs}" if n_dirs else "")
    if PHOTOS_INDEX.exists():
        pidx = pd.read_parquet(PHOTOS_INDEX)
        both = len(set(pidx.listing_id.astype(str)) & lids)
        line(True, "photos.parquet (index)", f"{len(pidx)} фото; объявлений с фото и в таблице: {both}")
    else:
        line(False, "photos.parquet (index)")
    line(EMB_PATH.exists(), "эмбеддинги фото (embed)",
         f"битых: {(~np.load(VALID_PATH)).sum()}" if VALID_PATH.exists() else "")
    line(ROOMS_PATH.exists(), "типы комнат (rooms)")
    line(DESC_EMB_PATH.exists(), "эмбеддинги описаний (embed-text)",
         f"{len(np.load(DESC_IDS_PATH))} описаний" if DESC_IDS_PATH.exists() else "")
    ch = [c for c, ok in [("фото", EMB_PATH.exists() and ROOMS_PATH.exists()),
                          ("описания", DESC_EMB_PATH.exists())] if ok]
    print(f"\n  каналы поиска: {', '.join(ch) if ch else 'нет — поиск будет только фильтровать'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mock")
    m.add_argument("--n", type=int, default=800)
    m.add_argument("--photos", action="store_true", help="ещё и фото-заглушки (для проверки обвязки)")
    sub.add_parser("prepare").add_argument("--raw", help="путь к файлу парсера (по умолчанию LISTINGS_RAW)")
    sub.add_parser("status")
    sub.add_parser("index")
    sub.add_parser("embed").add_argument("--force", action="store_true")
    sub.add_parser("rooms")
    sub.add_parser("embed-text")

    s = sub.add_parser("search")
    s.add_argument("--text")
    s.add_argument("--image")
    s.add_argument("--photo-rooms", help="типы комнат для фото-канала: kitchen,living_room")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--city")
    s.add_argument("--district", help="через запятую, подстрока: бостандык,медеу")
    s.add_argument("--rooms", help="число комнат через запятую: 2,3")
    s.add_argument("--price-min", type=float)
    s.add_argument("--price-max", type=float)
    s.add_argument("--area-min", type=float)
    s.add_argument("--area-max", type=float)
    s.add_argument("--year-min", type=int)
    s.add_argument("--building", help="через запятую: монолит,кирпич")
    s.add_argument("--not-first-floor", action="store_true")
    s.add_argument("--not-last-floor", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("what", choices=["robust", "sample-rooms", "rooms", "review", "score"])
    v.add_argument("file", nargs="?")
    v.add_argument("--k", type=int, default=5)
    v.add_argument("--n", type=int, default=300)

    a = p.parse_args()

    if a.cmd == "mock":
        import mock_data
        mock_data.main(n=a.n, photos=a.photos)
    elif a.cmd == "prepare":
        from listings import prepare
        prepare(a.raw) if a.raw else prepare()
    elif a.cmd == "status":
        status()
    elif a.cmd == "index":
        from photos import build_photo_index
        build_photo_index()
    elif a.cmd == "embed":
        from embed import embed_all
        embed_all(force=a.force)
    elif a.cmd == "rooms":
        from rooms import classify_all
        classify_all()
    elif a.cmd == "embed-text":
        from text_embed import embed_descriptions
        embed_descriptions()
    elif a.cmd == "search":
        from listings import Filters
        from search import get_search
        f = Filters(city=a.city, districts=_strs(a.district), rooms=_ints(a.rooms),
                    price_min=a.price_min, price_max=a.price_max, area_min=a.area_min,
                    area_max=a.area_max, year_min=a.year_min, building_types=_strs(a.building),
                    not_first_floor=a.not_first_floor, not_last_floor=a.not_last_floor)
        se = get_search()
        print(f"каналы: {se.channels or 'нет (только фильтры)'}")
        res = se.search(text=a.text, filters=f, image=a.image, rooms=_strs(a.photo_rooms), k=a.k)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif a.cmd == "validate":
        import validate as V
        if a.what in ("rooms", "review", "score") and not a.file:
            p.error(f"validate {a.what} требует путь к файлу")
        {"robust": lambda: V.robust(n_val=a.n, k=a.k),
         "sample-rooms": lambda: V.sample_rooms(n=a.n),
         "rooms": lambda: V.rooms_accuracy(a.file),
         "review": lambda: V.make_review(a.file, k=a.k),
         "score": lambda: V.score(a.file, k=a.k)}[a.what]()


if __name__ == "__main__":   # обязательно: DataLoader с num_workers на macOS запускает процессы через spawn
    main()
