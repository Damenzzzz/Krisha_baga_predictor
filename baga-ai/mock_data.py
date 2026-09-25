"""Синтетические объявления, чтобы каркас работал до прихода реальных данных.

Цена считается по правдоподобной формуле (район, площадь, этаж, год, тип дома,
ремонт + шум), поэтому на моке уже можно отлаживать модель цены. Числа НЕ реальные:
выводы о рынке по ним не делать. Специально добавлен мусор (дубли, «1 тенге»,
цена строкой) — чтобы проверить, что listings.prepare его чистит.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from config import DATA_DIR, LISTINGS_RAW, PHOTOS_DIR

DISTRICTS = {   # город: {район: базовая цена ₸/м²}
    "алматы": {"медеуский": 1_150_000, "бостандыкский": 1_050_000, "алмалинский": 950_000,
               "ауэзовский": 700_000, "жетысуский": 560_000, "турксибский": 520_000,
               "наурызбайский": 550_000, "алатауский": 480_000},
    "астана": {"есильский": 850_000, "нура": 700_000, "сарыаркинский": 600_000,
               "байконурский": 560_000, "алматинский": 520_000},
}
BUILDING = {"монолитный": 1.08, "кирпичный": 1.05, "панельный": 0.92, "иное": 0.95}
CONDITION = {"дизайнерский ремонт": 1.18, "свежий ремонт": 1.08, "хорошее": 1.0,
             "среднее": 0.93, "требует ремонта": 0.82, "черновая отделка": 0.85}
STYLE = {
    "дизайнерский ремонт": ["дизайнерский ремонт", "встроенная техника", "тёплые полы",
                            "кухня-гостиная с островом", "мраморная плитка в санузле"],
    "свежий ремонт": ["светлая кухня с белыми фасадами", "ламинат", "натяжные потолки",
                      "скандинавский стиль", "новая сантехника"],
    "хорошее": ["аккуратный ремонт", "пластиковые окна", "встроенные шкафы", "линолеум"],
    "среднее": ["косметический ремонт", "обои", "кафель в ванной", "межкомнатные двери"],
    "требует ремонта": ["старый советский ремонт", "деревянные окна", "ковёр на стене", "паркет"],
    "черновая отделка": ["черновая отделка", "ремонт под себя", "бетонные стены", "стяжка пола"],
}
MOCK_COLORS = {"дизайнерский ремонт": (60, 60, 70), "свежий ремонт": (235, 235, 225),
               "хорошее": (200, 190, 170), "среднее": (170, 150, 120),
               "требует ремонта": (120, 90, 60), "черновая отделка": (150, 150, 150)}


def make_listings(n=800, seed=0) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        city = r.choice(list(DISTRICTS), p=[0.6, 0.4])
        district = r.choice(list(DISTRICTS[city]))
        rooms = int(r.choice([1, 2, 3, 4, 5], p=[0.25, 0.38, 0.25, 0.09, 0.03]))
        area = round(max(18.0, r.normal(22 + rooms * 20, 7)), 1)
        btype = r.choice(list(BUILDING), p=[0.35, 0.2, 0.35, 0.1])
        year = int(r.randint(2005, 2027) if btype == "монолитный" else r.randint(1960, 2027))
        floors_total = int(r.choice([5, 9, 12, 16, 20]))
        floor = int(r.randint(1, floors_total + 1))
        cond = r.choice(list(CONDITION))

        ppm = (DISTRICTS[city][district] * BUILDING[btype] * CONDITION[cond]
               * (0.93 if floor == 1 else 1.0) * (0.95 if floor == floors_total else 1.0)
               * (1 + (year - 1990) * 0.003) * r.lognormal(0, 0.10))
        price = round(ppm * area, -5)
        style = ", ".join(r.choice(STYLE[cond], 2, replace=False))
        rows.append({
            "listing_id": f"m{i:05d}", "deal_type": "sale",
            "price": f"{int(price):,}".replace(",", " ") + " 〒",   # как на сайте — строкой
            "area": area, "rooms": rooms, "city": city.capitalize(),
            "district": f"{district} р-н", "floor": floor, "floors_total": floors_total,
            "year_built": year, "building_type": btype, "condition": cond,
            "kitchen_area": round(area * r.uniform(0.12, 0.2), 1),
            "description": (f"Продаётся {rooms}-комнатная квартира {area} м², этаж {floor}/{floors_total}, "
                            f"{btype} дом {year} г. {style.capitalize()}."),
        })
    df = pd.DataFrame(rows)
    junk = df.sample(3, random_state=seed).copy()
    junk["listing_id"] = ["junk0", "junk1", "junk2"]   # свои id, иначе уйдут как дубли и чистка не проверится
    junk.loc[junk.index[0], "price"] = "1 〒"
    junk.loc[junk.index[1], "area"] = 0
    junk.loc[junk.index[2], "price"] = "договорная"
    return pd.concat([df, junk, df.iloc[[0]]], ignore_index=True)  # + дубль id


def make_placeholder_photos(df, seed=0):
    """Заглушки: цветной фон + подпись. Только чтобы прогнать обвязку —
    эмбеддинги у них бессмысленные, качество поиска по ним не оценивать."""
    r = np.random.RandomState(seed)
    rooms = ["кухня", "гостиная", "спальня", "санузел", "фасад"]
    for row in df.drop_duplicates("listing_id").itertuples():
        d = PHOTOS_DIR / row.listing_id
        d.mkdir(parents=True, exist_ok=True)
        for n in range(r.randint(2, 6)):
            im = Image.new("RGB", (640, 480), MOCK_COLORS.get(row.condition, (128, 128, 128)))
            ImageDraw.Draw(im).text((20, 20), f"MOCK {rooms[n % len(rooms)]} {row.listing_id}", fill=(255, 0, 0))
            im.save(d / f"{n}.jpg", quality=85)


def main(n=800, photos=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = make_listings(n)
    df.to_parquet(LISTINGS_RAW, index=False)
    print(f"мок: {len(df)} строк -> {LISTINGS_RAW}")
    if photos:
        make_placeholder_photos(df)
        print(f"фото-заглушки -> {PHOTOS_DIR}")
