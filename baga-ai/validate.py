"""Проверки качества. Три разных вопроса — три разных теста.

robust  — насколько вектор фото сдвигается от качества, света, кадра и «ИИ-глянца».
          Идея псевдо-query из museum: портим реальное фото, ответ известен по
          построению, Hit@K по каждому типу искажения отдельно.
          ОГРАНИЧЕНИЕ: это мерит чувствительность к виду снимка, а не полный разрыв
          «сгенерированная комната vs реальная». Тот мерит только ручная разметка (review).
          Как и в museum: синтетика проще реальности, доверять порядку, а не числам.

review  — HTML со топ-K по вашим запросам (текст и ИИ-картинки) + CSV, где руками
          ставите 1/0 в колонке relevant. score считает precision@K.

rooms   — точность zero-shot классификатора комнат по ручной разметке.
"""
import html
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from config import RENDER_THR, SEED
from embed import embed_images, load_embeddings, open_rgb
from rooms import INTERIOR, load_rooms
from search import get_photo_search


# ---------------------------------------------------------------- robust

def _fill(im):
    a = np.array(im)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0)[:3])


def d_identity(im, r):
    return im   # контроль: Hit должен быть ~1.0, иначе сломан сам пайплайн


def d_blur(im, r):
    return im.filter(ImageFilter.GaussianBlur(r.uniform(1.0, 2.5)))


def d_lowres(im, r):
    w, h = im.size
    k = r.uniform(0.2, 0.4)
    return im.resize((max(32, int(w * k)), max(32, int(h * k))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def d_dark(im, r):
    """вечернее фото на телефон"""
    return ImageEnhance.Brightness(im).enhance(r.uniform(0.45, 0.7))


def d_crop(im, r):
    """другой кадр той же комнаты — грубая замена другому ракурсу"""
    w, h = im.size
    k = r.uniform(0.6, 0.85)
    cw, ch = int(w * k), int(h * k)
    x, y = r.randint(0, w - cw + 1), r.randint(0, h - ch + 1)
    return im.crop((x, y, x + cw, y + ch))


def d_perspective(im, r):
    w, h = im.size
    d = r.uniform(0.06, 0.18)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * d * r.uniform(0, 1), h * d * r.uniform(0, 1)],
                      [w * (1 - d * r.uniform(0, 1)), h * d * r.uniform(0, 1)],
                      [w * (1 - d * r.uniform(0, 1)), h * (1 - d * r.uniform(0, 1))],
                      [w * d * r.uniform(0, 1), h * (1 - d * r.uniform(0, 1))]])
    a = cv2.warpPerspective(np.array(im), cv2.getPerspectiveTransform(src, dst), (w, h),
                            borderValue=_fill(im))
    return Image.fromarray(a)


def d_polish(im, r):
    """«ИИ-глянец»: гладко, контрастно, насыщенно, светло — как у сгенерированных картинок"""
    im = im.filter(ImageFilter.SMOOTH_MORE)
    im = ImageEnhance.Contrast(im).enhance(r.uniform(1.15, 1.35))
    im = ImageEnhance.Color(im).enhance(r.uniform(1.2, 1.5))
    im = ImageEnhance.Brightness(im).enhance(r.uniform(1.05, 1.2))
    return ImageEnhance.Sharpness(im).enhance(1.8)


DISTORTIONS = {"identity": d_identity, "blur": d_blur, "lowres": d_lowres, "dark": d_dark,
               "crop": d_crop, "perspective": d_perspective, "polish": d_polish}


def robust(n_val=300, k=5, batch=64):
    idx, E, valid = load_embeddings()
    rooms = load_rooms()
    pool = np.flatnonzero(valid & rooms.room_type.isin(INTERIOR).to_numpy()
                          & (rooms.render_prob.to_numpy() < RENDER_THR))
    pick = np.random.RandomState(SEED).choice(pool, min(n_val, len(pool)), replace=False)
    listing = idx["listing_id"].to_numpy()
    originals = [open_rgb(idx.abs_path[i], max_side=1024) for i in pick]

    rows = {}
    for name, fn in DISTORTIONS.items():
        ims = [fn(im, np.random.RandomState(SEED + j)) for j, im in enumerate(originals)]
        Q = np.concatenate([embed_images(ims[b:b + batch]) for b in range(0, len(ims), batch)])
        S = Q @ E.T
        S[:, ~valid] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        rows[name] = {
            "cos_к_оригиналу": float((Q * E[pick]).sum(1).mean()),
            f"hit@{k}_фото": float(np.mean([p in t for p, t in zip(pick, top)])),
            f"hit@{k}_квартира": float(np.mean([listing[p] in set(listing[t]) for p, t in zip(pick, top)])),
        }
        print(f"{name:12s} {rows[name]}")

    table = pd.DataFrame(rows).T.round(3)
    print("\n", table.to_string())
    print("\ncos < ~0.85 у polish/dark -> стиль съёмки сильно сдвигает вектор; тогда пробуйте "
          "VIEWS=full,center или модель so400m и сравнивайте ПОРЯДОК строк, не абсолют.")
    return table


# ---------------------------------------------------------------- review / score

def _contact_sheet(blocks, out_html, title):
    """blocks: [(заголовок, [(путь, подпись), ...]), ...] -> HTML-страница с превью."""
    parts = [f"<meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font-family:sans-serif;margin:16px}.row{display:flex;gap:8px;flex-wrap:wrap}"
             ".c{width:220px;font-size:12px}.c img{width:220px;height:165px;object-fit:cover}</style>"]
    for head, items in blocks:
        parts.append(f"<h3>{html.escape(head)}</h3><div class='row'>")
        for path, cap in items:
            parts.append(f"<div class='c'><img src='{Path(path).resolve().as_uri()}'>"
                         f"<div>{html.escape(cap)}</div></div>")
        parts.append("</div>")
    Path(out_html).write_text("\n".join(parts), encoding="utf-8")


def make_review(queries_csv, out_dir="review", k=5):
    """queries_csv: qid,text,image,rooms. Заполняется text ИЛИ image (путь к ИИ-картинке);
    rooms необязательно, через ';'. На выходе review.html (смотреть) и review.csv (размечать)."""
    qs = pd.read_csv(queries_csv, dtype=str).fillna("")
    se = get_photo_search()
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    rows, blocks = [], []

    for q in qs.itertuples():
        rooms = q.rooms.split(";") if q.rooms else None
        if q.image:
            res = se.search_by_image(q.image, rooms=rooms or "auto", k=k)
            qtype, qtext = "image", q.image
        else:
            res = se.search_by_text(q.text, rooms=rooms, k=k)
            qtype, qtext = "text", q.text

        items = [(q.image, "ЗАПРОС")] if q.image else []
        for rank, r in enumerate(res["results"], 1):
            ph = r["photos"][0]
            rows.append({"qid": q.qid, "type": qtype, "query": qtext, "rank": rank,
                         "listing_id": r["listing_id"], "score": round(r["score"], 4),
                         "photo": ph["path"], "relevant": ""})
            items.append((ph["path"], f"#{rank} {r['listing_id']} · {ph['room_type']} · {r['score']:.3f}"))
        head = f"[{q.qid}] {qtext}" + (f"   (комната: {res['query_room']})" if res["query_room"] else "")
        blocks.append((head, items))

    pd.DataFrame(rows).to_csv(out / "review.csv", index=False)
    _contact_sheet(blocks, out / "review.html", "review")
    print(f"откройте {out / 'review.html'}, поставьте 1/0 в {out / 'review.csv'} -> validate.py score")


def score(review_csv, k=5):
    df = pd.read_csv(review_csv)
    if df["relevant"].isna().any():
        print(f"не размечено строк: {df['relevant'].isna().sum()} — они не учитываются")
        df = df.dropna(subset=["relevant"])
    df = df[df["rank"] <= k]
    per_q = df.groupby(["type", "qid"])["relevant"].mean()
    print(f"precision@{k} по типам запросов:")
    print(per_q.groupby("type").agg(["mean", "count"]).round(3).to_string())
    print(f"\nв целом: {per_q.mean():.3f}")
    print("\nразница text vs image = цена разрыва «сгенерированное фото -> реальное»")
    return per_q


# ---------------------------------------------------------------- rooms

def sample_rooms(n=200, out_dir="review"):
    """Случайные фото с предсказанием -> rooms_todo.csv (заполнить true_room) + HTML для просмотра."""
    rooms = load_rooms()
    s = rooms[rooms.room_type != "invalid"].sample(min(n, len(rooms)), random_state=SEED)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    s.assign(true_room="")[["path", "room_type", "room_conf", "render_prob", "true_room"]] \
        .to_csv(out / "rooms_todo.csv", index=False)
    _contact_sheet([("предсказания", [(p, f"{r} {c:.2f} render={pr:.2f}") for p, r, c, pr in
                                      s[["path", "room_type", "room_conf", "render_prob"]].itertuples(index=False)])],
                   out / "rooms.html", "rooms")
    print(f"заполните true_room в {out / 'rooms_todo.csv'} (и 'render' для рендеров) -> validate.py rooms")


def rooms_accuracy(labeled_csv):
    df = pd.read_csv(labeled_csv).dropna(subset=["true_room"])
    print(f"точность: {(df.room_type == df.true_room).mean():.3f} на {len(df)} фото\n")
    print(pd.crosstab(df.true_room, df.room_type).to_string())
