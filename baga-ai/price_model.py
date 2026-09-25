"""Модель справедливой цены аренды.

Предсказываем не одно число, а ИНТЕРВАЛ (10-й, 50-й, 90-й перцентили): пользователю
нужно «похожие сдают за 380–470», а не «модель считает 427 381». Для этого квантильная
регрессия: функция потерь pinball штрафует недооценку и переоценку по-разному, и модель
выучивает границу, а не среднее.

Учим на log(price): у цен длинный правый хвост, а ошибка становится относительной
(10% от 300 и от 600 тысяч — разные деньги, в логарифме одинаковая величина).

Всегда сравниваем с тупым рыночным бейзлайном «медиана ₸/м² по району и комнатам».
Если CatBoost его не обгоняет — проблема в признаках, а не в модели.

Предсказания out-of-fold: каждая строка предсказана моделью, которая её не видела,
поэтому вердикт «дороже похожих» можно честно показать на всех объявлениях сразу.
"""
import json

import numpy as np
import pandas as pd

from features import CAT_COLS, FLAG_COLS, NUM_COLS, build_features
from listings import load_listings
from splits import add_folds

QUANTILES = (0.1, 0.5, 0.9)
DESC_NUM = ["kitchen_area_txt", "living_area_txt", "ceiling", "year_txt", "desc_len"]

# лестница наборов признаков: каждая ступень добавляет блок к предыдущей
FEATURE_SETS = {
    "catboost база":      [c for c in NUM_COLS if c not in DESC_NUM] + [c for c in CAT_COLS if c != "condition"],
    "+ ремонт":           [c for c in NUM_COLS if c not in DESC_NUM] + CAT_COLS,
    "+ признаки текста":  NUM_COLS + FLAG_COLS + CAT_COLS,
}


def baseline_quantiles(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Рыночный бейзлайн: перцентили цены за м² у похожих (район + комнаты) × площадь.
    Если группа мелкая, откатываемся на район, потом на город."""
    t = train.assign(ppm=train.price / train.area)
    keys = [["city", "district", "rooms"], ["city", "district"], ["city"]]
    tables = [t.groupby(k).ppm.quantile(list(QUANTILES)).unstack() for k in keys]
    out = np.full((len(test), len(QUANTILES)), np.nan)
    for k, tab in zip(keys, tables):
        idx = pd.MultiIndex.from_frame(test[k]) if len(k) > 1 else pd.Index(test[k[0]])
        vals = tab.reindex(idx).to_numpy()
        out = np.where(np.isnan(out), vals, out)
    out = np.where(np.isnan(out), np.nanmedian(t.ppm), out)
    return out * test.area.to_numpy()[:, None]


def fit_predict_catboost(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
                         X: pd.DataFrame, iterations=1200) -> np.ndarray:
    from catboost import CatBoostRegressor, Pool
    cats = [c for c in cols if c in CAT_COLS]
    xtr, xte = X.loc[train.index, cols], X.loc[test.index, cols]
    for c in cats:
        xtr[c], xte[c] = xtr[c].astype(str), xte[c].astype(str)
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={','.join(str(q) for q in QUANTILES)}",
        iterations=iterations, depth=6, learning_rate=0.05, verbose=0, random_seed=0)
    model.fit(Pool(xtr, np.log(train.price), cat_features=cats))
    return np.exp(model.predict(Pool(xte, cat_features=cats)))


def conformal_q_for_fold(train: pd.DataFrame, cols: list[str], X: pd.DataFrame,
                         calib_frac=0.25, seed=0) -> float:
    """Конформная калибровка интервала (CQR).

    Квантильная регрессия сама по себе даёт слишком узкий коридор: на наших данных
    покрытие 66% вместо обещанных 80%. CQR это чинит: часть обучающих данных
    откладывается, на ней смотрим, насколько сильно правда вылезает за границы, и
    расширяем коридор ровно на столько. В логарифме сдвиг границ — это умножение
    в тенге, поэтому дорогие квартиры получают пропорционально более широкий интервал.
    """
    calib = train.sample(frac=calib_frac, random_state=seed)
    fit = train.drop(calib.index)
    pc = np.sort(fit_predict_catboost(fit, calib, cols, X), axis=1)
    yy = np.log(calib.price.to_numpy())
    scores = np.maximum(np.log(pc[:, 0]) - yy, yy - np.log(pc[:, 2]))   # насколько правда вышла за коридор
    return float(np.quantile(scores, 0.8))                              # 80% — целевое покрытие


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """pred: (n, 3) — p10, p50, p90. Сортируем: три независимых квантиля иногда пересекаются."""
    pred = np.sort(pred, axis=1)
    p10, p50, p90 = pred.T
    return {
        "MAE, ₸": int(np.mean(np.abs(y - p50))),
        "MAPE, %": round(float(np.mean(np.abs(y - p50) / y) * 100), 1),
        "покрытие 10-90, %": round(float(np.mean((y >= p10) & (y <= p90)) * 100), 1),
        "ширина, % от цены": round(float(np.median((p90 - p10) / p50) * 100)),
    }


def run(n_splits=5) -> pd.DataFrame:
    df = add_folds(load_listings(), n_splits=n_splits)
    X = build_features(df)
    y = df.price.to_numpy()
    best_set = "+ признаки текста"
    q_by_fold = []
    names = ["бейзлайн: район × м²", *FEATURE_SETS, "+ конформная калибровка"]
    preds = {name: np.full((len(df), 3), np.nan) for name in names}

    for k in range(n_splits):
        tr, te = df[df.fold != k], df[df.fold == k]
        preds["бейзлайн: район × м²"][te.index] = baseline_quantiles(tr, te)
        for name, cols in FEATURE_SETS.items():
            preds[name][te.index] = fit_predict_catboost(tr, te, cols, X)
        q = conformal_q_for_fold(tr, FEATURE_SETS[best_set], X)
        q_by_fold.append(q)
        p = np.sort(fit_predict_catboost(tr, te, FEATURE_SETS[best_set], X), axis=1)
        preds["+ конформная калибровка"][te.index] = np.stack(
            [p[:, 0] * np.exp(-q), p[:, 1], p[:, 2] * np.exp(q)], axis=1)
        print(f"фолд {k + 1}/{n_splits} готов (расширение ×{np.exp(q):.2f})")

    table = pd.DataFrame({name: metrics(y, p) for name, p in preds.items()}).T
    print(f"\nout-of-fold на {len(df)} объявлениях, цель — месячная аренда\n")
    print(table.to_string())
    print("\nпокрытие должно быть ≈80%: больше — интервал слишком широкий, меньше — слишком узкий")

    final = np.sort(preds["+ конформная калибровка"], axis=1)
    seg = df.assign(p50=final[:, 1], hit=(y >= final[:, 0]) & (y <= final[:, 2]),
                    ape=np.abs(y - final[:, 1]) / y * 100)
    print("\nитоговая модель по сегментам:")
    print(seg.groupby("city").agg(объявлений=("price", "size"), MAPE=("ape", "mean"),
                                  покрытие=("hit", "mean")).round(2).to_string())
    seg["цена"] = pd.qcut(seg.price, 4, labels=["дешёвые", "ниже медианы", "выше медианы", "дорогие"])
    print(seg.groupby("цена", observed=True).agg(объявлений=("price", "size"), MAPE=("ape", "mean"),
                                                 покрытие=("hit", "mean")).round(2).to_string())
    # Сохраняем out-of-fold предсказания: для объявлений, что уже в базе, только они честные —
    # обученная на всех данных модель их «видела» и занижает ошибку. Модель + поправка нужны
    # для НОВЫХ объявлений (сайт, MCP).
    from config import ART_DIR
    oof = pd.DataFrame({"listing_id": df.listing_id.to_numpy(),
                        "p10": final[:, 0], "p50": final[:, 1], "p90": final[:, 2]})
    oof.to_parquet(ART_DIR / "price_oof.parquet", index=False)

    model_full, cats_full = None, None
    from catboost import CatBoostRegressor, Pool
    cols = FEATURE_SETS[best_set]
    cats = [c for c in cols if c in CAT_COLS]
    xall = X[cols].copy()
    for c in cats:
        xall[c] = xall[c].astype(str)
    model_full = CatBoostRegressor(
        loss_function="MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES),
        iterations=1200, depth=6, learning_rate=0.05, verbose=0, random_seed=0)
    model_full.fit(Pool(xall, np.log(y), cat_features=cats))
    model_full.save_model(str(ART_DIR / "price_model.cbm"))
    (ART_DIR / "price_model.json").write_text(json.dumps({
        "features": cols, "cat_features": cats, "quantiles": list(QUANTILES),
        "conformal_q": float(np.mean(q_by_fold)), "target": "log(price), ₸ в месяц",
        "metrics_oof": metrics(y, preds["+ конформная калибровка"]), "n_train": int(len(df)),
    }, ensure_ascii=False, indent=2))
    print(f"\nсохранено: price_model.cbm, price_model.json, price_oof.parquet ({len(oof)} строк)")
    return table


if __name__ == "__main__":
    run()
