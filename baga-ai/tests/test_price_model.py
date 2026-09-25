"""Регрессия модели цены по закоммиченным out-of-fold предсказаниям.

Если кто-то переобучит модель или поменяет признаки и закоммитит артефакты, CI сравнит
метрики с порогами. Значения на момент сдачи: MAPE 16.4%, покрытие 10–90 81.1%.
"""
import pandas as pd
import pytest

from config import ART_DIR


@pytest.fixture(scope="module")
def oof():
    listings = pd.read_parquet(ART_DIR / "listings.parquet", columns=["listing_id", "price"])
    return listings.merge(pd.read_parquet(ART_DIR / "price_oof.parquet"), on="listing_id")


def test_every_listing_has_oof_prediction(oof):
    assert len(oof) == len(pd.read_parquet(ART_DIR / "listings.parquet", columns=["listing_id"]))


def test_mape_beats_baseline(oof):
    mape = ((oof.p50 - oof.price).abs() / oof.price).mean()
    assert mape < 0.18, f"MAPE {mape:.1%}: хуже, чем при сдаче (16.4%), бейзлайн 22.3%"


def test_interval_is_calibrated(oof):
    """Интервал 10–90 должен накрывать ~80% цен. Меньше — нормальные квартиры получат
    вердикт «дороже похожих»; больше — интервал бесполезно широкий."""
    cover = ((oof.price >= oof.p10) & (oof.price <= oof.p90)).mean()
    assert 0.76 <= cover <= 0.86, f"покрытие {cover:.1%}"
