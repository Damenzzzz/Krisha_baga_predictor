"""Группы дублей из пар, найденных DINOv3 + геометрией (ноутбук notebooks/kaggle_dedup.ipynb).

Пары приходят с заниженным порогом: dupe_pairs.csv намеренно содержит и слабые
совпадения. Собирать группы по всем парам нельзя — объявления с типовым ремонтом
от застройщика цепляются в одну цепочку (в первом прогоне так слиплись 210 штук).
Поэтому в граф групп идут только рёбра с надёжной геометрией.
"""
from pathlib import Path

import pandas as pd

ART = Path("artifacts")
PAIRS = ART / "dupe_pairs.csv"
MIN_INLIERS = 40        # ниже начинается сцепление цепочками, выше и до 200 результат не меняется


def _components(edges) -> dict:
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return {x: find(x) for x in parent}


def rebuild(min_inliers: int = MIN_INLIERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(PAIRS, dtype={"a": str, "b": str})
    strong = pairs[pairs.inliers >= min_inliers]

    groups = pd.DataFrame(
        _components(zip(strong.a, strong.b)).items(), columns=["listing_id", "dup_group"]
    )

    both = pd.concat([strong[["a", "b"]], strong[["b", "a"]].rename(columns={"b": "a", "a": "b"})])
    trust = (both.groupby("a").b.nunique()
             .rename("shared_photo_listings").reset_index()
             .rename(columns={"a": "listing_id"}))

    sizes = groups.groupby("dup_group").size()
    if len(sizes) and sizes.max() > 20:
        raise RuntimeError(
            f"группа из {sizes.max()} объявлений — порог {min_inliers} слишком низкий, "
            "рёбра сцепляются цепочкой"
        )
    return groups, trust


def load_dupes() -> pd.DataFrame:
    p = ART / "listing_dupes.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["listing_id", "dup_group"])


def load_trust() -> dict:
    """listing_id -> в скольких ещё объявлениях встречаются эти фото."""
    p = ART / "listing_trust.parquet"
    if not p.exists():
        return {}
    t = pd.read_parquet(p)
    return dict(zip(t.listing_id.astype(str), t.shared_photo_listings.astype(int)))


if __name__ == "__main__":
    groups, trust = rebuild()
    groups.to_parquet(ART / "listing_dupes.parquet", index=False)
    trust.to_parquet(ART / "listing_trust.parquet", index=False)
    sizes = groups.groupby("dup_group").size()
    print(f"объявлений в группах: {len(groups)}, групп: {len(sizes)}, максимум в группе: {sizes.max()}")
    print(f"объявлений с чужими фото: {len(trust)}")
