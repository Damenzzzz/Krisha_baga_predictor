"""Поиск объявлений: жёсткие условия + ранжирование по смыслу.

    фильтры (город, комнаты, цена, площадь)  ──┐
    текст/картинка -> SigLIP 2 -> фото         ├─→ RRF (k=60) → топ-10
    текст -> ALEM text-1024 -> описания        ┘

Два бэкенда поиска по фото:
  - Qdrant (основной): фильтры выполняются внутри базы вместе с векторным поиском,
    группировка по объявлению — запросом query_points_groups;
  - numpy (запасной): если коллекция не залита, работает по матрице из artifacts.

Каналы включаются сами по наличию артефактов: нет фото — ищем по описаниям, нет
описаний — по фото, нет ничего — просто фильтруем.
"""
import numpy as np
import pandas as pd

from config import DESC_EMB_PATH, EMB_PATH, RENDER_THR, ROOM_AUTO_CONF, ROOMS_PATH, RRF_K
from listings import Filters, apply_filters, load_listings

CARD_FIELDS = ["price", "area", "rooms", "price_per_m2", "city", "district", "floor", "floors_total",
               "condition", "address", "url", "description"]


def resolve_districts(df: pd.DataFrame, substrings) -> list[str]:
    """LLM говорит «бостандык», в данных «бостандыкский р-н» — разворачиваем в точные значения."""
    if not substrings:
        return []
    known = df.district.dropna().astype(str).unique()
    return sorted({d for s in substrings for d in known if s.lower() in d})


class NumpyPhotoSearch:
    """Запасной бэкенд: матрица эмбеддингов в памяти."""

    def __init__(self):
        from embed import load_embeddings
        self.idx, self.E, self.valid = load_embeddings()
        self.has_rooms = ROOMS_PATH.exists()
        if self.has_rooms:
            from rooms import load_rooms
            r = load_rooms()
            assert len(r) == len(self.idx), "photo_rooms.parquet устарел"
            self.room, self.render = r.room_type.to_numpy(), r.render_prob.to_numpy()
        else:
            self.room, self.render = np.full(len(self.idx), "unknown"), np.zeros(len(self.idx))
        self.listing = self.idx.listing_id.to_numpy()
        self.paths = self.idx.abs_path.to_numpy()

    def rank(self, q, rooms=None, listing_ids=None, allow_renders=False, k=10, photos_per_listing=3, **_):
        mask = self.valid.copy()
        if rooms and self.has_rooms:
            mask &= np.isin(self.room, list(rooms))
        if not allow_renders and self.has_rooms:
            mask &= self.render < RENDER_THR
        if listing_ids is not None:
            mask &= np.isin(self.listing, [str(x) for x in listing_ids])
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return []
        s = self.E @ q
        n = min(cand.size, k * 50)      # с запасом: у квартиры до 20 фото
        best = cand[np.argpartition(-s[cand], n - 1)[:n]]
        out = {}
        for r in best[np.argsort(-s[best])]:
            lid = self.listing[r]
            if lid not in out:
                if len(out) == k:
                    continue
                out[lid] = {"listing_id": lid, "score": float(s[r]), "photos": []}
            if len(out[lid]["photos"]) < photos_per_listing:
                out[lid]["photos"].append({"path": self.paths[r], "room_type": self.room[r],
                                           "score": float(s[r])})
        return list(out.values())


class QdrantPhotoSearch:
    """Основной бэкенд: фильтры и группировка выполняются в Qdrant."""

    def __init__(self):
        import qdrant_store as Q
        self.Q = Q

    def rank(self, q, rooms=None, f=None, districts_exact=None, allow_renders=False,
             k=10, photos_per_listing=3, **_):
        return self.Q.search_photos(q, f=f, rooms=rooms, districts_exact=districts_exact,
                                    allow_renders=allow_renders, k=k,
                                    photos_per_listing=photos_per_listing)


def _plain(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


class ListingSearch:
    def __init__(self, backend: str | None = None):
        self.df = load_listings().set_index("listing_id", drop=False)
        self.backend = backend or self._pick_backend()
        self.photos = {"qdrant": QdrantPhotoSearch, "numpy": NumpyPhotoSearch,
                       "none": lambda: None}[self.backend]()
        self.desc = None
        if DESC_EMB_PATH.exists():
            try:
                import qdrant_store as Q
                self.desc = Q if Q.has_collection(Q.DESC_COLLECTION) else None
            except Exception:
                self.desc = None
        self.channels = [c for c, on in [("photo", self.photos), ("desc", self.desc)] if on]

    @staticmethod
    def _pick_backend() -> str:
        try:
            import qdrant_store as Q
            if Q.has_collection(Q.PHOTOS_COLLECTION):
                return "qdrant"
        except Exception:
            pass
        return "numpy" if EMB_PATH.exists() else "none"

    def search(self, text=None, filters: Filters | None = None, image=None, images=None,
               rooms=None, k=10, weights=None, candidates=200):
        """text — стиль («светлая кухня, скандинавский»), filters — жёсткие условия,
        image/images — одна или несколько картинок-запросов.

        Несколько фото НЕ усредняются в один вектор: среднее кухни и детской не похоже
        ни на кухню, ни на детскую. Вместо этого каждое фото — отдельный запрос со своим
        типом комнаты, а объявления ранжируются по слиянию этих запросов; отдельно
        считается покрытие — сколько из присланных фото объявление закрыло."""
        weights = {"photo": 1.0, "desc": 1.0, **(weights or {})}
        f = filters or Filters()
        districts_exact = resolve_districts(self.df, f.districts)
        cand = apply_filters(self.df, f)
        ids = cand.index.tolist()
        out = {"n_candidates": len(ids), "channels_used": [], "query_room": None,
               "backend": self.backend, "results": []}
        if not ids:
            return out

        queries = [im for im in (images or ([image] if image else [])) if im]
        rankings, photo_hits, coverage = {}, {}, {}
        if self.photos is not None and (text or queries):
            if queries:
                out["query_room"] = []
                for n, im in enumerate(queries):
                    q, detected = self._query_vector(None, im, rooms)
                    qrooms = ([detected["room"]] if detected and detected.get("used")
                              else (rooms if rooms and rooms != "auto" else None))
                    out["query_room"].append({"n": n, **(detected or {})})
                    hits = self.photos.rank(q, rooms=qrooms, f=f, districts_exact=districts_exact,
                                            listing_ids=ids, k=candidates)
                    hits = [h for h in hits if h["listing_id"] in cand.index]
                    rankings[f"photo{n}"] = [h["listing_id"] for h in hits]
                    for h in hits:
                        coverage.setdefault(h["listing_id"], []).append(
                            {"query_n": n, "room": (detected or {}).get("room"),
                             "score": h["score"], "photo": h["photos"][0]["path"] if h["photos"] else None})
                        prev = photo_hits.get(h["listing_id"])
                        if prev is None:
                            photo_hits[h["listing_id"]] = h
                        else:   # копим фото от всех запросов, без повторов
                            seen = {p["path"] for p in prev["photos"]}
                            prev["photos"] += [p for p in h["photos"] if p["path"] not in seen]
            else:
                q, detected = self._query_vector(text, None, rooms)
                out["query_room"] = detected
                hits = self.photos.rank(q, rooms=rooms, f=f, districts_exact=districts_exact,
                                        listing_ids=ids, k=candidates)
                hits = [h for h in hits if h["listing_id"] in cand.index]
                rankings["photo"] = [h["listing_id"] for h in hits]
                photo_hits = {h["listing_id"]: h for h in hits}
        if self.desc is not None and text:
            from text_embed import embed_texts as embed_desc_query
            qd = embed_desc_query([text])[0]
            hits = self.desc.search_descriptions(qd, f=f, districts_exact=districts_exact, k=candidates)
            rankings["desc"] = [lid for lid, _ in hits if lid in cand.index]
        out["channels_used"] = list(rankings)

        if rankings:
            score = {}
            for ch, lst in rankings.items():
                w = weights.get(ch, weights.get("photo", 1.0) if ch.startswith("photo") else 1.0)
                for rank, lid in enumerate(lst, 1):
                    score[lid] = score.get(lid, 0.0) + w / (RRF_K + rank)
            # сначала те, кто закрыл больше присланных фото: объявление с кухней И детской
            # важнее того, у кого идеально совпала только кухня
            n_queries = len(queries)
            order = sorted(score, key=lambda lid: (len(coverage.get(lid, [])) if n_queries > 1 else 0,
                                                   score[lid]), reverse=True)
        else:   # ни стиля, ни каналов — сверху дешевле за м²
            order, score = cand.sort_values("price_per_m2").index.tolist(), {}

        pos = {ch: {lid: i + 1 for i, lid in enumerate(lst)} for ch, lst in rankings.items()}
        for lid in order[:k]:
            row = self.df.loc[lid]
            out["results"].append({
                "listing_id": lid,
                "score": round(score.get(lid, 0.0), 5),
                "rank_by": {ch: p[lid] for ch, p in pos.items() if lid in p},
                "listing": {c: _plain(row[c]) for c in CARD_FIELDS if c in row.index},
                "photos": photo_hits.get(lid, {}).get("photos", []),
                "matched_queries": coverage.get(lid, []),
                "coverage": f"{len(coverage.get(lid, []))}/{len(queries)}" if len(queries) > 1 else None,
            })
        return out

    def _query_vector(self, text, image, rooms):
        """Вектор запроса: из текста или из картинки. Для картинки заодно определяем комнату —
        иначе сгенерированная гостиная найдёт кухню в тех же тонах."""
        from embed import embed_images, embed_texts
        if image is not None:
            q = embed_images([image])[0]
            if rooms == "auto" or rooms is None:
                from rooms import INTERIOR, room_of
                room, conf = room_of(q)
                return q, {"room": room, "conf": round(conf, 3),
                           "used": room in INTERIOR and conf >= ROOM_AUTO_CONF}
            return q, None
        return embed_texts([text])[0], None


_search = None


def get_search() -> ListingSearch:
    """Один экземпляр на процесс: индексы и модель грузятся один раз."""
    global _search
    if _search is None:
        _search = ListingSearch()
    return _search
