# Baga AI — krisha.kz Rent Parser (AGENTS.md)

Source of truth for any agent (Claude, Codex, …) continuing this project.

## Purpose
Baga AI estimates the fair monthly rent of an apartment from its photos +
description, so a tenant can tell whether they are overpaying. This repo is the
**data pipeline**: it scrapes monthly apartment-rent listings from krisha.kz for
**Kaskelen + Almaty**, stores structured fields + photos in SQLite, and embeds
photos into Qdrant for similarity search. It is intentionally a *working, stable*
scraper, not a perfect one.

## Directory map
```
KrishaParser/
  krisha/                  # the package (run as `python -m krisha`)
    __main__.py            # argparse CLI dispatch
    config.py              # paths, HARD scraping limits, URL builders, headers
    logging_setup.py       # file + console logging (logs/krisha.log)
    controller.py          # AIMD self-tuning controller + runtime profile
    fetcher/
      base.py              # Fetcher ABC + FetchResult (ok/is_block/is_captcha)
      http_engine.py       # httpx.Client engine (list pages + photos) — PRIMARY
      pw_engine.py         # Playwright Chromium engine (detail pages) — FALLBACK
      cache.py             # raw-response disk cache keyed by sha1(url)
    parsers/
      window_data.py       # extract embedded `window.data` JSON
      listing_list.py      # search page -> ids + pagination + light cards
      listing_detail.py    # window.data.advert + offer__info-item -> fields
      selectors.py         # centralized selectors + data-name -> column map
      normalize.py         # numeric/text normalizers (never raise)
      dom.py               # selectolax fast-path, bs4 fallback
    db/
      schema.sql           # listings, photos, scrape_events (+ indexes)
      database.py          # Database: upserts, queries, event log
    commands/              # crawl / details / photos / embed / stats / export
    dedup.py               # near-duplicate grouping (union-find)
    photos.py              # photo download + sha256 dedup
    embed_worker.py        # open_clip -> Qdrant (Phase B, lazy heavy imports)
  config/runtime_profile.json  # best stable {engine, delay_ms, concurrency}
  docs/site_notes.md       # Phase-0 reconnaissance (read this!)
  tests/                   # pytest + saved HTML/JSON fixtures
  data/                    # gitignored: krisha.db, cache/, photos/, exports/
  requirements.txt  .env.example  README.md
docker-compose.yml         # qdrant service (repo root)
CLAUDE.md                  # `@AGENTS.md`
```

## Architecture
- **Two engines, one interface** (`fetcher/base.py::Fetcher`):
  `http` (httpx, primary) for list/search pages and photos; `playwright`
  (real Chromium) for detail pages. Detail pages return krisha's custom
  **HTTP 468** anti-bot status to plain HTTP clients, so they must go through
  Playwright. See `docs/site_notes.md`.
- **AIMD self-tuning controller** (`controller.py`) wraps every fetch:
  paces with a deterministic-jitter delay, logs each request to `scrape_events`,
  and adapts — success streak lowers the delay toward the floor; `429/403/468`
  doubles the delay, drops concurrency to 1, honors `Retry-After`, cools down,
  and stops after `MAX_RETRIES`; a CAPTCHA **stops the run** (never solved); a
  sustained `empty_parse` streak stops with a "layout changed?" error. The best
  profile is persisted to `config/runtime_profile.json` and reloaded next run.
- **Idempotent / resumable**: listing `id` is the PK; crawl only refreshes seen
  ids, `details` only fetches rows with `detail_fetched_at IS NULL`, photos only
  download rows without `local_path`, embed only photos with `embedded_at IS NULL`.
  Re-running never re-does finished work. A raw-response disk cache
  (`fetcher/cache.py`) is available for list pages.
- **Embedding worker** (`embed_worker.py`) is a separate step: open_clip on CPU
  (default `ViT-B-32`) → upsert into Qdrant collection `listing_photos`.

## Data model (SQLite, `data/krisha.db`)
- `listings` — one row per advert; **all fields nullable** (a missing field is
  logged, never fatal). Key columns: `id, url, deal_type, rent_period, price_kzt,
  rooms, area_total/living/kitchen, floor, floors_total, year_built, building_type,
  complex_name, city, district, address, lat, lon, description, furniture, bathroom,
  balcony, renovation_text, published_at, photo_urls(json), photos_count, raw_json,
  detail_fetched_at, first_seen_at, last_seen_at, duplicate_group_id`.
- `photos` — `(listing_id, idx)` PK; `url, local_path, sha256, width, height,
  downloaded_at, embedded_at`.
- `scrape_events` — `ts, run_id, url, engine, status_code, latency_ms, delay_ms,
  concurrency, outcome` where outcome ∈
  `ok|http_429|http_403|captcha|timeout|empty_parse|parse_error`.

### Qdrant payload (collection `listing_photos`)
point id = `uuid5(sha256)` (idempotent). Payload: `listing_id, photo_idx, city,
district, rooms, area_total, floor, price_kzt, price_per_m2, rent_period, lat, lon,
duplicate_group_id`. Payload indexes created for `city, district, rooms,
rent_period, price_kzt, duplicate_group_id`.

## Commands
```
python -m krisha crawl   --city kaskelen|almaty --deal rent [--max-pages N]
python -m krisha details [--city ...] [--limit N]      # Playwright
python -m krisha photos  [--limit N]                   # httpx + sha256 dedup
python -m krisha dedup                                  # assign duplicate_group_id
python -m krisha embed   [--limit N] [--batch-size N]  # Phase B (needs Qdrant + torch)
python -m krisha similar --image PATH [--city ...] [--limit N]  # Phase B similarity search
python -m krisha stats
python -m krisha export  --csv [--out path] [--no-dedup]
```
Typical flow: `crawl` → `details` → `photos` → `dedup` → (`embed`) → `export`.

## Code conventions
- Python 3.11+, standard library `argparse`, `sqlite3`; type hints; small modules.
- Parsers must **never raise** on missing/odd fields — normalize to `None` and log.
- Selectors and the `data-name -> column` map live only in `parsers/selectors.py`.
- Logging via `logging_setup.get_logger(name)`; no secrets in logs.

## Scraping rules — DO NOT CHANGE (hard limits from the brief)
- **No proxies / no residential rotation. No fingerprint spoofing. Never solve
  CAPTCHAs.** Realistic User-Agent + polite pace only.
- **Max concurrency = 2** (enforced in `config.MAX_CONCURRENCY`; current profile
  runs sequentially = 1, which is polite and stable). Do not raise for speed.
- Respect robots.txt: never request `/ajax/`, `/captcha`, `/a/show-map/`, `/map/`,
  or any URL with `raion=`.
- On block/CAPTCHA the correct response is to slow down / stop, not to evade.

## How to add a new listing field
1. Add the column to `krisha/db/schema.sql` (nullable) and to
   `db/database.py::LISTING_FIELDS`.
2. If it comes from a detail characteristic, add its `data-name -> column` to
   `parsers/selectors.py::PARAM_MAP` and handle any parsing in
   `parsers/listing_detail.py`.
3. Add it to `commands/export.py::COLUMNS` if it should reach the ML CSV.
4. Add/adjust a fixture assertion in `tests/test_parse_detail.py`.

## Known limitations / status
- **Detail fetching is IP-reputation limited.** krisha serves 468 on `/a/show/`
  once an IP is flagged; the Playwright engine bypasses it when the IP is clean
  but not when greylisted. Run `details` slowly and patiently; if it stops with
  `BlockedStop`, wait (minutes–hours) or use a clean network, then resume — it is
  idempotent.
- **Card-level fallback**: because detail pages are often blocked, `crawl` also
  persists what search cards expose — price, rooms, area_total, floor, address,
  and **1–2 full-size photo URLs** per listing (photos live on an unblocked CDN).
  This yields a usable dataset without detail pages. Fields only on the detail
  page (area_kitchen/living, year_built, building_type, lat/lon, full gallery,
  full description) stay NULL until `details` succeeds.
- **Current data (2026-09-22)**: Kaskelen + Almaty crawled — **7485 listings**,
  **7199 with photos**. Full albums recovered via CDN index-walk (`photos --expand`):
  **67,850 photos, all downloaded** (0 pending), avg **9.4 photos/listing** (max 40),
  ~66.8k files on disk. Deduped to **7287 groups**; `export --csv` emits **7287 rows**.
  Card-level fields present for ~all rows (price, rooms, area_total, floor,
  floors_total, address, description snippet). Detail-only fields
  (area_kitchen/living, year_built, building_type, lat/lon, full description,
  amenities) remain NULL — `/a/show/` is IP-blocked (468) from this network;
  `detail_fetched_at` count = 1 (fixture).
- `crawl`, `photos`, `dedup`, `stats`, `export` and all parsing are validated;
  parsers are covered by pytest (12 tests) against real saved fixtures.
- Complex name (`complex_name`) is only populated when present in the params.
- **Phase B is live.** Qdrant (v1.19, `docker compose up -d qdrant`) holds the
  `listing_photos` collection; `embed` encodes photos with open_clip
  **`ViT-B-32-quickgelu` + `openai`** (the matched pair — plain `ViT-B-32` mispairs
  the activation and degrades quality) and upserts (point id = `uuid5(sha256)`, so
  identical photos collapse — idempotent). `similar --image PATH [--city ...]`
  runs the smoke query: cosine search over the collection, optional city filter,
  prints ranked matches with price/rooms/url. Validated end to end (self-match
  score = 1.000, visually-similar neighbours ranked below, city filter honoured).
  Full 67,850-photo embed is a ~2–3 h CPU run; re-runnable and resumable
  (`embedded_at` gates it). Override the model via `CLIP_MODEL`/`CLIP_PRETRAINED`,
  Qdrant via `QDRANT_URL`, and the data dir via `KRISHA_DATA_DIR` (lets a git
  worktree run against the primary checkout's DB + photos).

## TODO
- [ ] Scale `details` to ≥300 unique monthly listings with photos (Kaskelen → Almaty)
      once the IP is not greylisted.
- [x] Phase B: Qdrant up, `embed` + `similar` smoke query working end to end
      (quickgelu/openai CLIP, city-filtered cosine search). Full 67.8k embed runs
      in the background; `stats` / collection `points_count` show progress.
- [ ] Optional: parse `complexId -> complex_name` via the complexes endpoint.
