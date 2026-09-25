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
  `details --refresh-missing` revisits incomplete rows once per parser version;
  `--cached-only` reparses saved successful detail HTML without network requests.
  Photos are merged by URL to preserve existing photo/embedding identities.
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
python -m scripts.backfill_details [--limit N]        # refresh + dedup + resync + smoke
python -m krisha photos  [--limit N]                   # httpx + sha256 dedup
python -m krisha dedup                                  # assign duplicate_group_id
python -m krisha embed   [--limit N] [--batch-size N]  # Phase B (needs Qdrant + torch)
python -m krisha stats
python -m krisha export  --csv [--out path] [--no-dedup]
```
Typical flow: `crawl` → `details` → `photos` → `dedup` → (`embed`) → `export`.

## Code conventions
- Python 3.11+, standard library `argparse`, `sqlite3`; type hints; small modules.
- Parsers must **never raise** on missing/odd fields — normalize to `None` and log.
- Selectors and the `data-name -> column` map live only in `parsers/selectors.py`.
- Logging via `logging_setup.get_logger(name)`; no secrets in logs.

## Scraping rules (updated by owner, 2026-09-23)
- Owner authorizes proxies, IP rotation, and VPN for detail backfill. Configure
  private proxy endpoints via `KRISHA_PROXIES_FILE` (JSON in ignored `data/`).
  Playwright rotates configured routes after block cooldown within MAX_RETRIES.
  This machine uses WARP local proxy with opt-in reconnects after 20 requests and
  on block retries; settings are private in `.env`. See `docs/detail_backfill.md`.
  Never log credentials. No fingerprint spoofing or CAPTCHA solving.
- **Max concurrency = 2** (enforced in `config.MAX_CONCURRENCY`; current profile
  runs sequentially = 1, which is polite and stable). Do not raise for speed.
- Respect robots.txt: never request `/ajax/`, `/captcha`, `/a/show-map/`, `/map/`,
  or any URL with `raion=`.
- On block, slow down and honor cooldown before retrying a configured route.
  Stop on CAPTCHA or exhausted retries; do not loop indefinitely through routes.

## How to add a new listing field
1. Add the column to `krisha/db/schema.sql` (nullable) and to
   `db/database.py::LISTING_FIELDS`.
2. If it comes from a detail characteristic, add its `data-name -> column` to
   `parsers/selectors.py::PARAM_MAP` and handle any parsing in
   `parsers/listing_detail.py`.
3. Add it to `commands/export.py::COLUMNS` if it should reach the ML CSV.
4. Add/adjust a fixture assertion in `tests/test_parse_detail.py`.

## Known limitations / status
- **2026-09-23 handoff**: background `scripts.auto_backfill` processes batches of 25
  with photo downloads, new embeddings, dedup, incremental Qdrant sync and latest
  CSV export. See `data/auto_backfill_status.json` for live counts/PID; logs in
  `logs/auto_backfill.*.log`. At launch: 3106 detail rows, 6104 first fetches pending,
  3083 older rows needing refresh; bathroom 19, balcony/loggia 14. The first full
  resync updated 22,045 points; subsequent audit 0 stale; oblast smoke PASS.
  Do not report full backfill complete until the live job confirms it.
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
- **Current data (2026-09-22)**: Almaty city + **whole Almaty oblast** + Kaskelen
  crawled — **9210 listings** (`almaty` 7805, `almaty_oblast` 1137, `kaskelen` 268).
  Full albums recovered via CDN index-walk (`photos --expand`): **85,611 photos
  downloaded** (1 dead 404 URL), **all embedded** into Qdrant `listing_photos`
  (**84,169 unique points** after sha256 dedup). Deduped to **8966 groups** (203 with
  >1 member); `export --csv` emits **8966 rows**. **Details fetched for 3084 listings**
  (Playwright, clean IP): `lat/lon` (3084), `area_kitchen` (1510), `complex_name` (1607).
  **Descriptions: 9144/9210 (99.3%)** parsed — the NULLs are card-only rows whose search
  card had no snippet (expected).
- **Qdrant is hosted on Qdrant Cloud** (set `QDRANT_URL` + `QDRANT_API_KEY` in `.env`;
  `make_client()` in `embed_worker.py` reads both — empty key ⇒ local Docker). Vectors
  were migrated local→cloud (no re-encode) via `python -m scripts.migrate_qdrant`.
- **Phase B validated (against cloud)**: `python -m scripts.smoke_similar --city <c>`
  encodes one photo, searches Qdrant unfiltered + filtered-by-city, and asserts the
  `city` filter leaks nothing (PASS). Confirms the vector index + KEYWORD `city` payload.
- **Detail-field diagnosis (2026-09-23)**: live pages fetched through Cloudflare
  WARP local proxy confirm `separated_toilet`, `balcony_count`, `loggia_count` in
  `.offer__parameters dl` were missed. Parser now reads both summary and definition
  list layouts, with old aliases retained. `house.year` and `flat.building` still
  work on a freshly fetched `673910188` (1975, brick); the first 21 new pages omit
  them entirely. Do not fabricate year/type values for listings without them.
  See `docs/detail_backfill.md` and ignored `data/detail_diagnostics/` for evidence.
- **City is canonicalized** to lowercase ASCII keys (`almaty`, `kaskelen`) in
  `parsers/normalize.py::normalize_city`, applied at the DB layer (`upsert_stub` +
  `update_detail`). This fixed a split where crawl stored the CLI arg (`almaty`) and
  detail overwrote it from JSON (`Almaty`), breaking the Qdrant `city` filter. Existing
  DB rows were migrated. **Payload resync**: after any crawl/embed/dedup run
  `python -m scripts.resync_qdrant_payload` — it patches all mutable listing fields,
  including `city`, `duplicate_group_id`, price, area and coordinates from the DB
  (dedup runs after embed, so fresh points
  carry a NULL group until resynced). Idempotent. (`scripts/reindex_qdrant_city.py` is
  the older city-only version, superseded by the resync script.)
- `crawl`, `photos`, `dedup`, `stats`, `export` and all parsing are validated;
  parsers are covered by pytest (12 tests) against real saved fixtures.
- Complex name (`complex_name`) is only populated when present in the params.

## TODO
- [x] Scale `details` to ≥300 unique monthly listings with photos — **done: 3084**.
- [x] Phase B embed — **done: all 85,611 photos embedded** into Qdrant `listing_photos`.
- [x] Phase B smoke query — **done: `scripts/smoke_similar.py`**, city filter airtight.
- [x] Move Qdrant to cloud — **done**: `make_client()` + `scripts/migrate_qdrant.py`.
- [x] Crawl whole Almaty oblast + city — **done: 9210 listings** (`almaty_oblast` path).
- [x] Diagnose missing fields against live pages and fix definition-list parsing
      plus confirmed current aliases; add regression tests. Year/type use existing
      correct selectors and remain nullable when the source does not provide them.
- [ ] Finish `details --refresh-missing` for the full dataset (resumable by parser
      version), followed by dedup / full payload resync / oblast smoke test.
- [ ] Backfill remaining detail pages: 6104 pending at the 2026-09-23 handoff;
      consult `data/auto_backfill_status.json` for the current queue. Idempotent.
- [ ] Optional: parse `complexId -> complex_name` via the complexes endpoint.
