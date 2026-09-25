# Baga AI — krisha.kz Rent Parser

Scrapes **monthly apartment-rent** listings from krisha.kz (Kaskelen + Almaty),
stores structured fields + photos in SQLite, and (Phase B) embeds photos into
Qdrant for similarity search. Full architecture and rules: see [AGENTS.md](AGENTS.md)
and [docs/site_notes.md](docs/site_notes.md).

## Quick start

```bash
# 1) install (Windows, Python 3.11+)
python -m pip install -r requirements.txt
python -m playwright install chromium          # detail pages need a real browser

cp .env.example .env                            # optional; sane defaults work

# 2) collect (start small, then scale)
python -m krisha crawl   --city kaskelen --deal rent --max-pages 2
python -m krisha details --city kaskelen --limit 20     # Playwright; slow & polite
python -m krisha photos  --limit 200                    # httpx + sha256 dedup
python -m krisha dedup                                   # near-duplicate groups
python -m krisha stats                                   # events + current profile
python -m krisha export  --csv                           # ML dataset -> data/exports/
```

Scale up by re-running `crawl` without `--max-pages`, then `details` (Kaskelen
first ≈268 listings, then `--city almaty` for volume). Every step is idempotent
and resumable.

## Phase B — photo embeddings (optional, heavier)

```bash
docker compose up -d qdrant                     # from repo root
python -m krisha embed --limit 500              # open_clip (ViT-B-32, CPU) -> Qdrant
```

## Tests

```bash
python -m pytest -q                             # parses real saved fixtures
```

## Rules (do not change)
No proxies, no fingerprint spoofing, never solve CAPTCHAs, max concurrency 2,
respect robots.txt. On a block the parser slows down / stops — it does not evade.
See AGENTS.md → "Scraping rules".

## Notes
- **List/search pages** are fetched with **httpx**; **detail pages** with
  **Playwright** (krisha serves an anti-bot HTTP 468 to plain clients on `/a/show/`).
- Detail fetching is IP-reputation sensitive: if `details` stops with `BlockedStop`,
  wait a while and resume — no data is lost.
