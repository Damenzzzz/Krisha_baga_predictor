-- Baga AI krisha.kz parser schema. All listing fields are nullable: the
-- parser logs a missing field rather than failing.

CREATE TABLE IF NOT EXISTS listings (
    id                 INTEGER PRIMARY KEY,        -- krisha advert id
    url                TEXT,
    deal_type          TEXT,                       -- 'rent'
    rent_period        TEXT,                       -- 'month' | 'day'
    price_kzt          INTEGER,
    rooms              INTEGER,
    area_total         REAL,
    area_living        REAL,
    area_kitchen       REAL,
    floor              INTEGER,
    floors_total       INTEGER,
    year_built         INTEGER,
    building_type      TEXT,
    complex_name       TEXT,
    city               TEXT,
    district           TEXT,
    address            TEXT,
    lat                REAL,
    lon                REAL,
    description        TEXT,
    furniture          TEXT,
    bathroom           TEXT,
    balcony            TEXT,
    renovation_text    TEXT,
    published_at       TEXT,
    photo_urls         TEXT,                       -- json array
    photos_count       INTEGER,
    raw_json           TEXT,                       -- json blob of window.data.advert
    detail_fetched_at  TEXT,                       -- set once card is enriched
    first_seen_at      TEXT,
    last_seen_at       TEXT,
    duplicate_group_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_listings_city ON listings(city);
CREATE INDEX IF NOT EXISTS idx_listings_dupgroup ON listings(duplicate_group_id);
CREATE INDEX IF NOT EXISTS idx_listings_detail ON listings(detail_fetched_at);

CREATE TABLE IF NOT EXISTS photos (
    listing_id     INTEGER NOT NULL,
    idx            INTEGER NOT NULL,
    url            TEXT,
    local_path     TEXT,
    sha256         TEXT,
    width          INTEGER,
    height         INTEGER,
    downloaded_at  TEXT,
    embedded_at    TEXT,
    PRIMARY KEY (listing_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_photos_sha ON photos(sha256);
CREATE INDEX IF NOT EXISTS idx_photos_embed ON photos(embedded_at);

-- Every request is logged here; the adaptive controller reads it back.
CREATE TABLE IF NOT EXISTS scrape_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,
    run_id       TEXT,
    url          TEXT,
    engine       TEXT,       -- http | playwright
    status_code  INTEGER,
    latency_ms   INTEGER,
    delay_ms     INTEGER,
    concurrency  INTEGER,
    outcome      TEXT        -- ok|http_429|http_403|captcha|timeout|empty_parse|parse_error
);

CREATE INDEX IF NOT EXISTS idx_events_run ON scrape_events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_outcome ON scrape_events(outcome);
