# krisha.kz — site notes (Phase 0 reconnaissance)

Date: 2026-09-21. Section: **АРЕНДА квартир, помесячно** (monthly apartment rent).
Region: Kaskelen + Almaty (all districts).

## robots.txt (`User-agent: *`)
- **Allowed**: `/arenda/kvartiry/...` (search), `/a/show/{id}` (detail).
- **Disallowed** (we avoid): `/ajax/`, `/captcha`, `/a/show-map/`, `/map/...`,
  `/*raion=` (district query param), `/comments*`, tracking params
  (`utm_source`, `gclid`, …). No `Crawl-delay`. `Sitemap: https://krisha.kz/sitemap.xml`.
- Consequence: we crawl the **whole-city path** and never filter by district via
  `raion=`. "All Almaty districts" is the default city page.

## URL templates & pagination
- Monthly rent search: `https://krisha.kz/arenda/kvartiry/{city}/`
  - Kaskelen: `/arenda/kvartiry/kaskelen/` — `nbTotal` ≈ **268**.
  - Almaty:   `/arenda/kvartiry/almaty/`  — `nbTotal` ≈ **7261**.
- Pagination: `?page=N`.
- Detail: `https://krisha.kz/a/show/{id}`.
- Daily rent ("Посуточно") is a **separate section** — we do NOT crawl it. Everything
  from the monthly path is stored `rent_period='month'`.

## Data sources (both pages are server-rendered + carry embedded JSON)
- **List page**: `window.data.search` = `{ currentPage, nbTotal, ids[20], regionId, ... }`.
  `ids` is the authoritative 20-per-page result set (excludes promoted "hot" cards
  that also render as `div.a-card[data-id]`). Card DOM: `.a-card__title` (link+title),
  `.a-card__price`, `.a-card__subtitle` (address), `.a-card__descr`.
  There is **no** per-advert array in list `window.data` — full fields come from detail.
- **Detail page**: `window.data.advert` = rich object:
  `id, price, rooms, square, title, addressTitle, photos[]{src,w,h}, map{lat,lon},
  address{district,country,city,street,house_num}, complexId, sectionAlias(=arenda),
  categoryAlias(=kvartiry), userType, status`.
  Full-size photo src ends `.../{n}-full.jpg`.
  Characteristics rendered as `div.offer__info-item[data-name=...]`:
  `flat.building`(Тип дома), `house.year`(Год постройки), `flat.floor`("1 из 5"),
  `live.square`("74 м², Площадь кухни — 9 м²"), `flat.renovation`, `flat.rent_renovation`,
  `flat.toilet`. Description in `.offer__description` (leads with an "О квартире" heading).
  Published date appears as `"createdAt":"YYYY-MM-DD"` in the embedded JSON.
  `digitalData` is analytics only — ignored.

## Photos
- Host: `https://krisha-photos.kcdn.online/...` — a **separate CDN**, NOT anti-bot
  protected (plain httpx GET returns 200). Downloaded via the httpx engine.

## Blocking / anti-bot (observed)
- Homepage and **list/search pages** load fine over plain httpx (200) once the
  homepage warmup sets cookies `krssid, krishauid, kraid`.
- **Detail pages `/a/show/` are protected**: they return custom status **HTTP 468**
  ("blocked") to automated clients when the source IP is even mildly flagged. During
  recon a fresh Chromium (Playwright) fetched details with 200, but after cumulative
  requests from one IP the 468 applied to Playwright too — i.e. it is **IP-reputation /
  rate based**, not defeated by headers or cookies alone.
- No CAPTCHA was surfaced in this recon; markers to watch: `/captcha`, `g-recaptcha`,
  `captcha__`, "Доступ ограничен".

## Consequences baked into the parser
- **List → httpx engine; Detail → Playwright engine** (single `Fetcher` interface).
- `468/403/429` are treated as blocks → AIMD doubles delay, concurrency→1, cooldown,
  honor `Retry-After`, and stop after `MAX_RETRIES`. CAPTCHA → stop run, never solve.
- Detail scraping must be **slow and patient**; run it when the IP is not greylisted.
  Owner authorized proxies / IP rotation / VPN on 2026-09-23. Explicit proxy
  configuration is supported; pacing, bounded retries and CAPTCHA stop remain.
