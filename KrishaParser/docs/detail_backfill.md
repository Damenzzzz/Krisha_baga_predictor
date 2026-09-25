# Detail backfill and field diagnosis (2026-09-23)

## Connection

The owner authorized proxy rotation / VPN for this run. Cloudflare WARP was
installed from the official `Cloudflare.Warp` winget package (2026.7.1376.0).
It runs in **local proxy mode**, listening on `127.0.0.1:40000`; other applications
retain their normal connection. A request through it confirmed `warp=on`.

Local `.env` selects `KRISHA_PROXIES_FILE=data/warp-proxy.json`. That private JSON
file contains `[{"server":"http://127.0.0.1:40000"}]`. No credentials are committed.
More routes can be configured as a JSON list with optional `username` / `password`
keys. HTTP(S) and unauthenticated SOCKS5 endpoints are supported by the loader.
Rotation advances through the list after block cooldown, within MAX_RETRIES.
WARP itself does not promise a new IP for each request or connection.

This machine also sets `KRISHA_WARP_RECONNECT=1` and
`KRISHA_WARP_ROTATE_EVERY=20`. For this exact loopback endpoint the engine closes
the browser session, reconnects WARP, waits for a healthy connection and starts a
new session after 20 detail requests or a block cooldown. The reconnect helper
refuses to operate if WARP is in a system-wide tunnel mode. A blocked detail page
was successfully retrieved after this automatic reconnect.

An optional `python -m scripts.find_proxies --limit 20` checks HTTPS connectivity
of a bounded sample from [ProxyScrape's official public list](https://github.com/proxyscrape/free-proxy-list).
It saved 13 reachable candidates to ignored `data/public-proxies.json` during this
run. This does not establish Krisha compatibility; they are not enabled by default.

Manage the Windows client with
`C:\Program Files\Cloudflare\Cloudflare WARP\warp-cli.exe`:

```powershell
& 'C:/Program Files/Cloudflare/Cloudflare WARP/warp-cli.exe' status
& 'C:/Program Files/Cloudflare/Cloudflare WARP/warp-cli.exe' connect
& 'C:/Program Files/Cloudflare/Cloudflare WARP/warp-cli.exe' disconnect
```

Clear `KRISHA_PROXIES_FILE` to use direct access. An unavailable configured proxy
never silently falls back to direct access. CAPTCHA still stops the run.

Official reference: https://developers.cloudflare.com/warp-client/warp-modes/

## Evidence and parser changes

Live listing `1015959523` was fetched through WARP with HTTP 200. The HTML and full
characteristic dump are in ignored `data/detail_diagnostics/1015959523.{html,json}`.
The report was generated before adding the two new aliases, so its `column: null`
entries record the original parser gap.

| Meaning | Live data-name | Location | Stored field |
|---|---|---|---|
| Combined/separate toilet | `separated_toilet` | `.offer__parameters dl > dt/dd` | `bathroom` |
| Balcony count | `balcony_count` | `.offer__parameters dl > dt/dd` | `balcony` (`балкон: 1`, for example) |
| Loggia count (subsequent live pages) | `loggia_count` | `.offer__parameters dl > dt/dd` | `balcony` (`лоджия: 2`; combined if both present) |
| Bath/shower equipment | `bathroom` | `.offer__parameters dl > dt/dd` | Not mapped to the toilet-type column |
| Construction year | Absent on this page | — | Remains NULL |
| Building material/type | Absent on this page | — | Remains NULL |

The older saved listing `673910188`, also fetched live again through WARP, has `house.year=1975` and
`flat.building=кирпичный` in summary cards; `flat.toilet=раздельный` and
`flat.balcony=лоджия` are in definition lists. Previously the parser read only
summary cards. Both layouts are now read and both old and observed new names
are supported. Where old and new forms coexist, the new toilet value and labeled
balcony/loggia counts take precedence. Year/building names still work on the live
site; these characteristics were absent from the first 21 newly fetched listings.
No new year/building aliases are invented without source evidence.

`tests/fixtures/detail_rental_20260923_params.html` contains a minimal reconstruction
of the observed characteristic blocks, with personal data and scripts omitted.
Tests cover the old layout, new names, missing fields and empty duplicate values.

## Resume and finalize

For unattended operation that survives the chat session:

```powershell
python -m scripts.auto_backfill --batch-size 25 --fallback-proxies data/public-proxies-ranked.json
```

The background worker was started on 2026-09-23. It immediately publishes existing
progress, then processes batches of 25: details → pending photos → new embeddings
→ dedup → incremental Qdrant payload updates → `data/exports/listings_latest.csv`.
Incremental updates use an INTEGER index on the payload's owning `listing_id`,
so batches do not repeatedly scan all vectors or change shared-photo ownership.
The final audit is a full resync check plus the oblast smoke test.

- Live state (including PID, phase and remaining counts): `data/auto_backfill_status.json`.
- Output: `logs/auto_backfill.stdout.log`, `logs/auto_backfill.stderr.log`.
- Only one worker may hold `data/auto_backfill.lock` at a time.
- To stop after the current batch, create `data/STOP_AUTO_BACKFILL`.
- Exhausted proxy routes are not recycled; CAPTCHA and layout-change signals stop
  the job. `routes_exhausted` means new working routes are needed, not completion.
- HTTP failures/removed listings that do not trigger a stop are recorded as
  `unavailable_ids` at the end; rerunning retries them using the DB checkpoints.

At background handoff: 3,106/9,210 listings have detail pages; 6,104 still need
their first detail fetch, and 3,083 old detail rows need a parser refresh (9,187
queued in total). Bathroom: 19; balcony/loggia: 14. Year/building: 1 each, with
source absence confirmed on the other sampled pages. These are a snapshot;
the state file is authoritative while the worker runs.

```powershell
python -m scripts.backfill_details
# Or a bounded batch (also runs dedup / full payload resync / oblast smoke):
python -m scripts.backfill_details --limit 100
```

The wrapper runs `details --refresh-missing`, then `dedup`,
`scripts.resync_qdrant_payload`, and `scripts.smoke_similar --city almaty_oblast`.
Controlled block/CAPTCHA exits still trigger finalization of partial progress.

Individual commands:

```powershell
python -m scripts.inspect_detail 1015959523
python -m krisha details --refresh-missing
python -m krisha details --cached-only --refresh-missing
python -m krisha dedup
python -m scripts.resync_qdrant_payload
python -m scripts.resync_qdrant_payload --dry-run
python -m scripts.smoke_similar --city almaty_oblast
```

`detail_parser_version` is migrated additively. `--refresh-missing` includes older
incomplete detail rows once per parser version, alongside pending rows. A missing
field is not proof of failure: source pages often omit optional characteristics.
Successful HTML is cached for offline reparsing after future parser changes.

Photos are merged by URL, retaining existing photo indices, hashes and embedding
checkpoints even when the detail gallery is reordered. The `city` search bucket
is retained when a detail's address names a specific town in Almaty oblast.

Payload synchronization now covers all mutable fields in the existing payload,
including coordinates, price, area and derived price/m², with acknowledged batch
writes. It leaves vector data, photo indices and unrelated payload keys intact.
The initial dry-run found 21,825 stale points across 2,173 listings out of 84,169.
The first update, including the initial detail batch, changed 22,045 points. A
subsequent audit found zero stale points, and the oblast smoke test passed.
Float comparisons tolerate JSON serialization roundoff (about 1e-12) so resync
does not endlessly rewrite unchanged derived prices.
