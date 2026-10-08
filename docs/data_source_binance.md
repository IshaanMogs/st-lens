# Primary data source: Binance Spot public market data (Phase 0 record)

*Technical details checked 2026-10-08 against the official API docs (github.com/binance/binance-spot-api-docs, `web-socket-streams.md` and `CHANGELOG.md`). Re-check before Phase 1 because Binance changes these without notice. The terms are covered separately at the end of this document.*

## Streams (BTCUSDT; ETHUSDT later)

| Purpose | Stream | Documented speed | Key fields |
|---|---|---|---|
| Diff depth | `btcusdt@depth@100ms` (default `btcusdt@depth` = 1000 ms) | 1000 ms or 100 ms | `E` event time, `U` first update ID, `u` final update ID, `b`/`a` = `[price, qty]` (**qty 0 = remove level**) |
| Trades | `btcusdt@trade` | real-time | `t` trade ID, `p`, `q`, `T` trade time, `m` buyer is maker |
| Snapshot (REST) | `GET /api/v3/depth?symbol=BTCUSDT&limit=5000` | on demand (weight 250 at limit 1001–5000) | `lastUpdateId`, ≤ 5000 levels per side |
| Instrument rules (REST) | `GET /api/v3/exchangeInfo?symbol=BTCUSDT` | on demand (weight 20) | `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize` (0 = rule disabled) |

- **Endpoints:**
  - WebSocket: `wss://stream.binance.com:9443` (or `:443`). `wss://data-stream.binance.vision` serves market data only.
  - REST: `https://api.binance.com`. For public market-data-only calls, the docs recommend `https://data-api.binance.vision`; the Phase 1 REST client uses that.
  - The `*.binance.vision` hostnames are live API endpoints. **They are not the Binance Vision historical datasets** covered by the Vision Dataset Terms (source B below).
- **REST errors:** HTTP 429 = rate limit exceeded; HTTP 418 = IP auto-banned for continuing after 429s. Both carry `Retry-After` (seconds). The client raises on any non-200 and never retries automatically.
- **Combined stream:** `/stream?streams=btcusdt@depth@100ms/btcusdt@trade`.
- **Timestamps:** milliseconds by default; `timeUnit=MICROSECOND` gives µs. **Decision: record in µs.** Store both the exchange time (`E`/`T`) and the local receive time, both as UTC (spec B.1).
- **Aggressor side:** `m = true` means the buyer is the maker, so the aggressor is the **seller**.

## Update IDs and gap detection: supported for depth

Binance documents the local-book procedure:

1. Buffer stream events.
2. Fetch the snapshot. If `lastUpdateId < U` of the first buffered event, re-fetch.
3. Drop buffered events with `u ≤ lastUpdateId`. Per the docs, the first remaining event should have `lastUpdateId` within its `[U; u]` range.
4. Apply events in order:
   - ignore an event with `u <` the local ID;
   - **if `U >` local ID + 1, events were missed → discard the book and resync from step 1**;
   - normally `U` = previous `u` + 1.

This gives the deterministic gap rule that the BookBuilder (Phase 2) and the quarantine/re-sync counts (Phase 1 gate) need.

## Caveats that affect later phases (manually reviewed: time, causality, leakage)

1. **Trades have no update-ID link to depth events.** Execution attribution (spec B.5) must align trades to depth diffs by timestamp. Each diff event batches up to 100 ms of changes, so attribution is approximate within a batch. Alignment must use only trades with `T ≤` the diff's `E` (never later trades), or it leaks the future.
2. **Trade-ID contiguity is not documented.** The docs don't say that `t` increments by 1 per symbol. **Verify empirically in Phase 1** before using it for trade-gap detection.
3. **Update speed is inconsistent between sources.** The 2025-11-11 changelog entry mentions `@depth` changing to 50 ms, but the stream page still lists 1000 ms / 100 ms. **Measure the actual inter-event spacing in Phase 1.** Choose the 250 ms grid only after that measurement.
4. **Snapshot depth is 5000 levels per side.** Levels outside the snapshot are unknown until they change, so the price-grid ladder (P buckets) must stay well inside the snapshot range.
5. **Every reconnect needs a resync, and each resync is logged.** Triggers:
   - the 24 h connection limit;
   - a `serverShutdown` event;
   - a missed pong (server pings every 20 s; disconnects if no pong within 1 min).

   Other connection limits: at most 5 incoming messages/s per connection and 300 connection attempts per 5 minutes per IP. Data across a resync gap is a discontinuity and must never be bridged by forward-filling.
6. **Exchange time vs receive time.** Features and labels use exchange time. Receive time is used for latency and data-health monitoring only.

## Terms of use: two different data sources

ST-LENS plans to collect **only source A**. Source B is not used.

### A. Live public market data (WebSocket streams + REST depth snapshot): what we collect

- **Data:**
  - `btcusdt@depth@100ms` and `btcusdt@trade` from `stream.binance.com` (or `data-stream.binance.vision`);
  - `GET /api/v3/depth` snapshots from `api.binance.com`;
  - recorded by us in real time.
- **Applicable terms:** the general **Binance Terms of Use** (binance.com/en/terms), plus any API-specific terms Binance publishes. The API documentation (`web-socket-streams.md`) gives technical rules (rate limits, connection limits), not licence terms.
- **Review status: NOT REVIEWED (UNRESOLVED).**
  - The Terms of Use page could not be retrieved programmatically (client-rendered page). Its clauses on data use, redistribution and restricted jurisdictions are therefore unknown to this project.
  - We found no Binance document that explicitly licenses live API market data for research use, and none that explicitly prohibits it.
  - **Action before Phase 1 recording:** the user reads the current Terms of Use and records the relevant clauses and their date here.
- **Jurisdiction (UNRESOLVED):** confirm that accessing Binance.com market data is permitted from where the recorder will run.
- **Interim working policy (our own conservative choice, not derived from any Binance terms):**
  - non-commercial research use only;
  - raw and derived data are never committed (`/data/` is git-ignored) or published;
  - attribute Binance as the source;
  - make no claim of Binance endorsement;
  - stay within published rate and connection limits.

### B. Binance Vision historical datasets (data.binance.vision): not used

- **Applicable terms:** **Binance Vision Dataset Terms**, v1.0, last updated 26 Aug 2026. They are supplemental to the Binance Terms of Use.
- **Scope:** "Datasets" made available through www.data.binance.vision and associated endpoints, i.e. published historical archives. **They do not state that they govern live WebSocket/REST data, and this project does not assume they do.**
- **Summary:**
  - licence: CC BY-NC-SA 4.0;
  - permitted: academic research and personal non-production research;
  - not permitted: commercial use, redistribution or resale of the data or derived feeds, or implying Binance endorsement;
  - required: derivative works must credit Binance Vision and use the same licence.
- **Relevance:** applies only if a Vision dataset is ever downloaded (e.g. a future trade backfill). That would need a separate decision and its own entry in `decisions.md`.

## Not used for the core system

- **Tardis.dev:** optional future source only.
- **data.binance.vision** historical files: these may help with trade backfill. Whether they contain diff-depth data suitable for book reconstruction was **not verified**. The core pipeline does not depend on them.
