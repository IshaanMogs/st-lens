# Canonical events, Binance mapping and raw capture (Phase 1)

**Code:**

| Layer | Module | Knows Binance? |
|---|---|---|
| Canonical contract | `stlens/schemas/events.py`, `schemas/instruments.py`, `schemas/tables.py` | no |
| Raw boundary + persistence | `stlens/ingestion/raw.py`, `ingestion/storage.py` | no |
| Binance parser / envelope | `stlens/ingestion/binance/parser.py` | **yes** |
| Binance REST client | `stlens/ingestion/binance/rest.py` | **yes** |
| Binance recorder | `stlens/ingestion/binance/recorder.py` | **yes** |

## 1. Pipeline boundary

```
wire text ──► RawMessage ──► RawMessageStore (data/raw/…)               [recorder]
                  │
                  └─► BinanceAdapter ──► ParsedMessage(raw, events, depth_message, unknown_fields)
                              └─► BinanceParseError ──► QuarantinedMessage ──► QuarantineStore
```

- **Downstream code** sees only the canonical `BookSnapshot`, `BookDelta`, `Trade` and `InstrumentSpec`.
- **Binance field names** (`E`, `U`, `m`, …) appear only inside `ingestion/binance/`.
- **Every canonical object** is frozen and validates its invariants on construction, so an invalid event cannot exist.

## 2. Raw-message preservation

`RawMessage` holds:
- `exchange`;
- `stream` (`/stream` for the combined WebSocket, or the REST request line);
- `payload`: the **exact text received**, never re-serialised;
- `recv_time_us`;
- `session_id`;
- `recv_seq`: a local, gap-free per-session receipt counter.

Every canonical event carries `source: SourceRef(exchange, stream, session_id, recv_seq)`, which points back to its raw record. `session_id` and `recv_seq` are assigned locally; **they are not exchange identifiers**.

## 3. Canonical schemas

| Type | Fields |
|---|---|
| `SourceRef` | `exchange`, `stream`, `session_id`, `recv_seq` |
| `PriceLevel` | `price: Decimal > 0`, `size: Decimal ≥ 0` |
| `BookSnapshot` | `symbol`, `last_update_id`, `bids`, `asks` (`PriceLevel` tuples, exchange order), `exchange_time_us: int \| None`, `recv_time_us`, `source` |
| `BookDelta` | `symbol`, `side: Side(bid/ask)`, `price > 0`, `new_size ≥ 0` (0 = remove level), `exchange_time_us`, `recv_time_us`, `first_update_id ≤ final_update_id`, `level_index < level_count`, `source` |
| `Trade` | `symbol`, `trade_id`, `price > 0`, `size > 0`, `aggressor_side: AggressorSide(buy/sell)`, `exchange_time_us` (trade time), `exchange_event_time_us`, `recv_time_us`, `source` |
| `InstrumentSpec` | `exchange`, `symbol`, `status`, `base_asset`, `quote_asset`, `tick_size ≥ 0`, `step_size ≥ 0` (0 = rule disabled), `recv_time_us`, `source` |

### `BookDelta` semantics (spec B.2: `BookDelta(side, price, new_size)`)

- **Granularity.** One `BookDelta` is **one price-level change**. One exchange depth message with N level changes yields **N** `BookDelta` events.
- **Shared message metadata.** All N events carry the same message metadata:
  - `first_update_id` / `final_update_id`: the message's exchange sequence range (Binance `U` / `u`), verbatim;
  - `exchange_time_us`: the message's event time (Binance `E`);
  - `recv_time_us` and `source`: the raw message's receipt time and trace key.
- **Position within the message.** `level_index` (0…N−1) and `level_count` (N) give each change's position and the message size, so a consumer can regroup a message and check it is complete without reordering anything.
- **Order.**
  - Within a side, order is exactly the exchange array order.
  - Binance sends `b` and `a` as separate arrays, so there is **no cross-side order** in the message. Bids-then-asks is a fixed convention, not exchange information.
- **Empty messages.** A message with **zero level changes** yields **zero** `BookDelta` events. Its `U`/`u`, `E`, symbol and raw payload are kept in the exchange envelope (`BinanceDepthMessage`) and in the `depth_messages` table. The Phase 2 BookBuilder must therefore read sequence continuity from the message envelopes / `depth_messages`, **not** from `BookDelta` rows alone.

### Exchange envelope: `BinanceDepthMessage` (Binance-specific, not canonical)

- **Fields:** `symbol`, `event_time_us` (`E`), `first_update_id` (`U`), `final_update_id` (`u`), `bids` (`b`), `asks` (`a`) as `(price, qty)` pairs in message order, `stream`, `raw` (the `RawMessage`).
- **`to_book_deltas()`** produces the canonical events.
- **Validation:** the envelope validates `U ≤ u` and a plausible `E` itself, so empty messages are checked too.

### Numbers

- Prices and sizes are `Decimal`, parsed from Binance's decimal strings. Only plain non-negative decimals are accepted: no sign, exponent, whitespace or NaN.
- JSON numbers in price or size fields are rejected, not converted.
- **Tables use float64 for prices and sizes. This is PROVISIONAL:**
  - no precision requirement has been established yet that would justify a different on-disk type;
  - the exact strings stay in the raw payload;
  - the final Parquet representation is decided when canonical tables are first persisted.

## 4. Binance → canonical mapping

| Binance | Canonical |
|---|---|
| **Diff depth** `{"e":"depthUpdate"}` | `BinanceDepthMessage` + N × `BookDelta` |
| `E` | `exchange_time_us` on every delta (unit per §5) |
| `s` | `symbol` (must equal the adapter symbol) |
| `U` / `u` | `first_update_id` / `final_update_id` on the envelope and every delta |
| `b[i] = [p, q]` | `BookDelta(BID, p, q)`, `level_index = i` |
| `a[j] = [p, q]` | `BookDelta(ASK, p, q)`, `level_index = len(b) + j` |
| **Trade** `{"e":"trade"}` | `Trade` |
| `t` | `trade_id` |
| `p`, `q` | `price`, `size` |
| `T` | `exchange_time_us` (trade time) |
| `E` | `exchange_event_time_us` |
| `m = true` / `false` | `aggressor_side = SELL` / `BUY` (`m` = buyer is maker) |
| `M` | documented "Ignore"; not mapped (still in raw) |
| **REST** `GET /api/v3/depth` | `BookSnapshot` (`lastUpdateId`, `bids`, `asks`; symbol = requested symbol; `exchange_time_us = None`) |
| **REST** `GET /api/v3/exchangeInfo?symbol=` | `InstrumentSpec`: `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize`, `status`, `baseAsset`, `quoteAsset` |

**Accepted forms:**
- Raw-stream payloads.
- Combined-stream wrappers `{"stream", "data"}`. `source.stream` is then the wrapper's stream name.

**Rejected:** any other event type (e.g. `aggTrade`, subscription responses, `serverShutdown`), which raises `BinanceParseError`. The raw message itself is still stored by the recorder.

## 5. Timestamp semantics

- **Unit:** `int` microseconds since the Unix epoch, UTC (`utils/timestamps.py`). Naive datetimes are rejected.
- **Exchange time** is Binance's clock.
  - Unit is set by `BinanceAdapter(time_unit=...)`, which must match the stream URL's `timeUnit`. The recorder subscribes with `timeUnit=MICROSECOND`.
  - Milliseconds are converted by exact ×1000.
  - The unit is never guessed. A value outside 2017–2100 fails validation.
- **Receipt time** is this machine's wall clock: for stream frames, on arrival; for REST, when the full response has arrived. It is subject to clock error and latency.
- **The two clocks are never substituted for each other.** The REST snapshot and `exchangeInfo` have no exchange event time.
- **Ordering:** nothing is sorted or reordered anywhere in Phase 1.

## 6. Update-ID and sequence semantics

- `U`/`u` and `lastUpdateId` are carried verbatim.
- **Within one message**, the only check is `U ≤ u`.
- **Between messages**, continuity, snapshot alignment and resync belong to the BookBuilder (Phase 2) and are **not** checked here.
- **Trade IDs** are preserved verbatim. Their contiguity is not documented and is not assumed.
- **`recv_seq`** is gap-free within a session and follows receipt order for stream frames and REST responses alike. A REST response takes its number when it arrives, not when it was requested, and a failed request takes none.

## 7. Validation and quarantine

- **Per-message checks:**
  - JSON object;
  - no duplicate keys;
  - no NaN/Infinity;
  - required fields with exact types (no bool-as-int; `m` is a real boolean);
  - matching symbol;
  - the number and timestamp rules above.
- **Failures** raise `BinanceParseError` with a specific reason. `parse_stream_batch` returns them as `QuarantinedMessage(raw, reason)`, and `QuarantineStore` persists them. Nothing is dropped silently.
- **Unknown fields** are tolerated and listed in `ParsedMessage.unknown_fields`.
- **Pandera tables** (`trades`, `depth_messages`, `depth_levels`) are `strict=True` and check per-row invariants only.

## 8. Raw capture and persistence (minimal; not the research database)

- **Stores.** `RawMessageStore` and `QuarantineStore` write gzip JSON lines to:
  - `data/raw/{exchange}/{symbol}/{YYYY-MM-DD}/{session_id}.jsonl.gz`
  - `data/quarantine/{exchange}/{symbol}/{YYYY-MM-DD}/{session_id}.jsonl.gz`
- **Day partition:** taken from the UTC receipt time.
- **Write rules:**
  - records are written in arrival order;
  - files are created exclusively and never appended to or overwritten;
  - a session's files stay open until the store is closed.
- **Readers** (`read_raw_messages`, `read_quarantined`) fail loudly on corrupt or incomplete records. A file truncated by a crash raises an error rather than being silently skipped.
- **Recorder.** `BinanceRecorder` runs one session per WebSocket connection:
  1. connects to the combined `btcusdt@depth@100ms/btcusdt@trade` stream with `timeUnit=MICROSECOND`;
  2. fetches `exchangeInfo` once (failure is logged, not fatal);
  3. fetches a REST depth snapshot **after the first stream frame is stored**, following the documented buffer-then-snapshot order. If the snapshot fails, the session ends;
  4. optionally re-snapshots every `snapshot_interval_s` (cadence not yet decided);
  5. stores everything raw.
- **Recorder `run()`** reconnects with exponential backoff (minimum 5 s). Each reconnect is a new session with a new snapshot.
- **What the recorder never does:** parse, filter or repair messages, or check sequence continuity.
- **REST base URL:** `data-api.binance.vision` is Binance's documented base URL for live public market-data REST calls. It is **not** the Binance Vision historical dataset service (see `data_source_binance.md`).
- **Not run against Binance.** The recorder is tested only against fakes. It must not be pointed at Binance until the terms and jurisdiction review is complete.

## 9. Known limitations

- **No trade-to-depth linkage.** Binance documents no field linking `t` to `U`/`u`, so none is created or implied. Any later association (Phase 2 attribution) is a timestamp-based approximation.
- **Diff-depth messages batch changes** over the update interval. Intermediate states and the cross-side order within a message are not observable.
- **Snapshot coverage:** at most 5000 levels per side; levels outside are unknown until they change.
- **Receipt time is local wall-clock time,** subject to NTP drift.
- **Tick-size alignment is not yet enforced.** `InstrumentSpec` provides the tick size, but the check is deferred because specs can change over time and must be matched by retrieval time.
- **Fixtures are hand-constructed** from documented formats, not captured.
- **Canonical-event Parquet persistence** (`data/canonical/`) is not implemented.
