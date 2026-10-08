# Decision log

## Resolved: 2026-10-08 (before Phase 0)

| # | Topic | Decision |
|---|---|---|
| 1 | Phase numbering | Phases **0–12** (13 phases). Spec section O's list renders as 1–13; read it as 0–12. |
| 2 | Primary data source | Binance Spot **live** public market data: diff-depth + trade WebSocket streams, plus REST depth snapshots (see `data_source_binance.md`, source A). Binance Vision historical datasets are not used. |
| 3 | Symbols | **BTCUSDT** primary. ETHUSDT later as a cross-symbol generalisation test; not a Phase 1 blocker. |
| 4 | Python | **3.11** for the project, CI and Docker. |
| 5 | PyTorch | Must run on CPU; code stays device-agnostic so CUDA can be used later. |
| 6 | CI | GitHub Actions (`.github/workflows/ci.yml`): Ruff lint, Ruff format check, pytest on 3.11. |
| 7 | Branch | `main`. |
| 8 | Specification location | `docs/ST-LENS_Technical_Specification.md` (moved verbatim from the pasted file). |
| 9 | Data volume | Phase 1 needs only enough data to prove ingestion and reconstruction. See D-EVAL-1 for the eventual dataset. |
| 10 | Tardis.dev | Optional future source; the core system must not depend on it. |
| 11 | ABIDES | Deferred; not part of the core implementation. |

## Decided during Phase 0

- **Dependencies are split into extras.** Core = data layer; `ingest`, `ml`, `dashboard` and `dev` are separate, so CI installs only `.[dev]` until a phase needs more.
- **Timestamps are recorded in microseconds** (`timeUnit=MICROSECOND`) as UTC. Ruff rule `DTZ` forbids naive datetimes.
- **Pytest treats warnings as errors** (`filterwarnings = error`), so pandas/numpy deprecations surface immediately. Any ignore must be explicit.
- **Python environment (D-ENV-1):**
  - The pre-existing local Python 3.13 `.venv` is **not** a project environment and must not be modified further.
  - During Phase 0 it received one editable install of `stlens` plus its core dependencies, used to run pytest once. Nothing else has been changed since.
  - Before Phase 1, a clean Python 3.11 project environment is created and the project is installed from `pyproject.toml`.
  - Done: the project environment is `.venv311` (Python 3.11.15).

## Decided during Phase 1 (details in `canonical_events.md`)

- **D-ING-1 `BookDelta` granularity.** One canonical `BookDelta` per price-level change, as in spec B.2. Exchange messages are kept as exchange-specific envelopes (`BinanceDepthMessage`), so messages with zero level changes keep their `U`/`u`.
- **D-ING-2 Raw storage.** Gzip JSON lines in `data/raw` and `data/quarantine`, partitioned by `exchange/symbol/UTC receipt day/session`. Files are exclusive-create and the payload is stored as the exact text. *Provisional; it may be revisited when volumes are measured.*
- **D-ING-3 Price/size table type.** float64 in Pandera tables. *PROVISIONAL:* no precision requirement has been established yet, and the exact decimal strings are preserved in raw.
- **D-ING-4 CI extras.** CI installs `.[dev,ingest]`, because the REST and recorder tests use httpx. No test touches the network.

## D-EVAL-1: evaluation design (PROVISIONAL, 2026-10-08)

**Status: provisional.** The split sizes below are decided. The **exact fold schedule is provisional** until the formal embargo calculation is finalised. That calculation needs window length T and label tolerance h, which are set from the episode-duration distribution in Phase 4 (spec D, J).

### Dataset and split

- **Primary dataset target:** 30 calendar days of BTCUSDT (UTC days D1–D30).
- **Development / walk-forward period:** the first ~25 days (D1–D25).
- **Final lock-box:** the last 5 completely untouched days (D26–D30).

### Lock-box rules

- **The lock-box must not be used for:**
  - feature selection;
  - hyperparameter tuning;
  - threshold selection (including alert-band thresholds);
  - model selection.
- **Also excluded from the lock-box, because each one fits something to the data:**
  - normalisation / scaler statistics;
  - heuristic-label thresholds derived from data quantiles;
  - injection-parameter distributions sampled from real large orders;
  - exploratory plots or summary statistics of features, labels or scores.
- **Permitted on lock-box days before the final evaluation:** only pipeline-integrity checks (capture completeness, schema validation, gap/re-sync counts) that do not compute or display features, labels or model outputs.
- **Final evaluation:** the lock-box is evaluated **once**, at the end of the project, via the explicit `--final` path (spec J), and each use is logged.

### Walk-forward schedule (PROVISIONAL)

Inside D1–D25, following spec I:
- **Fold k:** train on D1…Dk, validate on Dk+1, test on Dk+2, then roll forward one day.
- **Embargo:** at least T + h between consecutive blocks. Final value TBD.
- **Development/lock-box boundary:** an embargo of the same rule also separates D25 from D26.
- **Still TBD:**
  - minimum training length (first k);
  - number of folds;
  - whether embargoes are carved from the start of the following day or placed at day boundaries.

## Open

- **Phase 0 gate:** the repository has not yet been pushed to GitHub, and no CI run has executed.
- **Binance terms:** Terms of Use review (governs live data, source A) and jurisdiction confirmation.
- **Data facts to measure in Phase 1:** true depth-stream update speed; trade-ID contiguity.
- **Licence** for this repository.
- **Approvals:** Captum and DuckDB.
- **Evaluation:** finalise the D-EVAL-1 embargo and fold schedule (after Phase 4).
- **Spec rendering defects:** missing architecture diagram, truncated `d_depth` cell, signed-log on non-negative flows.
