# ST-LENS

Research prototype that scores how *spoof-like* recent limit-order-book activity looks, using spatial (price-ladder) and temporal models. It flags **patterns consistent with spoofing**. It cannot identify traders, intent or rule violations, it is not a compliance tool, and it is **not production-ready**.

> **Current state: fast-track prototype on SYNTHETIC data.** The end-to-end pipeline (book reconstruction → causal features → spoof injection and labels → leakage-safe splits → rule/XGBoost/deep baselines → ST-LENS-Net and ablations → evaluation → explanations → offline replay → dashboard) runs on a simulated order book with injected episodes. Live Binance data collection is not yet approved. No result here says anything about real markets.

## Documents

- [`docs/fast_track.md`](docs/fast_track.md): what is implemented, what was evaluated, what is synthetic, limitations, deferred work.
- [`docs/results.md`](docs/results.md): metrics, **generated automatically** by the experiment script.
- [`docs/ST-LENS_Technical_Specification.md`](docs/ST-LENS_Technical_Specification.md): design source.
- [`docs/related_work.md`](docs/related_work.md): related work. **No novelty is claimed**; ST-LENS-Net adapts the established CNN + temporal (DeepLOB) pattern.
- [`docs/canonical_events.md`](docs/canonical_events.md), [`docs/data_source_binance.md`](docs/data_source_binance.md), [`docs/decisions.md`](docs/decisions.md).

## Setup (Python 3.11)

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311 torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv311 -e ".[dev,ingest,ml,dashboard]"
```

## Run

```bash
python scripts/run_experiment.py --quick     # ~30 s smoke run (numbers meaningless)
python scripts/run_experiment.py             # full run, validation only
python scripts/run_experiment.py --final     # full run + one-time test evaluation -> docs/results.md
streamlit run dashboard/app.py               # read-only dashboard over artifacts/
```

## Checks (same as CI)

```bash
ruff check .
ruff format --check .
pytest
```

## Data

Market data is never committed (`/data/` and `/artifacts/` are git-ignored). Binance data is for non-commercial research only, and recording waits on the terms review in `docs/data_source_binance.md`.
