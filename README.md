# ST-LENS

Research prototype that scores how *spoof-like* recent limit-order-book activity looks, using spatial (price-ladder) and temporal models. It flags **patterns consistent with spoofing** in public L2 data. It cannot identify traders, intent, or rule violations, and it is not a compliance tool.

- Specification: [`docs/ST-LENS_Technical_Specification.md`](docs/ST-LENS_Technical_Specification.md)
- Related work and novelty: [`docs/related_work.md`](docs/related_work.md)
- Data source (Binance) notes: [`docs/data_source_binance.md`](docs/data_source_binance.md)
- Decisions: [`docs/decisions.md`](docs/decisions.md)

**Status:** Phase 0 (setup and literature review). No pipeline code yet.

## Setup (Python 3.11)

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"          # add ,ingest / ,ml / ,dashboard as phases need them
```

or, with plain pip on a Python 3.11 interpreter: `python -m pip install -e ".[dev]"`.

## Checks (same as CI)

```bash
ruff check .
ruff format --check .
pytest
```

## Data

Market data is never committed (`/data/` is git-ignored). It is used for non-commercial research only; see the terms notes in `docs/data_source_binance.md`.
