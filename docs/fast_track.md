# ST-LENS fast-track prototype: what exists, what was evaluated, what is deferred

This is a **research prototype** built quickly for a demonstration. It implements the smallest defensible version of each phase of the specification (`ST-LENS_Technical_Specification.md`). It is **not production-ready**. Results are in `results.md`, generated automatically by `scripts/run_experiment.py`.

## 1. What the data is

**Everything in the experiment is synthetic.** Live Binance recording is not yet approved (terms and jurisdiction review pending), so no real market data is used.

- **Background market** (`ingestion/synthetic.py`): a zero-intelligence limit order book, with Poisson limit orders at geometric distances, random partial cancellations and Poisson market orders with lognormal (heavy-tailed) sizes.
- **Event format:** it emits the same canonical events as the Binance parser (`BookSnapshot`, per-level `BookDelta` with gap-free update IDs, and `Trade`), in 100 ms batches like Binance diff-depth.
- **No reaction to spoofing.** Background participants do not react to anything. There is therefore **no behavioural price reaction to spoofing**; the only signal is the order lifecycle (size, placement, resting time, cancellation without execution, opposite-side activity).
- **Spoof episodes** (`labeling/injector.py`) are executed by the simulator like real orders. They queue behind existing volume, can be executed by incoming market orders, and appear only through ordinary canonical events.
- **Ground truth:** a spoof counts as positive only if it was placed and then cancelled with zero execution. Spoofs that the market partly executed are not positives.
- **Hard negatives:**
  - large orders that rest and are then **executed**;
  - large orders cancelled **without** opposite-side activity.
- **Episode parameters:**
  - sizes come from the background's own size distribution, conditioned on its 99th-percentile tail;
  - distances follow the background's distance law;
  - payoff is 3–8× a median market order.
  
  These were fixed **before** the final run. An earlier smoke run showed the 95th-percentile setting gave almost no learnable signal, and that is the only change made.
- **Artefact probe.** A classifier using placement features only (size, distance) cannot tell injected orders from background orders of the same size class: the test requires |AUC − 0.5| < 0.1.
- **Two label sources:**
  - *injected ground truth* (used for training and evaluation);
  - *heuristic weak labels* from the transparent rule engine (`labeling/rules.py`), which is also baseline B1.
  
  They are never mixed.

## 2. What is implemented, per phase

| Phase | Implemented (minimal) | Main tests |
|---|---|---|
| 1 Ingestion | Phase 1 Binance parser/REST/recorder (unchanged) + synthetic source emitting canonical events | determinism, gap-free IDs, trade timing, injection ground truth |
| 2 Book | `BookBuilder` (gap → unsynced until next snapshot, stale/incomplete messages rejected, tick alignment), executed/added/cancelled attribution, 250 ms grid | hand-made books, crafted attribution cases, prefix invariance |
| 3 Features | Mirrored price-grid ladder tensor `[T, 6, 20]`, level-indexed tensor (DeepLOB baseline), 9 context series, window features for classical models | **truncation tests** for every feature family and the rule engine |
| 4 Labels | Injection schedule, hard negatives, window labels (y, side, location), rule engine, artefact probe | round trip (logged = recovered), labels = episode log, probe ≈ chance |
| 5 Datasets | Chronological day splits, separate injection seeds per split, train-only scalers, lazy PyTorch windows | spans disjoint, episodes disjoint, scaler fit on train only, natural class rate |
| 6 Classical | B0 prior, B1 rule engine, B2 logistic, B3 XGBoost (each with and without rule-like features) | metrics on hand-made inputs, planted-signal learning |
| 7 Deep baselines | B4 spatial CNN, B5 causal TCN (no price axis), B6 DeepLOB-style (simplified) | shapes, causality, overfit planted pattern |
| 8 ST-LENS-Net | Side-symmetric multi-scale price encoder, cross-touch conv, context fusion, causal TCN, attention pooling, side and location heads; ablations spatial-only, temporal-only | shapes, TCN causality and receptive field, order invariance of the temporal ablation |
| 9 Explainability | Reason codes from book evidence, Integrated Gradients, bucket occlusion, localisation hit rate, randomised-weights sanity check | known-bucket attribution, IG completeness, randomisation changes attributions |
| 10 Replay | Step-by-step scorer: calibrated probability → EMA → band → alert (hysteresis, cooldown, High requires evidence) | streaming = batch EMA, no look-ahead, hysteresis/cooldown/evidence |
| 11 Dashboard | Streamlit (read-only): market monitor, alerts, models and metrics, data health | integration test renders all tabs from a real tiny run |
| 12 | Ruff, pytest, CI (CPU torch), docs | — |

**Not novel.** ST-LENS-Net adapts the established CNN-then-temporal pattern (DeepLOB). No novelty is claimed for the architecture; see `related_work.md`.

## 3. What was evaluated, and how

- **Protocol:**
  - 8 synthetic days of 30 min each: 5 train, 1 validation, 2 test, in time order;
  - window 10 s, label horizon 2 s;
  - 3 seeds per deep model.
- **Decisions on validation only:** early stopping, temperature scaling, F1-optimal thresholds and the alert-budget threshold.
- **Test days were evaluated once** (`--final`, logged in `artifacts/final_runs.log`).
- **Metrics:**
  - PR-AUC (primary) and ROC-AUC;
  - recall at a fixed window-alert budget;
  - episode recall, detection latency and hard-negative flag rate;
  - explanation localisation;
  - replay alert counts.

## 4. Limitations (apply to every number in results.md)

1. **Synthetic only.** The background is a zero-intelligence model with no strategic participants and no reaction to spoofing. Nothing transfers to real markets without re-running on recorded data.
2. **Labels reflect the designer's injection assumptions.** A model may learn the injector rather than spoofing. The artefact probe checks placement only.
3. **Background activity can look spoof-like.** Background large orders are sometimes cancelled quickly, and those count as negatives, so measured precision is pessimistic.
4. **L2 data has no order or trader IDs.** Outputs are "spoof-like pattern" scores. They cannot show intent or who placed an order.
5. **The attribution of executed versus added/cancelled volume is approximate.** It is aggregated per 100 ms message and matched to trades by price and time only.
6. **Small scale:**
   - one chronological fold instead of the full walk-forward;
   - one market configuration;
   - fixed hyperparameters with no search;
   - 3 seeds.
   
   Differences smaller than the seed spread are not findings.
7. **Crypto/synthetic ≠ regulated equities.**

## 5. Model checkpoint status

- **Every run now saves a checkpoint.** `scripts/run_experiment.py` writes `<out>/stlens_full.pt` for the ST-LENS-Net seed chosen on validation. It contains:
  - the architecture config and weights;
  - the train-only input scalers;
  - the validation-fitted temperature and the validation-chosen threshold;
  - metadata.
  
  `stlens.models.checkpoint.load_checkpoint` reloads it with the safe `torch.load(weights_only=True)` loader, and tests show the reload reproduces predictions exactly.
- **The final run reported in `results.md` (2026-10-08) has no checkpoint.** It predates this feature. Creating one would mean retraining, which would be a new experiment and could differ numerically, so it was not done. The dashboard and `results.md` use the stored scores from that run.

## 6. Deferred to a later or production phase

- Live recording and a Binance data-health report (the Phase 1 gate on real data) are blocked on the terms and jurisdiction review.
- Not yet built:
  - full rolling walk-forward with multiple folds and block bootstrap over days;
  - held-out injection-parameter ranges and cross-symbol and cross-regime tests;
  - MLflow tracking and model registry, and TreeSHAP (`tracking` extra);
  - incremental event-by-event streaming feature state;
  - a live WebSocket scoring loop;
  - a replay-equals-offline test at the event level (the current test covers the scoring layer, and feature causality is covered by truncation tests);
  - Docker / docker-compose;
  - an alert store with an analyst feedback loop (the dashboard is read-only);
  - canonical-event Parquet storage, and a final decision on the price storage type;
  - agent-based simulation (ABIDES).
