# ST-LENS Technical Specification

8 Oct 2026 · @Paras
ST-LENS is a research prototype that scores how *spoofing-like* the recent behaviour of a limit order book looks, using a model that reads the book across price levels (spatial) and over time (temporal). It is not a compliance tool and cannot prove manipulation: public crypto data has no trader or order identities, so the system flags patterns consistent with spoofing, not spoofing itself.

## Scope and honest framing

**What the system answers:** "Given the last few seconds of the order book and trades, how strongly does the current activity resemble a spoofing episode (large non-bona-fide orders placed to move price, then cancelled)?"
**What it does not answer:** who placed the orders, whether there was intent, or whether a rule was broken. Intent is a legal element of spoofing; it is not observable in aggregated public data.
**Design principles used throughout this spec**

1. **One causal code path.** The same book-builder and feature code runs in training, replay and live inference. Why: train/serve skew is the most common silent failure in market ML.
2. **Time is sacred.** Every split, normalisation and feature only looks backwards. Why: LOB data is heavily autocorrelated, so any look-ahead inflates results dramatically.
3. **Baselines before novelty.** The deep model must beat a rule engine and a gradient-boosted model on identical splits. Why: since labels partly come from rules, a deep model that only re-learns the rule adds nothing.
4. **Small, testable increments.** Each module has a contract (schema in, schema out) and tests. Why: you are building with AI assistance; contracts and tests are how you catch generated code that looks right but isn't.

## Related work and novelty caveat

Combining convolutions across price levels with a recurrent/temporal layer is **established**, not new. DeepLOB (Zhang, Zohren and Roberts, 2019) already stacks CNN, Inception-style blocks and an LSTM over LOB snapshots, though for mid-price movement prediction rather than manipulation detection. The FI-2010 benchmark (Ntakaris et al., 2018) is the usual public LOB dataset for that line of work. Order Flow Imbalance (Cont, Kukanov and Stoikov, 2014) is a standard microstructure feature.
Therefore ST-LENS should **not** claim a novel architecture. The defensible contributions are: (a) applying and ablating a spatial-temporal LOB model for spoofing-pattern scoring on public crypto data, (b) a transparent, reproducible labelling protocol, and (c) a leakage-controlled walk-forward evaluation. Whether even (a) is new must be checked in Phase 0: search for prior spoofing-detection work using deep learning on LOB data before writing any claim. These references are cited from general knowledge and should be verified against the papers themselves.

## A. System architecture

ST-LENS is organised as four layers connected by persisted, schema-checked artefacts, with the serving layer reusing the data-layer and feature code rather than re-implementing it.
[embed: node/3a85d6b2-722c]
The serving layer's event source feeds the same book builder, resampler and feature state used offline; the diagram draws that as one path to keep it readable. Training runs log to MLflow, and the model runner loads models from its registry.

|   |
| - |

Layer

|   |
| - |

Modules (from your list of 20)

|   |
| - |

Why it is a separate layer

|   |
| - |

Data

|   |
| - |

1 ingestion, 2 schema validation, 3 reconstruction/normalisation

|   |
| - |

Exchange-specific mess is contained here; everything downstream sees one canonical format

|   |
| - |

Research

|   |
| - |

4 features, 5 labelling, 6 sequences, 7–10 models, 11 training, 12 evaluation, 13 tracking

|   |
| - |

Offline, batch, reproducible; can be re-run from stored data with a config

|   |
| - |

Serving

|   |
| - |

14 inference, 15 risk scoring, 16 alerts, 17 explainability

|   |
| - |

Incremental and stateful; must behave identically to offline code on the same data

|   |
| - |

Presentation and ops

|   |
| - |

18 dashboard, 19 tests, 20 Docker

|   |
| - |

Consumes stores only, so the UI can never change results
**Why layered with files between stages, not one script:** each stage can be tested in isolation, re-run after a bug fix without re-downloading data, and inspected when results look wrong. For a beginner using AI-generated code, being able to open the intermediate Parquet file and look at it is the main debugging tool.
**Why exchange-agnostic adapters:** only the adapter knows exchange message formats. Adding another venue, or an equities source like LOBSTER later, means writing one adapter that emits canonical events, with no changes downstream.
**Initial data source (to confirm in Phase 0):** a major exchange's public market-data WebSocket (e.g. Binance diff-depth plus trade streams) recorded yourself, optionally supplemented by a historical vendor such as Tardis.dev. Verify each provider's terms of use and whether update IDs allow gap detection before committing.

## B. Data flow: raw event to prediction

Every prediction is produced by a strictly forward-moving pipeline; each stage persists its output so it can be inspected, tested and replayed.

1. **Raw capture.** Exchange messages (book snapshot, incremental depth updates, trades) are stored exactly as received, with exchange timestamp *and* local receive timestamp, as compressed JSON lines or Parquet partitioned by `exchange/symbol/date`. Why: raw data is the only thing you cannot regenerate; keeping it unmodified lets you fix bugs downstream and rebuild.
2. **Normalisation to a canonical event schema.** An exchange adapter converts each message into ST-LENS events: `BookSnapshot`, `BookDelta(side, price, new_size)`, `Trade(price, size, aggressor_side)`. Why: this is the exchange-agnostic seam; a new source only needs a new adapter.
3. **Schema validation (Pandera).** Types, non-negative sizes, positive prices, monotonic sequence IDs, tick-size alignment. Bad batches go to a quarantine table, not silently dropped. Why: crypto feeds drop messages and reconnect; you must know when.
4. **Book reconstruction.** A `BookBuilder` applies deltas to the snapshot, checks sequence continuity, and re-syncs from a fresh snapshot on gaps. Output: a sequence of book states.
5. **Event attribution.** For each update, the size change at a price level is split into *executed* volume (matched against trades at that price) and *inferred add/cancel* volume (the remainder). Why: spoofing is a cancellation pattern, and aggregated L2 data never says "cancel" directly.
6. **Resampling.** Book states are sampled on a fixed clock grid (default proposal: every 250 ms, to be tuned), carrying forward the last state and aggregating flow quantities between ticks.
7. **Feature and tensor construction.** Causal rolling features (section E) and the ladder tensor (section C).
8. **Labelling.** Synthetic spoof injection and/or heuristic weak labels attach episode labels (section F). Training only.
9. **Windowing.** Sliding windows of T grid steps become samples (section D).
10. **Model inference.** Model produces a logit per window (plus optional side/location heads).
11. **Calibration and risk scoring.** Logit → calibrated probability → smoothed risk score with severity band.
12. **Alert generation.** Threshold with hysteresis and cooldown → alert record with explanation attached.
13. **Storage and display.** Scores and alerts are written to an alert store the dashboard reads from.

Steps 1–7 and 10–13 are identical in training, replay and live mode; only the event source (file vs WebSocket) changes.

## C. Proposed LOB data representation

The core representation is a **ladder tensor** of shape `[C, T, P]`: C channels, T time steps, P price buckets, with the price axis laid out as one continuous ladder from deep bids through the spread to deep asks.

### Price axis: price-grid primary, level-indexed secondary

|   |
| - |

Representation

|   |
| - |

How the price axis is defined

|   |
| - |

Strength

|   |
| - |

Weakness

|   |
| - |

Level-indexed (DeepLOB style)

|   |
| - |

Index k = k-th best bid/ask

|   |
| - |

Compact; matches published baselines

|   |
| - |

Index meaning shifts when a level appears or vanishes, so a static spoof order "moves" across columns

|   |
| - |

Price-grid (proposed primary)

|   |
| - |

Fixed buckets at 0, 1, 2… ticks (or bps) from mid, per side

|   |
| - |

A resting order stays in the same column while the book moves around it

|   |
| - |

Sparse in thin books; needs a bucket width choice
**Why price-grid as primary:** a spoof order is defined by *where* it sits relative to the touch and how long it stays. On a price grid that becomes a stable spatial feature the convolution can see; on a level index it is smeared across columns. The level-indexed form is kept for the DeepLOB-style baseline so results are comparable to the literature, and comparing the two is itself an ablation.
**Layout:** `[bid_bucket_{P/2-1} … bid_bucket_0 | ask_bucket_0 … ask_bucket_{P/2-1}]`. Why mirrored: the spread sits in the middle, so a small convolution kernel can see both sides of the touch at once, and bucket distance is consistent on both sides.
**Bucket width:** start with 1 tick for the first buckets near the touch; consider wider (geometric) buckets further out so deep levels are covered without huge P. Exact widths are tuned per symbol in Phase 3 from its tick size and typical spread; no value is fixed here.

### Channels (per bucket, per time step)

|   |
| - |

Channel

|   |
| - |

Meaning

|   |
| - |

Normalisation

|   |
| - |

`depth`

|   |
| - |

Resting size in the bucket

|   |
| - |

`log1p`, then z-score using training-period statistics

|   |
| - |

`d_depth`

|   |
| - |

Change since previous step

|   |
| - |

Signed log (\`sign(x)·log1p(

|   |
| - |

`add_vol`

|   |
| - |

Inferred new volume added

|   |
| - |

Signed log

|   |
| - |

`cancel_vol`

|   |
| - |

Inferred volume removed without trade

|   |
| - |

Signed log

|   |
| - |

`exec_vol`

|   |
| - |

Volume executed against this bucket

|   |
| - |

Signed log

|   |
| - |

`rel_size`

|   |
| - |

Depth ÷ rolling median depth of that bucket

|   |
| - |

Ratio, clipped
**Why these channels:** spoofing is visible as *large relative size* (`rel_size`), *appearing quickly* (`add_vol`), *disappearing without trading* (`cancel_vol` vs `exec_vol`). Raw depth alone would force the model to infer flows by differencing, which is harder to learn from limited positives.
**Why prices are relative, not absolute:** absolute BTC prices drift by orders of magnitude across years, which makes them non-stationary and invites the model to memorise price regimes. Distance from mid in ticks/bps is comparable across time and symbols.
A separate **context vector** per time step carries non-spatial series (mid return, spread, realised volatility, trade imbalance) to be fused into the temporal stage (section H).

## D. Definition of a training sample

One sample is **the order book's history over a look-back window ending at decision time t**, labelled by whether a spoofing episode is active or has just completed at t. It is a *detection* task (what has happened), not a *forecasting* task (what will happen).

|   |
| - |

Component

|   |
| - |

Definition

|   |
| - |

Why

|   |
| - |

Decision time t

|   |
| - |

A grid timestamp

|   |
| - |

Matches how a surveillance system would score the market continuously

|   |
| - |

Ladder input X

|   |
| - |

`[C, T, P]` tensor covering (t − T·Δ, t]

|   |
| - |

Only past and present data, so the same sample can be built live

|   |
| - |

Context input z

|   |
| - |

`[T, F]` series of non-spatial features over the same window

|   |
| - |

Lets the temporal model use price/volatility context

|   |
| - |

Tabular input v

|   |
| - |

Aggregated window statistics (for XGBoost / logistic baselines)

|   |
| - |

Gives classical models a fair, comparable input

|   |
| - |

Primary label y

|   |
| - |

1 if a spoof episode's *cancellation phase* falls inside (t − h, t], else 0

|   |
| - |

Spoofing is only recognisable once the order is pulled; labelling earlier would ask the model to predict intent

|   |
| - |

Aux label: side

|   |
| - |

bid / ask / none

|   |
| - |

Tells the analyst which side was suspicious

|   |
| - |

Aux label: location

|   |
| - |

Price bucket(s) where the spoof order rested

|   |
| - |

Supports localisation and checks explanations

|   |
| - |

Metadata

|   |
| - |

symbol, t, split id, label source (synthetic/heuristic), episode id

|   |
| - |

Needed for grouped evaluation and leakage checks
**Window length T and label tolerance h** are hyperparameters chosen so that T covers a full spoof lifecycle (placement → rest → cancel → price reaction) observed in the labelling stage. They are set from the label generator's episode-duration distribution in Phase 4, not guessed now.
**Sampling stride.** Consecutive windows overlap heavily. For training, use a stride > 1 and keep *all* windows from the same episode in the same split. Why: overlapping windows are near-duplicates; random shuffling across splits would leak.
**Class imbalance.** Positives will be rare by construction. Keep the natural rate in validation/test (that is the operating reality) and only rebalance training via loss weighting or sampling.

## E. Proposed feature groups

Seven groups, each computed causally (rolling windows over past data only) and each mapped to a part of the spoofing mechanism. The deep model sees the raw ladder plus group 1–3 context; classical baselines see aggregated versions of all groups.

|   |
| - |

\#

|   |
| - |

Group

|   |
| - |

Example features

|   |
| - |

Spoofing mechanism it targets

|   |
| - |

1

|   |
| - |

Book shape

|   |
| - |

Spread, depth at top-k levels, cumulative depth curve slope, queue imbalance at k = 1, 3, 5, 10

|   |
| - |

A spoof order distorts visible depth and imbalance

|   |
| - |

2

|   |
| - |

Order flow

|   |
| - |

Order Flow Imbalance (Cont et al.), net add vs cancel flow per side, depth change rate

|   |
| - |

The fake pressure the spoofer creates

|   |
| - |

3

|   |
| - |

Cancellation dynamics

|   |
| - |

Inferred cancel volume per side, cancel-to-add ratio, cancel-to-execution ratio, cancels by distance from touch

|   |
| - |

The defining act: large orders removed without trading

|   |
| - |

4

|   |
| - |

Large-order lifecycle

|   |
| - |

Size relative to rolling depth distribution, distance from touch, time-in-book proxy, whether it was ever touched by trades

|   |
| - |

Large orders that live briefly and never execute

|   |
| - |

5

|   |
| - |

Trade activity

|   |
| - |

Trade count/volume, aggressor-side imbalance, trades on opposite side after a large order appears

|   |
| - |

The spoofer's real goal: execution on the other side

|   |
| - |

6

|   |
| - |

Price dynamics

|   |
| - |

Mid-price returns, micro-price, realised volatility, price move toward/away from large orders

|   |
| - |

The price reaction the spoof seeks to induce

|   |
| - |

7

|   |
| - |

Regime and context

|   |
| - |

Time of day (UTC hour, cyclically encoded), rolling volume regime, volatility regime

|   |
| - |

Separates normal busy periods from suspicious ones
**Why grouped:** groups make ablations meaningful ("remove cancellation dynamics, does performance collapse?") and give the explanation layer human-readable reason codes.
**Important caveat on group 4.** Aggregated L2 data shows a *level's* total size, not individual orders. "Large order" means a large *step change* at one price that is attributed to a single add; this is an approximation and will be wrong when many small orders arrive together. This limitation must be stated in results.
**Avoid label leakage through features.** Some features (e.g. "large add followed by cancel within N seconds") are close to the heuristic label definition. They are allowed for the classical models, but results must be reported both with and without them so it is clear whether a model learned anything beyond the rule (see J).

## F. Spoofing-label strategy and its limitations

There is no public ground truth for spoofing in crypto order books, so ST-LENS uses **two independent label sources** and never treats either as truth. This is the single biggest scientific risk in the project and should be stated openly in any report.

### Source 1 (primary): controlled synthetic injection

Inject artificial spoof episodes into *real* recorded book data, so the background is realistic and the positives are known exactly.
An episode is parameterised by:

- side (bid or ask) and distance from touch (in buckets)
- size relative to the local depth distribution
- resting duration before cancellation
- whether it is layered (several levels) or a single order
- whether a genuine opposite-side trade follows (the "payoff")

Injection modifies the reconstructed book *before* resampling and feature computation, so injected orders flow through exactly the same pipeline. Hard negatives are injected too: large orders that rest and then **execute**, and large orders that are cancelled *without* any opposite-side activity. Why: without hard negatives the model learns "big order = spoof".
**Why synthetic first:** it is the only way to get exact labels, measure recall, and run controlled experiments (e.g. "how does detection degrade as spoof size shrinks?").

### Source 2 (secondary): heuristic weak labels on real data

A transparent rule engine flags real episodes matching a spoof signature: a large relative add away from the touch → no meaningful execution against it → cancellation within a short time → activity on the opposite side. Thresholds are set from data quantiles and recorded in config.
**Why also heuristics:** they show whether anything spoof-like occurs naturally in the data and give an out-of-distribution test for models trained on injected data.

### Optional Source 3: agent-based simulation

An agent-based market simulator (ABIDES is one open-source option) can generate books with a spoofing agent present. Deferred to a later phase because it adds complexity and its realism must itself be validated.

### Limitations (must appear in every results section)

|   |
| - |

Limitation

|   |
| - |

Consequence

|   |
| - |

Mitigation

|   |
| - |

No trader/order IDs in L2 data

|   |
| - |

Cannot link the placing and cancelling party, or the spoof to the profiting trade

|   |
| - |

Phrase outputs as "spoof-like pattern", never "spoofing"

|   |
| - |

Injected spoofs reflect the designer's assumptions

|   |
| - |

Model may learn the injector, not spoofing

|   |
| - |

Hold out injection parameter ranges for test; check injected orders are not trivially distinguishable from real large orders

|   |
| - |

Heuristic labels are circular

|   |
| - |

A model trained on them can at best re-learn the rule

|   |
| - |

Use heuristics mainly for evaluation; always report the rule engine itself as a baseline

|   |
| - |

Unlabelled real spoofs exist as "negatives"

|   |
| - |

Measured precision is pessimistic and noisy

|   |
| - |

Treat real-data negatives as unlabelled; report alert review samples qualitatively

|   |
| - |

Legitimate behaviour looks similar

|   |
| - |

Market makers routinely cancel large quotes

|   |
| - |

Include hard negatives; review false positives

|   |
| - |

Crypto ≠ regulated equities

|   |
| - |

Different microstructure, fees and participants

|   |
| - |

Do not generalise findings to equity markets without new data

## G. Baseline models

Seven baselines form a ladder of increasing capacity; the proposed model is only interesting if it beats all of them on the same walk-forward splits. Each baseline isolates one question.

|   |
| - |

\#

|   |
| - |

Baseline

|   |
| - |

Input

|   |
| - |

Question it answers

|   |
| - |

B0

|   |
| - |

Prior / random scorer

|   |
| - |

none

|   |
| - |

What does chance look like at this positive rate? (PR-AUC of random = prevalence)

|   |
| - |

B1

|   |
| - |

Rule engine (the heuristic labeller)

|   |
| - |

Raw events

|   |
| - |

Can a transparent rule already do the job?

|   |
| - |

B2

|   |
| - |

Logistic regression

|   |
| - |

Tabular window features `v`

|   |
| - |

Is the signal mostly linear in hand-crafted features?

|   |
| - |

B3

|   |
| - |

XGBoost

|   |
| - |

Tabular window features `v`

|   |
| - |

Strong non-linear tabular baseline; often hard to beat in practice

|   |
| - |

B4

|   |
| - |

Spatial CNN

|   |
| - |

Ladder at the last step only (or a short stack) `[C, P]`

|   |
| - |

How much is visible in the book's shape without temporal modelling?

|   |
| - |

B5

|   |
| - |

Temporal model: TCN (LSTM as variant)

|   |
| - |

Flattened per-step features `[T, F]`

|   |
| - |

How much comes from time without price-axis structure?

|   |
| - |

B6

|   |
| - |

DeepLOB-style reimplementation

|   |
| - |

Level-indexed ladder

|   |
| - |

Reference point from the published literature
**Why B1 matters most:** if labels come from rules, a learned model must justify itself against the rule — by generalising to injection settings the rule misses, by ranking better, or by producing fewer false alerts at the same recall.
**Why TCN as the main temporal baseline (LSTM as a variant):** TCNs with causal dilated convolutions are straightforward to keep strictly causal, train in parallel, and have a fixed receptive field you can match to the window length. LSTM stays as a variant because it is the more familiar reference and is used in DeepLOB.
**Why B4 and B5 separately:** together they form the ablation that justifies a *spatial-temporal* model. If B4 alone ≈ proposed model, time isn't helping; if B5 alone ≈ proposed model, the price axis isn't helping.
**Fairness rule:** all models get the same splits, the same hyperparameter-search budget (number of trials), and their thresholds are tuned only on validation folds.

## H. Proposed spatial-temporal model (ST-LENS-Net)

ST-LENS-Net encodes each book snapshot along the price axis with convolutions, then models how those encodings evolve with a causal temporal network, and outputs a spoof-likeness logit plus side and location heads. It is an adaptation of the established CNN-then-temporal pattern (DeepLOB), specialised for spoof detection; its value is in the ablations, not the novelty of the blocks.

### Stages

1. **Input:** ladder tensor `[B, C, T, P]` and context series `[B, T, F]`.
2. **Side-symmetric spatial encoder (per time step).** Small 1-D convolutions along the price axis, applied with *shared weights* to the bid half and the mirrored ask half, then a cross-touch convolution over the full ladder. Why shared weights: a spoof on the bid is the mirror image of one on the ask, so sharing halves the parameters and doubles the effective training data per pattern.
3. **Multi-scale price kernels.** Parallel kernels of different widths (Inception-style) capture a single-bucket order and layered spoofs across several buckets. Why: spoof patterns occur at different spatial scales.
4. **Spatial output kept, not fully pooled.** Produce a per-bucket feature map `[B, D, T, P']` *and* a pooled vector `[B, D, T]`. Why: the location head and the explanation heatmaps need to know *where* on the ladder the signal is.
5. **Context fusion.** Concatenate the pooled spatial vector with the projected context series per time step.
6. **Temporal encoder.** Causal dilated TCN (primary) or GRU/LSTM (variant), receptive field ≥ T. Why causal: the same model must run in live mode, where the future does not exist.
7. **Temporal attention pooling.** Learned weights over time steps produce the window embedding. Why: lets the model focus on the cancellation moment; the weights are also a (weak) explanation signal.
8. **Heads.**
   - Main: spoof-likeness logit (binary).
   - Auxiliary: side (bid/ask/none) and location (which bucket), trained only on episodes with known location.

### Training choices

|   |
| - |

Choice

|   |
| - |

Proposal

|   |
| - |

Why

|   |
| - |

Loss

|   |
| - |

Weighted BCE (focal loss as variant) + down-weighted auxiliary losses

|   |
| - |

Rare positives; aux tasks act as regularisers that force the model to localise

|   |
| - |

Optimiser

|   |
| - |

AdamW with early stopping on validation PR-AUC

|   |
| - |

Standard, robust default

|   |
| - |

Regularisation

|   |
| - |

Dropout, weight decay, small model width to start

|   |
| - |

Limited effective positives → overfitting risk

|   |
| - |

Augmentation

|   |
| - |

Bid/ask mirroring with label flip; small size jitter

|   |
| - |

Exploits symmetry; cheap and label-safe

|   |
| - |

Calibration

|   |
| - |

Temperature scaling on validation fold

|   |
| - |

Turns logits into probabilities usable for risk scoring

|   |
| - |

Seeds

|   |
| - |

≥ 3 seeds per configuration

|   |
| - |

Deep models on rare events vary a lot run-to-run

### Planned ablations (the actual research contribution)

- remove spatial encoder (→ ≈ B5) · remove temporal encoder (→ ≈ B4)
- price-grid vs level-indexed price axis
- with vs without flow channels (`add_vol`, `cancel_vol`, `exec_vol`)
- shared vs separate bid/ask weights
- with vs without auxiliary heads
- train on injected labels → test on heuristic labels, and vice versa

## I. Evaluation methodology

Models are evaluated with **rolling walk-forward splits by calendar day**, with purge/embargo gaps, using precision-recall and alert-budget metrics rather than accuracy. Accuracy is meaningless when almost every window is negative.

### Walk-forward protocol

Fold k: train on days [1 … k], validate on day k+1, test on day k+2, with an embargo gap between each block; then roll forward one day. Hyperparameters and thresholds are chosen on validation only; test days are never looked at during development.
**Why walk-forward rather than random K-fold:** markets change (volatility, liquidity, participants). Walk-forward simulates the real use: train on the past, deploy on the future. Random K-fold puts near-identical neighbouring windows in train and test and reports fantasy numbers.
**Final hold-out:** reserve the last block of days as a lock-box test, evaluated once at the end of the project.

### Metrics

|   |
| - |

Level

|   |
| - |

Metric

|   |
| - |

Why

|   |
| - |

Window

|   |
| - |

PR-AUC (primary), ROC-AUC (secondary)

|   |
| - |

PR-AUC reflects performance on the rare positive class; ROC-AUC looks flattering under imbalance

|   |
| - |

Window

|   |
| - |

Recall at fixed false-alert rate (e.g. alerts per hour budget)

|   |
| - |

How surveillance teams actually operate: limited analyst time

|   |
| - |

Episode

|   |
| - |

Episode recall: share of spoof episodes with ≥ 1 alert during their span

|   |
| - |

An analyst cares about catching the episode, not every window

|   |
| - |

Episode

|   |
| - |

Detection latency: time from cancellation to first alert

|   |
| - |

Usefulness in near-real-time

|   |
| - |

Alert

|   |
| - |

Precision of alerts after de-duplication

|   |
| - |

Measures alert fatigue

|   |
| - |

Calibration

|   |
| - |

Brier score, reliability diagram

|   |
| - |

Risk scores must mean what they say

|   |
| - |

Localisation

|   |
| - |

Side accuracy, location hit rate on true positives

|   |
| - |

Validates the aux heads and explanations

### Generalisation tests

- **Injection-parameter shift:** train on some spoof sizes/durations/distances, test on held-out ranges.
- **Label-source shift:** train on injected, evaluate on heuristic labels.
- **Cross-symbol:** e.g. train on one pair, test on another from the same exchange.
- **Cross-regime:** report results separately for high- and low-volatility days.

### Statistical reporting

Report mean and spread across folds and seeds; use block bootstrap over *days* (not windows) for confidence intervals, because windows within a day are dependent. A difference smaller than the across-seed spread is not a finding.

## J. Data leakage prevention strategy

Leakage is prevented by construction (causal code, time-based splits, per-split injection) and then verified by automated tests, because in LOB research a leak usually shows up as "suspiciously good" results rather than an error.

|   |
| - |

Leakage route

|   |
| - |

How it happens

|   |
| - |

Prevention

|   |
| - |

Automated check

|   |
| - |

Look-ahead in features

|   |
| - |

Rolling stat uses a centred window or a future value

|   |
| - |

All features are incremental/causal state machines

|   |
| - |

**Truncation test:** feature at t computed on data up to t equals feature at t computed on the full day

|   |
| - |

Normalisation leakage

|   |
| - |

Z-score uses statistics of the whole dataset

|   |
| - |

Fit scalers on the training block only; store them as fold artefacts

|   |
| - |

Test that scaler fit timestamps ≤ training end

|   |
| - |

Overlapping windows across splits

|   |
| - |

Window ending just after split boundary contains pre-boundary data, label horizon crosses boundary

|   |
| - |

Embargo gap ≥ window length + label tolerance between train/val/test

|   |
| - |

Assert no sample's time span intersects another split's span

|   |
| - |

Episode split across sets

|   |
| - |

One spoof episode yields windows in both train and test

|   |
| - |

Group samples by episode id; assign whole episodes to one split

|   |
| - |

Assert episode ids are disjoint across splits

|   |
| - |

Injector leakage

|   |
| - |

Same random injection seeds/patterns in train and test

|   |
| - |

Inject independently per split with different seeds; held-out parameter ranges for test

|   |
| - |

Log injection configs per split in MLflow

|   |
| - |

Injection artefacts

|   |
| - |

Injected orders have tell-tale signatures (round sizes, impossible timing)

|   |
| - |

Sample sizes/timings from real large-order distributions

|   |
| - |

**Artefact probe:** a classifier trying to tell injected large orders from real large orders (ignoring cancel behaviour) should be near chance

|   |
| - |

Label-definition features

|   |
| - |

Feature directly encodes the heuristic rule

|   |
| - |

Report with/without those features

|   |
| - |

Feature list per experiment logged and diffed

|   |
| - |

Tuning on test

|   |
| - |

Repeatedly checking test results

|   |
| - |

Thresholds/hyperparameters from validation only; lock-box final test

|   |
| - |

Test-set evaluation function requires an explicit `--final` flag and logs each use

|   |
| - |

Train/serve skew

|   |
| - |

Offline and online feature code differ

|   |
| - |

Single shared implementation (section L)

|   |
| - |

Replay inference output equals offline batch output on the same day
**Why a truncation test is the most valuable single test:** it catches nearly every causal-feature bug automatically, including ones introduced later by AI-generated code that silently uses a pandas operation over the full column.

## K. Explainability strategy

Every alert carries two kinds of explanation: **evidence** (measurable facts about the book, independent of the model) and **attribution** (where on the ladder and when the model's score came from). Evidence is what an analyst trusts; attribution shows whether the model looked at the right thing.

|   |
| - |

Layer

|   |
| - |

Method

|   |
| - |

Output shown to analyst

|   |
| - |

Why

|   |
| - |

Evidence / reason codes

|   |
| - |

Rule-style checks run on the alert window

|   |
| - |

"Bid +6 ticks: size 9× local median, rested \~seconds, cancelled with no execution, ask-side buys followed"

|   |
| - |

Model-independent and auditable; the same language surveillance analysts use

|   |
| - |

Gradient attribution (deep models)

|   |
| - |

Integrated Gradients over the `[C, T, P]` input

|   |
| - |

Heatmap over time × price ladder, per channel

|   |
| - |

Faithful to the model's computation; maps directly onto the ladder picture

|   |
| - |

Occlusion

|   |
| - |

Mask one price bucket or time slice, re-score

|   |
| - |

"Score drops most when bucket ask+6 is removed"

|   |
| - |

Intuitive, model-agnostic cross-check of gradients

|   |
| - |

Temporal attention weights

|   |
| - |

Read from the attention-pooling layer

|   |
| - |

Highlighted time steps

|   |
| - |

Cheap, but **not** a faithful explanation on its own; shown as secondary only

|   |
| - |

Tree attribution (XGBoost)

|   |
| - |

TreeSHAP

|   |
| - |

Top contributing features per alert

|   |
| - |

Exact for tree models; lets you compare what classical vs deep models rely on

|   |
| - |

Global

|   |
| - |

Ablation results, SHAP summaries, attribution averaged over true positives

|   |
| - |

Model card in the dashboard

|   |
| - |

Shows what the model depends on overall
**Validating explanations (not just displaying them):**

- **Localisation check:** on injected episodes the true spoof bucket is known, so measure how often the attribution peak lands on it. This turns explainability into a measurable result.
- **Sanity check:** attributions of a model with randomised weights should look different from the trained model's; if not, the method is showing input structure, not model reasoning.

Integrated Gradients can be implemented directly in PyTorch or via the Captum library (an extra dependency to approve).

## L. Real-time and replay inference architecture

Replay mode is built first and is the main mode for the project; live mode reuses every component and only swaps the event source. Why replay first: it is deterministic, testable, and lets you "re-live" any historical day at any speed for demos and debugging.

### Components

1. **EventSource (interface).** `FileReplaySource` reads canonical events from Parquet in timestamp order, at 1×, N× or max speed. `WebSocketSource` (later) wraps an exchange adapter. Both emit the same event objects.
2. **BookBuilder** (same class as offline) maintains the book and detects gaps.
3. **StreamingFeatureState** keeps the rolling statistics and the ring buffer of the last T grid steps. It is the *same* code used offline, run incrementally. Why: guarantees no train/serve skew.
4. **ModelRunner** loads a versioned model and its fold scaler/calibrator from the MLflow registry, scores each new window on CPU in batches of one. Inference runs every grid step or every k steps, configurable.
5. **RiskScorer** turns model output into a risk score (below).
6. **AlertManager** applies alert rules, attaches explanation, writes to the alert store.
7. **Stores:** scores time series and alerts in Parquet/SQLite (DuckDB is an optional addition for fast queries). The dashboard reads only from these stores.

### Risk scoring

|   |
| - |

Step

|   |
| - |

Rule

|   |
| - |

Why

|   |
| - |

Calibrate

|   |
| - |

p = sigmoid(logit / temperature)

|   |
| - |

Makes the score an interpretable probability-like value

|   |
| - |

Smooth

|   |
| - |

Exponential moving average of p over recent steps

|   |
| - |

Suppresses single-step flicker

|   |
| - |

Band

|   |
| - |

Low / Medium / High bands from thresholds chosen on validation to hit an alert budget

|   |
| - |

Analysts act on bands, not decimals; thresholds tied to workload

|   |
| - |

Combine with evidence

|   |
| - |

Score shown alongside reason codes; High requires at least one supporting reason code

|   |
| - |

Avoids high-risk alerts with no human-checkable evidence

### Alert generation

- **Hysteresis:** open an alert when the score crosses the High threshold, close it only when it falls below a lower threshold. Why: prevents on/off chatter.
- **Cooldown and de-duplication:** one alert per symbol/side/episode window; repeated triggers update the existing alert.
- **Alert record:** id, symbol, side, start/end time, peak score, band, location bucket, reason codes, attribution artefact path, model version, status (open / reviewed / dismissed / escalated), analyst note.
- **Feedback loop:** analyst decisions are stored and can later become additional evaluation labels.

### Performance targets

Record measured per-step latency and throughput in replay; do not set a target until the first measurement exists. Streamlit is not a low-latency system, so the scoring loop runs as a separate process from the dashboard.

## M. Dashboard architecture

The Streamlit dashboard is a **read-only view** over the score, alert and experiment stores; its only write is the analyst's review decision on an alert, and it never runs model training or the scoring loop itself. Why: Streamlit re-runs the script on every interaction, so heavy work inside it would make the UI slow and the results non-reproducible.

### Pages

|   |
| - |

Page

|   |
| - |

Shows

|   |
| - |

Built with

|   |
| - |

Market monitor

|   |
| - |

Order-book heatmap (time × price, colour = depth), mid price, risk score timeline with band shading, alert markers

|   |
| - |

Plotly heatmap + line charts, auto-refresh during replay

|   |
| - |

Alert queue

|   |
| - |

Sortable/filterable table: time, symbol, side, band, peak score, status

|   |
| - |

`st.dataframe` with filters

|   |
| - |

Alert detail

|   |
| - |

Book replay around the alert, reason codes, attribution heatmap aligned to the ladder, analyst decision buttons and notes

|   |
| - |

Plotly + form that writes status back to the alert store

|   |
| - |

Model and experiments

|   |
| - |

Active model version, walk-forward metrics per fold, ablation table, calibration plot, link to MLflow UI

|   |
| - |

Reads MLflow tracking data

|   |
| - |

Data health

|   |
| - |

Feed gaps, re-syncs, quarantine counts, label prevalence per day

|   |
| - |

Reads validation logs

### Design decisions

- **Process separation:** scorer process → stores → dashboard. Why: each can restart independently; the dashboard can show a historical day without the scorer running.
- **Caching:** `st.cache_data` for loaded day files and `st.cache_resource` for connections. Why: avoids re-reading Parquet on every click.
- **Downsampling for plots:** heatmaps over a whole day are too large for the browser; aggregate to a display resolution and load full resolution only for the alert-detail window.
- **Wording:** the UI says "spoof-like pattern" and shows the limitation note on every alert detail page. Why: prevents over-interpretation in a research demo.

## N. Project directory structure

A `src/` layout with one package per pipeline stage, configs separated from code, and tests mirroring the package. Why `src/`: it forces tests to import the installed package, catching packaging mistakes early; one folder per stage maps directly onto the development phases.

```
st-lens/
├── pyproject.toml          # dependencies, Ruff + pytest config
├── README.md
├── Dockerfile
├── docker-compose.yml      # scorer, dashboard, mlflow services
├── .github/workflows/ci.yml  # ruff + pytest on every push
├── configs/
│   ├── data/               # exchange, symbols, dates, grid step
│   ├── features/           # feature groups on/off, window sizes
│   ├── labels/             # injection params, heuristic thresholds
│   ├── models/             # one file per model
│   └── experiments/        # walk-forward fold definitions
├── data/                   # git-ignored
│   ├── raw/                # exchange/symbol/date, untouched
│   ├── canonical/          # validated canonical events
│   ├── quarantine/
│   ├── processed/          # grid books, features, tensors
│   └── labels/
├── src/stlens/
│   ├── ingestion/          # adapters per exchange, recorder
│   ├── schemas/            # Pandera schemas, event dataclasses
│   ├── book/               # BookBuilder, gap detection, resampling, event attribution
│   ├── features/           # one module per feature group, causal state classes
│   ├── labeling/           # injector, heuristic rule engine, episode registry
│   ├── datasets/           # window builder, splits, PyTorch Dataset
│   ├── models/
│   │   ├── classical/      # logistic, XGBoost wrappers
│   │   ├── baselines/      # spatial CNN, TCN, LSTM, DeepLOB-style
│   │   └── stlens_net/     # proposed model
│   ├── training/           # trainer, losses, calibration
│   ├── evaluation/         # walk-forward runner, metrics, bootstrap
│   ├── tracking/           # MLflow helpers
│   ├── inference/          # event sources, streaming state, model runner
│   ├── scoring/            # risk scorer, alert manager, alert store
│   ├── explain/            # reason codes, IG, occlusion, SHAP
│   └── utils/              # time, logging, config loading
├── dashboard/
│   ├── app.py
│   └── pages/              # monitor, alerts, alert_detail, models, data_health
├── scripts/                # thin CLIs: record, build, train, evaluate, replay
├── notebooks/              # exploration only, never imported
└── tests/
    ├── unit/               # mirrors src/stlens
    ├── leakage/            # truncation, split-overlap, scaler-fit tests
    ├── integration/        # small end-to-end on a fixture day
    └── fixtures/           # tiny hand-made order books with known answers
```

**Testing priorities:** BookBuilder on hand-made books with known final state; event attribution (add/cancel/execute) on crafted sequences; the leakage suite in J; injector produces exactly the episodes it logs; replay output equals offline batch output.
**Why notebooks are never imported:** code that matters lives in `src/` with tests; notebooks are for looking at data. This keeps AI-generated exploratory code from leaking into the pipeline.

## O. Development phases

Thirteen phases, each ending in a gate: a concrete check that must pass before the next phase starts. Why gates: with AI-assisted coding it is easy to move fast on top of a broken foundation; gates force each layer to be verified first. Phases 0–5 are the foundation and will take the most care; the models are comparatively quick once the data is right.

1. **Setup and literature review.** Repo, `pyproject.toml`, Ruff, pytest, CI; read DeepLOB, OFI and existing spoof-detection literature; pick exchange and symbols; check the data source's terms of use. *Gate:* CI green on an empty package; a one-page related-work note stating what is and isn't new.
2. **Ingestion and schema validation.** One exchange adapter, raw recorder, canonical events, Pandera schemas, quarantine. *Gate:* a few days of data recorded; validation report shows gap/re-sync counts.
3. **Book reconstruction and event attribution.** BookBuilder, gap handling, grid resampling, add/cancel/execute split. *Gate:* unit tests on hand-made books pass; reconstructed top-of-book cross-checked against the exchange's own snapshots.
4. **Feature engineering.** Groups 1–7 as causal state classes; ladder tensor. *Gate:* truncation test passes for every feature; exploratory plots look sane.
5. **Labelling.** Injector with hard negatives, heuristic rule engine, episode registry. *Gate:* injector round-trip test (logged episodes = recovered episodes); artefact probe near chance.
6. **Dataset and splits.** Window builder, walk-forward folds with embargo, PyTorch Dataset. *Gate:* leakage suite (overlap, episode-disjointness, scaler-fit) passes.
7. **Classical baselines + MLflow.** B0–B3 with full walk-forward and tracking. *Gate:* reproducible metrics per fold logged in MLflow.
8. **Deep baselines.** B4 spatial CNN, B5 TCN/LSTM, B6 DeepLOB-style. *Gate:* each overfits a tiny batch (sanity), then runs full walk-forward.
9. **ST-LENS-Net and ablations.** Proposed model, calibration, ablation grid, multiple seeds. *Gate:* complete comparison table with spreads; conclusions only where differences exceed seed variance.
10. **Explainability.** Reason codes, Integrated Gradients, occlusion, SHAP; localisation check. *Gate:* localisation hit rate measured on injected episodes.
11. **Replay inference, risk scoring, alerts.** Event sources, streaming state, scorer, alert manager, stores. *Gate:* replay output equals offline output for a test day.
12. **Dashboard.** Pages from section M. *Gate:* full replay of a held-out day viewable end-to-end with alerts and explanations.
13. **Docker and hardening.** Dockerfile, compose (scorer, dashboard, MLflow), integration test in CI; optional live WebSocket source. *Gate:* fresh clone → `docker compose up` → working demo.

**How to work with AI assistance per phase:** ask for one module at a time with its tests, run the tests yourself, and read every function that touches time (rolling windows, shifts, merges, splits) line by line. Those are where generated code most often leaks the future.