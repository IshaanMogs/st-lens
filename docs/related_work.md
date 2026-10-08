# ST-LENS: related work and novelty note (Phase 0)

*Written 2026-10-08. Each source is marked by what was actually checked: **full text**, **abstract/arXiv page**, or **index only** (bibliographic record).*

## 1. Established: not ours to claim

- **CNN over price levels, then a recurrent temporal layer.** DeepLOB (Zhang, Zohren & Roberts, *IEEE Trans. Signal Processing* 67(11), 2019; DOI 10.1109/TSP.2019.2907260; arXiv:1808.03668) is checked against the **full text**.
  - Architecture: convolutional layers, an Inception module, then LSTM.
  - Input: 10 levels per side, price and volume, so 40 features per timestamp.
  - Task: mid-price *movement prediction*, not manipulation detection.
  - Data: FI-2010 and one year of LSE data for 5 stocks.
- **Order Flow Imbalance.** Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events", *J. Financial Econometrics* 12(1):47–88, 2014; DOI 10.1093/jjfinec/nbt003; arXiv:1011.6402. Checked at **abstract + index**.
  - Over short intervals, price changes are mainly driven by OFI at the best quotes.
  - The relation is linear, with slope inversely proportional to depth.
  - Data: NYSE TAQ, 50 stocks.
- **FI-2010 benchmark.** Ntakaris, Magris, Kanniainen, Gabbouj & Iosifidis, *J. Forecasting* 37(8):852–866, 2018; DOI 10.1002/for.2543. Checked at **abstract + index**.
  - 5 Nasdaq Nordic stocks, 10 consecutive days, about 4M samples, 10 levels.
  - Mid-price labels only, normalised data. **No manipulation labels and no cancellation flags**, so it cannot be used to evaluate spoof detection.
- **Causal dilated TCNs** are a standard sequence-model component. ST-LENS uses them; it does not contribute them.

## 2. Already done: spoof detection with learned models on order-book data

| Work | Checked | Data | Labels | Model | Overlap with ST-LENS |
|---|---|---|---|---|---|
| Tuccella, Nadler & Şerban 2021, arXiv:2110.03687 | full text | Crypto **L2** (Bitfinex, Kraken; e.g. ETH/USD) | Rule-based: large cancel vs cumulative depth, near touch, volatility rise | GRU on 200 book updates | Deep sequence model + rule labels on **crypto L2**. Reports accuracy on 65/15/20 splits |
| Kularatnam & Stathaki 2024, arXiv:2403.13429 (AAAI-24 workshop) | full text (HTML) | US equities, 30 levels, 5 venues | Algorithmic weak labels (not described) | **Causal TCN** over stacked level×time snapshots | **TCN on LOB tensors for spoof detection.** Preliminary; accuracy/F1; no baselines |
| Lin & Yang 2025, arXiv:2508.17086 | full text (HTML) | LOBSTER, 3 NASDAQ stocks, 1 day, 5 levels | **Synthetic spoof/layering injected into real LOB data** | CNN/LSTM/Transformer encoders + contrastive learning; OC-SVM / IForest | **Injection into real books + deep encoders + PR-AUC.** Chronology of splits not stated |
| Fabre & Challet 2025, arXiv:2504.15908 | abstract | Crypto **L3** | None. Estimates "spoofability" (expected manipulation gain) | Hawkes features + NN predicting mid-price distribution | Shows **posting distance from the touch matters**, which supports the price-grid axis |

**Non-deep detectors, for context:**
- Tao, Day, Ling & Drapeau, *Quant. Finance* 22(8), 2022 (abstract): microstructural spoofing model; Wasserstein-distance monitoring on TMX L2.
- Leangarun, Tangamchit & Thajchayapong, *IEEE Access* 2021 (abstract via search only): AE/GAN anomaly detection on SET.
- Li, Polukarova & Ventre 2023, arXiv:2308.08683 (abstract): statistical-physics "momentum" measure on LUNA and BTC.

Kang, Mu & Ning 2023 (transformer graph learning for conspiracy spoofing) could not be accessed and is **not relied on**.

## 3. What ST-LENS can legitimately claim

Every major building block has precedent: CNN plus temporal stacks, TCNs for spoofing, injection into real books, rule-based labels on crypto L2, and the importance of posting distance. **ST-LENS-Net is not a novel architecture, and ST-LENS is not the first deep-learning spoof detector, nor the first on crypto data.**

> **No novelty claim is made at this stage.** The search in section 4 was not systematic. Novelty must not be claimed in any report, paper or presentation until a systematic search has been done and documented.

Within the literature reviewed so far, we did not identify a study combining all of the methodological elements below. If that holds after further verification, the intended contribution is the **combination as a careful empirical study**, not any single component:

1. **Public data, reproducible.** Binance L2 diff-depth plus trades, with an open, versioned labelling protocol.
2. **Hard-tested injection.** Injection that includes hard negatives (large orders that execute; cancels with no payoff), plus an **artefact probe** showing injected orders are not trivially separable.
3. **Leakage-controlled, multi-day walk-forward evaluation.** Embargo, episode-disjoint splits, a lock-box, and day-level block bootstrap. Prior studies above use a single day, within-sample splits, or unstated chronology.
4. **Comparison against the labelling rule itself (B1) and XGBoost**, with and without label-defining features. None of the four deep-learning works in section 2 reports this, as far as we could read.
5. **Ablations.** Price-grid vs level-indexed axis, inferred add/cancel/execute channels, spatial vs temporal contribution, and **cross-label-source transfer** (train injected, test heuristic).
6. **Measured explanation localisation** on injected episodes, not just displayed heatmaps.

A legitimate result includes a **negative** one, e.g. "XGBoost on window features matches the deep model". Such an outcome must be reported, not tuned away.

## 4. Limitations of this review and of the claim

- **The search was not systematic.** It used ad-hoc web search (about 10 queries, English, arXiv-heavy) on 2026-10-08. There was no protocol, no defined inclusion criteria, and no Google Scholar, SSRN, IEEE Xplore, ACM or citation-chaining search. Paywalled journal versions were not read. Relevant work, including 2025–2026 preprints and non-English work, may have been missed. **Novelty must not be claimed without further verification.**
- **Prior results are taken as reported by their authors.** None were reproduced.
- **The claim is about the evaluation, not the detection of spoofing.** Labels are synthetic or rule-derived. L2 data has no trader or order IDs, and intent is unobservable. Outputs are "spoof-like patterns".
- **Crypto ≠ regulated equities.** No generalisation to equities without new data.
