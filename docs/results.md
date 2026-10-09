# ST-LENS fast-track prototype: results

*Generated automatically from `results.json` at 2026-10-08T21:38:32.581022+00:00. Do not edit numbers by hand; re-run `python scripts/run_experiment.py --final`.*

> **All data is SYNTHETIC.** A zero-intelligence simulated order book with injected spoof-like episodes. These numbers say nothing about real markets, real spoofing or trader intent. Labels are injected ground truth; outputs are "spoof-like pattern" scores.

## Setup

- Days: 8 simulated sessions of 30 min (train/val/test by day, in time order: [0, 1, 2, 3, 4] / [5] / [6, 7]).
- Grid 250 ms; window T = 40 steps; label horizon h = 8 steps.
- Seeds per learned deep model: [0, 1, 2] (mean ± std across seeds; classical models are single runs; B0 has one random run per seed).
- Thresholds and temperatures chosen on validation only. Recall@budget uses 600 flagged windows/hour, threshold set on validation.
- Final test evaluation performed: **True**. Runtime 1021.9 s.

## Data

| Split | Windows | Positive windows | Prevalence | Positive episodes | Spoof / HN-cancel / HN-executed | Book gaps |
|---|---|---|---|---|---|---|
| train | 17905 | 572 | 0.0319 | 143 | 208 / 93 / 104 | 0 |
| val | 7161 | 232 | 0.0324 | 29 | 40 / 23 / 19 | 0 |
| test | 14322 | 536 | 0.0374 | 67 | 92 / 40 / 45 | 0 |

## Validation results

| Model | PR-AUC | ROC-AUC | Recall@budget | Precision | Recall | Episode recall | Hard-neg flag rate | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| B0_prior | 0.033 ± 0.001 | 0.490 ± 0.013 | 0.037 ± 0.015 | 0.036 ± 0.003 | 0.621 ± 0.287 | 0.977 ± 0.033 | 0.929 ± 0.101 | 0.167 ± 0.236 |
| B1_rule_engine | 0.075 | 0.616 | 0.151 | 0.116 | 0.267 | 0.345 | 0.214 | 0.000 |
| B2_logistic | 0.048 | 0.619 | 0.091 | 0.068 | 0.151 | 0.310 | 0.119 | 0.000 |
| B2_logistic_no_rule_feats | 0.062 | 0.628 | 0.108 | 0.066 | 0.272 | 0.517 | 0.071 | 0.000 |
| B3_xgboost | 0.047 | 0.646 | 0.052 | 0.057 | 0.483 | 0.759 | 0.405 | 0.000 |
| B3_xgboost_no_rule_feats | 0.057 | 0.631 | 0.030 | 0.055 | 0.513 | 0.862 | 0.500 | 0.000 |
| B4_spatial_cnn | 0.109 ± 0.003 | 0.651 ± 0.020 | 0.214 ± 0.012 | 0.177 ± 0.005 | 0.220 ± 0.030 | 0.724 ± 0.028 | 0.349 ± 0.074 | 0.000 ± 0.000 |
| B5_temporal_tcn | 0.128 ± 0.004 | 0.804 ± 0.014 | 0.211 ± 0.012 | 0.149 ± 0.014 | 0.376 ± 0.060 | 0.724 ± 0.049 | 0.635 ± 0.062 | 0.167 ± 0.236 |
| B6_deeplob_lite | 0.167 ± 0.017 | 0.837 ± 0.018 | 0.270 ± 0.019 | 0.202 ± 0.028 | 0.381 ± 0.050 | 0.598 ± 0.086 | 0.421 ± 0.081 | 0.250 ± 0.000 |
| STLENS_spatial_only | 0.061 ± 0.001 | 0.698 ± 0.004 | 0.066 ± 0.009 | 0.077 ± 0.000 | 0.599 ± 0.009 | 0.701 ± 0.016 | 0.373 ± 0.022 | 0.000 ± 0.000 |
| STLENS_temporal_only | 0.198 ± 0.025 | 0.804 ± 0.010 | 0.241 ± 0.020 | 0.236 ± 0.049 | 0.274 ± 0.059 | 0.391 ± 0.065 | 0.135 ± 0.056 | 0.000 ± 0.000 |
| STLENS_full | 0.365 ± 0.034 | 0.920 ± 0.016 | 0.491 ± 0.050 | 0.414 ± 0.068 | 0.499 ± 0.059 | 0.621 ± 0.049 | 0.175 ± 0.030 | 0.000 ± 0.000 |

## Test results (evaluated once)

| Model | PR-AUC | ROC-AUC | Recall@budget | Precision | Recall | Episode recall | Hard-neg flag rate | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| B0_prior | 0.037 ± 0.002 | 0.487 ± 0.003 | 0.034 ± 0.010 | 0.035 ± 0.003 | 0.577 ± 0.312 | 0.915 ± 0.120 | 0.945 ± 0.078 | 0.250 ± 0.354 |
| B1_rule_engine | 0.086 | 0.616 | 0.131 | 0.109 | 0.164 | 0.224 | 0.153 | 0.000 |
| B2_logistic | 0.065 | 0.620 | 0.114 | 0.068 | 0.153 | 0.239 | 0.235 | 0.000 |
| B2_logistic_no_rule_feats | 0.070 | 0.634 | 0.104 | 0.068 | 0.272 | 0.478 | 0.376 | 0.000 |
| B3_xgboost | 0.061 | 0.637 | 0.091 | 0.061 | 0.491 | 0.701 | 0.435 | 0.000 |
| B3_xgboost_no_rule_feats | 0.057 | 0.620 | 0.050 | 0.055 | 0.459 | 0.657 | 0.459 | 0.000 |
| B4_spatial_cnn | 0.096 ± 0.003 | 0.636 ± 0.007 | 0.172 ± 0.005 | 0.149 ± 0.012 | 0.170 ± 0.020 | 0.597 ± 0.042 | 0.337 ± 0.087 | 0.000 ± 0.000 |
| B5_temporal_tcn | 0.114 ± 0.002 | 0.775 ± 0.008 | 0.165 ± 0.002 | 0.141 ± 0.008 | 0.304 ± 0.065 | 0.682 ± 0.090 | 0.573 ± 0.069 | 0.583 ± 0.118 |
| B6_deeplob_lite | 0.173 ± 0.008 | 0.833 ± 0.015 | 0.226 ± 0.022 | 0.207 ± 0.018 | 0.310 ± 0.069 | 0.507 ± 0.095 | 0.361 ± 0.064 | 0.250 ± 0.000 |
| STLENS_spatial_only | 0.056 ± 0.002 | 0.625 ± 0.003 | 0.063 ± 0.005 | 0.056 ± 0.003 | 0.368 ± 0.015 | 0.468 ± 0.019 | 0.357 ± 0.024 | 0.000 ± 0.000 |
| STLENS_temporal_only | 0.095 ± 0.008 | 0.735 ± 0.022 | 0.144 ± 0.011 | 0.134 ± 0.014 | 0.141 ± 0.048 | 0.234 ± 0.043 | 0.184 ± 0.040 | 0.292 ± 0.212 |
| STLENS_full | 0.193 ± 0.034 | 0.845 ± 0.012 | 0.257 ± 0.069 | 0.242 ± 0.053 | 0.236 ± 0.065 | 0.343 ± 0.095 | 0.188 ± 0.058 | 0.125 ± 0.177 |

## Explanation checks (ST-LENS full, best seed by validation, test true positives)

- `n_true_positive_windows`: 135
- `n_with_known_location`: 131
- `ig_localisation_hit_rate_pm1`: 0.893
- `occlusion_localisation_hit_rate_pm1`: 0.916
- `random_weights_ig_localisation_hit_rate_pm1`: 0.168
- `chance_hit_rate_pm1`: 0.150
- `mean_corr_trained_vs_random_ig`: -0.223
- `location_head_hit_rate_pm1`: 0.840
- `side_head_accuracy`: 0.044

## Offline replay (test days)

| Day | Alerts | Alerts overlapping a positive | Positive episodes | Episodes with an alert |
|---|---|---|---|---|
| 6 | 31 | 9 | 34 | 9 |
| 7 | 23 | 8 | 33 | 8 |

## How to read this

- B0's PR-AUC is the chance level (≈ prevalence).
- B1 is the transparent rule. A learned model only adds value if it beats B1 and B3.
- Differences smaller than the across-seed spread are not findings.
- `*_no_rule_feats` rows drop window features that encode the heuristic rule (spec E/J).

See `docs/fast_track.md` for limitations and deferred work.
