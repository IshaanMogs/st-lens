"""Render ``docs/results.md`` from ``results.json``. Every number comes from the file."""

import json
import math
from pathlib import Path

ORDER = [
    "B0_prior",
    "B1_rule_engine",
    "B2_logistic",
    "B2_logistic_no_rule_feats",
    "B3_xgboost",
    "B3_xgboost_no_rule_feats",
    "B4_spatial_cnn",
    "B5_temporal_tcn",
    "B6_deeplob_lite",
    "STLENS_spatial_only",
    "STLENS_temporal_only",
    "STLENS_full",
]


def _fmt(cell: dict) -> str:
    m, s, n = cell["mean"], cell["std"], cell["n"]
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "n/a"
    return f"{m:.3f} ± {s:.3f}" if n > 1 else f"{m:.3f}"


def _table(summary: dict) -> list[str]:
    cols = [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("recall_at_budget", "Recall@budget"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("episode_recall", "Episode recall"),
        ("hard_negative_flag_rate", "Hard-neg flag rate"),
        ("median_latency_s", "Latency (s)"),
    ]
    lines = ["| Model | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name in [n for n in ORDER if n in summary] + [n for n in summary if n not in ORDER]:
        row = summary[name]
        lines.append(f"| {name} | " + " | ".join(_fmt(row[k]) for k, _ in cols) + " |")
    return lines


def render(results_path: Path) -> str:
    r = json.loads(results_path.read_text(encoding="utf-8"))
    cfg = r["config"]
    d = cfg["data"]
    out = [
        "# ST-LENS fast-track prototype: results",
        "",
        f"*Generated automatically from `{results_path.name}` at {r['generated_utc']}. "
        "Do not edit numbers by hand; re-run `python scripts/run_experiment.py --final`.*",
        "",
        "> **All data is SYNTHETIC.** A zero-intelligence simulated order book with injected "
        "spoof-like episodes. These numbers say nothing about real markets, real spoofing or "
        'trader intent. Labels are injected ground truth; outputs are "spoof-like pattern" scores.',
        "",
        "## Setup",
        "",
        f"- Days: {d['n_days']} simulated sessions of {d['sim_steps_per_day'] * 100 / 60_000:.0f} min "
        f"(train/val/test by day, in time order: "
        f"{r['data']['train']['days']} / {r['data']['val']['days']} / {r['data']['test']['days']}).",
        f"- Grid 250 ms; window T = {d['window']} steps; label horizon h = {d['horizon']} steps.",
        f"- Seeds per learned deep model: {list(cfg['seeds'])} (mean ± std across seeds; "
        "classical models are single runs; B0 has one random run per seed).",
        f"- Thresholds and temperatures chosen on validation only. Recall@budget uses "
        f"{cfg['alerts_per_hour']:.0f} flagged windows/hour, threshold set on validation.",
        f"- Final test evaluation performed: **{r['final_test_evaluated']}**. Runtime {r['runtime_s']} s.",
        "",
        "## Data",
        "",
        "| Split | Windows | Positive windows | Prevalence | Positive episodes | Spoof / HN-cancel / HN-executed | Book gaps |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        x = r["data"][s]
        e = x["episodes"]
        out.append(
            f"| {s} | {x['windows']} | {x['positive_windows']} | {x['prevalence']:.4f} | "
            f"{x['positive_episodes']} | {e['spoof']} / {e['hn_cancel_no_payoff']} / {e['hn_executed']} | "
            f"{x['book_gaps']} |"
        )
    out += ["", "## Validation results", "", *_table(r["summary"]["val"])]
    if r["final_test_evaluated"]:
        out += ["", "## Test results (evaluated once)", "", *_table(r["summary"]["test"])]
    if r.get("explanation"):
        out += [
            "",
            "## Explanation checks (ST-LENS full, best seed by validation, test true positives)",
            "",
        ]
        out += [
            f"- `{k}`: {v:.3f}" if isinstance(v, float) else f"- `{k}`: {v}"
            for k, v in r["explanation"].items()
        ]
    if r.get("replay"):
        out += [
            "",
            "## Offline replay (test days)",
            "",
            "| Day | Alerts | Alerts overlapping a positive | Positive episodes | Episodes with an alert |",
            "|---|---|---|---|---|",
        ]
        for day, x in r["replay"].items():
            out.append(
                f"| {day} | {x['alerts']} | {x['hits']} | {x['positive_episodes']} | {x['episodes_alerted']} |"
            )
    out += [
        "",
        "## How to read this",
        "",
        "- B0's PR-AUC is the chance level (≈ prevalence).",
        "- B1 is the transparent rule. A learned model only adds value if it beats B1 and B3.",
        "- Differences smaller than the across-seed spread are not findings.",
        "- `*_no_rule_feats` rows drop window features that encode the heuristic rule (spec E/J).",
        "",
        "See `docs/fast_track.md` for limitations and deferred work.",
        "",
    ]
    return "\n".join(out)


def write(results_path: Path, out_path: Path) -> None:
    out_path.write_text(render(results_path), encoding="utf-8")
