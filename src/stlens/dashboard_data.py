"""Read-only loaders for the dashboard (spec M). The dashboard never trains or scores;
it only reads artefacts written by ``scripts/run_experiment.py``."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DayView:
    day: int
    grid_times_us: np.ndarray
    mid: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    depth: np.ndarray  # [T, P]
    cancel_vol: np.ndarray  # [T, P]
    y: np.ndarray
    rule_score: np.ndarray
    prob: np.ndarray
    smooth: np.ndarray
    band: np.ndarray
    episodes: list[dict]


def load_results(root: Path) -> dict | None:
    path = root / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def list_days(root: Path) -> list[int]:
    return sorted(int(p.stem[3:]) for p in root.glob("day*.npz"))


def load_day(root: Path, day: int) -> DayView:
    z = np.load(root / f"day{day}.npz", allow_pickle=False)
    episodes = json.loads((root / f"day{day}_episodes.json").read_text(encoding="utf-8"))
    return DayView(day=day, episodes=episodes, **{k: z[k] for k in z.files})


def load_alerts(root: Path) -> list[dict]:
    path = root / "alerts.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_pr_curves(root: Path) -> dict:
    path = root / "pr_curves.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metrics_rows(results: dict, split: str) -> list[dict]:
    """Flat rows (model, metric means/stds) for a table."""
    rows = []
    for model, m in results["summary"].get(split, {}).items():
        row = {"model": model}
        for k, v in m.items():
            if isinstance(v, dict):
                row[k] = v["mean"]
                row[f"{k}_std"] = v["std"]
        rows.append(row)
    return rows


def confusion(results: dict, split: str, model: str) -> np.ndarray:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] summed over seeds/runs."""
    runs = results["per_run"][split][model]
    return np.array(
        [
            [sum(r["tn"] for r in runs), sum(r["fp"] for r in runs)],
            [sum(r["fn"] for r in runs), sum(r["tp"] for r in runs)],
        ]
    )


def downsample(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices for plotting long series in the browser."""
    return np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)
