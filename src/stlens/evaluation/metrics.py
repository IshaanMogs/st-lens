"""Evaluation metrics (spec I): window, alert-budget, episode and calibration levels.

Thresholds are always chosen on validation scores and then applied unchanged to test.
"""

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class EvalResult:
    pr_auc: float
    roc_auc: float
    prevalence: float
    threshold: float
    precision: float
    recall: float
    alerts_per_hour: float
    tp: int
    fp: int
    fn: int
    tn: int
    episode_recall: float
    hard_negative_flag_rate: float
    median_latency_s: float
    brier: float | None
    n_windows: int
    n_pos_episodes: int

    def as_dict(self) -> dict:
        return asdict(self)


def threshold_for_budget(scores: np.ndarray, hours: float, alerts_per_hour: float) -> float:
    """Smallest threshold whose number of flagged windows stays within the budget."""
    k = max(int(np.floor(alerts_per_hour * hours)), 1)
    if k >= len(scores):
        return float(np.min(scores))
    return float(np.sort(scores)[::-1][k - 1])


@dataclass
class EvalSet:
    """Scores for one split, aligned with window metadata."""

    y: np.ndarray  # [N] 0/1
    step: np.ndarray  # [N] grid step of the window end
    day: np.ndarray  # [N] day position
    kind: np.ndarray  # [N] episode kind at that step ("" if none)
    episode: np.ndarray  # [N] episode key ("day:idx") or ""
    grid_s: float  # seconds per grid step
    hours: float  # observed hours represented by the windows


def episode_metrics(flag: np.ndarray, es: EvalSet) -> tuple[float, float, float, int]:
    """Episode recall, hard-negative flag rate, median detection latency (s), #positives."""
    pos_hit: dict[str, bool] = {}
    hn_hit: dict[str, bool] = {}
    first: dict[str, int] = {}
    start: dict[str, int] = {}
    for i in range(len(flag)):
        ep = es.episode[i]
        if not ep:
            continue
        if es.kind[i] == "spoof" and es.y[i] == 1:
            pos_hit.setdefault(ep, False)
            start[ep] = min(start.get(ep, es.step[i]), es.step[i])
            if flag[i]:
                pos_hit[ep] = True
                first[ep] = min(first.get(ep, es.step[i]), es.step[i])
        elif es.kind[i] in ("hn_executed", "hn_cancel_no_payoff"):
            hn_hit.setdefault(ep, False)
            hn_hit[ep] |= bool(flag[i])
    recall = float(np.mean(list(pos_hit.values()))) if pos_hit else float("nan")
    hn_rate = float(np.mean(list(hn_hit.values()))) if hn_hit else float("nan")
    lat = [(first[e] - start[e]) * es.grid_s for e in first]
    return recall, hn_rate, float(np.median(lat)) if lat else float("nan"), len(pos_hit)


def evaluate(scores: np.ndarray, es: EvalSet, threshold: float, probs: bool = False) -> EvalResult:
    y = es.y.astype(int)
    flag = scores >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    ep_recall, hn_rate, latency, n_pos = episode_metrics(flag, es)
    return EvalResult(
        pr_auc=float(average_precision_score(y, scores)) if y.any() else float("nan"),
        roc_auc=float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan"),
        prevalence=float(y.mean()),
        threshold=float(threshold),
        precision=tp / (tp + fp) if tp + fp else float("nan"),
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        alerts_per_hour=float(flag.sum() / es.hours) if es.hours else float("nan"),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        episode_recall=ep_recall,
        hard_negative_flag_rate=hn_rate,
        median_latency_s=latency,
        brier=float(brier_score_loss(y, np.clip(scores, 0, 1))) if probs else None,
        n_windows=len(y),
        n_pos_episodes=n_pos,
    )


def reliability(
    scores: np.ndarray, y: np.ndarray, bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in pairwise(edges):
        m = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if m.any():
            out.append((float(scores[m].mean()), float(y[m].mean()), int(m.sum())))
    return out
