"""Transparent heuristic rule engine: weak labels (spec F, Source 2) and baseline B1.

Signature, per side, on attributed grid flows only (never on injection ground truth):

1. a large relative ADD (>= ``min_rel_add`` x typical level size) at least one tick away
   from the touch (far buckets);
2. a CANCEL of at least ``min_cancel_frac`` of it at the same or an adjacent bucket
   within ``max_life_steps``;
3. little EXECUTION at those buckets in between (< ``max_exec_frac`` of the add);
4. opposite-side aggressive volume in between (soft-weighted).

A detected pattern is stamped at its cancellation step ``c`` and uses only grid rows in
``[a, c]``, so the score at ``t`` depends on data up to ``t`` only. Thresholds are
config values; they are not tuned on test data.
"""

from dataclasses import dataclass

import numpy as np

from stlens.book.builder import GridDay
from stlens.features.ladder import size_scale


@dataclass(frozen=True)
class RuleParams:
    min_rel_add: float = 8.0
    min_cancel_frac: float = 0.7
    max_exec_frac: float = 0.2
    max_life_steps: int = 24  # 6 s at 250 ms
    opp_vol_scale: float = 0.2  # opposite volume giving a 0.5 soft weight
    flag_threshold: float = 4.0  # score >= threshold -> weak label / alert


DEFAULT_RULES = RuleParams()


@dataclass(frozen=True)
class RulePattern:
    side: str  # "bid" / "ask"
    add_step: int
    cancel_step: int
    bucket: int  # ladder index at the cancellation step
    rel_add: float
    score: float


def detect_patterns(g: GridDay, params: RuleParams = DEFAULT_RULES) -> list[RulePattern]:
    """All rule patterns in a day, each stamped at its cancellation step."""
    H = g.half_buckets
    scale = size_scale(g)
    patterns: list[RulePattern] = []
    sides = {
        "bid": (list(range(0, H - 1)), g.buy_vol),  # opposite: buyers lifting asks
        "ask": (list(range(H + 1, 2 * H)), g.sell_vol),  # opposite: sellers hitting bids
    }
    for side, (far, opp_vol) in sides.items():
        for a, b in zip(
            *np.nonzero(g.add_vol[:, far] >= params.min_rel_add * scale[:, None]), strict=True
        ):
            bucket = far[b]
            add = g.add_vol[a, bucket]
            near = [x for x in (bucket - 1, bucket, bucket + 1) if 0 <= x < 2 * H]
            for c in range(a + 1, min(a + params.max_life_steps, g.n_steps - 1) + 1):
                cancel = g.cancel_vol[c, near].max()
                if cancel >= params.min_cancel_frac * add:
                    ex = g.exec_vol[a : c + 1][:, near].sum()
                    if ex < params.max_exec_frac * add:
                        opp = float(opp_vol[a : c + 1].sum())
                        weight = opp / (opp + params.opp_vol_scale)
                        rel = float(add / scale[a])
                        cb = near[int(np.argmax(g.cancel_vol[c, near]))]
                        patterns.append(RulePattern(side, int(a), c, cb, rel, rel * weight))
                    break
    return sorted(patterns, key=lambda p: (p.cancel_step, p.side, p.bucket))


def rule_scores(
    g: GridDay, horizon: int, params: RuleParams = DEFAULT_RULES
) -> tuple[np.ndarray, list[RulePattern]]:
    """B1 score per grid step: max score of patterns cancelled in ``(t - horizon, t]``."""
    patterns = detect_patterns(g, params)
    scores = np.zeros(g.n_steps)
    for p in patterns:
        end = min(p.cancel_step + horizon, g.n_steps)
        scores[p.cancel_step : end] = np.maximum(scores[p.cancel_step : end], p.score)
    return scores, patterns
