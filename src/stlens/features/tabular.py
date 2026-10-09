"""Aggregated window features ``v`` for the classical baselines (spec D, E).

A window ending at grid step ``t`` covers rows ``[t - T + 1, t]``; only those rows are
read. Features are grouped so ablations and the "with/without rule-like features"
comparison (spec E, J) are possible: names in :data:`RULE_LIKE` encode (parts of) the
heuristic labelling rule and must be reported both ways.
"""

import numpy as np

from stlens.book.builder import GridDay
from stlens.features.ladder import CONTEXT

RULE_LIKE = (
    "bid_max_bucket_cancel_far",
    "ask_max_bucket_cancel_far",
    "bid_large_add_then_cancel",
    "ask_large_add_then_cancel",
)


def _names(H: int) -> list[str]:
    names: list[str] = []
    for s in ("bid", "ask"):
        names += [
            f"{s}_max_rel_size",
            f"{s}_last_max_rel_size",
            f"{s}_cancel_far_sum",
            f"{s}_add_far_sum",
            f"{s}_exec_sum",
            f"{s}_cancel_add_ratio_far",
            f"{s}_max_bucket_cancel_far",
            f"{s}_large_add_then_cancel",
        ]
    names += ["buy_vol_sum", "sell_vol_sum", "buy_vol_last_q", "sell_vol_last_q"]
    for c in CONTEXT:
        names += [f"{c}_mean", f"{c}_std", f"{c}_last"]
    return names


def window_features(
    g: GridDay,
    ladder: np.ndarray,
    context: np.ndarray,
    scale: np.ndarray,
    ends: np.ndarray,
    window: int,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix ``[len(ends), K]`` for windows ending at ``ends`` (each >= window-1)."""
    H = g.half_buckets
    rel = ladder[:, 5, :]
    halves = {"bid": (slice(0, H), slice(0, H - 1)), "ask": (slice(H, 2 * H), slice(H + 1, 2 * H))}
    rows = []
    for t in ends:
        lo = t - window + 1
        if lo < 0:
            raise ValueError(f"window ending at {t} needs {window} steps")
        sl = slice(lo, t + 1)
        f: list[float] = []
        for half, far in halves.values():
            add_far = g.add_vol[sl, far] / scale[sl, None]
            cancel_far = g.cancel_vol[sl, far] / scale[sl, None]
            # large add somewhere in the window followed by a large cancel later in it:
            # max over k of min(max add in [0, k], max cancel in (k, end])
            add_prefix = np.maximum.accumulate(add_far.max(1))
            cancel_suffix = np.maximum.accumulate(cancel_far.max(1)[::-1])[::-1]
            add_then_cancel = float(np.minimum(add_prefix[:-1], cancel_suffix[1:]).max())
            f += [
                float(rel[sl, half].max()),
                float(rel[t, half].max()),
                float(np.log1p(cancel_far.sum())),
                float(np.log1p(add_far.sum())),
                float(np.log1p((g.exec_vol[sl, half] / scale[sl, None]).sum())),
                float(cancel_far.sum() / (add_far.sum() + 1.0)),
                float(np.log1p(cancel_far.max())),
                float(np.log1p(add_then_cancel)),
            ]
        q = max(window // 4, 1)
        f += [
            float(np.log1p(g.buy_vol[sl].sum())),
            float(np.log1p(g.sell_vol[sl].sum())),
            float(np.log1p(g.buy_vol[t - q + 1 : t + 1].sum())),
            float(np.log1p(g.sell_vol[t - q + 1 : t + 1].sum())),
        ]
        ctx = context[sl]
        for j in range(ctx.shape[1]):
            f += [float(ctx[:, j].mean()), float(ctx[:, j].std()), float(ctx[-1, j])]
        rows.append(f)
    return np.asarray(rows, dtype=np.float32).reshape(len(ends), -1), _names(H)
