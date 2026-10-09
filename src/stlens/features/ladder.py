"""Causal per-step features: the ladder tensor and the context series (Phase 3).

Every value at grid step ``t`` is computed from grid rows ``<= t`` only (rolling windows
look backwards and include the current step). Normalisation statistics are NOT applied
here; they are fitted on the training split only in Phase 5.

Ladder tensor ``[T, C, P]`` (spec C), channels:

* ``depth``      log1p(resting size)
* ``d_depth``    signed log of the change since the previous step (0 at t=0)
* ``add_vol``    log1p(inferred added volume in the interval)
* ``cancel_vol`` log1p(inferred cancelled volume)
* ``exec_vol``   log1p(executed volume)
* ``rel_size``   size / rolling median of non-zero level sizes over the past window,
                 clipped to [0, 20]

Deviation from spec C, stated: the spec says "signed log" for add/cancel/exec, but those
flows are non-negative, so ``log1p`` is used (identical on that domain). ``rel_size`` uses
a pooled (all-bucket) rolling median rather than a per-bucket one, because per-bucket
medians are zero for sparsely populated far buckets.
"""

import numpy as np

from stlens.book.builder import GridDay

CHANNELS = ("depth", "d_depth", "add_vol", "cancel_vol", "exec_vol", "rel_size")
CONTEXT = (
    "mid_ret",  # group 6: mid change in ticks since previous step
    "spread",  # group 1
    "realised_vol",  # group 6: std of mid_ret over the past window
    "trade_imbalance",  # group 5: (buy - sell) / (buy + sell)
    "log_trade_vol",  # group 5
    "ofi",  # group 2: signed log of order flow imbalance (Cont et al. 2014)
    "queue_imb_1",  # group 1: (q_bid1 - q_ask1) / (q_bid1 + q_ask1)
    "queue_imb_5",  # group 1: same over the 5 best levels
    "vol_regime",  # group 7: rolling mean of log trade volume
)
# Context features whose sign flips when bids and asks are swapped (mirror augmentation).
SIGNED_CONTEXT = ("mid_ret", "trade_imbalance", "ofi", "queue_imb_1", "queue_imb_5")

REL_WINDOW = 240  # 60 s at 250 ms
VOL_WINDOW = 40  # 10 s
REGIME_WINDOW = 240


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _rolling_past(values: np.ndarray, window: int, fn) -> np.ndarray:
    """fn over values[max(0, t-window+1) : t+1] for each t (causal, includes t)."""
    out = np.empty(len(values))
    for t in range(len(values)):
        out[t] = fn(values[max(0, t - window + 1) : t + 1])
    return out


def size_scale(g: GridDay, window: int = REL_WINDOW) -> np.ndarray:
    """Typical level size per step: causal rolling median of per-step median level sizes."""
    step_med = np.array([np.median(row[row > 0]) if (row > 0).any() else np.nan for row in g.depth])

    def med(w: np.ndarray) -> float:
        return float(np.nanmedian(w)) if np.isfinite(w).any() else 1.0

    denom = _rolling_past(step_med, window, med)
    return np.where(np.isfinite(denom) & (denom > 0), denom, 1.0)


def ladder_tensor(g: GridDay, rel_window: int = REL_WINDOW) -> np.ndarray:
    """Causal ladder tensor ``[T, C, P]`` (float32, un-normalised)."""
    depth = g.depth
    d_depth = np.zeros_like(depth)
    d_depth[1:] = depth[1:] - depth[:-1]
    rel = np.clip(depth / size_scale(g, rel_window)[:, None], 0.0, 20.0)
    stacked = np.stack(
        [
            np.log1p(depth),
            signed_log(d_depth),
            np.log1p(g.add_vol),
            np.log1p(g.cancel_vol),
            np.log1p(g.exec_vol),
            rel,
        ],
        axis=1,
    )
    return stacked.astype(np.float32)


def level_tensor(g: GridDay) -> np.ndarray:
    """Level-indexed input for the DeepLOB-style baseline: ``[T, 2, 2L]`` (log size, distance)."""
    return np.stack([np.log1p(g.lvl_size), g.lvl_dist], axis=1).astype(np.float32)


def context_series(g: GridDay) -> np.ndarray:
    """Causal context features ``[T, F]`` in the order of :data:`CONTEXT`."""
    mid = g.mid
    mid_ret = np.zeros(len(mid))
    mid_ret[1:] = np.nan_to_num(mid[1:] - mid[:-1])
    spread = np.nan_to_num(g.best_ask - g.best_bid)
    vol = _rolling_past(mid_ret, VOL_WINDOW, np.std)
    tv = g.buy_vol + g.sell_vol
    imb = np.where(tv > 0, (g.buy_vol - g.sell_vol) / np.where(tv > 0, tv, 1.0), 0.0)
    L = g.half_levels
    qb, qa = g.lvl_size[:, L - 1], g.lvl_size[:, L]
    q1 = np.where(qb + qa > 0, (qb - qa) / np.where(qb + qa > 0, qb + qa, 1.0), 0.0)
    qb5, qa5 = g.lvl_size[:, L - 5 : L].sum(1), g.lvl_size[:, L : L + 5].sum(1)
    q5 = np.where(qb5 + qa5 > 0, (qb5 - qa5) / np.where(qb5 + qa5 > 0, qb5 + qa5, 1.0), 0.0)
    log_tv = np.log1p(tv)
    regime = _rolling_past(log_tv, REGIME_WINDOW, np.mean)
    cols = [mid_ret, spread, vol, imb, log_tv, signed_log(g.ofi), q1, q5, regime]
    return np.stack(cols, axis=1).astype(np.float32)
