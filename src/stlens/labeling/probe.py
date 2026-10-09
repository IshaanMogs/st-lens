"""Injection artefact probe (spec J): can placement features alone tell injected large
orders from the background's own large orders? It should be near chance (AUC ~ 0.5).

Only placement attributes are used (size, distance from the own-side touch); cancel or
execution behaviour is deliberately excluded, since that is the signal being studied.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from stlens.ingestion.synthetic import SimulatedDay


def probe_auc(days: list[SimulatedDay], min_size: float, seed: int = 0) -> tuple[float, int, int]:
    """Mean 5-fold CV ROC-AUC, plus (n_injected, n_background) orders compared.

    Single-layer, away-from-touch injections are compared with background orders of the
    same size class (``size >= min_size``, the injector's "large" threshold) placed at
    distance >= 1 (injected spoofs never sit at the touch).
    """
    inj, bg = [], []
    for d in days:
        for rec in d.episodes:
            if rec.spec.layers == 1 and rec.spec.distance >= 1:
                inj.append((np.log(rec.spec.size), rec.spec.distance))
        bg += [
            (np.log(s), dist)
            for (_, _, dist, s) in d.background_large_orders
            if dist >= 1 and s >= min_size
        ]
    if len(inj) < 10 or len(bg) < 10:
        raise ValueError("not enough orders for the probe")
    rng = np.random.default_rng(seed)
    bg_arr = np.asarray(bg)[rng.choice(len(bg), size=min(len(bg), 5 * len(inj)), replace=False)]
    X = np.vstack([np.asarray(inj), bg_arr])
    y = np.r_[np.ones(len(inj)), np.zeros(len(bg_arr))]
    auc = cross_val_score(LogisticRegression(), X, y, cv=5, scoring="roc_auc").mean()
    return float(auc), len(inj), len(bg_arr)
