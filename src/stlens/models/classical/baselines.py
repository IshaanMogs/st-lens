"""Classical baselines B0-B3 (spec G). All use the same window index per split.

* B0 prior: random scores; PR-AUC ~= prevalence (what chance looks like).
* B1 rule engine: the heuristic labeller's own score (``labeling.rules``).
* B2 logistic regression on window features ``v`` (scaler fitted on train).
* B3 XGBoost on ``v``. Fixed hyperparameters; early stopping on validation only.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def prior_scores(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int = 0):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3_000, C=0.5, random_state=seed),
    )
    return model.fit(X, y)


def fit_xgboost(
    X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int = 0
) -> XGBClassifier:
    pos = max(int(y.sum()), 1)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2.0,
        scale_pos_weight=(len(y) - pos) / pos,
        eval_metric="aucpr",
        early_stopping_rounds=40,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )
    return model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
