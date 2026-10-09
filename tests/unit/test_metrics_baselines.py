"""Metrics on hand-made inputs and classical baselines on controlled data (Phase 6)."""

import numpy as np
import pytest

from stlens.evaluation.metrics import EvalSet, evaluate, threshold_for_budget
from stlens.models.classical.baselines import fit_logistic, fit_xgboost, prior_scores


def eval_set(y, kind=None, episode=None):
    n = len(y)
    return EvalSet(
        y=np.asarray(y),
        step=np.arange(n),
        day=np.zeros(n, int),
        kind=np.asarray(kind if kind is not None else [""] * n),
        episode=np.asarray(episode if episode is not None else [""] * n),
        grid_s=0.25,
        hours=1.0,
    )


def test_perfect_and_confusion_counts():
    y = [0, 0, 1, 1, 0]
    r = evaluate(np.array([0.1, 0.2, 0.9, 0.8, 0.3]), eval_set(y), threshold=0.5)
    assert r.pr_auc == pytest.approx(1.0) and r.roc_auc == pytest.approx(1.0)
    assert (r.tp, r.fp, r.fn, r.tn) == (2, 0, 0, 3)
    assert r.alerts_per_hour == 2.0


def test_episode_recall_latency_and_hard_negatives():
    y = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    kind = ["", "spoof", "spoof", "spoof", "hn_executed", "hn_executed", "", "spoof", "spoof", ""]
    ep = ["", "e1", "e1", "e1", "h1", "h1", "", "e2", "e2", ""]
    scores = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], float)
    r = evaluate(scores, eval_set(y, kind, ep), threshold=0.5)
    assert r.episode_recall == 0.5  # e1 caught, e2 missed
    assert r.median_latency_s == pytest.approx(0.25)  # first flag one step after start
    assert r.hard_negative_flag_rate == 1.0
    assert r.n_pos_episodes == 2


def test_threshold_respects_alert_budget():
    scores = np.linspace(0, 1, 101)
    thr = threshold_for_budget(scores, hours=2.0, alerts_per_hour=5)
    assert (scores >= thr).sum() == 10


def test_prior_pr_auc_close_to_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.05).astype(int)
    r = evaluate(prior_scores(len(y), seed=1), eval_set(y), threshold=0.5)
    assert r.pr_auc == pytest.approx(y.mean(), abs=0.01)


def test_logistic_and_xgboost_learn_a_planted_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3_000, 6))
    y = (X[:, 2] + 0.3 * rng.normal(size=3_000) > 1.6).astype(int)
    Xv = rng.normal(size=(1_000, 6))
    yv = (Xv[:, 2] + 0.3 * rng.normal(size=1_000) > 1.6).astype(int)
    lr = fit_logistic(X, y)
    xgb = fit_xgboost(X, y, Xv, yv)
    for p in (lr.predict_proba(Xv)[:, 1], xgb.predict_proba(Xv)[:, 1]):
        assert evaluate(p, eval_set(yv), 0.5).pr_auc > 0.8
