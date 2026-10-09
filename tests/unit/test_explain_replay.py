"""Explainability (Phase 9) and offline replay scoring (Phase 10)."""

import numpy as np
import pytest
import torch
from torch import nn

from stlens.explain.attribution import (
    ReasonCode,
    bucket_importance,
    integrated_gradients,
    localisation_hit_rate,
    occlusion_buckets,
    randomised_copy,
)
from stlens.models.stlens_net.model import STLENSNet
from stlens.scoring.replay import AlertConfig, StreamingScorer, ema_batch, replay_day


class BucketModel(nn.Module):
    """Logit = sum of channel 3 at bucket 6: attribution must point at bucket 6."""

    def forward(self, batch):
        return {"logit": batch["ladder"][:, 3, :, 6].sum(-1)}


def batch(B=2, C=6, T=10, P=20):
    g = torch.Generator().manual_seed(0)
    return {"ladder": torch.rand(B, C, T, P, generator=g), "context": torch.zeros(B, T, 9)}


def test_ig_and_occlusion_localise_known_bucket():
    b = batch()
    imp = bucket_importance(integrated_gradients(BucketModel(), b))
    assert (imp.argmax(1) == 6).all()
    assert (occlusion_buckets(BucketModel(), b).argmax(1) == 6).all()
    assert localisation_hit_rate(imp, np.array([6, 7]), tol=1) == 1.0
    assert localisation_hit_rate(imp, np.array([10, -1]), tol=1) == 0.0


def test_ig_completeness_for_linear_model():
    b = batch()
    ig = integrated_gradients(BucketModel(), b)
    zero = BucketModel()({**b, "ladder": torch.zeros_like(b["ladder"])})["logit"]
    full = BucketModel()(b)["logit"]
    torch.testing.assert_close(ig.sum(dim=(1, 2, 3)), full - zero)


def test_randomised_model_changes_attributions():
    m = STLENSNet(6, 20, 9)
    b = batch()
    a = bucket_importance(integrated_gradients(m, b, steps=4))
    r = bucket_importance(integrated_gradients(randomised_copy(m), b, steps=4))
    assert not np.allclose(a, r)


def test_reason_code_evidence_rule():
    strong = ReasonCode(
        "bid", 3, rel_size_max=9.0, cancelled_rel=8.0, executed_rel=0.0, opposite_volume=1.0
    )
    executed = ReasonCode(
        "bid", 0, rel_size_max=9.0, cancelled_rel=0.5, executed_rel=8.0, opposite_volume=0.0
    )
    assert strong.supports_high and not executed.supports_high
    assert "bid +3 ticks" in strong.text()


def test_streaming_scorer_matches_batch_ema_and_never_looks_ahead():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    p[[5, 50]] = np.nan
    s = StreamingScorer(AlertConfig(ema_alpha=0.3))
    stream = np.array([s.update(i, x, True)[0] for i, x in enumerate(p)])
    np.testing.assert_allclose(stream, ema_batch(p, 0.3), equal_nan=True)
    # Changing the future does not change past scores.
    p2 = p.copy()
    p2[100:] = 1.0
    s2 = StreamingScorer(AlertConfig(ema_alpha=0.3))
    stream2 = np.array([s2.update(i, x, True)[0] for i, x in enumerate(p2)])
    np.testing.assert_array_equal(stream[:100], stream2[:100])


def test_hysteresis_cooldown_and_evidence():
    cfg = AlertConfig(ema_alpha=1.0, high=0.8, low=0.3, cooldown_steps=5)
    prob = np.array([0.1, 0.9, 0.7, 0.5, 0.2, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.2])
    times = np.arange(len(prob)) * 250_000
    _, _, alerts = replay_day(0, times, prob, np.ones(len(prob), bool), cfg)
    # opens at 1, stays open through 0.7/0.5 (above low), closes at 4; re-trigger at 5-6
    # is inside the cooldown; next alert opens at 12.
    assert [(a.start_step, a.end_step) for a in alerts] == [(1, 4), (12, 13)]
    assert alerts[0].peak_score == pytest.approx(0.9)
    _, band, none = replay_day(0, times, prob, np.zeros(len(prob), bool), cfg)
    assert none == [] and "high" not in set(band)  # High requires supporting evidence
