"""Trainer: tiny-batch overfit sanity (spec phase gate), mirror augmentation, calibration."""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from stlens.models.baselines.deep import DeepLOBLite, SpatialCNN, TemporalTCN
from stlens.models.stlens_net.model import STLENSNet
from stlens.training.trainer import (
    TrainConfig,
    calibrate,
    fit_temperature,
    mirror_batch,
    predict,
    train_model,
)

C, T, P, F, L2 = 6, 40, 20, 9, 20


class Toy(Dataset):
    """Positives carry a large cancel at one far bid bucket near the end of the window."""

    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.ladder = torch.randn(n, C, T, P, generator=g) * 0.3
        self.context = torch.randn(n, T, F, generator=g) * 0.3
        self.level = torch.randn(n, 2, T, L2, generator=g) * 0.3
        self.y = (torch.rand(n, generator=g) < 0.3).float()
        for i in torch.nonzero(self.y).flatten():
            self.ladder[i, 3, -4:, 6] += 4.0  # cancel_vol spike
            self.level[i, 0, -4:, 6] += 4.0
            self.context[i, -4:, 4] += 2.0
        self.side = (self.y * 1).long()
        self.loc = torch.where(self.y > 0, 6, -1).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "ladder": self.ladder[i],
            "context": self.context[i],
            "level": self.level[i],
            "y": self.y[i],
            "side": self.side[i],
            "loc": self.loc[i],
        }


@pytest.mark.parametrize(
    "make",
    [
        lambda: SpatialCNN(C),
        lambda: TemporalTCN(C, F),
        lambda: DeepLOBLite(L2),
        lambda: STLENSNet(C, P, F),
    ],
)
def test_models_learn_a_planted_pattern(make):
    tr, va = Toy(256, 0), Toy(128, 1)
    cfg = TrainConfig(epochs=8, batch_size=64, lr=3e-3, patience=8)
    _, hist = train_model(make(), tr, va, tr.y.numpy(), va.y.numpy(), cfg)
    assert hist[-1]["val_pr_auc"] > 0.9 or max(h["val_pr_auc"] for h in hist) > 0.9


def test_mirror_is_an_involution_and_flips_labels():
    b = dict(Toy(8, 2).__dict__)
    m = mirror_batch(b)
    assert torch.equal(m["ladder"], b["ladder"].flip(-1))
    assert ((b["side"] == 1) == (m["side"] == 2)).all()
    assert torch.equal(m["loc"][b["loc"] >= 0], P - 1 - b["loc"][b["loc"] >= 0])
    back = mirror_batch(m)
    for k in ("ladder", "context", "side", "loc"):
        assert torch.equal(back[k], b[k])


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20_000)
    y = (rng.random(20_000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    t = fit_temperature(true_logit * 3.0, y)  # over-confident by a factor 3
    assert t == pytest.approx(3.0, rel=0.1)
    p = calibrate(true_logit * 3.0, t)
    assert ((p >= 0) & (p <= 1)).all()


def test_predict_is_deterministic():
    ds = Toy(32, 3)
    m = STLENSNet(C, P, F)
    np.testing.assert_array_equal(predict(m, ds), predict(m, ds))


def test_temperature_bounded_and_calibration_preserves_ranking():
    # Perfectly separable logits push the NLL optimum towards T -> 0; it must be bounded.
    logits = np.r_[np.full(50, -3.0), np.full(50, 3.0)] + np.linspace(0, 0.01, 100)
    y = np.r_[np.zeros(50), np.ones(50)]
    t = fit_temperature(logits, y)
    assert 0.05 <= t <= 20.0
    p = calibrate(logits * 100, t)
    assert np.isfinite(p).all()
    assert (np.argsort(p, kind="stable") == np.argsort(logits, kind="stable")).all()
