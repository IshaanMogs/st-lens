"""Deep baselines and ST-LENS-Net: shapes, ablation switches, temporal causality."""

import pytest
import torch

from stlens.models.baselines.deep import DeepLOBLite, SpatialCNN, TemporalTCN
from stlens.models.common import TCN
from stlens.models.stlens_net.model import STLENSNet

B, C, T, P, F, L2 = 3, 6, 40, 20, 9, 20


def batch():
    g = torch.Generator().manual_seed(0)
    return {
        "ladder": torch.randn(B, C, T, P, generator=g),
        "context": torch.randn(B, T, F, generator=g),
        "level": torch.randn(B, 2, T, L2, generator=g),
    }


@pytest.mark.parametrize(
    "model",
    [
        SpatialCNN(C),
        TemporalTCN(C, F),
        DeepLOBLite(L2),
        STLENSNet(C, P, F),
        STLENSNet(C, P, F, use_spatial=False),
        STLENSNet(C, P, F, use_temporal=False),
        STLENSNet(C, P, F, shared_sides=False),
    ],
)
def test_forward_shapes(model):
    out = model.eval()(batch())
    assert out["logit"].shape == (B,)
    if "loc" in out:
        assert out["loc"].shape == (B, P)


def test_stlens_heads_and_ablation_outputs():
    full = STLENSNet(C, P, F).eval()(batch())
    assert full["side"].shape == (B, 3) and full["attn"].shape == (B, T)
    assert "loc" not in STLENSNet(C, P, F, use_spatial=False).eval()(batch())


def test_tcn_is_causal_and_covers_window():
    tcn = TCN(8).eval()
    assert tcn.receptive_field >= T
    x = torch.randn(1, 8, T)
    y = tcn(x)
    x2 = x.clone()
    x2[..., 25:] += 10.0  # change only the future of step 24
    torch.testing.assert_close(tcn(x2)[..., :25], y[..., :25])


def test_temporal_ablation_is_order_invariant():
    m = STLENSNet(C, P, F, use_temporal=False).eval()
    b = batch()
    perm = torch.randperm(T)
    shuffled = {**b, "ladder": b["ladder"][:, :, perm], "context": b["context"][:, perm]}
    torch.testing.assert_close(m(b)["logit"], m(shuffled)["logit"], rtol=1e-4, atol=1e-5)
