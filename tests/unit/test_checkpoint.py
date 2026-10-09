"""ST-LENS checkpoint: reload reproduces predictions on a fixed input exactly."""

import numpy as np
import pytest
import torch

from stlens.datasets.build import Standardizer
from stlens.models.checkpoint import load_checkpoint, save_checkpoint
from stlens.models.stlens_net.model import STLENSNet

C, T, P, F, L2 = 6, 40, 20, 9, 20


def fixed_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    return {
        "ladder": torch.randn(4, C, T, P, generator=g),
        "context": torch.randn(4, T, F, generator=g),
        "level": torch.randn(4, 2, T, L2, generator=g),
    }


def standardizers() -> dict[str, Standardizer]:
    rng = np.random.default_rng(0)
    return {
        "ladder": Standardizer(rng.normal(size=(C, 1)), rng.random((C, 1)) + 0.5, "train", 10),
        "context": Standardizer(rng.normal(size=F), rng.random(F) + 0.5, "train", 10),
        "level": Standardizer(rng.normal(size=(2, 1)), rng.random((2, 1)) + 0.5, "train", 10),
    }


@pytest.mark.parametrize(
    "kwargs", [{}, {"use_temporal": False}, {"use_spatial": False}, {"shared_sides": False}]
)
def test_reload_reproduces_predictions(tmp_path, kwargs):
    torch.manual_seed(7)
    model = STLENSNet(C, P, F, **kwargs).eval()
    batch = fixed_batch()
    with torch.no_grad():
        expected = model(batch)["logit"].numpy()
    stds = standardizers()
    path = tmp_path / "stlens_full.pt"
    save_checkpoint(path, model, stds, temperature=1.7, threshold=0.42, meta={"seed": 7})

    ck = load_checkpoint(path)
    with torch.no_grad():
        got = ck.model(batch)["logit"].numpy()
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_allclose(
        ck.probabilities(batch), 1 / (1 + np.exp(-expected / 1.7)), rtol=1e-6
    )
    assert (ck.temperature, ck.threshold, ck.meta) == (1.7, 0.42, {"seed": 7})
    assert ck.model.config == model.config
    for name, std in stds.items():
        np.testing.assert_array_equal(ck.standardizers[name].mean, std.mean)
        np.testing.assert_array_equal(ck.standardizers[name].std, std.std)
        assert ck.standardizers[name].fitted_on_split == "train"


def test_reload_into_fresh_process_state_is_independent(tmp_path):
    torch.manual_seed(1)
    model = STLENSNet(C, P, F).eval()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, standardizers(), 1.0, 0.5, {})
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # mutate the original after saving
    ck = load_checkpoint(path)
    assert not torch.equal(next(ck.model.parameters()), next(model.parameters()))


def test_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 99, "model_class": "STLENSNet"}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        load_checkpoint(path)
