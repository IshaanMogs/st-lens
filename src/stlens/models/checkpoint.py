"""Reproducible ST-LENS-Net checkpoints: weights + everything needed to score new windows.

A checkpoint stores:

* the architecture config (``STLENSNet.config``) and the trained ``state_dict``;
* the train-only standardisation statistics for the ladder, context and level inputs;
* the validation-fitted temperature and the validation-chosen operating threshold;
* metadata (seed, validation PR-AUC, data/feature settings, versions).

Loading uses ``torch.load(weights_only=True)``: only tensors and plain Python values are
stored, so no arbitrary code is unpickled.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stlens.datasets.build import Standardizer
from stlens.models.stlens_net.model import STLENSNet
from stlens.training.trainer import calibrate

FORMAT_VERSION = 1
INPUTS = ("ladder", "context", "level")


@dataclass
class LoadedCheckpoint:
    model: STLENSNet
    standardizers: dict[str, Standardizer]
    temperature: float
    threshold: float
    meta: dict

    def probabilities(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """Calibrated spoof-likeness for an already-standardised batch."""
        with torch.no_grad():
            logits = self.model.eval()(batch)["logit"].cpu().numpy()
        return calibrate(logits, self.temperature)


def save_checkpoint(
    path: Path,
    model: STLENSNet,
    standardizers: dict[str, Standardizer],
    temperature: float,
    threshold: float,
    meta: dict,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "model_class": "STLENSNet",
        "model_config": dict(model.config),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "standardizers": {
            name: {
                "mean": torch.from_numpy(np.asarray(std.mean, dtype=np.float64)),
                "std": torch.from_numpy(np.asarray(std.std, dtype=np.float64)),
                "fitted_on_split": std.fitted_on_split,
                "fit_data_end_us": int(std.fit_data_end_us),
            }
            for name, std in standardizers.items()
        },
        "temperature": float(temperature),
        "threshold": float(threshold),
        "meta": meta,
        "torch_version": str(torch.__version__),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> LoadedCheckpoint:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("model_class") != "STLENSNet":
        raise ValueError(f"unsupported checkpoint format in {path}")
    model = STLENSNet(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    standardizers = {
        name: Standardizer(
            mean=s["mean"].numpy(),
            std=s["std"].numpy(),
            fitted_on_split=s["fitted_on_split"],
            fit_data_end_us=s["fit_data_end_us"],
        )
        for name, s in payload["standardizers"].items()
    }
    missing = set(INPUTS) - set(standardizers)
    if missing:
        raise ValueError(f"checkpoint lacks standardizers for {sorted(missing)}")
    return LoadedCheckpoint(
        model=model,
        standardizers=standardizers,
        temperature=payload["temperature"],
        threshold=payload["threshold"],
        meta=payload["meta"],
    )
