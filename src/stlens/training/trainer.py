"""Training loop shared by all deep models (spec H training choices).

* weighted BCE on the main head (positive weight = neg/pos in the training windows, capped)
* down-weighted auxiliary losses: side (all windows) and location (positives with a
  known location only)
* AdamW, early stopping on VALIDATION PR-AUC, best weights restored
* bid/ask mirror augmentation with label flip
* temperature scaling fitted on validation logits

Device-agnostic: runs on CPU, uses CUDA if available and requested.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stlens.features.ladder import CONTEXT, SIGNED_CONTEXT

SIGNED_IDX = [CONTEXT.index(c) for c in SIGNED_CONTEXT]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 3
    aux_weight: float = 0.2
    max_pos_weight: float = 30.0
    mirror: bool = True
    seed: int = 0
    device: str = "cpu"


def mirror_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Swap bids and asks: reverse the price axis, negate signed context, flip labels."""
    P = batch["ladder"].shape[-1]
    ctx = batch["context"].clone()
    ctx[..., SIGNED_IDX] = -ctx[..., SIGNED_IDX]
    side = batch["side"].clone()
    side[batch["side"] == 1], side[batch["side"] == 2] = 2, 1
    loc = torch.where(batch["loc"] >= 0, P - 1 - batch["loc"], batch["loc"])
    return {
        **batch,
        "ladder": batch["ladder"].flip(-1),
        "level": batch["level"].flip(-1),
        "context": ctx,
        "side": side,
        "loc": loc,
    }


def _to(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(
    model: nn.Module, ds: Dataset, batch_size: int = 512, device: str = "cpu"
) -> np.ndarray:
    model.eval().to(device)
    out = [
        model(_to(b, device))["logit"].cpu().numpy()
        for b in DataLoader(ds, batch_size=batch_size, shuffle=False)
    ]
    return np.concatenate(out) if out else np.zeros(0)


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(cfg.seed)
    model.to(cfg.device)
    pos = max(float(y_train.sum()), 1.0)
    pos_weight = torch.tensor(
        min((len(y_train) - pos) / pos, cfg.max_pos_weight), device=cfg.device
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    best, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for b in loader:
            b = _to(b, cfg.device)
            if cfg.mirror:
                flip = torch.rand(len(b["y"]), generator=gen) < 0.5
                if flip.any():
                    m = mirror_batch({k: v[flip.to(v.device)] for k, v in b.items()})
                    b = {k: torch.cat([v[~flip.to(v.device)], m[k]]) for k, v in b.items()}
            out = model(b)
            loss = bce(out["logit"], b["y"])
            if "side" in out:
                loss = loss + cfg.aux_weight * ce(out["side"], b["side"])
            if "loc" in out:
                known = b["loc"] >= 0
                if known.any():
                    loss = loss + cfg.aux_weight * ce(out["loc"][known], b["loc"][known])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(b["y"]), n + len(b["y"])
        val_ap = float(average_precision_score(y_val, predict(model, val_ds, device=cfg.device)))
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_pr_auc": val_ap})
        log(f"  epoch {epoch}: loss {total / max(n, 1):.4f}  val PR-AUC {val_ap:.4f}")
        if val_ap > best:
            best, stale = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


T_MIN, T_MAX = 0.05, 20.0


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T in [T_MIN, T_MAX] minimising validation NLL of sigmoid(logit / T).

    Bounded so a degenerate fit cannot collapse scores to 0/1 (which would destroy the
    ranking); falls back to T = 1 if the optimiser produces a non-finite value."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    if not np.isfinite(t):
        return 1.0
    return float(np.clip(t, T_MIN, T_MAX))


def calibrate(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable sigmoid(logit / T); monotonic, so rankings are preserved."""
    from scipy.special import expit

    return expit(np.asarray(logits, dtype=np.float64) / temperature)
