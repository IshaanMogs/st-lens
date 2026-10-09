"""Basic explainability (spec K): evidence reason codes plus model attribution.

* Reason codes are model-independent facts about the window, from the reconstructed
  book (what an analyst can check).
* Attribution: Integrated Gradients over the ladder input and price-bucket occlusion.
  Temporal attention weights are NOT treated as explanations.
* Validation: localisation hit rate on injected spoofs (true bucket known) and a
  randomised-weights sanity check (attributions should change when weights do).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stlens.datasets.build import DayData


def integrated_gradients(
    model: nn.Module, batch: dict[str, torch.Tensor], steps: int = 24
) -> torch.Tensor:
    """IG of the main logit w.r.t. the (standardised) ladder; baseline = 0 (train mean).
    Returns ``[B, C, T, P]``."""
    model.eval()
    x = batch["ladder"]
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (a * x).requires_grad_(True)
        logit = model({**batch, "ladder": xi})["logit"].sum()
        (grad,) = torch.autograd.grad(logit, xi)
        total += grad
    return x * total / steps


@torch.no_grad()
def occlusion_buckets(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Drop in logit when one price bucket (all channels, all steps) is set to the train
    mean (0 after standardisation). ``[B, P]``; larger = more important."""
    model.eval()
    base = model(batch)["logit"]
    P = batch["ladder"].shape[-1]
    drops = []
    for p in range(P):
        x = batch["ladder"].clone()
        x[..., p] = 0.0
        drops.append(base - model({**batch, "ladder": x})["logit"])
    return torch.stack(drops, dim=1)


def bucket_importance(ig: torch.Tensor) -> np.ndarray:
    """Positive evidence per bucket: sum of IG over channels and time. ``[B, P]``."""
    return ig.sum(dim=(1, 2)).detach().cpu().numpy()


def localisation_hit_rate(importance: np.ndarray, true_loc: np.ndarray, tol: int = 1) -> float:
    """Share of windows whose most important bucket is within ``tol`` of the true one."""
    keep = true_loc >= 0
    if not keep.any():
        return float("nan")
    peak = importance[keep].argmax(axis=1)
    return float(np.mean(np.abs(peak - true_loc[keep]) <= tol))


def randomised_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    import copy

    clone = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in clone.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return clone


@dataclass
class ReasonCode:
    side: str
    bucket_ticks: int  # ticks beyond the mid reference on that side
    rel_size_max: float
    cancelled_rel: float
    executed_rel: float
    opposite_volume: float

    def text(self) -> str:
        return (
            f"{self.side} +{self.bucket_ticks} ticks: "
            f"size {self.rel_size_max:.1f}x typical level, "
            f"cancelled {self.cancelled_rel:.1f}x / "
            f"executed {self.executed_rel:.1f}x typical size, "
            f"opposite-side aggressive volume {self.opposite_volume:.3f}"
        )

    @property
    def supports_high(self) -> bool:
        """Evidence check for a High band: a large level that was cancelled, not executed."""
        return (
            self.rel_size_max >= 5
            and self.cancelled_rel >= 3
            and self.executed_rel < self.cancelled_rel
        )


def reason_codes(day: DayData, t: int, window: int) -> list[ReasonCode]:
    """Evidence per side for the window ending at ``t`` (rows ``t-window+1 .. t`` only)."""
    g = day.grid
    H = g.half_buckets
    sl = slice(t - window + 1, t + 1)
    rel = day.ladder[sl, 5, :]
    scale = day.scale[sl, None]
    codes = []
    for side, half, opp in (("bid", range(0, H), g.buy_vol), ("ask", range(H, 2 * H), g.sell_vol)):
        cols = list(half)
        b = cols[int(np.argmax(rel[:, cols].max(axis=0)))]
        near = [x for x in (b - 1, b, b + 1) if x in cols]
        codes.append(
            ReasonCode(
                side=side,
                bucket_ticks=(H - 1 - b) if side == "bid" else (b - H),
                rel_size_max=float(rel[:, b].max()),
                cancelled_rel=float((g.cancel_vol[sl][:, near] / scale).sum(axis=1).max()),
                executed_rel=float((g.exec_vol[sl][:, near] / scale).sum(axis=1).max()),
                opposite_volume=float(opp[sl].sum()),
            )
        )
    return sorted(codes, key=lambda c: -c.rel_size_max)
