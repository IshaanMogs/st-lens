"""ST-LENS-Net (spec H): a CNN-then-temporal model over the price ladder.

This is an adaptation of the established CNN + temporal pattern (e.g. DeepLOB, Zhang,
Zohren & Roberts 2019). It is NOT a novel architecture; its value is in the ablations.

Stages: side-symmetric multi-scale price convolutions per time step (bid half mirrored,
weights shared with the ask half) -> cross-touch convolution -> per-bucket map + pooled
vector -> fusion with the context series -> causal dilated TCN -> temporal attention
pooling -> heads (spoof-likeness logit, side, location).

Ablation switches:
* ``use_spatial=False``: the price-axis encoder is replaced by per-step side aggregates
  (no price structure); the location head is unavailable.
* ``use_temporal=False``: no TCN and mean pooling over time instead of attention, so the
  model is invariant to the order of time steps (no temporal modelling).
* ``shared_sides=False``: separate bid/ask convolution weights.
"""

import torch
from torch import nn

from stlens.models.common import TCN, side_aggregates


class _MultiScale(nn.Module):
    def __init__(self, c_in: int, dim: int, kernels=(1, 3, 5)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(c_in, dim, k, padding=k // 2) for k in kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(torch.cat([c(x) for c in self.convs], dim=1))


class STLENSNet(nn.Module):
    def __init__(
        self,
        channels: int,
        buckets: int,
        context: int,
        dim: int = 32,
        use_spatial: bool = True,
        use_temporal: bool = True,
        shared_sides: bool = True,
    ) -> None:
        super().__init__()
        self.P, self.H = buckets, buckets // 2
        self.use_spatial, self.use_temporal = use_spatial, use_temporal
        branch = dim // 2
        if use_spatial:
            self.bid_enc = _MultiScale(channels, branch)
            self.ask_enc = self.bid_enc if shared_sides else _MultiScale(channels, branch)
            self.cross = nn.Conv1d(3 * branch, dim, 3, padding=1)
        else:
            self.agg = nn.Linear(4 * channels, dim)
        self.ctx = nn.Linear(context, dim)
        self.fuse = nn.Linear(2 * dim, dim)
        self.tcn = TCN(dim) if use_temporal else None
        self.attn = nn.Linear(dim, 1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 1))
        self.side_head = nn.Linear(dim, 3)
        self.loc_head = nn.Conv1d(dim, 1, 1) if use_spatial else None

    def spatial(self, ladder: torch.Tensor) -> torch.Tensor:
        """``[B, C, T, P] -> [B, T, D, P]`` per-bucket feature map."""
        B, C, T, P = ladder.shape
        x = ladder.permute(0, 2, 1, 3).reshape(B * T, C, P)
        bid = self.bid_enc(x[..., : self.H].flip(-1)).flip(-1)  # mirrored: touch first
        ask = self.ask_enc(x[..., self.H :])
        h = nn.functional.gelu(self.cross(torch.cat([bid, ask], dim=-1)))
        return h.reshape(B, T, -1, P)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ladder, context = batch["ladder"], batch["context"]
        fmap = None
        if self.use_spatial:
            fmap = self.spatial(ladder)  # [B, T, D, P]
            pooled = fmap.amax(-1)
        else:
            pooled = self.agg(side_aggregates(ladder))
        h = self.fuse(torch.cat([pooled, self.ctx(context)], dim=-1))  # [B, T, D]
        if self.tcn is not None:
            h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
            w = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # [B, T]
        else:
            w = torch.full(h.shape[:2], 1.0 / h.shape[1], device=h.device)
        emb = (w.unsqueeze(-1) * h).sum(1)
        out = {"logit": self.head(emb).squeeze(-1), "side": self.side_head(emb), "attn": w}
        if fmap is not None and self.loc_head is not None:
            loc_map = (w[:, :, None, None] * fmap).sum(1)  # [B, D, P]
            out["loc"] = self.loc_head(loc_map).squeeze(1)  # [B, P]
        return out
