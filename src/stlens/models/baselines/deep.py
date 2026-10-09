"""Compact deep baselines (spec G, B4-B6). Small on purpose: limited positives.

* B4 :class:`SpatialCNN` - price-axis convolutions over the last few steps only.
* B5 :class:`TemporalTCN` - causal TCN over per-step features with no price structure.
* B6 :class:`DeepLOBLite` - DeepLOB-style (conv over the level axis, Inception-like
  multi-kernel block, LSTM) on the level-indexed book. A simplified re-implementation
  for reference, not a faithful reproduction of Zhang, Zohren & Roberts (2019).
"""

import torch
from torch import nn

from stlens.models.common import TCN, side_aggregates


class SpatialCNN(nn.Module):
    def __init__(self, channels: int, stack: int = 4, dim: int = 32) -> None:
        super().__init__()
        self.stack = stack
        self.net = nn.Sequential(
            nn.Conv1d(channels * stack, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["ladder"][:, :, -self.stack :, :]  # [B, C, S, P]
        h = self.net(x.flatten(1, 2))  # [B, D, P]
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return {"logit": self.head(pooled).squeeze(-1)}


class TemporalTCN(nn.Module):
    def __init__(self, channels: int, context: int, dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(4 * channels + context, dim)
        self.tcn = TCN(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat([side_aggregates(batch["ladder"]), batch["context"]], dim=-1)
        h = self.tcn(self.proj(x).transpose(1, 2))  # [B, D, T]
        return {"logit": self.head(h[:, :, -1]).squeeze(-1)}


class DeepLOBLite(nn.Module):
    def __init__(self, levels2: int, dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        # Conv over the level axis (pairs of neighbouring levels, then the whole side).
        self.conv = nn.Sequential(
            nn.Conv2d(2, dim, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim, dim, (1, levels2 // 2)),
            nn.LeakyReLU(0.01),
        )
        # Inception-like temporal block (kernels 1, 3, 5) - causal via left padding.
        self.branches = nn.ModuleList(nn.Conv2d(dim, dim, (k, 1)) for k in (1, 3, 5))
        self.lstm = nn.LSTM(3 * dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.conv(batch["level"])  # [B, D, T, 1]
        outs = [
            br(nn.functional.pad(x, (0, 0, k - 1, 0)))
            for br, k in zip(self.branches, (1, 3, 5), strict=True)
        ]
        h = torch.cat(outs, dim=1).squeeze(-1).transpose(1, 2)  # [B, T, 3D]
        out, _ = self.lstm(h)
        return {"logit": self.head(out[:, -1]).squeeze(-1)}
