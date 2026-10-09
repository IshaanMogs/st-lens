"""Shared model building blocks. Every model maps a batch dict to a dict with ``logit``
(and optionally ``side``, ``loc``, ``attn``). Inputs: ``ladder [B, C, T, P]``,
``context [B, T, F]``, ``level [B, 2, T, 2L]``."""

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    """1-D convolution over time that only sees the current and past steps."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int = 1) -> None:
        super().__init__(c_in, c_out, kernel, dilation=dilation)
        self.left_pad = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, T]
        return super().forward(nn.functional.pad(x, (self.left_pad, 0)))


class TCN(nn.Module):
    """Causal dilated residual TCN. Receptive field = 1 + (kernel-1) * sum(dilations)."""

    def __init__(self, dim: int, dilations=(1, 2, 4, 8, 16), kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(CausalConv1d(dim, dim, kernel, d), nn.GELU(), nn.Dropout(dropout))
            for d in dilations
        )
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D, T] -> [B, D, T]
        for block in self.blocks:
            x = x + block(x)
        return x


def side_aggregates(ladder: torch.Tensor) -> torch.Tensor:
    """Per-step features WITHOUT price-axis structure: mean and max of every channel over
    each half of the ladder. ``[B, C, T, P] -> [B, T, 4C]``."""
    P = ladder.shape[-1]
    bid, ask = ladder[..., : P // 2], ladder[..., P // 2 :]
    feats = [bid.mean(-1), bid.amax(-1), ask.mean(-1), ask.amax(-1)]
    return torch.cat(feats, dim=1).transpose(1, 2)
