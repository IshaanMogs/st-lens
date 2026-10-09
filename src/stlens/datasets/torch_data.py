"""PyTorch views over day data: windows are sliced lazily, standardised with train-only
statistics. Mirror augmentation (bid/ask swap) is applied in the trainer, not here."""

import numpy as np
import torch
from torch.utils.data import Dataset

from stlens.datasets.build import DayData, Standardizer


class WindowDataset(Dataset):
    """Items: ladder ``[C, T, P]``, context ``[T, F]``, level ``[2, T, 2L]`` and labels."""

    def __init__(
        self,
        days: list[DayData],
        index: np.ndarray,
        window: int,
        ladder_std: Standardizer,
        context_std: Standardizer,
        level_std: Standardizer,
    ) -> None:
        self.days, self.index, self.window = days, index, window
        self.ladder = [ladder_std(d.ladder) for d in days]
        self.context = [context_std(d.context) for d in days]
        self.level = [level_std(d.level) for d in days]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        pos, t = self.index[i]
        sl = slice(t - self.window + 1, t + 1)
        lab = self.days[pos].labels
        return {
            "ladder": torch.from_numpy(self.ladder[pos][sl]).permute(1, 0, 2),
            "context": torch.from_numpy(self.context[pos][sl]),
            "level": torch.from_numpy(self.level[pos][sl]).permute(1, 0, 2),
            "y": torch.tensor(float(lab.y[t])),
            "side": torch.tensor(int(lab.side[t])),
            "loc": torch.tensor(int(lab.loc[t])),
        }
