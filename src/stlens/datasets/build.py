"""Day data, chronological splits, windows, train-only scalers (Phase 5).

Splits are by whole (synthetic) day, in time order: train days, then validation days,
then test days. Injection uses a different seed stream per split. Windows never cross a
day boundary, and a window ending at grid step ``t`` covers ``[t - T + 1, t]`` with the
label of step ``t`` (which depends only on episodes ending at or before ``t``).

Fast-track simplification (documented): one chronological split instead of the full
rolling walk-forward of spec I; ``walk_forward_folds`` exists and is tested, but the
experiment evaluates a single fold.
"""

from dataclasses import dataclass, field

import numpy as np

from stlens.book.builder import GridDay, reconstruct
from stlens.features.ladder import context_series, ladder_tensor, level_tensor, size_scale
from stlens.ingestion.synthetic import MarketParams, SimulatedDay, SyntheticMarket
from stlens.labeling.injector import DayLabels, InjectionParams, label_day, schedule
from stlens.labeling.rules import RuleParams, RulePattern, rule_scores

SPLITS = ("train", "val", "test")
DAY_US = 86_400_000_000
START_US = 1_790_812_800_000_000  # 2026-10-01T00:00:00Z (synthetic calendar)


@dataclass(frozen=True)
class DataConfig:
    n_days: int = 8
    n_val_days: int = 1
    n_test_days: int = 2
    sim_steps_per_day: int = 18_000  # 30 simulated minutes at 100 ms
    grid_us: int = 250_000
    window: int = 40  # T: 10 s, covers placement -> rest (<= 4 s) -> cancel
    horizon: int = 8  # h: 2 s label tolerance after cancellation
    train_stride: int = 2
    eval_stride: int = 1
    base_seed: int = 7
    market: MarketParams = field(default_factory=MarketParams)
    injection: InjectionParams = field(default_factory=InjectionParams)
    rules: RuleParams = field(default_factory=RuleParams)

    @property
    def grid_steps_per_day(self) -> int:
        return self.sim_steps_per_day * self.market.step_ms * 1_000 // self.grid_us


def assign_splits(cfg: DataConfig) -> list[str]:
    n_train = cfg.n_days - cfg.n_val_days - cfg.n_test_days
    if n_train < 1:
        raise ValueError("need at least one training day")
    return ["train"] * n_train + ["val"] * cfg.n_val_days + ["test"] * cfg.n_test_days


def injection_seed(cfg: DataConfig, split: str, day: int) -> int:
    """Independent injection seed stream per split (spec J: injector leakage)."""
    return cfg.base_seed * 100_000 + (SPLITS.index(split) + 1) * 1_000 + day


def walk_forward_folds(n_days: int, min_train: int) -> list[tuple[list[int], int, int]]:
    """Spec I folds: train on days [0..k], validate on k+1, test on k+2."""
    return [(list(range(k + 1)), k + 1, k + 2) for k in range(min_train - 1, n_days - 2)]


@dataclass
class DayData:
    day_index: int
    split: str
    sim: SimulatedDay
    grid: GridDay
    ladder: np.ndarray  # [T, C, P]
    level: np.ndarray  # [T, 2, 2L]
    context: np.ndarray  # [T, F]
    scale: np.ndarray  # [T]
    labels: DayLabels
    rule_score: np.ndarray  # [T]  B1 / weak-label score
    rule_patterns: list[RulePattern]


def build_day(cfg: DataConfig, day: int, split: str) -> DayData:
    specs = schedule(
        cfg.injection,
        cfg.market,
        cfg.sim_steps_per_day,
        injection_seed(cfg, split, day),
        f"{split}-d{day}",
    )
    sim = SyntheticMarket(
        cfg.market, seed=cfg.base_seed * 1_000 + day, injection_seed=injection_seed(cfg, split, day)
    ).simulate(cfg.sim_steps_per_day, START_US + day * DAY_US, day, specs)
    g = reconstruct(
        sim.events, cfg.market.tick_size, sim.start_time_us, cfg.grid_steps_per_day, cfg.grid_us
    )
    rule, patterns = rule_scores(g, cfg.horizon, cfg.rules)
    return DayData(
        day_index=day,
        split=split,
        sim=sim,
        grid=g,
        ladder=ladder_tensor(g),
        level=level_tensor(g),
        context=context_series(g),
        scale=size_scale(g),
        labels=label_day(sim, g, cfg.horizon),
        rule_score=rule,
        rule_patterns=patterns,
    )


def build_days(cfg: DataConfig) -> list[DayData]:
    return [build_day(cfg, d, s) for d, s in enumerate(assign_splits(cfg))]


def window_index(days: list[DayData], split: str, window: int, stride: int) -> np.ndarray:
    """``[N, 2]`` (day position, end step) for windows of ``split`` whose rows are all valid."""
    rows = []
    for pos, d in enumerate(days):
        if d.split != split:
            continue
        valid = d.grid.valid
        bad = np.convolve(~valid, np.ones(window, int), mode="full")[: len(valid)]
        for t in range(window - 1, len(valid), stride):
            if bad[t] == 0:
                rows.append((pos, t))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


@dataclass
class Standardizer:
    """Per-feature mean/std fitted on training data only; records what it was fitted on."""

    mean: np.ndarray
    std: np.ndarray
    fitted_on_split: str
    fit_data_end_us: int  # latest grid time used for fitting

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def fit_standardizer(days: list[DayData], attr: str, axis_keep: int) -> Standardizer:
    """Fit on ``train`` days' valid steps only. ``axis_keep``: feature axis after step axis."""
    arrays, end = [], 0
    for d in days:
        if d.split != "train":
            continue
        a = getattr(d, attr)[d.grid.valid]
        arrays.append(a)
        end = max(end, int(d.grid.grid_times_us[d.grid.valid].max()))
    x = np.concatenate(arrays)
    axes = tuple(i for i in range(x.ndim) if i != axis_keep)
    mean = x.mean(axis=axes, keepdims=True)[0]
    std = x.std(axis=axes, keepdims=True)[0] + 1e-6
    return Standardizer(mean, std, "train", end)


def split_time_range(days: list[DayData], split: str) -> tuple[int, int]:
    times = [d.grid.grid_times_us for d in days if d.split == split]
    return int(min(t[0] for t in times)), int(max(t[-1] for t in times))
