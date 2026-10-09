"""Leakage suite for datasets (spec J): split overlap, episode disjointness, scaler fit,
injector seeds, natural class rate."""

import dataclasses

import numpy as np
import pytest

from stlens.datasets.build import (
    DataConfig,
    assign_splits,
    build_days,
    fit_standardizer,
    injection_seed,
    split_time_range,
    walk_forward_folds,
    window_index,
)
from stlens.datasets.torch_data import WindowDataset

pytestmark = pytest.mark.leakage

CFG = DataConfig(n_days=4, n_val_days=1, n_test_days=1, sim_steps_per_day=3_000)


@pytest.fixture(scope="module")
def days():
    return build_days(CFG)


def test_splits_are_chronological(days):
    assert assign_splits(CFG) == ["train", "train", "val", "test"]
    tr, va, te = (split_time_range(days, s) for s in ("train", "val", "test"))
    assert tr[1] < va[0] <= va[1] < te[0]


def test_window_spans_never_cross_splits(days):
    spans = {}
    for split in ("train", "val", "test"):
        idx = window_index(days, split, CFG.window, 1)
        times = [
            (days[p].grid.grid_times_us[t - CFG.window + 1], days[p].grid.grid_times_us[t])
            for p, t in idx
        ]
        spans[split] = (min(a for a, _ in times), max(b for _, b in times))
        assert all(days[p].split == split for p, _ in idx)
    assert spans["train"][1] < spans["val"][0] and spans["val"][1] < spans["test"][0]


def test_windows_cover_only_valid_steps_within_one_day(days):
    idx = window_index(days, "train", CFG.window, CFG.train_stride)
    assert (idx[:, 1] >= CFG.window - 1).all()
    for p, t in idx:
        assert days[p].grid.valid[t - CFG.window + 1 : t + 1].all()


def test_episode_ids_disjoint_across_splits(days):
    ids = {
        s: {r.spec.episode_id for d in days if d.split == s for r in d.sim.episodes}
        for s in ("train", "val", "test")
    }
    assert all(ids.values())
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])


def test_injection_seeds_differ_per_split():
    seeds = {(s, d): injection_seed(CFG, s, d) for s in ("train", "val", "test") for d in range(4)}
    assert len(set(seeds.values())) == len(seeds)


def test_scaler_fitted_on_train_only(days):
    std = fit_standardizer(days, "ladder", 1)
    assert std.fitted_on_split == "train"
    assert std.fit_data_end_us <= split_time_range(days, "train")[1]
    assert std.fit_data_end_us < split_time_range(days, "val")[0]
    # Changing validation/test data must not change the fitted statistics.
    tampered = [
        dataclasses.replace(d, ladder=d.ladder * 0 + 99.0) if d.split != "train" else d
        for d in days
    ]
    np.testing.assert_array_equal(fit_standardizer(tampered, "ladder", 1).mean, std.mean)


def test_eval_splits_keep_natural_class_rate(days):
    for split in ("val", "test"):
        idx = window_index(days, split, CFG.window, CFG.eval_stride)
        ys = np.array([days[p].labels.y[t] for p, t in idx])
        full = np.concatenate([d.labels.y[CFG.window - 1 :] for d in days if d.split == split])
        assert ys.mean() == pytest.approx(full.mean(), abs=1e-9)


def test_dataset_items_shapes_and_labels(days):
    std = {a: fit_standardizer(days, a, 1) for a in ("ladder", "context", "level")}
    idx = window_index(days, "val", CFG.window, 1)
    ds = WindowDataset(days, idx, CFG.window, std["ladder"], std["context"], std["level"])
    item = ds[len(ds) // 2]
    C, P = days[0].ladder.shape[1:]
    assert item["ladder"].shape == (C, CFG.window, P)
    assert item["context"].shape == (CFG.window, days[0].context.shape[1])
    p, t = idx[len(ds) // 2]
    assert item["y"].item() == days[p].labels.y[t]


def test_walk_forward_folds_are_ordered():
    folds = walk_forward_folds(6, min_train=2)
    assert folds[0] == ([0, 1], 2, 3)
    assert all(max(tr) < va < te for tr, va, te in folds)
