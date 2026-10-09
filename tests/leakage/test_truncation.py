"""Truncation tests (spec J): a feature at t computed on data up to t equals the same
feature at t computed on the full day. Catches any look-ahead in feature code."""

import numpy as np
import pytest

from stlens.book.builder import reconstruct
from stlens.features.ladder import (
    CHANNELS,
    CONTEXT,
    context_series,
    ladder_tensor,
    level_tensor,
    size_scale,
)
from stlens.features.tabular import window_features
from stlens.ingestion.synthetic import (
    EpisodeKind,
    InjectionSpec,
    MarketParams,
    SyntheticMarket,
)
from stlens.labeling.rules import RuleParams, detect_patterns, rule_scores
from stlens.schemas.events import Side

pytestmark = pytest.mark.leakage

T0 = 1_790_812_800_000_000
P = MarketParams()
N_STEPS = 600  # grid steps (250 ms)
CUTS = [45, 199, 377, 599]


@pytest.fixture(scope="module")
def grid():
    inj = [
        InjectionSpec(f"s{i}", EpisodeKind.SPOOF, side, 150 + 400 * i, 20, 3, 1, 3.0, 1.0, 8)
        for i, side in enumerate([Side.BID, Side.ASK, Side.BID, Side.ASK, Side.BID])
    ]
    day = SyntheticMarket(P, seed=11, injection_seed=3).simulate(N_STEPS * 3, T0, 0, inj)
    return reconstruct(day.events, P.tick_size, T0, N_STEPS)


@pytest.mark.parametrize("cut", CUTS)
def test_ladder_and_context_truncation(grid, cut):
    full_l, full_c = ladder_tensor(grid), context_series(grid)
    part = grid.prefix(cut + 1)
    np.testing.assert_array_equal(ladder_tensor(part)[cut], full_l[cut], err_msg=str(CHANNELS))
    np.testing.assert_array_equal(context_series(part)[cut], full_c[cut], err_msg=str(CONTEXT))
    np.testing.assert_array_equal(level_tensor(part)[cut], level_tensor(grid)[cut])


@pytest.mark.parametrize("cut", CUTS[1:])
def test_window_feature_truncation(grid, cut):
    window = 40
    full = window_features(
        grid, ladder_tensor(grid), context_series(grid), size_scale(grid), np.array([cut]), window
    )[0]
    part_grid = grid.prefix(cut + 1)
    part = window_features(
        part_grid,
        ladder_tensor(part_grid),
        context_series(part_grid),
        size_scale(part_grid),
        np.array([cut]),
        window,
    )[0]
    np.testing.assert_array_equal(part, full)


@pytest.mark.parametrize("cut", CUTS)
def test_rule_engine_truncation(grid, cut):
    full, _ = rule_scores(grid, horizon=4)
    part, _ = rule_scores(grid.prefix(cut + 1), horizon=4)
    np.testing.assert_array_equal(part[: cut + 1], full[: cut + 1])


def test_rule_engine_fires_on_injected_spoofs(grid):
    # Sanity (not a performance claim): the rule finds at least one of the injected spoofs.
    patterns = detect_patterns(grid, RuleParams())
    assert any(p.score > 0 for p in patterns)
