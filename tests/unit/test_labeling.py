"""Injection schedule, ground-truth labels, round trip and artefact probe (Phase 4)."""

from itertools import pairwise

import numpy as np
import pytest

from stlens.book.builder import reconstruct
from stlens.ingestion.synthetic import EpisodeKind, MarketParams, SyntheticMarket
from stlens.labeling.injector import (
    InjectionParams,
    label_day,
    large_size_threshold,
    schedule,
)
from stlens.labeling.probe import probe_auc

T0 = 1_790_812_800_000_000
P = MarketParams()
IP = InjectionParams()
N_SIM = 6_000  # 10 simulated minutes
HORIZON = 8


def simulate(day: int, inj_seed: int):
    specs = schedule(IP, P, N_SIM, seed=inj_seed, tag=f"d{day}")
    sim = SyntheticMarket(P, seed=day, injection_seed=inj_seed)
    return sim.simulate(N_SIM, T0 + day * 86_400_000_000, day, specs)


@pytest.fixture(scope="module")
def days():
    return [simulate(d, 500 + d) for d in range(3)]


def test_schedule_deterministic_and_non_overlapping():
    a = schedule(IP, P, N_SIM, seed=9, tag="x")
    assert a == schedule(IP, P, N_SIM, seed=9, tag="x")
    assert a != schedule(IP, P, N_SIM, seed=10, tag="x")
    for prev, nxt in pairwise(a):
        assert nxt.place_step - (prev.place_step + prev.rest_steps) >= IP.min_gap_steps
    assert len({s.episode_id for s in a}) == len(a)
    assert {s.kind for s in a} == set(EpisodeKind)


def test_every_scheduled_episode_is_logged(days):
    for d in days:
        specs = schedule(IP, P, N_SIM, seed=500 + d.day_index, tag=f"d{d.day_index}")
        assert [r.spec for r in d.episodes] == specs
        placed = [r for r in d.episodes if r.placed]
        assert all(r.end_time_us > r.place_time_us for r in placed)
        assert not any(r.is_positive for r in d.episodes if not r.placed)
        assert len(placed) >= 0.9 * len(specs)


def test_round_trip_injected_spoofs_visible_in_reconstruction(days):
    """Logged spoof placements/cancels are recovered as add/cancel flow on the right side."""
    d = days[0]
    g = reconstruct(d.events, P.tick_size, d.start_time_us, 2_400)
    lab = label_day(d, g, HORIZON)
    H = g.half_buckets
    checked = 0
    for rec in d.episodes:
        if not rec.is_positive or rec.spec.layers != 1:
            continue
        c = lab.cancel_step[rec.spec.episode_id]
        a = lab.place_step[rec.spec.episode_id]
        if lab.loc[c] < 0:
            continue  # spoof outside the ladder window: not observable on it
        half = slice(0, H) if rec.spec.side.value == "bid" else slice(H, 2 * H)
        assert g.cancel_vol[c, half].sum() >= rec.cancelled_size - 1e-9
        assert g.add_vol[a, half].sum() >= rec.spec.size - 1e-9
        checked += 1
    assert checked >= 3


def test_labels_match_episode_log(days):
    d = days[1]
    g = reconstruct(d.events, P.tick_size, d.start_time_us, 2_400)
    lab = label_day(d, g, HORIZON)
    expected = np.zeros(g.n_steps, np.int8)
    for rec in d.episodes:
        if rec.is_positive:
            c = lab.cancel_step[rec.spec.episode_id]
            expected[c : c + HORIZON] = 1
    assert set(lab.cancel_step) == {r.spec.episode_id for r in d.episodes if r.placed}
    np.testing.assert_array_equal(lab.y, expected)
    # Hard negatives never produce positive labels, and side labels exist only on positives.
    assert set(np.unique(lab.kind[lab.y == 1])) == {"spoof"}
    assert ((lab.side > 0) == (lab.y == 1)).all()
    assert 0 < lab.y.mean() < 0.1


def test_artefact_probe_near_chance(days):
    more = days + [simulate(d, 900 + d) for d in range(3, 9)]
    auc, n_inj, n_bg = probe_auc(more, large_size_threshold(P, IP.large_quantile_z))
    assert n_inj >= 30 and n_bg >= n_inj
    assert abs(auc - 0.5) < 0.1, f"injected orders distinguishable at placement: AUC={auc:.3f}"
