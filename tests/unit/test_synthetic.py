"""Synthetic market source: determinism, canonical validity, injection ground truth."""

from itertools import groupby

import pytest

from stlens.ingestion.synthetic import (
    EpisodeKind,
    InjectionSpec,
    MarketParams,
    SyntheticMarket,
)
from stlens.schemas.events import AggressorSide, BookDelta, BookSnapshot, Side, Trade

T0 = 1_790_812_800_000_000
P = MarketParams()


def simulate(n_steps=1500, seed=1, injections=None):
    return SyntheticMarket(P, seed=seed, injection_seed=7).simulate(n_steps, T0, 0, injections)


def messages(day):
    deltas = [e for e in day.events if isinstance(e, BookDelta)]
    return [list(g) for _, g in groupby(deltas, key=lambda d: d.source.recv_seq)]


def test_deterministic_given_seed():
    assert simulate(seed=3).events == simulate(seed=3).events
    assert simulate(seed=3).events != simulate(seed=4).events


def test_starts_with_snapshot_and_has_trades():
    day = simulate()
    assert isinstance(day.events[0], BookSnapshot)
    assert any(isinstance(e, Trade) for e in day.events)


def test_update_ids_are_gap_free_and_messages_complete():
    day = simulate()
    last = day.events[0].last_update_id
    for msg in messages(day):
        assert [d.level_index for d in msg] == list(range(msg[0].level_count))
        assert {(d.first_update_id, d.final_update_id) for d in msg} == {
            (msg[0].first_update_id, msg[0].final_update_id)
        }
        assert msg[0].first_update_id == last + 1
        last = msg[0].final_update_id


def test_trades_never_later_than_following_depth_message():
    day = simulate()
    pending: list[Trade] = []
    for e in day.events:
        if isinstance(e, Trade):
            pending.append(e)
        elif isinstance(e, BookDelta):
            assert all(t.exchange_time_us <= e.exchange_time_us for t in pending)
            pending = []


def test_spoof_episode_ground_truth():
    spec = InjectionSpec("s1", EpisodeKind.SPOOF, Side.BID, 100, 20, 3, 1, 2.0, 1.0, 8)
    day = simulate(injections=[spec])
    (rec,) = day.episodes
    assert rec.spec == spec
    assert rec.is_positive
    assert rec.executed_size == 0.0
    assert rec.cancelled_size == pytest.approx(2.0)
    assert rec.place_time_us < min(rec.payoff_trade_times_us) <= rec.end_time_us
    # Payoff trades are on the opposite side: buyers lifting asks for a bid spoof.
    payoff = [
        e
        for e in day.events
        if isinstance(e, Trade) and e.exchange_time_us in rec.payoff_trade_times_us
    ]
    assert payoff
    assert {t.aggressor_side for t in payoff} == {AggressorSide.BUY}


def test_spoof_cancellation_visible_only_through_canonical_deltas():
    spec = InjectionSpec("s1", EpisodeKind.SPOOF, Side.ASK, 100, 20, 2, 1, 2.0, 1.0, 8)
    day = simulate(injections=[spec])
    (rec,) = day.episodes
    price = rec.prices_ticks[0]
    at_level = [
        e
        for e in day.events
        if isinstance(e, BookDelta) and e.side is Side.ASK and int(e.price / P.tick_size) == price
    ]
    placed = [d for d in at_level if d.exchange_time_us >= rec.place_time_us]
    # The placement and the cancellation both appear as level-size changes, nothing else.
    assert placed[0].exchange_time_us - rec.place_time_us <= P.step_ms * 1_000
    assert any(abs(d.exchange_time_us - rec.end_time_us) <= P.step_ms * 1_000 for d in placed)


def test_hard_negative_executed_is_not_positive():
    spec = InjectionSpec("h1", EpisodeKind.HN_EXECUTED, Side.ASK, 100, 15, 0, 1, 2.0)
    (rec,) = simulate(injections=[spec]).episodes
    assert rec.executed_size == pytest.approx(2.0)
    assert not rec.is_positive


def test_injection_does_not_change_background_randomness():
    # Background draws come from the background RNG only, so the event stream before the
    # first injection is identical with and without injections.
    spec = InjectionSpec("s1", EpisodeKind.SPOOF, Side.BID, 500, 20, 3, 1, 2.0, 1.0, 8)
    plain, injected = simulate().events, simulate(injections=[spec]).events
    cutoff = T0 + 500 * P.step_ms * 1_000
    assert [e for e in plain if e.exchange_time_us <= cutoff] == [
        e for e in injected if e.exchange_time_us <= cutoff
    ]
