"""BookBuilder on hand-made books, attribution on crafted sequences, grid causality."""

from decimal import Decimal

import numpy as np
import pytest

from stlens.book.builder import (
    BookBuilder,
    BookIntegrityError,
    ladder_bucket,
    reconstruct,
    to_ticks,
)
from stlens.ingestion.synthetic import MarketParams, SyntheticMarket
from stlens.schemas.events import (
    AggressorSide,
    BookDelta,
    BookSnapshot,
    PriceLevel,
    Side,
    SourceRef,
    Trade,
)

TICK = Decimal("0.01")
T0 = 1_790_812_800_000_000
D = Decimal


def snap(bids, asks, last=100, t=T0):
    return BookSnapshot(
        "SYN",
        last,
        tuple(PriceLevel(D(p), D(s)) for p, s in bids),
        tuple(PriceLevel(D(p), D(s)) for p, s in asks),
        t,
        t,
        SourceRef("synthetic", "snap", "s", 0),
    )


def msg(changes, first, t, seq, final=None):
    final = final if final is not None else first + len(changes) - 1
    src = SourceRef("synthetic", "depth", "s", seq)
    return [
        BookDelta("SYN", side, D(p), D(s), t, t, first, final, i, len(changes), src)
        for i, (side, p, s) in enumerate(changes)
    ]


def trade(p, s, aggr, t, seq):
    return Trade("SYN", seq, D(p), D(s), aggr, t, t, t, SourceRef("synthetic", "trade", "s", seq))


BOOK = snap([("100.00", "1.0"), ("99.99", "2.0")], [("100.01", "1.5"), ("100.02", "3.0")])


def test_known_final_state_after_messages():
    b = BookBuilder(TICK)
    b.apply_snapshot(BOOK)
    b.apply_message(msg([(Side.BID, "100.00", "0.5"), (Side.ASK, "100.03", "4.0")], 101, T0 + 1, 1))
    b.apply_message(msg([(Side.BID, "99.99", "0")], 103, T0 + 2, 2))
    assert b.levels[Side.BID] == {10000: 0.5}
    assert b.levels[Side.ASK] == {10001: 1.5, 10002: 3.0, 10003: 4.0}
    assert (b.best(Side.BID), b.best(Side.ASK), b.last_update_id) == (10000, 10001, 103)


def test_gap_unsyncs_book_until_next_snapshot():
    b = BookBuilder(TICK)
    b.apply_snapshot(BOOK)
    assert b.apply_message(msg([(Side.BID, "100.00", "0.5")], 105, T0 + 1, 1)) is None
    assert (b.synced, b.gaps) == (False, 1)
    # Later contiguous-looking messages are still ignored: no guessing.
    assert b.apply_message(msg([(Side.BID, "100.00", "0.7")], 106, T0 + 2, 2)) is None
    b.apply_snapshot(snap([("100.00", "9.0")], [("100.01", "1.0")], last=200))
    assert b.synced and b.levels[Side.BID] == {10000: 9.0}


def test_stale_message_ignored():
    b = BookBuilder(TICK)
    b.apply_snapshot(BOOK)
    assert b.apply_message(msg([(Side.BID, "100.00", "7")], 99, T0 + 1, 1, final=100)) is None
    assert b.stale_messages == 1 and b.levels[Side.BID][10000] == 1.0


def test_incomplete_message_rejected():
    b = BookBuilder(TICK)
    b.apply_snapshot(BOOK)
    m = msg([(Side.BID, "100.00", "1"), (Side.BID, "99.99", "1")], 101, T0 + 1, 1)
    with pytest.raises(BookIntegrityError, match="incomplete"):
        b.apply_message(m[:1])


def test_tick_alignment_enforced():
    assert to_ticks(D("67012.34"), TICK) == 6701234
    with pytest.raises(BookIntegrityError, match="not aligned"):
        to_ticks(D("67012.345"), TICK)


def test_ladder_layout_is_mirrored_around_mid():
    mid = 10000.5  # bid 10000 / ask 10001
    assert ladder_bucket(Side.BID, 10000, mid, 3) == 2  # bid_0 next to the spread
    assert ladder_bucket(Side.BID, 9998, mid, 3) == 0  # bid_2 at the far left
    assert ladder_bucket(Side.ASK, 10001, mid, 3) == 3  # ask_0
    assert ladder_bucket(Side.ASK, 10003, mid, 3) == 5
    assert ladder_bucket(Side.ASK, 10004, mid, 3) is None


def grid_of(events, n=4):
    return reconstruct(events, TICK, T0, n, grid_us=1_000, half_buckets=2, half_levels=2)


def test_attribution_cancel_without_trade():
    events = [BOOK, *msg([(Side.BID, "99.99", "0.5")], 101, T0 + 500, 1)]
    g = grid_of(events)
    # bid 99.99 is 1 tick beyond the bid touch -> bucket bid_1 = index 0
    assert g.cancel_vol[0].tolist() == [1.5, 0.0, 0.0, 0.0]
    assert g.add_vol[0].sum() == 0.0 and g.exec_vol[0].sum() == 0.0


def test_attribution_execution_matched_to_trades():
    events = [
        BOOK,
        trade("100.01", "1.0", AggressorSide.BUY, T0 + 400, 1),
        *msg([(Side.ASK, "100.01", "0.5")], 101, T0 + 500, 2),
    ]
    g = grid_of(events)
    assert g.exec_vol[0].tolist() == [0.0, 0.0, 1.0, 0.0]  # ask_0
    assert g.cancel_vol[0].sum() == 0.0 and g.add_vol[0].sum() == 0.0
    assert (g.buy_vol[0], g.sell_vol[0], g.n_trades[0]) == (1.0, 0.0, 1)


def test_attribution_mixed_execution_and_cancel():
    events = [
        BOOK,
        trade("100.01", "0.5", AggressorSide.BUY, T0 + 400, 1),
        *msg([(Side.ASK, "100.01", "0.2")], 101, T0 + 500, 2),
    ]
    g = grid_of(events)
    assert g.exec_vol[0, 2] == pytest.approx(0.5)
    assert g.cancel_vol[0, 2] == pytest.approx(0.8)


def test_grid_state_uses_only_past_messages():
    events = [BOOK, *msg([(Side.BID, "100.00", "5.0")], 101, T0 + 1_500, 1)]
    g = grid_of(events)
    # grid times are T0+1000, T0+2000, ...: the message at T0+1500 affects only step 1+.
    assert g.depth[0, 1] == 1.0 and g.depth[1, 1] == 5.0
    assert g.add_vol[0].sum() == 0.0 and g.add_vol[1, 1] == pytest.approx(4.0)


def test_grid_prefix_invariance_on_synthetic_day():
    """Truncating the event stream after time t must not change grid rows <= t."""
    p = MarketParams()
    day = SyntheticMarket(p, seed=5).simulate(800, T0)
    full = reconstruct(day.events, p.tick_size, T0, 300)
    cut_t = T0 + 150 * 250_000
    part = reconstruct([e for e in day.events if e.exchange_time_us <= cut_t], p.tick_size, T0, 150)
    for name in ("depth", "add_vol", "cancel_vol", "exec_vol", "mid", "buy_vol", "ofi"):
        np.testing.assert_array_equal(getattr(full, name)[:150], getattr(part, name)[:150])
    assert full.health["gaps"] == 0 and full.valid.all()
