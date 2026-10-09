"""Deterministic order-book reconstruction, flow attribution and grid resampling (Phase 2).

Input: canonical events in emission order (one ``BookSnapshot`` first; depth messages as
consecutive ``BookDelta`` groups; ``Trade`` events). Output: a :class:`GridDay` of numpy
arrays on a fixed clock grid. Everything is strictly forward-moving: the state at grid
time ``g`` uses only messages with exchange time ``<= g``.

Attribution (spec B.5) per depth message and price level::

    exec   = traded volume at that price against that side, from trades received since
             the previous message whose time is <= this message's time
    resid  = (new_size - old_size) + exec
    add    = max(resid, 0);   cancel = max(-resid, 0)

This is an approximation: aggregated L2 data cannot separate an add and a cancel at the
same level within one message, and trades are matched by price and time only (there is
no trade-to-depth linkage). Both limitations are inherent and documented.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from stlens.schemas.events import AggressorSide, BookDelta, BookSnapshot, Side, Trade


class BookIntegrityError(ValueError):
    """A depth message is malformed or does not fit the book's sequence."""


def to_ticks(price: Decimal, tick_size: Decimal) -> int:
    """Exact price -> integer ticks; rejects prices not on the tick grid (spec B.3)."""
    q = price / tick_size
    if q != q.to_integral_value():
        raise BookIntegrityError(f"price {price} is not aligned to tick size {tick_size}")
    return int(q)


@dataclass(frozen=True)
class LevelChange:
    side: Side
    price: int
    old_size: float
    new_size: float


class BookBuilder:
    """L2 book in integer ticks with sequence checking (Binance-style ``U``/``u`` rules)."""

    def __init__(self, tick_size: Decimal) -> None:
        self.tick_size = tick_size
        self.levels: dict[Side, dict[int, float]] = {Side.BID: {}, Side.ASK: {}}
        self.last_update_id: int | None = None
        self.synced = False
        self.gaps = 0
        self.stale_messages = 0

    def apply_snapshot(self, snap: BookSnapshot) -> None:
        self.levels = {
            Side.BID: {
                to_ticks(lv.price, self.tick_size): float(lv.size)
                for lv in snap.bids
                if lv.size > 0
            },
            Side.ASK: {
                to_ticks(lv.price, self.tick_size): float(lv.size)
                for lv in snap.asks
                if lv.size > 0
            },
        }
        self.last_update_id = snap.last_update_id
        self.synced = True

    def apply_message(self, deltas: Sequence[BookDelta]) -> list[LevelChange] | None:
        """Apply one depth message. Returns the level changes, or None if not applied.

        * ``final_update_id <= last`` -> stale, ignored.
        * ``first_update_id > last + 1`` -> gap: the book is marked unsynced and every
          message is ignored until the next snapshot (no repair, no guessing).
        """
        if not deltas:
            raise BookIntegrityError("empty message")
        first, final = deltas[0].first_update_id, deltas[0].final_update_id
        n = deltas[0].level_count
        if len(deltas) != n or [d.level_index for d in deltas] != list(range(n)):
            raise BookIntegrityError("incomplete or reordered depth message")
        if any((d.first_update_id, d.final_update_id) != (first, final) for d in deltas):
            raise BookIntegrityError("mixed update ids within one message")
        if not self.synced or self.last_update_id is None:
            return None
        if final <= self.last_update_id:
            self.stale_messages += 1
            return None
        if first > self.last_update_id + 1:
            self.gaps += 1
            self.synced = False
            return None
        changes = []
        for d in deltas:
            price = to_ticks(d.price, self.tick_size)
            book = self.levels[d.side]
            old = book.get(price, 0.0)
            new = float(d.new_size)
            if new == 0.0:
                book.pop(price, None)
            else:
                book[price] = new
            changes.append(LevelChange(d.side, price, old, new))
        self.last_update_id = final
        return changes

    def best(self, side: Side) -> int | None:
        lv = self.levels[side]
        if not lv:
            return None
        return max(lv) if side is Side.BID else min(lv)

    def top(self, side: Side, n: int) -> list[tuple[int, float]]:
        items = sorted(self.levels[side].items(), reverse=side is Side.BID)
        return items[:n]


@dataclass
class GridDay:
    """Book and flow state sampled on a fixed grid. Arrays are indexed by grid step.

    Ladder axis (length ``2*H``): ``[bid_{H-1} ... bid_0 | ask_0 ... ask_{H-1}]`` where bucket
    ``k`` holds prices ``k`` ticks beyond the mid reference on that side (spec C).
    Level-indexed arrays (length ``2*L``) hold the k-th best bid/ask instead (DeepLOB-style).
    """

    grid_times_us: np.ndarray  # [T] int64, grid timestamps (end of each interval)
    valid: np.ndarray  # [T] bool, book synced and two-sided at this step
    best_bid: np.ndarray  # [T] float64 ticks (nan if invalid)
    best_ask: np.ndarray  # [T]
    mid: np.ndarray  # [T] float64 ticks; the ladder reference
    depth: np.ndarray  # [T, 2H] resting size per price bucket
    add_vol: np.ndarray  # [T, 2H] inferred added volume in (g_{t-1}, g_t]
    cancel_vol: np.ndarray  # [T, 2H] inferred cancelled volume
    exec_vol: np.ndarray  # [T, 2H] executed volume
    lvl_size: np.ndarray  # [T, 2L] size at k-th best level, layout like the ladder
    lvl_dist: np.ndarray  # [T, 2L] distance of k-th best level from mid (ticks)
    buy_vol: np.ndarray  # [T] aggressive buy volume in interval
    sell_vol: np.ndarray  # [T]
    n_trades: np.ndarray  # [T]
    ofi: np.ndarray  # [T] order flow imbalance at the best quotes (Cont et al. 2014)
    tick_size: Decimal = Decimal("0.01")
    half_buckets: int = 10
    half_levels: int = 10
    health: dict[str, int] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.grid_times_us)

    def prefix(self, n: int) -> "GridDay":
        """The first ``n`` grid steps (what a live system would have seen at step n-1)."""
        arrays = {
            name: getattr(self, name)[:n]
            for name in (
                "grid_times_us",
                "valid",
                "best_bid",
                "best_ask",
                "mid",
                "depth",
                "add_vol",
                "cancel_vol",
                "exec_vol",
                "lvl_size",
                "lvl_dist",
                "buy_vol",
                "sell_vol",
                "n_trades",
                "ofi",
            )
        }
        return GridDay(
            **arrays,
            tick_size=self.tick_size,
            half_buckets=self.half_buckets,
            half_levels=self.half_levels,
            health=dict(self.health),
        )


def ladder_bucket(side: Side, price: int, mid: float, half: int) -> int | None:
    """Index of ``price`` on the mirrored ladder relative to ``mid``; None if outside."""
    if side is Side.BID:
        k = int(np.floor(mid - price))
        return half - 1 - k if 0 <= k < half else None
    k = int(np.floor(price - mid))
    return half + k if 0 <= k < half else None


def reconstruct(
    events: Iterable[BookSnapshot | BookDelta | Trade],
    tick_size: Decimal,
    start_time_us: int,
    n_steps: int,
    grid_us: int = 250_000,
    half_buckets: int = 10,
    half_levels: int = 10,
) -> GridDay:
    """Replay canonical events in order and sample the book on the grid ``start + (j+1)*grid``."""
    H, L = half_buckets, half_levels
    grid = start_time_us + grid_us * (np.arange(n_steps, dtype=np.int64) + 1)
    out = {
        "valid": np.zeros(n_steps, bool),
        "best_bid": np.full(n_steps, np.nan),
        "best_ask": np.full(n_steps, np.nan),
        "mid": np.full(n_steps, np.nan),
        "depth": np.zeros((n_steps, 2 * H)),
        "add_vol": np.zeros((n_steps, 2 * H)),
        "cancel_vol": np.zeros((n_steps, 2 * H)),
        "exec_vol": np.zeros((n_steps, 2 * H)),
        "lvl_size": np.zeros((n_steps, 2 * L)),
        "lvl_dist": np.zeros((n_steps, 2 * L)),
        "buy_vol": np.zeros(n_steps),
        "sell_vol": np.zeros(n_steps),
        "n_trades": np.zeros(n_steps),
        "ofi": np.zeros(n_steps),
    }
    book = BookBuilder(tick_size)
    flows: dict[tuple[Side, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    trades_since_msg: list[Trade] = []
    interval_trades = [0.0, 0.0, 0]
    ofi = 0.0
    prev_top: tuple[int, float, int, float] | None = None
    j = 0
    health = {"messages": 0, "gaps": 0, "stale": 0, "unmatched_trade_volume_events": 0}

    def flush_until(t_us: int) -> None:
        """Emit grid rows for all grid times < t_us (state is final for them)."""
        nonlocal j, ofi
        while j < n_steps and grid[j] < t_us:
            emit(j)
            flows.clear()
            interval_trades[:] = [0.0, 0.0, 0]
            ofi = 0.0
            j += 1

    def emit(i: int) -> None:
        bb, ba = book.best(Side.BID), book.best(Side.ASK)
        out["buy_vol"][i], out["sell_vol"][i], out["n_trades"][i] = interval_trades
        out["ofi"][i] = ofi
        if not book.synced or bb is None or ba is None:
            return
        mid = (bb + ba) / 2.0
        out["valid"][i] = True
        out["best_bid"][i], out["best_ask"][i], out["mid"][i] = bb, ba, mid
        for side in (Side.BID, Side.ASK):
            for price, size in book.levels[side].items():
                b = ladder_bucket(side, price, mid, H)
                if b is not None:
                    out["depth"][i, b] += size
        for (side, price), (add, cancel, ex) in flows.items():
            b = ladder_bucket(side, price, mid, H)
            if b is not None:
                out["add_vol"][i, b] += add
                out["cancel_vol"][i, b] += cancel
                out["exec_vol"][i, b] += ex
        for k, (price, size) in enumerate(book.top(Side.BID, L)):
            out["lvl_size"][i, L - 1 - k] = size
            out["lvl_dist"][i, L - 1 - k] = mid - price
        for k, (price, size) in enumerate(book.top(Side.ASK, L)):
            out["lvl_size"][i, L + k] = size
            out["lvl_dist"][i, L + k] = price - mid

    def apply(msg: list[BookDelta]) -> None:
        nonlocal ofi, prev_top
        changes = book.apply_message(msg)
        health["messages"] += 1
        t_msg = msg[0].exchange_time_us
        # Trades seen since the previous message with time <= this message's time are
        # matched to it; any later-stamped trade waits for the next message.
        traded: dict[tuple[Side, int], float] = defaultdict(float)
        later = [tr for tr in trades_since_msg if tr.exchange_time_us > t_msg]
        for tr in trades_since_msg:
            if tr.exchange_time_us <= t_msg:
                hit = Side.ASK if tr.aggressor_side is AggressorSide.BUY else Side.BID
                traded[(hit, to_ticks(tr.price, tick_size))] += float(tr.size)
        trades_since_msg[:] = later
        if changes is None:
            return
        for ch in changes:
            ex = traded.pop((ch.side, ch.price), 0.0)
            resid = (ch.new_size - ch.old_size) + ex
            f = flows[(ch.side, ch.price)]
            f[0] += max(resid, 0.0)
            f[1] += max(-resid, 0.0)
            f[2] += ex
        health["unmatched_trade_volume_events"] += len(traded)
        bb, ba = book.best(Side.BID), book.best(Side.ASK)
        if bb is not None and ba is not None:
            qb, qa = book.levels[Side.BID][bb], book.levels[Side.ASK][ba]
            if prev_top is not None:
                pb, pqb, pa, pqa = prev_top
                ofi += (qb if bb >= pb else 0.0) - (pqb if bb <= pb else 0.0)
                ofi -= (qa if ba <= pa else 0.0) - (pqa if ba >= pa else 0.0)
            prev_top = (bb, qb, ba, qa)

    pending: list[BookDelta] = []
    for ev in events:
        if pending and (not isinstance(ev, BookDelta) or ev.source != pending[0].source):
            flush_until(pending[0].exchange_time_us)
            apply(pending)
            pending = []
        if isinstance(ev, BookSnapshot):
            flush_until(ev.recv_time_us)
            book.apply_snapshot(ev)
            prev_top = None
        elif isinstance(ev, BookDelta):
            pending.append(ev)
        elif isinstance(ev, Trade):
            flush_until(ev.exchange_time_us)
            trades_since_msg.append(ev)
            if ev.aggressor_side is AggressorSide.BUY:
                interval_trades[0] += float(ev.size)
            else:
                interval_trades[1] += float(ev.size)
            interval_trades[2] += 1
    if pending:
        flush_until(pending[0].exchange_time_us)
        apply(pending)
    # Grid times at or after the last event: state is final.
    while j < n_steps:
        emit(j)
        flows.clear()
        interval_trades[:] = [0.0, 0.0, 0]
        ofi = 0.0
        j += 1
    health["gaps"], health["stale"] = book.gaps, book.stale_messages
    return GridDay(
        grid_times_us=grid,
        tick_size=tick_size,
        half_buckets=H,
        half_levels=L,
        health=health,
        **out,
    )
