"""Synthetic limit-order-book source emitting canonical events (fast-track prototype).

WHY THIS EXISTS: live Binance recording is not yet approved (terms/jurisdiction review
pending), so the end-to-end prototype runs on a simulated market. Everything produced
here is SYNTHETIC and labelled ``exchange="synthetic"``. Results obtained on it do not
transfer to real markets without re-running on recorded data.

Model: a zero-intelligence order flow in the spirit of Smith-Farmer-type models
(Poisson limit orders placed at a geometric distance from the opposite best price,
random partial cancellations, Poisson market orders with heavy-tailed sizes). Background
traders do not react to anything, in particular not to spoof orders. This is a stated
limitation: there is no behavioural price reaction to spoofing in this market.

Output mirrors the Binance diff-depth design: every ``step_ms`` the changed levels are
emitted as one depth message (a group of ``BookDelta`` with shared, gap-free update ids
and the message time), trades are emitted individually with times inside the step
(always at or before the message time), and a ``BookSnapshot`` opens the day.

Injected orders (Phase 4) are executed by the same simulator as real participants'
orders: they occupy queue space (behind existing volume), can be partially or fully
executed by incoming market orders, and their cancellations and payoff trades show up
only through the ordinary canonical events. The episode log is the ground truth.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

import numpy as np

from stlens.schemas.events import (
    AggressorSide,
    BookDelta,
    BookSnapshot,
    PriceLevel,
    Side,
    SourceRef,
    Trade,
)

EXCHANGE = "synthetic"
SIZE_DECIMALS = 5  # lot step 0.00001, like BTCUSDT


@dataclass(frozen=True)
class MarketParams:
    """Background order-flow parameters (per 100 ms step unless stated)."""

    symbol: str = "SYNUSDT"
    tick_size: Decimal = Decimal("0.01")
    start_price_ticks: int = 6_701_234
    step_ms: int = 100
    limit_rate: float = 5.0
    cancel_rate: float = 4.0
    market_rate: float = 0.5
    distance_p: float = 0.3  # geometric parameter for limit distance from opposite best
    max_distance: int = 40  # levels further than this from the touch are pruned
    size_median: float = 0.08
    size_sigma: float = 1.1  # lognormal sigma -> heavy right tail ("large orders")
    market_size_mult: float = 1.5
    initial_levels: int = 25


class EpisodeKind(StrEnum):
    """Injected episode types. Only SPOOF episodes can become positive labels."""

    SPOOF = "spoof"  # away from touch, payoff on the opposite side, cancelled unfilled
    HN_CANCEL_NO_PAYOFF = "hn_cancel_no_payoff"  # same shape, no opposite-side activity
    HN_EXECUTED = "hn_executed"  # large order at the touch that gets executed


@dataclass(frozen=True)
class InjectionSpec:
    """One injected episode, decided by the injector before simulation (Phase 4)."""

    episode_id: str
    kind: EpisodeKind
    side: Side
    place_step: int
    rest_steps: int  # steps between placement and cancel / execution
    distance: int  # ticks from the own-side best price at placement (0 = at touch)
    layers: int  # consecutive price levels used (1 = single order)
    size: float  # total size across layers
    payoff_size: float = 0.0
    payoff_delay_steps: int = 0


@dataclass
class EpisodeRecord:
    """What actually happened to an injected episode (ground truth log)."""

    spec: InjectionSpec
    placed: bool = False  # False if the side was empty at placement time (logged, never positive)
    prices_ticks: list[int] = field(default_factory=list)
    place_time_us: int = 0
    end_time_us: int = 0  # cancel time, or execution-completion time
    executed_size: float = 0.0
    cancelled_size: float = 0.0
    payoff_trade_times_us: list[int] = field(default_factory=list)

    @property
    def is_positive(self) -> bool:
        """Ground-truth positive: a placed spoof cancelled without any execution."""
        return self.placed and self.spec.kind is EpisodeKind.SPOOF and self.executed_size == 0.0


@dataclass
class SimulatedDay:
    """One simulated session: canonical events in emission order plus the episode log."""

    day_index: int
    start_time_us: int
    params: MarketParams
    events: list[BookSnapshot | BookDelta | Trade]
    episodes: list[EpisodeRecord]
    background_large_orders: list[tuple[int, Side, int, float]]  # (time_us, side, dist, size)


def _q(size: float) -> Decimal:
    return Decimal(f"{size:.{SIZE_DECIMALS}f}")


class _Book:
    """Simulator-internal book: total size and injected ('own') size per price level."""

    def __init__(self) -> None:
        self.levels: dict[Side, dict[int, float]] = {Side.BID: {}, Side.ASK: {}}
        self.own: dict[Side, dict[int, float]] = {Side.BID: {}, Side.ASK: {}}

    def best(self, side: Side) -> int | None:
        lv = self.levels[side]
        if not lv:
            return None
        return max(lv) if side is Side.BID else min(lv)


class SyntheticMarket:
    """Generate one day of canonical events. Deterministic given the seeds."""

    def __init__(self, params: MarketParams, seed: int, injection_seed: int = 0) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        # Separate stream for injection-side randomness (payoff timing jitter etc.).
        self.inj_rng = np.random.default_rng(injection_seed)

    # ------------------------------------------------------------------ helpers

    def _size(self, mult: float = 1.0) -> float:
        s = float(self.rng.lognormal(np.log(self.p.size_median * mult), self.p.size_sigma))
        return max(round(s, SIZE_DECIMALS), 10**-SIZE_DECIMALS)

    def _set(self, book: _Book, side: Side, price: int, size: float, changed: set) -> None:
        size = round(size, SIZE_DECIMALS)
        if size <= 0:
            book.levels[side].pop(price, None)
            book.own[side].pop(price, None)
        else:
            book.levels[side][price] = size
        changed.add((side, price))

    def _consume(
        self,
        book: _Book,
        aggressor: AggressorSide,
        size: float,
        t_us: int,
        changed: set,
        trades: list,
        episodes_by_price: dict,
    ) -> list[tuple[int, float]]:
        """Market order: walk the opposite side. Own (injected) volume queues last."""
        side = Side.ASK if aggressor is AggressorSide.BUY else Side.BID
        fills: list[tuple[int, float]] = []
        remaining = size
        while remaining > 1e-12:
            best = book.best(side)
            if best is None:
                break
            level = book.levels[side][best]
            take = min(level, remaining)
            take = round(take, SIZE_DECIMALS)
            if take <= 0:
                break
            own = book.own[side].get(best, 0.0)
            new_level = round(level - take, SIZE_DECIMALS)
            if own > 0 and new_level < own:
                executed_own = round(own - max(new_level, 0.0), SIZE_DECIMALS)
                book.own[side][best] = max(new_level, 0.0)
                rec = episodes_by_price.get((side, best))
                if rec is not None:
                    rec.executed_size = round(rec.executed_size + executed_own, SIZE_DECIMALS)
            self._set(book, side, best, new_level, changed)
            trades.append((t_us, best, take, aggressor))
            fills.append((best, take))
            remaining = round(remaining - take, SIZE_DECIMALS)
        return fills

    # ------------------------------------------------------------------ generation

    def simulate(
        self,
        n_steps: int,
        start_time_us: int,
        day_index: int = 0,
        injections: list[InjectionSpec] | None = None,
    ) -> SimulatedDay:
        p = self.p
        step_us = p.step_ms * 1_000
        book = _Book()
        mid = p.start_price_ticks
        for d in range(1, p.initial_levels + 1):
            book.levels[Side.BID][mid - d] = self._size()
            book.levels[Side.ASK][mid + d] = self._size()

        session = f"synthetic-day{day_index:03d}"
        stream_depth, stream_trade = "synthetic@depth", "synthetic@trade"
        events: list[BookSnapshot | BookDelta | Trade] = []
        recv_seq = 0
        update_id = 1_000_000

        def src(stream: str) -> SourceRef:
            nonlocal recv_seq
            ref = SourceRef(EXCHANGE, stream, session, recv_seq)
            recv_seq += 1
            return ref

        events.append(
            BookSnapshot(
                symbol=p.symbol,
                last_update_id=update_id,
                bids=tuple(
                    PriceLevel(Decimal(px) * p.tick_size, _q(sz))
                    for px, sz in sorted(book.levels[Side.BID].items(), reverse=True)
                ),
                asks=tuple(
                    PriceLevel(Decimal(px) * p.tick_size, _q(sz))
                    for px, sz in sorted(book.levels[Side.ASK].items())
                ),
                exchange_time_us=start_time_us,
                recv_time_us=start_time_us,
                source=src("synthetic@snapshot"),
            )
        )

        # Injection bookkeeping.
        by_step: dict[int, list[tuple[str, InjectionSpec]]] = {}
        for spec in injections or []:
            by_step.setdefault(spec.place_step, []).append(("place", spec))
            if spec.kind is EpisodeKind.SPOOF and spec.payoff_size > 0:
                by_step.setdefault(spec.place_step + spec.payoff_delay_steps, []).append(
                    ("payoff", spec)
                )
            by_step.setdefault(spec.place_step + spec.rest_steps, []).append(("end", spec))
        records: dict[str, EpisodeRecord] = {}
        episodes_by_price: dict[tuple[Side, int], EpisodeRecord] = {}
        large_bg: list[tuple[int, Side, int, float]] = []
        large_threshold = float(np.exp(np.log(p.size_median) + 1.645 * p.size_sigma))

        for k in range(n_steps):
            t0 = start_time_us + k * step_us
            t_end = t0 + step_us
            changed: set[tuple[Side, int]] = set()
            trades: list[tuple[int, int, float, AggressorSide]] = []
            actions = (
                ["L"] * int(self.rng.poisson(p.limit_rate))
                + ["C"] * int(self.rng.poisson(p.cancel_rate))
                + ["M"] * int(self.rng.poisson(p.market_rate))
            )
            inj = by_step.get(k, [])
            actions += [f"I{i}" for i in range(len(inj))]
            order = self.rng.permutation(len(actions))
            times = np.sort(self.rng.integers(t0 + 1, t_end + 1, size=len(actions)))

            for slot, idx in enumerate(order):
                a = actions[idx]
                t_us = int(times[slot])
                if a == "L":
                    side = Side.BID if self.rng.random() < 0.5 else Side.ASK
                    opp = book.best(Side.ASK if side is Side.BID else Side.BID)
                    ref = opp if opp is not None else book.best(side)
                    if ref is None:
                        ref = mid
                    dist = int(self.rng.geometric(p.distance_p))
                    price = ref - dist if side is Side.BID else ref + dist
                    if opp is not None and (
                        (side is Side.BID and price >= opp) or (side is Side.ASK and price <= opp)
                    ):
                        continue
                    size = self._size()
                    own_best = book.best(side)
                    if size >= large_threshold and own_best is not None:
                        own_dist = own_best - price if side is Side.BID else price - own_best
                        large_bg.append((t_us, side, max(own_dist, 0), size))
                    self._set(book, side, price, book.levels[side].get(price, 0.0) + size, changed)
                elif a == "C":
                    side = Side.BID if self.rng.random() < 0.5 else Side.ASK
                    lv = book.levels[side]
                    if not lv:
                        continue
                    prices = list(lv)
                    price = prices[int(self.rng.integers(len(prices)))]
                    own = book.own[side].get(price, 0.0)
                    background = round(lv[price] - own, SIZE_DECIMALS)
                    if background <= 0:
                        continue
                    frac = float(self.rng.uniform(0.2, 1.0))
                    cut = background if frac > 0.8 else round(background * frac, SIZE_DECIMALS)
                    self._set(book, side, price, lv[price] - cut, changed)
                elif a == "M":
                    aggressor = AggressorSide.BUY if self.rng.random() < 0.5 else AggressorSide.SELL
                    self._consume(
                        book,
                        aggressor,
                        self._size(p.market_size_mult),
                        t_us,
                        changed,
                        trades,
                        episodes_by_price,
                    )
                else:
                    what, spec = inj[int(a[1:])]
                    self._inject(
                        what, spec, book, t_us, changed, trades, records, episodes_by_price
                    )

            # Keep both sides populated and prune far levels (emitted as removals).
            for side in (Side.BID, Side.ASK):
                if not book.levels[side]:
                    opp = book.best(Side.ASK if side is Side.BID else Side.BID) or mid
                    price = opp - 1 if side is Side.BID else opp + 1
                    self._set(book, side, price, self._size(), changed)
            bb, ba = book.best(Side.BID), book.best(Side.ASK)
            mid = (bb + ba) // 2 if bb is not None and ba is not None else mid
            for side in (Side.BID, Side.ASK):
                for price in list(book.levels[side]):
                    if abs(price - mid) > p.max_distance and book.own[side].get(price, 0.0) == 0:
                        self._set(book, side, price, 0.0, changed)

            for t_us, price, size, aggressor in trades:
                events.append(
                    Trade(
                        symbol=p.symbol,
                        trade_id=len(events),
                        price=Decimal(price) * p.tick_size,
                        size=_q(size),
                        aggressor_side=aggressor,
                        exchange_time_us=t_us,
                        exchange_event_time_us=t_us,
                        recv_time_us=t_us,
                        source=src(stream_trade),
                    )
                )
            if changed:
                ordered = sorted(changed, key=lambda sp: (sp[0] is Side.ASK, sp[1]))
                first = update_id + 1
                update_id += len(ordered)
                source = src(stream_depth)
                for i, (side, price) in enumerate(ordered):
                    events.append(
                        BookDelta(
                            symbol=p.symbol,
                            side=side,
                            price=Decimal(price) * p.tick_size,
                            new_size=_q(book.levels[side].get(price, 0.0)),
                            exchange_time_us=t_end,
                            recv_time_us=t_end,
                            first_update_id=first,
                            final_update_id=update_id,
                            level_index=i,
                            level_count=len(ordered),
                            source=source,
                        )
                    )

        return SimulatedDay(
            day_index=day_index,
            start_time_us=start_time_us,
            params=p,
            events=events,
            # Every scheduled episode is logged, placed or not, in schedule order.
            episodes=[records.get(s.episode_id, EpisodeRecord(spec=s)) for s in injections or []],
            background_large_orders=large_bg,
        )

    def _inject(
        self,
        what: str,
        spec: InjectionSpec,
        book: _Book,
        t_us: int,
        changed: set,
        trades: list,
        records: dict[str, EpisodeRecord],
        episodes_by_price: dict,
    ) -> None:
        side = spec.side
        if what == "place":
            own_best = book.best(side)
            if own_best is None:
                records[spec.episode_id] = EpisodeRecord(spec=spec, placed=False)
                return
            rec = EpisodeRecord(spec=spec, placed=True, place_time_us=t_us)
            per_layer = round(spec.size / spec.layers, SIZE_DECIMALS)
            for j in range(spec.layers):
                d = spec.distance + j
                price = own_best - d if side is Side.BID else own_best + d
                book.own[side][price] = round(
                    book.own[side].get(price, 0.0) + per_layer, SIZE_DECIMALS
                )
                self._set(book, side, price, book.levels[side].get(price, 0.0) + per_layer, changed)
                rec.prices_ticks.append(price)
                episodes_by_price[(side, price)] = rec
            records[spec.episode_id] = rec
            return
        rec = records.get(spec.episode_id)
        if rec is None or not rec.placed:
            return
        if what == "payoff":
            # Opposite-side activity the spoof is meant to induce: aggressive orders that
            # lift the opposite touch (bid spoof -> buyers lift asks, and vice versa).
            aggressor = AggressorSide.BUY if side is Side.BID else AggressorSide.SELL
            before = len(trades)
            self._consume(
                book, aggressor, spec.payoff_size, t_us, changed, trades, episodes_by_price
            )
            rec.payoff_trade_times_us.extend(t for t, *_ in trades[before:])
            return
        # what == "end"
        if spec.kind is EpisodeKind.HN_EXECUTED:
            # Aggressive order on the same side as the injected order's counterparty,
            # large enough to execute everything resting down to and including it.
            aggressor = AggressorSide.SELL if side is Side.BID else AggressorSide.BUY
            for price in rec.prices_ticks:
                while book.own[side].get(price, 0.0) > 0 and price in book.levels[side]:
                    best = book.best(side)
                    if best is None:
                        break
                    self._consume(
                        book,
                        aggressor,
                        book.levels[side][best],
                        t_us,
                        changed,
                        trades,
                        episodes_by_price,
                    )
        # Cancel whatever injected volume is left (for HN_EXECUTED normally nothing).
        for price in rec.prices_ticks:
            own = book.own[side].pop(price, 0.0)
            if own > 0:
                rec.cancelled_size = round(rec.cancelled_size + own, SIZE_DECIMALS)
                self._set(book, side, price, book.levels[side].get(price, 0.0) - own, changed)
            episodes_by_price.pop((side, price), None)
        rec.end_time_us = t_us


def iter_days(
    params: MarketParams,
    n_days: int,
    n_steps: int,
    base_seed: int,
    injections_by_day: dict[int, list[InjectionSpec]] | None = None,
    injection_seed_by_day: dict[int, int] | None = None,
    start_time_us: int = 1_790_812_800_000_000,  # 2026-10-01T00:00:00Z
) -> Iterator[SimulatedDay]:
    """Simulate ``n_days`` independent days (independent background seeds)."""
    day_us = 86_400_000_000
    for d in range(n_days):
        market = SyntheticMarket(
            params,
            seed=base_seed * 1_000 + d,
            injection_seed=(injection_seed_by_day or {}).get(d, 0),
        )
        yield market.simulate(
            n_steps,
            start_time_us + d * day_us,
            day_index=d,
            injections=(injections_by_day or {}).get(d, []),
        )
