"""Canonical, exchange-agnostic market-data events (the stable Phase 1 contract).

Downstream code (BookBuilder, features, replay) depends only on these types, never on
exchange message formats. Every event is immutable and validates its own invariants on
construction, so an invalid event cannot exist.

Conventions (see docs/canonical_events.md):

* Timestamps are ``int`` UTC microseconds. ``exchange_time_us`` is the exchange's clock,
  ``recv_time_us`` is the local receipt clock; they are never substituted for each other.
* Prices and sizes are :class:`decimal.Decimal`, parsed exactly from exchange strings.
* ``source`` links each event back to the raw message it was parsed from.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from stlens.utils.timestamps import is_plausible_us


class EventValidationError(ValueError):
    """A canonical event violates its contract."""


class Side(StrEnum):
    """Book side of a resting price level."""

    BID = "bid"
    ASK = "ask"


class AggressorSide(StrEnum):
    """Side of the incoming (taker) order that caused a trade."""

    BUY = "buy"
    SELL = "sell"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EventValidationError(message)


def _check_int(name: str, value: object, *, minimum: int = 0) -> None:
    # bool is a subclass of int; reject it explicitly.
    if not isinstance(value, int) or isinstance(value, bool):
        raise EventValidationError(f"{name} must be int, got {type(value).__name__}")
    _require(value >= minimum, f"{name} must be >= {minimum}, got {value}")


def _check_time(name: str, value: object) -> None:
    _check_int(name, value)
    assert isinstance(value, int)
    _require(is_plausible_us(value), f"{name}={value} is outside the plausible UTC-us range")


def _check_decimal(name: str, value: object, *, positive: bool) -> None:
    if not isinstance(value, Decimal):
        raise EventValidationError(f"{name} must be Decimal, got {type(value).__name__}")
    _require(value.is_finite(), f"{name} must be finite, got {value}")
    if positive:
        _require(value > 0, f"{name} must be > 0, got {value}")
    else:
        _require(value >= 0, f"{name} must be >= 0, got {value}")


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Pointer from a canonical event to the raw message it came from.

    ``session_id`` + ``recv_seq`` identify the raw record; they are assigned locally by
    the capture process and are NOT exchange identifiers.
    """

    exchange: str
    stream: str
    session_id: str
    recv_seq: int

    def __post_init__(self) -> None:
        _require(bool(self.exchange), "exchange must be non-empty")
        _require(bool(self.stream), "stream must be non-empty")
        _require(bool(self.session_id), "session_id must be non-empty")
        _check_int("recv_seq", self.recv_seq)


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """A price and the total resting size at that price (aggregated L2)."""

    price: Decimal
    size: Decimal

    def __post_init__(self) -> None:
        _check_decimal("price", self.price, positive=True)
        _check_decimal("size", self.size, positive=False)


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Full (depth-limited) book state at exchange update id ``last_update_id``.

    ``exchange_time_us`` is ``None`` when the exchange does not timestamp the snapshot
    (Binance REST depth does not); it is never filled from ``recv_time_us``.
    Levels are kept in the order the exchange sent them.
    """

    symbol: str
    last_update_id: int
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    exchange_time_us: int | None
    recv_time_us: int
    source: SourceRef

    def __post_init__(self) -> None:
        _require(bool(self.symbol), "symbol must be non-empty")
        _check_int("last_update_id", self.last_update_id)
        if self.exchange_time_us is not None:
            _check_time("exchange_time_us", self.exchange_time_us)
        _check_time("recv_time_us", self.recv_time_us)
        for name, levels in (("bids", self.bids), ("asks", self.asks)):
            _require(isinstance(levels, tuple), f"{name} must be a tuple")
            for level in levels:
                _require(isinstance(level, PriceLevel), f"{name} must contain PriceLevel")


@dataclass(frozen=True, slots=True)
class BookDelta:
    """New absolute size at one price level (spec B.2: ``BookDelta(side, price, new_size)``).

    ``new_size == 0`` removes the level. One exchange depth message with N level changes
    yields N ``BookDelta`` events that share the message metadata:

    * ``first_update_id``/``final_update_id``: the exchange sequence range of the whole
      message (Binance ``U``/``u``), carried verbatim for gap detection by the future
      BookBuilder. Nothing here checks continuity between messages.
    * ``level_index``/``level_count``: this change's position within the message and the
      number of changes in it, in exchange order, so the message can be reassembled and
      its completeness checked without reordering.

    A message with zero level changes produces no ``BookDelta``; its sequence range is
    retained in the exchange-specific message envelope (see ``stlens.ingestion``).
    """

    symbol: str
    side: Side
    price: Decimal
    new_size: Decimal
    exchange_time_us: int
    recv_time_us: int
    first_update_id: int
    final_update_id: int
    level_index: int
    level_count: int
    source: SourceRef

    def __post_init__(self) -> None:
        _require(bool(self.symbol), "symbol must be non-empty")
        _require(isinstance(self.side, Side), f"side must be Side, got {self.side!r}")
        _check_decimal("price", self.price, positive=True)
        _check_decimal("new_size", self.new_size, positive=False)
        _check_time("exchange_time_us", self.exchange_time_us)
        _check_time("recv_time_us", self.recv_time_us)
        _check_int("first_update_id", self.first_update_id)
        _check_int("final_update_id", self.final_update_id)
        _require(
            self.first_update_id <= self.final_update_id,
            f"first_update_id {self.first_update_id} > final_update_id {self.final_update_id}",
        )
        _check_int("level_count", self.level_count, minimum=1)
        _check_int("level_index", self.level_index)
        _require(
            self.level_index < self.level_count,
            f"level_index {self.level_index} must be < level_count {self.level_count}",
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """One exchange trade print.

    ``exchange_time_us`` is the trade (match) time; ``exchange_event_time_us`` is when
    the exchange emitted the message. No link to depth updates exists or is implied.
    """

    symbol: str
    trade_id: int
    price: Decimal
    size: Decimal
    aggressor_side: AggressorSide
    exchange_time_us: int
    exchange_event_time_us: int
    recv_time_us: int
    source: SourceRef

    def __post_init__(self) -> None:
        _require(bool(self.symbol), "symbol must be non-empty")
        _check_int("trade_id", self.trade_id)
        _check_decimal("price", self.price, positive=True)
        _check_decimal("size", self.size, positive=True)
        _require(
            isinstance(self.aggressor_side, AggressorSide),
            f"aggressor_side must be AggressorSide, got {self.aggressor_side!r}",
        )
        _check_time("exchange_time_us", self.exchange_time_us)
        _check_time("exchange_event_time_us", self.exchange_event_time_us)
        _check_time("recv_time_us", self.recv_time_us)


CanonicalEvent = BookSnapshot | BookDelta | Trade
