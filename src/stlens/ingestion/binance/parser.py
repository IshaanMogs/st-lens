"""Binance Spot message parser: raw Binance messages -> canonical events.

Together with the rest of :mod:`stlens.ingestion.binance`, this is the ONLY code that
knows Binance field names. Supported inputs:

* WebSocket diff-depth (``<symbol>@depth`` / ``<symbol>@depth@100ms``)
  -> :class:`BinanceDepthMessage` envelope + one canonical ``BookDelta`` per level change
* WebSocket trades (``<symbol>@trade``) -> ``Trade``
* REST ``GET /api/v3/depth`` -> ``BookSnapshot``
* REST ``GET /api/v3/exchangeInfo`` -> ``InstrumentSpec``

Raw-stream payloads and combined-stream wrappers (``{"stream": ..., "data": ...}``) are
both accepted. Parsing is strict: anything that does not match the documented format
raises :class:`BinanceParseError`; nothing is repaired, reordered or filled in. Fields
outside the documented set are tolerated but reported in
:attr:`ParsedMessage.unknown_fields`, so format changes are observable.

Field semantics follow the Binance Spot API documentation (docs/data_source_binance.md).
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

import pandas as pd

from stlens.ingestion.raw import QuarantinedMessage, RawMessage
from stlens.schemas.events import (
    AggressorSide,
    BookDelta,
    BookSnapshot,
    CanonicalEvent,
    EventValidationError,
    PriceLevel,
    Side,
    SourceRef,
    Trade,
)
from stlens.schemas.instruments import InstrumentSpec
from stlens.schemas.tables import DEPTH_MESSAGES_SCHEMA, validated_frame
from stlens.utils.timestamps import is_plausible_us, ms_to_us

EXCHANGE = "binance"

# Documented payload fields. "M" in trades is documented as "Ignore".
DEPTH_FIELDS = frozenset({"e", "E", "s", "U", "u", "b", "a"})
TRADE_FIELDS = frozenset({"e", "E", "s", "t", "p", "q", "T", "m", "M"})
SNAPSHOT_FIELDS = frozenset({"lastUpdateId", "bids", "asks"})
COMBINED_FIELDS = frozenset({"stream", "data"})

# Plain non-negative decimal string, as Binance sends prices/quantities ("67012.34000000").
# Rejects signs, exponents, whitespace, NaN/Infinity.
_DECIMAL_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")


class TimeUnit(StrEnum):
    """Unit of Binance ``E``/``T`` fields, set by the stream URL (``timeUnit=``)."""

    MILLISECOND = "ms"
    MICROSECOND = "us"


class BinanceParseError(ValueError):
    """A raw message does not match the documented Binance format."""


@dataclass(frozen=True, slots=True)
class BinanceDepthMessage:
    """Exchange-specific envelope for one Binance diff-depth message.

    Keeps the whole message as Binance sent it (``U``, ``u``, ``E``, ``s``, ``b``, ``a``)
    plus the raw message it was parsed from. It exists even when the message carries no
    level changes, so the sequence range is never lost.

    Attributes:
        symbol: ``s``.
        event_time_us: ``E`` converted to UTC microseconds.
        first_update_id: ``U``.
        final_update_id: ``u``.
        bids: ``b`` as ``(price, quantity)`` pairs, in message order.
        asks: ``a`` as ``(price, quantity)`` pairs, in message order.
        stream: Stream name (from the combined-stream wrapper if present).
        raw: The raw message, payload text unchanged.
    """

    symbol: str
    event_time_us: int
    first_update_id: int
    final_update_id: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    stream: str
    raw: RawMessage

    def __post_init__(self) -> None:
        if self.first_update_id > self.final_update_id:
            raise BinanceParseError(
                f"first_update_id (U={self.first_update_id}) > "
                f"final_update_id (u={self.final_update_id})"
            )
        if not is_plausible_us(self.event_time_us):
            raise BinanceParseError(
                f"event time E={self.event_time_us} us is outside the plausible UTC-us range"
            )

    @property
    def level_count(self) -> int:
        return len(self.bids) + len(self.asks)

    @property
    def source(self) -> SourceRef:
        return SourceRef(EXCHANGE, self.stream, self.raw.session_id, self.raw.recv_seq)

    def to_book_deltas(self) -> tuple[BookDelta, ...]:
        """One canonical ``BookDelta`` per level change, all sharing this message's metadata.

        Order: ``b`` entries in message order, then ``a`` entries in message order. Binance
        sends bids and asks as separate arrays, so no cross-side order exists in the message;
        bids-first is a fixed convention, not exchange information.
        """
        changes = [(Side.BID, p, q) for p, q in self.bids] + [
            (Side.ASK, p, q) for p, q in self.asks
        ]
        source = self.source
        try:
            return tuple(
                BookDelta(
                    symbol=self.symbol,
                    side=side,
                    price=price,
                    new_size=size,
                    exchange_time_us=self.event_time_us,
                    recv_time_us=self.raw.recv_time_us,
                    first_update_id=self.first_update_id,
                    final_update_id=self.final_update_id,
                    level_index=index,
                    level_count=len(changes),
                    source=source,
                )
                for index, (side, price, size) in enumerate(changes)
            )
        except EventValidationError as exc:
            raise BinanceParseError(f"canonical validation failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    """A successfully parsed raw message.

    Attributes:
        raw: The raw message.
        events: Canonical events in message order (zero or more; a depth message with no
            level changes yields none).
        depth_message: The Binance envelope for diff-depth messages, else ``None``.
        unknown_fields: Payload fields outside the documented set (sorted).
    """

    raw: RawMessage
    events: tuple[CanonicalEvent, ...]
    depth_message: BinanceDepthMessage | None
    unknown_fields: tuple[str, ...]


# ---------------------------------------------------------------------- JSON helpers


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise BinanceParseError(f"duplicate JSON keys: {duplicates}")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise BinanceParseError(f"non-finite JSON constant {name}")


def _load_json(payload: str) -> dict[str, Any]:
    try:
        obj = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise BinanceParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BinanceParseError(f"payload must be a JSON object, got {type(obj).__name__}")
    return obj


def _get(obj: dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise BinanceParseError(f"missing field {key!r}")
    return obj[key]


def _int_field(obj: dict[str, Any], key: str) -> int:
    value = _get(obj, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BinanceParseError(f"field {key!r} must be an integer, got {value!r}")
    if value < 0:
        raise BinanceParseError(f"field {key!r} must be >= 0, got {value}")
    return value


def _str_field(obj: dict[str, Any], key: str) -> str:
    value = _get(obj, key)
    if not isinstance(value, str) or not value:
        raise BinanceParseError(f"field {key!r} must be a non-empty string, got {value!r}")
    return value


def _decimal(value: Any, what: str) -> Decimal:
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        raise BinanceParseError(f"{what} must be a plain decimal string, got {value!r}")
    return Decimal(value)


def _levels(value: Any, what: str) -> tuple[tuple[Decimal, Decimal], ...]:
    if not isinstance(value, list):
        raise BinanceParseError(f"{what} must be a list, got {type(value).__name__}")
    out = []
    for i, entry in enumerate(value):
        if not isinstance(entry, list) or len(entry) != 2:
            raise BinanceParseError(f"{what}[{i}] must be a [price, quantity] pair, got {entry!r}")
        out.append(
            (_decimal(entry[0], f"{what}[{i}] price"), _decimal(entry[1], f"{what}[{i}] quantity"))
        )
    return tuple(out)


# ---------------------------------------------------------------------- adapter


class BinanceAdapter:
    """Parse raw Binance messages for one symbol into canonical events.

    Args:
        symbol: Exchange symbol, e.g. ``"BTCUSDT"``. Messages for other symbols are rejected.
        time_unit: Unit of ``E``/``T`` on the subscribed streams. Must match the
            ``timeUnit`` URL parameter used when connecting (project default: microseconds).
    """

    def __init__(self, symbol: str, time_unit: TimeUnit = TimeUnit.MICROSECOND) -> None:
        if not symbol or symbol != symbol.upper():
            raise ValueError(f"symbol must be upper-case, e.g. 'BTCUSDT', got {symbol!r}")
        self.symbol = symbol
        self.time_unit = TimeUnit(time_unit)

    def parse_stream_message(self, raw: RawMessage) -> ParsedMessage:
        """Parse a WebSocket diff-depth or trade message (raw or combined-stream form)."""
        self._check_raw(raw)
        obj = _load_json(raw.payload)
        stream = raw.stream
        unknown: list[str] = []
        if "data" in obj and "stream" in obj:
            unknown += sorted(set(obj) - COMBINED_FIELDS)
            stream = obj["stream"]
            if not isinstance(stream, str) or not stream:
                raise BinanceParseError(
                    f"combined-stream 'stream' must be a non-empty string, got {stream!r}"
                )
            obj = obj["data"]
            if not isinstance(obj, dict):
                raise BinanceParseError("combined-stream 'data' must be a JSON object")
        event_type = _get(obj, "e")
        if event_type == "depthUpdate":
            message = self._depth_message(obj, raw, stream)
            unknown += sorted(set(obj) - DEPTH_FIELDS)
            return ParsedMessage(raw, message.to_book_deltas(), message, tuple(unknown))
        if event_type == "trade":
            trade = self._trade(obj, raw, SourceRef(EXCHANGE, stream, raw.session_id, raw.recv_seq))
            unknown += sorted(set(obj) - TRADE_FIELDS)
            return ParsedMessage(raw, (trade,), None, tuple(unknown))
        raise BinanceParseError(f"unsupported event type {event_type!r}")

    def parse_depth_snapshot(self, raw: RawMessage) -> ParsedMessage:
        """Parse a REST ``GET /api/v3/depth`` response for this adapter's symbol.

        The response carries no symbol and no timestamp; the symbol is the one that was
        requested (this adapter's) and ``exchange_time_us`` is left ``None``.
        """
        self._check_raw(raw)
        obj = _load_json(raw.payload)
        source = SourceRef(EXCHANGE, raw.stream, raw.session_id, raw.recv_seq)
        last_update_id = _int_field(obj, "lastUpdateId")
        bids = _levels(_get(obj, "bids"), "bids")
        asks = _levels(_get(obj, "asks"), "asks")
        try:
            snapshot = BookSnapshot(
                symbol=self.symbol,
                last_update_id=last_update_id,
                bids=tuple(PriceLevel(p, q) for p, q in bids),
                asks=tuple(PriceLevel(p, q) for p, q in asks),
                exchange_time_us=None,
                recv_time_us=raw.recv_time_us,
                source=source,
            )
        except EventValidationError as exc:
            raise BinanceParseError(f"canonical validation failed: {exc}") from exc
        return ParsedMessage(raw, (snapshot,), None, tuple(sorted(set(obj) - SNAPSHOT_FIELDS)))

    def parse_exchange_info(self, raw: RawMessage) -> InstrumentSpec:
        """Extract this symbol's tick and lot sizes from a ``GET /api/v3/exchangeInfo`` response.

        Uses the documented ``PRICE_FILTER.tickSize`` and ``LOT_SIZE.stepSize``. A value of
        0 means the filter is disabled and is kept as 0, not replaced.
        """
        self._check_raw(raw)
        obj = _load_json(raw.payload)
        symbols = _get(obj, "symbols")
        if not isinstance(symbols, list):
            raise BinanceParseError("field 'symbols' must be a list")
        matches = [s for s in symbols if isinstance(s, dict) and s.get("symbol") == self.symbol]
        if len(matches) != 1:
            raise BinanceParseError(
                f"expected exactly one entry for {self.symbol!r} in 'symbols', found {len(matches)}"
            )
        entry = matches[0]
        filters = _get(entry, "filters")
        if not isinstance(filters, list):
            raise BinanceParseError("field 'filters' must be a list")
        by_type: dict[str, dict[str, Any]] = {}
        for f in filters:
            if isinstance(f, dict) and isinstance(f.get("filterType"), str):
                if f["filterType"] in by_type:
                    raise BinanceParseError(f"duplicate filter {f['filterType']!r}")
                by_type[f["filterType"]] = f
        for required in ("PRICE_FILTER", "LOT_SIZE"):
            if required not in by_type:
                raise BinanceParseError(f"missing filter {required!r} for {self.symbol!r}")
        try:
            return InstrumentSpec(
                exchange=EXCHANGE,
                symbol=self.symbol,
                status=_str_field(entry, "status"),
                base_asset=_str_field(entry, "baseAsset"),
                quote_asset=_str_field(entry, "quoteAsset"),
                tick_size=_decimal(_get(by_type["PRICE_FILTER"], "tickSize"), "tickSize"),
                step_size=_decimal(_get(by_type["LOT_SIZE"], "stepSize"), "stepSize"),
                recv_time_us=raw.recv_time_us,
                source=SourceRef(EXCHANGE, raw.stream, raw.session_id, raw.recv_seq),
            )
        except EventValidationError as exc:
            raise BinanceParseError(f"instrument validation failed: {exc}") from exc

    def parse_stream_batch(
        self, raws: Iterable[RawMessage]
    ) -> tuple[list[ParsedMessage], list[QuarantinedMessage]]:
        """Parse many stream messages in receipt order; failures are quarantined, not dropped."""
        parsed: list[ParsedMessage] = []
        quarantined: list[QuarantinedMessage] = []
        for raw in raws:
            try:
                parsed.append(self.parse_stream_message(raw))
            except BinanceParseError as exc:
                quarantined.append(QuarantinedMessage(raw=raw, reason=str(exc)))
        return parsed, quarantined

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _check_raw(raw: RawMessage) -> None:
        if raw.exchange != EXCHANGE:
            raise BinanceParseError(f"raw message is from {raw.exchange!r}, not {EXCHANGE!r}")

    def _time_us(self, obj: dict[str, Any], key: str) -> int:
        value = _int_field(obj, key)
        return ms_to_us(value) if self.time_unit is TimeUnit.MILLISECOND else value

    def _symbol(self, obj: dict[str, Any]) -> str:
        symbol = _get(obj, "s")
        if symbol != self.symbol:
            raise BinanceParseError(
                f"symbol {symbol!r} does not match adapter symbol {self.symbol!r}"
            )
        return symbol

    def _depth_message(
        self, obj: dict[str, Any], raw: RawMessage, stream: str
    ) -> BinanceDepthMessage:
        return BinanceDepthMessage(
            symbol=self._symbol(obj),
            event_time_us=self._time_us(obj, "E"),
            first_update_id=_int_field(obj, "U"),
            final_update_id=_int_field(obj, "u"),
            bids=_levels(_get(obj, "b"), "b"),
            asks=_levels(_get(obj, "a"), "a"),
            stream=stream,
            raw=raw,
        )

    def _trade(self, obj: dict[str, Any], raw: RawMessage, source: SourceRef) -> Trade:
        buyer_is_maker = _get(obj, "m")
        if not isinstance(buyer_is_maker, bool):
            raise BinanceParseError(f"field 'm' must be a boolean, got {buyer_is_maker!r}")
        # m = true: the buyer was the resting (maker) order, so the taker/aggressor sold.
        aggressor = AggressorSide.SELL if buyer_is_maker else AggressorSide.BUY
        try:
            return Trade(
                symbol=self._symbol(obj),
                trade_id=_int_field(obj, "t"),
                price=_decimal(_get(obj, "p"), "price 'p'"),
                size=_decimal(_get(obj, "q"), "quantity 'q'"),
                aggressor_side=aggressor,
                exchange_time_us=self._time_us(obj, "T"),
                exchange_event_time_us=self._time_us(obj, "E"),
                recv_time_us=raw.recv_time_us,
                source=source,
            )
        except EventValidationError as exc:
            raise BinanceParseError(f"canonical validation failed: {exc}") from exc


def depth_messages_frame(messages: Iterable[BinanceDepthMessage]) -> pd.DataFrame:
    """One validated row per depth message (``DEPTH_MESSAGES_SCHEMA``), in the given order.

    Keeps messages with zero level changes, which produce no ``BookDelta`` rows.
    """
    rows = [
        {
            "symbol": m.symbol,
            "first_update_id": m.first_update_id,
            "final_update_id": m.final_update_id,
            "level_count": m.level_count,
            "exchange_time_us": m.event_time_us,
            "recv_time_us": m.raw.recv_time_us,
            "source_exchange": EXCHANGE,
            "source_stream": m.stream,
            "source_session_id": m.raw.session_id,
            "source_recv_seq": m.raw.recv_seq,
        }
        for m in messages
    ]
    return validated_frame(rows, DEPTH_MESSAGES_SCHEMA)
