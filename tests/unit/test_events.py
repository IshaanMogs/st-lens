"""Canonical event contracts: invalid events cannot be constructed."""

import dataclasses
from decimal import Decimal

import pytest

from stlens.schemas import (
    AggressorSide,
    BookDelta,
    BookSnapshot,
    EventValidationError,
    PriceLevel,
    Side,
    SourceRef,
    Trade,
)

T0 = 1_790_856_000_123_456
SRC = SourceRef("binance", "btcusdt@trade", "session-1", 0)


def make_trade(**overrides):
    fields = {
        "symbol": "BTCUSDT",
        "trade_id": 1,
        "price": Decimal("67012.34"),
        "size": Decimal("0.012"),
        "aggressor_side": AggressorSide.BUY,
        "exchange_time_us": T0,
        "exchange_event_time_us": T0 + 44,
        "recv_time_us": T0 + 1_000,
        "source": SRC,
    }
    return Trade(**(fields | overrides))


def make_delta(**overrides):
    fields = {
        "symbol": "BTCUSDT",
        "side": Side.BID,
        "price": Decimal("67012.34"),
        "new_size": Decimal("0.5"),
        "exchange_time_us": T0,
        "recv_time_us": T0 + 1_000,
        "first_update_id": 10,
        "final_update_id": 12,
        "level_index": 0,
        "level_count": 2,
        "source": SRC,
    }
    return BookDelta(**(fields | overrides))


def test_valid_events_construct():
    make_trade()
    make_delta()
    BookSnapshot("BTCUSDT", 9, (PriceLevel(Decimal("1"), Decimal("2")),), (), None, T0, SRC)


def test_events_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_trade().price = Decimal("1")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"price": Decimal("0")}, "price must be > 0"),
        ({"price": 67012.34}, "price must be Decimal"),
        ({"size": Decimal("0")}, "size must be > 0"),
        ({"size": Decimal("NaN")}, "size must be finite"),
        ({"trade_id": True}, "trade_id must be int"),
        ({"trade_id": -1}, "trade_id must be >= 0"),
        ({"aggressor_side": "buy"}, "aggressor_side must be AggressorSide"),
        ({"exchange_time_us": 1_790_856_000_123}, "plausible"),
        ({"recv_time_us": None}, "recv_time_us must be int"),
        ({"symbol": ""}, "symbol must be non-empty"),
    ],
)
def test_invalid_trade_rejected(overrides, message):
    with pytest.raises(EventValidationError, match=message):
        make_trade(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"first_update_id": 13}, "first_update_id 13 > final_update_id 12"),
        ({"side": "bid"}, "side must be Side"),
        ({"price": Decimal("0")}, "price must be > 0"),
        ({"new_size": Decimal("-1")}, "new_size must be >= 0"),
        ({"level_index": 2}, "level_index 2 must be < level_count 2"),
        ({"level_count": 0}, "level_count must be >= 1"),
        ({"exchange_time_us": 0}, "plausible"),
    ],
)
def test_invalid_delta_rejected(overrides, message):
    with pytest.raises(EventValidationError, match=message):
        make_delta(**overrides)


def test_delta_zero_size_means_level_removal_and_is_valid():
    assert make_delta(new_size=Decimal("0")).new_size == 0


def test_snapshot_exchange_time_optional_but_validated_when_present():
    with pytest.raises(EventValidationError, match="plausible"):
        BookSnapshot("BTCUSDT", 9, (), (), 5, T0, SRC)


def test_source_ref_requires_identity():
    with pytest.raises(EventValidationError, match="session_id"):
        SourceRef("binance", "btcusdt@trade", "", 0)
