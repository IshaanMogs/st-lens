"""Binance parser: documented message formats -> canonical events (offline fixtures only)."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from stlens.ingestion import RawMessage, RawMessageError
from stlens.ingestion.binance import (
    BinanceAdapter,
    BinanceDepthMessage,
    BinanceParseError,
    TimeUnit,
    depth_messages_frame,
)
from stlens.schemas import AggressorSide, BookDelta, BookSnapshot, Side, Trade
from stlens.utils.timestamps import is_plausible_us

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "binance"
RECV_US = 1_790_856_000_200_000
SNAPSHOT_STREAM = "GET /api/v3/depth?symbol=BTCUSDT&limit=5000"

# depth_update_us.json: b = [67012.34 -> 0.5, 67012.33 -> 0], a = [67012.35 -> 1.25, 67015 -> 3]
EXPECTED_LEVELS = [
    (Side.BID, Decimal("67012.34000000"), Decimal("0.50000000")),
    (Side.BID, Decimal("67012.33000000"), Decimal("0.00000000")),
    (Side.ASK, Decimal("67012.35000000"), Decimal("1.25000000")),
    (Side.ASK, Decimal("67015.00000000"), Decimal("3.00000000")),
]


def payload(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8").rstrip("\n")


def raw(text: str, stream: str = "btcusdt@depth@100ms", seq: int = 7) -> RawMessage:
    return RawMessage("binance", stream, text, RECV_US, "session-A", seq)


@pytest.fixture
def adapter() -> BinanceAdapter:
    return BinanceAdapter("BTCUSDT")


def parse_depth(adapter: BinanceAdapter, name: str, **kw):
    parsed = adapter.parse_stream_message(raw(payload(name), **kw))
    assert isinstance(parsed.depth_message, BinanceDepthMessage)
    assert all(isinstance(e, BookDelta) for e in parsed.events)
    return parsed


# ------------------------------------------------------------------ depth: envelope + deltas


def test_depth_message_produces_one_book_delta_per_level(adapter):
    parsed = parse_depth(adapter, "depth_update_us.json")
    assert len(parsed.events) == 4
    assert [(d.side, d.price, d.new_size) for d in parsed.events] == EXPECTED_LEVELS
    assert parsed.unknown_fields == ()


def test_every_book_delta_keeps_message_update_ids(adapter):
    parsed = parse_depth(adapter, "depth_update_us.json")
    assert {(d.first_update_id, d.final_update_id) for d in parsed.events} == {
        (75100000001, 75100000004)
    }
    assert {d.exchange_time_us for d in parsed.events} == {1790856000123456}
    assert {d.recv_time_us for d in parsed.events} == {RECV_US}
    assert {d.symbol for d in parsed.events} == {"BTCUSDT"}


def test_book_delta_order_matches_message_order(adapter):
    parsed = parse_depth(adapter, "depth_update_us.json")
    assert [d.level_index for d in parsed.events] == [0, 1, 2, 3]
    assert {d.level_count for d in parsed.events} == {4}
    # Within each side, order is exactly the order of the b / a arrays; nothing is sorted
    # (the bid prices are descending here and the ask prices ascending, as sent).
    message = parsed.depth_message
    assert [(d.price, d.new_size) for d in parsed.events if d.side is Side.BID] == list(
        message.bids
    )
    assert [(d.price, d.new_size) for d in parsed.events if d.side is Side.ASK] == list(
        message.asks
    )


def test_book_deltas_carry_no_information_from_other_messages(adapter):
    first = parse_depth(adapter, "depth_update_us.json", seq=0)
    second = parse_depth(adapter, "depth_update_combined_us.json", stream="/stream", seq=1)
    # Each delta carries only its own message's ids, times and source; nothing is
    # copied forward or backward between messages.
    assert {d.final_update_id for d in first.events} == {75100000004}
    assert {d.final_update_id for d in second.events} == {75100000005}
    assert {d.exchange_time_us for d in second.events} == {1790856000223456}
    assert {d.source.recv_seq for d in first.events} == {0}
    assert {d.source.recv_seq for d in second.events} == {1}


def test_zero_quantity_level_is_kept_as_removal(adapter):
    parsed = parse_depth(adapter, "depth_update_us.json")
    removed = [d for d in parsed.events if d.new_size == 0]
    assert [(d.side, d.price) for d in removed] == [(Side.BID, Decimal("67012.33000000"))]


def test_empty_depth_message_keeps_envelope_with_zero_deltas(adapter):
    parsed = parse_depth(adapter, "depth_update_empty_us.json")
    assert parsed.events == ()
    message = parsed.depth_message
    assert (message.first_update_id, message.final_update_id) == (75100000006, 75100000006)
    assert message.event_time_us == 1790856000323456
    assert message.level_count == 0
    assert message.raw.payload == payload("depth_update_empty_us.json")


def test_envelope_preserves_whole_message(adapter):
    parsed = parse_depth(adapter, "depth_update_us.json")
    message = parsed.depth_message
    assert message.symbol == "BTCUSDT"
    assert (message.first_update_id, message.final_update_id) == (75100000001, 75100000004)
    assert message.event_time_us == 1790856000123456
    assert [(Side.BID, p, q) for p, q in message.bids] + [
        (Side.ASK, p, q) for p, q in message.asks
    ] == EXPECTED_LEVELS
    assert message.raw is parsed.raw
    assert message.to_book_deltas() == parsed.events


def test_depth_messages_table_keeps_empty_messages(adapter):
    messages = [
        parse_depth(adapter, "depth_update_us.json", seq=0).depth_message,
        parse_depth(adapter, "depth_update_empty_us.json", seq=1).depth_message,
    ]
    frame = depth_messages_frame(messages)
    assert frame["first_update_id"].tolist() == [75100000001, 75100000006]
    assert frame["final_update_id"].tolist() == [75100000004, 75100000006]
    assert frame["level_count"].tolist() == [4, 0]
    assert frame["source_recv_seq"].tolist() == [0, 1]


def test_combined_stream_depth_uses_wrapper_stream_name(adapter):
    parsed = parse_depth(adapter, "depth_update_combined_us.json", stream="/stream")
    assert parsed.depth_message.stream == "btcusdt@depth@100ms"
    assert {d.source.stream for d in parsed.events} == {"btcusdt@depth@100ms"}
    assert [d.side for d in parsed.events] == [Side.ASK]


# ------------------------------------------------------------------ trades


def test_valid_trade_message(adapter):
    parsed = adapter.parse_stream_message(raw(payload("trade_us.json"), stream="btcusdt@trade"))
    assert parsed.depth_message is None
    (trade,) = parsed.events
    assert isinstance(trade, Trade)
    assert trade.trade_id == 5200000001
    assert trade.price == Decimal("67012.34000000")
    assert trade.size == Decimal("0.01200000")
    assert trade.exchange_time_us == 1790856000123456  # T: trade time
    assert trade.exchange_event_time_us == 1790856000123500  # E: event time
    assert parsed.unknown_fields == ()  # "M" is documented ("Ignore")


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("trade_us.json", AggressorSide.SELL),  # m=true: buyer is maker -> seller aggressed
        ("trade_combined_us.json", AggressorSide.BUY),  # m=false: buyer is taker
    ],
)
def test_trade_side_conversion(adapter, fixture, expected):
    (trade,) = adapter.parse_stream_message(raw(payload(fixture), stream="/stream")).events
    assert trade.aggressor_side is expected


# ------------------------------------------------------------------ snapshot / exchangeInfo


def test_valid_depth_snapshot(adapter):
    parsed = adapter.parse_depth_snapshot(
        raw(payload("depth_snapshot.json"), stream=SNAPSHOT_STREAM)
    )
    (snap,) = parsed.events
    assert isinstance(snap, BookSnapshot)
    assert snap.last_update_id == 75100000000
    assert snap.exchange_time_us is None  # REST depth has no timestamp; never filled in
    assert snap.recv_time_us == RECV_US
    assert [lvl.price for lvl in snap.bids] == [Decimal("67012.34"), Decimal("67012.33")]
    assert [lvl.size for lvl in snap.asks] == [Decimal("1.00000000"), Decimal("0.30000000")]


def test_exchange_info_tick_and_step_size(adapter):
    spec = adapter.parse_exchange_info(raw(payload("exchange_info.json"), stream="GET x"))
    assert spec.symbol == "BTCUSDT"
    assert spec.tick_size == Decimal("0.01000000")
    assert spec.step_size == Decimal("0.00001000")
    assert (spec.base_asset, spec.quote_asset, spec.status) == ("BTC", "USDT", "TRADING")
    assert spec.recv_time_us == RECV_US


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda o: o["symbols"].clear(), "found 0"),
        (lambda o: o["symbols"].append(dict(o["symbols"][0])), "found 2"),
        (lambda o: o["symbols"][0]["filters"].pop(0), "missing filter 'PRICE_FILTER'"),
        (lambda o: o["symbols"][0]["filters"][0].update(tickSize="-1"), "plain decimal"),
        (lambda o: o.pop("symbols"), "missing field 'symbols'"),
    ],
)
def test_exchange_info_malformed(adapter, mutate, error):
    obj = json.loads(payload("exchange_info.json"))
    mutate(obj)
    with pytest.raises(BinanceParseError, match=error):
        adapter.parse_exchange_info(raw(json.dumps(obj), stream="GET x"))


def test_exchange_info_disabled_tick_size_kept_as_zero(adapter):
    obj = json.loads(payload("exchange_info.json"))
    obj["symbols"][0]["filters"][0]["tickSize"] = "0.00000000"
    spec = adapter.parse_exchange_info(raw(json.dumps(obj), stream="GET x"))
    assert spec.tick_size == 0


# ------------------------------------------------------------------ timestamps


def test_microsecond_stream_timestamps_used_as_is(adapter):
    parsed = parse_depth(adapter, "depth_update_us.json")
    assert parsed.depth_message.event_time_us == 1790856000123456


def test_millisecond_stream_timestamps_converted_exactly():
    ms_adapter = BinanceAdapter("BTCUSDT", time_unit=TimeUnit.MILLISECOND)
    parsed = parse_depth(ms_adapter, "depth_update_ms.json")
    assert {d.exchange_time_us for d in parsed.events} == {1790856000123000}


def test_wrong_time_unit_fails_instead_of_guessing():
    ms_adapter = BinanceAdapter("BTCUSDT", time_unit=TimeUnit.MILLISECOND)
    with pytest.raises(BinanceParseError, match="plausible"):
        ms_adapter.parse_stream_message(raw(payload("depth_update_us.json")))


def test_exchange_and_receipt_times_are_kept_separate(adapter):
    (delta, *_) = parse_depth(adapter, "depth_update_us.json").events
    assert delta.recv_time_us == RECV_US
    assert delta.exchange_time_us == 1790856000123456


# ------------------------------------------------------------------ numeric conversion


def test_numeric_conversion_is_exact_decimal(adapter):
    (trade,) = adapter.parse_stream_message(
        raw(payload("trade_us.json"), stream="btcusdt@trade")
    ).events
    assert isinstance(trade.price, Decimal)
    assert str(trade.price) == "67012.34000000"
    assert str(trade.size) == "0.01200000"


# ------------------------------------------------------------------ raw preservation


@pytest.mark.parametrize(
    "fixture",
    [
        "depth_update_us.json",
        "depth_update_combined_us.json",
        "trade_us.json",
        "trade_combined_us.json",
    ],
)
def test_raw_payload_and_trace_are_preserved(adapter, fixture):
    text = payload(fixture)
    message = raw(text, stream="/stream", seq=42)
    parsed = adapter.parse_stream_message(message)
    assert parsed.raw is message
    assert parsed.raw.payload == text  # byte-for-byte, not re-serialised
    for event in parsed.events:
        assert event.source.session_id == "session-A"
        assert event.source.recv_seq == 42
        assert event.source.exchange == "binance"


# ------------------------------------------------------------------ unknown fields


def test_unknown_fields_are_tolerated_and_reported(adapter):
    obj = json.loads(payload("trade_us.json"))
    obj["X"] = "NEW"
    obj["b"] = 123  # removed from the trade stream in 2024; must not be silently ignored
    parsed = adapter.parse_stream_message(raw(json.dumps(obj), stream="btcusdt@trade"))
    assert parsed.unknown_fields == ("X", "b")
    assert isinstance(parsed.events[0], Trade)


def test_unknown_combined_wrapper_fields_are_reported(adapter):
    obj = json.loads(payload("depth_update_combined_us.json"))
    obj["extra"] = 1
    parsed = adapter.parse_stream_message(raw(json.dumps(obj), stream="/stream"))
    assert parsed.unknown_fields == ("extra",)


# ------------------------------------------------------------------ malformed messages

MALFORMED = json.loads((FIXTURES / "malformed_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", MALFORMED, ids=[c["name"] for c in MALFORMED])
def test_malformed_message_fails_clearly(adapter, case):
    message = raw(case["payload"])
    parse = (
        adapter.parse_depth_snapshot
        if case["parser"] == "snapshot"
        else adapter.parse_stream_message
    )
    with pytest.raises(BinanceParseError) as excinfo:
        parse(message)
    assert case["error"] in str(excinfo.value)


def test_batch_quarantines_failures_in_order(adapter):
    good = raw(payload("depth_update_us.json"), seq=0)
    bad = raw('{"e":"depthUpdate"}', seq=1)
    good2 = raw(payload("trade_us.json"), seq=2)
    parsed, quarantined = adapter.parse_stream_batch([good, bad, good2])
    assert [p.raw.recv_seq for p in parsed] == [0, 2]
    assert len(quarantined) == 1
    assert quarantined[0].raw is bad
    assert "missing field" in quarantined[0].reason


def test_batch_does_not_reorder_messages(adapter):
    later = raw(payload("depth_update_combined_us.json"), stream="/stream", seq=0)
    earlier = raw(payload("depth_update_us.json"), seq=1)
    parsed, _ = adapter.parse_stream_batch([later, earlier])
    assert [p.raw.recv_seq for p in parsed] == [0, 1]


def test_raw_from_other_exchange_rejected(adapter):
    message = RawMessage("other", "x", payload("trade_us.json"), RECV_US, "s", 0)
    with pytest.raises(BinanceParseError, match="not 'binance'"):
        adapter.parse_stream_message(message)


def test_adapter_requires_upper_case_symbol():
    with pytest.raises(ValueError, match="upper-case"):
        BinanceAdapter("btcusdt")


# ------------------------------------------------------------------ raw envelope


def test_raw_message_envelope_validation():
    with pytest.raises(RawMessageError, match="payload must be str"):
        RawMessage("binance", "s", b"{}", RECV_US, "session", 0)  # type: ignore[arg-type]
    with pytest.raises(RawMessageError, match="recv_time_us"):
        RawMessage("binance", "s", "{}", 1_790_856_000_123, "session", 0)
    with pytest.raises(RawMessageError, match="recv_seq"):
        RawMessage("binance", "s", "{}", RECV_US, "session", -1)


def test_raw_message_received_now_stamps_receipt_time():
    message = RawMessage.received_now(
        exchange="binance", stream="btcusdt@trade", payload="{}", session_id="s", recv_seq=0
    )
    assert is_plausible_us(message.recv_time_us)
