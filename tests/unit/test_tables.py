"""Pandera batch schemas over flattened canonical events."""

from pathlib import Path

import pandera.errors
import pytest

from stlens.ingestion import RawMessage
from stlens.ingestion.binance import BinanceAdapter
from stlens.schemas.tables import (
    DEPTH_LEVELS_SCHEMA,
    TRADES_SCHEMA,
    depth_levels_frame,
    trades_frame,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "binance"
RECV_US = 1_790_856_000_200_000
SCHEMA_ERRORS = (pandera.errors.SchemaError, pandera.errors.SchemaErrors)


def parse(name: str, seq: int):
    text = (FIXTURES / name).read_text(encoding="utf-8").rstrip("\n")
    message = RawMessage("binance", "/stream", text, RECV_US, "session-A", seq)
    return BinanceAdapter("BTCUSDT").parse_stream_message(message).events


def test_depth_levels_table():
    deltas = [*parse("depth_update_us.json", 0), *parse("depth_update_empty_us.json", 1)]
    levels = depth_levels_frame(deltas)
    assert len(levels) == 4  # the empty message contributes no level rows
    assert levels["side"].tolist() == ["bid", "bid", "ask", "ask"]
    assert levels["new_size"].tolist() == [0.5, 0.0, 1.25, 3.0]
    assert levels["level_index"].tolist() == [0, 1, 2, 3]
    assert set(levels["level_count"]) == {4}
    assert set(levels["first_update_id"]) == {75100000001}
    assert set(levels["final_update_id"]) == {75100000004}
    assert set(levels["source_recv_seq"]) == {0}


def test_trades_table():
    frame = trades_frame([*parse("trade_us.json", 0), *parse("trade_combined_us.json", 1)])
    assert frame["trade_id"].tolist() == [5200000001, 5200000002]
    assert frame["aggressor_side"].tolist() == ["sell", "buy"]
    assert frame["exchange_time_us"].dtype == "int64"


def test_empty_tables_validate():
    assert trades_frame([]).empty
    assert depth_levels_frame([]).empty


def test_schema_rejects_bad_batch():
    frame = trades_frame(parse("trade_us.json", 0))
    frame.loc[0, "price"] = -1.0
    with pytest.raises(SCHEMA_ERRORS):
        TRADES_SCHEMA.validate(frame)


def test_schema_rejects_level_index_out_of_range():
    frame = depth_levels_frame(parse("depth_update_us.json", 0))
    frame.loc[0, "level_index"] = 4
    with pytest.raises(SCHEMA_ERRORS):
        DEPTH_LEVELS_SCHEMA.validate(frame)


def test_schema_rejects_unexpected_columns():
    frame = depth_levels_frame(parse("depth_update_us.json", 0))
    frame["extra"] = 1
    with pytest.raises(SCHEMA_ERRORS):
        DEPTH_LEVELS_SCHEMA.validate(frame)
