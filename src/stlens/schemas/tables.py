"""Pandera schemas for flattened canonical-event tables (spec B.3).

Canonical event objects validate themselves one at a time; these schemas validate
*batches* once events are flattened into DataFrames (the form later stored as Parquet).
They check per-row invariants only. Cross-row properties (sequence continuity, gaps)
belong to the BookBuilder (Phase 2) and are deliberately NOT checked here.

Prices and sizes are float64 in these tables. This is PROVISIONAL (see
docs/canonical_events.md): the exact decimal strings remain in the raw messages
(``source_session_id`` + ``source_recv_seq``), and the final on-disk representation is
not yet decided.
"""

from collections.abc import Iterable

import pandas as pd
import pandera.pandas as pa

from stlens.schemas.events import AggressorSide, BookDelta, Side, Trade
from stlens.utils.timestamps import MAX_PLAUSIBLE_US, MIN_PLAUSIBLE_US

_TIME_CHECK = pa.Check.in_range(MIN_PLAUSIBLE_US, MAX_PLAUSIBLE_US, include_max=False)


def _source_columns() -> dict[str, pa.Column]:
    return {
        "source_exchange": pa.Column(str, pa.Check.str_length(min_value=1)),
        "source_stream": pa.Column(str, pa.Check.str_length(min_value=1)),
        "source_session_id": pa.Column(str, pa.Check.str_length(min_value=1)),
        "source_recv_seq": pa.Column("int64", pa.Check.ge(0)),
    }


TRADES_SCHEMA = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str, pa.Check.str_length(min_value=1)),
        "trade_id": pa.Column("int64", pa.Check.ge(0)),
        "price": pa.Column("float64", pa.Check.gt(0)),
        "size": pa.Column("float64", pa.Check.gt(0)),
        "aggressor_side": pa.Column(str, pa.Check.isin([s.value for s in AggressorSide])),
        "exchange_time_us": pa.Column("int64", _TIME_CHECK),
        "exchange_event_time_us": pa.Column("int64", _TIME_CHECK),
        "recv_time_us": pa.Column("int64", _TIME_CHECK),
        **_source_columns(),
    },
    strict=True,
    ordered=True,
    name="trades",
)

# One row per exchange depth message (keeps messages with zero level changes, which
# produce no BookDelta). Filled by exchange adapters from their message envelopes.
DEPTH_MESSAGES_SCHEMA = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str, pa.Check.str_length(min_value=1)),
        "first_update_id": pa.Column("int64", pa.Check.ge(0)),
        "final_update_id": pa.Column("int64", pa.Check.ge(0)),
        "level_count": pa.Column("int64", pa.Check.ge(0)),
        "exchange_time_us": pa.Column("int64", _TIME_CHECK),
        "recv_time_us": pa.Column("int64", _TIME_CHECK),
        **_source_columns(),
    },
    checks=[
        pa.Check(
            lambda df: df["first_update_id"] <= df["final_update_id"],
            error="first_update_id <= final_update_id",
        )
    ],
    strict=True,
    ordered=True,
    name="depth_messages",
)

# One row per canonical BookDelta (one price-level change); joins to depth_messages on
# the source columns.
DEPTH_LEVELS_SCHEMA = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str, pa.Check.str_length(min_value=1)),
        "side": pa.Column(str, pa.Check.isin([s.value for s in Side])),
        "price": pa.Column("float64", pa.Check.gt(0)),
        "new_size": pa.Column("float64", pa.Check.ge(0)),
        "exchange_time_us": pa.Column("int64", _TIME_CHECK),
        "recv_time_us": pa.Column("int64", _TIME_CHECK),
        "first_update_id": pa.Column("int64", pa.Check.ge(0)),
        "final_update_id": pa.Column("int64", pa.Check.ge(0)),
        "level_index": pa.Column("int64", pa.Check.ge(0)),
        "level_count": pa.Column("int64", pa.Check.ge(1)),
        **_source_columns(),
    },
    checks=[
        pa.Check(
            lambda df: df["first_update_id"] <= df["final_update_id"],
            error="first_update_id <= final_update_id",
        ),
        pa.Check(
            lambda df: df["level_index"] < df["level_count"],
            error="level_index < level_count",
        ),
    ],
    strict=True,
    ordered=True,
    name="depth_levels",
)


def _source(event: Trade | BookDelta) -> dict[str, object]:
    return {
        "source_exchange": event.source.exchange,
        "source_stream": event.source.stream,
        "source_session_id": event.source.session_id,
        "source_recv_seq": event.source.recv_seq,
    }


def validated_frame(rows: list[dict[str, object]], schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Build a DataFrame with the schema's column order and validate it."""
    columns = list(schema.columns)
    frame = pd.DataFrame(rows, columns=columns)
    if not rows:
        frame = frame.astype({name: col.dtype.type for name, col in schema.columns.items()})
    return schema.validate(frame)


def trades_frame(trades: Iterable[Trade]) -> pd.DataFrame:
    """Flatten trades (in the given order) into a validated table."""
    rows = [
        {
            "symbol": t.symbol,
            "trade_id": t.trade_id,
            "price": float(t.price),
            "size": float(t.size),
            "aggressor_side": t.aggressor_side.value,
            "exchange_time_us": t.exchange_time_us,
            "exchange_event_time_us": t.exchange_event_time_us,
            "recv_time_us": t.recv_time_us,
            **_source(t),
        }
        for t in trades
    ]
    return validated_frame(rows, TRADES_SCHEMA)


def depth_levels_frame(deltas: Iterable[BookDelta]) -> pd.DataFrame:
    """One validated row per canonical BookDelta, in the given order."""
    rows = [
        {
            "symbol": d.symbol,
            "side": d.side.value,
            "price": float(d.price),
            "new_size": float(d.new_size),
            "exchange_time_us": d.exchange_time_us,
            "recv_time_us": d.recv_time_us,
            "first_update_id": d.first_update_id,
            "final_update_id": d.final_update_id,
            "level_index": d.level_index,
            "level_count": d.level_count,
            **_source(d),
        }
        for d in deltas
    ]
    return validated_frame(rows, DEPTH_LEVELS_SCHEMA)
