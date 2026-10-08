"""Raw and quarantine persistence: exact round trip, partitioning, no overwrite."""

import gzip

import pytest

from stlens.ingestion import QuarantinedMessage, RawMessage
from stlens.ingestion.storage import (
    QuarantineStore,
    RawMessageStore,
    StorageError,
    read_quarantined,
    read_raw_messages,
)

DAY1_US = 1_790_856_000_123_456  # 2026-10-01T12:00:00.123456Z
DAY2_US = 1_790_899_200_000_001  # 2026-10-02T00:00:00.000001Z
# Payload with odd whitespace, key order and non-ASCII: must survive byte-for-byte.
PAYLOAD = '{"e":"trade",  "s":"BTCUSDT","note":"é", "p":"67012.34000000"}'


def make_raw(seq: int, recv_us: int = DAY1_US, session: str = "S1") -> RawMessage:
    return RawMessage("binance", "/stream", PAYLOAD, recv_us, session, seq)


def test_raw_round_trip_is_exact_and_ordered(tmp_path):
    messages = [make_raw(0), make_raw(1, DAY1_US + 5), make_raw(2, DAY1_US + 3)]
    with RawMessageStore(tmp_path, "binance", "BTCUSDT") as store:
        for m in messages:
            store.write(m)
    (path,) = store.paths
    assert path == tmp_path / "raw/binance/BTCUSDT/2026-10-01/S1.jsonl.gz"
    # Stored in write order (not sorted by receipt time), every field identical.
    assert list(read_raw_messages(path)) == messages


def test_raw_store_partitions_by_utc_receipt_day_and_session(tmp_path):
    with RawMessageStore(tmp_path, "binance", "BTCUSDT") as store:
        store.write(make_raw(0, DAY1_US, "S1"))
        store.write(make_raw(1, DAY2_US, "S1"))
        store.write(make_raw(0, DAY2_US, "S2"))
    rel = [p.relative_to(tmp_path).as_posix() for p in store.paths]
    assert rel == [
        "raw/binance/BTCUSDT/2026-10-01/S1.jsonl.gz",
        "raw/binance/BTCUSDT/2026-10-02/S1.jsonl.gz",
        "raw/binance/BTCUSDT/2026-10-02/S2.jsonl.gz",
    ]


def test_raw_store_never_overwrites_or_appends(tmp_path):
    with RawMessageStore(tmp_path, "binance", "BTCUSDT") as store:
        store.write(make_raw(0))
    with (
        RawMessageStore(tmp_path, "binance", "BTCUSDT") as again,
        pytest.raises(StorageError, match="refusing to append"),
    ):
        again.write(make_raw(1))


def test_raw_store_rejects_other_exchange(tmp_path):
    with (
        RawMessageStore(tmp_path, "binance", "BTCUSDT") as store,
        pytest.raises(StorageError, match="store is for"),
    ):
        store.write(RawMessage("other", "/s", "{}", DAY1_US, "S1", 0))


def test_corrupt_raw_file_fails_loudly(tmp_path):
    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"exchange": "binance", "stream": "/s"\n')
    with pytest.raises(StorageError, match=r"bad.jsonl.gz:1"):
        list(read_raw_messages(path))


def test_raw_record_missing_field_fails_loudly(tmp_path):
    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"exchange": "binance"}\n')
    with pytest.raises(StorageError, match="invalid raw record"):
        list(read_raw_messages(path))


def test_quarantine_round_trip_keeps_raw_and_reason(tmp_path):
    item = QuarantinedMessage(raw=make_raw(3), reason="missing field 'U'")
    with QuarantineStore(tmp_path, "binance", "BTCUSDT") as store:
        store.write(item)
    (path,) = store.paths
    assert path == tmp_path / "quarantine/binance/BTCUSDT/2026-10-01/S1.jsonl.gz"
    assert list(read_quarantined(path)) == [item]


def test_session_alternating_between_days_keeps_files_open(tmp_path):
    # E.g. a wall-clock step backwards across midnight: day 2, then day 1, then day 2.
    with RawMessageStore(tmp_path, "binance", "BTCUSDT") as store:
        store.write(make_raw(0, DAY2_US))
        store.write(make_raw(1, DAY1_US))
        store.write(make_raw(2, DAY2_US))
    day2, day1 = store.paths
    assert [m.recv_seq for m in read_raw_messages(day2)] == [0, 2]
    assert [m.recv_seq for m in read_raw_messages(day1)] == [1]
