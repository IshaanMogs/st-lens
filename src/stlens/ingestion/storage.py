"""Minimal persistence for raw and quarantined messages (spec B.1, B.3).

Layout (``data/`` is git-ignored)::

    {root}/raw/{exchange}/{symbol}/{YYYY-MM-DD}/{session_id}.jsonl.gz
    {root}/quarantine/{exchange}/{symbol}/{YYYY-MM-DD}/{session_id}.jsonl.gz

* One gzip JSON-lines file per capture session per UTC day; the day comes from the
  message's local receipt time, the only timestamp every raw message has.
* Records are written in the order they are handed to the store; nothing is sorted.
* Files are created exclusively: an existing file is never appended to or overwritten.
* ``payload`` is stored as the exact received text (a JSON string), never re-encoded.

This is deliberately not the research database: canonical-event Parquet tables come
with the normalisation step.
"""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from stlens.ingestion.raw import QuarantinedMessage, RawMessage
from stlens.utils.timestamps import us_to_datetime, wall_clock_us


class StorageError(RuntimeError):
    """A persisted file could not be written or read back faithfully."""


def _raw_record(raw: RawMessage) -> dict[str, Any]:
    return {
        "exchange": raw.exchange,
        "stream": raw.stream,
        "session_id": raw.session_id,
        "recv_seq": raw.recv_seq,
        "recv_time_us": raw.recv_time_us,
        "payload": raw.payload,
    }


def _raw_from_record(record: dict[str, Any]) -> RawMessage:
    return RawMessage(
        exchange=record["exchange"],
        stream=record["stream"],
        payload=record["payload"],
        recv_time_us=record["recv_time_us"],
        session_id=record["session_id"],
        recv_seq=record["recv_seq"],
    )


class _DailySessionWriter:
    """Writes JSON lines to one gzip file per (UTC day, session).

    Every file opened stays open until :meth:`close`, so records of one session whose
    receipt day alternates (e.g. the wall clock stepping back across midnight) still go
    to the right file without reopening it.
    """

    def __init__(self, root: Path, kind: str, exchange: str, symbol: str) -> None:
        self._base = Path(root) / kind / exchange / symbol
        self._handles: dict[tuple[str, str], IO[str]] = {}
        self.paths: list[Path] = []

    def _open(self, day: str, session_id: str) -> IO[str]:
        key = (day, session_id)
        if key not in self._handles:
            path = self._base / day / f"{session_id}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Kept open across writes and closed in close(); a with-block cannot span calls.
                self._handles[key] = gzip.open(path, "xt", encoding="utf-8")  # noqa: SIM115
            except FileExistsError as exc:
                raise StorageError(f"refusing to append to existing file {path}") from exc
            self.paths.append(path)
        return self._handles[key]

    def write_line(self, record: dict[str, Any], recv_time_us: int, session_id: str) -> None:
        day = us_to_datetime(recv_time_us).strftime("%Y-%m-%d")
        self._open(day, session_id).write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        for handle in self._handles.values():
            handle.flush()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "_DailySessionWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class RawMessageStore(_DailySessionWriter):
    """Append-only store of raw messages for one exchange/symbol."""

    def __init__(self, root: Path, exchange: str, symbol: str) -> None:
        super().__init__(root, "raw", exchange, symbol)
        self._exchange = exchange

    def write(self, raw: RawMessage) -> None:
        if raw.exchange != self._exchange:
            raise StorageError(f"store is for {self._exchange!r}, got {raw.exchange!r}")
        self.write_line(_raw_record(raw), raw.recv_time_us, raw.session_id)


class QuarantineStore(_DailySessionWriter):
    """Append-only store of messages that failed validation, with the failure reason."""

    def __init__(self, root: Path, exchange: str, symbol: str) -> None:
        super().__init__(root, "quarantine", exchange, symbol)

    def write(self, item: QuarantinedMessage) -> None:
        record = {
            "reason": item.reason,
            "quarantined_at_us": wall_clock_us(),
            "raw": _raw_record(item.raw),
        }
        self.write_line(record, item.raw.recv_time_us, item.raw.session_id)


def _read_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise StorageError(f"{path}:{line_no}: unreadable record ({exc})") from exc


def read_raw_messages(path: Path) -> Iterator[RawMessage]:
    """Yield raw messages from a raw file in stored order. Corrupt records raise."""
    for line_no, record in _read_lines(path):
        try:
            yield _raw_from_record(record)
        except (KeyError, ValueError) as exc:
            raise StorageError(f"{path}:{line_no}: invalid raw record ({exc})") from exc


def read_quarantined(path: Path) -> Iterator[QuarantinedMessage]:
    """Yield quarantined messages from a quarantine file in stored order."""
    for line_no, record in _read_lines(path):
        try:
            yield QuarantinedMessage(raw=_raw_from_record(record["raw"]), reason=record["reason"])
        except (KeyError, ValueError) as exc:
            raise StorageError(f"{path}:{line_no}: invalid quarantine record ({exc})") from exc
