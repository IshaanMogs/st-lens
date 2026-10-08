"""Binance raw-capture recorder (spec B.1). Stores messages exactly as received.

One *session* is one WebSocket connection lifetime. Within a session the recorder:

1. connects to the combined diff-depth + trade stream;
2. fetches ``exchangeInfo`` once (tick size; failure is logged, not fatal);
3. after the first stream message is stored, fetches a REST depth snapshot, following
   the documented order (buffer stream events first, then snapshot). If it fails, the
   session ends: without a snapshot the stream cannot be used to build a book;
4. optionally re-fetches a snapshot every ``snapshot_interval_s``;
5. writes every message (stream and REST) to the raw store with a gap-free local
   ``recv_seq`` in receipt order.

It never parses, filters, reorders or repairs; normalisation is a separate step. Book
synchronisation and gap detection belong to the BookBuilder (Phase 2).

THIS MODULE IS NOT TO BE RUN AGAINST BINANCE until the Binance Terms of Use and
jurisdiction review in docs/data_source_binance.md is complete.
"""

import asyncio
import contextlib
import itertools
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Literal

import httpx

from stlens.ingestion.binance.parser import EXCHANGE, TimeUnit
from stlens.ingestion.binance.rest import BinanceRestClient, BinanceRestError
from stlens.ingestion.raw import RawMessage
from stlens.ingestion.storage import RawMessageStore
from stlens.utils.timestamps import US_PER_S, us_to_datetime, wall_clock_us

log = logging.getLogger(__name__)

DEFAULT_WS_BASE = "wss://stream.binance.com:9443"
COMBINED_STREAM_PATH = "/stream"

# A connection factory: given a URL, an async context manager yielding an async iterator
# of frames. ``websockets.asyncio.client.connect`` satisfies this; tests inject fakes.
Connect = Callable[[str], AbstractAsyncContextManager[AsyncIterator[str | bytes]]]


def combined_stream_url(
    symbol: str,
    *,
    base: str = DEFAULT_WS_BASE,
    depth_speed: Literal["100ms", "1000ms"] = "100ms",
    time_unit: TimeUnit = TimeUnit.MICROSECOND,
) -> str:
    """URL for the combined ``<symbol>@depth[@100ms]`` + ``<symbol>@trade`` stream."""
    s = symbol.lower()
    depth = f"{s}@depth@100ms" if depth_speed == "100ms" else f"{s}@depth"
    url = f"{base}{COMBINED_STREAM_PATH}?streams={depth}/{s}@trade"
    if time_unit is TimeUnit.MICROSECOND:
        url += "&timeUnit=MICROSECOND"
    return url


def new_session_id(now_us: int) -> str:
    """Filesystem-safe, time-sortable, unique session id."""
    return f"{us_to_datetime(now_us):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"


def _default_connect(url: str) -> AbstractAsyncContextManager[AsyncIterator[str | bytes]]:
    # Imported lazily so the module imports without the optional dependency.
    from websockets.asyncio.client import connect

    return connect(url)  # type: ignore[return-value]


@dataclass
class SessionStats:
    session_id: str
    stream_messages: int = 0
    snapshots: int = 0
    exchange_info: int = 0
    rest_errors: list[str] = field(default_factory=list)
    end_reason: str = ""


class BinanceRecorder:
    """Record one symbol's raw Binance stream + REST snapshots into a :class:`RawMessageStore`."""

    def __init__(
        self,
        symbol: str,
        store: RawMessageStore,
        rest: BinanceRestClient,
        *,
        url: str | None = None,
        connect: Connect | None = None,
        snapshot_interval_s: float | None = None,
        clock: Callable[[], int] = wall_clock_us,
    ) -> None:
        if symbol != symbol.upper():
            raise ValueError(f"symbol must be upper-case, got {symbol!r}")
        self.symbol = symbol
        self.url = url or combined_stream_url(symbol)
        self._store = store
        self._rest = rest
        self._connect = connect or _default_connect
        self._snapshot_interval_us = (
            None if snapshot_interval_s is None else int(snapshot_interval_s * US_PER_S)
        )
        self._clock = clock

    async def record_session(
        self, *, stop: asyncio.Event | None = None, max_messages: int | None = None
    ) -> SessionStats:
        """Record until the connection closes, ``stop`` is set or ``max_messages`` frames."""
        session_id = new_session_id(self._clock())
        stats = SessionStats(session_id)
        next_seq = itertools.count().__next__
        tasks: set[asyncio.Task[None]] = set()
        snapshot_failed = asyncio.Event()

        async def fetch(kind: Literal["snapshot", "exchange_info"]) -> None:
            try:
                if kind == "snapshot":
                    raw = await self._rest.fetch_depth_snapshot(
                        self.symbol, session_id=session_id, next_seq=next_seq
                    )
                else:
                    raw = await self._rest.fetch_exchange_info(
                        self.symbol, session_id=session_id, next_seq=next_seq
                    )
            except (BinanceRestError, httpx.HTTPError, OSError, TimeoutError) as exc:
                stats.rest_errors.append(f"{kind}: {exc}")
                log.warning("session %s: %s fetch failed: %s", session_id, kind, exc)
                if kind == "snapshot":
                    snapshot_failed.set()
                return
            # No await between receipt and write: store order == recv_seq order.
            self._store.write(raw)
            if kind == "snapshot":
                stats.snapshots += 1
            else:
                stats.exchange_info += 1

        def start(kind: Literal["snapshot", "exchange_info"]) -> None:
            task = asyncio.create_task(fetch(kind))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        log.info("session %s: connecting to %s", session_id, self.url)
        last_snapshot_us: int | None = None
        try:
            async with self._connect(self.url) as frames:
                start("exchange_info")
                async for frame in frames:
                    recv_time_us = self._clock()
                    payload = frame.decode("utf-8") if isinstance(frame, bytes) else frame
                    self._store.write(
                        RawMessage(
                            exchange=EXCHANGE,
                            stream=COMBINED_STREAM_PATH,
                            payload=payload,
                            recv_time_us=recv_time_us,
                            session_id=session_id,
                            recv_seq=next_seq(),
                        )
                    )
                    stats.stream_messages += 1
                    due = last_snapshot_us is None or (
                        self._snapshot_interval_us is not None
                        and recv_time_us - last_snapshot_us >= self._snapshot_interval_us
                    )
                    if due:
                        last_snapshot_us = recv_time_us
                        start("snapshot")
                    if snapshot_failed.is_set():
                        stats.end_reason = "snapshot fetch failed"
                        break
                    if stop is not None and stop.is_set():
                        stats.end_reason = "stop requested"
                        break
                    if max_messages is not None and stats.stream_messages >= max_messages:
                        stats.end_reason = "max_messages reached"
                        break
                else:
                    stats.end_reason = "connection closed"
                # Let in-flight REST fetches finish so their results are stored.
                if tasks:
                    await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            self._store.flush()
        log.info("session %s ended: %s (%s)", session_id, stats.end_reason, stats)
        return stats

    async def run(
        self,
        *,
        stop: asyncio.Event,
        min_backoff_s: float = 5.0,
        max_backoff_s: float = 300.0,
        max_sessions: int | None = None,
    ) -> list[SessionStats]:
        """Record sessions back to back until ``stop`` is set, reconnecting with backoff.

        Every reconnect starts a new session (new id, new snapshot). The minimum backoff
        keeps well under the documented 300 connection attempts per 5 minutes per IP.
        """
        history: list[SessionStats] = []
        backoff = min_backoff_s
        while not stop.is_set() and (max_sessions is None or len(history) < max_sessions):
            try:
                stats = await self.record_session(stop=stop)
            except Exception as exc:  # connection-level failure: log, back off, retry
                log.warning("session failed to run: %r", exc)
                stats = SessionStats(session_id="", end_reason=f"error: {exc!r}")
            history.append(stats)
            if stop.is_set():
                break
            backoff = (
                min_backoff_s if stats.stream_messages > 0 else min(backoff * 2, max_backoff_s)
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=backoff)
        return history
