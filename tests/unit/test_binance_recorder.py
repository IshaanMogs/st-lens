"""Recorder against a fake WebSocket and httpx.MockTransport (no network, no Binance)."""

import asyncio
import itertools
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from stlens.ingestion.binance import TimeUnit
from stlens.ingestion.binance.recorder import BinanceRecorder, combined_stream_url
from stlens.ingestion.binance.rest import BinanceRestClient
from stlens.ingestion.storage import RawMessageStore, read_raw_messages

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "binance"
SNAPSHOT = (FIXTURES / "depth_snapshot.json").read_text(encoding="utf-8")
EXCHANGE_INFO = (FIXTURES / "exchange_info.json").read_text(encoding="utf-8")
FRAMES = [
    (FIXTURES / name).read_text(encoding="utf-8").rstrip("\n")
    for name in ("depth_update_combined_us.json", "trade_combined_us.json")
] * 3

T0 = 1_790_856_000_000_000


def fake_clock():
    counter = itertools.count()
    return lambda: T0 + next(counter) * 1_000


def fake_connect(frames, urls, *, pause: float = 0.0):
    @asynccontextmanager
    async def connect(url):
        urls.append(url)

        async def stream():
            for frame in frames:
                await asyncio.sleep(pause)
                yield frame

        yield stream()

    return connect


def rest(snapshot_status=200, snapshot_delay=0.0, info_status=200, clock=None):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/depth":
            await asyncio.sleep(snapshot_delay)
            return httpx.Response(snapshot_status, text=SNAPSHOT)
        return httpx.Response(info_status, text=EXCHANGE_INFO)

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="https://test")
    return BinanceRestClient(http, clock=clock or fake_clock())


def record(tmp_path, *, frames=FRAMES, rest_kw=None, **kw):
    urls: list[str] = []
    clock = fake_clock()  # one shared time source for stream frames and REST responses
    store = RawMessageStore(tmp_path, "binance", "BTCUSDT")
    recorder = BinanceRecorder(
        "BTCUSDT",
        store,
        rest(**(rest_kw or {}), clock=clock),
        connect=fake_connect(frames, urls, pause=kw.pop("pause", 0.0)),
        clock=clock,
    )
    stats = asyncio.run(recorder.record_session(**kw))
    store.close()
    stored = [m for p in store.paths for m in read_raw_messages(p)]
    return stats, stored, urls


def test_combined_stream_url():
    assert combined_stream_url("BTCUSDT") == (
        "wss://stream.binance.com:9443/stream"
        "?streams=btcusdt@depth@100ms/btcusdt@trade&timeUnit=MICROSECOND"
    )
    assert combined_stream_url("BTCUSDT", depth_speed="1000ms", time_unit=TimeUnit.MILLISECOND) == (
        "wss://stream.binance.com:9443/stream?streams=btcusdt@depth/btcusdt@trade"
    )


def test_session_stores_every_frame_exactly_with_gap_free_sequence(tmp_path):
    stats, stored, urls = record(tmp_path)
    assert urls == [combined_stream_url("BTCUSDT")]
    assert stats.stream_messages == len(FRAMES)
    assert (stats.snapshots, stats.exchange_info, stats.rest_errors) == (1, 1, [])
    assert stats.end_reason == "connection closed"
    assert [m.recv_seq for m in stored] == list(range(len(stored)))
    assert {m.session_id for m in stored} == {stats.session_id}
    stream_payloads = [m.payload for m in stored if m.stream == "/stream"]
    assert stream_payloads == FRAMES  # exact text, receipt order
    rest_streams = sorted(m.stream for m in stored if m.stream != "/stream")
    assert rest_streams == [
        "GET /api/v3/depth?symbol=BTCUSDT&limit=5000",
        "GET /api/v3/exchangeInfo?symbol=BTCUSDT",
    ]


def test_snapshot_is_requested_after_first_stream_message(tmp_path):
    _, stored, _ = record(tmp_path, pause=0.001)
    first_stream = next(i for i, m in enumerate(stored) if m.stream == "/stream")
    snapshot = next(i for i, m in enumerate(stored) if m.stream.startswith("GET /api/v3/depth"))
    assert first_stream < snapshot


def test_slow_snapshot_gets_sequence_at_arrival_not_request(tmp_path):
    _, stored, _ = record(tmp_path, pause=0.005, rest_kw={"snapshot_delay": 0.012})
    snapshot = next(m for m in stored if m.stream.startswith("GET /api/v3/depth"))
    before = [m for m in stored if m.recv_seq < snapshot.recv_seq and m.stream == "/stream"]
    # Stream frames received while the request was in flight come first.
    assert len(before) > 1
    # Receipt times never decrease along recv_seq: nothing is reordered.
    times = [m.recv_time_us for m in sorted(stored, key=lambda m: m.recv_seq)]
    assert times == sorted(times)


def test_snapshot_failure_ends_session_without_fabrication(tmp_path):
    stats, stored, _ = record(tmp_path, pause=0.001, rest_kw={"snapshot_status": 429})
    assert stats.end_reason == "snapshot fetch failed"
    assert stats.snapshots == 0
    assert any("HTTP 429" in e for e in stats.rest_errors)
    assert not any(m.stream.startswith("GET /api/v3/depth") for m in stored)
    assert [m.recv_seq for m in stored] == list(range(len(stored)))


def test_exchange_info_failure_is_logged_not_fatal(tmp_path):
    stats, _, _ = record(tmp_path, rest_kw={"info_status": 500})
    assert stats.end_reason == "connection closed"
    assert stats.exchange_info == 0
    assert stats.snapshots == 1
    assert any("exchange_info" in e for e in stats.rest_errors)


def test_max_messages_stops_session(tmp_path):
    stats, stored, _ = record(tmp_path, max_messages=2)
    assert stats.stream_messages == 2
    assert stats.end_reason == "max_messages reached"
    assert sum(m.stream == "/stream" for m in stored) == 2


def test_run_reconnects_with_new_session_after_failure(tmp_path):
    attempts: list[str] = []
    ok = fake_connect(FRAMES[:2], attempts)

    @asynccontextmanager
    async def flaky(url):
        if not attempts:
            attempts.append(url)
            raise OSError("connection refused")
        async with ok(url) as frames:
            yield frames

    clock = fake_clock()
    store = RawMessageStore(tmp_path, "binance", "BTCUSDT")
    recorder = BinanceRecorder("BTCUSDT", store, rest(clock=clock), connect=flaky, clock=clock)
    history = asyncio.run(
        recorder.run(stop=asyncio.Event(), min_backoff_s=0, max_backoff_s=0, max_sessions=3)
    )
    store.close()
    assert history[0].end_reason.startswith("error: OSError")
    assert [h.end_reason for h in history[1:]] == ["connection closed"] * 2
    assert len({h.session_id for h in history[1:]}) == 2  # each reconnect is a new session


def test_recorder_requires_upper_case_symbol(tmp_path):
    with pytest.raises(ValueError, match="upper-case"):
        BinanceRecorder("btcusdt", RawMessageStore(tmp_path, "binance", "BTCUSDT"), rest())
