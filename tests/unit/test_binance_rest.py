"""Binance REST client against httpx.MockTransport (no network)."""

import asyncio
import itertools
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from stlens.ingestion.binance import BinanceAdapter
from stlens.ingestion.binance.rest import BinanceRestClient, BinanceRestError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "binance"
SNAPSHOT = (FIXTURES / "depth_snapshot.json").read_text(encoding="utf-8")
EXCHANGE_INFO = (FIXTURES / "exchange_info.json").read_text(encoding="utf-8")


def client(handler) -> BinanceRestClient:
    transport = httpx.MockTransport(handler)
    return BinanceRestClient(httpx.AsyncClient(transport=transport, base_url="https://test"))


def run(coro):
    return asyncio.run(coro)


def test_depth_snapshot_request_and_raw_response():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=SNAPSHOT)

    raw = run(client(handler).fetch_depth_snapshot("BTCUSDT", session_id="S1", next_seq=lambda: 9))
    (request,) = seen
    assert request.method == "GET"
    assert request.url.path == "/api/v3/depth"
    assert dict(request.url.params) == {"symbol": "BTCUSDT", "limit": "5000"}
    assert raw.payload == SNAPSHOT  # exact response text
    assert raw.stream == "GET /api/v3/depth?symbol=BTCUSDT&limit=5000"
    assert (raw.exchange, raw.session_id, raw.recv_seq) == ("binance", "S1", 9)
    # The raw response parses into a canonical snapshot.
    (snap,) = BinanceAdapter("BTCUSDT").parse_depth_snapshot(raw).events
    assert snap.last_update_id == 75100000000


def test_exchange_info_request():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/exchangeInfo"
        assert dict(request.url.params) == {"symbol": "BTCUSDT"}
        return httpx.Response(200, text=EXCHANGE_INFO)

    raw = run(client(handler).fetch_exchange_info("BTCUSDT", session_id="S1", next_seq=lambda: 0))
    assert BinanceAdapter("BTCUSDT").parse_exchange_info(raw).tick_size == Decimal("0.01")


@pytest.mark.parametrize("status", [400, 418, 429, 500])
def test_http_errors_raise_with_retry_after_and_consume_no_sequence(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text='{"code":-1003}', headers={"Retry-After": "30"})

    counter = itertools.count()
    with pytest.raises(BinanceRestError) as excinfo:
        run(
            client(handler).fetch_depth_snapshot(
                "BTCUSDT", session_id="S1", next_seq=counter.__next__
            )
        )
    assert excinfo.value.status_code == status
    assert excinfo.value.retry_after == "30"
    assert next(counter) == 0  # failed request did not use a recv_seq


def test_depth_limit_is_bounded():
    with pytest.raises(ValueError, match="limit"):
        run(
            client(lambda r: httpx.Response(200)).fetch_depth_snapshot(
                "BTCUSDT", session_id="S1", next_seq=lambda: 0, limit=5001
            )
        )
