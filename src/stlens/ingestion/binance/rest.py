"""Binance Spot REST client for public market data: depth snapshots and exchangeInfo.

Returns :class:`RawMessage` objects (exact response text + receipt time); parsing is
done by :class:`stlens.ingestion.binance.parser.BinanceAdapter`. No automatic retries:
HTTP errors (including 429 rate-limit and 418 IP-ban responses) raise
:class:`BinanceRestError` carrying the ``Retry-After`` value, and the caller decides.
"""

from collections.abc import Callable

import httpx

from stlens.ingestion.raw import RawMessage
from stlens.utils.timestamps import wall_clock_us

# Documented base endpoint for public market data only.
DEFAULT_BASE_URL = "https://data-api.binance.vision"
DEPTH_PATH = "/api/v3/depth"
EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
MAX_DEPTH_LIMIT = 5000  # documented maximum; weight 250 for limit 1001-5000


class BinanceRestError(RuntimeError):
    """Non-200 response from the Binance REST API."""

    def __init__(self, request: str, status_code: int, body: str, retry_after: str | None) -> None:
        super().__init__(
            f"{request} -> HTTP {status_code}"
            + (f" (Retry-After: {retry_after}s)" if retry_after is not None else "")
            + f": {body[:500]}"
        )
        self.request = request
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after


class BinanceRestClient:
    """Async client for the two public endpoints Phase 1 needs.

    Args:
        http: An ``httpx.AsyncClient``; injected so tests can use ``httpx.MockTransport``
            and never touch the network. Its ``base_url`` must be set.
        clock: Receipt-time source (UTC microseconds); share the recorder's clock.
    """

    def __init__(self, http: httpx.AsyncClient, clock: Callable[[], int] = wall_clock_us) -> None:
        if not str(http.base_url):
            raise ValueError("http client must have a base_url")
        self._http = http
        self._clock = clock

    @classmethod
    def create(
        cls, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 10.0
    ) -> "BinanceRestClient":
        return cls(httpx.AsyncClient(base_url=base_url, timeout=timeout_s))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(
        self, path: str, params: dict[str, str], *, session_id: str, next_seq: Callable[[], int]
    ) -> RawMessage:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        request = f"GET {path}?{query}"
        response = await self._http.get(path, params=params)
        # Stamped when the full response has arrived. The receipt counter is taken at the
        # same moment (no await in between), so recv_seq follows actual receipt order even
        # when stream messages arrive while this request is in flight.
        # Failed requests consume no sequence number, so recv_seq stays gap-free.
        recv_time_us = self._clock()
        if response.status_code != 200:
            raise BinanceRestError(
                request, response.status_code, response.text, response.headers.get("Retry-After")
            )
        return RawMessage(
            exchange="binance",
            stream=request,
            payload=response.text,
            recv_time_us=recv_time_us,
            session_id=session_id,
            recv_seq=next_seq(),
        )

    async def fetch_depth_snapshot(
        self,
        symbol: str,
        *,
        session_id: str,
        next_seq: Callable[[], int],
        limit: int = MAX_DEPTH_LIMIT,
    ) -> RawMessage:
        """``GET /api/v3/depth`` as a raw message."""
        if not 1 <= limit <= MAX_DEPTH_LIMIT:
            raise ValueError(f"limit must be in [1, {MAX_DEPTH_LIMIT}], got {limit}")
        return await self._get(
            DEPTH_PATH,
            {"symbol": symbol, "limit": str(limit)},
            session_id=session_id,
            next_seq=next_seq,
        )

    async def fetch_exchange_info(
        self, symbol: str, *, session_id: str, next_seq: Callable[[], int]
    ) -> RawMessage:
        """``GET /api/v3/exchangeInfo?symbol=...`` as a raw message."""
        return await self._get(
            EXCHANGE_INFO_PATH, {"symbol": symbol}, session_id=session_id, next_seq=next_seq
        )
