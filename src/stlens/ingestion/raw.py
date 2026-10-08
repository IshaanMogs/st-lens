"""Raw-message boundary: exchange messages exactly as received, before any parsing.

A :class:`RawMessage` is the unit that will be persisted by the recorder (spec B.1) and
the only input an exchange adapter accepts. ``payload`` is the exact text received;
it is never re-serialised, so the original bytes stay auditable.
"""

from dataclasses import dataclass

from stlens.utils.timestamps import is_plausible_us, wall_clock_us


class RawMessageError(ValueError):
    """A raw message envelope is itself invalid (not a payload problem)."""


@dataclass(frozen=True, slots=True)
class RawMessage:
    """One message as received from an exchange.

    Attributes:
        exchange: Exchange name, e.g. ``"binance"``.
        stream: Where the message came from: a WebSocket stream/URL path or a REST
            request description, e.g. ``"btcusdt@depth@100ms"`` or
            ``"GET /api/v3/depth?symbol=BTCUSDT&limit=5000"``.
        payload: Exact message text as received (UTF-8 decoded).
        recv_time_us: Local wall-clock receipt time, UTC microseconds.
        session_id: Identifier of the capture session (one connection lifetime).
        recv_seq: Local receipt counter within the session, starting at 0. Assigned
            by this machine; not an exchange identifier.
    """

    exchange: str
    stream: str
    payload: str
    recv_time_us: int
    session_id: str
    recv_seq: int

    def __post_init__(self) -> None:
        for name in ("exchange", "stream", "session_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RawMessageError(f"{name} must be a non-empty str")
        if not isinstance(self.payload, str):
            raise RawMessageError("payload must be str (decoded text as received)")
        for name in ("recv_time_us", "recv_seq"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RawMessageError(f"{name} must be a non-negative int")
        if not is_plausible_us(self.recv_time_us):
            raise RawMessageError(
                f"recv_time_us={self.recv_time_us} is outside the plausible UTC-us range"
            )

    @classmethod
    def received_now(
        cls, *, exchange: str, stream: str, payload: str, session_id: str, recv_seq: int
    ) -> "RawMessage":
        """Stamp a message with the current wall-clock receipt time."""
        return cls(
            exchange=exchange,
            stream=stream,
            payload=payload,
            recv_time_us=wall_clock_us(),
            session_id=session_id,
            recv_seq=recv_seq,
        )


@dataclass(frozen=True, slots=True)
class QuarantinedMessage:
    """A raw message that failed validation, kept with the reason (spec B.3: never dropped)."""

    raw: RawMessage
    reason: str
