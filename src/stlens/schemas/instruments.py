"""Instrument reference data (tick size, lot step), exchange-agnostic."""

from dataclasses import dataclass
from decimal import Decimal

from stlens.schemas.events import SourceRef, _check_decimal, _check_time, _require


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Trading rules for one symbol as published by the exchange at ``recv_time_us``.

    ``tick_size``/``step_size`` of 0 mean the exchange has disabled that rule; they are
    kept as 0. Specs can change over time, so each one is tied to when it was retrieved.
    """

    exchange: str
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    step_size: Decimal
    recv_time_us: int
    source: SourceRef

    def __post_init__(self) -> None:
        for name in ("exchange", "symbol", "status", "base_asset", "quote_asset"):
            _require(bool(getattr(self, name)), f"{name} must be non-empty")
        _check_decimal("tick_size", self.tick_size, positive=False)
        _check_decimal("step_size", self.step_size, positive=False)
        _check_time("recv_time_us", self.recv_time_us)
