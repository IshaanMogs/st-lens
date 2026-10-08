"""Timestamp conventions for ST-LENS.

All timestamps inside ST-LENS are ``int`` microseconds since the Unix epoch (UTC).
Integers avoid float rounding and the naive/aware ``datetime`` ambiguity; use
:func:`us_to_datetime` only for display.

Two clocks are never mixed:

* exchange time: stamped by the exchange (e.g. Binance ``E``/``T``);
* receipt time: this machine's wall clock when the message arrived.
"""

import time
from datetime import UTC, datetime

US_PER_MS = 1_000
US_PER_S = 1_000_000

# Plausibility bounds for exchange timestamps after unit conversion. A value outside
# them almost always means the wrong unit was assumed (e.g. ms read as us).
MIN_PLAUSIBLE_US = 1_483_228_800 * US_PER_S  # 2017-01-01T00:00:00Z
MAX_PLAUSIBLE_US = 4_102_444_800 * US_PER_S  # 2100-01-01T00:00:00Z


def wall_clock_us() -> int:
    """Current local wall-clock time in UTC microseconds (receipt time source)."""
    return time.time_ns() // 1_000


def ms_to_us(ms: int) -> int:
    """Convert integer milliseconds to integer microseconds (exact)."""
    return ms * US_PER_MS


def us_to_datetime(us: int) -> datetime:
    """Timezone-aware UTC datetime for display; exact to the microsecond."""
    seconds, micros = divmod(us, US_PER_S)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=micros)


def datetime_to_us(dt: datetime) -> int:
    """Convert an aware datetime to UTC microseconds. Naive datetimes are rejected."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("naive datetime: timezone is required")
    delta = dt - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * US_PER_S + delta.microseconds


def is_plausible_us(us: int) -> bool:
    """True if ``us`` lies within the plausibility bounds above."""
    return MIN_PLAUSIBLE_US <= us < MAX_PLAUSIBLE_US
