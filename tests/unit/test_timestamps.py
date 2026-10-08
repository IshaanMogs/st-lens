from datetime import UTC, datetime, timedelta, timezone

import pytest

from stlens.utils.timestamps import (
    MAX_PLAUSIBLE_US,
    MIN_PLAUSIBLE_US,
    datetime_to_us,
    is_plausible_us,
    ms_to_us,
    us_to_datetime,
    wall_clock_us,
)

FIXTURE_US = 1_790_856_000_123_456  # 2026-10-01T12:00:00.123456Z


def test_ms_to_us_is_exact():
    assert ms_to_us(1_790_856_000_123) == 1_790_856_000_123_000


def test_us_to_datetime_is_utc_and_exact():
    dt = us_to_datetime(FIXTURE_US)
    assert dt == datetime(2026, 10, 1, 12, 0, 0, 123_456, tzinfo=UTC)
    assert dt.utcoffset() == timedelta(0)


def test_datetime_round_trip():
    assert datetime_to_us(us_to_datetime(FIXTURE_US)) == FIXTURE_US


def test_datetime_to_us_converts_other_timezones_to_utc():
    plus_two = datetime(2026, 10, 1, 14, 0, 0, 123_456, tzinfo=timezone(timedelta(hours=2)))
    assert datetime_to_us(plus_two) == FIXTURE_US


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="naive"):
        datetime_to_us(datetime(2026, 10, 1))  # noqa: DTZ001 - deliberately naive


def test_plausibility_bounds():
    assert is_plausible_us(FIXTURE_US)
    assert is_plausible_us(MIN_PLAUSIBLE_US)
    assert not is_plausible_us(MAX_PLAUSIBLE_US)
    # A millisecond value mistaken for microseconds lands in 1970 and is implausible.
    assert not is_plausible_us(1_790_856_000_123)


def test_wall_clock_is_plausible_microseconds():
    assert is_plausible_us(wall_clock_us())
