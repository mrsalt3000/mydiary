"""Unit tests for the day-selection helpers of T05 (pure, no database).

The boundary math and the ``local_date_of`` mapping are pure functions of
(explicit date, explicit timezone), so they are testable without a session:

* a local calendar day is bounded by local midnight and the instant before the
  next local midnight — a DST day is 23 or 25 hours long, never a fixed 24, so
  a UTC-only implementation would put messages on the wrong day;
* a moment belongs to the day of its *local* date, which for ``APP_TIMEZONE``
  is not the date of its UTC instant;
* a naive (offset-less) moment is rejected: its local day is undefined.

All values are synthetic (AGENTS.md §13).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from diary.domain.pipelines.sum_daily.day_input import day_bounds, local_date_of

MOSCOW = ZoneInfo("Europe/Moscow")
NEW_YORK = ZoneInfo("America/New_York")

#: The local day used by most cases below (a Monday, no DST transition).
DAY = date(2026, 9, 14)


def test_a_local_day_spans_local_midnight_to_the_next_local_midnight() -> None:
    start, end = day_bounds(DAY, MOSCOW)

    # Moscow is UTC+3 all year: local midnight is 21:00 UTC of the day before.
    assert start == datetime(2026, 9, 13, 21, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 14, 20, 59, 59, 999999, tzinfo=timezone.utc)
    assert end - start == timedelta(days=1) - timedelta(microseconds=1)
    assert start.utcoffset() == timedelta(0)  # both bounds are UTC instants


def test_a_spring_forward_day_is_twenty_three_hours_long() -> None:
    """2026-03-08 in New York loses an hour: the local day is not 24 hours."""
    start, end = day_bounds(date(2026, 3, 8), NEW_YORK)

    assert start == datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc)  # 00:00 EST
    # 2026-03-08 23:59:59.999999 EDT (the next local midnight is 04:00 UTC).
    assert end == datetime(2026, 3, 9, 3, 59, 59, 999999, tzinfo=timezone.utc)
    assert end - start == timedelta(hours=23) - timedelta(microseconds=1)


def test_a_fall_back_day_is_twenty_five_hours_long() -> None:
    """2026-11-01 in New York repeats an hour: the local day is 25 hours."""
    start, end = day_bounds(date(2026, 11, 1), NEW_YORK)

    assert start == datetime(2026, 11, 1, 4, 0, tzinfo=timezone.utc)  # 00:00 EDT
    # 2026-11-01 23:59:59.999999 EST (the next local midnight is 05:00 UTC).
    assert end == datetime(2026, 11, 2, 4, 59, 59, 999999, tzinfo=timezone.utc)
    assert end - start == timedelta(hours=25) - timedelta(microseconds=1)


def test_day_bounds_of_adjacent_days_touch_without_overlap() -> None:
    _, end = day_bounds(DAY, MOSCOW)
    next_start, _ = day_bounds(DAY + timedelta(days=1), MOSCOW)

    assert next_start - end == timedelta(microseconds=1)


def test_local_date_of_uses_the_local_day_not_the_utc_day() -> None:
    # 2026-09-13 22:00 UTC is 2026-09-14 01:00 in Moscow: the next local day.
    assert local_date_of(datetime(2026, 9, 13, 22, 0, tzinfo=timezone.utc), MOSCOW) == DAY
    # 2026-09-14 20:59:59.999999 UTC is still 23:59:59.999999 of that local day.
    assert (
        local_date_of(datetime(2026, 9, 14, 20, 59, 59, 999999, tzinfo=timezone.utc), MOSCOW) == DAY
    )
    # One microsecond later the local day has turned over.
    tomorrow = date(2026, 9, 15)
    assert local_date_of(datetime(2026, 9, 14, 21, 0, tzinfo=timezone.utc), MOSCOW) == tomorrow


def test_local_date_of_accepts_a_moment_in_another_offset() -> None:
    # The same instant, expressed in Moscow local time, keeps its local day.
    assert local_date_of(datetime(2026, 9, 14, 1, 0, tzinfo=MOSCOW), MOSCOW) == DAY


def test_local_date_of_rejects_a_naive_moment() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        local_date_of(datetime(2026, 9, 14, 13, 20), MOSCOW)
