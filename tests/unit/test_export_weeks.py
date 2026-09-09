"""Unit tests for pure calendar-week helpers (Task 10).

Covers week-boundary math around Monday/Sunday and across timezones with a
non-whole-hour UTC offset (Asia/Kathmandu, UTC+05:45), the Sunday-date
derivation for arbitrary dates of a week, and validation of the Sunday input.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from diary.domain.exports.weeks import is_sunday, sunday_date_for, week_bounds, week_monday


def test_sunday_date_for_every_day_of_week() -> None:
    sunday = date(2026, 9, 13)
    for day in range(7):  # Monday 2026-09-07 … Sunday 2026-09-13
        assert sunday_date_for(date(2026, 9, 7 + day)) == sunday


def test_sunday_date_for_sunday_is_identity() -> None:
    assert sunday_date_for(date(2026, 9, 13)) == date(2026, 9, 13)


def test_sunday_date_across_year_boundary() -> None:
    # 2026-12-31 is a Thursday; its week runs 2026-12-28 (Mon) … 2027-01-03.
    assert sunday_date_for(date(2026, 12, 31)) == date(2027, 1, 3)


def test_is_sunday() -> None:
    assert is_sunday(date(2026, 9, 13))
    assert not is_sunday(date(2026, 9, 7))  # Monday
    assert not is_sunday(date(2026, 9, 12))  # Saturday


def test_week_monday() -> None:
    assert week_monday(date(2026, 9, 13)) == date(2026, 9, 7)


def test_week_monday_rejects_non_sunday() -> None:
    with pytest.raises(ValueError):
        week_monday(date(2026, 9, 7))


def test_week_bounds_moscow() -> None:
    tz = ZoneInfo("Europe/Moscow")  # UTC+03:00, no DST.
    start, end = week_bounds(date(2026, 9, 13), tz)
    assert start == datetime(2026, 9, 6, 21, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 13, 20, 59, 59, 999999, tzinfo=timezone.utc)
    assert end - start == timedelta(days=7, microseconds=-1)


def test_week_bounds_kathmandu() -> None:
    tz = ZoneInfo("Asia/Kathmandu")  # UTC+05:45: not a whole hour.
    start, end = week_bounds(date(2026, 9, 13), tz)
    # Local Monday 2026-09-07 00:00:00 = 2026-09-06 18:15 UTC.
    assert start == datetime(2026, 9, 6, 18, 15, 0, tzinfo=timezone.utc)
    # Local Sunday 2026-09-13 23:59:59.999999 = 2026-09-13 18:14:59.999999 UTC.
    assert end == datetime(2026, 9, 13, 18, 14, 59, 999999, tzinfo=timezone.utc)


def test_week_bounds_rejects_non_sunday() -> None:
    with pytest.raises(ValueError):
        week_bounds(date(2026, 9, 7), ZoneInfo("Europe/Moscow"))


def test_week_bounds_span_exactly_seven_local_days() -> None:
    """The week covers seven local calendar days (not just seven UTC days)."""
    tz = ZoneInfo("Europe/Moscow")
    sunday = date(2026, 9, 13)
    start, end = week_bounds(sunday, tz)
    start_local = start.astimezone(tz)
    end_local = end.astimezone(tz)
    assert start_local.isoformat().startswith("2026-09-07T00:00:00")
    assert end_local.isoformat().startswith("2026-09-13T23:59:59")
