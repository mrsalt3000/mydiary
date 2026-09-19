"""Unit tests for the ISO-week helpers of T06 (pure, no database).

The week math and the membership guard are pure functions of (explicit week id,
explicit dates), so they are testable without a session:

* the ``YYYY`` part of ``week_id`` is the **ISO year**, not the calendar year —
  around the New Year they differ, and a calendar-year implementation would put
  days of one week into two different buckets;
* an ISO week is seven consecutive local days, Monday → Sunday, and it may span
  two calendar years (``2026-W53`` runs from 2026-12-28 to 2027-01-03);
* a week id that does not exist in its ISO year (``2026-W54``) or has the wrong
  shape is rejected loudly;
* every summary fed to a run must belong to the run's week id.

All values are synthetic (AGENTS.md §13).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from diary.domain.contracts import DailySummary
from diary.domain.pipelines.sum_weekly.week_input import (
    WeeklyInputError,
    ensure_summaries_match_week,
    week_days,
    week_id_of,
    week_monday,
)


def _summary(day: date) -> DailySummary:
    return DailySummary(date=day, facts=[f"Синтетический факт {day.isoformat()}."])


# ---------------------------------------------------------------------------
# the ISO year, not the calendar year
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        # A Monday in the middle of a year.
        ("2026-09-14", "2026-W38"),
        # The last ISO week of 2026 starts in December and ends in 2027.
        ("2026-12-28", "2026-W53"),
        ("2026-12-31", "2026-W53"),
        ("2027-01-01", "2026-W53"),
        ("2027-01-03", "2026-W53"),
        # The first ISO week of 2027 starts on 2027-01-04.
        ("2027-01-04", "2027-W01"),
        # The mirror case: the first days of 2026 belong to ISO week 2026-W01,
        # whose Monday falls in December 2025.
        ("2025-12-29", "2026-W01"),
        ("2026-01-01", "2026-W01"),
        ("2026-01-04", "2026-W01"),
        ("2026-01-05", "2026-W02"),
        # A year with 53 ISO weeks (2020 had one).
        ("2021-01-01", "2020-W53"),
    ],
)
def test_week_id_uses_the_iso_year(day: str, expected: str) -> None:
    assert week_id_of(date.fromisoformat(day)) == expected


def test_week_days_of_a_week_spanning_two_calendar_years() -> None:
    """2026-W53 is one local week even though it crosses the New Year."""
    days = week_days("2026-W53")

    assert [day.isoformat() for day in days] == [
        "2026-12-28",
        "2026-12-29",
        "2026-12-30",
        "2026-12-31",
        "2027-01-01",
        "2027-01-02",
        "2027-01-03",
    ]
    assert days[0].weekday() == 0  # Monday
    assert days[-1].weekday() == 6  # Sunday


def test_week_days_are_seven_consecutive_days() -> None:
    days = week_days("2026-W38")

    assert days[0] == date(2026, 9, 14)
    assert len(days) == 7
    assert all(days[index + 1] - days[index] == timedelta(days=1) for index in range(6))
    # Every day of the week maps back to the same week id.
    assert {week_id_of(day) for day in days} == {"2026-W38"}


@pytest.mark.parametrize("week_id", ["2026-W01", "2026-W38", "2026-W53", "2027-W01", "2020-W53"])
def test_week_monday_round_trips_through_week_id_of(week_id: str) -> None:
    assert week_id_of(week_monday(week_id)) == week_id


@pytest.mark.parametrize("week_id", ["2026-W54", "2026-W00", "2026-W1", "2026-38", "abc", ""])
def test_a_malformed_or_nonexistent_week_id_fails_loudly(week_id: str) -> None:
    with pytest.raises(WeeklyInputError):
        week_monday(week_id)
    with pytest.raises(WeeklyInputError):
        week_days(week_id)


# ---------------------------------------------------------------------------
# the membership guard
# ---------------------------------------------------------------------------


def test_summaries_of_the_week_pass_the_guard() -> None:
    summaries = [_summary(day) for day in week_days("2026-W38")]

    ensure_summaries_match_week(summaries, "2026-W38")  # does not raise


def test_a_summary_of_another_week_fails_the_guard() -> None:
    summaries = [_summary(date(2026, 9, 14)), _summary(date(2026, 9, 21))]  # W38 and W39

    with pytest.raises(WeeklyInputError):
        ensure_summaries_match_week(summaries, "2026-W38")


def test_the_guard_rejects_a_malformed_week_id() -> None:
    with pytest.raises(WeeklyInputError):
        ensure_summaries_match_week([], "2026-W54")
