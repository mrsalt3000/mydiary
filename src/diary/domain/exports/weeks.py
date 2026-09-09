"""Pure calendar-week helpers for weekly exports (Task 10).

A diary week is Monday 00:00:00 through Sunday 23:59:59.999999 in the
configured application timezone (product decisions; architecture brief §7).
This module keeps that math pure: no database access and no settings import —
the timezone is passed explicitly, so unit tests can exercise any zone.

The exported week is identified by its **Sunday date** (the date used in
artifact file names). Two entry points are provided:

- given a Sunday date, the week is fully determined
  (:func:`week_monday`, :func:`week_bounds`);
- given any date of a week, :func:`sunday_date_for` finds that week's Sunday.

:func:`week_bounds` returns the two boundary *instants* — Monday 00:00:00
local and Sunday 23:59:59.999999 local — as timezone-aware datetimes
normalised to UTC. ``messages.created_at`` is stored in UTC, so the export
service filters messages with a plain ``start <= created_at <= end``
comparison without further conversion.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo

#: ``weekday()`` of a Sunday (the last day of a diary week).
_SUNDAY_WEEKDAY = 6


def is_sunday(value: date) -> bool:
    """Whether ``value`` is a Sunday (the diary week's last day)."""
    return value.weekday() == _SUNDAY_WEEKDAY


def sunday_date_for(value: date) -> date:
    """The Sunday date of the calendar week that contains ``value``.

    Weeks start on Monday, so for example the Sunday of the week of
    ``2026-09-10`` (a Thursday) is ``2026-09-13``; the Sunday of a Sunday is
    the date itself.
    """
    monday = value - timedelta(days=value.weekday())
    return monday + timedelta(days=6)


def week_monday(sunday: date) -> date:
    """The Monday of the week whose Sunday date is ``sunday``.

    Raises :class:`ValueError` when ``sunday`` is not actually a Sunday.
    """
    if not is_sunday(sunday):
        raise ValueError(
            f"{sunday.isoformat()} is not a Sunday; "
            "the export week must be identified by its Sunday date."
        )
    return sunday - timedelta(days=6)


def week_bounds(sunday: date, tz: tzinfo) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` — the boundary instants of the week of ``sunday``.

    ``start`` is the local wall-clock Monday 00:00:00 converted to its UTC
    instant; ``end`` is the local Sunday 23:59:59.999999 (i.e. one microsecond
    before the following local Monday 00:00:00). Both results are
    timezone-aware UTC datetimes, directly comparable with the UTC-aware
    ``messages.created_at`` timestamps.

    The end is derived from the *following* local midnight rather than from
    ``start + 7 days`` so that the local week always spans exactly seven local
    calendar days even across a DST transition.

    Raises :class:`ValueError` when ``sunday`` is not actually a Sunday.
    """
    if not is_sunday(sunday):
        raise ValueError(
            f"{sunday.isoformat()} is not a Sunday; "
            "the export week must be identified by its Sunday date."
        )
    start_local = datetime.combine(week_monday(sunday), time.min, tzinfo=tz)
    next_monday_local = datetime.combine(sunday + timedelta(days=1), time.min, tzinfo=tz)
    end_local = next_monday_local - timedelta(microseconds=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


__all__ = [
    "is_sunday",
    "sunday_date_for",
    "week_bounds",
    "week_monday",
]
