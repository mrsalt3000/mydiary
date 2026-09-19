"""ISO-week input selection for the weekly-summary layer (task T06).

The weekly-summary layer consumes **the daily summaries of one ISO week** — the
``DailySummary`` contracts of the layer ``daily_summary`` (task T05), one per
local calendar day (ADR-007 §1). It never reads the raw source tables
(``messages``, ``audio_files``, ``transcriptions``) and never reads a weekly
export: the day summaries are already the grounded rollup of the raw messages,
and this layer is a *merge* of them (T06 DoD: "только merge дневных саммари, не
raw"). That selection is a distinct concern from the LLM call, so it lives in
this small module inside the layer package:

* :func:`week_id_of` — the ISO week id ``YYYY-Www`` of a local date. The year
  part is the **ISO year** (``date.isocalendar().year``, ``%G`` in ``strftime``
  terms), which around the New Year differs from the calendar year;
* :func:`week_monday` / :func:`week_days` — the Monday and the seven local days
  of an ISO week (project week: Monday through Sunday in ``APP_TIMEZONE``);
* :func:`ensure_summaries_match_week` — the loud guard that every summary fed
  to the run belongs to the run's ``week_id``;
* :func:`collect_week_summaries` — the week's daily summaries, in day order.

Read-only by construction: the module reads ``artifacts`` rows of the
``daily_summary`` layer through the task-T08 store (``latest_layer_result``) and
never writes anywhere. The *newest* version of each day's lineage is used: a
re-run of a day summary produces ``v+1`` and the newest result is the current
one. A day with no summary yet is simply absent — a week may be partially
summarized — but a day that *has* a summary is never skipped, and a stored
payload that does not validate back into ``DailySummary``, or whose ``date``
does not match the day it is anchored to, raises :class:`WeeklyInputError`
instead of being silently dropped (a lost or duplicated day is a veto of T06).

Privacy: the module returns validated contract objects; it never logs and never
puts the source text into an error message (AGENTS.md §7-8).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, timedelta

from pydantic import ValidationError
from sqlalchemy.orm import Session

from diary.db.models import ArtifactLayer
from diary.db.results import latest_layer_result
from diary.domain.contracts import DailySummary
from diary.domain.contracts.weekly_summary import WEEK_ID_PATTERN, validate_week_id
from diary.domain.pipelines.base import PipelinePermanentError

__all__ = [
    "WeeklyInputError",
    "collect_week_summaries",
    "ensure_summaries_match_week",
    "week_days",
    "week_id_of",
    "week_monday",
]

_WEEK_ID_RE = re.compile(WEEK_ID_PATTERN)

#: Static, non-sensitive error summaries (AGENTS.md §8).
_BAD_WEEK_ID_MESSAGE = "week_id must be an existing ISO week (YYYY-Www, ISO year)."
_FOREIGN_DAY_MESSAGE = (
    "A daily summary belongs to a different ISO week than the week_id of this run."
)
_CORRUPT_DAY_MESSAGE = (
    "A stored daily summary of this week does not validate into the daily_summary contract."
)
_MISPLACED_DAY_MESSAGE = (
    "A stored daily summary is anchored to a different local day than the date it reports."
)


class WeeklyInputError(PipelinePermanentError):
    """The week cannot be turned into a valid ``daily_summary[]`` input.

    Raised for an unusable weekly input — a malformed ``week_id``, a daily
    summary that belongs to another week, or a stored day summary that no
    longer validates. The message is a static, non-sensitive summary
    (AGENTS.md §8); a failed run is never retried.
    """


def week_id_of(local_date: date) -> str:
    """The ISO week id ``YYYY-Www`` of the local ``local_date``.

    The year is the **ISO year** of the date's ISO week, not its calendar
    year: ``2027-01-01`` (a Friday) belongs to ``2026-W53``. The project week
    is Monday–Sunday in ``APP_TIMEZONE``, so the ISO week of a local date is
    the week of its Monday (ADR-007 §5).
    """
    iso_year, iso_week, _ = local_date.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def week_monday(week_id: str) -> date:
    """The Monday of the ISO week ``week_id``.

    Raises :class:`WeeklyInputError` for a wrong shape or a week that does not
    exist in that ISO year (``2026-W54``).
    """
    if not _WEEK_ID_RE.fullmatch(week_id):
        raise WeeklyInputError(_BAD_WEEK_ID_MESSAGE)
    try:
        validate_week_id(week_id)
        return date.fromisocalendar(int(week_id[:4]), int(week_id[6:]), 1)
    except ValueError as exc:
        raise WeeklyInputError(_BAD_WEEK_ID_MESSAGE) from exc


def week_days(week_id: str) -> list[date]:
    """The seven local days (Monday → Sunday) of the ISO week ``week_id``.

    The list may span two calendar years (``2026-W53`` runs from 2026-12-28 to
    2027-01-03), which is exactly why the anchor is the ISO week id and not a
    calendar date.
    """
    monday = week_monday(week_id)
    return [monday + timedelta(days=offset) for offset in range(7)]


def ensure_summaries_match_week(summaries: Sequence[DailySummary], week_id: str) -> None:
    """Fail loudly unless ``week_id`` is valid and every summary belongs to it.

    The run's anchor is the ``week_id``; a summary of another week inside the
    run would make the stored ``daily_ids`` a lie. The collected summaries
    satisfy this by construction — the check is what makes the caller-supplied
    ``summaries`` seam (tests, the T09 orchestrator) safe.
    """
    week_monday(week_id)  # validates the id shape and existence
    for summary in summaries:
        if week_id_of(summary.date) != week_id:
            raise WeeklyInputError(_FOREIGN_DAY_MESSAGE)


def collect_week_summaries(
    session: Session,
    *,
    week_id: str,
    local_tz: str,
) -> list[DailySummary]:
    """The daily summaries of the ISO week ``week_id``, in day order.

    For each of the seven local days the **newest** ``daily_summary`` result of
    that day's lineage is taken (a re-run of a day appends ``v+1``; the newest
    version is the day's current summary). Days without a summary are skipped —
    the week is then partially summarized, which is not a loss. Only artifacts
    of the ``daily_summary`` layer are read; ``messages``, ``transcriptions``
    and ``weekly_exports`` are never touched. A day whose stored payload does
    not validate, or whose ``date`` disagrees with its day anchor, fails loudly.
    """
    summaries: list[DailySummary] = []
    for day in week_days(week_id):
        artifact = latest_layer_result(
            session,
            layer=ArtifactLayer.daily_summary,
            local_date=day,
            local_tz=local_tz,
        )
        if artifact is None:
            continue
        payload = artifact.result_json
        if not isinstance(payload, dict):
            raise WeeklyInputError(_CORRUPT_DAY_MESSAGE)
        try:
            summary = DailySummary.model_validate(payload)
        except ValidationError as exc:
            raise WeeklyInputError(_CORRUPT_DAY_MESSAGE) from exc
        if summary.date != day:
            raise WeeklyInputError(_MISPLACED_DAY_MESSAGE)
        summaries.append(summary)
    return summaries
