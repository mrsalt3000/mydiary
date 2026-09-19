"""Contract ``weekly_summary`` (layer 5, task T00).

Fixes the weekly rollup contract of the phase-2 family (variant B; ADR-007)::

    weekly_summary: week_id, daily_ids[], rollup

Purpose: roll the per-day summaries of one week up into a single grounded text.
The anchor is the **ISO week id**, not the original weekly export: the phase-2
family must work for weeks that were never exported (ADR-007, second anchor
family).

``week_id`` format: ``YYYY-Www`` (pattern ``^\\d{4}-W\\d{2}$``) with a real ISO
year and a real ISO week number (``2026-W54`` is rejected). The year part is
the **ISO year** (``%G``), which may differ from the calendar year around the
New Year; the project week is Monday–Sunday in ``APP_TIMEZONE`` and maps 1:1
onto the ISO week of its Monday (ADR-007). The existing export anchor — the
Sunday date ``YYYY-MM-DD`` of ``weekly_exports`` — is not replaced by this
identifier; the two anchors are related by the mapping recorded in
``docs/data-lineage.md``.

``daily_ids`` identifies the ``DailySummary`` objects that were rolled up. In
this contract a daily summary is identified by its ISO date, so every entry is
the ``YYYY-MM-DD`` string of a referenced ``DailySummary.date`` (a
machine-checkable grounding link; ``set(daily_ids) ⊆ declared`` in the golden
invariants). At least one day is required and a day must not repeat.

``rollup`` is the grounded rollup text. It is deliberately not pinned to a
reference string: it is evaluated by invariants (no new facts, no advice, no
diagnoses), not by equality with an expected wording.
"""

from __future__ import annotations

import datetime
import re
from typing import Annotated, ClassVar

from pydantic import Field, field_validator

from diary.domain.contracts._base import ContractModel

#: Required shape of ``week_id``: ISO year plus ISO week number.
WEEK_ID_PATTERN = r"^\d{4}-W\d{2}$"

#: Required shape of one ``daily_ids`` entry: an ISO calendar date.
ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

_WEEK_ID_RE = re.compile(WEEK_ID_PATTERN)
_ISO_DATE_RE = re.compile(ISO_DATE_PATTERN)

#: A non-empty rollup line.
NonEmptyText = Annotated[str, Field(min_length=1)]


def validate_week_id(value: str) -> str:
    """Return ``value`` if it is a well-formed, existing ISO week id.

    Raises :class:`ValueError` for a wrong shape, a week number outside
    1-53, or a week that does not exist in that ISO year (e.g. ``2026-W53``).
    """
    if not _WEEK_ID_RE.fullmatch(value):
        raise ValueError("week_id must look like YYYY-Www (for example 2026-W38)")
    iso_year, iso_week = int(value[:4]), int(value[6:])
    try:
        datetime.date.fromisocalendar(iso_year, iso_week, 1)
    except ValueError as exc:
        raise ValueError(f"week_id {value!r} is not an existing ISO week") from exc
    return value


def _validate_daily_id(value: str) -> str:
    if not _ISO_DATE_RE.fullmatch(value):
        raise ValueError("daily_ids entries must be ISO dates (YYYY-MM-DD)")
    try:
        datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"daily_ids entry {value!r} is not a real date") from exc
    return value


class WeeklySummary(ContractModel):
    """Grounded rollup of the per-day summaries of one ISO week."""

    contract_name: ClassVar[str] = "weekly_summary"

    #: ISO week identifier ``YYYY-Www`` (project week Monday–Sunday; ISO year).
    week_id: str = Field(pattern=WEEK_ID_PATTERN)

    #: ISO dates of the rolled-up ``DailySummary`` objects; non-empty, unique.
    daily_ids: list[NonEmptyText] = Field(min_length=1)

    #: Grounded rollup text; evaluated by invariants, not by exact wording.
    rollup: str = Field(min_length=1)

    @field_validator("week_id")
    @classmethod
    def _week_id_exists(cls, value: str) -> str:
        return validate_week_id(value)

    @field_validator("daily_ids")
    @classmethod
    def _daily_ids_are_iso_dates(cls, value: list[str]) -> list[str]:
        checked = [_validate_daily_id(item) for item in value]
        if len(set(checked)) != len(checked):
            raise ValueError("daily_ids must not repeat a day")
        return checked
