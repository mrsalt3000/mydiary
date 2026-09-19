"""Contract ``daily_summary`` (layer 5, task T00).

Fixes the per-day aggregation contract of the phase-2 family (variant B;
ADR-007)::

    daily_summary: date, facts[], themes[], energy, open_questions[]

Purpose: aggregate the ``raw_entry`` messages of one local calendar day into
grounded facts, themes and open questions. The anchor is the **day**
(``date``), i.e. ``(APP_TIMEZONE day)`` — not the weekly export, so this layer
exists even for days that will never be exported.

Decision — ``energy`` must never be invented:

``energy`` is ``low`` / ``medium`` / ``high`` **or** ``None``. ``None`` means
"the source of this day does not say", and it is the only honest value when the
messages do not state the day's energy — a model must not infer or guess a
level to fill the field. The same rule is asserted for golden chunks through
``GoldenInvariants.expected_energy`` (``eval/golden``): a chunk whose source is
silent about energy expects ``energy: null`` in the output.

Grounding: ``facts`` are statements supported by the day's messages; numbers
and names must come from the source, and the day's messages are never rewritten
(ADR-005). ``themes`` and ``open_questions`` may be empty lists; ``facts`` is
never empty — a day summary without a single fact would be a fabrication.
"""

from __future__ import annotations

import datetime
from typing import Annotated, ClassVar, Literal

from pydantic import Field

from diary.domain.contracts._base import ContractModel

#: A summary item that must not be an empty string.
NonEmptyText = Annotated[str, Field(min_length=1)]

#: Allowed energy levels; ``None`` (field left unset) means "not stated".
EnergyLevel = Literal["low", "medium", "high"]


class DailySummary(ContractModel):
    """Grounded rollup of one local calendar day of diary messages."""

    contract_name: ClassVar[str] = "daily_summary"

    #: Local calendar day (``APP_TIMEZONE``) the summary covers.
    date: datetime.date

    #: Facts stated by the day's messages; at least one, never invented.
    facts: list[NonEmptyText] = Field(min_length=1)

    #: Recurring themes of the day; may be empty.
    themes: list[NonEmptyText] = Field(default_factory=list)

    #: Day energy as stated by the source; ``None`` = the source does not say
    #: (never guess a level to fill the field).
    energy: EnergyLevel | None = None

    #: Open questions / unknowns the day leaves behind; may be empty.
    open_questions: list[NonEmptyText] = Field(default_factory=list)
