"""Contract ``coach_note`` (new layer 5, task T00).

Fixes the coaching-note contract of the phase-2 family (variant B; ADR-007)::

    coach_note: summary_id, observations[3], hypotheses[2], experiment_7d,
                disclaimer

Purpose: turn one ``weekly_summary`` into a short, cautious note — exactly
three observations, exactly two hypotheses and one small experiment for the
next 7 days. This is a **new layer**: it does not replace or modify layer 4
(``wellbeing``), which keeps its own pipeline, prompt, safety rules and
disclaimer (ADR-006).

Safety position (inherits ADR-006 by construction):

- observations are supported by the weekly summary; hypotheses are explicitly
  hypotheses, never asserted causes;
- no diagnoses, no treatment plans, no invented symptoms, no advice phrased as
  an instruction (the golden invariants check the forbidden lexemes on the
  generated fields);
- ``disclaimer`` is validated by **equality** with :data:`COACH_DISCLAIMER`, so
  a report can never go out with a paraphrased or missing limitation. The
  constant is intentionally a separate one: it is not the layer-4
  ``wellbeing.safety.DISCLAIMER`` and must not be replaced by it.

``summary_id`` identifies the upstream ``WeeklySummary``. In this contract that
summary is identified by its ``week_id`` (there is no other identity for it),
so ``summary_id`` carries the ``YYYY-Www`` week id here; the field stays a
plain string so a later version can carry the persisted artifact id instead
without breaking the layer.

``experiment_7d`` must state the 7-day horizon explicitly (the digit ``7``):
the field is the actionable part of the note, and the horizon is what makes it
a bounded experiment rather than open-ended advice.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field, field_validator

from diary.domain.contracts._base import ContractModel

#: Mandatory limitation of every coaching note (layer 5). Validated by
#: equality with ``CoachNote.disclaimer``; separate from the layer-4
#: ``diary.domain.pipelines.wellbeing.safety.DISCLAIMER`` by design (ADR-007).
COACH_DISCLAIMER = (
    "Заметка собрана только из недельных саммари дневника. Это не медицинская "
    "и не психологическая помощь, не диагноз и не замена консультации "
    "специалиста; гипотезы — это предположения, а не установленные причины."
)

#: Number of observations required by the contract.
OBSERVATIONS_COUNT = 3

#: Number of hypotheses required by the contract.
HYPOTHESES_COUNT = 2

#: Horizon of the suggested experiment, in days.
EXPERIMENT_HORIZON_DAYS = 7

#: A note item that must not be an empty string.
NonEmptyText = Annotated[str, Field(min_length=1)]


class CoachNote(ContractModel):
    """Cautious, non-diagnostic note over one weekly summary."""

    contract_name: ClassVar[str] = "coach_note"

    #: Identity of the upstream ``WeeklySummary`` (its ``week_id`` here).
    summary_id: str = Field(min_length=1)

    #: Exactly three observations grounded in the weekly summary.
    observations: list[NonEmptyText] = Field(
        min_length=OBSERVATIONS_COUNT, max_length=OBSERVATIONS_COUNT
    )

    #: Exactly two hypotheses (explicitly tentative, never asserted causes).
    hypotheses: list[NonEmptyText] = Field(min_length=HYPOTHESES_COUNT, max_length=HYPOTHESES_COUNT)

    #: One small experiment for the next 7 days; states the horizon explicitly.
    experiment_7d: str = Field(min_length=1)

    #: Mandatory limitation; must equal :data:`COACH_DISCLAIMER` exactly.
    disclaimer: str

    @field_validator("experiment_7d")
    @classmethod
    def _experiment_states_the_horizon(cls, value: str) -> str:
        if str(EXPERIMENT_HORIZON_DAYS) not in value:
            raise ValueError(
                "experiment_7d must state its horizon explicitly "
                f"(the digit {EXPERIMENT_HORIZON_DAYS})"
            )
        return value

    @field_validator("disclaimer")
    @classmethod
    def _disclaimer_is_the_fixed_constant(cls, value: str) -> str:
        if value != COACH_DISCLAIMER:
            raise ValueError("disclaimer must equal COACH_DISCLAIMER verbatim")
        return value
