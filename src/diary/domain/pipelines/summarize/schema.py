"""Strict Pydantic schema of the summarization structured output (Task 14).

The model summarizes *only* the normalized document (layer 2) it was sent and
returns a source-grounded weekly summary with five fixed sections (tasklist
Task 14, architecture brief §10.2):

- ``key_events``        — events that actually happened, as stated in the source;
- ``work_and_load``     — work/study/chores and load, when present in the source;
- ``wellbeing``         — wellbeing/emotions **only when explicitly supported
                          by the source**: ``explicit_support`` must be ``true``
                          exactly then, and ``points`` must be empty otherwise;
- ``supportive_factors``— factors that helped/supported, when present;
- ``open_questions``    — open questions/unknowns named in the source.

The wellbeing section is the one with an explicit "no explicit data" state:
the pipeline renders that state as a plain statement instead of inventing an
emotion reading of the week. Every field is strictly typed, extra keys are
rejected (``extra="forbid"``) and a partial (map) response uses the very same
schema as a full (single-pass) one, so the deterministic reduce step can merge
partial summaries of identical shape.

Privacy contract: diary-derived text appears only inside the in-memory
structured model / the rendered layer-3 file — never in logs, error summaries
or persisted run metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WellbeingSection(BaseModel):
    """Wellbeing/emotion statements of the week, gated on explicit source
    support.

    ``explicit_support`` must be ``true`` only when the normalized source
    itself states wellbeing or emotions (mood, sleep, energy, named feelings).
    When it is ``false`` the model may not attach any ``points``; this is
    enforced so the pipeline can never render an invented emotional reading.
    """

    model_config = ConfigDict(extra="forbid")

    explicit_support: bool
    points: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _points_require_explicit_support(self) -> "WellbeingSection":
        if not self.explicit_support and self.points:
            raise ValueError("wellbeing points require explicit support in the source text.")
        return self


class SummaryResult(BaseModel):
    """The whole structured output of one summarization response.

    A single-pass run returns one ``SummaryResult`` for the full document; a
    map-reduce run returns one ``SummaryResult`` per chunk (partial, same
    schema), which the pipeline merges deterministically into the final
    structure. Section lists keep the source order and may be empty when the
    source has nothing for a section.
    """

    model_config = ConfigDict(extra="forbid")

    key_events: list[str]
    work_and_load: list[str]
    wellbeing: WellbeingSection
    supportive_factors: list[str]
    open_questions: list[str]


__all__ = [
    "SummaryResult",
    "WellbeingSection",
]
