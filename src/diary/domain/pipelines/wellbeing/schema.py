"""Strict Pydantic schema of the wellbeing structured output (Task 15).

The model reflects *only* the normalized document (layer 2) it was sent and
returns four fixed sections (tasklist Task 15, architecture brief §10.3):

- ``observed_facts`` — what the records literally say, one item per dated
  diary entry. Every item MUST begin with the entry date ``[YYYY-MM-DD]`` so
  each fact is anchored to a concrete source record (enforced by a validator
  below — a fact without a date anchor fails the run);
- ``cautious_interpretation`` — possible readings of the week's patterns,
  explicitly hedged and only over the observed facts;
- ``recommendations`` — one to three low-risk, practical suggestions;
- ``confidence_and_limitations`` — how confident the reflection is and what
  limits it (single week, incomplete records, non-diagnostic nature).

Safety design decisions (ADR-006, AGENTS.md §16):

* the schema carries **no** safety-level field. The level that drives the
  report and the crisis flow is computed by the deterministic classifier
  (``diary.domain.pipelines.wellbeing.safety``), never by the model, so an LLM
  output can never downgrade or suppress an ``urgent`` level;
* the mandatory disclaimer and the urgent escalation content are code
  constants added by the renderer — the model cannot omit them;
* diagnostics, treatment plans, invented symptoms and causal claims are
  prohibited by the prompt invariants; the structural guarantees enforced here
  are the fact/date anchoring and the 1-3 recommendations bound.

Privacy contract: diary-derived text appears only inside the in-memory
structured model / the rendered layer-4 file — never in logs, error summaries
or persisted run metadata.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: A date anchor a fact item must begin with (``[YYYY-MM-DD] ...`` or
#: ``YYYY-MM-DD ...``).
_FACT_DATE_PREFIX_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}\]?")


class WellbeingResult(BaseModel):
    """The whole structured output of one wellbeing reflection response.

    A single validated object per run — the wellbeing pipeline always reflects
    over the complete normalized document in one pass (MVP; no chunking, see
    the pipeline module docstring). All list items are non-blank; extra keys
    are rejected (``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")

    #: Facts anchored to dated diary entries; at least one is required because
    #: the pipeline refuses to analyze a document without entries.
    observed_facts: list[str] = Field(min_length=1)
    #: Hedged interpretations; may be empty when there is nothing to interpret.
    cautious_interpretation: list[str] = Field(default_factory=list)
    #: One to three low-risk practical recommendations (tasklist invariant).
    recommendations: list[str] = Field(min_length=1, max_length=3)
    #: Confidence and limitations; never empty (the model must state limits).
    confidence_and_limitations: list[str] = Field(min_length=1)

    @field_validator(
        "observed_facts",
        "cautious_interpretation",
        "recommendations",
        "confidence_and_limitations",
    )
    @classmethod
    def _strip_and_reject_blank_items(cls, values: list[str]) -> list[str]:
        """Trim every item and reject blank/whitespace-only list entries."""
        cleaned: list[str] = []
        for item in values:
            stripped = item.strip()
            if not stripped:
                raise ValueError("wellbeing list sections must not contain blank items.")
            cleaned.append(stripped)
        return cleaned

    @model_validator(mode="after")
    def _facts_require_date_anchors(self) -> "WellbeingResult":
        """Every observed fact must begin with an entry date ``[YYYY-MM-DD]``.

        The date anchor keeps facts tied to concrete source records and is the
        structural half of "result explicitly grounded in observed diary
        facts": a fact the model cannot anchor to a record fails the run
        instead of being written into the report.
        """
        for fact in self.observed_facts:
            if _FACT_DATE_PREFIX_RE.match(fact) is None:
                raise ValueError(
                    "observed_facts items must begin with the record date "
                    "(e.g. '[2026-09-07] ...')."
                )
        return self


__all__ = ["WellbeingResult"]
