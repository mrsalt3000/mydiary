"""Contract ``stt_corrected`` (phase-2 layer, task T00).

Fixes the STT-correction contract of the phase-2 family (variant B; ADR-007)::

    stt_corrected: raw_id, text, uncertain[], model, prompt_ver

Purpose: turn one raw ``raw_entry`` (a voice message with raw STT text, or a
text message with obvious typos) into a corrected text **without inventing
content**. The layer is per message, not per week, so it is anchored to the
message identity (``raw_id``) that it was produced from.

Decision — doubt is expressed only through ``uncertain[]``:

- ``text`` carries the corrected phrase **with no marker of any kind inside
  it**. The literal ``[?]`` is deliberately *not* part of this contract (the
  original brief sketched "doubt → ``[?]``"; that was replaced by this
  structured field — see ADR-007 and the T04 definition of done in
  ``docs/tasklist-phase2.md``).
- ``uncertain`` is a list of **verbatim substrings of ``text``** — the exact
  fragments the correction could not resolve. A fragment that is not a
  substring of ``text`` is a validation error, so the two fields can never
  drift apart, and a renderer can highlight the raw text without re-parsing
  a marker.

Decision — ``model`` and ``prompt_ver`` are *envelope* fields:

the backend fills them in from the processing run (the resolved profile model
and the ``prompts/<task>/v<N>.md`` version, AGENTS.md §15), **never** the LLM:
a model cannot report its own identity or prompt version reliably. They are
required on the record so that every persisted correction is traceable
(AGENTS.md §6); no separate copy of this data is created in
``processing_runs.{model_name,prompt_version}`` — the pair (contract record,
run row) stays the single place for each.

``prompt_ver`` follows the ``prompts/README.md`` convention ``v<N>``.
"""

from __future__ import annotations

import uuid
from typing import ClassVar

from pydantic import Field, field_validator, model_validator

from diary.domain.contracts._base import ContractModel

#: Accepted form of the prompt version: ``v1``, ``v2``, ... (prompts/README.md).
PROMPT_VER_PATTERN = r"^v\d+$"


class SttCorrection(ContractModel):
    """One corrected message plus the fragments that stayed uncertain."""

    contract_name: ClassVar[str] = "stt_corrected"

    #: Identity of the corrected source message (``RawEntry.id``).
    raw_id: uuid.UUID

    #: Corrected text; no doubt marker is embedded in it (see ``uncertain``).
    text: str = Field(min_length=1)

    #: Verbatim substrings of ``text`` that the correction could not resolve.
    #: Uncapped on purpose: a fully uncertain message may list several pieces;
    #: an unambiguous one lists none.
    uncertain: list[str] = Field(default_factory=list)

    #: Envelope: model identifier that produced the correction (filled by the
    #: backend from the resolved profile, never read from LLM output).
    model: str = Field(min_length=1)

    #: Envelope: prompt version ``v<N>`` used for the correction (filled by the
    #: backend, never read from LLM output).
    prompt_ver: str = Field(pattern=PROMPT_VER_PATTERN)

    @field_validator("uncertain")
    @classmethod
    def _fragments_are_not_empty(cls, value: list[str]) -> list[str]:
        for fragment in value:
            if not fragment:
                raise ValueError("uncertain entries must not be empty strings")
        return value

    @model_validator(mode="after")
    def _fragments_are_verbatim_substrings(self) -> SttCorrection:
        for fragment in self.uncertain:
            if fragment not in self.text:
                raise ValueError(
                    f"uncertain entry {fragment!r} is not a verbatim substring of text"
                )
        return self
