"""Daily-summary pipeline (layer ``daily_summary``; task T05).

Responsibilities
================

* Take **the messages of one local calendar day** as ``raw_entry[]`` (selected
  by :mod:`diary.domain.pipelines.sum_daily.day_input`) and build exactly one
  :class:`LLMRequest` from the versioned prompt template
  ``prompts/sum_daily/v1.md`` plus that day's entries, in chronological order.
  The entries are read only: nothing here rewrites a message, its audio or its
  raw transcript.
* Validate the completion against the T00 contract
  :class:`~diary.domain.contracts.DailySummary` through the shared strict
  bridge (:mod:`diary.domain.llm.structured`). The completion must be one JSON
  object with exactly the contract's fields; anything else is an
  :class:`~diary.domain.llm.errors.LLMStructuredOutputError` — a failed run
  with no stored artifact (AGENTS.md §15).
* Fill the **anchor field** of the contract from the backend, never from the
  model (ADR-007 §7): ``date`` is the local day the entries were selected for,
  so the stored summary can never claim a different day than its anchor, even
  if the model tries to (the bridge overwrites the key with the backend value).

Definition of done (``docs/tasklist-phase2.md`` §6, T05): the result is strictly
the four content fields of the contract — ``facts``, ``themes``, ``energy``,
``open_questions`` — with no advice, morality, evaluation or speculation, and
``energy`` is ``null`` unless the source states it (ADR-007 §4). The prompt
carries those rules; the contract's ``extra="forbid"`` and ``min_length=1``
reject an answer that breaks the shape.

Storage is not this module's business: :mod:`.service` hands the validated
summary to the append-only result store of task T08 under the
``local_date`` + ``local_tz`` anchor. The layer renders no file: its result is
stored inline as validated JSON (ADR-008 §1), so ``output_ext`` exists only for
registry consistency.

Privacy: the day's texts are in-memory request/contract content only; they are
never logged, and every error summary is static (AGENTS.md §7-8).
"""

from __future__ import annotations

import json
from datetime import date

from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import DailySummary, RawEntry
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.llm.structured import parse_contract_response
from diary.domain.pipelines.base import PipelinePermanentError, SecondFamilyLayer
from diary.domain.pipelines.sum_daily.day_input import DayInputError

#: Versioned prompt template of this layer (``prompts/sum_daily/v1.md``).
PROMPT_VERSION = "v1"
#: Prompt template location: ``prompts/sum_daily/v1.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "sum_daily" / f"{PROMPT_VERSION}.md"
#: Phase-2 profile field of ``configs/models.yaml`` this layer runs under.
PROFILE_NAME = "diary_sum"
#: Line that introduces the day's entries inside a request.
ENTRIES_HEADER = "Записи дня (JSON, raw_entry[] в хронологическом порядке):"

#: Static, non-sensitive error summary (AGENTS.md §8).
_PROMPT_MISSING_MESSAGE = "The sum_daily prompt template is unavailable."
_EMPTY_DAY_MESSAGE = (
    "The day has no messages to summarize; a day summary must state at least one fact."
)


class SumDailyPipeline(SecondFamilyLayer[DailySummary]):
    """The T05 daily-summary layer (second anchor family, one local day per run)."""

    pipeline_type = PipelineType.daily_summary
    layer = ArtifactLayer.daily_summary
    output_ext = ".md"
    profile_name = PROFILE_NAME
    result_schema = DailySummary

    def load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    def build_request(self, *, entries: list[RawEntry]) -> LLMRequest:
        """One request containing the prompt and exactly this day's entries.

        The entries are handed over as the JSON array of ``raw_entry`` objects
        in chronological order — the layer's declared input contract, mapped
        1:1 from the source rows by :mod:`.day_input`. A day without a single
        message cannot produce a grounded summary (``facts`` is never empty),
        so it fails loudly instead of asking the model to invent facts.
        """
        if not entries:
            raise DayInputError(_EMPTY_DAY_MESSAGE)
        prompt = self.load_prompt()
        payload = json.dumps(
            [entry.model_dump(mode="json") for entry in entries],
            ensure_ascii=False,
        )
        content = f"{prompt}\n\n{ENTRIES_HEADER}\n{payload}"
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    def parse_result(self, response: LLMResponse, *, local_date: date) -> DailySummary:
        """Validate one completion into the :class:`DailySummary` contract.

        The contract's anchor field ``date`` is supplied by the backend from the
        day the entries were selected for: the model does not have to echo it,
        and a value it does report cannot change the stored day.
        """
        envelope: dict[str, object] = {"date": local_date.isoformat()}
        return parse_contract_response(response, DailySummary, envelope=envelope)


__all__ = [
    "ENTRIES_HEADER",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SumDailyPipeline",
]
