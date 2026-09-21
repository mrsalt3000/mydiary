"""Daily-summary pipeline (layer ``daily_summary``; task T05).

Responsibilities
================

* Take **the messages of one local calendar day** as ``raw_entry[]`` (selected
  by :mod:`diary.domain.pipelines.sum_daily.day_input`) and build the
  :class:`LLMRequest` from the versioned prompt template
  ``prompts/sum_daily/v1.md`` plus that day's entries, in chronological order.
  The entries are read only: nothing here rewrites a message, its audio or its
  raw transcript.
* Keep the request inside the model's context window (T23). A day that fits is
  **one** request, exactly as before; a day that does not fit is split by entries
  (:meth:`~SumDailyPipeline.chunk_entries`), each part is summarized with the same
  template, and the partial summaries are folded back into one day by a second
  versioned template (:meth:`~SumDailyPipeline.build_merge_request`). The budget
  is ``num_ctx - profile.max_tokens`` estimated with the eval contract's
  ``CHARS_PER_TOKEN_EST`` — the rule the weekly layer and the coach layer already
  use, so no layer can drift from the harness.
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
from collections.abc import Sequence
from datetime import date

from diary.core.models_config import OpenAICompatibleProfile
from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import DailySummary, RawEntry
from diary.domain.contracts.golden import estimate_tokens
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
#: Line that introduces the partial summaries inside a merge request.
PARTS_HEADER = "Части сводки этого дня (JSON, в порядке частей):"

#: Versioned prompt of the **merge** step (``prompts/sum_daily/v2.md``).
#:
#: The day is split by entries only when it does not fit one context window
#: (T23); each part then produces a partial day summary with the ordinary prompt
#: (``v1``, unchanged), and this template folds the parts back into one day.
#: The weekly layer works the same way (``sum_weekly`` v1 rollup + v2 merge).
MERGE_PROMPT_VERSION = "v2"
#: Merge prompt template location: ``prompts/sum_daily/v2.md``.
MERGE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "sum_daily" / f"{MERGE_PROMPT_VERSION}.md"

#: Context window of the day layer's model, in tokens.
#:
#: ``num_ctx`` is a property of the created model (``ollama/Modelfile.sum``, and
#: ``num_ctx 8192`` in its CPU twin), not a profile field — the same rule the
#: weekly layer and the coach layer document. A day that does not fit is split by
#: entries (T23); raising this number is a separate decision (T02).
NUM_CTX = 8192

#: Static, non-sensitive error summary (AGENTS.md §8).
_PROMPT_MISSING_MESSAGE = "The sum_daily prompt template is unavailable."
_EMPTY_DAY_MESSAGE = (
    "The day has no messages to summarize; a day summary must state at least one fact."
)
_MESSAGE_TOO_LARGE_MESSAGE = (
    "One message of the day does not fit the context window on its own; "
    "the day cannot be split further without splitting the message."
)
_MERGE_TOO_LARGE_MESSAGE = (
    "The partial day summaries do not fit one context window; the day cannot be folded back."
)
_MERGE_PROMPT_MISSING_MESSAGE = "The sum_daily merge prompt template is unavailable."
_EMPTY_PARTS_MESSAGE = "The day has no partial summaries to fold back."
_PART_NOT_OF_DAY_MESSAGE = "A partial summary belongs to another day than the one being folded."


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

    # ------------------------------------------------------------------
    # Context budget, chunking and folding back (T23)
    # ------------------------------------------------------------------

    @staticmethod
    def input_budget_tokens(profile: OpenAICompatibleProfile) -> int:
        """Tokens available to the request content of one call.

        The whole content (prompt + entries) shares the model's context window
        with the generation cap: what is left for the content is ``num_ctx``
        minus the profile's ``max_tokens``. The rule is the weekly layer's, and
        the estimate is the eval contract's ``CHARS_PER_TOKEN_EST``, so the two
        layers and the harness cannot drift apart. A cap that leaves no room
        yields a non-positive budget, and the chunker then fails loudly instead
        of sending an oversized request.
        """
        return NUM_CTX - profile.max_tokens

    def request_fits(self, request: LLMRequest, *, profile: OpenAICompatibleProfile) -> bool:
        """Whether one built request fits the input budget of ``profile``."""
        return estimate_tokens(len(request.messages[0].content)) <= self.input_budget_tokens(
            profile
        )

    def require_merge_fits(self, request: LLMRequest, *, profile: OpenAICompatibleProfile) -> None:
        """Fail loudly when the merge request does not fit the context budget.

        The merge input is bounded by the partial summaries, which are
        themselves capped by ``max_tokens``; a day that produces more partial
        text than one window can hold fails loudly rather than being silently
        truncated.
        """
        if not self.request_fits(request, profile=profile):
            raise PipelinePermanentError(_MERGE_TOO_LARGE_MESSAGE)

    def _entries_fit(
        self,
        prompt: str,
        entries: Sequence[RawEntry],
        profile: OpenAICompatibleProfile,
    ) -> bool:
        """Whether a day request with exactly ``entries`` fits the budget."""
        payload = json.dumps(
            [entry.model_dump(mode="json") for entry in entries],
            ensure_ascii=False,
        )
        content = f"{prompt}\n\n{ENTRIES_HEADER}\n{payload}"
        return estimate_tokens(len(content)) <= self.input_budget_tokens(profile)

    def chunk_entries(
        self,
        entries: Sequence[RawEntry],
        *,
        profile: OpenAICompatibleProfile,
    ) -> list[list[RawEntry]]:
        """Partition the day's entries into ordered parts that each fit.

        The partition is greedy and deterministic: entries are taken in
        chronological order and a part is closed only when the next entry would
        overflow the context budget of ``profile``. The union of the returned
        parts is exactly ``entries`` — every message appears in exactly one part,
        in order, so no message can be lost or duplicated by chunking. One
        message that does not fit on its own cannot be split without splitting
        the message itself, so that fails loudly. A day that fits one window is
        returned as a single part, i.e. the ordinary single-call behaviour is
        preserved exactly (T23: a day that fits is not touched).
        """
        if not entries:
            raise DayInputError(_EMPTY_DAY_MESSAGE)
        prompt = self.load_prompt()
        parts: list[list[RawEntry]] = []
        current: list[RawEntry] = []
        for entry in entries:
            if self._entries_fit(prompt, [*current, entry], profile):
                current.append(entry)
                continue
            if not current:
                raise PipelinePermanentError(_MESSAGE_TOO_LARGE_MESSAGE)
            parts.append(current)
            current = [entry]
            if not self._entries_fit(prompt, current, profile):
                raise PipelinePermanentError(_MESSAGE_TOO_LARGE_MESSAGE)
        if current:
            parts.append(current)
        return parts

    def load_merge_prompt(self) -> str:
        """The versioned merge template text (static, no diary content)."""
        try:
            return MERGE_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_MERGE_PROMPT_MISSING_MESSAGE) from exc

    def build_merge_request(
        self,
        parts: Sequence[DailySummary],
        *,
        local_date: date,
    ) -> LLMRequest:
        """One request folding the day's partial summaries back into one day.

        The parts are handed over as their four content fields in part order —
        the anchor ``date`` is the backend's, as everywhere else in this layer,
        and a summary of another day cannot be folded in: that would make the
        merged day a lie.
        """
        if not parts:
            raise PipelinePermanentError(_EMPTY_PARTS_MESSAGE)
        for part in parts:
            if part.date != local_date:
                raise PipelinePermanentError(_PART_NOT_OF_DAY_MESSAGE)
        prompt = self.load_merge_prompt()
        payload = json.dumps(
            [
                {
                    "facts": part.facts,
                    "themes": part.themes,
                    "energy": part.energy,
                    "open_questions": part.open_questions,
                }
                for part in parts
            ],
            ensure_ascii=False,
        )
        content = f"{prompt}\n\n{PARTS_HEADER}\n{payload}"
        return LLMRequest.from_prompt(content, prompt_version=MERGE_PROMPT_VERSION)


__all__ = [
    "ENTRIES_HEADER",
    "MERGE_PROMPT_PATH",
    "MERGE_PROMPT_VERSION",
    "NUM_CTX",
    "PARTS_HEADER",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SumDailyPipeline",
]
