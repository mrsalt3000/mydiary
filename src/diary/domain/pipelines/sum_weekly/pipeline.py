"""Weekly-summary pipeline (layer ``weekly_summary``; task T06).

Responsibilities
================

* Take **the daily summaries of one ISO week** (selected by
  :mod:`diary.domain.pipelines.sum_weekly.week_input`) and build the
  :class:`LLMRequest` from the versioned prompt template
  ``prompts/sum_weekly/v1.md`` plus those summaries, in day order. The input is
  the ``DailySummary`` contract of the layer ``daily_summary`` — never raw
  messages and never a weekly export (T06 DoD: "только merge дневных саммари,
  не raw").
* Validate the completion against the T00 contract
  :class:`~diary.domain.contracts.WeeklySummary` through the shared strict
  bridge (:mod:`diary.domain.llm.structured`). The completion must be one JSON
  object with exactly the contract's fields; anything else is an
  :class:`~diary.domain.llm.errors.LLMStructuredOutputError` — a failed run
  with no stored artifact (AGENTS.md §15).
* Fill the **envelope fields** of the contract from the backend, never from the
  model (ADR-007 §7): ``week_id`` is the ISO week the run is anchored to and
  ``daily_ids`` are the days of the summaries actually fed to the model. The
  bridge overwrites those keys with the backend values, so a completion that
  reports its own week or day list cannot influence the stored record.
* Keep the **context budget** and split the input when it does not fit
  (``docs/tasklist-phase2.md`` §9.11, rule 4 of the T06 brief). The request
  content must fit ``NUM_CTX - profile.max_tokens`` tokens, estimated with the
  eval-contract constant :data:`~diary.domain.contracts.golden.CHARS_PER_TOKEN_EST`
  (ADR-007 §9). :meth:`SumWeeklyPipeline.chunk_summaries` partitions the days
  into ordered, non-overlapping chunks that each fit; no day can be dropped or
  repeated, and the union of the chunks is always exactly the input. When the
  input needs more than one chunk, the partial rollups are reconciled by a
  second kind of call (:meth:`SumWeeklyPipeline.build_merge_request`) into one
  coherent ``rollup`` — not by concatenating strings.

Storage is not this module's business: :mod:`.service` hands the validated
summary to the append-only result store of task T08 under the ``week_id``
anchor. The layer renders no file: its result is stored inline as validated
JSON (ADR-008 §1), so ``output_ext`` exists only for registry consistency.

Privacy: the day summaries are in-memory request/contract content only; they are
never logged, and every error summary is static (AGENTS.md §7-8).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from diary.core.models_config import OpenAICompatibleProfile
from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import DailySummary, WeeklySummary
from diary.domain.contracts.golden import estimate_tokens
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.llm.structured import parse_contract_response
from diary.domain.pipelines.base import PipelinePermanentError, SecondFamilyLayer
from diary.domain.pipelines.sum_weekly.week_input import WeeklyInputError

#: Versioned prompt template of this layer (``prompts/sum_weekly/v1.md``).
PROMPT_VERSION = "v1"
#: Prompt template location: ``prompts/sum_weekly/v1.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "sum_weekly" / f"{PROMPT_VERSION}.md"
#: Phase-2 profile field of ``configs/models.yaml`` this layer runs under.
PROFILE_NAME = "diary_sum"
#: Line that introduces the week's daily summaries inside a rollup request.
SUMMARIES_HEADER = "Дневные сводки недели (JSON, daily_summary[] в порядке дней недели):"
#: Line that introduces the partial rollups inside a merge request.
MERGE_HEADER = "Частичные сводки недели (JSON, в порядке дней недели):"

#: Context window of the ``diary-sum`` tag, in tokens. ``num_ctx`` is a
#: property of the created model (``ollama/Modelfile.sum``), not a profile
#: field: the OpenAI-compatible API does not accept it (ADR-007 §9).
NUM_CTX = 8192

#: Static, non-sensitive error summaries (AGENTS.md §8).
_PROMPT_MISSING_MESSAGE = "The sum_weekly prompt template is unavailable."
_EMPTY_WEEK_MESSAGE = (
    "The week has no daily summaries to roll up; a weekly summary must state at least one day."
)
_SUMMARY_TOO_LARGE_MESSAGE = (
    "A single daily summary does not fit the context budget of the weekly layer."
)
_MERGE_TOO_LARGE_MESSAGE = (
    "The partial weekly rollups do not fit the context budget; the week cannot be merged."
)


class SumWeeklyPipeline(SecondFamilyLayer[WeeklySummary]):
    """The T06 weekly-summary layer (second anchor family, one ISO week per run)."""

    pipeline_type = PipelineType.weekly_summary
    layer = ArtifactLayer.weekly_summary
    output_ext = ".md"
    profile_name = PROFILE_NAME
    result_schema = WeeklySummary

    def load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    @staticmethod
    def _payload(summaries: Sequence[DailySummary]) -> str:
        """The day summaries as the JSON array of ``daily_summary`` objects."""
        return json.dumps(
            [summary.model_dump(mode="json") for summary in summaries],
            ensure_ascii=False,
        )

    @staticmethod
    def _compose(prompt: str, header: str, payload: str) -> str:
        """One request content: prompt, the input header and the JSON payload."""
        return f"{prompt}\n\n{header}\n{payload}"

    def build_request(self, *, summaries: Sequence[DailySummary]) -> LLMRequest:
        """The rollup request: the prompt and exactly the week's day summaries.

        The summaries are handed over as the JSON array of ``daily_summary``
        objects in day order — the layer's declared input contract, mapped 1:1
        from the ``daily_summary`` results of the week. A week without a single
        day summary cannot produce a grounded rollup, so it fails loudly
        instead of asking the model to invent the week.
        """
        if not summaries:
            raise WeeklyInputError(_EMPTY_WEEK_MESSAGE)
        content = self._compose(self.load_prompt(), SUMMARIES_HEADER, self._payload(summaries))
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    def build_merge_request(self, *, partials: Sequence[WeeklySummary]) -> LLMRequest:
        """The merge request: reconcile already-produced partial rollups.

        Each partial carries the days it covered and its own rollup text; the
        request asks the model to produce one coherent week text from them, in
        day order, without adding anything (mode B of the prompt). The model
        answers with the same single-field shape, so a partial and a final
        rollup go through the same contract validation.
        """
        if not partials:
            raise WeeklyInputError(_EMPTY_WEEK_MESSAGE)
        payload = json.dumps(
            [
                {"daily_ids": list(partial.daily_ids), "rollup": partial.rollup}
                for partial in partials
            ],
            ensure_ascii=False,
        )
        content = self._compose(self.load_prompt(), MERGE_HEADER, payload)
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    # ------------------------------------------------------------------
    # Context budget and chunking
    # ------------------------------------------------------------------

    @staticmethod
    def input_budget_tokens(profile: OpenAICompatibleProfile) -> int:
        """Tokens available to the request content of one call.

        The whole content (prompt + input) shares the model's context window
        with the generation cap: what is left for the input is ``num_ctx``
        minus the profile's ``max_tokens``. ``num_ctx`` lives in the Modelfile
        (ADR-007 §9), the cap in the profile; a cap that leaves no room yields a
        non-positive budget, and the chunker then fails loudly instead of
        sending an oversized request.
        """
        return NUM_CTX - profile.max_tokens

    def request_fits(self, request: LLMRequest, *, profile: OpenAICompatibleProfile) -> bool:
        """Whether one built request fits the input budget of ``profile``."""
        return estimate_tokens(len(request.messages[0].content)) <= self.input_budget_tokens(
            profile
        )

    def require_merge_fits(self, request: LLMRequest, *, profile: OpenAICompatibleProfile) -> None:
        """Fail loudly when the merge request does not fit the context budget.

        The merge input is bounded by the chunk rollups, which are themselves
        capped by ``max_tokens``; a week that produces more partial text than
        one window can hold fails loudly rather than being silently truncated.
        """
        if not self.request_fits(request, profile=profile):
            raise PipelinePermanentError(_MERGE_TOO_LARGE_MESSAGE)

    def _summaries_fit(
        self,
        prompt: str,
        summaries: Sequence[DailySummary],
        profile: OpenAICompatibleProfile,
    ) -> bool:
        """Whether a rollup request with exactly ``summaries`` fits the budget."""
        content = self._compose(prompt, SUMMARIES_HEADER, self._payload(summaries))
        return estimate_tokens(len(content)) <= self.input_budget_tokens(profile)

    def chunk_summaries(
        self,
        summaries: Sequence[DailySummary],
        *,
        profile: OpenAICompatibleProfile,
    ) -> list[list[DailySummary]]:
        """Partition the week's days into ordered requests that each fit.

        The partition is greedy and deterministic: days are taken in the given
        (day) order and a chunk is closed only when the next day would overflow
        the context budget of ``profile``. The union of the returned chunks is
        exactly ``summaries`` — every day appears in exactly one chunk, in
        order, so no day can be lost or duplicated by chunking. A single day
        that does not fit on its own cannot be split without splitting a day
        (which is what the "no lost day" rule forbids), so that fails loudly.
        """
        if not summaries:
            raise WeeklyInputError(_EMPTY_WEEK_MESSAGE)
        prompt = self.load_prompt()
        chunks: list[list[DailySummary]] = []
        current: list[DailySummary] = []
        for summary in summaries:
            if self._summaries_fit(prompt, [*current, summary], profile):
                current.append(summary)
                continue
            if not current:
                raise PipelinePermanentError(_SUMMARY_TOO_LARGE_MESSAGE)
            chunks.append(current)
            current = [summary]
            if not self._summaries_fit(prompt, current, profile):
                raise PipelinePermanentError(_SUMMARY_TOO_LARGE_MESSAGE)
        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def parse_result(
        self,
        response: LLMResponse,
        *,
        week_id: str,
        daily_ids: Sequence[str],
    ) -> WeeklySummary:
        """Validate one completion into the :class:`WeeklySummary` contract.

        The contract's envelope fields are supplied by the backend: ``week_id``
        is the ISO week of the run and ``daily_ids`` are the days of the
        summaries the run actually fed (all of them for the final rollup, the
        chunk's days for a partial one). The model does not have to echo them,
        and a value it does report cannot change the stored anchor or day list.
        """
        envelope: dict[str, object] = {"week_id": week_id, "daily_ids": list(daily_ids)}
        return parse_contract_response(response, WeeklySummary, envelope=envelope)


__all__ = [
    "MERGE_HEADER",
    "NUM_CTX",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SUMMARIES_HEADER",
    "SumWeeklyPipeline",
]
