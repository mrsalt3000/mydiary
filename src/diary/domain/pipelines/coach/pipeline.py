"""Coaching-note pipeline — layer ``coach_note`` (phase 2, task T07).

Responsibilities
================

* Take **one weekly summary** (the ``WeeklySummary`` contract of the layer
  ``weekly_summary``, selected by :mod:`.coach_input`) and build exactly one
  :class:`LLMRequest` from the versioned prompt template
  ``prompts/coach_note/v3.md`` plus that summary as JSON. The input is the
  grounded week rollup — never raw messages, never the day summaries directly
  and never a weekly export (T07 decision 1).
* Validate the completion against the T00 contract
  :class:`~diary.domain.contracts.CoachNote` through the shared strict bridge
  (:mod:`diary.domain.llm.structured`). The completion must be one JSON object
  with exactly the contract's fields; anything else is an
  :class:`~diary.domain.llm.errors.LLMStructuredOutputError` — a failed run with
  no stored artifact (AGENTS.md §15).
* Fill the **envelope fields** of the contract from the backend, never from the
  model (ADR-007 §7, T07 decision 4):

  - ``summary_id`` — the ``week_id`` of the upstream weekly summary. The model
    cannot name the source of its own note;
  - ``disclaimer`` — :data:`~diary.domain.contracts.COACH_DISCLAIMER` verbatim,
    the **layer-5** constant. It is deliberately not
    ``wellbeing.safety.DISCLAIMER``: layer 4 and layer 5 are separate layers
    with separate limitations (ADR-007 §8) and the two constants are never
    merged or re-used one for the other.

  The bridge overwrites both keys with the backend values, so a completion that
  reports a foreign week id or its own disclaimer wording cannot influence the
  stored note.

DoD (``docs/tasklist-phase2.md`` §6, T07): **3 observations + 2 hypotheses + 1
experiment**, a mandatory disclaimer, and an explicit launch only. The prompt
forbids asserted causes (hypotheses must read as assumptions), diagnoses,
treatment, morality, prescriptive advice and new facts; the contract's shape
(exactly 3 observations, exactly 2 hypotheses, the digit ``7`` in
``experiment_7d``, disclaimer equality) is what the layer enforces before any
write. The layer does not weaken or extend those validators.

Context budget (T20): the request must fit ``NUM_CTX - profile.max_tokens``,
estimated with the eval-contract constant
:data:`~diary.domain.contracts.golden.CHARS_PER_TOKEN_EST` — the same rule the
weekly layer applies. Unlike the weekly layer, the coach layer has nothing to
chunk: its input is one week rollup, so an oversized request is refused loudly
(:meth:`CoachNotePipeline.require_request_fits`) instead of being split or
silently truncated. The check needs the resolved profile, so the caller passes
it (see :mod:`.service`).

Definition of done is structural where it must be: the layer registers no
automatic trigger (registration is a caller decision, see :mod:`.registration`)
and no job handler exists for it, so it can only run when explicitly invoked.

Storage is not this module's business: :mod:`.service` hands the validated note
to the append-only result store of task T08 under the **artifact-id** anchor
``summary_id``. The layer renders no file: its result is stored inline as
validated JSON (ADR-008 §1), so ``output_ext`` exists only for registry
consistency.

Privacy: the weekly summary is in-memory request/contract content only; it is
never logged, and every error summary is static (AGENTS.md §7-8).
"""

from __future__ import annotations

import json

from diary.core.models_config import OpenAICompatibleProfile
from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import COACH_DISCLAIMER, CoachNote, WeeklySummary
from diary.domain.contracts.golden import estimate_tokens
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.llm.structured import parse_contract_response
from diary.domain.pipelines.base import PipelinePermanentError, SecondFamilyLayer

#: Versioned prompt template of this layer (``prompts/coach_note/v3.md``).
PROMPT_VERSION = "v3"
#: Prompt template location: ``prompts/coach_note/v3.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "coach_note" / f"{PROMPT_VERSION}.md"
#: Phase-2 profile field of ``configs/models.yaml`` this layer runs under.
PROFILE_NAME = "diary_coach"
#: Line that introduces the weekly summary inside the request.
SUMMARY_HEADER = "Недельная сводка (JSON, weekly_summary):"

#: Context window of the coach tag family, in tokens (``ollama/Modelfile.coach``).
#: ``num_ctx`` is a property of the created model, not a profile field: the
#: OpenAI-compatible API does not accept it (ADR-007 §9).
NUM_CTX = 8192

#: Static, non-sensitive error summary (AGENTS.md §8).
_PROMPT_MISSING_MESSAGE = "The coach_note prompt template is unavailable."
_REQUEST_TOO_LARGE_MESSAGE = (
    "The coaching-note request does not fit the context budget of the coach layer."
)


class CoachNotePipeline(SecondFamilyLayer[CoachNote]):
    """The T07 coaching-note layer (layer 5, one weekly summary per run).

    A separate layer from the layer-4 ``wellbeing`` pipeline: its own prompt,
    its own profile (``diary_coach``), its own disclaimer constant and its own
    anchor (an upstream ``weekly_summary`` artifact). It shares no code path
    with the wellbeing pipeline and does not import from it.
    """

    pipeline_type = PipelineType.coach_note
    layer = ArtifactLayer.coach_note
    output_ext = ".md"
    profile_name = PROFILE_NAME
    result_schema = CoachNote

    def load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    def build_request(self, *, summary: WeeklySummary) -> LLMRequest:
        """One request containing the prompt and exactly the weekly summary.

        The summary is handed over as the ``weekly_summary`` object of the T00
        contract (``week_id``/``daily_ids``/``rollup``) — the layer's declared
        input. The model is not given the raw messages, the day summaries or
        the weekly export; it may only work with what the grounded week rollup
        already states.
        """
        payload = json.dumps(summary.model_dump(mode="json"), ensure_ascii=False)
        content = f"{self.load_prompt()}\n\n{SUMMARY_HEADER}\n{payload}"
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    # ------------------------------------------------------------------
    # Context budget
    # ------------------------------------------------------------------

    @staticmethod
    def input_budget_tokens(profile: OpenAICompatibleProfile) -> int:
        """Tokens available to the request content of one call.

        The whole content (prompt and summary) shares the model's context window
        with the generation cap: what is left for the input is ``num_ctx`` minus
        the profile's ``max_tokens``. ``num_ctx`` lives in the Modelfile
        (ADR-007 §9), the cap in the profile; a cap that leaves no room yields a
        non-positive budget, and the caller then fails loudly instead of sending
        an oversized request.
        """
        return NUM_CTX - profile.max_tokens

    def request_fits(self, request: LLMRequest, *, profile: OpenAICompatibleProfile) -> bool:
        """Whether one built request fits the input budget of ``profile``."""
        return estimate_tokens(len(request.messages[0].content)) <= self.input_budget_tokens(
            profile
        )

    def require_request_fits(
        self, request: LLMRequest, *, profile: OpenAICompatibleProfile
    ) -> None:
        """Fail loudly when the note request does not fit the context budget.

        The coach layer has no chunking to fall back on: its input is exactly one
        week rollup, and a rollup too large for the window cannot be split
        without splitting the week. Refusing loudly keeps the failure visible and
        stores nothing, the same policy as ADR-012 — a silently truncated request
        would produce a note about a week the model never fully saw (T20).
        """
        if not self.request_fits(request, profile=profile):
            raise PipelinePermanentError(_REQUEST_TOO_LARGE_MESSAGE)

    def parse_result(self, response: LLMResponse, *, week_id: str) -> CoachNote:
        """Validate one completion into the :class:`CoachNote` contract.

        The contract's envelope fields are supplied by the backend:
        ``summary_id`` is the ``week_id`` of the upstream weekly summary of
        this run (not the artifact id — that one is the **storage** anchor, see
        :mod:`.coach_input`) and ``disclaimer`` is the layer-5
        :data:`~diary.domain.contracts.COACH_DISCLAIMER`. The model does not
        have to echo them, and any value it does report under those keys is
        replaced, so the stored note can never carry a foreign week id or a
        paraphrased limitation.
        """
        envelope: dict[str, object] = {
            "summary_id": week_id,
            "disclaimer": COACH_DISCLAIMER,
        }
        return parse_contract_response(response, CoachNote, envelope=envelope)


__all__ = [
    "NUM_CTX",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SUMMARY_HEADER",
    "CoachNotePipeline",
]
