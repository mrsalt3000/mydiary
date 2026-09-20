"""Coaching-note pipeline — layer ``coach_note`` (phase 2, task T07).

Responsibilities
================

* Take **one weekly summary** (the ``WeeklySummary`` contract of the layer
  ``weekly_summary``, selected by :mod:`.coach_input`) and build exactly one
  :class:`LLMRequest` from the versioned prompt template
  ``prompts/coach_note/v2.md`` plus that summary as JSON. The input is the
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

from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import COACH_DISCLAIMER, CoachNote, WeeklySummary
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.llm.structured import parse_contract_response
from diary.domain.pipelines.base import PipelinePermanentError, SecondFamilyLayer

#: Versioned prompt template of this layer (``prompts/coach_note/v2.md``).
PROMPT_VERSION = "v3"
#: Prompt template location: ``prompts/coach_note/v2.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "coach_note" / f"{PROMPT_VERSION}.md"
#: Phase-2 profile field of ``configs/models.yaml`` this layer runs under.
PROFILE_NAME = "diary_coach"
#: Line that introduces the weekly summary inside the request.
SUMMARY_HEADER = "Недельная сводка (JSON, weekly_summary):"

#: Static, non-sensitive error summary (AGENTS.md §8).
_PROMPT_MISSING_MESSAGE = "The coach_note prompt template is unavailable."


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
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SUMMARY_HEADER",
    "CoachNotePipeline",
]
