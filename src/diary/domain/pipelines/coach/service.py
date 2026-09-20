"""Run one coaching note and store it (task T07).

:func:`coach_note` is the run itself: take one weekly summary
(:func:`~diary.domain.pipelines.coach.coach_input.collect_week_summary`), build
the request, call the provider, validate the completion into
:class:`~diary.domain.contracts.CoachNote` and append the result to the
phase-2 store under the anchor ``summary_id`` = **the upstream
``weekly_summary`` artifact id** (task T08; ADR-008 §5). :func:`run_coach_note`
is the convenience entry point that resolves the optional ``diary_coach``
profile and the provider from the configuration — the shape task T09's
orchestrator needs for a ``coach_note`` job.

Two ``summary_id`` values, one name
===================================

The run produces both of them and they are **not** the same value:

* storage anchor ``artifacts.summary_id`` (UUID) = the **artifact id** of the
  input weekly summary (``coach_input.artifact.id``);
* contract field ``CoachNote.summary_id`` (string) = the **week id** of that
  summary (``coach_input.week_id``, e.g. ``2026-W38``), which
  :meth:`~diary.domain.pipelines.coach.pipeline.CoachNotePipeline.parse_result`
  fills from the backend.

Both come from the same already validated :class:`CoachInput`, so they cannot
disagree with each other or with the stored row; the model supplies neither.

Properties that matter for the T07 definition of done:

* **only an explicit call runs the layer**: nothing here is triggered by an
  import, a job handler, a route or another layer. The module registers no
  handler and schedules nothing — the caller invokes :func:`coach_note` /
  :func:`run_coach_note` itself;
* **one weekly summary per run**, read only; the layer's single write is
  ``save_layer_result`` into ``artifacts``;
* a week without a weekly summary, or a summary of another week, fails loudly
  before any request is sent;
* an invalid completion raises
  :class:`~diary.domain.llm.errors.LLMStructuredOutputError` **before** any
  result is stored, so a failed run leaves no partial artifact;
* the stored result is the validated contract — 3 observations, 2 hypotheses,
  one 7-day experiment and the backend's ``summary_id``/``disclaimer`` — and
  nothing else;
* the version of the (upstream artifact, layer) pair is computed by the store:
  a repeated run appends ``v+1`` and keeps the previous result intact
  (ADR-005);
* the caller owns the transaction (``save_layer_result`` flushes but never
  commits), so the result row and the processing run it belongs to are
  committed atomically by the orchestrator.

Privacy: the weekly summary exists only inside the in-memory request and the
validated contract; nothing here logs it (AGENTS.md §7-8).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, load_models_config
from diary.db.models import Artifact, ArtifactLayer
from diary.db.results import save_layer_result
from diary.domain.contracts import CoachNote
from diary.domain.llm.providers import LLMProvider, create_llm_provider
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines.coach.coach_input import (
    CoachInput,
    collect_week_summary,
    ensure_input_matches_week,
)
from diary.domain.pipelines.coach.pipeline import PROFILE_NAME, CoachNotePipeline
from diary.domain.pipelines.quality import generate_with_invariants


@dataclass(frozen=True)
class CoachNoteOutcome:
    """The validated coaching note and the appended result row of one run."""

    note: CoachNote
    artifact: Artifact


def coach_note(
    session: Session,
    *,
    week_id: str,
    provider: LLMProvider,
    coach_input: CoachInput | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> CoachNoteOutcome:
    """Build one coaching note for the ISO week ``week_id`` and append it (no commit).

    The input is the week's newest stored ``weekly_summary`` artifact, or the
    caller-supplied ``coach_input`` (the seam used by tests and by the T09
    orchestrator); a supplied input must belong to ``week_id``, otherwise the
    run fails loudly before any provider call. The provider is injected so
    tests (and the GPU-arbitrated orchestrator of T09) control the call; it
    carries this layer's ``diary_coach`` completion parameters.

    The result is anchored to the **artifact id** of the input weekly summary
    (``save_layer_result(..., summary_id=coach_input.artifact.id)``) while the
    contract's ``summary_id`` field carries that summary's ``week_id``. The
    version is computed by the store: a repeated run appends ``v+1`` and keeps
    the previous result (ADR-005). This function is the only way to run the
    layer — nothing calls it automatically.
    """
    run_input = (
        coach_input if coach_input is not None else collect_week_summary(session, week_id=week_id)
    )
    ensure_input_matches_week(run_input, week_id)
    pipeline = CoachNotePipeline()

    def generate(_attempt: int) -> CoachNote:
        request = pipeline.build_request(summary=run_input.summary)
        response = provider.complete(request)
        return pipeline.parse_result(response, week_id=run_input.week_id)

    note = generate_with_invariants(
        layer=ArtifactLayer.coach_note.value,
        source=[run_input.summary],
        generate=generate,
    )
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.coach_note,
        # Storage anchor: the ARTIFACT id of the upstream weekly summary.
        summary_id=run_input.artifact.id,
        result=note,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )
    return CoachNoteOutcome(note=note, artifact=artifact)


def run_coach_note(
    session: Session,
    *,
    week_id: str,
    models_config: ModelsConfig | None = None,
    provider: LLMProvider | None = None,
    coach_input: CoachInput | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> CoachNoteOutcome:
    """Resolve the ``diary_coach`` profile/provider and build one coaching note.

    An absent ``diary_coach`` profile means the layer is switched off and raises
    :class:`~diary.domain.llm.errors.LLMProviderUnavailableError` (never
    retryable) before any request is sent. ``provider`` overrides the factory —
    tests inject a fake, and T09 may pass its own instance. Calling this
    function **is** the explicit launch: no import, job handler or route calls
    it for the user.
    """
    config = models_config if models_config is not None else load_models_config()
    profile = resolve_phase2_profile(config, PROFILE_NAME)
    active_provider = provider if provider is not None else create_llm_provider(profile)
    return coach_note(
        session,
        week_id=week_id,
        provider=active_provider,
        coach_input=coach_input,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )


__all__ = ["CoachNoteOutcome", "coach_note", "run_coach_note"]
