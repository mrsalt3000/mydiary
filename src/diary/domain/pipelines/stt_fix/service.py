"""Run one STT correction and store it (task T04).

:func:`correct_stt` is the run itself: build the request for one
:class:`~diary.domain.contracts.RawEntry`, call the provider, validate the
completion into the contract and append the result to the phase-2 store under
the ``raw_id`` anchor (task T08). :func:`run_stt_fix` is the convenience entry
point that resolves the optional ``diary_stt`` profile and the provider from the
configuration — the shape task T09's orchestrator needs for a ``stt_corrected``
job.

Properties that matter for the T04 definition of done:

* the **input is never written**: the entry is read only, and the only write is
  ``save_layer_result`` into ``artifacts`` (``messages``, ``audio_files`` and
  ``transcriptions`` are not touched);
* an invalid completion raises
  :class:`~diary.domain.llm.errors.LLMStructuredOutputError` **before** any
  result is stored, so a failed run leaves no partial artifact;
* the result is stored inline as the validated contract — there is no rendered
  file for this layer, so ``path``/``sha256`` stay ``None``;
* the caller owns the transaction (``save_layer_result`` flushes but never
  commits), so the result row and the processing run it belongs to are
  committed atomically by the orchestrator.

Privacy: diary text exists only inside the in-memory request and the validated
contract; nothing here logs it (AGENTS.md §7-8).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.db.models import Artifact, ArtifactLayer
from diary.db.results import save_layer_result
from diary.domain.contracts import RawEntry, SttCorrection
from diary.domain.llm.providers import LLMProvider, create_llm_provider
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines.stt_fix.pipeline import PROFILE_NAME, SttFixPipeline


@dataclass(frozen=True)
class SttFixOutcome:
    """The validated correction and the appended result row of one run."""

    correction: SttCorrection
    artifact: Artifact


def correct_stt(
    session: Session,
    *,
    entry: RawEntry,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> SttFixOutcome:
    """Correct one entry with ``provider`` and append the result (no commit).

    ``profile`` is the resolved phase-2 profile whose ``model`` becomes the
    envelope value of the stored correction; the provider is injected so tests
    (and the GPU-arbitrated orchestrator of T09) control the call. The result is
    anchored to ``entry.id`` — the message the correction belongs to — and the
    version of the (message, layer) pair is computed by the store: a repeated
    run appends ``v+1`` and keeps the previous result intact (ADR-005).
    """
    pipeline = SttFixPipeline()
    request = pipeline.build_request(entry=entry)
    response = provider.complete(request)
    correction = pipeline.parse_result(response, entry=entry, profile=profile)
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=entry.id,
        result=correction,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )
    return SttFixOutcome(correction=correction, artifact=artifact)


def run_stt_fix(
    session: Session,
    *,
    entry: RawEntry,
    models_config: ModelsConfig | None = None,
    provider: LLMProvider | None = None,
    processing_run_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
) -> SttFixOutcome:
    """Resolve the ``diary_stt`` profile/provider and run one correction.

    An absent ``diary_stt`` profile means the layer is switched off and raises
    :class:`~diary.domain.llm.errors.LLMProviderUnavailableError` (never
    retryable) before any request is sent. ``provider`` overrides the factory —
    tests inject a fake, and T09 may pass its own instance.
    """
    config = models_config if models_config is not None else load_models_config()
    profile = resolve_phase2_profile(config, PROFILE_NAME)
    active_provider = provider if provider is not None else create_llm_provider(profile)
    return correct_stt(
        session,
        entry=entry,
        profile=profile,
        provider=active_provider,
        processing_run_id=processing_run_id,
        created_at=created_at,
    )


__all__ = ["SttFixOutcome", "correct_stt", "run_stt_fix"]
