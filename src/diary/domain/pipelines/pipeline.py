"""Phase-2 orchestrator: the second-family chain (task T09).

This module is the only place that drives **more than one** layer of the second
anchor family (ADR-007 §1; persistence shape ADR-008) in one sequence. The
phase-2 chain is::

    raw (messages / transcriptions)
      → stt_fix      (one run per message, anchor ``raw_id``)
      → sum_daily    (one run per local day, anchor ``local_date`` + ``local_tz``)
      → sum_weekly   (one run per ISO week, anchor ``week_id``)
      → coach        (layer 5, **explicit call only**, anchor ``summary_id``)

Entry points, one per unit of work (T09; the message unit is added by T11):

* :func:`run_day_chain` — one local calendar day: the day's messages through
  ``stt_fix`` and then the day through ``sum_daily``;
* :func:`run_week_chain` — one ISO week: the stored daily summaries through
  ``sum_weekly`` (optionally refreshing the week's days first, ``run_days``);
* :func:`run_coach_step` — **the only path to the coaching note** (T07 decision
  3: «запуск только явно»). No parameter of the day/week chain can reach it and
  nothing else in the worker calls it; see :func:`run_week_chain`'s note.

Task T11 adds one more unit of work, the smallest one: :func:`run_stt_fix_step`
corrects exactly one message. It is the unit behind a ``stt_corrected`` job
(the day chain is not a substitute — a single-message job must not also
summarize the day), and the day chain runs the same helper once per message, so
both paths produce identical steps.

Why a layer step is not just "call the service"
==============================================

Every step is executed under **its own** :class:`~diary.db.models.ProcessingRun`
and the run id is passed to the layer as ``processing_run_id``, which the layer
stores on the artifact. That is what closes the lineage loop
``artifact → run → source`` for the second family (ADR-007 §7,
``docs/data-lineage.md`` §7) and what the T04/T05/T07 acceptance found missing
while the parameter was optional. Per step the orchestrator:

1. resolves the layer through
   :func:`~diary.domain.pipelines.registry.get_second_family_pipeline` — the
   second family is **never** driven through the export-shaped
   :class:`~diary.domain.pipelines.base.Pipeline`/``run_pipeline`` path, and a
   layer that was never registered fails loudly instead of silently doing
   nothing;
2. records the run as ``running`` and **commits it before the LLM call**, so a
   crash mid-call leaves a visible running run rather than nothing;
3. calls the layer service with a **recording provider wrapper**: the wrapper
   is the only way the orchestrator can learn the token usage and request
   digests of the calls it triggered, and it keeps the layer's own code (and
   its API) untouched;
4. on success writes the provenance — ``profile_name`` (the configuration
   profile field, e.g. ``diary_stt``), ``model_name`` (the effective model of
   the request) and ``prompt_version`` (the version of the layer's prompt
   template) — and commits the run **together with** the artifact the layer
   flushed, so a stored result is never orphaned from its run;
5. on failure rolls the session back first (a step that raised cannot leave a
   partially written artifact behind — the layers already guarantee this, and
   the rollback makes it structural), records the failed run with a static,
   non-sensitive summary and **re-raises the original exception**, so a caller
   (or the worker's retry policy) still sees exactly the error that occurred:
   a permanent domain error stays permanent, a transient LLM error stays
   retryable.

An absent phase-2 profile means "this layer is switched off"
(``docs/tasklist-phase2.md`` §9.8): the step is **skipped**, no run is recorded
and the outcome names it in ``skipped`` with the reason ``profile_off``. A step
whose input is empty (a day without messages, a week without daily summaries, a
voice message that was never transcribed) is skipped with the reason
``no_input`` — never invented, never a fabricated artifact.

Isolation from the export views (ADR-008 §7)
===========================================

Runs of the second family are created with ``input_export_id = NULL`` (the
second family has no weekly export at all — ADR-007 §1) and therefore cannot be
returned by
:meth:`~diary.db.repositories.ProcessingRunRepository.list_by_input_export`,
which is the only way ``_export_runs_view``/``_export_management_view``
(``api/routes/pages.py``) and ``GET /api/v1/exports/{id}/runs`` reach runs.
``_processing_run_view`` — which takes the last artifact of a run and requires
a rendered file — is therefore unreachable for these runs by construction, not
by a check that could rot. The export-scoped artifact listings
(``ArtifactRepository.list_by_export*``) filter on ``export_id`` and never see
the inline (file-less) results of this chain either. The only artifact column
the orchestrator links back is ``input_artifact_id``, and only for the coach
step, whose input is exactly one upstream weekly-summary artifact.

Privacy (AGENTS.md §7-8)
========================

Nothing here logs. Run metadata carries identifiers and provenance only
(``input_ref`` is ``message:<uuid>``/``day:<YYYY-MM-DD>``/``week:<YYYY-Www>``),
request content enters metadata solely as an SHA-256 digest, and failure
summaries are ``<ErrorClass>: <static message>`` — never diary text, prompts or
LLM output.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Generic, TypeVar
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfig, OpenAICompatibleProfile, load_models_config
from diary.core.settings import get_settings
from diary.db.base import utc_now
from diary.db.models import Artifact, PipelineType, ProcessingRun, RunStatus
from diary.db.repositories import (
    MAX_ERROR_SUMMARY_LENGTH,
    MessageRepository,
    ProcessingRunRepository,
)
from diary.domain.contracts import (
    CoachNote,
    DailySummary,
    RawEntry,
    SttCorrection,
    WeeklySummary,
)
from diary.domain.llm.errors import LLMProviderUnavailableError
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage, request_input_sha256
from diary.domain.llm.providers import LLMProvider, create_llm_provider, resolve_completion_params
from diary.domain.llm.service import resolve_phase2_profile
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.base import PipelineInputError, PipelineNotRegisteredError
from diary.domain.pipelines.coach.coach_input import collect_week_summary
from diary.domain.pipelines.coach.pipeline import PROMPT_VERSION as COACH_PROMPT_VERSION
from diary.domain.pipelines.coach.service import coach_note
from diary.domain.pipelines.naming import PIPELINE_PROFILE_NAME
from diary.domain.pipelines.stt_fix.pipeline import PROMPT_VERSION as STT_FIX_PROMPT_VERSION
from diary.domain.pipelines.stt_fix.service import correct_stt
from diary.domain.pipelines.sum_daily.day_input import (
    collect_day_entries,
    local_date_of,
)
from diary.domain.pipelines.sum_daily.pipeline import PROMPT_VERSION as SUM_DAILY_PROMPT_VERSION
from diary.domain.pipelines.sum_daily.service import sum_daily
from diary.domain.pipelines.sum_weekly.pipeline import PROMPT_VERSION as SUM_WEEKLY_PROMPT_VERSION
from diary.domain.pipelines.sum_weekly.service import sum_weekly
from diary.domain.pipelines.sum_weekly.week_input import collect_week_summaries, week_days

__all__ = [
    "ORCHESTRATOR",
    "SKIP_NO_INPUT",
    "SKIP_PROFILE_OFF",
    "ChainStep",
    "CoachStepOutcome",
    "DayChainOutcome",
    "SkippedStep",
    "SttFixStepOutcome",
    "WeekChainOutcome",
    "run_coach_step",
    "run_day_chain",
    "run_stt_fix_step",
    "run_week_chain",
]

#: Marker written into ``processing_runs.metadata_json`` of every run created
#: here, so the runs of this chain are recognisable as such.
ORCHESTRATOR = "phase2_chain"

#: Skip reason: the optional phase-2 profile is absent, so the layer is off
#: (``docs/tasklist-phase2.md`` §9.8) — not an error.
SKIP_PROFILE_OFF = "profile_off"

#: Skip reason: the step has no input (empty text, a day without messages, a
#: week without daily summaries) — nothing is invented for it.
SKIP_NO_INPUT = "no_input"

#: Type of the validated contract a step produces (``DailySummary`` for the day
#: step, ``WeeklySummary`` for the week step, ``CoachNote`` for the coach step).
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class ChainStep:
    """One executed step of a chain run.

    Metadata and identifiers only — never diary content. ``input_ref`` is a
    safe reference to the step's input (``message:<uuid>``,
    ``day:<YYYY-MM-DD>`` or ``week:<YYYY-Www>``), ``run_id`` is the processing
    run that recorded the step and ``artifact_id`` the result row it produced
    (the artifact carries the same ``processing_run_id``).
    """

    layer: PipelineType
    run_id: uuid.UUID
    artifact_id: uuid.UUID | None
    profile_name: str
    model_name: str
    prompt_version: str
    input_ref: str


@dataclass(frozen=True)
class SkippedStep:
    """A step that was not executed, with the reason and the safe subject.

    ``reason`` is :data:`SKIP_PROFILE_OFF` (the layer is switched off) or
    :data:`SKIP_NO_INPUT` (the step has nothing to work on); ``subject`` names
    the would-be input the same way :attr:`ChainStep.input_ref` does.
    """

    layer: PipelineType
    reason: str
    subject: str


@dataclass(frozen=True)
class DayChainOutcome:
    """Result of :func:`run_day_chain` for one local calendar day.

    ``summary``/``summary_artifact_id`` are ``None`` when the day step did not
    run (its profile is off, or the day has no messages) — the reason is then in
    :attr:`skipped`.
    """

    local_date: date
    steps: tuple[ChainStep, ...]
    skipped: tuple[SkippedStep, ...]
    summary: DailySummary | None = None
    summary_artifact_id: uuid.UUID | None = None

    @property
    def layers(self) -> tuple[PipelineType, ...]:
        """The layers that actually ran, in order."""
        return tuple(step.layer for step in self.steps)


@dataclass(frozen=True)
class SttFixStepOutcome:
    """Result of :func:`run_stt_fix_step` — the correction of exactly one message.

    Exactly one of ``step`` (the run that stored the correction) and ``skipped``
    (the layer is switched off, or the message has no text yet) is set;
    ``correction`` is the validated contract when the step ran.
    """

    raw_id: uuid.UUID
    step: ChainStep | None = None
    skipped: SkippedStep | None = None
    correction: SttCorrection | None = None


@dataclass(frozen=True)
class WeekChainOutcome:
    """Result of :func:`run_week_chain` for one ISO week.

    ``summary``/``summary_artifact_id`` are ``None`` when the week step did not
    run (profile off, or no day summaries stored yet); ``steps`` then still
    contains the day steps when the call refreshed the week's days.
    """

    week_id: str
    steps: tuple[ChainStep, ...]
    skipped: tuple[SkippedStep, ...]
    summary: WeeklySummary | None = None
    summary_artifact_id: uuid.UUID | None = None

    @property
    def layers(self) -> tuple[PipelineType, ...]:
        """The layers that actually ran, in order."""
        return tuple(step.layer for step in self.steps)


@dataclass(frozen=True)
class CoachStepOutcome:
    """Result of :func:`run_coach_step` — the explicit layer-5 launch.

    Exactly one of ``step`` (the run that produced the note) and ``skipped``
    (the layer is switched off) is set; ``note`` is the validated contract when
    the step ran.
    """

    week_id: str
    step: ChainStep | None = None
    skipped: SkippedStep | None = None
    note: CoachNote | None = None


@dataclass(frozen=True)
class _StepProduct(Generic[PayloadT]):
    """What a step's ``perform`` callable hands back to the orchestrator."""

    #: The result row the layer appended (already flushed, not committed).
    artifact: Artifact
    #: The validated contract of the layer, when the caller needs it.
    payload: PayloadT | None = None
    #: The single upstream artifact this step consumed (the coach step only).
    input_artifact_id: uuid.UUID | None = None


#: One step: build the run's product from the provider and the run id.
_Perform = Callable[[LLMProvider, uuid.UUID], _StepProduct[PayloadT]]


class _RecordingProvider:
    """A provider wrapper that records the **successful** calls of one step.

    The layer services return their validated outcome, not the raw response, so
    the only way to record real provenance (token usage, request digests,
    effective model, number of calls for a chunked layer) without touching the
    accepted layer API is to wrap the provider the orchestrator injects. A call
    that raised is not recorded — the step fails and only the failure is
    recorded on the run.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.requests: list[LLMRequest] = []
        self.responses: list[LLMResponse] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete ``request`` and record it when the provider returned."""
        response = self._inner.complete(request)
        self.requests.append(request)
        self.responses.append(response)
        return response


# ---------------------------------------------------------------------------
# Profile / provider resolution
# ---------------------------------------------------------------------------


def _layer_profile(
    models_config: ModelsConfig, layer: PipelineType
) -> tuple[str, OpenAICompatibleProfile | None]:
    """The profile field name of ``layer`` and its profile, or ``None`` if off.

    The name comes from :data:`~diary.domain.pipelines.naming.PIPELINE_PROFILE_NAME`
    (the same map the registry validates a layer's own ``profile_name``
    against), and an absent profile is *not* an error: it means the layer is
    switched off (``docs/tasklist-phase2.md`` §9.8).
    """
    profile_name = PIPELINE_PROFILE_NAME[layer]
    try:
        return profile_name, resolve_phase2_profile(models_config, profile_name)
    except LLMProviderUnavailableError:
        return profile_name, None


def _provider_for(
    *,
    profile_name: str,
    profile: OpenAICompatibleProfile,
    providers: Mapping[str, LLMProvider] | None,
) -> LLMProvider:
    """The provider of one step: the injected one, or a fresh real one.

    ``providers`` is the test/orchestrator-injection seam keyed by profile field
    name. An injected mapping missing the layer's entry is a caller bug, so it
    fails loudly instead of silently contacting a real endpoint.
    """
    if providers is None:
        return create_llm_provider(profile)
    try:
        return providers[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"The injected providers mapping has no entry for profile {profile_name!r}."
        ) from exc


# ---------------------------------------------------------------------------
# Run lifecycle of one step
# ---------------------------------------------------------------------------


def _run_step(
    session: Session,
    *,
    layer: PipelineType,
    profile_name: str,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
    prompt_version: str,
    subject: str,
    perform: _Perform[PayloadT],
) -> tuple[ChainStep, _StepProduct[PayloadT]]:
    """Execute one layer step under its own processing run (see module docstring).

    The run is committed as ``running`` before the layer is called, and
    finalised (``succeeded`` + provenance) in the same transaction as the
    artifact the layer flushed. On failure the session is rolled back — so a
    step that raised leaves no partial artifact — the run is recorded as
    ``failed`` with a static summary, and the original exception is re-raised
    unchanged (a permanent domain error stays permanent, a transient LLM error
    stays retryable for the caller).
    """
    # The registry lookup is the guard that the second family is driven through
    # its own half of the registry (never through the export-shaped framework)
    # and that the layer the run names is actually implemented.
    registered = pipeline_registry.get_second_family_pipeline(layer)
    run = ProcessingRunRepository(session).create(
        pipeline_type=layer,
        status=RunStatus.running,
        metadata_json={
            "orchestrator": ORCHESTRATOR,
            "layer": layer.value,
            "input_ref": subject,
        },
    )
    run.started_at = utc_now()
    session.commit()
    run_id = run.id
    if registered is None:
        # Mirrors the export framework: the run exists, then fails permanently
        # with a clear summary (a layer that was never registered is a
        # misconfiguration, not an input problem).
        error = PipelineNotRegisteredError(
            f"The phase-2 layer {layer.value!r} is not registered; the orchestrator cannot run it."
        )
        _record_failed_run(session, run_id, error)
        raise error
    recorder = _RecordingProvider(provider)
    try:
        product = perform(recorder, run_id)
    except Exception as exc:
        _record_failed_run(session, run_id, exc)
        raise
    step = _finalize_run(
        session,
        run_id,
        layer=layer,
        profile_name=profile_name,
        profile=profile,
        prompt_version=prompt_version,
        subject=subject,
        product=product,
        recorder=recorder,
    )
    return step, product


def _finalize_run(
    session: Session,
    run_id: uuid.UUID,
    *,
    layer: PipelineType,
    profile_name: str,
    profile: OpenAICompatibleProfile,
    prompt_version: str,
    subject: str,
    product: _StepProduct[PayloadT],
    recorder: _RecordingProvider,
) -> ChainStep:
    """Record a succeeded run (status, timestamps, provenance) and commit it.

    The artifact the layer appended is committed **together with** its run, so
    the pair «result ↔ run» is atomic and the provenance the acceptance of T05
    asked for (``profile_name``/``model_name``/``prompt_version``) is always
    present on the run that produced a stored result. Nothing here is diary
    content: profile/model/prompt version, content digests and token usage only.
    """
    run = session.get(ProcessingRun, run_id)
    if run is None:  # pragma: no cover - the run row was committed before the call
        raise PipelineNotRegisteredError("The processing run of this step disappeared.")
    model_name = (
        resolve_completion_params(recorder.requests[0], profile)["model"]
        if recorder.requests
        else profile.model
    )
    run.status = RunStatus.succeeded
    run.finished_at = utc_now()
    run.profile_name = profile_name
    run.model_name = model_name
    run.prompt_version = prompt_version
    run.input_artifact_id = product.input_artifact_id
    run.metadata_json = _success_metadata(
        layer=layer,
        subject=subject,
        profile_name=profile_name,
        profile=profile,
        model_name=model_name,
        prompt_version=prompt_version,
        recorder=recorder,
    )
    session.commit()
    return ChainStep(
        layer=layer,
        run_id=run_id,
        artifact_id=product.artifact.id,
        profile_name=profile_name,
        model_name=model_name,
        prompt_version=prompt_version,
        input_ref=subject,
    )


def _success_metadata(
    *,
    layer: PipelineType,
    subject: str,
    profile_name: str,
    profile: OpenAICompatibleProfile,
    model_name: str,
    prompt_version: str,
    recorder: _RecordingProvider,
) -> dict[str, object]:
    """Content-free run metadata of a succeeded step.

    Mirrors the shape the export framework writes (single request: a digest
    string and one usage object; several requests: the ordered digest list, the
    summed usage and a call count), so the worker's completion metrics and any
    reader of ``metadata_json`` behave the same for both families.
    """
    metadata: dict[str, object] = {
        "orchestrator": ORCHESTRATOR,
        "layer": layer.value,
        "input_ref": subject,
        "profile": profile_name,
        "provider_type": profile.provider,
        "model": model_name,
        "prompt_version": prompt_version,
    }
    if len(recorder.requests) == 1:
        usage = recorder.responses[0].usage
        metadata["input_sha256"] = request_input_sha256(recorder.requests[0])
        metadata["usage"] = usage.model_dump() if usage is not None else None
        return metadata
    metadata["input_sha256"] = [request_input_sha256(request) for request in recorder.requests]
    metadata["usage"] = _merged_usage(recorder.responses)
    metadata["call_count"] = len(recorder.requests)
    return metadata


def _merged_usage(responses: list[LLMResponse]) -> dict[str, int] | None:
    """Sum token usage across a step's responses (``None`` when any is missing)."""
    if not responses or any(response.usage is None for response in responses):
        return None
    totals = {"input_tokens": 0, "output_tokens": 0}
    for response in responses:
        usage: LLMUsage | None = response.usage
        assert usage is not None  # guaranteed by the guard above
        totals["input_tokens"] += usage.input_tokens
        totals["output_tokens"] += usage.output_tokens
    return totals


def _record_failed_run(session: Session, run_id: uuid.UUID, error: BaseException) -> None:
    """Roll back the step and record its run as ``failed`` (commits).

    The rollback comes first on purpose: it discards anything the failed step
    flushed (the layers validate before writing, and this makes "no partial
    artifact" a property of the orchestrator too), after which the already
    committed run row is re-read and finalised. ``error`` is either a domain
    error with a static message or is safely truncated — no diary content, no
    prompts, no provider payloads (AGENTS.md §8).
    """
    session.rollback()
    run = session.get(ProcessingRun, run_id)
    if run is None:  # pragma: no cover - the run row was committed before the call
        return
    metadata = dict(run.metadata_json) if isinstance(run.metadata_json, dict) else {}
    metadata.update({"error_class": type(error).__name__, "terminal": True})
    run.status = RunStatus.failed
    run.finished_at = utc_now()
    run.error_summary = _safe_summary(error)
    run.metadata_json = metadata
    session.commit()


def _safe_summary(error: BaseException) -> str:
    """``<ErrorClass>: <whitespace-collapsed message>``, truncated and static."""
    message = " ".join(str(error).split())
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message[:MAX_ERROR_SUMMARY_LENGTH]}"


# ---------------------------------------------------------------------------
# One message: the stt_fix step (shared by the day chain and the job dispatch)
# ---------------------------------------------------------------------------


def _stt_fix_step(
    session: Session,
    *,
    entry: RawEntry,
    profile_name: str,
    profile: OpenAICompatibleProfile,
    provider: LLMProvider,
    created_at: datetime | None,
) -> tuple[ChainStep, SttCorrection | None]:
    """Run one ``stt_fix`` step for ``entry`` under its own processing run.

    The single-message unit of the chain: the day chain calls it once per
    message of the day, and :func:`run_stt_fix_step` calls it for the message a
    ``stt_corrected`` job names. ``entry`` is never empty here — the callers
    skip a message with no text before calling.
    """
    subject = f"message:{entry.id}"

    def perform_stt_fix(
        step_provider: LLMProvider,
        run_id: uuid.UUID,
        *,
        corrected_entry: RawEntry = entry,
    ) -> _StepProduct[SttCorrection]:
        outcome = correct_stt(
            session,
            entry=corrected_entry,
            profile=profile,
            provider=step_provider,
            processing_run_id=run_id,
            created_at=created_at,
        )
        return _StepProduct(artifact=outcome.artifact, payload=outcome.correction)

    step, product = _run_step(
        session,
        layer=PipelineType.stt_corrected,
        profile_name=profile_name,
        profile=profile,
        provider=provider,
        prompt_version=STT_FIX_PROMPT_VERSION,
        subject=subject,
        perform=perform_stt_fix,
    )
    return step, product.payload


def run_stt_fix_step(
    session: Session,
    *,
    raw_id: uuid.UUID,
    models_config: ModelsConfig | None = None,
    providers: Mapping[str, LLMProvider] | None = None,
    local_tz: str | None = None,
    created_at: datetime | None = None,
) -> SttFixStepOutcome:
    """Correct exactly one message — the ``stt_corrected`` unit of work.

    The message is resolved by identity (never by a caller-supplied text) and
    mapped onto the ``raw_entry`` contract by the day layer's own selector, so
    the correction always describes exactly the stored row a summary would see:
    the message's local day is derived from its timestamp in ``local_tz``
    (default ``APP_TIMEZONE``) and the entry is looked up in that day's
    ``raw_entry[]``. A message that does not exist or is soft-deleted is a loud
    :class:`~diary.domain.pipelines.base.PipelineInputError` — no correction is
    invented for it.

    Outcomes mirror the day chain: an absent ``diary_stt`` profile means the
    layer is switched off (``profile_off``) and a message with no text yet (a
    voice message without a succeeded transcription) is skipped with
    ``no_input``; otherwise the step runs under its own processing run and the
    validated :class:`~diary.domain.contracts.SttCorrection` is returned.
    """
    config = models_config if models_config is not None else load_models_config()
    timezone_name = local_tz or get_settings().timezone
    subject = f"message:{raw_id}"
    entry = _entry_for_message(session, raw_id=raw_id, local_tz=timezone_name)
    stt_name, stt_profile = _layer_profile(config, PipelineType.stt_corrected)
    if stt_profile is None:
        return SttFixStepOutcome(
            raw_id=raw_id,
            skipped=SkippedStep(PipelineType.stt_corrected, SKIP_PROFILE_OFF, subject),
        )
    if not entry.text.strip():
        return SttFixStepOutcome(
            raw_id=raw_id,
            skipped=SkippedStep(PipelineType.stt_corrected, SKIP_NO_INPUT, subject),
        )
    active_stt_profile = stt_profile
    stt_provider = _provider_for(
        profile_name=stt_name, profile=active_stt_profile, providers=providers
    )
    step, correction = _stt_fix_step(
        session,
        entry=entry,
        profile_name=stt_name,
        profile=active_stt_profile,
        provider=stt_provider,
        created_at=created_at,
    )
    return SttFixStepOutcome(raw_id=raw_id, step=step, correction=correction)


def _entry_for_message(session: Session, *, raw_id: uuid.UUID, local_tz: str) -> RawEntry:
    """The ``raw_entry`` of the stored message ``raw_id`` (loud when absent).

    The day's own selector is reused on purpose: a single-message step must map
    a row onto the contract exactly as the day step does (text messages by
    ``body_text``, voice messages by the raw text of the newest succeeded
    transcription), so nothing can diverge between the two paths. The local day
    of the message is derived from its UTC timestamp in ``local_tz``.
    """
    message = MessageRepository(session).get(raw_id)
    if message is None:
        raise PipelineInputError("The message this step was asked to correct does not exist.")
    day = local_date_of(message.created_at, ZoneInfo(local_tz))
    for entry in collect_day_entries(session, local_date=day, tz=ZoneInfo(local_tz)):
        if entry.id == raw_id:
            return entry
    # The row exists but the day selector did not return it: it is soft-deleted
    # (or its audio file row is gone). Either way there is nothing to correct.
    raise PipelineInputError(
        "The message this step was asked to correct is not available for correction."
    )


# ---------------------------------------------------------------------------
# Day chain
# ---------------------------------------------------------------------------


def run_day_chain(
    session: Session,
    *,
    local_date: date,
    models_config: ModelsConfig | None = None,
    providers: Mapping[str, LLMProvider] | None = None,
    local_tz: str | None = None,
    created_at: datetime | None = None,
) -> DayChainOutcome:
    """Run one local calendar day through ``stt_fix`` and then ``sum_daily``.

    The day's messages are selected **once**, read-only, by the day layer's own
    selector (``collect_day_entries`` — soft-deleted excluded, voice messages
    with the raw text of their newest succeeded transcription): the same
    ``raw_entry[]`` list is fed to the ``stt_fix`` steps and to the day step, so
    the correction and the summary always describe the same set of source rows.

    ``providers`` is the injection seam (keyed by profile field name:
    ``diary_stt``, ``diary_sum``, ``diary_week``, ``diary_coach``); ``local_tz``
    defaults to ``APP_TIMEZONE`` and is stored with the day anchor. Nothing is
    written to ``messages``/``audio_files``/``transcriptions`` — the only write
    path of the chain is the append-only result store, and every step runs under
    its own processing run (see the module docstring).
    """
    config = models_config if models_config is not None else load_models_config()
    timezone_name = local_tz or get_settings().timezone
    day_subject = f"day:{local_date.isoformat()}"
    entries = collect_day_entries(session, local_date=local_date, tz=ZoneInfo(timezone_name))
    steps: list[ChainStep] = []
    skipped: list[SkippedStep] = []

    # --- step 1: stt_fix, one run per message of the day --------------------
    stt_name, stt_profile = _layer_profile(config, PipelineType.stt_corrected)
    if stt_profile is None:
        skipped.append(SkippedStep(PipelineType.stt_corrected, SKIP_PROFILE_OFF, day_subject))
    else:
        active_stt_profile = stt_profile
        stt_provider = _provider_for(
            profile_name=stt_name, profile=active_stt_profile, providers=providers
        )
        for entry in entries:
            if not entry.text.strip():
                # A voice message that has not been transcribed yet has nothing
                # to correct: the step is skipped, never asked to invent text.
                skipped.append(
                    SkippedStep(PipelineType.stt_corrected, SKIP_NO_INPUT, f"message:{entry.id}")
                )
                continue
            step, _ = _stt_fix_step(
                session,
                entry=entry,
                profile_name=stt_name,
                profile=active_stt_profile,
                provider=stt_provider,
                created_at=created_at,
            )
            steps.append(step)

    # --- step 2: sum_daily, one run for the day -----------------------------
    day_name, day_profile = _layer_profile(config, PipelineType.daily_summary)
    summary: DailySummary | None = None
    summary_artifact_id: uuid.UUID | None = None
    if day_profile is None:
        skipped.append(SkippedStep(PipelineType.daily_summary, SKIP_PROFILE_OFF, day_subject))
    elif not entries:
        skipped.append(SkippedStep(PipelineType.daily_summary, SKIP_NO_INPUT, day_subject))
    else:
        active_day_profile = day_profile
        day_provider = _provider_for(
            profile_name=day_name, profile=active_day_profile, providers=providers
        )

        def perform_sum_daily(
            step_provider: LLMProvider, run_id: uuid.UUID
        ) -> _StepProduct[DailySummary]:
            outcome = sum_daily(
                session,
                local_date=local_date,
                profile=active_day_profile,
                provider=step_provider,
                local_tz=timezone_name,
                entries=entries,
                processing_run_id=run_id,
                created_at=created_at,
            )
            return _StepProduct(artifact=outcome.artifact, payload=outcome.summary)

        step, product = _run_step(
            session,
            layer=PipelineType.daily_summary,
            profile_name=day_name,
            profile=active_day_profile,
            provider=day_provider,
            prompt_version=SUM_DAILY_PROMPT_VERSION,
            subject=day_subject,
            perform=perform_sum_daily,
        )
        steps.append(step)
        summary = product.payload
        summary_artifact_id = product.artifact.id

    return DayChainOutcome(
        local_date=local_date,
        steps=tuple(steps),
        skipped=tuple(skipped),
        summary=summary,
        summary_artifact_id=summary_artifact_id,
    )


# ---------------------------------------------------------------------------
# Week chain
# ---------------------------------------------------------------------------


def run_week_chain(
    session: Session,
    *,
    week_id: str,
    run_days: bool = False,
    models_config: ModelsConfig | None = None,
    providers: Mapping[str, LLMProvider] | None = None,
    local_tz: str | None = None,
    created_at: datetime | None = None,
) -> WeekChainOutcome:
    """Run one ISO week through ``sum_weekly`` (optionally refreshing its days).

    The normative input of the weekly layer is **the stored daily summaries of
    the week** (T06: only a merge of day summaries, never raw), so the default
    call is exactly one step: ``sum_weekly`` over the week's stored day
    summaries. ``run_days=True`` prepends :func:`run_day_chain` for each of the
    week's seven local days (Monday → Sunday), which is how the full
    ``raw → stt_fix → sum_daily → sum_weekly`` chain is produced in one call;
    the day chains are the same explicit per-day runs, so nothing about them
    changes — only the number of runs an operator asked for.

    A week without a single stored day summary is skipped with
    :data:`SKIP_NO_INPUT` (the layer refuses to invent a week); a malformed or
    non-existent ``week_id`` fails loudly before anything runs, because the ISO
    week calendar is consulted to enumerate the days.

    **The coaching note is not reachable from here.** Layer 5 is a separate,
    explicit launch (:func:`run_coach_step`, T07 decision 3) and this function
    takes no parameter for it; a week run never produces a ``coach_note``.
    """
    config = models_config if models_config is not None else load_models_config()
    timezone_name = local_tz or get_settings().timezone
    week_subject = f"week:{week_id}"
    steps: list[ChainStep] = []
    skipped: list[SkippedStep] = []

    if run_days:
        for day in week_days(week_id):  # validates week_id, Monday → Sunday
            day_outcome = run_day_chain(
                session,
                local_date=day,
                models_config=config,
                providers=providers,
                local_tz=timezone_name,
                created_at=created_at,
            )
            steps.extend(day_outcome.steps)
            skipped.extend(day_outcome.skipped)

    week_name, week_profile = _layer_profile(config, PipelineType.weekly_summary)
    summary: WeeklySummary | None = None
    summary_artifact_id: uuid.UUID | None = None
    if week_profile is None:
        skipped.append(SkippedStep(PipelineType.weekly_summary, SKIP_PROFILE_OFF, week_subject))
    else:
        active_week_profile = week_profile
        # Read-only: the week's stored daily summaries (newest version of each
        # day). A malformed week_id fails inside the week math, before the
        # skip decision.
        stored = collect_week_summaries(session, week_id=week_id, local_tz=timezone_name)
        if not stored:
            skipped.append(SkippedStep(PipelineType.weekly_summary, SKIP_NO_INPUT, week_subject))
        else:
            week_provider = _provider_for(
                profile_name=week_name, profile=active_week_profile, providers=providers
            )

            def perform_sum_weekly(
                step_provider: LLMProvider, run_id: uuid.UUID
            ) -> _StepProduct[WeeklySummary]:
                outcome = sum_weekly(
                    session,
                    week_id=week_id,
                    profile=active_week_profile,
                    provider=step_provider,
                    local_tz=timezone_name,
                    summaries=stored,
                    processing_run_id=run_id,
                    created_at=created_at,
                )
                return _StepProduct(artifact=outcome.artifact, payload=outcome.summary)

            step, product = _run_step(
                session,
                layer=PipelineType.weekly_summary,
                profile_name=week_name,
                profile=active_week_profile,
                provider=week_provider,
                prompt_version=SUM_WEEKLY_PROMPT_VERSION,
                subject=week_subject,
                perform=perform_sum_weekly,
            )
            steps.append(step)
            summary = product.payload
            summary_artifact_id = product.artifact.id

    return WeekChainOutcome(
        week_id=week_id,
        steps=tuple(steps),
        skipped=tuple(skipped),
        summary=summary,
        summary_artifact_id=summary_artifact_id,
    )


# ---------------------------------------------------------------------------
# Coach (layer 5): explicit launch only
# ---------------------------------------------------------------------------


def run_coach_step(
    session: Session,
    *,
    week_id: str,
    models_config: ModelsConfig | None = None,
    providers: Mapping[str, LLMProvider] | None = None,
    created_at: datetime | None = None,
) -> CoachStepOutcome:
    """Run the coaching note of one ISO week — **only when called explicitly**.

    This function is the single entry point of layer 5 in the orchestrator and
    nothing calls it on the user's behalf: neither :func:`run_day_chain` nor
    :func:`run_week_chain` can reach it, the worker registers the layer but has
    no handler for its job type (an enqueued ``coach_note`` job still fails
    permanently instead of running the layer), and the layer module itself
    starts nothing on import (T07 decision 3). Its input is exactly one stored
    ``weekly_summary`` artifact (the newest version of the week), which is also
    the run's ``input_artifact_id`` — the only artifact input the chain links
    back.

    An absent ``diary_coach`` profile means the layer is switched off and the
    outcome carries the ``profile_off`` skip; a week without a weekly summary
    fails loudly inside the run (recorded as a failed run), never silently.
    """
    config = models_config if models_config is not None else load_models_config()
    week_subject = f"week:{week_id}"
    coach_name, coach_profile = _layer_profile(config, PipelineType.coach_note)
    if coach_profile is None:
        return CoachStepOutcome(
            week_id=week_id,
            skipped=SkippedStep(PipelineType.coach_note, SKIP_PROFILE_OFF, week_subject),
        )
    active_coach_profile = coach_profile
    coach_provider = _provider_for(
        profile_name=coach_name, profile=active_coach_profile, providers=providers
    )

    def perform_coach_note(
        step_provider: LLMProvider, run_id: uuid.UUID
    ) -> _StepProduct[CoachNote]:
        run_input = collect_week_summary(session, week_id=week_id)
        outcome = coach_note(
            session,
            week_id=week_id,
            provider=step_provider,
            profile=active_coach_profile,
            coach_input=run_input,
            processing_run_id=run_id,
            created_at=created_at,
        )
        return _StepProduct(
            artifact=outcome.artifact,
            payload=outcome.note,
            input_artifact_id=run_input.artifact_id,
        )

    step, product = _run_step(
        session,
        layer=PipelineType.coach_note,
        profile_name=coach_name,
        profile=active_coach_profile,
        provider=coach_provider,
        prompt_version=COACH_PROMPT_VERSION,
        subject=week_subject,
        perform=perform_coach_note,
    )
    return CoachStepOutcome(week_id=week_id, step=step, note=product.payload)
