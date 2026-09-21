"""Job-handler protocol and registry for the durable queue (Task 08, ADR-003).

A handler executes one job of a :class:`~diary.db.models.JobType`:

- signature ``(session, job) -> None`` — ``session`` is a fresh SQLAlchemy
  ``Session`` opened by the worker for this job; the handler may use it for its
  side effects and may commit. The worker commits again after a successful
  return, so a handler that never touches the session is safe too.
- after a normal return the worker marks the job ``succeeded``;
- after :class:`PermanentJobError` the worker marks the job ``failed``
  immediately (deterministic failures that cannot succeed on retry);
- after any other exception the worker applies the retry policy
  (``retrying`` with exponential backoff, ``failed`` once ``max_attempts``
  are exhausted).

Privacy contract (architecture brief §11, AGENTS.md §7–8): a job payload and
the exceptions/messages a handler raises must never carry diary content or
provider secrets. The worker stores only a short, non-sensitive summary of an
exception in ``jobs.error_summary``.

Registered handlers:

- ``transcribe_audio`` (Task 09) — the real STT pipeline: delegates to
  :func:`diary.domain.transcription.service.run_transcription` and translates
  deterministic domain failures (missing audio, invalid payload, provider not
  installed, broken model configuration) into a permanent failure;
- ``normalize``/``summarize``/``wellbeing`` (Task 12) — the generic pipeline
  handler: delegates to :func:`diary.domain.pipelines.service.run_pipeline`,
  which drives the processing-run lifecycle. Deterministic failures
  (pipeline not registered yet, invalid input, invalid structured output,
  config errors, non-retryable LLM errors) already fail the processing run and
  are re-raised as permanent job failures; transient LLM errors propagate
  unchanged and follow the retry policy.
- ``stt_corrected``/``daily_summary``/``weekly_summary`` (task T11) — the three
  phase-2 layers of the second anchor family, dispatched to the orchestrator
  :mod:`diary.domain.pipelines.pipeline`: one message through
  ``run_stt_fix_step``, one local day through ``run_day_chain``, one ISO week
  through ``run_week_chain``. The job payload carries **only the anchor**
  (``raw_id`` / ``local_date`` + ``local_tz`` / ``week_id`` + ``run_days``),
  never diary content.

Deliberately **not** registered: ``coach_note``. Layer 5 keeps its
«explicit launch only» rule (T07 decision 3): an enqueued ``coach_note`` job
still fails permanently instead of running the layer, and the coaching note is
produced only by an explicit call (the T11 CLI).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import date
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfigError
from diary.db.models import Job, JobType
from diary.domain.pipelines import pipeline as phase2_pipeline
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.base import PipelinePermanentError
from diary.domain.pipelines.nli_guard import default_nli_judge
from diary.domain.transcription import service as transcription_service
from diary.domain.transcription.providers import ProviderUnavailableError

logger = logging.getLogger("diary.workers.handlers")

#: Phase-2 job payload keys (identifiers/anchors only — never diary content).
PAYLOAD_RAW_ID = "raw_id"
PAYLOAD_LOCAL_DATE = "local_date"
PAYLOAD_LOCAL_TZ = "local_tz"
PAYLOAD_WEEK_ID = "week_id"
PAYLOAD_RUN_DAYS = "run_days"


class PermanentJobError(Exception):
    """A handler error that must not be retried.

    Raising this marks the job ``failed`` immediately, regardless of the
    remaining ``max_attempts`` — for deterministic failures such as "this job
    type has no handler yet" or "the input no longer exists".
    """


class JobHandler(Protocol):
    """A callable that executes one job of its registered type."""

    def __call__(self, session: Session, job: Job) -> None: ...


#: Registered handlers, keyed by ``JobType.value`` (the stored string form).
REGISTRY: dict[str, JobHandler] = {}


def register(job_type: JobType, handler: JobHandler) -> None:
    """Register ``handler`` for ``job_type`` (raises on a duplicate)."""
    if job_type.value in REGISTRY:
        raise ValueError(f"A job handler for job type {job_type.value!r} is already registered.")
    REGISTRY[job_type.value] = handler


def get_handler(job_type: JobType | str) -> JobHandler | None:
    """Return the handler registered for ``job_type``, or ``None``."""
    key = job_type.value if isinstance(job_type, JobType) else job_type
    return REGISTRY.get(key)


def known_job_types() -> frozenset[str]:
    """The job types that currently have a registered handler."""
    return frozenset(REGISTRY)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _transcribe_audio(session: Session, job: Job) -> None:
    """Execute one ``transcribe_audio`` job (Task 09).

    Deterministic failures are wrapped in :class:`PermanentJobError` so the
    job fails immediately with a safe summary instead of burning retries:

    - the payload has no valid ``audio_file_id`` (``TranscriptionPermanentError``);
    - the audio row/bytes are gone;
    - the optional faster-whisper runtime is not installed
      (``ProviderUnavailableError``);
    - the models configuration is missing/broken (``ModelsConfigError``).

    Any other exception (provider runtime failure, decoder error) propagates
    unchanged and follows the worker's transient retry policy; the domain
    service writes the single terminal ``failed`` transcription row when the
    last attempt fails.
    """
    try:
        transcription_service.run_transcription(session, job)
    except transcription_service.TranscriptionPermanentError as exc:
        raise PermanentJobError(str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise PermanentJobError(str(exc)) from exc
    except ModelsConfigError as exc:
        raise PermanentJobError(str(exc)) from exc


def _run_pipeline_job(session: Session, job: Job) -> None:
    """Execute one ``normalize``/``summarize``/``wellbeing`` job (Task 12).

    Delegates to :func:`diary.domain.pipelines.service.run_pipeline`, which
    owns the processing-run lifecycle and already commits terminal *failed*
    runs. Only the deterministic-failure surface is translated here: a
    :class:`PipelinePermanentError` (raised for unregistered pipelines, invalid
    inputs, invalid structured output, config errors and non-retryable LLM
    errors) marks the job failed immediately. Transient LLM errors propagate
    unchanged so the worker's retry policy applies.
    """
    try:
        pipeline_service.run_pipeline(session, job)
    except pipeline_service.PipelinePermanentError as exc:
        raise PermanentJobError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase-2 layer dispatch (task T11)
# ---------------------------------------------------------------------------
# The three dispatched layers are exactly the ones an explicit request can ask
# for without breaking a guarantee: the message correction, the day and the
# week. ``coach_note`` is deliberately absent (T07 decision 3, module
# docstring).


def _payload_anchor(job: Job, key: str, *, kind: str) -> str:
    """A non-empty string anchor ``key`` from a phase-2 job payload.

    A missing or malformed anchor is a deterministic input failure (the job can
    never succeed by being retried), so it raises :class:`PermanentJobError`
    with a static message naming only the payload key (AGENTS.md §8).
    """
    payload = job.payload_json if isinstance(job.payload_json, dict) else None
    value = payload.get(key) if payload is not None else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise PermanentJobError(f"The {kind} job payload has no valid {key!r}.")


def _payload_raw_id(job: Job) -> uuid.UUID:
    """The message id a ``stt_corrected`` job names."""
    value = _payload_anchor(job, PAYLOAD_RAW_ID, kind="stt_corrected")
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise PermanentJobError(
            f"The stt_corrected job payload has no valid {PAYLOAD_RAW_ID!r}."
        ) from exc


def _payload_local_date(job: Job) -> date:
    """The local calendar day a ``daily_summary`` job names."""
    value = _payload_anchor(job, PAYLOAD_LOCAL_DATE, kind="daily_summary")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PermanentJobError(
            f"The daily_summary job payload has no valid {PAYLOAD_LOCAL_DATE!r}."
        ) from exc


def _payload_run_days(job: Job) -> bool:
    """Whether a ``weekly_summary`` job also refreshes the week's days."""
    payload = job.payload_json if isinstance(job.payload_json, dict) else {}
    return payload.get(PAYLOAD_RUN_DAYS) is True


def _payload_local_tz(job: Job) -> str | None:
    """The optional ``local_tz`` of a ``daily_summary`` payload, validated.

    An absent or blank value means "use ``APP_TIMEZONE``". A present value that
    is not an IANA timezone name is a deterministic input failure (retrying
    cannot fix a typo), so it raises :class:`PermanentJobError` naming only the
    payload key — otherwise the unknown zone would surface as a raw
    ``ZoneInfoNotFoundError`` deep inside the chain and burn the retry budget
    (AGENTS.md §8: no diary content in the message).
    """
    payload = job.payload_json if isinstance(job.payload_json, dict) else {}
    value = payload.get(PAYLOAD_LOCAL_TZ)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PermanentJobError(
            f"The daily_summary job payload has no valid {PAYLOAD_LOCAL_TZ!r} "
            "(it must be an IANA timezone name)."
        )
    name = value.strip()
    if not name:
        return None
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise PermanentJobError(
            f"The daily_summary job payload has no valid {PAYLOAD_LOCAL_TZ!r} "
            "(it must be an IANA timezone name)."
        ) from exc
    return name


def _skips_of(outcome: object) -> tuple[phase2_pipeline.SkippedStep, ...]:
    """The skipped steps of a phase-2 outcome, normalized to a tuple.

    ``run_stt_fix_step``/``run_coach_step`` carry a single skip (or ``None``),
    while ``run_day_chain``/``run_week_chain`` carry a tuple of them — this
    helper hides that difference from the dispatch policy below.
    """
    skipped = getattr(outcome, "skipped", None)
    if isinstance(skipped, phase2_pipeline.SkippedStep):
        return (skipped,)
    if isinstance(skipped, tuple):
        return tuple(item for item in skipped if isinstance(item, phase2_pipeline.SkippedStep))
    return ()


def _require_layer_enabled(skips: tuple[phase2_pipeline.SkippedStep, ...], *, layer: str) -> None:
    """Translate a ``profile_off`` skip of the requested layer into a failure.

    An absent phase-2 profile means "this layer is switched off"
    (``docs/tasklist-phase2.md`` §9.8) — the orchestrator skips the step and
    records no run. A job that was enqueued **for that layer** must not report
    success while doing nothing (the export family behaves the same way: a
    missing profile fails the run permanently), so the skip is reported as a
    deterministic failure naming the configuration field. A ``no_input`` skip —
    an empty day, a week without stored day summaries, a voice message without
    a transcription — is *not* a failure: those are legitimate states with
    nothing to do, and the orchestrator treats them as skips as well.
    """
    for skipped in skips:
        if skipped.layer.value != layer:
            continue
        if skipped.reason != phase2_pipeline.SKIP_PROFILE_OFF:
            continue
        profile_name = _PROFILE_FIELD_BY_JOB.get(layer, layer)
        raise PermanentJobError(
            f"The phase-2 layer {layer!r} is switched off: the model profile "
            f"{profile_name!r} is not configured."
        )


#: Job type value → the configuration profile field the layer needs.
_PROFILE_FIELD_BY_JOB: dict[str, str] = {
    JobType.stt_corrected.value: "diary_stt",
    JobType.daily_summary.value: "diary_sum",
    JobType.weekly_summary.value: "diary_week",
}


def _log_skips(layer: str, skips: tuple[phase2_pipeline.SkippedStep, ...], *, job: Job) -> None:
    """Log the skips of a dispatched phase-2 job (reason + safe subject only)."""
    for skipped in skips:
        logger.info(
            "worker job=%s layer=%s step=skipped reason=%s subject=%s",
            job.id,
            layer,
            skipped.reason,
            skipped.subject,
        )


def _run_phase2_job(
    session: Session,
    job: Job,
    *,
    layer: str,
    perform: Callable[[Session, Job], object],
) -> None:
    """Run one dispatched phase-2 layer step (the shared error translation).

    Deterministic failures (:class:`~diary.domain.pipelines.base.PipelinePermanentError`
    — an unusable payload, a message that no longer exists, a layer that was
    never registered — and a missing/broken models configuration, as
    :class:`~diary.core.models_config.ModelsConfigError`) become permanent job
    failures; transient LLM errors propagate unchanged so the retry policy
    applies. Anything the layer wrote is committed by the worker after a normal
    return.
    """
    try:
        outcome = perform(session, job)
    except PipelinePermanentError as exc:
        raise PermanentJobError(str(exc)) from exc
    except ModelsConfigError as exc:
        # The same convention ``_transcribe_audio``/``_run_pipeline_job`` use: a
        # configuration that cannot be read can never be fixed by a retry, so it
        # must not burn the retry budget (its message names no secret value).
        raise PermanentJobError(str(exc)) from exc
    skips = _skips_of(outcome)
    _require_layer_enabled(skips, layer=layer)
    _log_skips(layer, skips, job=job)


def _stt_corrected_job(session: Session, job: Job) -> None:
    """Execute one ``stt_corrected`` job: correct exactly one message.

    The payload names the message (``raw_id``); the message's own stored text is
    the input, so the job never carries diary content. The step runs under its
    own processing run and the correction is stored inline by the layer (T04).
    """

    def perform(current_session: Session, current_job: Job) -> object:
        return phase2_pipeline.run_stt_fix_step(
            current_session, raw_id=_payload_raw_id(current_job)
        )

    _run_phase2_job(session, job, layer=JobType.stt_corrected.value, perform=perform)


def _daily_summary_job(session: Session, job: Job) -> None:
    """Execute one ``daily_summary`` job: the day chain of one local day.

    The day chain is the orchestrator's day unit — the day's messages through
    ``stt_fix`` (one run each) and then the day through ``sum_daily`` — so a day
    summary requested as a job gets exactly the run the CLI produces. The
    payload names the day (``local_date``, optional ``local_tz``); an invalid
    ``local_tz`` is a permanent payload failure (``_payload_local_tz``).
    """

    def perform(current_session: Session, current_job: Job) -> object:
        return phase2_pipeline.run_day_chain(
            current_session,
            local_date=_payload_local_date(current_job),
            local_tz=_payload_local_tz(current_job),
            nli_judge=default_nli_judge(),
        )

    _run_phase2_job(session, job, layer=JobType.daily_summary.value, perform=perform)


def _weekly_summary_job(session: Session, job: Job) -> None:
    """Execute one ``weekly_summary`` job: the week chain of one ISO week.

    The normative input is the stored day summaries of the week (never raw
    text); ``run_days`` additionally refreshes the week's days first, which is
    how the full ``raw → stt_fix → sum_daily → sum_weekly`` chain is produced by
    one job. The payload names the week (``week_id``).
    """

    def perform(current_session: Session, current_job: Job) -> object:
        return phase2_pipeline.run_week_chain(
            current_session,
            week_id=_payload_anchor(current_job, PAYLOAD_WEEK_ID, kind="weekly_summary"),
            run_days=_payload_run_days(current_job),
            nli_judge=default_nli_judge(),
        )

    _run_phase2_job(session, job, layer=JobType.weekly_summary.value, perform=perform)


register(JobType.transcribe_audio, _transcribe_audio)
for _pipeline_job_type in (
    JobType.normalize,
    JobType.summarize,
    JobType.wellbeing,
):
    register(_pipeline_job_type, _run_pipeline_job)
for _phase2_job_type, _phase2_handler in (
    (JobType.stt_corrected, _stt_corrected_job),
    (JobType.daily_summary, _daily_summary_job),
    (JobType.weekly_summary, _weekly_summary_job),
):
    register(_phase2_job_type, _phase2_handler)
# ``JobType.coach_note`` is intentionally left unregistered: layer 5 is launched
# only by an explicit call (T07 decision 3), and this handler table is the
# machine-checked form of that rule.
