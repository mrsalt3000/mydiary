"""Transcription use cases: request and execute a manual STT job (Task 09).

Responsibilities
================

* :func:`request_transcription` — the explicit user action behind the
  ``Transcribe``/``Retranscribe`` button. It never runs STT in the HTTP
  request: it enqueues a durable ``transcribe_audio`` job and returns it.
  While a job for the audio file is in flight (``pending``/``running``/
  ``retrying``) the existing job is returned instead of creating a duplicate;
  after a terminal outcome an explicit repeat request starts a fresh run, so
  retranscription stays possible. The Task-08 idempotency-key reuse of a
  *succeeded* job would block retranscribe, so deduplication is done here by
  the active-job check instead (see :func:`find_active_transcription_job`).
* :func:`run_transcription` — executes one claimed ``transcribe_audio`` job
  (called by the worker handler): resolves the stored audio file, loads the
  ``transcription`` profile from the models configuration (Task 03), runs the
  configured provider and persists the result.

Transcription lifecycle and idempotency (ADR-005, architecture brief §8)
========================================================================

* Every job that reaches a terminal outcome records **exactly one** new
  ``transcriptions`` row for its audio file (``succeeded`` with the raw text,
  or ``failed`` with an empty ``raw_text``). Audio and the message are never
  modified. The Task-07 ``not_requested`` sentinel row is left untouched.
* ``succeeded``/``failed`` rows carry a profile snapshot (provider/model/
  language columns) plus ``metadata_json`` with the originating ``job_id``,
  so each result is traceable to its exact run.
* A provider failure is transient: the worker retries it with backoff; the
  ``failed`` row is written only when the last attempt fails
  (``job.attempts >= max_attempts``), so a retrying job never multiplies
  failure rows.
* Re-execution safety (worker crash between the handler commit and
  ``mark_succeeded``; stale-``running`` reset): before doing any work the
  handler checks whether a *succeeded* row already records this job's
  ``job_id`` and returns without re-running the model — no duplicate rows.

Privacy contract (AGENTS.md §7–8): job payloads carry only identifiers and a
profile reference; raw transcription text is persisted in ``transcriptions``
but is never logged; every exception raised here has a static, non-sensitive
message (provider errors are sanitised inside the adapter boundary).
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from diary.core.models_config import FasterWhisperProfile, load_models_config
from diary.core.settings import get_settings
from diary.db.models import Job, JobType, Transcription, TranscriptionStatus
from diary.db.repositories import (
    AudioFileRepository,
    JobRepository,
    TranscriptionRepository,
)
from diary.domain.audio.service import AudioNotFoundError, resolve_audio_file
from diary.domain.transcription.providers import (
    ProviderUnavailableError,
    TranscriptionProvider,
    create_provider,
)

#: ``profile_ref`` value stored in job payloads (the Task-03 ``transcription``
#: profile is the only STT profile in the MVP).
PROFILE_REF_TRANSCRIPTION = "transcription"


class TranscriptionServiceError(Exception):
    """Base class for transcription service errors (never diary content)."""


class TranscriptionPermanentError(TranscriptionServiceError):
    """The job input is unusable and retrying cannot fix it.

    Mapped to :class:`~diary.workers.handlers.PermanentJobError` at the worker
    boundary (audio row missing, malformed payload, ...).
    """


def _load_transcription_profile() -> FasterWhisperProfile:
    """Return the current ``transcription`` profile from the models config.

    Module-level hook so tests can substitute a profile without a config file
    (no models YAML is needed in CI; monkeypatched in integration tests).
    """
    return load_models_config().transcription


def find_active_transcription_job(session: Session, audio_file_id: uuid.UUID) -> Job | None:
    """The in-flight ``transcribe_audio`` job for ``audio_file_id``, if any.

    A job is "in flight" when it is ``pending``, ``running`` or ``retrying``.
    Terminal jobs do not count, so an explicit repeat request after a success
    or a failure starts a fresh run.
    """
    target = str(audio_file_id)
    for job in JobRepository(session).list_active_of_type(JobType.transcribe_audio):
        payload = job.payload_json if isinstance(job.payload_json, dict) else {}
        if str(payload.get("audio_file_id", "")) == target:
            return job
    return None


def request_transcription(
    session: Session,
    audio_file_id: uuid.UUID,
    *,
    profile_ref: str = PROFILE_REF_TRANSCRIPTION,
) -> Job:
    """Queue a manual transcription for one voice message; return the job.

    Raises :class:`~diary.domain.audio.service.AudioNotFoundError` (→ HTTP 404)
    when no audio-file row exists. When a job for the audio file is already in
    flight the existing job is returned unchanged — no duplicate row, and STT
    never runs inside the HTTP request.
    """
    if AudioFileRepository(session).get(audio_file_id) is None:
        raise AudioNotFoundError("Audio file not found.")
    active = find_active_transcription_job(session, audio_file_id)
    if active is not None:
        return active
    job = JobRepository(session).enqueue(
        job_type=JobType.transcribe_audio,
        payload_json={"audio_file_id": str(audio_file_id), "profile_ref": profile_ref},
    )
    session.commit()
    return job


def run_transcription(session: Session, job: Job) -> None:
    """Execute one claimed ``transcribe_audio`` job (worker handler core).

    The caller (``diary.workers.runner``) owns the transaction: on a normal
    return it commits the session and marks the job ``succeeded``; on an
    exception it applies the retry policy. This function commits itself only
    when it writes the single terminal ``failed`` row on the final attempt
    (the runner would otherwise roll the handler session back after the raise).
    """
    audio_file_id = _payload_audio_id(job)
    if _has_recorded_success(session, audio_file_id, job):
        # A previous execution of this very job already committed a succeeded
        # row (worker crashed between the handler commit and mark_succeeded and
        # the stale-running reset replayed the job). Re-running the model would
        # duplicate the result, so this replay is a no-op.
        return
    if _has_recorded_failure(session, audio_file_id, job):
        # The job already recorded its single terminal ``failed`` row and the
        # worker crashed before finalizing it (attempts are exhausted). Do not
        # run the model beyond the retry budget; fail the replay immediately.
        raise TranscriptionPermanentError(
            "This transcription job already recorded a terminal failure."
        )
    try:
        audio_path, _mime = resolve_audio_file(session, audio_file_id)
    except AudioNotFoundError as exc:
        raise TranscriptionPermanentError(
            "The audio file for this transcription job no longer exists."
        ) from exc
    if not audio_path.is_file():
        raise TranscriptionPermanentError(
            "The audio file for this transcription job is missing on disk."
        )
    profile = _load_transcription_profile()
    provider = create_provider(profile)
    try:
        raw_text = provider.transcribe(audio_path, profile)
    except ProviderUnavailableError:
        # Environment problem (faster-whisper not installed): never retryable;
        # the handler maps it to a permanent job failure. No result row is
        # written — no provider run happened.
        raise
    except Exception:
        if _is_terminal_attempt(job) and not _has_recorded_failure(session, audio_file_id, job):
            _record_failed_row(session, job, profile, audio_file_id)
            session.commit()
        raise
    _record_success_row(session, job, provider, profile, audio_file_id, raw_text)


def list_transcription_history(session: Session, audio_file_id: uuid.UUID) -> list[Transcription]:
    """Actual transcription-run rows for one audio file, oldest first.

    The Task-07 ``not_requested`` sentinel row is an upload-time placeholder,
    not a transcription result, and is excluded from the history. Raises
    :class:`AudioNotFoundError` when the audio file does not exist.
    """
    if AudioFileRepository(session).get(audio_file_id) is None:
        raise AudioNotFoundError("Audio file not found.")
    rows = TranscriptionRepository(session).list_by_audio_file(audio_file_id)
    return [row for row in rows if row.status is not TranscriptionStatus.not_requested]


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _payload_audio_id(job: Job) -> uuid.UUID:
    """The ``audio_file_id`` from a transcribe payload as a UUID."""
    payload = job.payload_json if isinstance(job.payload_json, dict) else None
    value = payload.get("audio_file_id") if payload is not None else None
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError:
            pass
    raise TranscriptionPermanentError(
        "The transcribe_audio job payload has no valid audio_file_id."
    )


def _rows_for_job(session: Session, audio_file_id: uuid.UUID, job: Job) -> list[Transcription]:
    """Transcription rows of one audio file that record ``job.id``."""
    job_id = str(job.id)
    return [
        row
        for row in TranscriptionRepository(session).list_by_audio_file(audio_file_id)
        if isinstance(row.metadata_json, dict) and row.metadata_json.get("job_id") == job_id
    ]


def _has_recorded_success(session: Session, audio_file_id: uuid.UUID, job: Job) -> bool:
    """Whether a ``succeeded`` row already records this job's outcome."""
    return any(
        row.status is TranscriptionStatus.succeeded
        for row in _rows_for_job(session, audio_file_id, job)
    )


def _has_recorded_failure(session: Session, audio_file_id: uuid.UUID, job: Job) -> bool:
    """Whether a ``failed`` row already records this job's outcome."""
    return any(
        row.status is TranscriptionStatus.failed
        for row in _rows_for_job(session, audio_file_id, job)
    )


def _is_terminal_attempt(job: Job) -> bool:
    """Whether this claim is the job's last allowed attempt."""
    effective_max = (
        job.max_attempts if job.max_attempts is not None else get_settings().job_max_attempts
    )
    return job.attempts >= effective_max


def _profile_metadata(profile: FasterWhisperProfile) -> dict[str, object]:
    """Non-sensitive run snapshot for a transcription row."""
    return {
        "profile_ref": PROFILE_REF_TRANSCRIPTION,
        "device": profile.device,
        "compute_type": profile.compute_type,
        "beam_size": profile.beam_size,
    }


def _provider_version(provider: TranscriptionProvider) -> str | None:
    """Optional ``version`` attribute of a provider instance (never required)."""
    value = getattr(provider, "version", None)
    if callable(value):
        value = value()
    return value if isinstance(value, str) and value else None


def _record_success_row(
    session: Session,
    job: Job,
    provider: TranscriptionProvider,
    profile: FasterWhisperProfile,
    audio_file_id: uuid.UUID,
    raw_text: str,
) -> None:
    """Insert one ``succeeded`` row for this job (flush; caller commits)."""
    metadata = _profile_metadata(profile)
    metadata["job_id"] = str(job.id)
    provider_version = _provider_version(provider)
    if provider_version is not None:
        metadata["provider_version"] = provider_version
    TranscriptionRepository(session).create(
        audio_file_id=audio_file_id,
        raw_text=raw_text,
        status=TranscriptionStatus.succeeded,
        profile_name=PROFILE_REF_TRANSCRIPTION,
        provider_name=profile.provider,
        model_name=profile.model,
        language=profile.language,
        metadata_json=metadata,
    )


def _record_failed_row(
    session: Session,
    job: Job,
    profile: FasterWhisperProfile,
    audio_file_id: uuid.UUID,
) -> None:
    """Insert the single terminal ``failed`` row for this job (no error text).

    The safe error summary lives on the job record (``jobs.error_summary``),
    written by the worker; the transcription row keeps only the status and the
    profile/run snapshot, so no diary-adjacent details end up in the history.
    """
    metadata = _profile_metadata(profile)
    metadata["job_id"] = str(job.id)
    metadata["attempts"] = job.attempts
    metadata["terminal"] = True
    TranscriptionRepository(session).create(
        audio_file_id=audio_file_id,
        raw_text="",
        status=TranscriptionStatus.failed,
        profile_name=PROFILE_REF_TRANSCRIPTION,
        provider_name=profile.provider,
        model_name=profile.model,
        language=profile.language,
        metadata_json=metadata,
    )


__all__ = [
    "PROFILE_REF_TRANSCRIPTION",
    "TranscriptionPermanentError",
    "TranscriptionServiceError",
    "find_active_transcription_job",
    "list_transcription_history",
    "request_transcription",
    "run_transcription",
]
