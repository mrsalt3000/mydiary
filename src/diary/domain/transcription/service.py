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
  configured provider **through the sequential access point**
  :func:`diary.domain.transcription.asr.run` and persists the result.

Sequential GPU access (task T11, ADR-010)
=========================================

The manual «Расшифровать» button of phase 1 reaches this module through a
durable ``transcribe_audio`` job, so this function is the path the phase-2 veto
«Whisper and the LLM are never resident in VRAM at the same time» has to hold
for. It therefore never calls the provider directly: speech-to-text runs inside
:func:`~diary.domain.transcription.asr.run`, which takes the process-wide GPU
slot, calls the **injected** ``unload_llm`` callable first and releases Whisper
afterwards. ``unload_llm`` is a parameter (default:
:func:`_default_llm_unloader`, the production unloader built from the model
configuration by :mod:`diary.domain.llm.unload`); the ASR module and this
service name no LLM tool, so a test can assert the order with a fake callable.

The provider instance this module creates is used **only** for its optional
provenance hook (``version``) and never loads a model — loading happens lazily
inside the transcription itself.

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
from collections.abc import Callable, Collection

from sqlalchemy.orm import Session

from diary.core.models_config import (
    GigaamCtcLmProfile,
    TranscriptionProfile,
    load_models_config,
    transcription_model_name,
)
from diary.core.settings import get_settings
from diary.db.models import Job, JobType, Transcription, TranscriptionStatus
from diary.db.repositories import (
    AudioFileRepository,
    JobRepository,
    TranscriptionRepository,
)
from diary.domain.audio.service import AudioNotFoundError, resolve_audio_file
from diary.domain.llm.unload import unload_local_llms
from diary.domain.transcription import asr
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


def _load_transcription_profile() -> TranscriptionProfile:
    """Return the current ``transcription`` profile from the models config.

    The profile is whichever STT backend the configuration names —
    ``faster_whisper`` or ``gigaam2_ctc_lm`` (T43). Module-level hook so tests
    can substitute a profile without a config file (no models YAML is needed in
    CI; monkeypatched in integration tests).
    """
    return load_models_config().transcription


def _default_llm_unloader() -> None:
    """The production ``unload_llm`` callable of a manual STT turn (ADR-010).

    Evicts every configured local LLM (``keep_alive: 0`` == ``ollama stop``)
    before Whisper starts, so the phase-2 VRAM veto no longer depends on the
    server being started with ``OLLAMA_KEEP_ALIVE=0``. The model configuration
    is read on every call — the worker is long-lived, and
    :func:`~diary.domain.llm.unload.unload_local_llms` is exactly the "config at
    call time" unload.

    Module-level for the same reason as ``_load_transcription_profile``: tests
    replace it with a fake and never contact (nor unload) a real LLM server.
    """
    unload_local_llms()


def find_active_transcription_jobs(
    session: Session, audio_file_ids: Collection[uuid.UUID]
) -> dict[uuid.UUID, Job]:
    """The in-flight ``transcribe_audio`` jobs of many audio files, in one query.

    Batch counterpart of :func:`find_active_transcription_job` (task T31): the
    whole day's worth of audio files is resolved with a single read of the
    active ``transcribe_audio`` jobs instead of one read per file. The single
    lookup delegates here, so the two can never disagree about which job is in
    flight for a file.

    An audio file with several in-flight jobs keeps the **oldest** one — the
    same job :func:`find_active_transcription_job` returns, because both walk
    the same queue order (``created_at``, then ``id``). Files without an
    in-flight job are absent from the mapping; payloads of other job types and
    malformed payloads are skipped.
    """
    wanted = {str(audio_file_id) for audio_file_id in audio_file_ids}
    if not wanted:
        return {}
    active: dict[uuid.UUID, Job] = {}
    for job in JobRepository(session).list_active_of_type(JobType.transcribe_audio):
        payload = job.payload_json if isinstance(job.payload_json, dict) else {}
        # The very condition the single-file lookup uses (``str(payload) ==
        # str(audio_file_id)``), so both paths match the same payloads.
        raw = str(payload.get("audio_file_id", ""))
        if raw not in wanted:
            continue
        try:
            key = uuid.UUID(raw)
        except ValueError:  # pragma: no cover - ``wanted`` holds canonical UUIDs
            continue
        # First match wins: the queue is oldest first, so the mapping agrees
        # with the single-file lookup.
        active.setdefault(key, job)
    return active


def find_active_transcription_job(session: Session, audio_file_id: uuid.UUID) -> Job | None:
    """The in-flight ``transcribe_audio`` job for ``audio_file_id``, if any.

    A job is "in flight" when it is ``pending``, ``running`` or ``retrying``.
    Terminal jobs do not count, so an explicit repeat request after a success
    or a failure starts a fresh run.
    """
    return find_active_transcription_jobs(session, (audio_file_id,)).get(audio_file_id)


def request_transcription(
    session: Session,
    audio_file_id: uuid.UUID,
    *,
    profile_ref: str = PROFILE_REF_TRANSCRIPTION,
) -> Job:
    """Queue a manual transcription for one voice message; return the job.

    Raises :class:`~diary.domain.audio.service.AudioNotFoundError` (→ HTTP 404)
    when the audio cannot be resolved: no audio-file row exists, the message has
    been deleted (ADR-011 / task T15 — a deleted message must not get a new
    transcription job), or the stored ``storage_path`` is unusable. The check
    goes through :func:`~diary.domain.audio.service.resolve_audio_file`, the
    same resolver playback uses, so the route refuses with a 404 instead of
    enqueuing a job that would only fail later. When a job for the audio file is
    already in flight the existing job is returned unchanged — no duplicate row,
    and STT never runs inside the HTTP request.
    """
    resolve_audio_file(session, audio_file_id)
    active = find_active_transcription_job(session, audio_file_id)
    if active is not None:
        return active
    job = JobRepository(session).enqueue(
        job_type=JobType.transcribe_audio,
        payload_json={"audio_file_id": str(audio_file_id), "profile_ref": profile_ref},
    )
    session.commit()
    return job


def run_transcription(
    session: Session,
    job: Job,
    *,
    unload_llm: Callable[[], None] | None = None,
) -> None:
    """Execute one claimed ``transcribe_audio`` job (worker handler core).

    Speech-to-text runs through :func:`diary.domain.transcription.asr.run`,
    which holds the exclusive GPU slot and calls ``unload_llm`` **before** any
    recognition work, so the phase-2 veto holds for the manual button too
    (ADR-010). ``unload_llm`` defaults to the production unloader built from
    the model configuration; tests inject a fake callable (see
    :func:`_default_llm_unloader`).

    The caller (``diary.workers.runner``) owns the transaction: on a normal
    return it commits the session and marks the job ``succeeded``; on an
    exception it applies the retry policy. This function commits itself only
    when it writes the single terminal ``failed`` row on the final attempt
    (the runner would otherwise roll the handler session back after the raise).

    A failed LLM unload raises :class:`~diary.domain.transcription.asr.LlmUnloadError`
    (no STT is started; the handler/core records no result row and the retry
    policy applies), which is the loud direction of the veto.

    A configured STT backend with no registered adapter (``gigaam2_ctc_lm``
    until T44) makes :func:`~diary.domain.transcription.providers.create_provider`
    raise :class:`~diary.domain.transcription.providers.TranscriptionProviderError`.
    This service resolves the provider itself, **before** calling
    :func:`~diary.domain.transcription.asr.run`, so for this path the refusal
    surfaces before the GPU slot is taken and before any result row is written.
    (``asr.run`` takes the slot itself when called directly and releases it in
    its ``finally``.) Either way it is the explicit failure ADR-017 requires
    instead of a silent fallback to Whisper.
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
    # Provenance probe only: it reads the adapter's optional ``version`` hook
    # (an installed-distribution lookup) and loads nothing — the transcription
    # itself runs on the instance ``asr.run`` creates under the GPU slot.
    provider = create_provider(profile)
    resolved_unload = unload_llm if unload_llm is not None else _default_llm_unloader
    try:
        raw_text = asr.run(audio_path, profile, unload_llm=resolved_unload)
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


def _profile_metadata(profile: TranscriptionProfile) -> dict[str, object]:
    """Non-sensitive run snapshot for a transcription row.

    Provider-dependent on purpose: the Whisper snapshot keeps exactly the keys
    history already carries (``device``/``compute_type``/``beam_size``), so rows
    written before T43 read as before, while the GigaAM-v2 CTC+LM backend
    records its own decoding knobs (``beam_width``/``alpha``/``beta``) and the
    pinned ``revision``. Nothing here can carry diary content.
    """
    if isinstance(profile, GigaamCtcLmProfile):
        return {
            "profile_ref": PROFILE_REF_TRANSCRIPTION,
            "device": profile.device,
            "beam_width": profile.beam_width,
            "alpha": profile.alpha,
            "beta": profile.beta,
            "revision": profile.revision,
        }
    return {
        "profile_ref": PROFILE_REF_TRANSCRIPTION,
        "device": profile.device,
        "compute_type": profile.compute_type,
        "beam_size": profile.beam_size,
    }


#: Recognition language of the GigaAM-v2 CTC+LM backend. Russian only, so it is
#: a property of the provider rather than a profile knob — the same reasoning
#: that keeps ``sample_rate`` out of :class:`GigaamCtcLmProfile`.
_GIGAAM_LANGUAGE = "ru"


def _profile_language(profile: TranscriptionProfile) -> str:
    """The language code recorded on a transcription row.

    Whisper carries it in the profile; GigaAM is Russian-only, so its constant
    is used (``transcriptions.language`` is NOT NULL).
    """
    if isinstance(profile, GigaamCtcLmProfile):
        return _GIGAAM_LANGUAGE
    return profile.language


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
    profile: TranscriptionProfile,
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
        model_name=transcription_model_name(profile),
        language=_profile_language(profile),
        metadata_json=metadata,
    )


def _record_failed_row(
    session: Session,
    job: Job,
    profile: TranscriptionProfile,
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
        model_name=transcription_model_name(profile),
        language=_profile_language(profile),
        metadata_json=metadata,
    )


__all__ = [
    "PROFILE_REF_TRANSCRIPTION",
    "TranscriptionPermanentError",
    "TranscriptionServiceError",
    "find_active_transcription_job",
    "find_active_transcription_jobs",
    "list_transcription_history",
    "request_transcription",
    "run_transcription",
]
