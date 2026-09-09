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

``transcribe_audio`` (Task 09) is registered to the real STT pipeline: the
handler delegates to :func:`diary.domain.transcription.service.run_transcription`
and translates deterministic domain failures (missing audio, invalid payload,
provider not installed, broken model configuration) into a permanent failure.
The ``normalize``/``summarize``/``wellbeing`` types have no handler yet and are
rejected by the worker as unknown job types (also permanent failure).
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from diary.core.models_config import ModelsConfigError
from diary.db.models import Job, JobType
from diary.domain.transcription import service as transcription_service
from diary.domain.transcription.providers import ProviderUnavailableError


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


register(JobType.transcribe_audio, _transcribe_audio)
