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

Task 08 ships no real processing (STT is Task 09, LLM pipelines arrive with
Task 11+). ``transcribe_audio`` is registered with an explicit placeholder that
fails permanently with a clear message instead of silently succeeding; the
``normalize``/``summarize``/``wellbeing`` types have no handler yet and are
rejected by the worker as unknown job types (also permanent failure).
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from diary.db.models import Job, JobType


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
# Task 08 placeholders
# ---------------------------------------------------------------------------


def _transcribe_audio_not_implemented(session: Session, job: Job) -> None:
    """Task 08 placeholder: real STT arrives with Task 09.

    The job fails permanently with a clear, non-sensitive summary; a retry
    would never succeed, so it must not burn ``max_attempts`` in backoff.
    """
    raise PermanentJobError(
        "The transcribe_audio handler is not implemented yet "
        "(planned in Task 09); the job was failed permanently."
    )


register(JobType.transcribe_audio, _transcribe_audio_not_implemented)
