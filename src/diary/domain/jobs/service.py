"""Durable-job retry use case (Task 16).

``POST /api/v1/jobs/{job_id}/retry`` lets the UI offer an explicit "retry"
action for jobs that ended in a terminal *failure* state (``failed`` or
``cancelled``).

Semantics (deliberate, consistent with Task 08/09):

* the original terminal job row is left untouched — history is never erased;
* a **brand-new** ``pending`` job is enqueued with the same ``job_type`` and
  the same ``payload_json`` (which holds identifiers only, never diary
  content — AGENTS.md §8);
* the new job deliberately carries **no ``idempotency_key``**, so it never
  aliases or replays the original row and always gets a fresh attempt budget
  (``max_attempts`` defaults to the configured value);
* ``metadata_json`` is not copied: worker/job-specific metadata of the old
  run must not leak into the retry.

The only persistence mechanics used are the existing Task-08 repository
operations (``get`` + ``enqueue``); the retry *policy* (which terminal states
are retryable, what is copied) lives here so the router stays thin. This
service commits its transaction (the new job row is the unit of work).
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from diary.db.models import Job, JobStatus
from diary.db.repositories import JobRepository


class JobNotFoundError(Exception):
    """No job row exists for the given id (HTTP 404)."""


class JobNotRetryableError(Exception):
    """The job is not in a retryable terminal state (HTTP 409)."""


def retry_job(session: Session, *, job_id: uuid.UUID) -> Job:
    """Re-enqueue a failed/cancelled job as a fresh job; return the new job.

    Raises :class:`JobNotFoundError` for an unknown id and
    :class:`JobNotRetryableError` when the job has not ended in a retryable
    terminal state (``failed`` or ``cancelled``). Commits the new job row.
    """
    repository = JobRepository(session)
    existing = repository.get(job_id)
    if existing is None:
        raise JobNotFoundError("Job not found.")
    if existing.status not in (JobStatus.failed, JobStatus.cancelled):
        raise JobNotRetryableError(
            "Only failed or cancelled jobs can be retried; "
            "this job is still in progress or already succeeded."
        )
    payload = dict(existing.payload_json) if existing.payload_json is not None else None
    new_job = repository.enqueue(
        job_type=existing.job_type,
        payload_json=payload,
        # No idempotency_key and no metadata copy: the retry is a fresh,
        # independent job (see the module docstring for the semantics).
    )
    session.commit()
    return new_job


__all__ = ["JobNotFoundError", "JobNotRetryableError", "retry_job"]
