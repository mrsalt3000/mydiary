"""Thin HTTP routes for job status polling (Task 08) and retry (Task 16).

- ``GET /api/v1/jobs``       — recent jobs, newest first (id/type/status/
  attempts/timestamps only, in the application timezone);
- ``GET /api/v1/jobs/{id}``  — one job's status surface plus its safe
  ``error_summary``; 404 for unknown ids, 422 for malformed UUIDs;
- ``POST /api/v1/jobs/{id}/retry`` — Task 16: re-enqueue a ``failed`` or
  ``cancelled`` job as a fresh pending job (202 + the new ``JobRead``); 404
  for unknown ids, 409 when the job is not in a retryable terminal state.

Routers stay thin: the handlers only map a service/repository call to a
schema. Privacy rule: job ``payload_json``/``metadata_json`` are never
returned — responses carry the status surface only (no diary content, no raw
payload).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.api.schemas import JobPage, JobRead, job_read_from
from diary.db.repositories import JobRepository
from diary.domain.jobs.service import JobNotFoundError, JobNotRetryableError, retry_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobPage)
def list_jobs(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> JobPage:
    """Return the most recent jobs, newest first, without any payload data."""
    jobs = JobRepository(session).list_recent(limit=limit)
    return JobPage(items=[job_read_from(job) for job in jobs], limit=limit)


@router.get("/{job_id}", response_model=JobRead)
def get_job(
    job_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> JobRead:
    """Return the polling status of one job (safe error summary included)."""
    job = JobRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_read_from(job)


@router.post("/{job_id}/retry", response_model=JobRead, status_code=202)
def retry_failed_job(
    job_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> JobRead:
    """Re-enqueue a failed/cancelled job as a fresh job (explicit UI action).

    The retry is a new pending job of the same type with the same identifier
    payload and a fresh attempt budget (see
    :func:`diary.domain.jobs.service.retry_job`); the original terminal job
    stays unchanged. The response is the new job's status surface only.
    """
    try:
        job = retry_job(session, job_id=job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except JobNotRetryableError as exc:
        raise HTTPException(
            status_code=409,
            detail="Only failed or cancelled jobs can be retried.",
        ) from exc
    return job_read_from(job)
