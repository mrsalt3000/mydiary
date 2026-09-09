"""Thin HTTP routes for job status polling (Task 08).

- ``GET /api/v1/jobs``       — recent jobs, newest first (id/type/status/
  attempts/timestamps only, in the application timezone);
- ``GET /api/v1/jobs/{id}``  — one job's status surface plus its safe
  ``error_summary``; 404 for unknown ids, 422 for malformed UUIDs.

Routers stay thin: the handlers only map a repository read to a schema.
Privacy rule: job ``payload_json``/``metadata_json`` are never returned —
responses carry the status surface only (no diary content, no raw payload).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.api.schemas import JobPage, JobRead, job_read_from
from diary.db.repositories import JobRepository

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
