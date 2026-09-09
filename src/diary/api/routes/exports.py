"""Thin HTTP routes for weekly original exports (Task 10).

- ``POST /api/v1/exports/weeks/{sunday_date}``               — create the first
  original export of a week (201); ``sunday_date`` must be a Sunday; an empty
  week or an already-exported week is rejected without creating anything;
- ``POST /api/v1/exports/weeks/{sunday_date}/re-export``     — explicit action
  that creates the next original revision of the week (201); never overwrites
  earlier files;
- ``GET /api/v1/exports``                                    — list exports
  (most recent week first) without any diary content — used for links and by
  Task 16;
- ``GET /api/v1/exports/{export_id}/file``                   — download one
  export as ``text/plain``.

The handlers stay thin: week math, numbering/revisions, rendering and the
atomic file + database writes live in :mod:`diary.domain.exports.service`.
Error details are static and never echo diary content or file paths
(AGENTS.md §8). Status codes: 422 for a non-Sunday date, 409 for empty /
already-exported / never-exported week states, 404 for unknown ids or missing
files.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.api.schemas import ExportRead, export_read_from
from diary.domain.exports.service import (
    EmptyWeekError,
    ExportNotFoundError,
    ExportStorageError,
    InvalidSundayDateError,
    WeekAlreadyExportedError,
    WeekNotExportedError,
    export_week,
    list_exports,
    resolve_export_file,
)

router = APIRouter(prefix="/exports", tags=["exports"])


@router.post("/weeks/{sunday_date}", response_model=ExportRead, status_code=201)
def create_weekly_export(
    sunday_date: date,
    session: Annotated[Session, Depends(get_session)],
) -> ExportRead:
    """Create the first original export of the week of ``sunday_date``."""
    try:
        record = export_week(session, sunday_date=sunday_date)
    except InvalidSundayDateError as exc:
        raise HTTPException(
            status_code=422,
            detail="sunday_date must be the Sunday date of the week to export.",
        ) from exc
    except EmptyWeekError as exc:
        raise HTTPException(
            status_code=409,
            detail="This week has no messages, so no export was created.",
        ) from exc
    except WeekAlreadyExportedError as exc:
        raise HTTPException(
            status_code=409,
            detail="This week is already exported; use POST .../re-export "
            "to create a new revision.",
        ) from exc
    except ExportStorageError as exc:
        raise HTTPException(status_code=500, detail="Failed to create the weekly export.") from exc
    return export_read_from(record)


@router.post("/weeks/{sunday_date}/re-export", response_model=ExportRead, status_code=201)
def re_export_week(
    sunday_date: date,
    session: Annotated[Session, Depends(get_session)],
) -> ExportRead:
    """Create the next original revision (rN+1) of an already exported week.

    This is the explicit re-export action: the lineage keeps its sequence
    number and the previous revision file is never modified.
    """
    try:
        record = export_week(session, sunday_date=sunday_date, re_export=True)
    except InvalidSundayDateError as exc:
        raise HTTPException(
            status_code=422,
            detail="sunday_date must be the Sunday date of the week to export.",
        ) from exc
    except EmptyWeekError as exc:
        raise HTTPException(
            status_code=409,
            detail="This week has no messages, so no export was created.",
        ) from exc
    except WeekNotExportedError as exc:
        raise HTTPException(
            status_code=409,
            detail="This week has no export yet; create the first export first.",
        ) from exc
    except WeekAlreadyExportedError as exc:
        raise HTTPException(
            status_code=409,
            detail="This week and revision are already exported.",
        ) from exc
    except ExportStorageError as exc:
        raise HTTPException(status_code=500, detail="Failed to create the weekly export.") from exc
    return export_read_from(record)


@router.get("", response_model=list[ExportRead])
def read_exports(
    session: Annotated[Session, Depends(get_session)],
) -> list[ExportRead]:
    """List every original export, most recent week first, newest revision first."""
    return [export_read_from(record) for record in list_exports(session)]


@router.get("/{export_id}/file")
def download_export_file(
    export_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> FileResponse:
    """Download one original export as a plain-text file (server-side path only)."""
    try:
        path, filename = resolve_export_file(session, export_id)
    except ExportNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Export not found.") from exc
    if not path.is_file():
        # Row present but the artifact is gone: still a plain 404.
        raise HTTPException(status_code=404, detail="Export file not found.")
    return FileResponse(str(path), media_type="text/plain", filename=filename)
