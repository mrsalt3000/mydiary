"""Thin HTTP routes for voice messages (Tasks 07/09).

- ``POST /api/v1/audio``                          — multipart upload of one audio
  file; 201 with the stored voice-message metadata, 400/413/422 for invalid input;
- ``GET /api/v1/audio/{audio_file_id}/file``      — stream the stored bytes back
  for in-page playback (``audio/*`` media type; 404 for unknown ids);
- ``POST /api/v1/audio/{audio_file_id}/transcribe`` — the explicit manual
  transcription action (Task 09): enqueues a durable ``transcribe_audio`` job
  and returns it (202). STT never runs inside the HTTP request; a repeated
  request while a job is in flight returns the existing job (no duplicate);
- ``GET /api/v1/audio/{audio_file_id}/transcriptions`` — raw-result history of
  real transcription runs (succeeded/failed), oldest first; ``raw_text`` is
  present on succeeded rows only.

Routers stay thin: HTTP/upload boundary concerns live here, validation,
atomic storage and the single-transaction persistence live in
:mod:`diary.domain.audio.service`, and transcription request/history logic in
:mod:`diary.domain.transcription.service`. The user-supplied file name is
deliberately never used — only the server-generated UUID path is stored
(ADR-002).

Error details are static and never echo uploaded content or file names
(AGENTS.md §8).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.api.schemas import (
    AudioRead,
    JobRead,
    TranscriptionRead,
    audio_read_from,
    job_read_from,
    transcription_read_from,
)
from diary.core.settings import get_settings
from diary.domain.audio.service import (
    AudioNotFoundError,
    AudioStorageError,
    AudioTooLargeError,
    EmptyAudioError,
    InvalidAudioContentError,
    UnsupportedAudioTypeError,
    resolve_audio_file,
    store_audio,
)
from diary.domain.transcription import service as transcription_service

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("", response_model=AudioRead, status_code=201)
def upload_audio(
    file: Annotated[UploadFile, File(description="The audio file to store.")],
    session: Annotated[Session, Depends(get_session)],
) -> AudioRead:
    """Store an uploaded voice message and return its metadata.

    The size limit is enforced while reading: at most limit+1 bytes are pulled
    from the stream, so an oversized upload is rejected without buffering the
    whole payload in memory.
    """
    settings = get_settings()
    limit = settings.max_audio_size_bytes
    data = file.file.read(limit + 1) if file.file is not None else b""
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file is too large (limit is {settings.max_audio_size_mb} MiB).",
        )
    try:
        stored = store_audio(session, data=data, content_type=file.content_type)
    except UnsupportedAudioTypeError as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio type.") from exc
    except EmptyAudioError as exc:
        raise HTTPException(status_code=400, detail="Audio file is empty.") from exc
    except InvalidAudioContentError as exc:
        raise HTTPException(
            status_code=400, detail="The file does not look like the declared audio type."
        ) from exc
    except AudioTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file is too large (limit is {settings.max_audio_size_mb} MiB).",
        ) from exc
    except AudioStorageError as exc:
        raise HTTPException(status_code=500, detail="Failed to store the audio file.") from exc
    return audio_read_from(
        audio_file_id=stored.audio_file_id,
        message_id=stored.message_id,
        mime=stored.original_mime,
        size_bytes=stored.size_bytes,
        transcription_status=stored.transcription_status,
        created_at=stored.created_at,
    )


@router.get("/{audio_file_id}/file")
def read_audio_file(
    audio_file_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> FileResponse:
    """Stream a stored audio file for playback (server-side path only)."""
    try:
        path, mime = resolve_audio_file(session, audio_file_id)
    except AudioNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Audio file not found.") from exc
    if not path.is_file():
        # Row present but the bytes are gone: still a plain 404, nothing more.
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(str(path), media_type=mime)


@router.post("/{audio_file_id}/transcribe", response_model=JobRead, status_code=202)
def start_transcription(
    audio_file_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> JobRead:
    """Queue a manual transcription for one voice message (Task 09).

    This is the explicit Transcribe/Retranscribe action: a durable
    ``transcribe_audio`` job is created and the request returns immediately
    (202) — STT never runs inside the HTTP request. When a job for this audio
    file is already in flight the existing job is returned instead of creating
    a duplicate; after a terminal outcome an explicit repeat request starts a
    fresh run (retranscription keeps the prior raw history intact).
    """
    try:
        job = transcription_service.request_transcription(session, audio_file_id)
    except AudioNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Audio file not found.") from exc
    return job_read_from(job)


@router.get("/{audio_file_id}/transcriptions", response_model=list[TranscriptionRead])
def read_transcriptions(
    audio_file_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> list[TranscriptionRead]:
    """Raw-result history of one voice message, oldest first (Task 09).

    Returns real transcription runs only (the upload-time ``not_requested``
    sentinel is not a result). ``raw_text`` is non-empty for succeeded rows
    only; failed rows return an empty string and their safe error summary
    lives on the corresponding job (``GET /api/v1/jobs/{id}``).
    """
    try:
        rows = transcription_service.list_transcription_history(session, audio_file_id)
    except AudioNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Audio file not found.") from exc
    return [transcription_read_from(row) for row in rows]
