"""Thin HTTP routes for voice messages (Task 07).

- ``POST /api/v1/audio``                 — multipart upload of one audio file;
  201 with the stored voice-message metadata, 400/413/422 for invalid input;
- ``GET /api/v1/audio/{audio_file_id}/file`` — stream the stored bytes back
  for in-page playback (``audio/*`` media type; 404 for unknown ids).

Routers stay thin: HTTP/upload boundary concerns live here, validation,
atomic storage and the single-transaction persistence live in
:mod:`diary.domain.audio.service`. The user-supplied file name is deliberately
never used — only the server-generated UUID path is stored (ADR-002).

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
from diary.api.schemas import AudioRead, audio_read_from
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
