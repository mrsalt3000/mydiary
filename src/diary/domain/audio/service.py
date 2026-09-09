"""Voice-message ingestion and storage use cases (Task 07).

Responsibilities
================

* Validate uploads server-side: MIME type against ``AppSettings`` allowlist,
  size against ``APP_MAX_AUDIO_MB``, and — where cheaply possible without a
  media library — container magic bytes so the payload matches the declared
  type.
* Store audio bytes atomically under ``<audio_dir>/YYYY/MM/<server-uuid>.<ext>``
  where ``<ext>`` comes from a fixed MIME→extension mapping and the file name is
  always a server-generated UUID. User-supplied file names and paths are never
  used (no path-traversal surface, ADR-002).
* Create, in a single transaction, the ``messages(kind=audio)`` row, the
  ``audio_files`` metadata row (MIME, size, SHA-256, server relative path) and
  the initial ``transcriptions`` row with status ``not_requested``. No
  transcription is ever started here (manual only, Task 09).
* Clean up the written file when the database transaction fails, so no orphaned
  "half-finished" audio remains (AGENTS.md §12, ``docs/data-lineage.md`` §5).
* Resolve stored audio by id for playback, returning only server-side paths.

Limitations (documented)
========================

* Duration (``audio_files.duration_seconds``) stays ``NULL``: computing it
  reliably needs a media demuxer (ffprobe/ffmpeg), which is a heavy dependency
  out of MVP scope. The metadata row keeps the column nullable.
* Container validation is a light magic-byte check for each supported family
  (EBML/WebM, Ogg, RIFF/WAVE, MPEG frame sync or ID3, ISO-BMFF "ftyp"). It does
  not decode the stream; a payload with a valid header but corrupt body would
  still be stored and only fail later at transcription time.
* Only MIME types with a built-in profile below can be stored. An allowlist
  entry without a profile (e.g. an exotic type added via environment) is
  rejected at upload time with a clear error rather than stored under a guessed
  extension.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from diary.core.settings import AppSettings, get_settings
from diary.db.base import utc_now
from diary.db.models import MessageKind, TranscriptionStatus
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    TranscriptionRepository,
)

# ---------------------------------------------------------------------------
# Domain errors (mapped to HTTP responses at the API boundary)
# ---------------------------------------------------------------------------


class AudioError(Exception):
    """Base class for audio-ingestion domain errors (never diary content)."""


class UnsupportedAudioTypeError(AudioError):
    """The MIME type is not in the allowlist or has no safe storage profile."""


class EmptyAudioError(AudioError):
    """The uploaded file contains no bytes."""


class AudioTooLargeError(AudioError):
    """The uploaded file exceeds ``AppSettings.max_audio_size_mb``."""


class InvalidAudioContentError(AudioError):
    """The payload does not match the container signature of the declared MIME."""


class AudioNotFoundError(AudioError):
    """No audio-file row with the given id exists."""


class AudioStorageError(AudioError):
    """The filesystem write or the persistence transaction failed."""


# ---------------------------------------------------------------------------
# MIME profiles: extension + lightweight container signature check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MimeProfile:
    """Safe file extension and (optional) magic-byte checker for one MIME."""

    ext: str
    checker: Callable[[bytes], bool] | None = None


def _is_ebml(data: bytes) -> bool:
    """WebM/Matroska: EBML magic ``1A 45 DF A3`` at offset 0."""
    return data.startswith(b"\x1a\x45\xdf\xa3")


def _is_ogg(data: bytes) -> bool:
    """Ogg container: ``OggS`` capture pattern."""
    return data.startswith(b"OggS")


def _is_wav(data: bytes) -> bool:
    """RIFF/WAVE: ``RIFF`` at 0 and ``WAVE`` at offset 8."""
    return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE"


def _is_mp3(data: bytes) -> bool:
    """MP3: an ID3 tag or an MPEG audio frame-sync header (11 set bits)."""
    if data.startswith(b"ID3"):
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _is_iso_bmff(data: bytes) -> bool:
    """MP4/M4A: ISO-BMFF ``....ftyp`` at offset 4."""
    return len(data) >= 8 and data[4:8] == b"ftyp"


#: Built-in storage profiles. ``audio/webm;codecs=opus`` arrives as the bare
#: ``audio/webm`` after parameter stripping (see :func:`normalize_mime`).
_AUDIO_MIME_PROFILES: dict[str, _MimeProfile] = {
    "audio/webm": _MimeProfile("webm", _is_ebml),
    "audio/ogg": _MimeProfile("ogg", _is_ogg),
    "audio/wav": _MimeProfile("wav", _is_wav),
    "audio/x-wav": _MimeProfile("wav", _is_wav),
    "audio/wave": _MimeProfile("wav", _is_wav),
    "audio/mpeg": _MimeProfile("mp3", _is_mp3),
    "audio/mp3": _MimeProfile("mp3", _is_mp3),
    "audio/mp4": _MimeProfile("m4a", _is_iso_bmff),
    "audio/x-m4a": _MimeProfile("m4a", _is_iso_bmff),
    "audio/mp4a-latm": _MimeProfile("m4a", _is_iso_bmff),
}

#: MIME types that can actually be stored (public, for tests/docs).
SUPPORTED_AUDIO_MIME_TYPES: frozenset[str] = frozenset(_AUDIO_MIME_PROFILES)


# ---------------------------------------------------------------------------
# Validation helpers (pure, unit-testable)
# ---------------------------------------------------------------------------


def normalize_mime(content_type: str | None) -> str | None:
    """Bare lower-case MIME type without ``;codecs=...`` parameters.

    ``None`` for an empty/whitespace value. ``"audio/webm;codecs=opus"``
    becomes ``"audio/webm"``; the codec parameter is browser/OS-dependent and
    is not part of the allowlist.
    """
    if content_type is None:
        return None
    bare = content_type.split(";", 1)[0].strip().lower()
    return bare or None


def validate_upload(
    content_type: str | None,
    data: bytes,
    *,
    settings: AppSettings | None = None,
) -> str:
    """Validate an upload and return its normalized MIME type.

    Raises :class:`UnsupportedAudioTypeError` for types outside the allowlist
    or with no storage profile, :class:`EmptyAudioError` for empty payloads,
    :class:`AudioTooLargeError` when the size exceeds the configured limit and
    :class:`InvalidAudioContentError` when the magic bytes contradict the
    declared type.
    """
    resolved = settings or get_settings()
    mime = normalize_mime(content_type)
    if mime is None or mime not in resolved.allowed_audio_mime_types:
        raise UnsupportedAudioTypeError("Unsupported audio type.")
    profile = _AUDIO_MIME_PROFILES.get(mime)
    if profile is None:
        # An allowlist entry without a safe storage profile cannot be stored
        # under a server-generated extension; reject rather than guess.
        raise UnsupportedAudioTypeError("Unsupported audio type.")
    if not data:
        raise EmptyAudioError("Audio file is empty.")
    if len(data) > resolved.max_audio_size_bytes:
        raise AudioTooLargeError(
            f"Audio file is too large (limit is {resolved.max_audio_size_mb} MiB)."
        )
    if profile.checker is not None and not profile.checker(data):
        raise InvalidAudioContentError("The uploaded data does not match the declared audio type.")
    return mime


def extension_for_mime(content_type: str | None) -> str | None:
    """Safe file extension for a MIME type, or ``None`` when unknown."""
    mime = normalize_mime(content_type)
    if mime is None:
        return None
    profile = _AUDIO_MIME_PROFILES.get(mime)
    return profile.ext if profile is not None else None


# ---------------------------------------------------------------------------
# Result value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredAudio:
    """What :func:`store_audio` returns after a successful commit."""

    audio_file_id: uuid.UUID
    message_id: uuid.UUID
    original_mime: str
    storage_path: str
    size_bytes: int
    sha256: str
    transcription_status: TranscriptionStatus
    created_at: datetime


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _unlink_safe(path: Path) -> None:
    """Remove a file, ignoring "already gone" errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _write_file_atomically(data: bytes, temp_path: Path, target_path: Path) -> None:
    """Write ``data`` to ``target_path`` atomically (AGENTS.md §12).

    The payload is written to a temporary file inside ``AppSettings.temp_dir``,
    flushed and fsynced, then renamed over the final path. When the temporary
    directory and the audio directory live on different filesystems (``rename``
    raises ``EXDEV``) a copy+delete fallback is used — the payload is already
    fsynced, so durability does not depend on the rename being atomic.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(temp_path, flags, 0o600)
    except OSError:
        raise
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _unlink_safe(temp_path)
        raise
    try:
        os.replace(temp_path, target_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            _unlink_safe(temp_path)
            raise
        try:
            shutil.move(str(temp_path), str(target_path))
        except BaseException:
            _unlink_safe(target_path)
            _unlink_safe(temp_path)
            raise


# ---------------------------------------------------------------------------
# Use cases
# ---------------------------------------------------------------------------


def store_audio(
    session: Session,
    *,
    data: bytes,
    content_type: str | None,
) -> StoredAudio:
    """Validate, store and persist one voice message (single transaction).

    File layout: ``<audio_dir>/YYYY/MM/<server-uuid>.<ext>`` where ``YYYY/MM``
    is the upload moment in the application timezone and the extension comes
    from the fixed MIME→extension mapping. On any persistence failure the
    already-written file is removed so no orphaned audio stays behind.
    """
    settings = get_settings()
    mime = validate_upload(content_type, data, settings=settings)
    profile = _AUDIO_MIME_PROFILES[mime]
    assert profile.ext

    now = utc_now()
    local_now = now.astimezone(settings.tzinfo)
    folder = local_now.strftime("%Y/%m")
    audio_file_id = uuid.uuid4()
    relative_path = f"{folder}/{audio_file_id}.{profile.ext}"

    digest = hashlib.sha256(data).hexdigest()
    temp_path = settings.temp_dir / f"audio-{audio_file_id}.tmp"
    target_path = settings.audio_dir / relative_path

    _write_file_atomically(data, temp_path, target_path)

    try:
        messages = MessageRepository(session)
        audio_files = AudioFileRepository(session)
        transcriptions = TranscriptionRepository(session)

        message = messages.create(kind=MessageKind.audio, body_text="", created_at=now)
        audio_file = audio_files.create(
            message_id=message.id,
            original_mime=mime,
            storage_path=relative_path,
            size_bytes=len(data),
            sha256=digest,
            duration_seconds=None,  # reliable duration needs a demuxer (Task 07 limitation).
            created_at=now,
            id=audio_file_id,
        )
        transcription = transcriptions.create(
            audio_file_id=audio_file.id,
            raw_text="",  # no transcription requested yet; status carries the state.
            status=TranscriptionStatus.not_requested,
            created_at=now,
        )
        session.commit()
    except Exception as exc:
        session.rollback()
        _unlink_safe(target_path)
        raise AudioStorageError("Failed to persist the audio file metadata.") from exc

    return StoredAudio(
        audio_file_id=audio_file.id,
        message_id=message.id,
        original_mime=mime,
        storage_path=relative_path,
        size_bytes=len(data),
        sha256=digest,
        transcription_status=transcription.status,
        created_at=now,
    )


def resolve_audio_file(session: Session, audio_file_id: uuid.UUID) -> tuple[Path, str]:
    """Resolve a stored audio file to an absolute path and its MIME type.

    The path always originates from the database row written by the server
    (never from user input). As defense in depth the resolved path is checked
    to stay inside ``AppSettings.audio_dir``.
    """
    settings = get_settings()
    row = AudioFileRepository(session).get(audio_file_id)
    if row is None:
        raise AudioNotFoundError("Audio file not found.")
    audio_root = settings.audio_dir.resolve()
    candidate = (settings.audio_dir / row.storage_path).resolve()
    if not candidate.is_relative_to(audio_root):
        # storage_path is server-generated; this branch is purely defensive.
        raise AudioNotFoundError("Audio file not found.")
    return candidate, row.original_mime


__all__ = [
    "AudioError",
    "AudioNotFoundError",
    "AudioStorageError",
    "AudioTooLargeError",
    "EmptyAudioError",
    "InvalidAudioContentError",
    "StoredAudio",
    "SUPPORTED_AUDIO_MIME_TYPES",
    "UnsupportedAudioTypeError",
    "extension_for_mime",
    "normalize_mime",
    "resolve_audio_file",
    "store_audio",
    "validate_upload",
]
