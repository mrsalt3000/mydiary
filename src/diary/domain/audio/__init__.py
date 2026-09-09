"""Audio domain package (Task 07: ingestion and storage).

Voice-message uploads are validated, stored under server-generated UUID names
and linked to a ``messages(kind=audio)`` row and an initial ``not_requested``
transcription. Transcription itself (Task 09) and the durable queue (Task 08)
live elsewhere; see :mod:`diary.domain.audio.service`.
"""

from diary.domain.audio.service import (
    SUPPORTED_AUDIO_MIME_TYPES,
    AudioError,
    AudioNotFoundError,
    AudioStorageError,
    AudioTooLargeError,
    EmptyAudioError,
    InvalidAudioContentError,
    StoredAudio,
    UnsupportedAudioTypeError,
    extension_for_mime,
    normalize_mime,
    resolve_audio_file,
    store_audio,
    validate_upload,
)

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
