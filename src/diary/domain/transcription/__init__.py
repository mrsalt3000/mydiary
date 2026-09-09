"""Transcription domain package (Task 09).

Manual, worker-executed local speech-to-text: provider abstraction
(:mod:`diary.domain.transcription.providers`), the request/execute use cases
(:mod:`diary.domain.transcription.service`) and the optional faster-whisper
adapter. Never runs STT inside an HTTP request and never stores or logs diary
content outside the ``transcriptions`` table (skills/local-stt).
"""

from diary.domain.transcription.providers import (
    PROVIDER_REGISTRY,
    FasterWhisperProvider,
    ProviderUnavailableError,
    TranscriptionExecutionError,
    TranscriptionProvider,
    TranscriptionProviderError,
    create_provider,
)
from diary.domain.transcription.service import (
    PROFILE_REF_TRANSCRIPTION,
    TranscriptionPermanentError,
    TranscriptionServiceError,
    find_active_transcription_job,
    list_transcription_history,
    request_transcription,
    run_transcription,
)

__all__ = [
    "FasterWhisperProvider",
    "PROFILE_REF_TRANSCRIPTION",
    "PROVIDER_REGISTRY",
    "ProviderUnavailableError",
    "TranscriptionExecutionError",
    "TranscriptionPermanentError",
    "TranscriptionProvider",
    "TranscriptionProviderError",
    "TranscriptionServiceError",
    "create_provider",
    "find_active_transcription_job",
    "list_transcription_history",
    "request_transcription",
    "run_transcription",
]
