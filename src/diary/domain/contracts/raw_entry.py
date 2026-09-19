"""Contract ``raw_entry``: one diary message as handed to a phase-2 pipeline (T00).

Mirrors the immutable *source* of the MVP data model (``messages`` /
``audio_files``, ADR-005): a text message or a voice message with its raw STT
text. The contract is deliberately read-only — it is the upstream input of the
``stt_corrected`` layer and of the per-day aggregation, never an output of an
LLM, so it carries no derived field.

Contract (fixing the user-facing definition given for T00)::

    raw_entry: id, ts, source=text|voice, text, audio_path?

Field notes:

- ``id`` — stable message identifier (UUID), the same identity the database
  uses (ADR-001). Downstream layers reference it (``stt_corrected.raw_id``),
  which is what makes provenance per *message* (not per week) possible.
- ``ts`` — timezone-aware moment of the message; an explicit UTC offset is
  required so a day boundary is never ambiguous (the project week/day math
  uses ``APP_TIMEZONE``).
- ``text`` — for ``source="text"`` the text exactly as the user typed it; for
  ``source="voice"`` the **raw** STT text when a transcription exists, and an
  empty string while the voice message has not been transcribed
  (``transcriptions`` is a separate, also immutable record).
- ``audio_path`` — data-root-relative storage path of the audio file for a
  voice message (``audio/YYYY/MM/<uuid>.<ext>``); ``None`` for a text message.
  Server-generated names only: absolute paths and traversal segments are
  rejected here as well, so no contract object can carry an unsafe path
  (AGENTS.md §7).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from diary.domain.contracts._base import ContractModel


def _validate_relative_audio_path(value: str) -> str:
    """Reject absolute paths, traversal segments and Windows separators."""
    if not value or value != value.strip():
        raise ValueError("audio_path must be a non-empty path without surrounding whitespace")
    if value.startswith("/") or value.startswith("\\"):
        raise ValueError("audio_path must be relative to the data root, not absolute")
    if "\\" in value:
        raise ValueError("audio_path must use forward slashes only")
    if ".." in value.split("/"):
        raise ValueError("audio_path must not contain '..' segments")
    return value


class RawEntry(ContractModel):
    """One diary message, exactly as recorded in the source tables."""

    contract_name: ClassVar[str] = "raw_entry"

    #: Stable message identity (same UUID as ``messages.id``).
    id: uuid.UUID

    #: Timezone-aware moment the message was created; an explicit offset is
    #: required (a naive timestamp would make the day boundary undefined).
    ts: datetime

    #: Message kind: typed text or a voice message with raw STT.
    source: Literal["text", "voice"]

    #: Original user text (``text``) or raw STT text (``voice``); ``""`` when a
    #: voice message has not been transcribed yet.
    text: str = ""

    #: Data-root-relative audio path; only for ``source="voice"``.
    audio_path: str | None = Field(default=None)

    @field_validator("ts")
    @classmethod
    def _ts_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must be timezone-aware (include an explicit UTC offset)")
        return value

    @field_validator("audio_path")
    @classmethod
    def _audio_path_is_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_relative_audio_path(value)

    @model_validator(mode="after")
    def _audio_path_matches_source(self) -> RawEntry:
        if self.source == "voice" and self.audio_path is None:
            raise ValueError("a voice entry must carry its audio_path")
        if self.source == "text" and self.audio_path is not None:
            raise ValueError("a text entry must not carry an audio_path")
        return self
