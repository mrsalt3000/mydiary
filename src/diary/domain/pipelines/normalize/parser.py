"""Deterministic parser for original weekly exports (layer 1; Task 10/13).

The normalization pipeline (Task 13) needs the byte-exact layer-1 TXT produced
by ``diary.domain.exports.service.render_original_export`` broken down into
structured, order-preserving entries: a text message's original body, an audio
message's raw STT text, or an explicit "not transcribed" state — each with the
rendered timestamp and the source kind.

Design notes
============

* Purely structural: no settings, no database access and no diary-text
  interpretation. Feeding the parsed entries back through
  ``render_original_export`` (via ``ExportItem``) reproduces the original
  bytes for canonical inputs — the ``render(parse(x)) == x`` round-trip
  property the parser tests rely on.
* Every entry carries its 0-based position in the export file (``index``),
  the rendered ``timestamp`` string and the source ``kind``; the pipeline
  echoes these through the LLM and checks them back, so a result can never
  silently reorder, re-time or change the source type of an entry.
* ``text`` is the entry's normalizable content: the message body for text
  entries and the raw STT text for transcribed audio entries. An audio entry
  that the export marked ``[не расшифровано]`` parses to ``text=None`` — there
  is nothing to normalize, and the layer-2 renderer keeps the marker.
* Input that does not match the fixed Task-10 structure raises
  :class:`OriginalParseError` with a static, non-sensitive message (no diary
  content, no file paths; AGENTS.md §8). The pipeline maps it to a permanent
  run failure.

Privacy: exception messages never echo diary text or stored paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from diary.db.models import MessageKind

#: Rendered entry header: ``[YYYY-MM-DD HH:MM] TEXT|AUDIO`` (Task 10 format).
_ENTRY_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (TEXT|AUDIO)$")
#: Line that ends the original-export header block.
_HEADER_SEPARATOR = "---"
#: Rendered marker for an audio entry without a successful transcription.
_UNTRANSCRIBED_MARKER = "[не расшифровано]"
#: Prefix of the audio file-reference line inside an AUDIO entry.
_AUDIO_FILE_PREFIX = "Файл: "
#: Prefix of the raw-STT line inside an AUDIO entry.
_STT_RAW_PREFIX = "STT raw:"


class OriginalParseError(Exception):
    """The text is not a well-formed Task-10 original export.

    Messages are static and never contain diary content or paths.
    """


@dataclass(frozen=True)
class ParsedEntry:
    """One parsed entry of an original weekly export.

    ``index`` is the entry's 0-based position in the export file (deterministic
    chronological order, as rendered by Task 10). ``timestamp`` is the rendered
    local-time stamp string ``YYYY-MM-DD HH:MM``. ``text`` is ``None`` exactly
    when the entry carries no normalizable content (an audio entry the export
    marked ``[не расшифровано]``); otherwise it holds the message body (text
    entries) or the raw STT text (audio entries). ``audio_path`` mirrors the
    ``Файл:`` line for audio entries and is ``None`` for text entries.
    """

    index: int
    kind: MessageKind
    timestamp: str
    text: str | None
    audio_path: str | None = None


@dataclass(frozen=True)
class OriginalDocument:
    """A fully parsed original export: header lineage + ordered entries.

    The header fields mirror what ``render_original_export`` emits and are what
    a round-trip re-render needs (the layer-2 header is built from the export
    database row instead — this document is the parser's own representation).
    """

    sunday: date
    seq_number: int
    revision: int
    created_at: datetime
    timezone_name: str
    entries: list[ParsedEntry]


def parse_original_text(text: str) -> OriginalDocument:
    """Parse one original weekly-export TXT into an :class:`OriginalDocument`.

    Raises :class:`OriginalParseError` (static messages) when the text does not
    match the Task-10 structure: a missing header separator, a missing or
    invalid header field, an entry whose header line is malformed, or an
    AUDIO entry without its ``Файл:``/``STT raw:`` metadata lines.
    """
    lines = text.split("\n")
    try:
        header_end = lines.index(_HEADER_SEPARATOR)
    except ValueError as exc:
        raise OriginalParseError("The export header separator is missing.") from exc
    header = _parse_header(lines[:header_end])
    entries: list[ParsedEntry] = []
    index = 0
    cursor = header_end + 1
    while cursor < len(lines):
        line = lines[cursor]
        if line == "":
            # Blank line between entries (or right after the separator).
            cursor += 1
            continue
        match = _ENTRY_RE.match(line)
        if match is None:
            raise OriginalParseError("An entry does not start with a valid timestamp line.")
        timestamp, kind_token = match.group(1), match.group(2)
        kind = MessageKind.text if kind_token == "TEXT" else MessageKind.audio
        cursor += 1
        content: list[str] = []
        while cursor < len(lines) and _ENTRY_RE.match(lines[cursor]) is None:
            content.append(lines[cursor])
            cursor += 1
        if kind is MessageKind.audio:
            entries.append(_parse_audio_entry(index, timestamp, content))
        else:
            body = "\n".join(content).rstrip("\n")
            entries.append(ParsedEntry(index=index, kind=kind, timestamp=timestamp, text=body))
        index += 1
    return OriginalDocument(
        sunday=header.sunday,
        seq_number=header.seq_number,
        revision=header.revision,
        created_at=header.created_at,
        timezone_name=header.timezone_name,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


class _Header:
    """Validated subset of the layer-1 header needed by the parser round-trip."""

    def __init__(
        self,
        *,
        sunday: date,
        seq_number: int,
        revision: int,
        created_at: datetime,
        timezone_name: str,
    ) -> None:
        self.sunday = sunday
        self.seq_number = seq_number
        self.revision = revision
        self.created_at = created_at
        self.timezone_name = timezone_name


def _parse_header(lines: list[str]) -> _Header:
    """Parse the header block (everything before the ``---`` separator)."""
    values: dict[str, str] = {}
    for line in lines:
        if line.startswith("# ") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()

    def required(name: str) -> str:
        try:
            return values[name]
        except KeyError as exc:
            raise OriginalParseError("A required export header field is missing.") from exc

    sunday_raw = required("Дата воскресенья")
    seq_raw = required("Номер выгрузки")
    revision_raw = required("Ревизия")
    created_raw = required("Создано")
    timezone_name = required("Timezone")
    if not timezone_name:
        raise OriginalParseError("The export header has an empty timezone field.")

    try:
        sunday = date.fromisoformat(sunday_raw)
    except ValueError as exc:
        raise OriginalParseError("The export header has an invalid Sunday date.") from exc
    try:
        seq_number = int(seq_raw)
    except ValueError as exc:
        raise OriginalParseError("The export header has an invalid sequence number.") from exc
    if not revision_raw.startswith("r"):
        raise OriginalParseError("The export header has an invalid revision.")
    try:
        revision = int(revision_raw[1:])
    except ValueError as exc:
        raise OriginalParseError("The export header has an invalid revision.") from exc
    return _Header(
        sunday=sunday,
        seq_number=seq_number,
        revision=revision,
        created_at=_parse_created(created_raw),
        timezone_name=timezone_name,
    )


def _parse_created(value: str) -> datetime:
    """Parse the ``Создано`` ISO-8601 line into a timezone-aware datetime."""
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise OriginalParseError("The export header has an invalid creation timestamp.") from exc
    if parsed.tzinfo is None:
        raise OriginalParseError("The export header creation timestamp is not timezone-aware.")
    return parsed


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def _parse_audio_entry(index: int, timestamp: str, content: list[str]) -> ParsedEntry:
    """Parse one AUDIO entry body into a :class:`ParsedEntry`.

    The body must start with ``Файл: <path>`` and ``STT raw:`` lines; the
    remaining lines are the raw STT text. A body whose raw text is exactly the
    ``[не расшифровано]`` marker parses to ``text=None``.
    """
    if len(content) < 2:
        raise OriginalParseError("An AUDIO entry is missing its metadata lines.")
    if not content[0].startswith(_AUDIO_FILE_PREFIX):
        raise OriginalParseError("An AUDIO entry has no valid file-reference line.")
    audio_path = content[0][len(_AUDIO_FILE_PREFIX) :]
    if content[1] != _STT_RAW_PREFIX:
        raise OriginalParseError("An AUDIO entry has no valid STT raw line.")
    raw_lines = content[2:]
    raw = "\n".join(raw_lines).rstrip("\n")
    text = None if raw == _UNTRANSCRIBED_MARKER else raw
    return ParsedEntry(
        index=index,
        kind=MessageKind.audio,
        timestamp=timestamp,
        text=text,
        audio_path=audio_path,
    )


__all__ = [
    "OriginalDocument",
    "OriginalParseError",
    "ParsedEntry",
    "parse_original_text",
]
