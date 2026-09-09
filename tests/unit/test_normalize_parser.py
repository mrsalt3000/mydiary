"""Unit tests for the original-TXT parser (Task 13).

The parser must deterministically split the byte-exact Task-10 export format
into structured, order-preserving entries. The core property tested here is
the round trip: re-rendering the parsed document with the layer-1 renderer
reproduces the original bytes for canonical inputs
(``render(parse(x)) == x``). Errors raise :class:`OriginalParseError` with
static messages.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from diary.db.models import MessageKind
from diary.domain.exports.service import ExportItem, render_original_export
from diary.domain.pipelines.normalize.parser import (
    OriginalParseError,
    parse_original_text,
)

MOSCOW = ZoneInfo("Europe/Moscow")
SUNDAY = date(2026, 9, 13)
CREATED = datetime(2026, 9, 13, 18, 0, 0, tzinfo=timezone.utc)  # 21:00 Moscow


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)


def _render(items: list[ExportItem]) -> str:
    return render_original_export(
        sunday=SUNDAY,
        seq_number=1,
        revision=1,
        created_at=CREATED,
        timezone_info=MOSCOW,
        timezone_name="Europe/Moscow",
        items=items,
    )


def _text(day: int, hour: int, body: str, minute: int = 0) -> ExportItem:
    return ExportItem(created_at=_at(day, hour, minute), kind=MessageKind.text, body_text=body)


def _audio(day: int, hour: int, raw_stt: str | None, minute: int = 0) -> ExportItem:
    return ExportItem(
        created_at=_at(day, hour, minute),
        kind=MessageKind.audio,
        body_text="",
        audio_path="audio/2026/09/abcd-1234.wav",
        raw_stt=raw_stt,
    )


def _rerender(doc) -> str:
    """Re-render a parsed document back into layer-1 bytes."""
    timezone_info = ZoneInfo(doc.timezone_name)
    items: list[ExportItem] = []
    for entry in doc.entries:
        created_at = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone_info
        )
        if entry.kind is MessageKind.audio:
            items.append(
                ExportItem(
                    created_at=created_at,
                    kind=entry.kind,
                    body_text="",
                    audio_path=entry.audio_path,
                    raw_stt=entry.text,
                )
            )
        else:
            items.append(
                ExportItem(created_at=created_at, kind=entry.kind, body_text=entry.text or "")
            )
    return render_original_export(
        sunday=doc.sunday,
        seq_number=doc.seq_number,
        revision=doc.revision,
        created_at=doc.created_at,
        timezone_info=timezone_info,
        timezone_name=doc.timezone_name,
        items=items,
    )


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "items",
    [
        [_text(7, 6, "Одна запись.", minute=5)],
        [
            _text(7, 6, "Первая.", minute=5),
            _text(7, 9, "Вторая.", minute=10),
            _text(9, 10, "Третья.", minute=30),
        ],
        [_text(7, 6, "Первая строка\n\nТретья строка (после пустой)", minute=5)],
        [_audio(8, 17, raw_stt="Расшифровка одной строкой.", minute=15)],
        [_audio(8, 17, raw_stt=None, minute=15)],
        [
            _text(7, 6, "Первая запись", minute=5),
            _audio(8, 17, raw_stt=None, minute=15),
            _audio(8, 18, raw_stt="строка STT одна\nстрока STT два", minute=0),
            _text(9, 10, "Последняя запись", minute=30),
        ],
        [
            _text(7, 6, "Многострочный\nтекст записи\nс тремя строками", minute=5),
            _audio(8, 18, raw_stt="многострочная\n\nрасшифровка", minute=0),
        ],
    ],
)
def test_parse_rerender_roundtrip_is_byte_exact(items: list[ExportItem]) -> None:
    original = _render(items)
    document = parse_original_text(original)
    assert [entry.kind for entry in document.entries] == [item.kind for item in items]
    assert _rerender(document) == original


def test_parse_records_indexes_timestamps_and_content() -> None:
    original = _render(
        [
            _text(7, 6, "Тело текста", minute=5),
            _audio(8, 17, raw_stt=None, minute=15),
            _audio(8, 18, raw_stt="Привет из транскрипции", minute=0),
        ]
    )
    document = parse_original_text(original)
    assert document.sunday == SUNDAY
    assert document.seq_number == 1
    assert document.revision == 1
    assert document.timezone_name == "Europe/Moscow"
    assert document.created_at == CREATED.astimezone(MOSCOW)

    assert len(document.entries) == 3
    text_entry, untranscribed, transcribed = document.entries
    assert text_entry.index == 0
    assert text_entry.kind is MessageKind.text
    assert text_entry.timestamp == "2026-09-07 09:05"
    assert text_entry.text == "Тело текста"
    assert text_entry.audio_path is None

    assert untranscribed.index == 1
    assert untranscribed.kind is MessageKind.audio
    assert untranscribed.timestamp == "2026-09-08 20:15"
    assert untranscribed.text is None  # marker → nothing to normalize
    assert untranscribed.audio_path == "audio/2026/09/abcd-1234.wav"

    assert transcribed.index == 2
    assert transcribed.kind is MessageKind.audio
    assert transcribed.text == "Привет из транскрипции"


def test_parse_preserves_sequence_and_revision_padding() -> None:
    original = render_original_export(
        sunday=SUNDAY,
        seq_number=7,
        revision=3,
        created_at=CREATED,
        timezone_info=MOSCOW,
        timezone_name="Europe/Moscow",
        items=[_text(7, 6, "Запись", minute=5)],
    )
    document = parse_original_text(original)
    assert document.seq_number == 7
    assert document.revision == 3
    assert _rerender(document) == original


# ---------------------------------------------------------------------------
# malformed input
# ---------------------------------------------------------------------------


def test_parse_rejects_text_without_header_separator() -> None:
    with pytest.raises(OriginalParseError, match="header separator"):
        parse_original_text("# Дневник — оригинал\nЧто-то без разделителя\n")


def test_parse_rejects_missing_header_field() -> None:
    header = (
        "# Дневник — оригинал\n"
        "Неделя: 2026-09-07 — 2026-09-13\n"
        "Дата воскресенья: 2026-09-13\n"
        "Номер выгрузки: 0001\n"
        "Создано: 2026-09-13T21:00:00+03:00\n"
        "Timezone: Europe/Moscow\n"
        "\n"
        "---\n"
        "\n"
        "[2026-09-07 09:05] TEXT\n"
        "Запись.\n"
    )
    with pytest.raises(OriginalParseError, match="header field"):
        parse_original_text(header)


def test_parse_rejects_body_line_outside_an_entry() -> None:
    """Text after the header that cannot belong to any entry must fail.

    (Garbage *inside* a text body is deliberately kept as body content — the
    format preserves arbitrary multiline text verbatim.)
    """
    original = _render([_text(7, 6, "Запись", minute=5)])
    # Place a non-entry line right after the "---" separator, before any stamp.
    corrupt = original.replace("---\n\n[", "---\n\nЭто не запись и не заголовок.\n[", 1)
    with pytest.raises(OriginalParseError, match="valid timestamp line"):
        parse_original_text(corrupt)


def test_parse_rejects_audio_entry_without_metadata() -> None:
    text = (
        "# Дневник — оригинал\n"
        "Неделя: 2026-09-07 — 2026-09-13\n"
        "Дата воскресенья: 2026-09-13\n"
        "Номер выгрузки: 0001\n"
        "Ревизия: r1\n"
        "Создано: 2026-09-13T21:00:00+03:00\n"
        "Timezone: Europe/Moscow\n"
        "\n"
        "---\n"
        "\n"
        "[2026-09-08 20:15] AUDIO\n"
        "только одна строка без метаданных\n"
    )
    with pytest.raises(OriginalParseError, match="AUDIO entry"):
        parse_original_text(text)
