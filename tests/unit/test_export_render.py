"""Unit tests for the original TXT renderer (Task 10).

Byte-for-byte checks of :func:`render_original_export` against the exact
template fixed in Task 10: header block, ``---`` separator, text and audio
entries, raw STT substitution, the ``[не расшифровано]`` marker, multiline
bodies and deterministic ordering of the passed items.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from diary.db.models import MessageKind
from diary.domain.exports.service import ExportItem, export_filename, render_original_export

MOSCOW = ZoneInfo("Europe/Moscow")
SUNDAY = date(2026, 9, 13)
CREATED = datetime(2026, 9, 13, 18, 0, 0, tzinfo=timezone.utc)  # 21:00 Moscow


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)


def _text_item(day: int, hour: int, body: str, minute: int = 0) -> ExportItem:
    return ExportItem(created_at=_at(day, hour, minute), kind=MessageKind.text, body_text=body)


def _audio_item(day: int, hour: int, *, raw_stt: str | None, minute: int = 0) -> ExportItem:
    return ExportItem(
        created_at=_at(day, hour, minute),
        kind=MessageKind.audio,
        body_text="",
        audio_path="audio/2026/09/abcd-1234.wav",
        raw_stt=raw_stt,
    )


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


def test_text_only_export_matches_template_byte_for_byte() -> None:
    text = _render([_text_item(7, 6, "Утром выпил кофе.", minute=5)])
    expected = (
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
        "[2026-09-07 09:05] TEXT\n"
        "Утром выпил кофе.\n"
    )
    assert text == expected


def test_audio_with_transcription_substitutes_raw_stt() -> None:
    text = _render([_audio_item(8, 17, raw_stt="Привет из транскрипции", minute=15)])
    expected = (
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
        "Файл: audio/2026/09/abcd-1234.wav\n"
        "STT raw:\n"
        "Привет из транскрипции\n"
    )
    assert text == expected


def test_audio_without_transcription_is_marked() -> None:
    text = _render([_audio_item(8, 17, raw_stt=None, minute=15)])
    assert "[2026-09-08 20:15] AUDIO\nФайл: audio/2026/09/abcd-1234.wav\nSTT raw:\n" in text
    assert text.rstrip().endswith("[не расшифровано]")
    assert "[не расшифровано]" in text


def test_mixed_entries_are_separated_by_exactly_one_blank_line() -> None:
    items = [
        _text_item(7, 6, "Первая запись", minute=5),
        _audio_item(8, 17, raw_stt=None, minute=15),
        _text_item(9, 10, "Вторая запись", minute=30),
    ]
    text = _render(items)
    assert text == (
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
        "[2026-09-07 09:05] TEXT\n"
        "Первая запись\n"
        "\n"
        "[2026-09-08 20:15] AUDIO\n"
        "Файл: audio/2026/09/abcd-1234.wav\n"
        "STT raw:\n"
        "[не расшифровано]\n"
        "\n"
        "[2026-09-09 13:30] TEXT\n"
        "Вторая запись\n"
    )


def test_multiline_text_body_is_preserved_verbatim() -> None:
    body = "Строка первая\n\nСтрока третья (с пустой между)"
    text = _render([_text_item(7, 6, body, minute=5)])
    assert "Строка первая\n\nСтрока третья (с пустой между)" in text


def test_multiline_raw_stt_is_preserved_verbatim() -> None:
    raw = "первая строка STT\nвторая строка STT"
    text = _render([_audio_item(8, 17, raw_stt=raw, minute=15)])
    assert "STT raw:\nпервая строка STT\nвторая строка STT" in text


def test_sequence_padding_in_filename_and_header() -> None:
    filename = export_filename(SUNDAY, 7, 3)
    assert filename == "2026-09-13_original_0007_r3.txt"
    text = render_original_export(
        sunday=SUNDAY,
        seq_number=7,
        revision=3,
        created_at=CREATED,
        timezone_info=MOSCOW,
        timezone_name="Europe/Moscow",
        items=[],
    )
    assert "Номер выгрузки: 0007\nРевизия: r3" in text
