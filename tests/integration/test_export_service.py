"""Integration tests for the weekly-export domain service (Task 10).

Runs against an isolated, migrated SQLite test database and a temporary data
root (so no file ever touches the production ``data/`` tree) with synthetic
messages only. Covers: byte-exact TXT generation, week-boundary inclusion in a
non-whole-hour timezone, deterministic ordering, empty weeks, no-gap sequence
numbering across skipped weeks, explicit re-export revisions (previous files
stay byte-identical), soft-deleted exclusion, the ``[не расшифровано]``
marker, and consistency between the file and the ``weekly_exports`` row on
success and on failure.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.core.settings import get_settings, reload_settings
from diary.db.models import MessageKind, TranscriptionStatus, WeeklyExport
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    TranscriptionRepository,
    WeeklyExportRepository,
)
from diary.domain.exports import service as exports_service
from diary.domain.exports.service import (
    EmptyWeekError,
    ExportStorageError,
    InvalidSundayDateError,
    WeekAlreadyExportedError,
    WeekNotExportedError,
    export_week,
)
from diary.domain.exports.weeks import week_bounds

MOSCOW_SUNDAY = date(2026, 9, 13)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in (
        "APP_TIMEZONE",
        "APP_MAX_MESSAGE_LENGTH",
        "APP_MAX_AUDIO_MB",
        "APP_ALLOWED_AUDIO_MIME_TYPES",
        "APP_DATA_ROOT",
        "APP_AUDIO_DIR",
        "APP_DB_DIR",
        "APP_TEMP_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


def _add_text(session: Session, body: str, created_at: datetime):
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


def _add_audio(
    session: Session,
    created_at: datetime,
    *,
    storage_path: str,
    transcriptions: list[tuple[str, TranscriptionStatus, datetime]] | None = None,
):
    """Insert an audio message with its audio_file and optional transcription
    rows (Task 07 stores a ``not_requested`` sentinel at upload time).

    Each transcription tuple carries its own ``created_at``: real STT runs get
    their own ``utc_now()`` timestamps, and "the newest succeeded result" is
    decided by ``(created_at, id)`` — rows sharing one instant would fall back
    to the arbitrary UUID tie-break.
    """
    messages = MessageRepository(session)
    audio_files = AudioFileRepository(session)
    transcription_rows = TranscriptionRepository(session)
    message = messages.create(kind=MessageKind.audio, body_text="", created_at=created_at)
    audio = audio_files.create(
        message_id=message.id,
        original_mime="audio/wav",
        storage_path=storage_path,
        size_bytes=321,
        sha256="0" * 64,
        created_at=created_at,
    )
    for text, status, row_created_at in transcriptions or []:
        transcription_rows.create(
            audio_file_id=audio.id, raw_text=text, status=status, created_at=row_created_at
        )
    return message, audio


def _original_files() -> list[Path]:
    """Regular files currently under ``data/exports/01_original`` (sorted)."""
    directory = get_settings().exports_dir / "01_original"
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _read_original(filename: str) -> str:
    path = get_settings().exports_dir / "01_original" / filename
    return path.read_text(encoding="utf-8")


def _export_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(WeeklyExport)) or 0


# ---------------------------------------------------------------------------
# creation, byte-exact content, consistency of file <-> row
# ---------------------------------------------------------------------------


def test_export_creates_byte_exact_file_and_consistent_row(
    session: Session,
) -> None:
    _add_text(
        session, "Синтетика: утренняя запись.", datetime(2026, 9, 7, 6, 5, tzinfo=timezone.utc)
    )
    _add_audio(
        session,
        datetime(2026, 9, 8, 17, 15, tzinfo=timezone.utc),
        storage_path="2026/09/voice-1.wav",
        transcriptions=[
            (
                "",
                TranscriptionStatus.not_requested,
                datetime(2026, 9, 8, 17, 15, tzinfo=timezone.utc),
            ),
            (
                "Синтетика: расшифровка голоса.",
                TranscriptionStatus.succeeded,
                datetime(2026, 9, 8, 17, 16, tzinfo=timezone.utc),
            ),
        ],
    )
    _add_text(
        session, "Синтетика: запись среды.", datetime(2026, 9, 9, 10, 30, tzinfo=timezone.utc)
    )
    session.commit()

    created_at = datetime(2026, 9, 13, 18, 0, 0, tzinfo=timezone.utc)
    record = export_week(session, sunday_date=MOSCOW_SUNDAY, now=created_at)

    assert record.week_end_date == MOSCOW_SUNDAY
    assert record.seq_number == 1
    assert record.revision == 1
    assert record.status.value == "created"
    filename = "2026-09-13_original_0001_r1.txt"
    assert record.path == f"exports/01_original/{filename}"

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
        "Синтетика: утренняя запись.\n"
        "\n"
        "[2026-09-08 20:15] AUDIO\n"
        "Файл: audio/2026/09/voice-1.wav\n"
        "STT raw:\n"
        "Синтетика: расшифровка голоса.\n"
        "\n"
        "[2026-09-09 13:30] TEXT\n"
        "Синтетика: запись среды.\n"
    )
    on_disk = _read_original(filename)
    assert on_disk == expected
    digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert record.sha256 == digest
    stored = WeeklyExportRepository(session).get(record.id)
    assert stored is not None
    assert stored.sha256 == record.sha256


def test_deterministic_order_follows_created_at_then_id(session: Session) -> None:
    _add_text(session, "ранняя-суббота", datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc))
    # Same-second trio on the Monday of the exported week: the deterministic
    # tie-breaker is the message id.
    for index in range(3):
        _add_text(session, f"trio-{index}", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    _add_text(session, "запись-вторника", datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc))
    session.commit()

    export_week(session, sunday_date=MOSCOW_SUNDAY)
    text = _read_original("2026-09-13_original_0001_r1.txt")

    # The Saturday message is outside the week.
    assert "ранняя-суббота" not in text
    # The rendered order equals the repository order (created_at, then id for
    # the same-second trio) — deterministic and id-stable across re-exports.
    start, end = week_bounds(MOSCOW_SUNDAY, get_settings().tzinfo)
    in_week = MessageRepository(session).list_active_between(start=start, end=end)
    assert len(in_week) == 4
    bodies = [message.body_text for message in in_week]
    positions = [text.index(body) for body in bodies]
    assert positions == sorted(positions)
    # Monday is before Tuesday in the file, regardless of the trio's id order.
    assert text.index("[2026-09-07 09:00] TEXT") < text.index("[2026-09-08 11:00] TEXT")


# ---------------------------------------------------------------------------
# timezone boundaries
# ---------------------------------------------------------------------------


def test_week_boundaries_respect_timezone(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Kathmandu")
    reload_settings()
    tz = ZoneInfo("Asia/Kathmandu")  # UTC+05:45
    sunday = date(2026, 9, 13)
    start, end = week_bounds(sunday, tz)
    _add_text(session, "ровно-в-начале", start)
    _add_text(session, "ровно-в-конце", end)
    _add_text(session, "за-миг-до-недели", start - timedelta(microseconds=1))
    _add_text(session, "через-миг-после-недели", end + timedelta(microseconds=1))
    session.commit()

    export_week(session, sunday_date=sunday, now=start)
    text = _read_original("2026-09-13_original_0001_r1.txt")
    # Local wall-clock stamps in the configured timezone.
    assert "[2026-09-07 00:00] TEXT\nровно-в-начале" in text
    assert "[2026-09-13 23:59] TEXT\nровно-в-конце" in text
    assert "за-миг-до-недели" not in text
    assert "через-миг-после-недели" not in text
    assert "Timezone: Asia/Kathmandu" in text


# ---------------------------------------------------------------------------
# empty weeks, numbering without gaps
# ---------------------------------------------------------------------------


def test_empty_week_creates_nothing_and_consumes_no_sequence(session: Session) -> None:
    # A message lives in a different week, so this week is empty.
    _add_text(session, "другая неделя", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    empty_sunday = date(2026, 9, 20)

    with pytest.raises(EmptyWeekError):
        export_week(session, sunday_date=empty_sunday)

    assert _export_count(session) == 0
    assert _original_files() == []
    # The neighbouring non-empty week still exports as sequence 1.
    record = export_week(session, sunday_date=MOSCOW_SUNDAY)
    assert record.seq_number == 1


def test_skipped_weeks_do_not_create_numbering_gaps(session: Session) -> None:
    week_a_sunday = date(2026, 8, 30)
    _add_text(session, "неделя A", datetime(2026, 8, 24, 6, 0, tzinfo=timezone.utc))
    session.commit()
    export_a = export_week(session, sunday_date=week_a_sunday)
    assert (export_a.seq_number, export_a.revision) == (1, 1)

    # Week B (2026-09-06) is skipped entirely.
    _add_text(session, "неделя C", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    export_c = export_week(session, sunday_date=MOSCOW_SUNDAY)
    assert export_c.seq_number == 2  # max+1: no gap left by the inactive week.
    assert export_c.revision == 1
    assert export_c.path == "exports/01_original/2026-09-13_original_0002_r1.txt"


# ---------------------------------------------------------------------------
# revisions and immutability of earlier files
# ---------------------------------------------------------------------------


def test_duplicate_without_reexport_is_rejected(session: Session) -> None:
    _add_text(session, "единственная запись", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    first = export_week(session, sunday_date=MOSCOW_SUNDAY)
    before = _read_original(first.path.rsplit("/", 1)[-1])

    with pytest.raises(WeekAlreadyExportedError):
        export_week(session, sunday_date=MOSCOW_SUNDAY)

    assert _export_count(session) == 1
    assert _read_original(first.path.rsplit("/", 1)[-1]) == before  # untouched


def test_reexport_creates_new_revision_and_keeps_prior_file(session: Session) -> None:
    _add_text(session, "первая версия текста", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    first = export_week(session, sunday_date=MOSCOW_SUNDAY)
    r1_name = "2026-09-13_original_0001_r1.txt"
    r1_bytes = _read_original(r1_name)

    # A new message arrives (e.g. STT finished) before the re-export.
    _add_text(
        session, "вторая запись для реэкспорта", datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)
    )
    session.commit()

    second = export_week(session, sunday_date=MOSCOW_SUNDAY, re_export=True)
    assert second.seq_number == first.seq_number
    assert second.revision == 2
    assert second.path == "exports/01_original/2026-09-13_original_0001_r2.txt"

    r2_text = _read_original("2026-09-13_original_0001_r2.txt")
    assert "первая версия текста" in r2_text
    assert "вторая запись для реэкспорта" in r2_text
    # The prior original is immutable: byte-for-byte unchanged.
    assert _read_original(r1_name) == r1_bytes
    assert r2_text != r1_bytes

    rows = WeeklyExportRepository(session).list_by_week(MOSCOW_SUNDAY)
    assert [row.revision for row in rows] == [1, 2]
    assert len({row.sha256 for row in rows}) == 2

    # A second re-export advances to r3 of the same lineage.
    third = export_week(session, sunday_date=MOSCOW_SUNDAY, re_export=True)
    assert (third.seq_number, third.revision) == (1, 3)


def test_reexport_of_never_exported_week_is_rejected(session: Session) -> None:
    _add_text(session, "запись", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    with pytest.raises(WeekNotExportedError):
        export_week(session, sunday_date=MOSCOW_SUNDAY, re_export=True)
    assert _export_count(session) == 0
    assert _original_files() == []


def test_invalid_sunday_date_is_rejected(session: Session) -> None:
    _add_text(session, "запись", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    with pytest.raises(InvalidSundayDateError):
        export_week(session, sunday_date=date(2026, 9, 7))  # a Monday
    assert _export_count(session) == 0


# ---------------------------------------------------------------------------
# soft-deleted messages and missing transcriptions
# ---------------------------------------------------------------------------


def test_soft_deleted_messages_are_excluded(session: Session) -> None:
    from diary.db.base import utc_now

    keep = _add_text(
        session, "остаётся в выгрузке", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc)
    )
    remove = _add_text(session, "удалена мягко", datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc))
    remove.deleted_at = utc_now()
    session.commit()
    assert keep.deleted_at is None

    export_week(session, sunday_date=MOSCOW_SUNDAY)
    text = _read_original("2026-09-13_original_0001_r1.txt")
    assert "остаётся в выгрузке" in text
    assert "удалена мягко" not in text


def test_audio_without_successful_transcription_is_marked(session: Session) -> None:
    _add_audio(
        session,
        datetime(2026, 9, 8, 17, 15, tzinfo=timezone.utc),
        storage_path="2026/09/no-stt.wav",
        transcriptions=[
            (
                "",
                TranscriptionStatus.not_requested,
                datetime(2026, 9, 8, 17, 15, tzinfo=timezone.utc),
            ),
            ("", TranscriptionStatus.failed, datetime(2026, 9, 8, 17, 16, tzinfo=timezone.utc)),  # noqa: E501
        ],
    )
    session.commit()

    export_week(session, sunday_date=MOSCOW_SUNDAY)
    text = _read_original("2026-09-13_original_0001_r1.txt")
    assert "Файл: audio/2026/09/no-stt.wav" in text
    assert "[не расшифровано]" in text


def test_audio_uses_newest_succeeded_transcription(session: Session) -> None:
    base = datetime(2026, 9, 8, 17, 15, tzinfo=timezone.utc)
    _add_audio(
        session,
        base,
        storage_path="2026/09/retranscribed.wav",
        transcriptions=[
            ("", TranscriptionStatus.not_requested, base),
            ("старая расшифровка", TranscriptionStatus.succeeded, base + timedelta(minutes=1)),
            ("", TranscriptionStatus.failed, base + timedelta(minutes=2)),
            ("новая расшифровка", TranscriptionStatus.succeeded, base + timedelta(minutes=3)),
        ],
    )
    session.commit()

    export_week(session, sunday_date=MOSCOW_SUNDAY)
    text = _read_original("2026-09-13_original_0001_r1.txt")
    assert "новая расшифровка" in text
    assert "старая расшифровка" not in text


# ---------------------------------------------------------------------------
# failure consistency: no row, no final file, no leftover
# ---------------------------------------------------------------------------


def test_file_write_failure_leaves_no_row_and_no_partial_file(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(data: bytes, target_path: Path, temp_dir: Path) -> None:
        raise OSError("simulated unwritable export directory")

    monkeypatch.setattr(exports_service, "_write_export_atomically", _boom)
    _add_text(session, "запись", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()

    with pytest.raises(ExportStorageError):
        export_week(session, sunday_date=MOSCOW_SUNDAY)

    assert _export_count(session) == 0
    assert _original_files() == []
    # No scratch leftovers in the temp directory either.
    temp_dir = get_settings().temp_dir
    leftovers = (
        [path for path in temp_dir.rglob("*") if path.is_file()] if temp_dir.exists() else []
    )
    assert leftovers == []


def test_db_failure_removes_the_written_file(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**kwargs):
        raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(WeeklyExportRepository, "create", _boom)
    _add_text(session, "запись", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()

    with pytest.raises(ExportStorageError):
        export_week(session, sunday_date=MOSCOW_SUNDAY)

    assert _export_count(session) == 0
    # The just-written final file was removed again: no orphaned artifact.
    assert _original_files() == []


def test_revision_uniqueness_is_enforced_in_database(session: Session) -> None:
    _add_text(session, "запись", datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))
    session.commit()
    first = export_week(session, sunday_date=MOSCOW_SUNDAY)

    # Directly inserting the same (week, revision) must fail at the DB level.
    from sqlalchemy.exc import IntegrityError

    repository = WeeklyExportRepository(session)
    with pytest.raises(IntegrityError):
        repository.create(
            week_end_date=first.week_end_date,
            seq_number=99,
            revision=first.revision,
            path="exports/01_original/duplicate.txt",
            sha256="0" * 64,
        )
        session.commit()
    session.rollback()
