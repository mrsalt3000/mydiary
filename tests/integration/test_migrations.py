"""Integration tests for Alembic migrations (Task 04).

A fresh database is created solely by running ``upgrade head`` on an empty
SQLite file; the tests verify that all expected tables, key indexes and
constraints exist and that a repeated upgrade is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import Engine, inspect

EXPECTED_TABLES = {
    "messages",
    "audio_files",
    "transcriptions",
    "weekly_exports",
    "processing_runs",
    "artifacts",
    "jobs",
    "alembic_version",
}

#: (table, index name, columns) — indexes required by the Task 04 criteria.
KEY_INDEXES = [
    ("messages", "ix_messages_created_at", ("created_at",)),
    ("jobs", "ix_jobs_status", ("status",)),
    ("audio_files", "ix_audio_files_message_id", ("message_id",)),
    ("transcriptions", "ix_transcriptions_audio_file_id", ("audio_file_id",)),
    ("weekly_exports", "ix_weekly_exports_seq_number", ("seq_number",)),
    ("artifacts", "ix_artifacts_export_id_layer", ("export_id", "layer")),
    ("artifacts", "ix_artifacts_processing_run_id", ("processing_run_id",)),
]


def test_upgrade_creates_all_expected_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    actual = set(inspector.get_table_names())
    assert EXPECTED_TABLES <= actual, f"missing tables: {EXPECTED_TABLES - actual}"


def test_upgrade_creates_key_indexes(engine: Engine) -> None:
    inspector = inspect(engine)
    indexed = {
        table: {i["name"]: tuple(i["column_names"]) for i in inspector.get_indexes(table)}
        for table in inspector.get_table_names()
    }
    for table, index_name, columns in KEY_INDEXES:
        table_indexes = indexed.get(table, {})
        assert index_name in table_indexes, f"{index_name} missing on {table}"
        assert table_indexes[index_name] == columns, (
            f"{index_name} on {table} has unexpected columns"
        )


def test_export_week_revision_is_unique(engine: Engine) -> None:
    inspector = inspect(engine)
    unique_names = {
        constraint["name"] for constraint in inspector.get_unique_constraints("weekly_exports")
    }
    assert "uq_weekly_exports_week_end_date_revision" in unique_names


def test_foreign_keys_are_declared(engine: Engine) -> None:
    inspector = inspect(engine)
    audio_fks = inspector.get_foreign_keys("audio_files")
    assert any(fk["referred_table"] == "messages" for fk in audio_fks)
    transcription_fks = inspector.get_foreign_keys("transcriptions")
    assert any(fk["referred_table"] == "audio_files" for fk in transcription_fks)


def test_status_columns_have_check_constraints(engine: Engine) -> None:
    inspector = inspect(engine)
    # Enum columns on SQLite are VARCHAR columns with a CHECK constraint.
    assert inspector.get_check_constraints("jobs"), "jobs.status has no CHECK constraint"
    assert inspector.get_check_constraints("processing_runs"), (
        "processing_runs.status has no CHECK constraint"
    )


def test_upgrade_is_repeatable_on_a_fresh_database(
    run_upgrade: Callable[[str], None], db_url: str
) -> None:
    # Second upgrade over the migrated database must be a clean no-op.
    run_upgrade(db_url)
    run_upgrade(db_url)
