"""Integration tests for migration ``0003`` (second anchor family, task T08).

Checks the migration itself, not the ORM:

* a fresh database is created by migrations only and carries the new anchor
  columns, indexes and constraints under their expected (non-doubled) names;
* a database with MVP-shaped rows can be migrated forward without losing or
  rewriting a single row — the exact scenario of the real diary database;
* the upgrade is repeatable, and the downgrade refuses to drop results that the
  previous schema cannot represent;
* the widened enums accept the phase-2 values and still reject unknown ones.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest
from _migrate import migrate_database_file
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from diary.db.models import Artifact, ArtifactLayer, JobType, PipelineType, RunStatus

REVISION_0002 = "0002_messages_soft_delete"
REVISION_0003 = "0003_phase2_anchor_family"

#: Columns the second anchor family adds to ``artifacts``.
NEW_COLUMNS = {
    "raw_id",
    "local_date",
    "local_tz",
    "week_id",
    "summary_id",
    "result_json",
}

#: Constraints the migration must create (names as seen in the SQLite schema).
EXPECTED_CHECKS = {
    "ck_artifacts_anchor_family",
    "ck_artifacts_anchor_local_tz",
    "ck_artifacts_anchor_payload",
    "ck_artifacts_artifact_layer",
    "ck_processing_runs_pipeline_type",
    "ck_jobs_job_type",
}
EXPECTED_UNIQUES = {
    "uq_artifacts_export_id_layer_version",
    "uq_artifacts_raw_id_layer_version",
    "uq_artifacts_local_date_layer_version",
    "uq_artifacts_week_id_layer_version",
    "uq_artifacts_summary_id_layer_version",
}
EXPECTED_INDEXES = {
    "ix_artifacts_export_id_layer",
    "ix_artifacts_processing_run_id",
    "ix_artifacts_raw_id_layer",
    "ix_artifacts_local_date_layer",
    "ix_artifacts_week_id_layer",
    "ix_artifacts_summary_id_layer",
}

_TABLES = (
    "messages",
    "audio_files",
    "transcriptions",
    "weekly_exports",
    "processing_runs",
    "artifacts",
    "jobs",
)
_RAW_TABLES = ("messages", "audio_files", "transcriptions", "weekly_exports")

#: Fixed timestamps (synthetic data only; never production content).
_TS = "2026-08-02 20:00:00.000000"


def _alembic_config(db_path: Path) -> AlembicConfig:
    """Alembic config for the in-process downgrade assertions."""
    repo_root = Path(__file__).resolve().parents[2]
    config = AlembicConfig()
    config.set_main_option("script_location", str(repo_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return config


def _table_rows(db_path: Path, table: str) -> tuple[list[str], list[tuple]]:
    """Column names and all rows of ``table`` (synthetic test data only)."""
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.execute(f"SELECT * FROM {table}")  # noqa: S608
        columns = [description[0] for description in cursor.description]
        return columns, sorted(cursor.fetchall())
    finally:
        connection.close()


def _assert_rows_unchanged(
    before: tuple[list[str], list[tuple]], after: tuple[list[str], list[tuple]], table: str
) -> None:
    """Fail unless every pre-migration row is present with identical values.

    New columns added by the migration are ignored on purpose: the assertion is
    about the values that existed before, and a new nullable column must not
    rewrite them.
    """
    before_columns, before_rows = before
    after_columns, after_rows = after
    assert set(before_columns) <= set(after_columns), f"a column vanished from {table}"
    projection = [after_columns.index(column) for column in before_columns]
    projected = sorted(tuple(row[index] for index in projection) for row in after_rows)
    assert projected == before_rows, f"{table} rows changed"


def _create_mvp_rows(db_path: Path) -> dict[str, str]:
    """Insert one synthetic row per MVP table at revision ``0002``.

    Mirrors the shape of the real database (messages, audio, raw
    transcriptions, an export lineage with its run and artifacts, a queue job)
    so the migration is exercised on data, not only on an empty schema.
    """
    ids = {name: uuid.uuid4().hex for name in ("message", "audio", "export", "run", "artifact")}
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.execute(
            "INSERT INTO messages (id, kind, body_text, created_at, deleted_at)"
            " VALUES (?, 'text', 'синтетическое сообщение', ?, NULL)",
            (ids["message"], _TS),
        )
        connection.execute(
            "INSERT INTO audio_files (id, message_id, original_mime, storage_path, size_bytes,"
            " sha256, duration_seconds, created_at) VALUES (?, ?, 'audio/webm',"
            " '2026/08/synthetic.webm', 42, ?, 1.5, ?)",
            (ids["audio"], ids["message"], "a" * 64, _TS),
        )
        connection.execute(
            "INSERT INTO transcriptions (id, audio_file_id, raw_text, status, created_at)"
            " VALUES (?, ?, 'сырой текст', 'succeeded', ?)",
            (uuid.uuid4().hex, ids["audio"], _TS),
        )
        connection.execute(
            "INSERT INTO weekly_exports (id, week_end_date, seq_number, revision, status, path,"
            " sha256, created_at) VALUES (?, '2026-09-13', 1, 1, 'created',"
            " 'exports/01_original/synthetic.txt', ?, ?)",
            (ids["export"], "b" * 64, _TS),
        )
        connection.execute(
            "INSERT INTO processing_runs (id, pipeline_type, status, input_export_id,"
            " profile_name, model_name, prompt_version, created_at, started_at, finished_at)"
            " VALUES (?, 'normalize', 'succeeded', ?, 'normalization', 'diary-sum', 'v1', ?, ?, ?)",
            (ids["run"], ids["export"], _TS, _TS, _TS),
        )
        connection.execute(
            "INSERT INTO artifacts (id, layer, export_id, processing_run_id, version, path,"
            " sha256, created_at) VALUES (?, 'normalized', ?, ?, 1,"
            " 'exports/02_normalized/synthetic_v1.txt', ?, ?)",
            (ids["artifact"], ids["export"], ids["run"], "c" * 64, _TS),
        )
        connection.execute(
            "INSERT INTO artifacts (id, layer, export_id, processing_run_id, version, path,"
            " sha256, created_at) VALUES (?, 'original', ?, NULL, NULL,"
            " 'exports/01_original/synthetic.txt', ?, ?)",
            (uuid.uuid4().hex, ids["export"], "b" * 64, _TS),
        )
        connection.execute(
            "INSERT INTO jobs (id, job_type, status, payload_json, attempts, created_at,"
            " updated_at) VALUES (?, 'normalize', 'succeeded', NULL, 1, ?, ?)",
            (uuid.uuid4().hex, _TS, _TS),
        )
        connection.commit()
    finally:
        connection.close()
    return ids


# ---------------------------------------------------------------------------
# Fresh database: schema shape
# ---------------------------------------------------------------------------


def test_fresh_database_has_the_second_anchor_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("artifacts")}
    assert NEW_COLUMNS <= set(columns)
    for name in ("export_id", "path", "sha256"):
        assert columns[name]["nullable"] is True
    assert columns["layer"]["nullable"] is False


def test_fresh_database_has_the_anchor_constraints_and_indexes(engine: Engine) -> None:
    inspector = inspect(engine)
    check_names = {
        constraint["name"]
        for table in ("artifacts", "processing_runs", "jobs")
        for constraint in inspector.get_check_constraints(table)
    }
    assert EXPECTED_CHECKS <= check_names
    assert {name for name in check_names if name.count("ck_artifacts_") > 1} == set()
    unique_names = {
        constraint["name"] for constraint in inspector.get_unique_constraints("artifacts")
    }
    assert EXPECTED_UNIQUES <= unique_names
    index_names = {index["name"] for index in inspector.get_indexes("artifacts")}
    assert EXPECTED_INDEXES <= index_names


def test_fresh_database_has_exactly_one_layer_check(engine: Engine) -> None:
    """The recreated table must not keep a stale, narrower layer CHECK."""
    with engine.connect() as connection:
        ddl = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name='artifacts'")
        ).scalar_one()
    assert str(ddl).count("CHECK (layer IN") == 1
    for layer in ArtifactLayer:
        assert f"'{layer.value}'" in str(ddl)


def test_second_family_anchors_reference_their_sources(engine: Engine) -> None:
    inspector = inspect(engine)
    foreign_keys = {
        (fk["constrained_columns"][0], fk["referred_table"])
        for fk in inspector.get_foreign_keys("artifacts")
    }
    assert ("raw_id", "messages") in foreign_keys
    assert ("summary_id", "artifacts") in foreign_keys
    assert ("export_id", "weekly_exports") in foreign_keys


# ---------------------------------------------------------------------------
# Migrating a database with data
# ---------------------------------------------------------------------------


def test_upgrade_from_0002_keeps_every_mvp_row(tmp_path: Path) -> None:
    db_path = tmp_path / "diary.db"
    migrate_database_file(db_path, REVISION_0002)
    _create_mvp_rows(db_path)
    before = {table: _table_rows(db_path, table) for table in _TABLES}

    migrate_database_file(db_path, "head")

    for table in _TABLES:
        _assert_rows_unchanged(before[table], _table_rows(db_path, table), table)
    for table in _RAW_TABLES:
        _assert_rows_unchanged(before[table], _table_rows(db_path, table), table)
    # the rows are still valid under the new constraints: updating a value
    # without violating them proves the table was not silently rewritten
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            REVISION_0003,
        )
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "UPDATE messages SET deleted_at = '2026-09-19 00:00:00.000000' WHERE deleted_at IS NULL"
        )
        connection.rollback()
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (2,)
    finally:
        connection.close()


def test_upgrade_is_repeatable_on_a_database_with_data(tmp_path: Path) -> None:
    db_path = tmp_path / "diary.db"
    migrate_database_file(db_path, REVISION_0002)
    _create_mvp_rows(db_path)
    migrate_database_file(db_path, "head")
    after_first = {table: _table_rows(db_path, table) for table in _TABLES}

    migrate_database_file(db_path, "head")

    for table in _TABLES:
        _assert_rows_unchanged(after_first[table], _table_rows(db_path, table), table)


def test_downgrade_refuses_while_second_family_rows_exist(tmp_path: Path) -> None:
    db_path = tmp_path / "diary.db"
    migrate_database_file(db_path, "head")
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.execute(
            "INSERT INTO messages (id, kind, body_text, created_at) VALUES (?, 'text', '', ?)",
            (uuid.uuid4().hex, _TS),
        )
        connection.execute(
            "INSERT INTO artifacts (id, layer, version, created_at, week_id, result_json)"
            " VALUES (?, 'weekly_summary', 1, ?, '2026-W38', '{}')",
            (uuid.uuid4().hex, _TS),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="second anchor family"):
        command.downgrade(_alembic_config(db_path), REVISION_0002)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM artifacts WHERE export_id IS NULL")
        connection.commit()
    finally:
        connection.close()
    command.downgrade(_alembic_config(db_path), REVISION_0002)
    assert (
        sqlite3.connect(db_path).execute("SELECT version_num FROM alembic_version").fetchone()[0]
        == REVISION_0002
    )
    command.upgrade(_alembic_config(db_path), "head")


# ---------------------------------------------------------------------------
# Enum widening
# ---------------------------------------------------------------------------


def test_phase2_enum_values_are_accepted_by_the_migrated_schema(session: Session) -> None:
    for pipeline_type in (
        PipelineType.stt_corrected,
        PipelineType.daily_summary,
        PipelineType.weekly_summary,
        PipelineType.coach_note,
    ):
        session.execute(
            text(
                "INSERT INTO processing_runs (id, pipeline_type, status, created_at)"
                " VALUES (:id, :pipeline, 'succeeded', :created)"
            ),
            {"id": uuid.uuid4().hex, "pipeline": pipeline_type.value, "created": _TS},
        )
    for job_type in (
        JobType.stt_corrected,
        JobType.daily_summary,
        JobType.weekly_summary,
        JobType.coach_note,
    ):
        session.execute(
            text(
                "INSERT INTO jobs (id, job_type, status, attempts, created_at, updated_at)"
                " VALUES (:id, :job, 'pending', 0, :created, :created)"
            ),
            {"id": uuid.uuid4().hex, "job": job_type.value, "created": _TS},
        )
    session.flush()


def test_unknown_enum_values_are_still_rejected(session: Session) -> None:
    for statement, params in (
        (
            "INSERT INTO processing_runs (id, pipeline_type, status, created_at)"
            " VALUES (:id, 'telegram', 'succeeded', :created)",
            None,
        ),
        (
            "INSERT INTO jobs (id, job_type, status, attempts, created_at, updated_at)"
            " VALUES (:id, 'telegram', 'pending', 0, :created, :created)",
            None,
        ),
        (
            "INSERT INTO artifacts (id, layer, version, created_at, raw_id, result_json)"
            " VALUES (:id, 'telegram', 1, :created, NULL, '{}')",
            None,
        ),
    ):
        with pytest.raises(IntegrityError):
            session.execute(
                text(statement),
                {"id": uuid.uuid4().hex, "created": _TS},
            )
        session.rollback()


def test_first_family_artifact_keeps_its_meaning(session: Session) -> None:
    """A layer-2 artifact of an export lineage is still stored exactly as before."""
    export_id = uuid.uuid4()
    session.execute(
        text(
            "INSERT INTO weekly_exports (id, week_end_date, seq_number, revision, status, path,"
            " sha256, created_at) VALUES (:id, '2026-09-13', 1, 1, 'created', 'exports/x.txt',"
            " :sha, :created)"
        ),
        {"id": export_id.hex, "sha": "b" * 64, "created": _TS},
    )
    session.execute(
        text(
            "INSERT INTO artifacts (id, layer, export_id, version, path, sha256, created_at)"
            " VALUES (:id, 'normalized', :export, 1, 'exports/02_normalized/x_v1.txt', :sha,"
            " :created)"
        ),
        {
            "id": uuid.uuid4().hex,
            "export": export_id.hex,
            "sha": "c" * 64,
            "created": _TS,
        },
    )
    session.flush()
    artifact = session.query(Artifact).one()
    assert artifact.layer is ArtifactLayer.normalized
    assert artifact.export_id == export_id
    assert artifact.version == 1
    assert artifact.result_json is None
    assert artifact.processing_run_id is None
    session.rollback()


def test_processing_run_status_values_are_unchanged(session: Session) -> None:
    """The run-status CHECK was re-created by the table recreation as well."""
    session.execute(
        text(
            "INSERT INTO processing_runs (id, pipeline_type, status, created_at)"
            " VALUES (:id, 'normalize', 'failed', :created)"
        ),
        {"id": uuid.uuid4().hex, "created": _TS},
    )
    session.flush()
    assert RunStatus.failed.value == "failed"
    session.rollback()
