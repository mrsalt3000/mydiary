"""Integration tests for the phase-2 append-only result store (task T08).

Runs against an isolated, migration-created test database (never the
production one) and covers the definition of done of T08:

* results of the second anchor family are stored without any weekly export and
  without a rendered file (``result_json`` carries the AI result);
* a repeated run **appends** ``v+1`` and never overwrites the previous version;
* the anchor invariants are enforced by the schema itself, not by the caller;
* traceability «result → source → run» resolves for every layer;
* saving results leaves the raw source tables byte-identical.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from diary.core.settings import get_settings
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    MessageKind,
    PipelineType,
    RunStatus,
    TranscriptionStatus,
)
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    ProcessingRunRepository,
    TranscriptionRepository,
)
from diary.db.results import (
    AnchorError,
    AnchorSourceError,
    LayerResultError,
    ResultPayloadError,
    UnsupportedLayerError,
    latest_layer_result,
    list_layer_results,
    next_result_version,
    save_layer_result,
    trace_layer_result,
)
from diary.domain.contracts import SttCorrection

UTC = "2026-09-19 12:00:00.000000"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _message(session: Session, *, kind: MessageKind = MessageKind.text, body: str = "Синтетика"):
    return MessageRepository(session).create(
        kind=kind, body_text=body, created_at=_utc("2026-09-19T10:00:00")
    )


def _run(session: Session, pipeline_type: PipelineType):
    return ProcessingRunRepository(session).create(
        pipeline_type=pipeline_type, status=RunStatus.succeeded
    )


# ---------------------------------------------------------------------------
# Append-only storage
# ---------------------------------------------------------------------------


def test_validated_phase2_contracts_are_stored_as_the_result_payload(session: Session) -> None:
    """The store accepts the real T00 contracts (a validated model, not raw text)."""
    message = _message(session)
    correction = SttCorrection(
        raw_id=message.id,
        text="Синтетическая правка текста",
        uncertain=["Синтетическая"],
        model="diary-stt",
        prompt_ver="v1",
    )
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=message.id,
        result=correction,
        processing_run_id=_run(session, PipelineType.stt_corrected).id,
    )
    session.commit()
    session.expunge_all()

    stored = session.get(Artifact, artifact.id)
    assert stored is not None
    assert stored.result_json == {
        "raw_id": str(message.id),
        "text": "Синтетическая правка текста",
        "uncertain": ["Синтетическая"],
        "model": "diary-stt",
        "prompt_ver": "v1",
    }


def test_stt_correction_is_stored_inline_without_a_weekly_export(session: Session) -> None:
    message = _message(session)
    run = _run(session, PipelineType.stt_corrected)

    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=message.id,
        result={"raw_id": str(message.id), "text": "Синтетическая правка", "uncertain": []},
        processing_run_id=run.id,
    )
    session.commit()

    assert artifact.export_id is None
    assert artifact.version == 1
    assert artifact.path is None and artifact.sha256 is None
    assert artifact.result_json == {
        "raw_id": str(message.id),
        "text": "Синтетическая правка",
        "uncertain": [],
    }
    assert artifact.processing_run_id == run.id
    assert artifact.created_at is not None
    assert session.scalar(text("SELECT COUNT(*) FROM weekly_exports")) == 0


def test_rerun_appends_a_new_version_and_keeps_the_previous_one(session: Session) -> None:
    message = _message(session)
    first = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=message.id,
        result={"text": "первая версия"},
    )
    session.commit()
    first_snapshot = (first.version, first.result_json, first.created_at)

    second = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=message.id,
        result={"text": "вторая версия"},
    )
    session.commit()

    assert (first.version, second.version) == (1, 2)
    assert first.id != second.id
    # the previous version is byte-identical: no field was rewritten
    assert (first.version, first.result_json, first.created_at) == first_snapshot
    lineage = list_layer_results(session, layer=ArtifactLayer.stt_corrected, raw_id=message.id)
    assert [row.version for row in lineage] == [1, 2]
    assert lineage[0].result_json == {"text": "первая версия"}
    assert (
        latest_layer_result(session, layer=ArtifactLayer.stt_corrected, raw_id=message.id) is second
    )
    assert next_result_version(session, layer=ArtifactLayer.stt_corrected, raw_id=message.id) == 3


def test_versions_are_per_anchor_and_layer_pair(session: Session) -> None:
    first_message = _message(session)
    second_message = _message(session)
    save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=first_message.id, result={"v": 1}
    )
    other = save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=second_message.id, result={"v": 1}
    )
    assert other.version == 1  # a different message is a different pair


def test_rendered_file_and_payload_are_stored_together(session: Session) -> None:
    digest = "a" * 64
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.weekly_summary,
        week_id="2026-W38",
        result={"week_id": "2026-W38", "rollup": "Синтетический rollup"},
        path="exports/05_weekly_summary/2026-W38_weekly_summary_v1.md",
        sha256=digest,
    )
    assert artifact.path == "exports/05_weekly_summary/2026-W38_weekly_summary_v1.md"
    assert artifact.sha256 == digest
    assert artifact.week_id == "2026-W38"


def test_every_phase2_layer_is_stored_and_found_by_its_anchor(session: Session) -> None:
    message = _message(session)
    run = _run(session, PipelineType.weekly_summary)

    correction = save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=message.id, result={"text": "x"}
    )
    daily = save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=_utc("2026-09-19T00:00:00").date(),
        result={"date": "2026-09-19"},
        processing_run_id=run.id,
    )
    weekly = save_layer_result(
        session, layer=ArtifactLayer.weekly_summary, week_id="2026-W38", result={"week_id": "x"}
    )
    note = save_layer_result(
        session, layer=ArtifactLayer.coach_note, summary_id=weekly.id, result={"note": "x"}
    )

    assert (
        latest_layer_result(session, layer=ArtifactLayer.stt_corrected, raw_id=message.id).id
        == correction.id
    )
    assert (
        latest_layer_result(
            session,
            layer=ArtifactLayer.daily_summary,
            local_date=daily.local_date,
            local_tz=get_settings().timezone,
        ).id
        == daily.id
    )
    assert (
        latest_layer_result(
            session, layer=ArtifactLayer.daily_summary, local_date=daily.local_date
        ).id
        == daily.id
    )
    assert (
        latest_layer_result(session, layer=ArtifactLayer.weekly_summary, week_id="2026-W38").id
        == weekly.id
    )
    assert (
        latest_layer_result(session, layer=ArtifactLayer.coach_note, summary_id=weekly.id).id
        == note.id
    )
    assert daily.local_tz == get_settings().timezone


def test_wrong_anchor_for_a_layer_is_rejected() -> None:
    # validation happens before any database access
    with pytest.raises(AnchorError, match="must not be set"):
        save_layer_result(
            None,  # type: ignore[arg-type]
            layer=ArtifactLayer.daily_summary,
            week_id="2026-W38",
            result={"x": 1},
        )
    with pytest.raises(AnchorError, match="requires"):
        save_layer_result(None, layer=ArtifactLayer.coach_note, result={"x": 1})  # type: ignore[arg-type]
    with pytest.raises(UnsupportedLayerError):
        save_layer_result(
            None,  # type: ignore[arg-type]
            layer=ArtifactLayer.summary,
            result={"x": 1},
        )
    with pytest.raises(ResultPayloadError, match="rendered file and/or"):
        save_layer_result(
            None,  # type: ignore[arg-type]
            layer=ArtifactLayer.weekly_summary,
            week_id="2026-W38",
        )


def test_result_sources_must_exist(session: Session) -> None:
    with pytest.raises(AnchorSourceError, match="message"):
        save_layer_result(
            session,
            layer=ArtifactLayer.stt_corrected,
            raw_id=uuid.uuid4(),
            result={"text": "x"},
        )
    with pytest.raises(AnchorSourceError, match="weekly summary"):
        save_layer_result(
            session,
            layer=ArtifactLayer.coach_note,
            summary_id=uuid.uuid4(),
            result={"note": "x"},
        )
    with pytest.raises(AnchorSourceError, match="processing run"):
        save_layer_result(
            session,
            layer=ArtifactLayer.weekly_summary,
            week_id="2026-W38",
            result={"week_id": "x"},
            processing_run_id=uuid.uuid4(),
        )


def test_coach_note_must_be_anchored_to_a_weekly_summary(session: Session) -> None:
    message = _message(session)
    correction = save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=message.id, result={"text": "x"}
    )
    with pytest.raises(AnchorSourceError, match="weekly_summary"):
        save_layer_result(
            session,
            layer=ArtifactLayer.coach_note,
            summary_id=correction.id,
            result={"note": "x"},
        )
    session.rollback()


# ---------------------------------------------------------------------------
# Schema-level anchor invariants (the caller cannot bypass them)
# ---------------------------------------------------------------------------


def _insert_artifact(session: Session, **columns: object) -> None:
    """Raw INSERT into ``artifacts`` so the schema constraints are hit directly."""
    statement = text(
        "INSERT INTO artifacts (id, layer, export_id, version, path, sha256, created_at,"
        " raw_id, local_date, local_tz, week_id, summary_id, result_json)"
        " VALUES (:id, :layer, :export_id, :version, :path, :sha256, :created_at,"
        " :raw_id, :local_date, :local_tz, :week_id, :summary_id, :result_json)"
    )
    values: dict[str, object] = {
        "id": uuid.uuid4().hex,
        "layer": None,
        "export_id": None,
        "version": None,
        "path": None,
        "sha256": None,
        "created_at": UTC,
        "raw_id": None,
        "local_date": None,
        "local_tz": None,
        "week_id": None,
        "summary_id": None,
        "result_json": None,
    }
    values.update(columns)
    session.execute(statement, values)


@pytest.mark.parametrize(
    ("label", "columns"),
    [
        ("no anchor at all", {"layer": "stt_corrected", "version": 1, "result_json": "{}"}),
        (
            "two anchors at once",
            {
                "layer": "stt_corrected",
                "version": 1,
                "raw_id": uuid.uuid4().hex,
                "week_id": "2026-W38",
                "result_json": "{}",
            },
        ),
        (
            "export anchor plus second-family anchor",
            {
                "layer": "normalized",
                "version": 1,
                "path": "x",
                "raw_id": uuid.uuid4().hex,
            },
        ),
        (
            "layer does not match its anchor",
            {"layer": "daily_summary", "version": 1, "week_id": "2026-W38", "result_json": "{}"},
        ),
        (
            "second-family result without a version",
            {"layer": "weekly_summary", "week_id": "2026-W38", "result_json": "{}"},
        ),
        (
            "second-family result without any payload",
            {"layer": "weekly_summary", "week_id": "2026-W38", "version": 1},
        ),
        (
            "date anchor without its timezone",
            {
                "layer": "daily_summary",
                "local_date": "2026-09-19",
                "version": 1,
                "result_json": "{}",
            },
        ),
        (
            "timezone without a date anchor",
            {
                "layer": "weekly_summary",
                "week_id": "2026-W38",
                "local_tz": "UTC",
                "version": 1,
                "result_json": "{}",
            },
        ),
        ("unknown layer value", {"layer": "telegram", "version": 1, "result_json": "{}"}),
    ],
)
def test_anchor_family_invariant_is_enforced_by_the_schema(
    session: Session, label: str, columns: dict[str, object]
) -> None:
    with pytest.raises(IntegrityError):
        _insert_artifact(session, **columns)
    session.rollback()


def test_first_family_rows_keep_their_file_only_shape(session: Session) -> None:
    """The MVP artifact shape stays exactly what it was: a file, no inline result."""
    export_id = uuid.uuid4()
    session.execute(
        text(
            "INSERT INTO weekly_exports (id, week_end_date, seq_number, revision, status, path,"
            " sha256, created_at) VALUES (:id, '2026-09-06', 5, 1, 'created',"
            " 'exports/01_original/x.txt', :sha, :created)"
        ),
        {"id": export_id.hex, "sha": "b" * 64, "created": UTC},
    )
    session.commit()  # committed so the rollbacks below keep the export row

    for columns in (
        {"result_json": "{}"},  # no file at all
        {"path": "exports/02_normalized/x.txt", "sha256": "c" * 64, "result_json": "{}"},
    ):
        with pytest.raises(IntegrityError):
            _insert_artifact(
                session, layer="normalized", export_id=export_id.hex, version=1, **columns
            )
        session.rollback()

    _insert_artifact(
        session,
        layer="normalized",
        export_id=export_id.hex,
        version=1,
        path="exports/02_normalized/x.txt",
        sha256="c" * 64,
    )
    session.flush()
    session.rollback()


def test_duplicate_version_of_a_pair_is_rejected_by_the_schema(session: Session) -> None:
    message = _message(session)
    save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=message.id, result={"text": "v1"}
    )
    session.flush()
    with pytest.raises(IntegrityError):
        _insert_artifact(
            session,
            layer="stt_corrected",
            raw_id=message.id.hex,
            version=1,
            result_json="{}",
        )
    session.rollback()


def test_first_family_rows_stay_valid_next_to_second_family_rows(session: Session) -> None:
    """The MVP artifact shape (export anchor, file, no result) is untouched."""
    export_id = uuid.uuid4()
    session.execute(
        text(
            "INSERT INTO weekly_exports (id, week_end_date, seq_number, revision, status, path,"
            " sha256, created_at) VALUES (:id, '2026-09-13', 1, 1, 'created',"
            " 'exports/01_original/x.txt', :sha, :created)"
        ),
        {"id": export_id.hex, "sha": "b" * 64, "created": UTC},
    )
    _insert_artifact(
        session,
        layer="original",
        export_id=export_id.hex,
        path="exports/01_original/x.txt",
        sha256="b" * 64,
    )
    _insert_artifact(
        session,
        layer="normalized",
        export_id=export_id.hex,
        version=1,
        path="exports/02_normalized/x.txt",
        sha256="c" * 64,
    )
    session.flush()  # both rows must satisfy the new constraints

    message = _message(session)
    save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=message.id, result={"text": "x"}
    )
    session.commit()
    assert session.scalar(text("SELECT COUNT(*) FROM artifacts")) == 3


# ---------------------------------------------------------------------------
# Traceability and raw immutability
# ---------------------------------------------------------------------------


def test_traceability_resolves_result_source_and_run(session: Session) -> None:
    message = _message(session, kind=MessageKind.audio, body="")
    run = _run(session, PipelineType.stt_corrected)
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=message.id,
        result={"text": "Синтетическая правка"},
        processing_run_id=run.id,
    )
    session.commit()

    trace = trace_layer_result(session, artifact.id)
    assert trace.layer is ArtifactLayer.stt_corrected
    assert trace.anchor_field == "raw_id"
    assert trace.anchor_value == message.id
    assert trace.source_kind == "message"
    assert trace.source_message is not None and trace.source_message.id == message.id
    assert trace.run is not None and trace.run.id == run.id
    assert trace.run.pipeline_type is PipelineType.stt_corrected


def test_traceability_of_a_coach_note_points_at_the_weekly_summary(session: Session) -> None:
    weekly = save_layer_result(
        session, layer=ArtifactLayer.weekly_summary, week_id="2026-W38", result={"week_id": "x"}
    )
    note = save_layer_result(
        session, layer=ArtifactLayer.coach_note, summary_id=weekly.id, result={"note": "x"}
    )
    session.commit()

    trace = trace_layer_result(session, note.id)
    assert trace.anchor_field == "summary_id"
    assert trace.source_kind == "artifact"
    assert trace.source_artifact is not None
    assert trace.source_artifact.id == weekly.id
    assert trace.source_artifact.layer is ArtifactLayer.weekly_summary


def test_traceability_rejects_unknown_and_first_family_artifacts(session: Session) -> None:
    with pytest.raises(LayerResultError, match="not found"):
        trace_layer_result(session, uuid.uuid4())

    export_id = uuid.uuid4()
    session.execute(
        text(
            "INSERT INTO weekly_exports (id, week_end_date, seq_number, revision, status, path,"
            " sha256, created_at) VALUES (:id, '2026-09-13', 2, 1, 'created', 'exports/x.txt',"
            " :sha, :created)"
        ),
        {"id": export_id.hex, "sha": "b" * 64, "created": UTC},
    )
    _insert_artifact(
        session,
        layer="normalized",
        export_id=export_id.hex,
        version=1,
        path="exports/02_normalized/x.txt",
        sha256="c" * 64,
    )
    session.flush()
    artifact = session.scalar(select(Artifact).where(Artifact.layer == ArtifactLayer.normalized))
    assert artifact is not None
    with pytest.raises(UnsupportedLayerError):
        trace_layer_result(session, artifact.id)
    session.rollback()


def test_saving_results_leaves_the_raw_sources_untouched(session: Session) -> None:
    """T08 definition of done: ``raw`` is immutable, only results are appended."""
    message = _message(session, kind=MessageKind.audio, body="")
    audio = AudioFileRepository(session).create(
        message_id=message.id,
        original_mime="audio/webm",
        storage_path=f"2026/09/{uuid.uuid4()}.webm",
        size_bytes=123,
        sha256="d" * 64,
        created_at=_utc("2026-09-19T10:00:01"),
    )
    transcription = TranscriptionRepository(session).create(
        audio_file_id=audio.id,
        raw_text="сырой синтетический текст",
        status=TranscriptionStatus.succeeded,
        created_at=_utc("2026-09-19T10:00:02"),
    )
    session.commit()

    def raw_state() -> list[tuple]:
        return [
            tuple(session.execute(text(f"SELECT * FROM {table} ORDER BY id")).all())  # noqa: S608
            for table in ("messages", "audio_files", "transcriptions", "weekly_exports")
        ]

    before = raw_state()
    message_text_before = message.body_text
    message_created_before = message.created_at
    raw_text_before = transcription.raw_text

    run = _run(session, PipelineType.coach_note)
    save_layer_result(
        session, layer=ArtifactLayer.stt_corrected, raw_id=message.id, result={"text": "правка"}
    )
    daily = save_layer_result(
        session,
        layer=ArtifactLayer.daily_summary,
        local_date=_utc("2026-09-19T00:00:00").date(),
        result={"date": "2026-09-19"},
        processing_run_id=run.id,
    )
    weekly = save_layer_result(
        session, layer=ArtifactLayer.weekly_summary, week_id="2026-W38", result={"week_id": "x"}
    )
    save_layer_result(
        session, layer=ArtifactLayer.coach_note, summary_id=weekly.id, result={"note": "x"}
    )
    session.commit()
    session.expunge_all()

    assert raw_state() == before
    stored_message = MessageRepository(session).get(message.id)
    stored_transcription = TranscriptionRepository(session).get(transcription.id)
    assert stored_message is not None
    assert stored_message.body_text == message_text_before
    assert stored_message.created_at == message_created_before
    assert stored_message.deleted_at is None
    assert stored_transcription is not None
    assert stored_transcription.raw_text == raw_text_before
    # the results themselves are appended, not overwritten
    assert daily.layer is ArtifactLayer.daily_summary
    assert session.scalar(text("SELECT COUNT(*) FROM artifacts")) == 4
