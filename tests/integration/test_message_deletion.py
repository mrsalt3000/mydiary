"""Integration tests for message deletion (task T15, ADR-011).

Every test runs against an isolated, migrated SQLite test database and a
temporary data root (``APP_DATA_ROOT``), so neither the production database nor
the production audio directory is ever touched. Payloads are synthetic.

Covers the deletion policy fixed in ADR-011 (which replaces the Task 05
"soft delete only" policy): a deleted message is hidden from the history, its
body text is erased, the stored audio bytes of a voice message are removed from
disk, the raw transcription text is erased while the rows and their lineage
stay, derived artifacts and traceability survive, a repeated delete is safe, and
playback of a deleted voice message is a plain 404. Task T30 adds two refusals
that used to look like success: a transcription request for a deleted message is
a 404 (no job is created), and a delete whose stored ``storage_path`` cannot be
resolved is a 409 that changes nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from _audio_samples import WAV_MIME, wav_bytes
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import Artifact, ArtifactLayer, Job, Message, MessageKind, TranscriptionStatus
from diary.db.repositories import (
    AudioFileRepository,
    MessageRepository,
    TranscriptionRepository,
)
from diary.db.results import save_layer_result, trace_layer_result
from diary.domain.audio.service import AudioNotFoundError, resolve_audio_file
from diary.domain.pipelines.sum_daily.day_input import collect_day_entries
from diary.main import app

MESSAGES_URL = "/api/v1/messages"
AUDIO_URL = "/api/v1/audio"

#: Fixed local day used by the pipeline-regression test (deterministic; the
#: default application timezone is Europe/Moscow, UTC+03:00 without DST).
LOCAL_DATE = date(2026, 9, 10)
LOCAL_INSTANT = datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc)
SYNTHETIC_TRANSCRIPTION = "Синтетический текст расшифровки."


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


@contextmanager
def _client_bound_to(session_factory) -> Iterator[TestClient]:
    """A TestClient whose ``get_session`` dependency serves ``session_factory``."""

    def _override_session() -> Iterator[Session]:
        with session_factory() as current_session:
            yield current_session

    app.dependency_overrides[get_session] = _override_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.fixture
def api_client(session_factory) -> Iterator[TestClient]:
    """TestClient bound to the per-test migrated database."""
    with _client_bound_to(session_factory) as client:
        yield client


def _create_text(api_client: TestClient, body: str) -> uuid.UUID:
    response = api_client.post(MESSAGES_URL, json={"body": body})
    assert response.status_code == 201
    return uuid.UUID(response.json()["id"])


def _create_text_at(session: Session, body: str, created_at: datetime) -> Message:
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


def _upload_wav(api_client: TestClient) -> dict:
    response = api_client.post(AUDIO_URL, files={"file": ("voice.wav", wav_bytes(), WAV_MIME)})
    assert response.status_code == 201
    return response.json()


def _delete(api_client: TestClient, message_id: uuid.UUID):
    return api_client.delete(f"{MESSAGES_URL}/{message_id}")


def _audio_path(storage_path: str) -> Path:
    """Absolute path of a stored audio file (server-generated relative name)."""
    return get_settings().audio_dir / storage_path


# ---------------------------------------------------------------------------
# text messages
# ---------------------------------------------------------------------------


def test_delete_text_message_erases_body_and_hides_it(
    api_client: TestClient, session: Session
) -> None:
    kept_id = _create_text(api_client, "Синтетическая запись, которая останется.")
    removed_id = _create_text(api_client, "Синтетическая запись к удалению.")

    assert _delete(api_client, removed_id).status_code == 204

    history = api_client.get(MESSAGES_URL)
    assert history.status_code == 200
    assert [item["id"] for item in history.json()["items"]] == [str(kept_id)]

    session.expunge_all()
    removed = MessageRepository(session).get(removed_id)
    assert removed is not None
    assert removed.deleted_at is not None  # the mark is still written
    assert removed.body_text == ""  # ... and the source text is gone
    kept = MessageRepository(session).get(kept_id)
    assert kept is not None and kept.body_text == "Синтетическая запись, которая останется."
    assert kept.deleted_at is None


def test_deleted_message_is_absent_from_the_chat_page(
    api_client: TestClient, session: Session
) -> None:
    kept = _create_text_at(session, "останется", LOCAL_INSTANT)
    removed = _create_text_at(session, "удалится", LOCAL_INSTANT)
    kept_id = kept.id
    removed_id = removed.id
    session.commit()
    session.expunge_all()

    assert _delete(api_client, removed_id).status_code == 204

    # The page shows one local day (T16b), so the day of the seeded messages
    # has to be named (LOCAL_DATE is 2026-09-10, the day both live in).
    page = api_client.get("/", params={"date": LOCAL_DATE.isoformat()})
    assert page.status_code == 200
    html = page.text
    assert f"message-{kept_id}" in html
    assert f"message-{removed_id}" not in html
    # The delete control of the remaining message is rendered with its
    # accessible name and the explicit confirmation step.
    assert f'data-message-id="{kept_id}"' in html
    assert 'data-action="delete-message-confirm"' in html


# ---------------------------------------------------------------------------
# voice messages: bytes on disk + transcription text
# ---------------------------------------------------------------------------


def test_delete_voice_message_removes_bytes_and_erases_text(
    api_client: TestClient, session: Session
) -> None:
    uploaded = _upload_wav(api_client)
    message_id = uuid.UUID(uploaded["message_id"])
    audio_file_id = uuid.UUID(uploaded["id"])

    audio = AudioFileRepository(session).get(audio_file_id)
    assert audio is not None
    path = _audio_path(audio.storage_path)
    assert path.is_file()  # fact before the deletion: the bytes are on disk

    # A synthetic STT result, as a real transcription run would leave it.
    TranscriptionRepository(session).create(
        audio_file_id=audio_file_id,
        raw_text=SYNTHETIC_TRANSCRIPTION,
        status=TranscriptionStatus.succeeded,
    )
    session.commit()
    session.expunge_all()

    assert _delete(api_client, message_id).status_code == 204

    # 1) the file is gone from disk (the fact the DoD asks to prove)
    assert not path.exists()
    # 2) the message is hidden from the history
    assert api_client.get(MESSAGES_URL).json()["items"] == []
    # 3) text is erased on both tables; the rows themselves stay for lineage
    session.expunge_all()
    message = MessageRepository(session).get(message_id)
    assert message is not None and message.deleted_at is not None and message.body_text == ""
    assert AudioFileRepository(session).get(audio_file_id) is not None
    rows = TranscriptionRepository(session).list_by_audio_file(audio_file_id)
    assert len(rows) == 2  # the not_requested sentinel + the synthetic run
    assert [row.raw_text for row in rows] == ["", ""]


def test_repeat_delete_is_idempotent_and_changes_nothing(
    api_client: TestClient, session: Session
) -> None:
    uploaded = _upload_wav(api_client)
    message_id = uuid.UUID(uploaded["message_id"])
    audio_file_id = uuid.UUID(uploaded["id"])
    audio = AudioFileRepository(session).get(audio_file_id)
    assert audio is not None
    path = _audio_path(audio.storage_path)
    session.expunge_all()

    assert _delete(api_client, message_id).status_code == 204
    session.expunge_all()
    marked = MessageRepository(session).get(message_id)
    assert marked is not None
    first_deleted_at = marked.deleted_at
    assert not path.exists()

    repeated = _delete(api_client, message_id)
    assert repeated.status_code == 404
    assert repeated.json() == {"detail": "Message not found."}

    session.expunge_all()
    after = MessageRepository(session).get(message_id)
    assert after is not None
    assert after.deleted_at == first_deleted_at  # the mark is not rewritten
    assert after.body_text == ""
    assert not path.exists()  # and a missing file is not an error
    assert api_client.get(MESSAGES_URL).json()["items"] == []
    # An unknown id behaves exactly like an already deleted one.
    assert _delete(api_client, uuid.uuid4()).status_code == 404


# ---------------------------------------------------------------------------
# links to deleted content
# ---------------------------------------------------------------------------


def test_playback_of_a_deleted_voice_message_is_a_plain_404(
    api_client: TestClient, session: Session
) -> None:
    uploaded = _upload_wav(api_client)
    message_id = uuid.UUID(uploaded["message_id"])
    audio_file_id = uuid.UUID(uploaded["id"])
    audio = AudioFileRepository(session).get(audio_file_id)
    assert audio is not None
    path = _audio_path(audio.storage_path)
    session.expunge_all()

    assert api_client.get(f"{AUDIO_URL}/{audio_file_id}/file").status_code == 200
    assert _delete(api_client, message_id).status_code == 204

    refused = api_client.get(f"{AUDIO_URL}/{audio_file_id}/file")
    assert refused.status_code == 404  # not a 500
    assert refused.json() == {"detail": "Audio file not found."}

    # Even if the unlink had failed and the bytes were still there, the audio of
    # a deleted message must not become playable (or transcribable) again.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes())
    try:
        assert api_client.get(f"{AUDIO_URL}/{audio_file_id}/file").status_code == 404
        session.rollback()
        with pytest.raises(AudioNotFoundError):
            resolve_audio_file(session, audio_file_id)
    finally:
        path.unlink(missing_ok=True)


def test_transcription_request_for_a_deleted_voice_message_is_a_404(
    api_client: TestClient, session: Session
) -> None:
    """A deleted message must not get a transcription job (T30).

    The request goes through the same resolver playback uses, so the route
    refuses with 404 and enqueues nothing — instead of a 202 whose job would
    only fail later.
    """
    uploaded = _upload_wav(api_client)
    message_id = uuid.UUID(uploaded["message_id"])
    audio_file_id = uuid.UUID(uploaded["id"])
    assert _delete(api_client, message_id).status_code == 204

    response = api_client.post(f"{AUDIO_URL}/{audio_file_id}/transcribe")

    assert response.status_code == 404
    assert response.json() == {"detail": "Audio file not found."}
    assert session.scalar(select(func.count()).select_from(Job)) == 0


def test_delete_with_a_broken_storage_path_is_a_409_and_changes_nothing(
    api_client: TestClient, session: Session
) -> None:
    """An unusable ``storage_path`` refuses the deletion instead of a 500 (T30).

    The audio paths are resolved *before* the deletion mark is written, so the
    refusal leaves the message visible, the transcription texts intact and the
    stored bytes untouched.
    """
    uploaded = _upload_wav(api_client)
    message_id = uuid.UUID(uploaded["message_id"])
    audio_file_id = uuid.UUID(uploaded["id"])
    audio = AudioFileRepository(session).get(audio_file_id)
    assert audio is not None
    path = _audio_path(audio.storage_path)
    assert path.is_file()
    TranscriptionRepository(session).create(
        audio_file_id=audio_file_id,
        raw_text=SYNTHETIC_TRANSCRIPTION,
        status=TranscriptionStatus.succeeded,
    )
    # Corrupt the server-generated relative path so it escapes the audio root.
    audio.storage_path = "../outside/escaped.wav"
    session.commit()
    session.expunge_all()

    response = _delete(api_client, message_id)

    assert response.status_code == 409
    assert response.json() == {"detail": "Message not deleted: its audio file cannot be resolved."}
    # Nothing changed: the message is still visible, its text is intact and the
    # stored bytes are still on disk.
    session.expunge_all()
    message = MessageRepository(session).get(message_id)
    assert message is not None
    assert message.deleted_at is None
    assert [item["id"] for item in api_client.get(MESSAGES_URL).json()["items"]] == [
        str(message_id)
    ]
    rows = TranscriptionRepository(session).list_by_audio_file(audio_file_id)
    assert any(row.raw_text == SYNTHETIC_TRANSCRIPTION for row in rows)
    assert path.is_file()
    assert path.read_bytes() == wav_bytes()


def test_deleting_a_message_keeps_artifacts_and_traceability(
    api_client: TestClient, session: Session
) -> None:
    message_id = _create_text(api_client, "Синтетическая запись с производным артефактом.")
    artifact = save_layer_result(
        session,
        layer=ArtifactLayer.stt_corrected,
        raw_id=message_id,
        result={
            "raw_id": str(message_id),
            "text": "Синтетическая правка текста",
            "uncertain": [],
            "model": "diary-stt",
            "prompt_ver": "v1",
        },
    )
    artifact_id = artifact.id
    session.commit()
    session.expunge_all()

    assert _delete(api_client, message_id).status_code == 204

    session.expunge_all()
    trace = trace_layer_result(session, artifact_id)
    assert trace.anchor_field == "raw_id"
    assert trace.source_kind == "message"
    # The source row is still there (and now carries the deletion mark), so the
    # derived result stays traceable after the deletion.
    assert trace.source_message is not None
    assert trace.source_message.id == message_id
    assert trace.source_message.deleted_at is not None

    stored = session.get(Artifact, artifact_id)
    assert stored is not None
    assert stored.result_json is not None
    assert stored.result_json["text"] == "Синтетическая правка текста"


# ---------------------------------------------------------------------------
# regression: pipeline input paths
# ---------------------------------------------------------------------------


def test_deleted_message_leaves_the_day_input_untouched(
    api_client: TestClient, session: Session
) -> None:
    kept = _create_text_at(session, "останется", LOCAL_INSTANT)
    removed = _create_text_at(session, "удалится", LOCAL_INSTANT)
    kept_id = kept.id
    removed_id = removed.id
    session.commit()
    session.expunge_all()

    assert _delete(api_client, removed_id).status_code == 204

    session.expunge_all()
    entries = collect_day_entries(session, local_date=LOCAL_DATE, tz=get_settings().tzinfo)
    assert [entry.text for entry in entries] == ["останется"]
    assert entries[0].id == kept_id
