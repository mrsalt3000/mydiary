"""Integration tests for the audio upload/storage API and UI (Task 07).

Every test runs against an isolated, migrated SQLite test database and a
temporary data root (fixtures from ``tests/integration/conftest.py`` plus an
``APP_DATA_ROOT`` override), so no file ever touches the production ``data/``
directory. Payloads are synthetic (see ``_audio_samples.py``).

Covers: upload → 201 with consistent DB rows and a server-UUID file path;
playback; persistence across an engine restart; validation errors (MIME, size,
empty, container mismatch); path-traversal resistance; cleanup when the
database transaction fails; 404s; and the chat page rendering audio messages.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import pytest
from _audio_samples import (
    WAV_MIME,
    WEBM_MIME,
    not_audio_bytes,
    wav_bytes,
    webm_bytes,
)
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import AudioFile, Job, Message, MessageKind, Transcription
from diary.db.repositories import AudioFileRepository, MessageRepository, TranscriptionRepository
from diary.db.session import create_engine, create_session_factory
from diary.main import app

AUDIO_URL = "/api/v1/audio"

#: A server-side storage path: YYYY/MM/<uuid>.<ext>.
SERVER_PATH_RE = re.compile(
    r"^\d{4}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.wav$"
)


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


def _upload(client: TestClient, data: bytes, mime: str, name: str = "voice.wav"):
    return client.post(AUDIO_URL, files={"file": (name, data, mime)})


def _files_below(root) -> list:
    """Relative paths of every regular file under ``root`` (sorted)."""
    root_path = root.resolve()
    return sorted(
        str(path.relative_to(root_path)) for path in root_path.rglob("*") if path.is_file()
    )


# ---------------------------------------------------------------------------
# upload success and consistency
# ---------------------------------------------------------------------------


def test_upload_creates_consistent_audio_message(api_client: TestClient, session: Session) -> None:
    payload = wav_bytes()
    response = _upload(api_client, payload, WAV_MIME)
    assert response.status_code == 201
    body = response.json()

    # Response shape: audio-file metadata, no user file name anywhere.
    assert set(body) == {
        "id",
        "message_id",
        "mime",
        "size_bytes",
        "transcription_status",
        "created_at",
    }
    assert body["mime"] == WAV_MIME
    assert body["size_bytes"] == len(payload)
    assert body["transcription_status"] == "not_requested"
    assert body["created_at"].endswith("+03:00")  # default Europe/Moscow
    datetime.fromisoformat(body["created_at"])
    audio_id = uuid.UUID(body["id"])
    message_id = uuid.UUID(body["message_id"])

    # --- database rows are consistent --------------------------------
    audio = AudioFileRepository(session).get(audio_id)
    assert audio is not None
    message = MessageRepository(session).get(message_id)
    assert message is not None
    assert message.kind is MessageKind.audio
    assert message.body_text == ""
    assert message.deleted_at is None

    transcriptions = TranscriptionRepository(session).list_by_audio_file(audio_id)
    assert len(transcriptions) == 1
    assert transcriptions[0].status.value == "not_requested"
    assert transcriptions[0].raw_text == ""

    assert audio.message_id == message_id
    assert audio.original_mime == WAV_MIME
    assert audio.size_bytes == len(payload)
    assert audio.sha256 == hashlib.sha256(payload).hexdigest()
    assert audio.duration_seconds is None  # no demuxer in MVP (documented)

    # --- file lives at a server-generated path -----------------------
    assert SERVER_PATH_RE.match(audio.storage_path)
    storage_path = audio.storage_path
    assert storage_path.endswith(f"{audio_id}.wav")
    audio_root = get_settings().audio_dir
    stored_file = (audio_root / storage_path).resolve()
    assert audio_root.resolve() in stored_file.parents
    assert stored_file.is_file()
    assert stored_file.read_bytes() == payload

    # --- no automatic transcription job is created -------------------
    assert session.scalar(select(func.count()).select_from(Job)) == 0


def test_upload_accepts_webm_container_and_sets_webm_extension(
    api_client: TestClient, session: Session
) -> None:
    payload = webm_bytes()
    response = _upload(api_client, payload, WEBM_MIME, name="my-recording.webm")
    assert response.status_code == 201
    audio_id = uuid.UUID(response.json()["id"])
    assert response.json()["mime"] == WEBM_MIME

    audio = AudioFileRepository(session).get(audio_id)
    assert audio is not None
    assert audio.storage_path.endswith(f"{audio_id}.webm")


def test_upload_ignores_user_supplied_file_names(api_client: TestClient, session: Session) -> None:
    payload = wav_bytes()
    hostile_names = [
        "../outside.wav",
        "..%2f..%2foutside.wav",
        "/tmp/absolute-outside.wav",
        "..\\..\\windows-outside.wav",
        "voice.wav",
    ]
    audio_root = get_settings().audio_dir
    data_root = get_settings().data_root
    before = set(_files_below(audio_root))
    for name in hostile_names:
        response = _upload(api_client, payload, WAV_MIME, name=name)
        assert response.status_code == 201, name
        audio_id = uuid.UUID(response.json()["id"])
        audio = AudioFileRepository(session).get(audio_id)
        assert audio is not None
        # The stored path is always server generated: YYYY/MM/<uuid>.wav.
        assert SERVER_PATH_RE.match(audio.storage_path)
        assert audio.storage_path.endswith(f"{audio_id}.wav")
        assert ".." not in audio.storage_path
        assert not audio.storage_path.startswith("/")
        # Exactly one new file appeared, at the server-side path, inside the
        # audio directory — nothing was written outside it.
        after = set(_files_below(audio_root))
        assert after == before | {audio.storage_path}
        before = after
    # And nothing at all was written outside the audio directory.
    assert set(_files_below(data_root)) == {f"audio/{path}" for path in before}


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------


def test_uploaded_file_plays_back_bytes(api_client: TestClient, session: Session) -> None:
    payload = wav_bytes()
    audio_id = _upload(api_client, payload, WAV_MIME).json()["id"]

    playback = api_client.get(f"{AUDIO_URL}/{audio_id}/file")
    assert playback.status_code == 200
    assert playback.headers["content-type"].startswith(WAV_MIME)
    assert playback.content == payload


def test_audio_survives_engine_restart_and_is_playable(
    api_client: TestClient, session: Session
) -> None:
    """Upload on one engine, then read DB + file with a brand-new engine."""
    payload = wav_bytes()
    created = _upload(api_client, payload, WAV_MIME)
    assert created.status_code == 201
    audio_id = created.json()["id"]
    message_id = created.json()["message_id"]

    # A second engine over the same migrated database file still sees the row
    # and the bytes are still served (application restart simulation).
    restarted_engine = create_engine(_db_url_of(session))
    try:
        restarted_factory = create_session_factory(restarted_engine)
        with _client_bound_to(restarted_factory) as restarted:
            playback = restarted.get(f"{AUDIO_URL}/{audio_id}/file")
            assert playback.status_code == 200
            assert playback.content == payload
            history = restarted.get("/api/v1/messages")
            assert history.status_code == 200
            items = history.json()["items"]
            assert [item["id"] for item in items] == [message_id]
            assert items[0]["kind"] == "audio"
    finally:
        restarted_engine.dispose()


def _db_url_of(session: Session) -> str:
    """The SQLite URL of the engine backing ``session``."""
    return session.get_bind().url.render_as_string(hide_password=False)


# ---------------------------------------------------------------------------
# chat page rendering of audio messages
# ---------------------------------------------------------------------------


def test_chat_page_renders_audio_message_with_player(
    api_client: TestClient,
) -> None:
    payload = wav_bytes()
    created = _upload(api_client, payload, WAV_MIME).json()
    audio_id = created["id"]
    message_id = created["message_id"]

    page = api_client.get("/")
    assert page.status_code == 200
    html = page.text
    assert f'id="message-{message_id}"' in html
    assert f'src="/api/v1/audio/{audio_id}/file"' in html
    assert "<audio" in html
    assert "Голосовое" in html
    assert "Не расшифровано" in html
    assert "message--audio" in html


def test_older_fragment_renders_audio_messages(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio messages inside older history windows render through the same
    partial, so progressive loading keeps working for both message kinds."""
    import diary.api.routes.pages as pages_module

    monkeypatch.setattr(pages_module, "PAGE_LIMIT", 1)
    audio = _upload(api_client, wav_bytes(), WAV_MIME).json()
    text = api_client.post("/api/v1/messages", json={"body": "новая текстовая запись"})
    assert text.status_code == 201

    page = api_client.get("/")
    html = page.text
    assert text.json()["id"] in html  # newest window shows the text message
    assert f'src="/api/v1/audio/{audio["id"]}/file"' not in html

    match = re.search(r'hx-get="/messages/fragments/older\?cursor=([^"]+)"', html)
    assert match is not None
    cursor = urllib.parse.unquote(match.group(1))
    fragment = api_client.get("/messages/fragments/older", params={"cursor": cursor, "limit": "1"})
    assert fragment.status_code == 200
    body = fragment.text
    assert f'id="message-{audio["message_id"]}"' in body
    assert f'src="/api/v1/audio/{audio["id"]}/file"' in body
    assert "Не расшифровано" in body


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------


def test_upload_rejects_unsupported_mime(api_client: TestClient) -> None:
    response = _upload(api_client, wav_bytes(), "application/octet-stream")
    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported audio type."}


def test_upload_rejects_content_mismatching_mime(api_client: TestClient) -> None:
    response = _upload(api_client, not_audio_bytes(), WAV_MIME)
    assert response.status_code == 400
    assert response.json() == {"detail": "The file does not look like the declared audio type."}


def test_upload_rejects_empty_file(api_client: TestClient) -> None:
    response = _upload(api_client, b"", WAV_MIME)
    assert response.status_code == 400
    assert response.json() == {"detail": "Audio file is empty."}


def test_upload_rejects_oversized_file(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_MAX_AUDIO_MB", "1")
    reload_settings()
    try:
        response = _upload(api_client, wav_bytes(payload_size=1024 * 1024), WAV_MIME)
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]
        assert "1 MiB" in response.json()["detail"]
    finally:
        reload_settings()


def test_upload_missing_file_field_returns_422(api_client: TestClient) -> None:
    response = api_client.post(AUDIO_URL, data={"note": "no file here"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# playback errors
# ---------------------------------------------------------------------------


def test_get_unknown_audio_file_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{AUDIO_URL}/{uuid.uuid4()}/file")
    assert response.status_code == 404
    assert response.json() == {"detail": "Audio file not found."}


def test_get_malformed_audio_id_returns_422(api_client: TestClient) -> None:
    response = api_client.get(f"{AUDIO_URL}/not-a-uuid/file")
    assert response.status_code == 422


def test_get_missing_bytes_returns_404(api_client: TestClient, session: Session) -> None:
    """A row whose bytes vanished from disk still answers a plain 404."""
    payload = wav_bytes()
    audio_id = uuid.UUID(_upload(api_client, payload, WAV_MIME).json()["id"])
    audio = AudioFileRepository(session).get(audio_id)
    assert audio is not None
    (get_settings().audio_dir / audio.storage_path).unlink()

    response = api_client.get(f"{AUDIO_URL}/{audio_id}/file")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# failure cleanup
# ---------------------------------------------------------------------------


def test_persistence_failure_removes_written_file_and_rolls_back(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("forced persistence failure")

    monkeypatch.setattr(AudioFileRepository, "create", _boom)

    response = _upload(api_client, wav_bytes(), WAV_MIME)
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to store the audio file."}

    # No rows survive the rollback.
    assert session.scalar(select(func.count()).select_from(Message)) == 0
    assert session.scalar(select(func.count()).select_from(AudioFile)) == 0
    assert session.scalar(select(func.count()).select_from(Transcription)) == 0

    # And no orphaned file remains anywhere under the data root.
    data_root = get_settings().data_root
    assert _files_below(data_root) == []
