"""Integration tests for the local STT pipeline (Task 09).

End-to-end through the real worker helpers and repository code, against an
isolated migrated test database and a temporary data root. Transcription uses
a fake provider and a fixed profile (monkeypatched into the domain service) —
no faster-whisper package, model or GPU is ever needed.

Covers: job creation/dedup for ``request_transcription``; worker success
(single ``succeeded`` row, audio/message untouched, file intact); replay
safety after a stale-running reset (no duplicate rows); provider failure with
retries ending in exactly one terminal ``failed`` row; retranscription
appending a new version without erasing history; the API (POST transcribe,
GET transcriptions, 404s, empty history); and the chat-page rendering of the
button, statuses and raw text.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from _audio_samples import WAV_MIME, wav_bytes
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.models_config import FasterWhisperProfile, GigaamCtcLmProfile
from diary.core.settings import get_settings, reload_settings
from diary.db.base import utc_now
from diary.db.models import Job, JobStatus, JobType, TranscriptionStatus
from diary.db.repositories import (
    AudioFileRepository,
    JobRepository,
    MessageRepository,
    TranscriptionRepository,
)
from diary.domain.audio.service import AudioNotFoundError, store_audio
from diary.domain.transcription import providers as transcription_providers
from diary.domain.transcription import service as transcription_service
from diary.main import app
from diary.workers import handlers
from diary.workers.runner import process_due_jobs

AUDIO_URL = "/api/v1/audio"

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_MAX_AUDIO_MB",
    "APP_ALLOWED_AUDIO_MIME_TYPES",
    "APP_JOB_MAX_ATTEMPTS",
    "APP_JOB_BACKOFF_BASE_S",
    "APP_JOB_BACKOFF_MULTIPLIER",
    "APP_JOB_MAX_BACKOFF_S",
    "APP_JOB_STALE_AFTER_S",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Restore the baseline handler registry after every test."""
    snapshot = dict(handlers.REGISTRY)
    yield
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(snapshot)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTranscriptionProvider:
    """Deterministic fake STT provider: returns queued outputs and can fail a
    fixed number of times before producing any text (the ``version`` attribute
    mirrors the real adapter's provenance hook).

    Since T11 the manual STT path runs through ``asr.run``, so the fake also
    implements the provider protocol's ``unload()`` (released after every turn)
    and appends to ``journal`` when one is given — that is how the order «LLM
    unloaded → STT → Whisper unloaded» is asserted end-to-end.
    """

    version = "9.9.9"

    def __init__(
        self, outputs: list[str] | None = None, failures: int = 0, journal: list[str] | None = None
    ) -> None:
        self.outputs = list(outputs) if outputs else ["Синтетический текст расшифровки."]
        self.failures = failures
        self.calls = 0
        self.unload_calls = 0
        self.journal = journal

    def transcribe(self, audio_path, profile: FasterWhisperProfile) -> str:
        self.calls += 1
        if self.journal is not None:
            self.journal.append("stt")
        if self.calls <= self.failures:
            raise RuntimeError("synthetic provider failure")
        index = min(self.calls - 1 - self.failures, len(self.outputs) - 1)
        return self.outputs[index]

    def unload(self) -> None:
        self.unload_calls += 1
        if self.journal is not None:
            self.journal.append("whisper_unload")


def _fake_profile() -> FasterWhisperProfile:
    return FasterWhisperProfile(
        provider="faster_whisper",
        model="tiny",
        language="ru",
        device="cpu",
        compute_type="int8",
        beam_size=1,
    )


def _install_fake_transcription(
    monkeypatch: pytest.MonkeyPatch,
    provider: FakeTranscriptionProvider,
    *,
    unload_journal: list[str] | None = None,
) -> None:
    """Replace the profile loader, the provider registry and the LLM unloader.

    The manual path runs through ``asr.run`` (T11), which resolves its provider
    from the providers **registry**; the service's provenance probe uses the
    same registry entry. ``unload_journal`` (when given) records the injected
    LLM unload, so a test can assert it happens *before* STT.
    """
    monkeypatch.setattr(
        transcription_service,
        "_load_transcription_profile",
        lambda: _fake_profile(),
    )
    monkeypatch.setitem(
        transcription_providers.PROVIDER_REGISTRY, "faster_whisper", lambda: provider
    )

    def _unload_llm() -> None:
        if unload_journal is not None:
            unload_journal.append("llm_unload")

    monkeypatch.setattr(transcription_service, "_default_llm_unloader", _unload_llm)


def _upload_audio(api_client: TestClient) -> tuple[uuid.UUID, uuid.UUID]:
    """Upload one synthetic WAV through the API; return audio id and message id."""
    payload = wav_bytes()
    response = api_client.post(AUDIO_URL, files={"file": ("voice.wav", payload, WAV_MIME)})
    assert response.status_code == 201
    return uuid.UUID(response.json()["id"]), uuid.UUID(response.json()["message_id"])


def _enqueue(session_factory, audio_id: uuid.UUID, *, max_attempts: int = 3) -> uuid.UUID:
    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.transcribe_audio,
            payload_json={"audio_file_id": str(audio_id), "profile_ref": "transcription"},
            max_attempts=max_attempts,
        )
        job_id = job.id
        session.commit()
    return job_id


def _make_retry_due(session_factory, job_id: uuid.UUID) -> None:
    """Force a ``retrying`` job's ``next_retry_at`` into the past."""
    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.retrying
        job.next_retry_at = utc_now() - timedelta(seconds=1)
        session.commit()


def _exhaust_job(session_factory, job_id: uuid.UUID, *, max_attempts: int = 3) -> None:
    """Run a job through every retry attempt until it becomes terminal."""
    for index in range(max_attempts):
        assert process_due_jobs(session_factory, worker_id="test-worker") == 1
        with session_factory() as session:
            job = JobRepository(session).get(job_id)
            assert job is not None
            if job.status is JobStatus.failed:
                return
        if index < max_attempts - 1:
            _make_retry_due(session_factory, job_id)


def _audio_bytes(session_factory, audio_id: uuid.UUID) -> bytes:
    with session_factory() as session:
        audio = AudioFileRepository(session).get(audio_id)
        assert audio is not None
        return (get_settings().audio_dir / audio.storage_path).read_bytes()


def _run_rows(session_factory, audio_id: uuid.UUID):
    """Transcription rows that represent real runs (sentinel excluded)."""
    with session_factory() as session:
        rows = TranscriptionRepository(session).list_by_audio_file(audio_id)
        return [row for row in rows if row.status is not TranscriptionStatus.not_requested]


def _job_count(session_factory) -> int:
    with session_factory() as session:
        return session.scalar(select(func.count()).select_from(Job)) or 0


# ---------------------------------------------------------------------------
# request_transcription (service level)
# ---------------------------------------------------------------------------


def test_request_transcription_creates_job_and_reuses_active_one(
    session,
    session_factory,
) -> None:
    stored = store_audio(session, data=wav_bytes(), content_type=WAV_MIME)
    first = transcription_service.request_transcription(session, stored.audio_file_id)
    second = transcription_service.request_transcription(session, stored.audio_file_id)
    assert second.id == first.id
    assert _job_count(session_factory) == 1
    # Payload holds identifiers only, plus the profile reference.
    assert first.payload_json == {
        "audio_file_id": str(stored.audio_file_id),
        "profile_ref": "transcription",
    }
    # A second request returns the existing active job, never a duplicate.
    with session_factory() as fresh:
        active = transcription_service.find_active_transcription_job(fresh, stored.audio_file_id)
        assert active is not None and active.id == first.id


def test_request_transcription_for_unknown_audio_raises(session) -> None:
    with pytest.raises(AudioNotFoundError):
        transcription_service.request_transcription(session, uuid.uuid4())


# ---------------------------------------------------------------------------
# unregistered STT backend (T43/ADR-017)
# ---------------------------------------------------------------------------


def test_unregistered_stt_backend_fails_before_stt_and_writes_no_row(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    """A configured backend with no adapter never reaches speech-to-text.

    The GigaAM-v2 CTC+LM profile has an adapter from T44 on, so the scenario is
    made by removing that adapter from the registry — which is exactly what a
    third, unknown backend would look like. The service resolves the provider
    **before** ``asr.run``, so the refusal is loud and cheap: no GPU slot, no
    decode, and — the observable half of "no silent fallback" — no transcription
    row that would present the audio as transcribed (a fallback to Whisper would
    have written one).
    """
    audio_id, _ = _upload_audio(api_client)
    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    monkeypatch.setattr(
        transcription_service,
        "_load_transcription_profile",
        lambda: GigaamCtcLmProfile(
            provider="gigaam2_ctc_lm",
            model_id="waveletdeboshir/gigaam-ctc-with-lm",
            revision="07d97cf8a4f178b68679cafae0600c8487877301",
        ),
    )
    monkeypatch.delitem(transcription_providers.PROVIDER_REGISTRY, "gigaam2_ctc_lm")
    asr_calls: list[str] = []

    def _forbidden_asr(*args: object, **kwargs: object) -> str:
        asr_calls.append("asr.run")
        raise AssertionError("asr.run must not be reached for an unregistered backend")

    monkeypatch.setattr(transcription_service.asr, "run", _forbidden_asr)

    _exhaust_job(session_factory, job_id)

    assert asr_calls == []
    assert _run_rows(session_factory, audio_id) == []
    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None
        # The job fails with a safe summary naming the missing provider — the
        # retry policy is the worker's business, the message is the user's.
        # The boundary wraps the refusal in ``PermanentJobError`` (the class the
        # worker stores), and the original message survives in the summary.
        assert job.status is JobStatus.failed
        assert job.error_summary is not None
        assert job.error_summary.startswith("PermanentJobError:")
        assert "gigaam2_ctc_lm" in job.error_summary


def test_unregistered_stt_backend_fails_on_the_first_attempt(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    """T44 DoD 6: a refusal retrying cannot fix must not burn the attempt budget.

    Before T44 the factory's refusal was a bare ``TranscriptionProviderError``
    and reached the worker as *transient*: the job was retried ``max_attempts``
    times for a configuration that cannot change between attempts. It now
    carries ``retryable = False``, so one attempt is the whole story — and no
    history row is written for a run that never happened.
    """
    audio_id, _ = _upload_audio(api_client)
    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    monkeypatch.setattr(
        transcription_service,
        "_load_transcription_profile",
        lambda: GigaamCtcLmProfile(
            provider="gigaam2_ctc_lm",
            model_id="waveletdeboshir/gigaam-ctc-with-lm",
            revision="07d97cf8a4f178b68679cafae0600c8487877301",
        ),
    )
    monkeypatch.delitem(transcription_providers.PROVIDER_REGISTRY, "gigaam2_ctc_lm")

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None
        assert job.status is JobStatus.failed
        assert job.attempts == 1
        assert job.error_summary is not None
        assert job.error_summary.startswith("PermanentJobError:")
        assert "No transcription provider is registered" in job.error_summary

    assert _run_rows(session_factory, audio_id) == []


def test_gigaam_run_records_backend_model_and_decoding_knobs(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    """T44 DoD 4: a GigaAM run's row names the backend, the model and the knobs.

    The adapter is faked (no sidecar venv, no GPU), but everything the row is
    built from is the real profile and the real service path: ``provider_name``
    comes from the discriminator, ``model_name`` from the shared
    ``transcription_model_name`` helper, ``language`` from the Russian-only
    constant, and ``metadata_json`` from the provider-dependent snapshot. The
    Whisper key set is pinned by its own unit test and is not touched here.
    """
    audio_id, _ = _upload_audio(api_client)
    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])

    profile = GigaamCtcLmProfile(
        provider="gigaam2_ctc_lm",
        model_id="waveletdeboshir/gigaam-ctc-with-lm",
        revision="07d97cf8a4f178b68679cafae0600c8487877301",
        device="cuda",
        beam_width=8,
        alpha=0.5,
        # beta stays None: "the checkpoint's own value", recorded as such.
    )

    class _FakeSidecar:
        version = "sidecar-test"

        def transcribe(self, audio_path, profile) -> str:  # type: ignore[no-untyped-def]
            return "Синтетический текст GigaAM."

        def unload(self) -> None:  # pragma: no cover - the service always calls it
            return None

    fake = _FakeSidecar()
    monkeypatch.setattr(transcription_service, "_load_transcription_profile", lambda: profile)
    monkeypatch.setitem(transcription_providers.PROVIDER_REGISTRY, "gigaam2_ctc_lm", lambda: fake)
    monkeypatch.setattr(transcription_service, "_default_llm_unloader", lambda: None)

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.succeeded and job.attempts == 1
        rows = _run_rows(session_factory, audio_id)
        assert len(rows) == 1
        row = rows[0]
        assert row.raw_text == "Синтетический текст GigaAM."
        assert row.provider_name == "gigaam2_ctc_lm"
        assert row.model_name == "waveletdeboshir/gigaam-ctc-with-lm"
        assert row.language == "ru"
        assert row.metadata_json is not None
        assert row.metadata_json["provider_version"] == "sidecar-test"
        # The decoding knobs and the pin are what a later reader needs to know
        # how this row was produced; ``alpha: 0.5`` is the configured value and
        # ``beta: null`` honestly says "the checkpoint's own".
        assert row.metadata_json["device"] == "cuda"
        assert row.metadata_json["beam_width"] == 8
        assert row.metadata_json["alpha"] == 0.5
        assert row.metadata_json["beta"] is None
        assert row.metadata_json["revision"] == "07d97cf8a4f178b68679cafae0600c8487877301"
        assert row.metadata_json["job_id"] == str(job_id)


# ---------------------------------------------------------------------------
# worker success / replay safety
# ---------------------------------------------------------------------------


def test_transcribe_job_succeeds_and_persists_raw_without_touching_audio(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    fake = FakeTranscriptionProvider(outputs=["Синтетический текст расшифровки."])
    _install_fake_transcription(monkeypatch, fake)
    audio_id, message_id = _upload_audio(api_client)
    before = _audio_bytes(session_factory, audio_id)

    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {
        "id",
        "job_type",
        "status",
        "attempts",
        "max_attempts",
        "error_summary",
        "created_at",
        "updated_at",
    }
    assert body["job_type"] == "transcribe_audio"
    assert body["status"] == "pending"
    job_id = uuid.UUID(body["id"])

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.succeeded
        assert job.attempts == 1

        rows = TranscriptionRepository(session).list_by_audio_file(audio_id)
        runs = [row for row in rows if row.status is not TranscriptionStatus.not_requested]
        assert len(runs) == 1
        row = runs[0]
        assert row.status is TranscriptionStatus.succeeded
        assert row.raw_text == "Синтетический текст расшифровки."
        assert row.profile_name == "transcription"
        assert row.provider_name == "faster_whisper"
        assert row.model_name == "tiny"
        assert row.language == "ru"
        assert row.metadata_json is not None
        assert row.metadata_json["job_id"] == str(job_id)
        assert row.metadata_json["provider_version"] == "9.9.9"
        assert row.metadata_json["device"] == "cpu"

        # Audio and message rows are untouched.
        audio = AudioFileRepository(session).get(audio_id)
        assert audio is not None and audio.sha256
        message = MessageRepository(session).get(message_id)
        assert message is not None and message.kind.value == "audio"

    # File bytes on disk are intact and the model ran exactly once.
    assert _audio_bytes(session_factory, audio_id) == before
    assert fake.calls == 1


def test_the_manual_stt_job_unloads_the_llm_before_it_recognises(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    """The manual button now goes through ``asr.run`` (T11, ADR-010).

    The whole point of the T09→T11 requirement: the LLM is released
    **deliberately** before speech-to-text starts (not because the server
    happens to be started with ``OLLAMA_KEEP_ALIVE=0``), and Whisper is
    released again afterwards — the phase-2 VRAM veto expressed as an order.
    """
    journal: list[str] = []
    fake = FakeTranscriptionProvider(outputs=["Синтетика расшифровки."], journal=journal)
    _install_fake_transcription(monkeypatch, fake, unload_journal=journal)
    audio_id, _ = _upload_audio(api_client)

    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    assert journal == ["llm_unload", "stt", "whisper_unload"]
    # The transcription was released exactly once, and the slot-free order is
    # the one ``asr.run`` guarantees (see tests/unit/test_asr_sequential.py).
    assert fake.unload_calls == 1


def test_replaying_a_succeeded_job_after_stale_reset_does_not_duplicate(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    """Worker "crash" after the handler committed the row, then a stale-running
    reset replays the job: the handler must detect the recorded outcome and
    not run the model a second time or create a second row."""
    fake = FakeTranscriptionProvider(outputs=["первая версия"])
    _install_fake_transcription(monkeypatch, fake)
    audio_id, _ = _upload_audio(api_client)

    with session_factory() as session:
        job = JobRepository(session).enqueue(
            job_type=JobType.transcribe_audio,
            payload_json={"audio_file_id": str(audio_id)},
            max_attempts=3,
        )
        job_id = job.id
        session.commit()
        claimed = JobRepository(session).claim_due()
        assert claimed is not None and claimed.id == job_id
        session.commit()

    # Execute the handler directly and commit; the job stays "running" (crash).
    handler = handlers.get_handler(JobType.transcribe_audio)
    assert handler is not None
    with session_factory() as session:
        handler(session, JobRepository(session).get(job_id))
        session.commit()

    # Backdate the abandoned running job and let the worker recover + replay it.
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None and stored.status is JobStatus.running
        stored.updated_at = utc_now() - timedelta(hours=1)
        session.commit()

    assert process_due_jobs(session_factory, worker_id="test-worker", reset_stale=True) == 1

    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.succeeded
        assert job.attempts == 2
        succeeded = [
            row
            for row in TranscriptionRepository(session).list_by_audio_file(audio_id)
            if row.status is TranscriptionStatus.succeeded
        ]
        assert len(succeeded) == 1  # no duplicate from the replay
    assert fake.calls == 1  # the model ran exactly once


# ---------------------------------------------------------------------------
# provider failure and retries
# ---------------------------------------------------------------------------


def test_provider_failure_retries_then_records_single_terminal_failed_row(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    fake = FakeTranscriptionProvider(outputs=[], failures=99)
    _install_fake_transcription(monkeypatch, fake)
    audio_id, _ = _upload_audio(api_client)
    before = _audio_bytes(session_factory, audio_id)
    job_id = _enqueue(session_factory, audio_id, max_attempts=3)

    # Attempt 1 → retrying, no transcription row yet.
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None and stored.status is JobStatus.retrying
        assert stored.attempts == 1
        assert stored.error_summary == "RuntimeError: synthetic provider failure"
    assert _run_rows(session_factory, audio_id) == []

    # Attempt 2 → retrying with backoff, still no row.
    _make_retry_due(session_factory, job_id)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    assert _run_rows(session_factory, audio_id) == []

    # Attempt 3 (terminal) → one failed row, job failed, audio intact.
    _make_retry_due(session_factory, job_id)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as session:
        stored = JobRepository(session).get(job_id)
        assert stored is not None and stored.status is JobStatus.failed
        assert stored.attempts == 3
        assert stored.error_summary == "RuntimeError: synthetic provider failure"
    rows = _run_rows(session_factory, audio_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.status is TranscriptionStatus.failed
    assert row.raw_text == ""
    assert row.metadata_json is not None
    assert row.metadata_json["job_id"] == str(job_id)
    assert row.metadata_json["terminal"] is True
    # Audio is not lost or corrupted; the fake failed all three times.
    assert _audio_bytes(session_factory, audio_id) == before
    assert fake.calls == 3


def test_retranscribe_appends_new_version_and_preserves_history(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    fake = FakeTranscriptionProvider(outputs=["первая версия", "вторая версия"])
    _install_fake_transcription(monkeypatch, fake)
    audio_id, _ = _upload_audio(api_client)

    first = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert first.status_code == 202
    first_job = uuid.UUID(first.json()["id"])
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    # Explicit repeat after a success starts a fresh job (retranscribe).
    second = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert second.status_code == 202
    second_job = uuid.UUID(second.json()["id"])
    assert second_job != first_job
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as session:
        rows = [
            row
            for row in TranscriptionRepository(session).list_by_audio_file(audio_id)
            if row.status is not TranscriptionStatus.not_requested
        ]
        assert [row.status for row in rows] == [
            TranscriptionStatus.succeeded,
            TranscriptionStatus.succeeded,
        ]
        assert rows[0].raw_text == "первая версия"  # history is never erased
        assert rows[1].raw_text == "вторая версия"
        assert rows[0].metadata_json is not None
        assert rows[1].metadata_json is not None
        assert rows[0].metadata_json["job_id"] == str(first_job)
        assert rows[1].metadata_json["job_id"] == str(second_job)
    assert fake.calls == 2


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------


def test_transcribe_api_returns_existing_active_job_without_duplicate(
    api_client: TestClient,
    session_factory,
) -> None:
    audio_id, _ = _upload_audio(api_client)
    first = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    second = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    with session_factory() as session:
        jobs = JobRepository(session).list_active_of_type(JobType.transcribe_audio)
        assert len(jobs) == 1


def test_transcribe_api_unknown_audio_returns_404(api_client: TestClient) -> None:
    response = api_client.post(f"{AUDIO_URL}/{uuid.uuid4()}/transcribe")
    assert response.status_code == 404
    assert response.json() == {"detail": "Audio file not found."}


def test_transcription_history_api(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    # Empty history before any run (the not_requested sentinel is not listed).
    audio_id, _ = _upload_audio(api_client)
    empty = api_client.get(f"{AUDIO_URL}/{audio_id}/transcriptions")
    assert empty.status_code == 200
    assert empty.json() == []

    fake = FakeTranscriptionProvider(outputs=["история версия один"])
    _install_fake_transcription(monkeypatch, fake)
    job_id = _enqueue(session_factory, audio_id)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    history = api_client.get(f"{AUDIO_URL}/{audio_id}/transcriptions")
    assert history.status_code == 200
    items = history.json()
    assert len(items) == 1
    item = items[0]
    assert set(item) == {
        "id",
        "audio_file_id",
        "status",
        "raw_text",
        "profile_name",
        "provider_name",
        "model_name",
        "language",
        "created_at",
    }
    assert item["audio_file_id"] == str(audio_id)
    assert item["status"] == "succeeded"
    assert item["raw_text"] == "история версия один"
    assert item["model_name"] == "tiny"
    assert item["created_at"].endswith("+03:00")  # default Europe/Moscow
    # The job id is provenance, not part of the public surface.
    assert str(job_id) not in history.text

    # A failed run is listed too, with an empty raw_text.
    failing = FakeTranscriptionProvider(outputs=[], failures=99)
    _install_fake_transcription(monkeypatch, failing)
    failed_job = _enqueue(session_factory, audio_id, max_attempts=1)
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    history = api_client.get(f"{AUDIO_URL}/{audio_id}/transcriptions")
    items = history.json()
    assert [item["status"] for item in items] == ["succeeded", "failed"]
    assert items[1]["raw_text"] == ""
    assert str(failed_job) not in history.text


def test_transcription_history_unknown_audio_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{AUDIO_URL}/{uuid.uuid4()}/transcriptions")
    assert response.status_code == 404
    assert response.json() == {"detail": "Audio file not found."}


# ---------------------------------------------------------------------------
# chat-page rendering
# ---------------------------------------------------------------------------


def test_chat_page_renders_transcribe_button_and_raw_after_success(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    fake = FakeTranscriptionProvider(outputs=["Первая <b>версия</b> записи."])
    _install_fake_transcription(monkeypatch, fake)
    audio_id, message_id = _upload_audio(api_client)
    assert api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe").status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    html = api_client.get("/").text
    assert f'id="message-{message_id}"' in html
    assert f'id="transcription-{audio_id}"' in html
    assert "message__transcription" in html
    assert "Расшифровано" in html
    assert "Перерасшифровать" in html
    # Raw text is rendered server-side and escaped.
    assert "Первая &lt;b&gt;версия&lt;/b&gt; записи." in html
    assert "<b>версия</b>" not in html
    assert "Не расшифровано" not in html


def test_chat_page_renders_queued_state_while_job_is_pending(
    api_client: TestClient,
) -> None:
    audio_id, _ = _upload_audio(api_client)
    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    job_id = response.json()["id"]

    html = api_client.get("/").text
    assert "Ожидает расшифровки" in html
    assert f'data-active-job-id="{job_id}"' in html
    # The busy button is disabled; the client resumes polling via the job id.
    assert "Расшифровка…" in html


def test_chat_page_renders_failed_state_after_terminal_failure(
    session_factory,
    api_client,
    monkeypatch,
) -> None:
    fake = FakeTranscriptionProvider(outputs=[], failures=99)
    _install_fake_transcription(monkeypatch, fake)
    audio_id, _ = _upload_audio(api_client)
    response = api_client.post(f"{AUDIO_URL}/{audio_id}/transcribe")
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["id"])
    _exhaust_job(session_factory, job_id)  # default policy: 3 attempts → failed

    html = api_client.get("/").text
    assert "Ошибка расшифровки" in html
    assert "Расшифровать" in html  # the button invites an explicit retry
    # Nothing presents this message as transcribed. The day page also carries
    # the processing panel (T16c), which counts the failure in words
    # ("Расшифровано 0 из 1 · с ошибкой 1"), so the guard is asserted on the
    # message's own state rather than on the bare word.
    assert 'data-transcription-state="succeeded"' not in html
    assert 'data-role="status-text">Расшифровано<' not in html


# ---------------------------------------------------------------------------
# no accidental behaviour regressions
# ---------------------------------------------------------------------------


def test_importing_stt_modules_does_not_require_faster_whisper() -> None:
    """The STT domain/worker modules must import without the optional
    faster-whisper package (it is imported lazily inside the adapter).

    Absence is *simulated* in a fresh interpreter instead of being assumed
    from the environment: the package is genuinely present as soon as the
    ``stt`` extra is installed, and the invariant has to hold either way.
    """
    probe = """
import sys

sys.modules["faster_whisper"] = None  # forces ImportError on import
try:
    import faster_whisper
except ImportError:
    pass
else:
    raise SystemExit("faster-whisper was importable: absence was not simulated")

import diary.domain.transcription
import diary.workers.handlers
import diary.workers.runner
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_no_automatic_transcription_on_upload(session_factory, api_client) -> None:
    """Uploading audio never creates a transcription job (Task 07 rule)."""
    _upload_audio(api_client)
    assert _job_count(session_factory) == 0
