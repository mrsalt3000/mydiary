"""Consolidated synthetic fixtures/helpers for the Task-19 e2e tests.

The Task-19 end-to-end scenario and its failure-safety tests share one
synthetic set (AGENTS.md §13, ``docs/privacy-and-safety.md``): text bodies are
made-up Russian phrases that carry a unique marker, audio is generated WAV
bytes with a real container header from ``_audio_samples``, STT uses a
deterministic fake provider and every LLM pipeline reuses the per-pipeline
fakes of this directory (``_normalize_fakes``, ``_summarize_fakes``,
``_wellbeing_fakes``, ``_pipeline_fakes``). No real diary content, network,
model, GPU or API key is ever used.

``tests/integration`` was chosen as the e2e home (instead of a separate
``tests/e2e/`` package) because it reuses the migrated-isolation fixtures of
``conftest.py`` and the existing ``_*_fakes`` without duplicating the Alembic
setup or sys.path plumbing.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from _audio_samples import WAV_MIME, wav_bytes

# Re-exports of the per-pipeline deterministic fakes — the consolidated "mock
# LLM results for all pipelines" set of the Task-19 plan.
from _normalize_fakes import NORM_PREFIX, FakeNormalizeLLM  # noqa: F401  (re-exported)
from _pipeline_fakes import FakeLLMProvider, install_fake_llm  # noqa: F401  (re-exported)
from _summarize_fakes import FakeSummarizeLLM  # noqa: F401  (re-exported)
from _wellbeing_fakes import FakeWellbeingLLM  # noqa: F401  (re-exported)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.models_config import FasterWhisperProfile
from diary.db.base import utc_now
from diary.db.models import JobStatus, Message, MessageKind
from diary.db.repositories import JobRepository, MessageRepository
from diary.domain.transcription import service as transcription_service
from diary.workers.runner import process_due_jobs

#: Prefix every synthetic body so content-leak scans can tell diary-like text
#: apart from static messages.
SYNTHETIC_PREFIX = "Синтетика (e2e): "


def unique_marker(label: str) -> str:
    """A content marker unique to one test run (never real diary text)."""
    return f"{label}-{uuid.uuid4().hex[:12]}"


def synthetic_body(label: str) -> str:
    """A deterministic synthetic text-message body embedding a unique marker."""
    return f"{SYNTHETIC_PREFIX}{unique_marker(label)} запись о прогулке и работе."


def add_synthetic_text(session: Session, body: str, *, created_at: datetime) -> Message:
    """Insert one synthetic text message (Task-05 repository path)."""
    return MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=created_at
    )


@contextmanager
def client_bound_to(session_factory) -> Iterator[TestClient]:
    """A TestClient whose ``get_session`` dependency serves ``session_factory``."""

    def _override_session() -> Iterator[Session]:
        with session_factory() as current_session:
            yield current_session

    from diary.main import app

    app.dependency_overrides[get_session] = _override_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)


def upload_synthetic_wav(api_client: TestClient) -> tuple[uuid.UUID, uuid.UUID]:
    """Upload one generated WAV through the real audio API.

    Returns ``(audio_file_id, message_id)``; asserts the upload succeeded.
    """
    payload = wav_bytes()
    response = api_client.post("/api/v1/audio", files={"file": ("voice.wav", payload, WAV_MIME)})
    assert response.status_code == 201, response.text
    body = response.json()
    return uuid.UUID(body["id"]), uuid.UUID(body["message_id"])


# ---------------------------------------------------------------------------
# Deterministic STT fake (Task-09 protocol) shared by the e2e scenario
# ---------------------------------------------------------------------------


class SyntheticSTTProvider:
    """Fake transcription provider: queued outputs and configurable failures.

    ``before_transcribe`` (optional) runs on every call right before the fake
    produces its outcome — used by the duplicate-claim test to hold one worker
    mid-job while a second worker attempts the claim.
    """

    version = "0.0.0-e2e"

    def __init__(
        self,
        *,
        outputs: list[str] | None = None,
        failures: int = 0,
        before_transcribe: Callable[[], None] | None = None,
    ) -> None:
        self.outputs = list(outputs) if outputs else ["Синтетический текст расшифровки (e2e)."]
        self.failures = failures
        self.before_transcribe = before_transcribe
        self.calls = 0

    def transcribe(self, audio_path, profile: FasterWhisperProfile) -> str:
        self.calls += 1
        if self.before_transcribe is not None:
            self.before_transcribe()
        if self.calls <= self.failures:
            raise RuntimeError("synthetic STT failure")
        index = min(self.calls - 1 - self.failures, len(self.outputs) - 1)
        return self.outputs[index]


def fake_stt_profile() -> FasterWhisperProfile:
    """The fixed faster-whisper profile the fake STT provider is wired as."""
    return FasterWhisperProfile(
        provider="faster_whisper",
        model="tiny",
        language="ru",
        device="cpu",
        compute_type="int8",
        beam_size=1,
    )


def install_fake_stt(monkeypatch: pytest.MonkeyPatch, provider: SyntheticSTTProvider) -> None:
    """Wire the fake STT provider into the transcription service (Task-09 seams)."""
    monkeypatch.setattr(
        transcription_service,
        "_load_transcription_profile",
        lambda: fake_stt_profile(),
    )
    monkeypatch.setattr(transcription_service, "create_provider", lambda profile: provider)


# ---------------------------------------------------------------------------
# Worker driving helpers
# ---------------------------------------------------------------------------


def run_due_jobs(session_factory, *, expected: int, worker_id: str = "e2e-worker") -> int:
    """Run the worker once and assert exactly ``expected`` jobs were handled."""
    processed = process_due_jobs(session_factory, worker_id=worker_id)
    assert processed == expected, f"worker handled {processed} jobs, expected {expected}"
    return processed


def make_retry_due(session_factory, job_id: uuid.UUID) -> None:
    """Force a ``retrying`` job's ``next_retry_at`` into the past."""
    with session_factory() as session:
        job = JobRepository(session).get(job_id)
        assert job is not None and job.status is JobStatus.retrying
        job.next_retry_at = utc_now() - timedelta(seconds=1)
        session.commit()


def enqueue_job(session_factory, *, job_type, payload_json: dict[str, object]) -> uuid.UUID:
    """Enqueue one durable job directly (identifiers-only payload); return its id."""
    with session_factory() as session:
        job = JobRepository(session).enqueue(job_type=job_type, payload_json=payload_json)
        job_id = job.id
        session.commit()
    return job_id
