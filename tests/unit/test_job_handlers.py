"""Unit tests for the job-handler registry (Task 08/09/12).

The registry is a module-level dict shared by the whole test session, so every
test restores the original handlers afterwards (a module fixture snapshots and
reinstates ``REGISTRY``).

Task 09 replaced the Task-08 ``transcribe_audio`` placeholder with the real
STT handler: a job whose payload carries no valid ``audio_file_id`` now fails
permanently with a payload error (not with the old "not implemented yet"
placeholder message).

Task 12 registered a generic handler for the pipeline job types
(``normalize``/``summarize``/``wellbeing``): they are no longer "unknown job
types". A pipeline job whose payload carries no valid ``export_id`` fails
permanently before any database work, so it is unit-testable without a
session.
"""

from __future__ import annotations

import pytest

from diary.db.models import Job, JobType
from diary.workers import handlers


def _noop_handler(session: object, job: object) -> None:
    """A fake handler whose callable shape matches the registry protocol."""


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    snapshot = dict(handlers.REGISTRY)
    yield
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(snapshot)


def test_transcribe_audio_handler_is_registered_by_default() -> None:
    assert handlers.get_handler(JobType.transcribe_audio) is not None
    assert JobType.transcribe_audio.value in handlers.known_job_types()


def test_pipeline_job_type_handlers_are_registered_by_default() -> None:
    """Task 12: normalize/summarize/wellbeing are handled, not unknown types."""
    for job_type in (JobType.normalize, JobType.summarize, JobType.wellbeing):
        assert handlers.get_handler(job_type) is not None
        assert job_type.value in handlers.known_job_types()


def test_phase2_layer_handlers_are_registered_by_default() -> None:
    """Task T11: the three dispatched phase-2 layers are handled.

    Before T11 these job types had no handler, so a phase-2 job enqueued for
    them failed permanently without running the layer.
    """
    for job_type in (JobType.stt_corrected, JobType.daily_summary, JobType.weekly_summary):
        assert handlers.get_handler(job_type) is not None
        assert job_type.value in handlers.known_job_types()


def test_the_coach_note_has_no_handler_by_design() -> None:
    """Layer 5 stays «explicit launch only» (T07 decision 3): no handler.

    A ``coach_note`` job therefore still fails permanently instead of producing
    a coaching note behind the user's back; the note is produced only through
    the explicit call (``run_coach_step`` / the T11 CLI).
    """
    assert handlers.get_handler(JobType.coach_note) is None
    assert JobType.coach_note.value not in handlers.known_job_types()


@pytest.mark.parametrize(
    ("job_type", "anchor_key"),
    [
        (JobType.stt_corrected, "raw_id"),
        (JobType.daily_summary, "local_date"),
        (JobType.weekly_summary, "week_id"),
    ],
)
def test_phase2_handler_fails_permanently_without_its_anchor(
    job_type: JobType, anchor_key: str
) -> None:
    """A phase-2 job whose payload has no anchor is a deterministic failure.

    It is raised before any database or orchestrator work, so the test needs no
    session (the payload check runs ahead of the layer call).
    """
    handler = handlers.get_handler(job_type)
    assert handler is not None
    job = Job(job_type=job_type, payload_json=None)
    with pytest.raises(handlers.PermanentJobError, match=anchor_key):
        handler(session=None, job=job)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("job_type", "payload"),
    [
        (JobType.stt_corrected, {"raw_id": "not-a-uuid"}),
        (JobType.daily_summary, {"local_date": "14.09.2026"}),
    ],
)
def test_phase2_handler_rejects_a_malformed_anchor(
    job_type: JobType, payload: dict[str, str]
) -> None:
    """A malformed anchor never reaches the layer (it can never be retried into
    a success)."""
    handler = handlers.get_handler(job_type)
    assert handler is not None
    job = Job(job_type=job_type, payload_json=payload)
    with pytest.raises(handlers.PermanentJobError, match="no valid"):
        handler(session=None, job=job)  # type: ignore[arg-type]


def test_override_and_get_handler_roundtrip() -> None:
    """Direct registry overrides (used by queue tests) are visible through
    ``get_handler``, including lookup by the stored string form."""
    handlers.REGISTRY[JobType.normalize.value] = _noop_handler  # type: ignore[assignment]
    assert handlers.get_handler(JobType.normalize) is _noop_handler
    assert handlers.get_handler("normalize") is _noop_handler
    assert "normalize" in handlers.known_job_types()


def test_get_handler_returns_none_for_unregistered_type() -> None:
    """Every JobType now has a handler; only unknown strings return ``None``."""
    assert handlers.get_handler("no-such-type") is None


def test_transcribe_handler_fails_permanently_without_valid_payload() -> None:
    """A transcribe job with no payload is a deterministic failure: the real
    handler must map it to PermanentJobError (payload error), never to the
    retired Task-08 "not implemented yet" placeholder message."""
    handler = handlers.get_handler(JobType.transcribe_audio)
    assert handler is not None
    job = Job(job_type=JobType.transcribe_audio, payload_json=None)
    with pytest.raises(handlers.PermanentJobError, match="audio_file_id"):
        handler(session=None, job=job)  # type: ignore[arg-type]


def test_pipeline_handler_fails_permanently_without_valid_payload() -> None:
    """A pipeline job whose payload has no valid ``export_id`` is a permanent
    failure raised before any database work (no session needed)."""
    handler = handlers.get_handler(JobType.normalize)
    assert handler is not None
    job = Job(job_type=JobType.normalize, payload_json=None)
    with pytest.raises(handlers.PermanentJobError, match="export_id"):
        handler(session=None, job=job)  # type: ignore[arg-type]


def test_registering_duplicate_job_type_raises() -> None:
    # ``normalize`` is already registered by the baseline, so ``register``
    # must refuse a second handler for it.
    with pytest.raises(ValueError, match="already registered"):
        handlers.register(JobType.normalize, _noop_handler)
