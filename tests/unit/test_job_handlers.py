"""Unit tests for the job-handler registry (Task 08).

The registry is a module-level dict shared by the whole test session, so every
test restores the original handlers afterwards (a module fixture snapshots and
reinstates ``REGISTRY``).
"""

from __future__ import annotations

import pytest

from diary.db.models import JobType
from diary.workers import handlers


def _noop_handler(session: object, job: object) -> None:
    """A fake handler whose callable shape matches the registry protocol."""


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    snapshot = dict(handlers.REGISTRY)
    yield
    handlers.REGISTRY.clear()
    handlers.REGISTRY.update(snapshot)


def test_transcribe_audio_stub_is_registered_by_default() -> None:
    assert handlers.get_handler(JobType.transcribe_audio) is not None
    assert JobType.transcribe_audio.value in handlers.known_job_types()


def test_register_and_get_handler_roundtrip() -> None:
    handlers.register(JobType.normalize, _noop_handler)
    assert handlers.get_handler(JobType.normalize) is _noop_handler
    # Lookup by the stored string form works too.
    assert handlers.get_handler("normalize") is _noop_handler
    assert "normalize" in handlers.known_job_types()


def test_registering_duplicate_job_type_raises() -> None:
    handlers.register(JobType.normalize, _noop_handler)
    with pytest.raises(ValueError, match="already registered"):
        handlers.register(JobType.normalize, _noop_handler)


def test_get_handler_returns_none_for_unregistered_type() -> None:
    assert handlers.get_handler(JobType.wellbeing) is None
    assert handlers.get_handler("no-such-type") is None


def test_transcribe_stub_fails_permanently_with_task09_message() -> None:
    stub = handlers.get_handler(JobType.transcribe_audio)
    assert stub is not None
    with pytest.raises(handlers.PermanentJobError, match="Task 09"):
        stub(session=None, job=None)  # type: ignore[arg-type]
