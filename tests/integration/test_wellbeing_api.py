"""Integration tests for the wellbeing API, download route and UI (Task 15).

Runs against an isolated, migrated SQLite test database and a temporary data
root with synthetic data only.

Covers:

* ``POST /api/v1/pipelines/wellbeing/runs`` — requires an explicit
  ``confirm: true`` (409 without it, 422 when the flag is missing), is
  rejected when the feature is disabled (403) or when the export has no
  normalized (layer 2) input (404), and enqueues a ``wellbeing`` job when all
  guards pass (202 → worker produces a layer-4 report);
* no-autostart: no wellbeing job/artifact is ever created without an explicit
  POST, even when pages are visited;
* ``GET /exports/{export_id}/wellbeing`` — read-only report page: escaped
  rendering, empty states, hidden run action when disabled, fixed escalation
  block + minimal metadata when the deterministic page-level classification
  is ``urgent`` (no diary text in that block), 404 for unknown exports;
* the exports list page links to the wellbeing page.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest
from _pipeline_fakes import install_fake_llm
from _wellbeing_fakes import FakeWellbeingLLM
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import ArtifactLayer, Job, MessageKind, WeeklyExport
from diary.db.repositories import ArtifactRepository, MessageRepository
from diary.domain.exports.service import export_week
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename
from diary.domain.pipelines.wellbeing.registration import (
    ensure_wellbeing_pipeline_registered,
)
from diary.domain.pipelines.wellbeing.safety import DISCLAIMER, ESCALATION_TITLE
from diary.main import app
from diary.workers.runner import process_due_jobs

PIPELINES_URL = "/api/v1/pipelines"
EXPORTS_URL = "/api/v1/exports"
SUNDAY = date(2026, 9, 13)
DIARY_PHRASE = "Синтетика: приватный текст для wellbeing API."
URGENT_PHRASE = "не хочу больше жить"
_REPORT_HEADER = "# Wellbeing-рефлексия\n"
_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_MAX_MESSAGE_LENGTH",
    "APP_MAX_AUDIO_MB",
    "APP_ALLOWED_AUDIO_MIME_TYPES",
    "APP_DATA_ROOT",
    "APP_AUDIO_DIR",
    "APP_DB_DIR",
    "APP_TEMP_DIR",
    "APP_NORMALIZE_CHUNK_ENTRIES",
    "APP_SUMMARIZE_MAPREDUCE_THRESHOLD",
    "APP_SUMMARIZE_CHUNK_SIZE",
    "APP_WELLBEING_ENABLED",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    monkeypatch.setenv("APP_WELLBEING_ENABLED", "true")
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
    with _client_bound_to(session_factory) as client:
        yield client


@pytest.fixture(autouse=True)
def _restore_pipeline_registry() -> None:
    """Restore the baseline pipeline registry after every test."""
    snapshot = dict(pipeline_registry.REGISTRY)
    yield
    pipeline_registry.REGISTRY.clear()
    pipeline_registry.REGISTRY.update(snapshot)


def _message_time(sunday: date) -> datetime:
    monday = sunday - timedelta(days=6)
    return datetime(monday.year, monday.month, monday.day, 9, 0, tzinfo=timezone.utc)


def _create_export(session: Session, sunday: date, body: str) -> WeeklyExport:
    MessageRepository(session).create(
        kind=MessageKind.text, body_text=body, created_at=_message_time(sunday)
    )
    session.commit()
    return export_week(session, sunday_date=sunday)


def _normalized_text(body: str) -> str:
    return (
        "# Дневник — нормализованный\n"
        "Дата воскресенья: 2026-09-13\n"
        "Номер выгрузки: 0001\n"
        "---\n"
        "[2026-09-07 09:00] TEXT\n"
        f"{body}\n"
    )


def _add_normalized(session: Session, export: WeeklyExport, body: str) -> None:
    """Write a real normalized (layer 2) artifact row + file."""
    filename = derived_filename(
        sunday=export.week_end_date,
        layer=ArtifactLayer.normalized,
        seq_number=export.seq_number,
        revision=export.revision,
        version=1,
    )
    relative = artifact_data_root_relative_path(layer=ArtifactLayer.normalized, filename=filename)
    target = get_settings().data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_normalized_text(body), encoding="utf-8")
    ArtifactRepository(session).create(
        layer=ArtifactLayer.normalized,
        export_id=export.id,
        path=relative,
        sha256=hashlib.sha256(_normalized_text(body).encode("utf-8")).hexdigest(),
        version=1,
    )
    session.commit()


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


def _post_run(api_client: TestClient, export_id: uuid.UUID, *, confirm: bool = True):
    return api_client.post(
        f"{PIPELINES_URL}/wellbeing/runs",
        json={"export_id": str(export_id), "confirm": confirm},
    )


# ---------------------------------------------------------------------------
# enqueueing guards
# ---------------------------------------------------------------------------


def test_without_confirm_flag_is_rejected_with_409(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись A.")
    _add_normalized(session, export, "Спокойная неделя.")
    response = _post_run(api_client, export.id, confirm=False)
    assert response.status_code == 409
    assert _job_count(session) == 0  # no job is created without confirmation


def test_missing_confirm_field_is_rejected_with_422(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись A.")
    response = api_client.post(
        f"{PIPELINES_URL}/wellbeing/runs", json={"export_id": str(export.id)}
    )
    assert response.status_code == 422
    assert _job_count(session) == 0


def test_confirmed_run_without_normalized_input_is_404(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись без нормализации.")
    response = _post_run(api_client, export.id)
    assert response.status_code == 404
    assert _job_count(session) == 0


def test_confirmed_run_for_unknown_export_is_404(api_client: TestClient) -> None:
    response = _post_run(api_client, uuid.uuid4())
    assert response.status_code == 404
    assert "Export not found." in response.text


def test_confirmed_run_with_missing_original_file_is_409(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись A.")
    _add_normalized(session, export, "Спокойная неделя.")
    original_path = get_settings().data_root / export.path
    original_path.unlink()  # simulate a lost original file
    response = _post_run(api_client, export.id)
    assert response.status_code == 409
    assert _job_count(session) == 0


def test_disabled_feature_rejects_the_run_with_403(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_WELLBEING_ENABLED", "false")
    reload_settings()
    export = _create_export(session, SUNDAY, "Запись A.")
    _add_normalized(session, export, "Спокойная неделя.")
    response = _post_run(api_client, export.id)
    assert response.status_code == 403
    assert _job_count(session) == 0


def test_confirmed_run_end_to_end_produces_a_report(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для рефлексии через API.")
    export_id = export.id
    _add_normalized(session, export, "Спокойная неделя, много гуляла.")
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True

    created = _post_run(api_client, export_id)
    assert created.status_code == 202
    body = created.json()
    assert body["job_type"] == "wellbeing"
    assert body["status"] == "pending"
    assert DIARY_PHRASE not in created.text  # no diary content in responses

    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as s:
        reports = ArtifactRepository(s).list_by_export_and_layer(export_id, ArtifactLayer.wellbeing)
        assert len(reports) == 1
        assert reports[0].version == 1
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# no automatic wellbeing runs
# ---------------------------------------------------------------------------


def test_nothing_runs_wellbeing_without_an_explicit_post(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visiting pages and even running normalization must never create a
    wellbeing job or artifact (AGENTS.md §16: never automatic)."""
    export = _create_export(session, SUNDAY, "Запись для проверки отсутствия автозапуска.")
    _add_normalized(session, export, "Обычный день.")

    assert api_client.get("/exports").status_code == 200
    assert api_client.get(f"/exports/{export.id}/wellbeing").status_code == 200
    assert _job_count(session) == 0
    with session_factory() as s:
        reports = ArtifactRepository(s).list_by_export_and_layer(export.id, ArtifactLayer.wellbeing)
        assert reports == []
    assert process_due_jobs(session_factory, worker_id="test-worker") == 0
    assert _job_count(session) == 0


# ---------------------------------------------------------------------------
# wellbeing page
# ---------------------------------------------------------------------------


def test_wellbeing_page_renders_report_escaped_and_lists_download(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для страницы.")
    export_id = export.id
    _add_normalized(session, export, "Спокойная неделя.")
    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True
    assert _post_run(api_client, export_id).status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    response = api_client.get(f"/exports/{export_id}/wellbeing")
    assert response.status_code == 200
    html = response.text
    assert "Wellbeing-рефлексия" in html
    assert _REPORT_HEADER in html  # markdown is shown escaped inside <pre>
    assert DISCLAIMER in html
    artifact = ArtifactRepository(session).list_by_export_and_layer(
        export_id, ArtifactLayer.wellbeing
    )[0]
    assert f"{EXPORTS_URL}/{export_id}/artifacts/{artifact.id}/file" in html
    assert "Скачать рефлексию" in html


def test_wellbeing_page_shows_run_action_when_normalized_but_no_report(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для действия.")
    _add_normalized(session, export, "Спокойная неделя.")
    html = api_client.get(f"/exports/{export.id}/wellbeing").text
    assert "wellbeing-run-form" in html
    assert "Запустить wellbeing-рефлексию" in html
    assert "психологический диагноз" in html  # the confirmation wording is shown


def test_wellbeing_page_hides_run_action_when_disabled(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_WELLBEING_ENABLED", "false")
    reload_settings()
    export = _create_export(session, SUNDAY, "Запись.")
    _add_normalized(session, export, "Спокойная неделя.")
    html = api_client.get(f"/exports/{export.id}/wellbeing").text
    assert "wellbeing-run-form" not in html
    assert "Wellbeing-анализ отключён" in html


def test_wellbeing_page_empty_state_without_normalized_version(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Только оригинал.")
    html = api_client.get(f"/exports/{export.id}/wellbeing").text
    text = " ".join(html.split())
    assert "сначала нужна нормализация (слой 2)" in text
    assert "wellbeing-run-form" not in html


def test_wellbeing_page_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"/exports/{uuid.uuid4()}/wellbeing")
    assert response.status_code == 404


def test_exports_list_links_to_the_wellbeing_page(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для ссылки.")
    html = api_client.get("/exports").text
    assert f'href="/exports/{export.id}/wellbeing"' in html
    assert "Wellbeing-рефлексия" in html


# ---------------------------------------------------------------------------
# deterministic urgent fallback on the page
# ---------------------------------------------------------------------------


def test_urgent_input_renders_fixed_escalation_without_diary_text(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An urgent normalized input shows the FIXED escalation guidance and safe
    minimal metadata on the page even before any LLM report exists — the
    fallback never depends on LLM generation and never leaks the diary text."""
    export = _create_export(session, SUNDAY, f"Сегодня думала, что {URGENT_PHRASE}.")
    _add_normalized(session, export, f"Сегодня думала, что {URGENT_PHRASE}.")
    html = api_client.get(f"/exports/{export.id}/wellbeing").text
    assert ESCALATION_TITLE in html
    assert "в службу экстренной помощи по месту жительства" in html
    assert "Уровень: Требуется внимание" in html
    assert str(export.id) in html  # minimal metadata: level, export id, time
    # The escalation block/metadata are deterministic and carry no diary text.
    assert URGENT_PHRASE not in html
    # The run action stays available for the deliberate user action.
    assert "wellbeing-run-form" in html
