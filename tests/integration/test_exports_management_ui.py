"""Integration tests for the exports-management pages (Task 16).

Uses an isolated, migrated SQLite test database and a temporary data root
with synthetic data only.

Covers:

* the weekly exports management page: weeks grouped with revisions, derived
  layer statuses, the export-week form, download/view/detail links and the
  normalize/summarize/wellbeing/re-export action forms (with their disabled
  gating);
* the per-export detail page: lineage sections across all layers with parent
  captions, artifact versions/sha/downloads, the runs table, sibling
  revisions and the run actions;
* wellbeing gating (disabled → action forms hidden and a notice shown) on
  both pages;
* external-provider consent disclosure: with an ``external: true`` profile
  the warning + checkbox are rendered; without one they are absent; a missing
  models configuration never breaks page rendering;
* HTML escaping of stored error summaries on the detail page;
* the full four-layer flow (text → export → normalize → summarize →
  wellbeing) driven through the exact API endpoints the page buttons use, run
  with fake LLM providers, and visible afterwards on the detail page.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from _normalize_fakes import FakeNormalizeLLM
from _pipeline_fakes import install_fake_llm
from _summarize_fakes import FakeSummarizeLLM
from _wellbeing_fakes import FakeWellbeingLLM
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import diary.api.routes.pages as pages_module
from diary.api.routes.messages import get_session
from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
)
from diary.core.settings import get_settings, reload_settings
from diary.db.models import (
    Artifact,
    ArtifactLayer,
    Job,
    MessageKind,
    PipelineType,
    RunStatus,
    WeeklyExport,
)
from diary.db.repositories import (
    ArtifactRepository,
    MessageRepository,
    ProcessingRunRepository,
)
from diary.domain.exports.service import export_filename, export_week
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.naming import artifact_data_root_relative_path, derived_filename
from diary.domain.pipelines.normalize.registration import (
    ensure_normalize_pipeline_registered,
)
from diary.domain.pipelines.summarize.registration import ensure_summarize_pipeline_registered
from diary.domain.pipelines.wellbeing.registration import (
    ensure_wellbeing_pipeline_registered,
)
from diary.main import app
from diary.workers.runner import process_due_jobs

#: Repository root: tests/integration/... -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

SUNDAY = date(2026, 9, 13)
SUNDAY2 = date(2026, 9, 20)
EXPORTS_URL = "/api/v1/exports"
PIPELINES_URL = "/api/v1/pipelines"
SCRIPT_SUMMARY = '<script>alert("xss")</script>'
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
def _restore_pipeline_registry() -> Iterator[None]:
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


def _artifact_filename(export: WeeklyExport, layer: ArtifactLayer, version: int) -> str:
    return derived_filename(
        sunday=export.week_end_date,
        layer=layer,
        seq_number=export.seq_number,
        revision=export.revision,
        version=version,
    )


def _add_artifact(
    session: Session,
    export: WeeklyExport,
    layer: ArtifactLayer,
    version: int,
    text: str,
    *,
    run_id: uuid.UUID | None = None,
) -> Artifact:
    """Write a real artifact row + file of ``layer`` for ``export`` (synthetic)."""
    filename = _artifact_filename(export, layer, version)
    relative = artifact_data_root_relative_path(layer=layer, filename=filename)
    target = get_settings().data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    artifact = ArtifactRepository(session).create(
        layer=layer,
        export_id=export.id,
        path=relative,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=version,
        processing_run_id=run_id,
    )
    session.commit()
    return artifact


def _add_run(
    session: Session,
    export: WeeklyExport,
    pipeline_type: PipelineType,
    *,
    status: RunStatus = RunStatus.succeeded,
    input_artifact_id: uuid.UUID | None = None,
    error_summary: str | None = None,
):
    run = ProcessingRunRepository(session).create(
        pipeline_type=pipeline_type,
        status=status,
        input_export_id=export.id,
        input_artifact_id=input_artifact_id,
        error_summary=error_summary,
    )
    session.commit()
    return run


def _llm_profile(model: str, *, external: bool) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:9/v1",  # never contacted by these tests
        model=model,
        api_key_env="FAKE_EXTERNAL_KEY",
        timeout_s=5.0,
        temperature=0.0,
        max_tokens=64,
        external=external,
    )


def _models_config(*, external: bool) -> ModelsConfig:
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=_llm_profile("norm-model", external=external),
        summarization=_llm_profile("summ-model", external=external),
        wellbeing_analysis=_llm_profile("wb-model", external=external),
    )


# ---------------------------------------------------------------------------
# exports management page
# ---------------------------------------------------------------------------


def test_exports_page_groups_weeks_and_lists_revisions(
    api_client: TestClient,
    session: Session,
) -> None:
    export1 = _create_export(session, SUNDAY, "Запись первой недели.")
    _add_artifact(session, export1, ArtifactLayer.normalized, 1, "Нормализованная запись.")
    export2 = _create_export(session, SUNDAY2, "Запись второй недели.")

    html = api_client.get("/exports").text
    assert "Недельные выгрузки" in html
    for export in (export1, export2):
        assert f'href="/exports/{export.id}"' in html  # detail links
        assert f'href="{EXPORTS_URL}/{export.id}/file"' in html  # original download
    # Derived statuses: normalized v1 present for export1, "нет" for others.
    assert "Нормализация: v1" in " ".join(html.split())
    # The export-week form and its default Sunday (current week) are present.
    assert 'name="sunday"' in html
    assert re.search(r'value="\d{4}-\d{2}-\d{2}"', html) is not None
    assert ">Выгрузить неделю</button>" in html or "Выгрузить неделю" in html
    # Re-export actions exist for each week, keyed by the week's Sunday.
    assert f"{EXPORTS_URL}/weeks/{SUNDAY.isoformat()}/re-export" in html
    assert f"{EXPORTS_URL}/weeks/{SUNDAY2.isoformat()}/re-export" in html


def test_exports_page_renders_run_action_forms_with_gating(
    api_client: TestClient,
    session: Session,
) -> None:
    normalized = _create_export(session, SUNDAY, "Уже нормализованная неделя.")
    _add_artifact(session, normalized, ArtifactLayer.normalized, 1, "Норм.")
    plain = _create_export(session, SUNDAY2, "Ещё не нормализованная неделя.")

    html = api_client.get("/exports").text
    # The action forms target the exact pipeline endpoints the buttons use.
    assert 'data-action="normalize"' in html
    assert f'data-url="{PIPELINES_URL}/normalize/runs"' in html
    assert f'data-url="{PIPELINES_URL}/summarize/runs"' in html
    assert f'data-url="{PIPELINES_URL}/wellbeing/runs"' in html
    assert 'data-export-id="' + str(normalized.id) + '"' in html
    assert 'data-export-id="' + str(plain.id) + '"' in html
    # Wellbeing runs need the explicit disclaimer confirmation.
    assert 'data-role="confirm-wellbeing"' in html
    # Gating hints: summarize/wellbeing of a not-yet-normalized export are
    # disabled with a reason.
    assert "Нужна нормализация (слой 2)." in html


def test_exports_page_reexport_and_links_stay_on_detail(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для проверки ссылок.")
    _add_artifact(session, export, ArtifactLayer.normalized, 1, "Норм.")
    _add_artifact(session, export, ArtifactLayer.summary, 1, "# Саммари\n")
    _add_artifact(session, export, ArtifactLayer.wellbeing, 1, "# Wellbeing\n")

    html = api_client.get("/exports").text
    assert f'href="/exports/{export.id}/normalize"' in html
    assert f'href="/exports/{export.id}/summary"' in html
    assert f'href="/exports/{export.id}/wellbeing"' in html
    assert "Детали и происхождение" in html
    assert "Скачать выгрузку" in html


def test_exports_page_empty_state(api_client: TestClient) -> None:
    html = api_client.get("/exports").text
    assert "Выгрузок пока нет" in html


# ---------------------------------------------------------------------------
# wellbeing disabled on the management pages
# ---------------------------------------------------------------------------


def test_exports_pages_hide_wellbeing_when_disabled(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_WELLBEING_ENABLED", "false")
    reload_settings()
    export = _create_export(session, SUNDAY, "Запись.")
    _add_artifact(session, export, ArtifactLayer.normalized, 1, "Норм.")

    list_html = api_client.get("/exports").text
    assert "Wellbeing-анализ отключён" in list_html
    assert 'data-action="wellbeing"' not in list_html
    assert 'data-action="normalize"' in list_html  # other actions stay available

    detail_html = api_client.get(f"/exports/{export.id}").text
    assert "Wellbeing-анализ отключён" in detail_html
    assert 'data-action="wellbeing"' not in detail_html


# ---------------------------------------------------------------------------
# external-provider consent disclosure
# ---------------------------------------------------------------------------


def test_exports_pages_show_external_warning_for_external_profile(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session, SUNDAY, "Запись.")
    _add_artifact(session, export, ArtifactLayer.normalized, 1, "Норм.")
    monkeypatch.setattr(
        pages_module,
        "_load_models_config",
        lambda: _models_config(external=True),
    )

    for url in ("/exports", f"/exports/{export.id}"):
        html = api_client.get(url).text
        # The disclosure names the external provider and demands consent.
        assert "будет отправлен внешнему провайдеру" in html
        assert "<code>127.0.0.1:9</code>" in html
        assert 'data-external="1"' in html
        assert 'data-role="confirm-external"' in html
        assert "обработку выполнит внешний провайдер" in html


def test_exports_pages_do_not_claim_external_without_external_profile(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _create_export(session, SUNDAY, "Запись.")
    monkeypatch.setattr(
        pages_module,
        "_load_models_config",
        lambda: _models_config(external=False),
    )
    html = api_client.get(f"/exports/{export.id}").text
    assert "внешнему провайдеру" not in html
    assert 'data-external="1"' not in html
    assert 'data-role="confirm-external"' not in html


def test_exports_pages_tolerate_missing_models_config(
    api_client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing/invalid models configuration never breaks page rendering and
    is treated as the local default (no external warning)."""
    export = _create_export(session, SUNDAY, "Запись.")
    monkeypatch.setattr(
        pages_module,
        "_load_models_config",
        _raise_models_config_error,
    )
    list_html = api_client.get("/exports").text
    detail_html = api_client.get(f"/exports/{export.id}").text
    assert "внешнему провайдеру" not in list_html + detail_html
    assert 'data-external="1"' not in list_html + detail_html
    filename = export_filename(export.week_end_date, export.seq_number, export.revision)
    assert filename in detail_html


def _raise_models_config_error():
    """Simulate an unreadable models configuration (see test above)."""
    from diary.core.models_config import ModelsConfigError  # noqa: PLC0415

    raise ModelsConfigError("model configuration file not found")


# ---------------------------------------------------------------------------
# per-export detail page
# ---------------------------------------------------------------------------


def test_export_detail_page_shows_lineage_artifacts_and_runs(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись для деталей.")
    normalized = _add_artifact(session, export, ArtifactLayer.normalized, 1, "Норм. текст")
    run_norm = _add_run(session, export, PipelineType.normalize, input_artifact_id=None)
    normalized.processing_run_id = run_norm.id
    session.commit()
    run_summ = _add_run(session, export, PipelineType.summarize, input_artifact_id=normalized.id)
    summary = _add_artifact(
        session, export, ArtifactLayer.summary, 1, "# Саммари\n", run_id=run_summ.id
    )
    run_wb = _add_run(session, export, PipelineType.wellbeing, input_artifact_id=normalized.id)
    wellbeing = _add_artifact(
        session, export, ArtifactLayer.wellbeing, 1, "# Wellbeing\n", run_id=run_wb.id
    )

    html = api_client.get(f"/exports/{export.id}").text
    assert "Детали выгрузки" in html
    # Lineage sections with all layers.
    assert "Оригинал (слой 1)" in html
    assert "Нормализованный (слой 2)" in html
    assert "Саммари (слой 3)" in html
    assert "Wellbeing-рефлексия (слой 4)" in html
    # Parent captions make the original→normalized→summary/wellbeing chain clear.
    assert "производный от оригинала 0001_r1" in html
    norm_filename = _artifact_filename(export, ArtifactLayer.normalized, 1)
    assert f"создан из нормализованного {norm_filename}" in html
    # Every artifact lists version/sha/download.
    for artifact in (normalized, summary, wellbeing):
        assert artifact.path.rsplit("/", 1)[-1] in html
        assert f"{EXPORTS_URL}/{export.id}/artifacts/{artifact.id}/file" in html
    assert "sha256" in html
    # Runs table rows.
    assert "Прогоны обработки" in html
    assert "Нормализация" in html
    assert "Саммаризация" in html
    assert "Wellbeing-рефлексия" in html
    assert "Успешно" in html
    # Run actions (with wellbeing confirmation) and downloads are present.
    assert 'data-action="normalize"' in html
    assert 'data-action="wellbeing"' in html
    assert 'data-role="confirm-wellbeing"' in html
    assert f"{EXPORTS_URL}/{export.id}/file" in html


def test_export_detail_page_lists_sibling_revisions(
    api_client: TestClient,
    session: Session,
) -> None:
    r1 = _create_export(session, SUNDAY, "Запись для ревизий.")
    r2 = export_week(session, sunday_date=SUNDAY, re_export=True)

    html = api_client.get(f"/exports/{r1.id}").text
    assert "Ревизии недели:" in html
    assert f'href="/exports/{r2.id}">r2<' in html
    assert f'href="/exports/{r1.id}">r1<' not in html  # current revision is unlinked
    assert ">r1</strong>" in html


def test_export_detail_page_escapes_run_error_summaries(
    api_client: TestClient,
    session: Session,
) -> None:
    export = _create_export(session, SUNDAY, "Запись.")
    _add_run(
        session,
        export,
        PipelineType.normalize,
        status=RunStatus.failed,
        error_summary=SCRIPT_SUMMARY,
    )
    html = api_client.get(f"/exports/{export.id}").text
    assert "&lt;script&gt;" in html
    assert "alert(&#34;xss&#34;)" in html
    assert SCRIPT_SUMMARY not in html


def test_export_detail_unknown_id_is_404(api_client: TestClient) -> None:
    response = api_client.get(f"/exports/{uuid.uuid4()}")
    assert response.status_code == 404


def test_export_detail_malformed_id_is_422(api_client: TestClient) -> None:
    response = api_client.get("/exports/not-a-uuid")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# full four-layer flow through the exact endpoints the buttons drive
# ---------------------------------------------------------------------------


def test_full_four_layer_flow_via_pipeline_endpoints_is_visible_on_detail_page(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text → export → normalize → summarize → wellbeing, each step enqueued
    with the exact request body its page button produces (exports.js) and
    executed by a fake LLM provider; afterwards the export detail page shows
    all layers. The wellbeing step mirrors the fixed exports.js path: the
    acknowledgement checkbox is checked, so the body carries ``confirm: true``
    — without it the server would answer 409 (see the JS-contract tests)."""
    export = _create_export(session, SUNDAY, "Запись для полного четырёхслойного флоу.")
    export_id = export.id

    # Layer 2 — normalize (button body: {export_id} only).
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as check:
        normalized = ArtifactRepository(check).list_by_export_and_layer(
            export_id, ArtifactLayer.normalized
        )
        assert len(normalized) == 1

    # Layer 3 — summarize over the normalized text (button body: {export_id}).
    install_fake_llm(monkeypatch, FakeSummarizeLLM())
    assert ensure_summarize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/summarize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    # Layer 4 — wellbeing over the normalized text. The button body is
    # {export_id, confirm: true} (checkbox checked, see _wellbeing_body_like_exports_js).
    install_fake_llm(monkeypatch, FakeWellbeingLLM())
    assert ensure_wellbeing_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/wellbeing/runs",
        json=_wellbeing_body_like_exports_js(export_id, confirmed=True),
    )
    assert response.status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    with session_factory() as check:
        for layer in (
            ArtifactLayer.normalized,
            ArtifactLayer.summary,
            ArtifactLayer.wellbeing,
        ):
            assert len(ArtifactRepository(check).list_by_export_and_layer(export_id, layer)) == 1

    html = api_client.get(f"/exports/{export_id}").text
    assert "производный от оригинала 0001_r1" in html
    normalized_filename = _artifact_filename(export, ArtifactLayer.normalized, 1)
    assert f"создан из нормализованного {normalized_filename}" in html
    for label in ("Нормализация", "Саммаризация", "Wellbeing-рефлексия"):
        assert label in html


# ---------------------------------------------------------------------------
# wellbeing JS/API contract (Task 16 r1 rework)
# ---------------------------------------------------------------------------


def _wellbeing_body_like_exports_js(export_id: uuid.UUID, *, confirmed: bool) -> dict:
    """Build the wellbeing request body exactly as ``exports.js`` does.

    ``startPipeline`` always sends ``{"export_id": ...}``; for wellbeing
    actions it adds ``confirm`` from the acknowledgement checkbox — checked →
    ``true``, unchecked/missing → no ``confirm`` key at all (r1 fix).
    """
    body: dict[str, object] = {"export_id": str(export_id)}
    if confirmed:
        body["confirm"] = True
    return body


def _job_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Job)) or 0


def test_wellbeing_without_confirm_as_old_exports_js_sent_is_409(
    api_client: TestClient,
    session: Session,
) -> None:
    """The r1 defect body ({export_id} only) is rejected with a clear 409 —
    never a 422 schema error — and creates no job."""
    export = _create_export(session, SUNDAY, "Запись для JS-контракта.")
    _add_artifact(session, export, ArtifactLayer.normalized, 1, "Норм. текст")
    response = api_client.post(
        f"{PIPELINES_URL}/wellbeing/runs",
        json=_wellbeing_body_like_exports_js(export.id, confirmed=False),
    )
    assert response.status_code == 409
    assert response.json()["detail"]  # non-empty, static error payload
    assert _job_count(session) == 0


def test_wellbeing_browser_flow_with_checked_checkbox_succeeds(
    api_client: TestClient,
    session: Session,
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed browser path (acknowledgement checkbox checked → body carries
    ``confirm: true``) yields 202 and a successful layer-4 run."""
    export = _create_export(session, SUNDAY, "Запись для JS-флоу.")
    export_id = export.id

    # Normalize first (the button sequence is normalize → wellbeing): the
    # wellbeing pipeline consumes only a valid layer-2 document.
    install_fake_llm(monkeypatch, FakeNormalizeLLM())
    assert ensure_normalize_pipeline_registered() is True
    response = api_client.post(
        f"{PIPELINES_URL}/normalize/runs", json={"export_id": str(export_id)}
    )
    assert response.status_code == 202
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1

    provider = FakeWellbeingLLM()
    install_fake_llm(monkeypatch, provider)
    assert ensure_wellbeing_pipeline_registered() is True
    created = api_client.post(
        f"{PIPELINES_URL}/wellbeing/runs",
        json=_wellbeing_body_like_exports_js(export_id, confirmed=True),
    )
    assert created.status_code == 202
    assert created.json()["job_type"] == "wellbeing"
    assert created.json()["status"] == "pending"
    assert process_due_jobs(session_factory, worker_id="test-worker") == 1
    with session_factory() as check:
        reports = ArtifactRepository(check).list_by_export_and_layer(
            export_id, ArtifactLayer.wellbeing
        )
        assert len(reports) == 1
        assert reports[0].version == 1
    assert provider.calls == 1
    # The export management page surfaces the new layer-4 result.
    html = api_client.get(f"/exports/{export_id}").text
    assert "Wellbeing-рефлексия (слой 4)" in html


def test_exports_js_maps_wellbeing_errors_and_sends_confirm_from_checkbox() -> None:
    """Static guards for the r1 fix (browser e2e is Task 19): the client sends
    the ``confirm`` flag read from the acknowledgement checkbox and maps every
    wellbeing server status (403/404/409/422) to a clear, static message."""
    js_path = REPO_ROOT / "src/diary/static/js/exports.js"
    script = js_path.read_text(encoding="utf-8")
    # The checkbox state really feeds the request body.
    assert 'input[data-role="confirm-wellbeing"]' in script
    assert "payload.confirm = " in script
    # Wellbeing error mapping covers the confirmation, disabled and schema cases.
    assert "wellbeing: {" in script
    for code in ('"403"', '"404"', '"409"', '"422"'):
        assert code in script
    assert "Wellbeing-анализ отключён в настройках приложения." in script
    assert "Требуется явное подтверждение запуска." in script
    # No HTML-string sinks in the client code.
    for sink in ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write"):
        assert sink not in script
