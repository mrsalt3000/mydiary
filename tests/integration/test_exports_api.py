"""Integration tests for the weekly-export HTTP API and minimal UI (Task 10).

Runs against an isolated, migrated SQLite test database and a temporary data
root with synthetic data only. Covers: creating an export through the API
(201 + file on disk + consistent row), the duplicate-export 409 with a
re-export hint, explicit re-export (new revision, prior file immutable), the
empty-week 409, the non-Sunday 422, listing without diary content, the
download endpoint, 404s, and the minimal UI page with download links.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.core.settings import get_settings, reload_settings
from diary.db.models import MessageKind, WeeklyExport
from diary.db.repositories import MessageRepository, WeeklyExportRepository
from diary.main import app

SUNDAY = "2026-09-13"
SUNDAY_DATE = date(2026, 9, 13)
EXPORTS_URL = "/api/v1/exports"


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


def _add_message(session: Session, body: str, created_at: datetime) -> None:
    MessageRepository(session).create(kind=MessageKind.text, body_text=body, created_at=created_at)
    session.commit()


def _add_week_message(session: Session, body: str) -> None:
    """A message inside the default exported week (Moscow timezone)."""
    _add_message(session, body, datetime(2026, 9, 7, 6, 0, tzinfo=timezone.utc))  # Mon 09:00 MSK


def _export_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(WeeklyExport)) or 0


def _original_txt_files() -> list:
    original_dir = get_settings().exports_dir / "01_original"
    if not original_dir.exists():
        return []
    return [path for path in original_dir.rglob("*.txt") if path.is_file()]


def _export_path(filename: str):
    return get_settings().exports_dir / "01_original" / filename


# ---------------------------------------------------------------------------
# creation through the API
# ---------------------------------------------------------------------------


def test_post_creates_export_with_file_and_consistent_row(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "API-запись недели.")

    response = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}")
    assert response.status_code == 201
    payload = response.json()
    assert set(payload) == {"id", "sunday", "seq_number", "revision", "filename", "created_at"}
    assert payload["sunday"] == SUNDAY
    assert payload["seq_number"] == 1
    assert payload["revision"] == 1
    assert payload["filename"] == "2026-09-13_original_0001_r1.txt"
    assert payload["created_at"].endswith("+03:00")

    path = _export_path(payload["filename"])
    assert path.is_file()
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# Дневник — оригинал\n")
    assert "API-запись недели." in content
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    row = WeeklyExportRepository(session).get(uuid.UUID(payload["id"]))
    assert row is not None
    assert row.week_end_date == SUNDAY_DATE
    assert row.path == f"exports/01_original/{payload['filename']}"
    assert row.sha256 == digest


def test_post_duplicate_week_returns_409_with_reexport_hint(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "первая запись")
    first = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}")
    assert first.status_code == 201

    duplicate = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}")
    assert duplicate.status_code == 409
    assert "re-export" in duplicate.json()["detail"]
    assert _export_count(session) == 1  # no second row
    assert "первая запись" not in duplicate.text  # no diary content echo


def test_post_re_export_creates_next_revision(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "запись для ревизии")
    first = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}").json()
    r1_path = _export_path(first["filename"])
    r1_bytes = r1_path.read_bytes()

    _add_week_message(session, "вторая запись для ревизии")
    revised = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}/re-export")
    assert revised.status_code == 201
    payload = revised.json()
    assert payload["seq_number"] == first["seq_number"] == 1
    assert payload["revision"] == 2
    assert payload["filename"] == "2026-09-13_original_0001_r2.txt"

    # The prior revision file is byte-identical after the re-export.
    assert r1_path.read_bytes() == r1_bytes
    r2_content = _export_path(payload["filename"]).read_text(encoding="utf-8")
    assert "вторая запись для ревизии" in r2_content


def test_post_re_export_of_never_exported_week_returns_409(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "запись")
    response = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}/re-export")
    assert response.status_code == 409
    assert "first export" in response.json()["detail"]


def test_post_empty_week_returns_409_and_creates_nothing(
    api_client: TestClient,
    session: Session,
) -> None:
    response = api_client.post(f"{EXPORTS_URL}/weeks/2026-09-20")
    assert response.status_code == 409
    assert "no messages" in response.json()["detail"]
    assert _export_count(session) == 0
    assert _original_txt_files() == []


def test_post_non_sunday_date_returns_422(api_client: TestClient) -> None:
    response = api_client.post(f"{EXPORTS_URL}/weeks/2026-09-07")  # a Monday
    assert response.status_code == 422


def test_post_malformed_date_returns_422(api_client: TestClient) -> None:
    response = api_client.post(f"{EXPORTS_URL}/weeks/not-a-date")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# listing and download
# ---------------------------------------------------------------------------


def test_list_orders_by_week_then_revision(api_client: TestClient, session: Session) -> None:
    # Week of 2026-09-13: r1 + r2.
    _add_week_message(session, "запись недели 13-го")
    assert api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}").status_code == 201
    assert api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}/re-export").status_code == 201
    # Week of 2026-09-20: r1 only (messages on its Monday 2026-09-14).
    _add_message(
        session,
        "запись недели 20-го",
        datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc),  # Mon 09:00 MSK
    )
    assert api_client.post(f"{EXPORTS_URL}/weeks/2026-09-20").status_code == 201

    listing = api_client.get(EXPORTS_URL)
    assert listing.status_code == 200
    items = listing.json()
    assert [item["sunday"] for item in items] == ["2026-09-20", "2026-09-13", "2026-09-13"]
    assert [item["revision"] for item in items] == [1, 2, 1]
    # The payload carries metadata only — never the diary text itself.
    for item in items:
        assert set(item) == {"id", "sunday", "seq_number", "revision", "filename", "created_at"}
    assert all("запись" not in item["filename"] for item in items)


def test_download_returns_file_bytes_and_filename(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "скачиваемая запись")
    created = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}")
    export_id = created.json()["id"]

    downloaded = api_client.get(f"{EXPORTS_URL}/{export_id}/file")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("text/plain")
    assert "attachment" in downloaded.headers.get("content-disposition", "")
    assert "2026-09-13_original_0001_r1.txt" in downloaded.headers["content-disposition"]
    path = _export_path(created.json()["filename"])
    assert downloaded.content == path.read_bytes()


def test_download_unknown_export_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"{EXPORTS_URL}/{uuid.uuid4()}/file")
    assert response.status_code == 404
    assert response.json() == {"detail": "Export not found."}


def test_download_missing_file_returns_404(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "запись")
    created = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}").json()
    _export_path(created["filename"]).unlink()

    response = api_client.get(f"{EXPORTS_URL}/{created['id']}/file")
    assert response.status_code == 404
    assert response.json() == {"detail": "Export file not found."}


def test_malformed_export_id_returns_422(api_client: TestClient) -> None:
    response = api_client.get(f"{EXPORTS_URL}/not-a-uuid/file")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# minimal UI page
# ---------------------------------------------------------------------------


def test_exports_page_lists_exports_with_download_links(
    api_client: TestClient,
    session: Session,
) -> None:
    _add_week_message(session, "видимая запись")
    created = api_client.post(f"{EXPORTS_URL}/weeks/{SUNDAY}")
    assert created.status_code == 201
    filename = created.json()["filename"]
    export_id = created.json()["id"]

    page = api_client.get("/exports")
    assert page.status_code == 200
    html = page.text
    assert "Недельные выгрузки" in html
    assert filename in html
    assert f'href="/api/v1/exports/{export_id}/file"' in html
    assert "Скачать выгрузку" in html


def test_exports_page_shows_empty_state(api_client: TestClient) -> None:
    page = api_client.get("/exports")
    assert page.status_code == 200
    assert "Выгрузок пока нет" in page.text


def test_chat_page_links_to_exports_page(api_client: TestClient) -> None:
    page = api_client.get("/")
    assert page.status_code == 200
    assert 'href="/exports"' in page.text
