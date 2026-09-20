"""Integration tests for the privacy disclosure page (Task 18, GET /privacy).

The page must render for every task a provider/model/host/local-or-external
summary and key/validity state computed from ``profile_statuses`` (Task 03),
never a secret value. A missing models configuration must render an explicit
"not configured" state instead of failing. Navigation to /privacy is verified
on the main templates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import diary.api.routes.pages as pages_module
from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    ModelsConfigError,
    OpenAICompatibleProfile,
)
from diary.core.settings import reload_settings
from diary.main import app

#: Repository root: tests/integration/... -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "src" / "diary" / "templates"

_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_DATA_ROOT",
    "APP_DB_DIR",
    "APP_AUDIO_DIR",
    "APP_EXPORTS_DIR",
    "APP_TEMP_DIR",
    "APP_BACKUPS_DIR",
    "APP_WELLBEING_ENABLED",
    "NORM_API",
    "SUMM_API",
    "WB_API",
)

ALL_TASK_LABELS = (
    "Транскрипция (голос → текст)",
    "Нормализация (слой 2)",
    "Саммаризация (слой 3)",
    "Wellbeing-рефлексия (слой 4)",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the app data root at a per-test directory and isolate the env."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def _llm_profile(
    model: str, *, external: bool, key_env: str, base_url: str
) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url=base_url,
        model=model,
        api_key_env=key_env,
        timeout_s=5.0,
        temperature=0.0,
        max_tokens=64,
        external=external,
    )


def _mixed_config() -> ModelsConfig:
    """One local STT, one local LLM, one external LLM, one missing-key LLM."""
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="small"),
        normalization=_llm_profile(
            "norm-model",
            external=False,
            key_env="NORM_API",
            base_url="http://127.0.0.1:11434/v1",
        ),
        summarization=_llm_profile(
            "summ-model",
            external=True,
            key_env="SUMM_API",
            base_url="https://api.example.com/v1",
        ),
        wellbeing_analysis=_llm_profile(
            "wb-model",
            external=False,
            key_env="WB_API",
            base_url="http://127.0.0.1:1234/v1",
        ),
    )


def _missing_config() -> ModelsConfigError:
    raise ModelsConfigError("model configuration file not found")


def test_privacy_page_renders_without_models_config(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pages_module, "_load_models_config", _missing_config)

    response = api_client.get("/privacy")

    assert response.status_code == 200
    html = response.text
    assert "Приватность" in html
    for label in ALL_TASK_LABELS:
        assert label in html
    assert "не задан или невалиден" in html
    assert "Не настроено" in html
    # Env-var names are only meaningful once a profile exists; with no config
    # the page shows nothing key-related at all.
    assert "api_key_env" not in html


def test_privacy_page_shows_profiles_and_never_key_values(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pages_module, "_load_models_config", _mixed_config)
    monkeypatch.setenv("NORM_API", "norm-secret-123")
    monkeypatch.setenv("WB_API", "wb-secret-456")
    # SUMM_API intentionally left unset → invalid, needs its env var.

    response = api_client.get("/privacy")

    assert response.status_code == 200
    html = response.text
    for label in ALL_TASK_LABELS:
        assert label in html
    # Provider/model/host summary.
    for needle in ("faster-whisper", "norm-model", "summ-model", "wb-model", "small"):
        assert needle in html
    assert "127.0.0.1:11434" in html
    assert "api.example.com" in html
    assert "127.0.0.1:1234" in html
    # Local vs external states.
    assert ">Локально<" in html or "Локально" in html
    assert ">Внешне<" in html
    # Key states: only variable NAMES are shown; values never leak.
    assert "NORM_API" in html
    assert "SUMM_API" in html
    assert "WB_API" in html
    assert "norm-secret-123" not in html
    assert "wb-secret-456" not in html
    # Local STT needs no key; the external/missing-key profile is not ready.
    assert "Не требуется" in html
    assert "Не настроен — нужен" in html
    assert "Готов" in html
    assert "Не готов" in html


def test_privacy_page_shows_local_only_without_external_marker(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _mixed_config()
    config.summarization = _llm_profile(
        "summ-model",
        external=False,
        key_env="SUMM_API",
        base_url="http://127.0.0.1:9999/v1",
    )
    monkeypatch.setattr(pages_module, "_load_models_config", lambda: config)
    monkeypatch.setenv("NORM_API", "x")
    monkeypatch.setenv("SUMM_API", "y")
    monkeypatch.setenv("WB_API", "z")

    html = api_client.get("/privacy").text

    assert ">Внешне<" not in html


def test_privacy_navigation_is_linked_from_main_templates() -> None:
    """The /privacy link is present in the navigation of every main page.

    Since T16a the navigation exists once, in the shared shell (base.html),
    and every main page renders inside that shell — so one check of the shell
    plus "every page extends it" covers all of them.
    """
    shell = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
    for href in ("/privacy", "/exports", "/processing"):
        assert f'href="{href}"' in shell, href
    templates = [
        "chat.html",
        "exports.html",
        "processing.html",
        "export_detail.html",
        "export_compare.html",
        "export_summary.html",
        "export_wellbeing.html",
    ]
    for name in templates:
        source = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
        assert '{% extends "base.html" %}' in source, name
    # The privacy page itself links back through the same shared navigation.
    assert 'href="/exports"' in shell


def test_privacy_page_is_reachable_without_database() -> None:
    # No session fixture involved: settings point at an empty data root and
    # the page still renders (it never touches the database).
    response = TestClient(app).get("/privacy")
    assert response.status_code == 200
    assert "Приватность" in response.text
