"""Integration tests for the phase-2 pipeline CLI (task T11).

The CLI is the single entry point of the phase-2 chain on this machine:
``day`` / ``week`` / ``coach`` over the orchestrator, reading the same ``APP_*``
settings as the web application. Every test points the data root at a per-test
temporary directory, creates the database **only** through the migrations, and
drives the commands in-process through :func:`diary.cli.main` with fake
per-layer providers — no endpoint, model, GPU or real configuration file is
touched.

Pinned here: the day/week/coach commands produce (and report) the run of their
layer; ``coach`` is the only command that produces a coaching note; the message
correction has no CLI verb (it is a job, not a chain); a switched-off layer is
an error, an empty-but-legitimate input is not; and the output carries
identifiers/counts only — never diary text.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from _migrate import upgrade_database_file
from sqlalchemy.orm import Session

from diary.cli import main
from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
)
from diary.core.settings import get_settings, reload_settings
from diary.db.models import ArtifactLayer, MessageKind
from diary.db.repositories import MessageRepository
from diary.db.results import list_layer_results
from diary.db.session import create_engine, create_session_factory
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.pipelines import pipeline as phase2_pipeline
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines.coach.registration import ensure_coach_pipeline_registered
from diary.domain.pipelines.stt_fix.pipeline import ENTRY_HEADER
from diary.domain.pipelines.stt_fix.registration import ensure_stt_fix_pipeline_registered
from diary.domain.pipelines.sum_daily.pipeline import ENTRIES_HEADER
from diary.domain.pipelines.sum_daily.registration import ensure_sum_daily_pipeline_registered
from diary.domain.pipelines.sum_weekly.pipeline import SUMMARIES_HEADER
from diary.domain.pipelines.sum_weekly.registration import ensure_sum_weekly_pipeline_registered

#: The synthetic local day (a Monday) and its ISO week.
DAY = date(2026, 9, 14)
WEEK_ID = "2026-W38"
LOCAL_TZ = "Europe/Moscow"

#: A UTC instant inside :data:`DAY` in ``Europe/Moscow`` (07:10 local).
MORNING_UTC = datetime(2026, 9, 14, 4, 10, tzinfo=timezone.utc)

RAW_TEXT = "утрам схадил в паликлинику за спрафкой"
CORRECTED_TEXT = "Утром сходил в поликлинику за справкой."
DAY_ROLLUP = "Неделя прошла вокруг бытовых дел."

_ENV_VARS = (
    "APP_TIMEZONE",
    "APP_DATA_ROOT",
    "APP_DB_DIR",
    "APP_AUDIO_DIR",
    "APP_EXPORTS_DIR",
    "APP_TEMP_DIR",
    "APP_BACKUPS_DIR",
    "MODELS_CONFIG",
)


class FakeLayerProvider:
    """The fake provider of one phase-2 layer (header-checked, deterministic)."""

    def __init__(self, *, header: str, payload: object, model: str) -> None:
        self.header = header
        self.payload = payload
        self.model = model
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        content = request.messages[0].content
        assert self.header in content, f"the request must be shaped like {self.header!r}"
        self.requests.append(request)
        text = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )
        return LLMResponse(
            text=text, model=self.model, usage=LLMUsage(input_tokens=9, output_tokens=5)
        )


def _llm_profile(model: str) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted
        model=model,
        api_key_env="SYNTHETIC_PHASE2_KEY",
        temperature=0.25,
        max_tokens=512,
    )


def _models_config(
    *, stt: bool = True, day: bool = True, week: bool = True, coach: bool = True
) -> ModelsConfig:
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=_llm_profile("fake-normalizer"),
        summarization=_llm_profile("fake-summarizer"),
        wellbeing_analysis=_llm_profile("fake-wellbeing"),
        diary_stt=_llm_profile("fake-stt") if stt else None,
        diary_sum=_llm_profile("fake-day") if day else None,
        diary_week=_llm_profile("fake-week") if week else None,
        diary_coach=_llm_profile("fake-coach") if coach else None,
    )


def _fake_providers() -> dict[str, FakeLayerProvider]:
    return {
        "fake-stt": FakeLayerProvider(
            header=ENTRY_HEADER,
            payload={"text": CORRECTED_TEXT, "uncertain": []},
            model="fake-stt",
        ),
        "fake-day": FakeLayerProvider(
            header=ENTRIES_HEADER,
            payload={
                "facts": ["Сходил в поликлинику за справкой."],
                "themes": ["поликлиника"],
                "energy": None,
                "open_questions": [],
            },
            model="fake-day",
        ),
        "fake-week": FakeLayerProvider(
            header=SUMMARIES_HEADER, payload={"rollup": DAY_ROLLUP}, model="fake-week"
        ),
        "fake-coach": FakeLayerProvider(
            header="Недельная сводка (JSON, weekly_summary):",
            payload={
                "observations": ["В сводке упомянуты бытовые дела."] * 3,
                "hypotheses": ["Возможно, неделя была спокойной."] * 2,
                "experiment_7d": "Отмечать бытовые дела в течение 7 дней.",
            },
            model="fake-coach",
        ),
    }


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the data root at a per-test directory and isolate the environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


@pytest.fixture
def migrated_settings() -> object:
    """A migrated database inside the per-test data root (migrations only)."""
    settings = get_settings()
    upgrade_database_file(settings.db_dir / "diary.db")
    return settings


@pytest.fixture(autouse=True)
def _registered_layers() -> None:
    snapshot = dict(pipeline_registry.SECOND_FAMILY_REGISTRY)
    ensure_stt_fix_pipeline_registered()
    ensure_sum_daily_pipeline_registered()
    ensure_sum_weekly_pipeline_registered()
    ensure_coach_pipeline_registered()
    yield
    pipeline_registry.SECOND_FAMILY_REGISTRY.clear()
    pipeline_registry.SECOND_FAMILY_REGISTRY.update(snapshot)


@pytest.fixture
def layer_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, FakeLayerProvider]:
    """Fake providers + in-memory models config wired into the CLI and the chain."""
    providers = _fake_providers()
    monkeypatch.setattr("diary.cli.load_models_config", lambda path=None: _models_config())
    monkeypatch.setattr(phase2_pipeline, "load_models_config", lambda: _models_config())
    monkeypatch.setattr(
        phase2_pipeline,
        "create_llm_provider",
        lambda profile: providers[profile.model],
    )
    return providers


def _open_session() -> Session:
    engine = create_engine(settings=get_settings())
    return create_session_factory(engine)()


def _add_message(body: str = RAW_TEXT, *, created_at: datetime = MORNING_UTC) -> None:
    with _open_session() as session:
        MessageRepository(session).create(
            kind=MessageKind.text, body_text=body, created_at=created_at
        )
        session.commit()


def _stored(layer: ArtifactLayer) -> list[object]:
    with _open_session() as session:
        return list_layer_results(session, layer=layer)


# ---------------------------------------------------------------------------
# day
# ---------------------------------------------------------------------------


def test_day_command_runs_the_chain_and_reports_safe_metadata(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_message()

    exit_code = main(["day", "--date", DAY.isoformat(), "--tz", LOCAL_TZ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"day {DAY.isoformat()}" in out
    assert "step layer=stt_corrected input=message:" in out
    assert "step layer=daily_summary input=day:" in out
    assert "summary layer=daily_summary artifact=" in out
    # Counts and identifiers only: no diary text, no prompt, no model output.
    assert RAW_TEXT not in out and CORRECTED_TEXT not in out and DAY_ROLLUP not in out
    assert len(_stored(ArtifactLayer.daily_summary)) == 1
    assert len(_stored(ArtifactLayer.stt_corrected)) == 1


def test_day_command_fails_when_the_day_layer_is_switched_off(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_message()
    monkeypatch.setattr("diary.cli.load_models_config", lambda path=None: _models_config(day=False))

    exit_code = main(["day", "--date", DAY.isoformat()])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "daily_summary" in captured.err and "diary_sum" in captured.err
    assert _stored(ArtifactLayer.daily_summary) == []


def test_day_command_without_messages_succeeds_as_a_noop(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["day", "--date", DAY.isoformat()])

    assert exit_code == 0
    assert "nothing to do" in capsys.readouterr().out
    assert _stored(ArtifactLayer.daily_summary) == []


def test_day_command_rejects_a_malformed_date(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["day", "--date", "14.09.2026"])

    assert exit_code == 1
    assert "ISO date" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# week
# ---------------------------------------------------------------------------


def test_week_command_merges_the_stored_day_summary(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_message()
    assert main(["day", "--date", DAY.isoformat(), "--tz", LOCAL_TZ]) == 0
    capsys.readouterr()

    exit_code = main(["week", "--week", WEEK_ID])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"week {WEEK_ID}" in out
    assert "step layer=weekly_summary input=week:" in out
    assert "summary layer=weekly_summary artifact=" in out
    assert len(_stored(ArtifactLayer.weekly_summary)) == 1
    # The week layer consumed a day summary, never the raw message.
    week_request = layer_providers["fake-week"].requests[0].messages[0].content
    assert DAY.isoformat() in week_request and RAW_TEXT not in week_request


def test_week_command_with_run_days_produces_the_whole_chain(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_message()

    exit_code = main(["week", "--week", WEEK_ID, "--run-days", "--tz", LOCAL_TZ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "step layer=daily_summary" in out
    assert "step layer=weekly_summary" in out
    assert len(_stored(ArtifactLayer.daily_summary)) == 1
    assert len(_stored(ArtifactLayer.weekly_summary)) == 1


def test_week_command_without_day_summaries_is_a_noop_with_a_hint(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["week", "--week", WEEK_ID])

    assert exit_code == 0
    assert "--run-days" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# coach — the only launcher of layer 5
# ---------------------------------------------------------------------------


def test_coach_command_produces_the_note(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_message()
    assert main(["week", "--week", WEEK_ID, "--run-days", "--tz", LOCAL_TZ]) == 0
    capsys.readouterr()
    # The weekly rollup store must be visible for the note's anchor.
    assert _stored(ArtifactLayer.weekly_summary)

    exit_code = main(["coach", "--week", WEEK_ID])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"coach {WEEK_ID}" in out
    assert "step layer=coach_note" in out
    assert "note layer=coach_note artifact=" in out
    assert "observations=3 hypotheses=2" in out
    assert len(_stored(ArtifactLayer.coach_note)) == 1


def test_coach_command_fails_when_the_layer_is_switched_off(
    migrated_settings: object,
    layer_providers: dict[str, FakeLayerProvider],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_message()
    monkeypatch.setattr(
        "diary.cli.load_models_config", lambda path=None: _models_config(coach=False)
    )

    exit_code = main(["coach", "--week", WEEK_ID])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "diary_coach" in captured.err
    assert _stored(ArtifactLayer.coach_note) == []


# ---------------------------------------------------------------------------
# the single entry point itself
# ---------------------------------------------------------------------------


def test_the_message_correction_has_no_cli_verb(
    migrated_settings: object, layer_providers: dict[str, FakeLayerProvider]
) -> None:
    """The single-message unit is a job (``stt_corrected``), not a CLI verb."""
    with pytest.raises(SystemExit):
        main(["stt-fix", "--message", "00000000-0000-0000-0000-000000000000"])


def test_the_cli_help_lists_the_phase2_commands(
    migrated_settings: object, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("backup", "recover", "day", "week", "coach"):
        assert command in out
