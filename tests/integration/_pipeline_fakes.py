"""Synthetic fakes for the Task-12 pipeline-framework integration tests.

Mirrors the Task-09 fake-provider approach: a fake LLM provider, an
in-memory models configuration and a fake ``normalize`` pipeline are
installed per test via monkeypatch and the module-level registries, so no
real endpoint, config file or model is ever touched. All content used here is
synthetic (AGENTS.md §13).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
    TaskType,
)
from diary.core.settings import get_settings
from diary.db.models import Artifact, ArtifactLayer, PipelineType, WeeklyExport
from diary.domain.llm.models import LLMRequest, LLMResponse, LLMUsage
from diary.domain.pipelines import registry as pipeline_registry
from diary.domain.pipelines import service as pipeline_service
from diary.domain.pipelines.base import Pipeline, PipelineOutputError

#: Static message used for invalid structured output (content-free).
INVALID_OUTPUT_MESSAGE = "The pipeline output did not pass structured validation."

FAKE_PROFILE_KEY = "FAKE_PIPELINE_API_KEY"


class FakeNormalized(BaseModel):
    """Structured output schema of the fake normalize pipeline."""

    lines: list[str] = Field(min_length=1)


def json_llm_response(payload: object, *, usage: LLMUsage | None = None) -> LLMResponse:
    """An LLMResponse whose text is the JSON encoding of ``payload``."""
    return LLMResponse(
        text=json.dumps(payload, ensure_ascii=False),
        model="fake-model",
        usage=usage,
    )


class FakeNormalizePipeline(Pipeline[FakeNormalized]):
    """A deterministic fake of the future normalization pipeline (Task 13).

    Reads the source original export file, embeds a marker line plus the
    content into the LLM request (diary content is allowed in-memory in a
    request, never logged), expects a ``{"lines": [...]}`` JSON response and
    renders the validated lines back as text.
    """

    pipeline_type = PipelineType.normalize
    layer = ArtifactLayer.normalized
    output_ext = ".txt"
    task_type = TaskType.normalization
    result_schema = FakeNormalized

    def build_request(self, *, export: WeeklyExport, input_artifact: Artifact | None) -> LLMRequest:
        path = (
            get_settings().data_root / export.path
            if input_artifact is None
            else get_settings().data_root / input_artifact.path
        )
        content = path.read_text(encoding="utf-8")
        return LLMRequest.from_prompt(
            f"Обработай выгрузку (синтетика):\n{content}", prompt_version="v1"
        )

    def parse_result(self, response: LLMResponse) -> FakeNormalized:
        try:
            data = json.loads(response.text)
            return FakeNormalized.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise PipelineOutputError(INVALID_OUTPUT_MESSAGE) from exc

    def render(self, structured: FakeNormalized) -> str:
        return "\n".join(structured.lines).rstrip("\n") + "\n"


class FakeLLMProvider:
    """Deterministic fake LLM provider (Task-11 protocol).

    Raises the given ``failures`` (in order, one per call) and afterwards
    returns ``response`` forever. Requests are recorded so tests can assert
    that the provider is not called twice on a replay.
    """

    def __init__(
        self,
        *,
        response: LLMResponse | None = None,
        failures: list[Exception] | None = None,
    ) -> None:
        self.response = response or json_llm_response({"lines": ["Синтетическая строка."]})
        self._failures = list(failures or [])
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        if self._failures:
            raise self._failures.pop(0)
        return self.response


def llm_profile(model: str) -> OpenAICompatibleProfile:
    """An ``openai_compatible`` profile for the fake endpoint."""
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url="http://127.0.0.1:1/v1",  # never contacted by the fake
        model=model,
        api_key_env=FAKE_PROFILE_KEY,
        timeout_s=10.0,
        temperature=0.0,
        max_tokens=256,
    )


def fake_models_config() -> ModelsConfig:
    """A complete in-memory Task-03 configuration (all four profiles)."""
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=llm_profile("fake-norm-model"),
        summarization=llm_profile("fake-summary-model"),
        wellbeing_analysis=llm_profile("fake-wellbeing-model"),
    )


def register_fake_normalize_pipeline() -> None:
    """Register the fake normalize pipeline in the pipeline registry."""
    pipeline_registry.register(FakeNormalizePipeline())


def install_fake_llm(monkeypatch, provider: FakeLLMProvider) -> None:
    """Wire the fake models config and provider factory into the pipeline
    service (module-level hooks, like the Task-09 STT fakes)."""
    monkeypatch.setattr(pipeline_service, "_load_models_config", lambda: fake_models_config())
    monkeypatch.setattr(pipeline_service, "create_llm_provider", lambda profile: provider)


class ScriptedNliJudge:
    """A judge that answers from a table, for tests that run a gated layer.

    The runtime NLI gate (T24b) is armed by the production entries, which means a
    test that goes through the CLI or a job handler would otherwise need the real
    712 MB instrument — and CI has neither it nor the optional extra. Tests install
    this double instead, exactly as they install a scripted provider: the gate's own
    behaviour is tested where it lives (``tests/unit/test_nli_guard.py`` and the day
    layer's storage tests), and the entries' wiring is pinned by a separate test.

    ``entailment`` is what every pair is answered with, so a test can choose between
    «the source supports every claim» (the default) and «nothing is supported».
    """

    def __init__(self, entailment: float = 0.99, contradiction: float = 0.005) -> None:
        self.entailment = entailment
        self.contradiction = contradiction

    def load(self) -> None:
        """The double needs no weights; the gate calls this to prove the instrument."""

    def __call__(self, pairs):
        neutral = max(0.0, 1.0 - self.entailment - self.contradiction)
        return [
            {
                "entailment": self.entailment,
                "contradiction": self.contradiction,
                "neutral": neutral,
            }
            for _ in pairs
        ]


def supported_nli_judge() -> ScriptedNliJudge:
    """The gate-armed entries in tests: everything is supported by its source."""
    return ScriptedNliJudge()


def contradicting_nli_judge() -> ScriptedNliJudge:
    """A source that contradicts every claim (used to exercise a refusal)."""
    return ScriptedNliJudge(entailment=0.01, contradiction=0.98)
