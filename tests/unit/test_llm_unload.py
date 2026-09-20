"""Unit tests for the explicit LLM unload (task T11, ADR-010).

Hermetic: every request goes through an ``httpx.MockTransport``, so no Ollama
server is contacted and nothing on the developer's machine is unloaded. What is
pinned here:

* the native endpoint is derived from the OpenAI-compatible ``base_url``
  (``scheme://host:port``) — ``keep_alive`` lives on ``/api/generate`` only;
* only **loopback** profiles are unloaded (an external endpoint holds nothing in
  this machine's VRAM) and a model named by several profiles is unloaded once;
* the request is exactly the ``ollama stop`` operation server-side
  (``{"model": ..., "keep_alive": 0}``);
* the failure policy that makes the veto hold without breaking the product:
  a confirmed unload → success, ``4xx`` (unknown model / no unload API) and an
  unreachable server → "nothing resident" (success), a reachable server
  answering ``5xx`` → a loud :class:`LlmUnloadFailedError` so STT never starts
  on an unconfirmed unload;
* the callable reads the model configuration per call (a long-lived worker must
  unload what its *current* configuration describes).
"""

from __future__ import annotations

import json

import httpx
import pytest

from diary.core.models_config import (
    FasterWhisperProfile,
    ModelsConfig,
    OpenAICompatibleProfile,
)
from diary.domain.llm import unload as unload_module


def _llm_profile(
    model: str, *, base_url: str = "http://127.0.0.1:11434/v1"
) -> OpenAICompatibleProfile:
    return OpenAICompatibleProfile(
        provider="openai_compatible",
        base_url=base_url,
        model=model,
        api_key_env="SYNTHETIC_KEY",
        temperature=0.2,
        max_tokens=64,
    )


def _config(
    *,
    phase2: dict[str, OpenAICompatibleProfile | None] | None = None,
    local: str = "http://127.0.0.1:11434/v1",
) -> ModelsConfig:
    defaults: dict[str, OpenAICompatibleProfile | None] = {
        "diary_stt": None,
        "diary_sum": None,
        "diary_week": None,
        "diary_coach": None,
    }
    defaults.update(phase2 or {})
    return ModelsConfig(
        transcription=FasterWhisperProfile(provider="faster_whisper", model="tiny"),
        normalization=_llm_profile("diary-stt", base_url=local),
        summarization=_llm_profile("diary-sum", base_url=local),
        wellbeing_analysis=_llm_profile("diary-coach", base_url=local),
        **defaults,
    )


def _recording_transport(
    journal: list[tuple[str, object]],
    *,
    status: int = 200,
    error: Exception | None = None,
) -> httpx.MockTransport:
    """A transport that journals every request and answers ``status``."""

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8")) if request.content else None
        journal.append((f"{request.method} {request.url}", payload))
        if error is not None:
            raise error
        return httpx.Response(status, json={"done": True})

    return httpx.MockTransport(handle)


# ---------------------------------------------------------------------------
# endpoint derivation
# ---------------------------------------------------------------------------


def test_native_root_drops_the_v1_prefix_and_keeps_host_and_port() -> None:
    assert unload_module.native_root_of("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"
    assert unload_module.native_root_of("https://localhost/v1") == "https://localhost"


def test_only_loopback_profiles_are_unloaded() -> None:
    external = _config(
        local="https://llm.example.com/v1",
    )
    assert unload_module.local_llm_endpoints(external) == ()


def test_every_local_profile_is_covered_and_duplicates_are_removed() -> None:
    """``diary-sum`` is named twice (MVP summarization + the phase-2 profile)."""
    endpoints = unload_module.local_llm_endpoints(
        _config(
            phase2={
                "diary_sum": _llm_profile("diary-sum"),
                "diary_week": _llm_profile("diary-sum-cpu"),
            }
        )
    )
    assert [(endpoint.native_root, endpoint.model) for endpoint in endpoints] == [
        ("http://127.0.0.1:11434", "diary-stt"),
        ("http://127.0.0.1:11434", "diary-sum"),
        ("http://127.0.0.1:11434", "diary-coach"),
        ("http://127.0.0.1:11434", "diary-sum-cpu"),
    ]


def test_coach_and_week_profiles_are_only_unloaded_when_configured() -> None:
    """An absent phase-2 profile names no model — nothing to evict for it."""
    endpoints = unload_module.local_llm_endpoints(_config())
    assert [endpoint.model for endpoint in endpoints] == ["diary-stt", "diary-sum", "diary-coach"]


# ---------------------------------------------------------------------------
# the request itself
# ---------------------------------------------------------------------------


def test_unload_model_sends_the_keep_alive_zero_request() -> None:
    journal: list[tuple[str, object]] = []
    endpoint = unload_module.LocalLlmEndpoint(
        native_root="http://127.0.0.1:11434", model="diary-stt"
    )

    confirmed = unload_module.unload_model(endpoint, transport=_recording_transport(journal))

    assert confirmed is True
    assert journal == [
        (
            "POST http://127.0.0.1:11434/api/generate",
            {"model": "diary-stt", "keep_alive": unload_module.KEEP_ALIVE_EVICT},
        )
    ]


# ---------------------------------------------------------------------------
# failure policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 404, 405])
def test_client_errors_mean_nothing_is_resident(status: int) -> None:
    """An unknown model or an endpoint without the unload API is not a failure."""
    endpoint = unload_module.LocalLlmEndpoint(native_root="http://127.0.0.1:11434", model="ghost")
    assert (
        unload_module.unload_model(endpoint, transport=_recording_transport([], status=status))
        is False
    )


def test_an_unreachable_server_means_nothing_is_resident() -> None:
    """The veto is about this machine's VRAM: no server, no resident model."""
    endpoint = unload_module.LocalLlmEndpoint(
        native_root="http://127.0.0.1:11434", model="diary-stt"
    )
    transport = _recording_transport(
        [], error=httpx.ConnectError("synthetic: nothing is listening")
    )
    assert unload_module.unload_model(endpoint, transport=transport) is False


def test_server_errors_are_loud() -> None:
    """A reachable server that did not confirm the unload must stop the turn."""
    endpoint = unload_module.LocalLlmEndpoint(
        native_root="http://127.0.0.1:11434", model="diary-stt"
    )
    with pytest.raises(unload_module.LlmUnloadFailedError, match="cannot be confirmed"):
        unload_module.unload_model(endpoint, transport=_recording_transport([], status=503))


# ---------------------------------------------------------------------------
# the injected callable
# ---------------------------------------------------------------------------


def test_the_callable_unloads_every_local_model() -> None:
    journal: list[tuple[str, object]] = []
    unloader = unload_module.build_llm_unloader(
        _config(
            phase2={
                "diary_stt": _llm_profile("diary-stt"),
                "diary_week": _llm_profile("diary-sum-cpu"),
            }
        ),
        transport=_recording_transport(journal),
    )

    unloader()

    assert [payload["model"] for _url, payload in journal] == [  # type: ignore[index]
        "diary-stt",
        "diary-sum",
        "diary-coach",
        "diary-sum-cpu",
    ]


def test_the_callable_is_a_noop_without_a_local_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    journal: list[tuple[str, object]] = []
    monkeypatch.setattr(
        unload_module,
        "load_models_config",
        lambda path=None: _config(local="https://llm.example.com/v1"),
    )
    unload_module.build_llm_unloader(transport=_recording_transport(journal))()

    assert journal == []


def test_the_configuration_is_read_when_the_callable_is_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-lived worker unloads what its **current** configuration describes."""
    journal: list[tuple[str, object]] = []
    seen: list[int] = []

    def _load(path=None) -> ModelsConfig:  # type: ignore[no-untyped-def]
        seen.append(1)
        return _config(phase2={"diary_stt": _llm_profile("diary-stt")})

    monkeypatch.setattr(unload_module, "load_models_config", _load)
    unloader = unload_module.build_llm_unloader(transport=_recording_transport(journal))

    assert seen == []  # nothing is read while the callable is only built
    unloader()
    assert seen == [1]


def test_a_missing_configuration_is_a_loud_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unload that cannot be built must not be skipped silently (veto)."""

    def _boom(path=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic: no configuration")

    monkeypatch.setattr(unload_module, "load_models_config", _boom)
    with pytest.raises(RuntimeError, match="no configuration"):
        unload_module.build_llm_unloader()()
