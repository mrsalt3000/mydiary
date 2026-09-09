"""Synthetic fakes for the Task-15 wellbeing integration tests.

Mirrors the Task-13/14 fake-provider approach: the fake LLM provider is wired
in through the module-level seams (``_load_models_config`` /
``create_llm_provider`` of ``diary.domain.pipelines.service``), and the real
:class:`~diary.domain.pipelines.wellbeing.pipeline.WellbeingPipeline` is
registered explicitly per test (never as an import side effect). All content
is synthetic (AGENTS.md §13).
"""

from __future__ import annotations

from collections.abc import Callable

from _pipeline_fakes import json_llm_response

from diary.domain.llm.models import LLMRequest, LLMResponse

#: Marker the fake embeds into observed facts so tests can prove which text
#: the request carried.
FACT_MARKER = "[2026-09-07] Синтетический факт:"


def wellbeing_payload(
    *,
    observed_facts: list[str],
    cautious_interpretation: list[str] | None = None,
    recommendations: list[str] | None = None,
    confidence_and_limitations: list[str] | None = None,
) -> dict[str, object]:
    """A valid wellbeing JSON payload over the strict schema."""
    return {
        "observed_facts": observed_facts,
        "cautious_interpretation": (
            cautious_interpretation if cautious_interpretation is not None else []
        ),
        "recommendations": (
            recommendations if recommendations is not None else ["Больше отдыхать."]
        ),
        "confidence_and_limitations": (
            confidence_and_limitations
            if confidence_and_limitations is not None
            else ["Только записи одной недели."]
        ),
    }


def default_wellbeing_payload(text: str) -> dict[str, object]:
    """Deterministic default payload independent of the request content.

    The fake records the request so tests can assert what the model saw; the
    returned payload is always the same benign reflection — the safety level
    is decided by the deterministic classifier, never by this payload.
    """
    return wellbeing_payload(
        observed_facts=[f"{FACT_MARKER} запись отражает нейтральный день."],
        cautious_interpretation=["Возможно, неделя была ровной."],
        recommendations=["Продолжать вести записи."],
        confidence_and_limitations=["Выводы только по одной неделе."],
    )


class FakeWellbeingLLM:
    """Deterministic fake LLM provider for the real wellbeing pipeline.

    ``responder(text)`` maps the request content to the JSON payload of the
    response (default: :func:`default_wellbeing_payload`); tests inject custom
    responders to exercise malformed/schema-invalid/urgent states.
    """

    def __init__(
        self,
        responder: Callable[[str], dict[str, object]] | None = None,
    ) -> None:
        self.responder = responder or default_wellbeing_payload
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        content = request.messages[-1].content
        return json_llm_response(self.responder(content))
