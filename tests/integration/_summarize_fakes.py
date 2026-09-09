"""Synthetic fakes for the Task-14 summarization integration tests.

Mirrors the Task-13 fake-provider approach: the fake LLM provider is wired in
through the module-level seams (``_load_models_config`` /
``create_llm_provider`` of ``diary.domain.pipelines.service``), the real
normalization pipeline produces the layer-2 input, and the real
:class:`~diary.domain.pipelines.summarize.pipeline.SummarizePipeline` is
registered explicitly per test (never as an import side effect). All content
is synthetic (AGENTS.md §13).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from _pipeline_fakes import json_llm_response

from diary.domain.llm.models import LLMRequest, LLMResponse

#: Request mode markers written by the pipeline after the static prompt.
_MODE_RE = re.compile(r"Режим: (финальный|частичный)(?:, блок (\d+) из (\d+))?")

#: Label of one request, used to make response content distinguishable.
ModeInfo = tuple[str, int, int]  # (mode, block index or 0, total or 0)


def extract_mode(content: str) -> ModeInfo:
    """Parse the mode line the pipeline appended to a request.

    Returns ``("final", 0, 0)`` for single-pass requests and
    ``("partial", index, total)`` for map-reduce requests.
    """
    match = _MODE_RE.search(content)
    if match is None:
        raise AssertionError("The request does not contain a summarize mode line.")
    if match.group(1) == "финальный":
        return ("final", 0, 0)
    return ("partial", int(match.group(2)), int(match.group(3)))


def summary_payload(
    *,
    key_events: list[str],
    work_and_load: list[str] | None = None,
    wellbeing_explicit: bool = False,
    wellbeing_points: list[str] | None = None,
    supportive_factors: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> dict[str, object]:
    """A valid summary JSON payload over the strict schema."""
    return {
        "key_events": key_events,
        "work_and_load": work_and_load if work_and_load is not None else ["Нагрузка умеренная."],
        "wellbeing": {
            "explicit_support": wellbeing_explicit,
            "points": wellbeing_points if wellbeing_points is not None else [],
        },
        "supportive_factors": (
            supportive_factors if supportive_factors is not None else ["Помогала семья."]
        ),
        "open_questions": open_questions if open_questions is not None else [],
    }


def default_summary_payload(mode: str, index: int, total: int) -> dict[str, object]:
    """Deterministic default payload that makes chunk outputs distinguishable.

    A single-pass request reports one "всей недели" event and no explicit
    wellbeing data. A map-reduce request reports one event per block
    (``блока N``); the *last* block turns the wellbeing section on, so a
    merged final summary must carry exactly one wellbeing point — proving the
    deterministic reduce combined the partials.
    """
    label = "всей недели" if mode == "final" else f"блока {index}"
    wellbeing_explicit = mode == "partial" and total > 1 and index == total
    wellbeing_points = [f"Явная усталость в конце недели ({label})."] if wellbeing_explicit else []
    return summary_payload(
        key_events=[f"Синтетическое событие {label}."],
        wellbeing_explicit=wellbeing_explicit,
        wellbeing_points=wellbeing_points,
    )


class FakeSummarizeLLM:
    """Deterministic fake LLM provider for the real summarization pipeline.

    ``responder(mode, index, total)`` maps a request to the JSON payload of
    the response (default: :func:`default_summary_payload`); tests inject
    custom responders to exercise malformed/special wellbeing states.
    """

    def __init__(
        self,
        responder: Callable[[str, int, int], dict[str, object]] | None = None,
    ) -> None:
        self.responder = responder or default_summary_payload
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        content = request.messages[-1].content
        mode, index, total = extract_mode(content)
        return json_llm_response(self.responder(mode, index, total))
