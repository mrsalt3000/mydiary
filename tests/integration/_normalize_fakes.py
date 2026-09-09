"""Synthetic fakes for the Task-13 normalization integration tests.

Mirrors the Task-12 fake-provider approach: the fake LLM provider is wired in
through the module-level seams (``_load_models_config`` /
``create_llm_provider`` of ``diary.domain.pipelines.service``), and the real
:class:`~diary.domain.pipelines.normalize.pipeline.NormalizePipeline` is
registered explicitly per test (never as an import side effect). All content
is synthetic (AGENTS.md §13).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from _pipeline_fakes import json_llm_response

from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.pipelines.normalize.pipeline import ENTRIES_HEADER

#: Prefix the deterministic fake writes into every normalized text so tests
#: can tell the normalized output apart from the original export text.
NORM_PREFIX = "Норм: "

#: Shape of one input record parsed from a chunk request.
InputRecord = dict[str, object]
#: Shape of one output record of the fake response.
OutputRecord = dict[str, object]


def echo_records(records: list[InputRecord]) -> list[OutputRecord]:
    """Deterministic echo-normalization: prefix each text and copy metadata.

    The ``index``/``timestamp``/``kind`` are copied unchanged (the pipeline
    checks them back), so this fake models a correct model output.
    """
    return [
        {
            "index": record["index"],
            "timestamp": record["timestamp"],
            "kind": record["kind"],
            "normalized_text": f"{NORM_PREFIX}{record['text']}",
        }
        for record in records
    ]


def extract_input_records(request: LLMRequest) -> list[InputRecord]:
    """Parse the chunk's input records out of a built request.

    The pipeline appends the entry JSON on the single line that follows
    :data:`~diary.domain.pipelines.normalize.pipeline.ENTRIES_HEADER`; this
    helper reads that block so the fake can answer any chunk without knowing
    the chunk boundaries in advance.
    """
    content = request.messages[-1].content
    tag = f"{ENTRIES_HEADER}\n"
    _, separator, tail = content.rpartition(tag)
    if not separator:
        raise AssertionError("The request does not contain the normalization entries block.")
    records = json.loads(tail.strip())
    if not isinstance(records, list):
        raise AssertionError("The normalization entries block is not a JSON array.")
    return records


class FakeNormalizeLLM:
    """Deterministic fake LLM provider for the real normalization pipeline.

    ``transform`` maps the input records of a request to the output records of
    the response (default: :func:`echo_records`); tests inject transforms that
    drop or corrupt records to exercise the completeness validation.
    """

    def __init__(
        self,
        transform: Callable[[list[InputRecord]], list[OutputRecord]] | None = None,
    ) -> None:
        self.transform = transform or echo_records
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        records = extract_input_records(request)
        entries = self.transform(records)
        return json_llm_response({"entries": entries, "entry_count": len(entries)})
