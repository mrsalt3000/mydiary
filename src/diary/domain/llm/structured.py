"""Structured-output bridge: one LLM completion → one validated contract (task T04).

Every phase-2 derivational layer (``stt_corrected``, and the daily/weekly
summaries and the coaching note built on the same pattern) exchanges a strict
Pydantic contract with the LLM boundary (``docs/tasklist-phase2.md`` §4,
AGENTS.md §15). This module is the single, reusable seam of that exchange:

* :func:`parse_contract_json` — **strict** JSON parsing of a completion. The
  response must be exactly one JSON object: no Markdown fences are stripped
  and no ``{...}`` span is hunted for inside free text. A model that returns
  prose, a fenced block or an array has produced an invalid output, and
  invalid output is a failed run, never a partial artifact;
* :func:`parse_contract_response` — parses a completion and validates it
  against the contract of the calling layer, with the **envelope** fields
  supplied by the backend.

Envelope fields are the part of a contract the model cannot know and must not
be trusted with (ADR-007 §7): the identity of the source record and the run
provenance (``model``, ``prompt_ver``). The caller passes them in; their values
**overwrite** anything the completion reported under the same keys, so a model
that echoes a foreign model name or prompt version cannot influence the stored
record. Keys the contract does not declare are rejected by the contract itself
(``extra="forbid"``), so an invented field fails the run as well.

Failures raise :class:`~diary.domain.llm.errors.LLMStructuredOutputError` (a
non-retryable LLM error with a static message) — the caller then records a
failed run and writes nothing. Completions, prompts and diary text never appear
in messages, logs or exceptions (AGENTS.md §7-8).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from diary.domain.llm.errors import LLMStructuredOutputError
from diary.domain.llm.models import LLMResponse

#: Type of the structured, validated contract of the calling layer.
SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Static, non-sensitive failure summaries (AGENTS.md §8).
_NOT_JSON_MESSAGE = "The model response is not a single JSON object."
_INVALID_SCHEMA_MESSAGE = "The model response did not pass structured validation."


def parse_contract_json(text: str) -> dict[str, object]:
    """Strictly parse a completion into a JSON object.

    The whole (whitespace-trimmed) response must be one JSON object. Markdown
    fences, surrounding prose, a JSON array or a scalar are all rejected: the
    bridge deliberately offers no "extract the JSON from the text" fallback,
    because silently repairing a malformed answer is exactly what lets a
    half-understood result reach a stored artifact (AGENTS.md §15).
    """
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise LLMStructuredOutputError(_NOT_JSON_MESSAGE) from exc
    if not isinstance(data, dict):
        raise LLMStructuredOutputError(_NOT_JSON_MESSAGE)
    return data


def parse_contract_response(
    response: LLMResponse,
    contract: type[SchemaT],
    *,
    envelope: Mapping[str, object],
) -> SchemaT:
    """Validate ``response`` into ``contract`` with backend-supplied envelope fields.

    ``envelope`` carries the fields the backend owns (source identity and run
    provenance). Its values win over the completion, so the returned model
    always reports the resolved profile model and prompt version rather than
    whatever the model claimed (ADR-007 §7). Any validation failure — a missing
    required field, an undeclared extra field, a ``uncertain`` entry that is not
    a verbatim substring of ``text`` — raises
    :class:`LLMStructuredOutputError`; the caller stores nothing.
    """
    payload: dict[str, object] = {**parse_contract_json(response.text), **envelope}
    try:
        return contract.model_validate(payload)
    except ValidationError as exc:
        raise LLMStructuredOutputError(_INVALID_SCHEMA_MESSAGE) from exc


__all__ = ["SchemaT", "parse_contract_json", "parse_contract_response"]
