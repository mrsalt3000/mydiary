"""Unit tests for the runtime invariant gate (T17b, closing finding F1).

The gate is what turns the shared checks from a measurement into a guarantee:
before it, a layer validated its completion against the Pydantic contract and
stored whatever text came back, so advice, causal claims and invented facts
reached the diary (phase-2 acceptance, items 4–5). These tests fix the contract
of the gate itself:

* a clean output is returned on the first attempt and the model is called once;
* a violating output is generated again — a fresh call, never a repaired result —
  and the first clean attempt wins;
* a persistently violating output raises a **permanent** error, names the checks
  it broke and stores nothing, with no generated text in the message or the log
  (AGENTS.md §7–8);
* the number of attempts is bounded, so a systematic prompt defect cannot become
  a loop of failed calls;
* a source with no content skips grounding (nothing to compare against) but
  still refuses advice and causal claims.

All text is synthetic (AGENTS.md §13); no provider, database or model is used.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from diary.domain.contracts.invariants import ADVICE_CHECK, CAUSAL_CHECK, NEW_FACTS_CHECK
from diary.domain.pipelines.base import (
    PipelineInvariantError,
    PipelineOutputError,
    PipelinePermanentError,
)
from diary.domain.pipelines.quality import GENERATION_ATTEMPTS, generate_with_invariants


class _Note(BaseModel):
    """Stand-in for a coaching note: the gate only reads the payload's fields."""

    observations: list[str] = []
    hypotheses: list[str] = []
    experiment_7d: str = "Эксперимент на 7 дней."


#: The week a note is written from: its quantities are spelled out as words.
SOURCE = {"rollup": "Просыпалась три раза за ночь."}

CLEAN = _Note(observations=["Просыпалась три раза за ночь."])
ADVICE = _Note(observations=["Ты молодец, так держать."])
CAUSAL = _Note(observations=["Просыпалась, что привело к усталости."])
INVENTED = _Note(observations=["Просыпалась 11 раз за ночь."])


def _generator(*models: _Note) -> tuple[list[int], object]:
    """A scripted ``generate``: answers with ``models`` in order, recording attempts."""
    attempts: list[int] = []

    def generate(attempt: int) -> _Note:
        attempts.append(attempt)
        index = min(attempt, len(models)) - 1
        return models[index]

    return attempts, generate


def test_the_attempt_budget_is_one_regeneration() -> None:
    """The number itself is the policy (ADR-012), so it is pinned literally."""
    assert GENERATION_ATTEMPTS == 2


def test_a_clean_output_is_accepted_on_the_first_attempt() -> None:
    attempts, generate = _generator(CLEAN)
    result = generate_with_invariants(layer="coach_note", source=SOURCE, generate=generate)
    assert result is CLEAN
    assert attempts == [1]


def test_a_violating_output_is_generated_again_and_the_clean_one_wins() -> None:
    attempts, generate = _generator(ADVICE, CLEAN)
    result = generate_with_invariants(layer="coach_note", source=SOURCE, generate=generate)
    assert result is CLEAN
    assert attempts == [1, 2]


def test_a_persistently_violating_output_is_refused() -> None:
    attempts, generate = _generator(ADVICE)
    with pytest.raises(PipelineInvariantError) as error:
        generate_with_invariants(layer="coach_note", source=SOURCE, generate=generate)
    assert error.value.checks == (ADVICE_CHECK,)
    # Bounded: a systematic defect costs a fixed number of calls, not a retry loop.
    assert attempts == list(range(1, GENERATION_ATTEMPTS + 1))


def test_the_refusal_is_permanent_so_the_job_is_not_retried() -> None:
    """The worker maps a permanent pipeline error to a permanent job failure."""
    assert issubclass(PipelineInvariantError, PipelineOutputError)
    assert issubclass(PipelineInvariantError, PipelinePermanentError)


def test_the_refusal_message_and_log_carry_no_generated_text() -> None:
    """Only check names and counts may leave the process (AGENTS.md §7–8)."""
    _, generate = _generator(ADVICE)
    with pytest.raises(PipelineInvariantError) as error:
        generate_with_invariants(layer="coach_note", source=SOURCE, generate=generate)
    assert ADVICE_CHECK in str(error.value)
    assert ADVICE_CHECK in error.value.checks
    assert "молодец" not in str(error.value)


def test_a_regeneration_is_logged_without_content(caplog: pytest.LogCaptureFixture) -> None:
    attempts, generate = _generator(CAUSAL, CLEAN)
    with caplog.at_level(logging.WARNING, logger="diary.domain.pipelines.quality"):
        generate_with_invariants(layer="coach_note", source=SOURCE, generate=generate)
    assert attempts == [1, 2]
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert CAUSAL_CHECK in message
    assert "привело" not in message


def test_every_violated_check_is_named() -> None:
    both = _Note(observations=["Ты молодец.", "Просыпалась, что привело к усталости."])
    _, generate = _generator(both)
    with pytest.raises(PipelineInvariantError) as error:
        generate_with_invariants(layer="coach_note", source=SOURCE, generate=generate)
    assert set(error.value.checks) == {ADVICE_CHECK, CAUSAL_CHECK}


def test_an_empty_source_still_refuses_advice_and_causal_claims() -> None:
    """Advice and causality do not compare against the source, so they always apply."""
    for violated, check in ((ADVICE, ADVICE_CHECK), (CAUSAL, CAUSAL_CHECK)):
        _, generate = _generator(violated)
        with pytest.raises(PipelineInvariantError) as error:
            generate_with_invariants(layer="coach_note", source=[], generate=generate)
        assert error.value.checks == (check,)


def test_an_empty_source_skips_grounding() -> None:
    """With no text to compare against, "not in the input" is unwinnable."""
    attempts, generate = _generator(INVENTED)
    result = generate_with_invariants(layer="coach_note", source=[], generate=generate)
    assert result is INVENTED  # accepted, grounding apart
    assert attempts == [1]  # no regeneration was spent on it
    # The same output is refused as soon as the source has content to ground in.
    _, only = _generator(INVENTED)
    with pytest.raises(PipelineInvariantError) as error:
        generate_with_invariants(layer="coach_note", source=SOURCE, generate=only)
    assert error.value.checks == (NEW_FACTS_CHECK,)
