"""Runtime gate: a layer may not store output that breaks an invariant (T17b).

Finding F1 of the phase-2 acceptance was that the three generated-text
invariants existed only in the eval harness: the layers validated their result
against a Pydantic contract, but nothing checked what the *text* claimed, so
causal connectors, a substituted quantity and a reversal of meaning reached
stored artifacts (`docs/acceptance-log.md`, T12 items 4–5). T17 moved the checks
into :mod:`diary.domain.contracts.invariants` and made them honest; this module
makes them binding.

**Policy: refuse, do not annotate.** A violating output is generated again, at
most
:data:`GENERATION_ATTEMPTS` times, and if it still violates an invariant the
layer raises :class:`~diary.domain.pipelines.base.PipelineInvariantError` and
stores nothing. The alternative — storing the artifact with a warning recorded
beside it — was rejected deliberately: the defect the acceptance measured is the
text itself reaching the diary, so an artifact that carries advice, a causal
claim or an invented fact contradicts the guarantee the product advertises, and
a note beside it does not make the guarantee true. The cost is honest and
visible: a day whose summary cannot be generated cleanly ends without a summary
(the raw entries are never touched), the failed run is recorded, and the day can
be re-run explicitly.

**What the gate does not do.** It applies the three invariants and nothing else;
it does not judge the quality of a summary, and it cannot see an invented fact
that is written as free-form prose rather than as a number, a numeral word or a
proper name (the documented limit of the grounding check). Re-generation is a
fresh call to the model, not a repair of its text: nothing rewrites the diary or
the artifact.

**Logs stay content-free.** A violation's `detail` quotes generated text, so the
gate reports only :func:`~diary.domain.contracts.invariants.violation_summary`
— check names and counts — in its log lines and in the exception message
(AGENTS.md §7–8).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from diary.domain.contracts.invariants import (
    InputFacts,
    InvariantViolation,
    advice_and_judgement_violations,
    causal_claim_violations,
    check_generated_text,
    input_facts_from_models,
    violation_summary,
)
from diary.domain.pipelines.base import PipelineInvariantError

logger = logging.getLogger(__name__)

#: Validated structured output of a derivational layer.
SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: How many times a layer may generate before the gate refuses to store anything.
#: Two means one re-generation: enough for a stochastic slip (a causal connector
#: the model can drop on a second try), not enough to turn a systematic prompt
#: defect into a loop of failed calls.
GENERATION_ATTEMPTS = 2


def generate_with_invariants(
    *,
    layer: str,
    source: object,
    generate: Callable[[int], SchemaT],
) -> SchemaT:
    """Generate once, checking the invariants; on violation generate again.

    ``layer`` is the contract name the layer validates against (``"stt_corrected"``,
    ``"daily_summary"``, ``"weekly_summary"``, ``"coach_note"``), ``source`` is
    the input the layer was given (contract models or their payloads) and
    ``generate`` performs one full attempt — the request, the provider call and
    the contract validation — returning the validated model. ``generate`` is
    called with the 1-based attempt number so a layer may vary the request; every
    attempt is a fresh call, never a repaired result.

    Returns the first output that satisfies every invariant. Raises
    :class:`~diary.domain.pipelines.base.PipelineInvariantError` — permanent, so
    the job does not enter the worker's retry loop — when every attempt violates
    an invariant; the caller stores nothing.
    """
    facts = input_facts_from_models(source)
    violations: list[InvariantViolation] = []
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        model = generate(attempt)
        violations = _violations_of(layer, model, facts)
        if not violations:
            return model
        logger.warning(
            "layer %s violated %s on attempt %d of %d",
            layer,
            violation_summary(violations),
            attempt,
            GENERATION_ATTEMPTS,
        )
    raise PipelineInvariantError(
        f"The {layer} output violated the generated-text invariants "
        f"({violation_summary(violations)}) in all {GENERATION_ATTEMPTS} generations; "
        "nothing was stored.",
        checks=tuple(dict.fromkeys(item.check for item in violations)),
    )


def _violations_of(layer: str, model: BaseModel, facts: InputFacts) -> list[InvariantViolation]:
    """The invariants one generated model breaks, given its source.

    Grounding is skipped when the source carries no content at all: with no text
    to compare against, "the number does not occur in the input" is unwinnable
    rather than informative, and a layer fed nothing has no facts to invent (the
    orchestrator's own empty-input paths skip such a layer before it is called).
    The advice and causality checks still apply — they do not compare against the
    source, and a correction layer's fidelity exemption degrades to "everything
    is new wording", which is the strict reading.
    """
    payload = model.model_dump(mode="json")
    if not facts.text.strip():
        return [
            *advice_and_judgement_violations(layer, payload, facts),
            *causal_claim_violations(layer, payload, facts),
        ]
    return check_generated_text(layer, payload, facts)


__all__ = ["GENERATION_ATTEMPTS", "SchemaT", "generate_with_invariants"]
