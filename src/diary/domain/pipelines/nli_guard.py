"""Runtime NLI gate: a layer refuses to store a summary that contradicts its source (T24b).

T24a built the instrument and measured it; this module makes the contradiction rule
binding in the runtime for the one contract it has been measured on — the **day
summary's `facts`**. The rule is the one the calibration justified
(``docs/nli-validation-analysis.md`` §8): claims are propositions of the layer's own
contract, the source is split into sentences on both sides, a claim that any source
sentence entails is safe whatever the others would say, and a claim nothing supports
is judged by its strongest contradiction against :data:`CONTRADICTION_THRESHOLD`.

**Scope is deliberate and narrow.** Nominal labels are exempt (a theme read as a
contradiction scores 0.90 on a faithful summary), the weekly rollup is exempt until
it has its own formulation (T26), and the coaching note was never in scope: the
gate covers what was measured, and the measurement is named next to each exemption
in :mod:`diary.domain.contracts.nli`.

**Policy: fail closed, and refuse rather than annotate** (ADR-012 and the owner's
decision for T24: «если NLI недоступен — Fail»). Two things follow:

* the instrument is proved **before** the layer generates: an unavailable extra, a
  missing weights file or a hash mismatch raises
  :class:`~diary.domain.pipelines.base.PipelinePermanentError`, so the run fails
  permanently instead of storing a summary nobody validated;
* a contradicting completion is generated again by the existing gate
  (:data:`~diary.domain.pipelines.quality.GENERATION_ATTEMPTS`) and, if it persists,
  the layer raises the invariant error and stores nothing — the artifact is never
  annotated, because the defect is the text itself.

**Logs and messages stay content-free.** A violation's detail carries the claim's
coordinate and probability (``facts#2 p_contradiction=0.91``) — an address and a
number, never the claim; the failure message is a fixed sentence. The claim text
itself goes only to the judge, which is the same disclosure the layer already makes
to its own model (AGENTS.md §7-8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from diary.core.settings import AppSettings
from diary.domain.contracts.invariants import InvariantViolation
from diary.domain.contracts.nli import (
    CONTRADICTION_CHECK,
    NliError,
    NliJudge,
    NliPremiseError,
    claims_for,
    judge_claims,
    premise_from_texts,
)
from diary.domain.nli.onnx_judge import NliUnavailableError, OnnxNliJudge
from diary.domain.pipelines.base import PipelinePermanentError

__all__ = [
    "GUARDED_CONTRACT",
    "contradiction_violations",
    "contradiction_guard",
    "default_nli_judge",
]

#: The contract the gate is written for today: the day summary (T24a measured its
#: ``facts``; the weekly rollup waits for T26, the coach note is out of scope).
GUARDED_CONTRACT = "daily_summary"

#: Fixed, content-free failure when the instrument cannot be used (AGENTS.md §8).
_INSTRUMENT_UNAVAILABLE = (
    "The NLI validation instrument is unavailable: the layer refuses to store a "
    "summary it cannot validate."
)
#: Fixed failure when there is nothing to validate against.
_SOURCE_UNUSABLE = "The NLI validation has no source to compare against: the layer refuses to run."


def default_nli_judge(settings: AppSettings | None = None) -> NliJudge:
    """The instrument of the runtime gate: the pinned ONNX cross-encoder (CPU).

    A module-level hook, so a test that runs a layer without the optional extra or
    the weights can install a scripted judge, and so the process builds one judge
    rather than one per step. Building it does not load 712 MB — the guard does
    that once, before the first generation, which is also where the failure
    surfaces when the instrument is not usable.
    """
    return OnnxNliJudge.from_settings(settings)


def contradiction_violations(
    payload: BaseModel, *, contract: str, premise: str, judge: NliJudge
) -> list[InvariantViolation]:
    """Judge the propositional claims of ``payload`` against ``premise``.

    Returns one violation per contradicted claim, carrying its coordinate and the
    probability — never the claim text (AGENTS.md §7-8).
    """
    claims = [
        claim
        for claim in claims_for(contract, payload.model_dump(mode="json"))
        if claim.is_propositional
    ]
    if not claims:
        return []
    judgements = judge_claims(premise, claims, judge)
    return [
        InvariantViolation(
            CONTRADICTION_CHECK,
            f"{judgement.claim.coordinate} "
            f"p_contradiction={judgement.contradiction_probability:.2f}",
        )
        for judgement in judgements
        if judgement.contradicts
    ]


def contradiction_guard(
    *,
    source_texts: Sequence[str],
    contract: str = GUARDED_CONTRACT,
    judge: NliJudge | None = None,
    settings: AppSettings | None = None,
) -> Callable[[BaseModel], Sequence[InvariantViolation]]:
    """Build the extra check for :func:`~diary.domain.pipelines.quality.generate_with_invariants`.

    The instrument is proved here, before any generation: an unavailable extra, a
    missing file or a wrong hash becomes
    :class:`~diary.domain.pipelines.base.PipelinePermanentError`. A source that
    cannot be reconstructed (no text at all) is a permanent failure too, because
    «nothing to compare against» must not silently mean «passed».
    """
    try:
        resolved = judge if judge is not None else default_nli_judge(settings)
        load = getattr(resolved, "load", None)
        if callable(load):
            # Force the instrument to prove itself now, so the failure is a property
            # of the run and not of the first claim that happens to be judged.
            load()
        premise = premise_from_texts(source_texts)
    except NliUnavailableError as exc:
        raise PipelinePermanentError(_INSTRUMENT_UNAVAILABLE) from exc
    except NliPremiseError as exc:
        raise PipelinePermanentError(_SOURCE_UNUSABLE) from exc

    def check(payload: BaseModel) -> Sequence[InvariantViolation]:
        try:
            return contradiction_violations(
                payload, contract=contract, premise=premise, judge=resolved
            )
        except NliError as exc:
            # A broken judge answer (wrong shape, mis-shaped probabilities) is a
            # failed validation, not a silent pass.
            raise PipelinePermanentError(_INSTRUMENT_UNAVAILABLE) from exc

    return check
