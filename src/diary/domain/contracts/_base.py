"""Shared base class of the phase-2 structured contracts (task T00).

The phase-2 backlog (``docs/tasklist-phase2.md``) introduces a second family
of derivational layers — ``stt_corrected``, ``daily_summary``,
``weekly_summary``, ``coach_note`` — that is anchored to messages/days instead
of the weekly original export (ADR-007). Every such layer exchanges a
validated Pydantic contract with the LLM boundary; this module holds the one
base class they all share.

Contract invariants (AGENTS.md §15):

- ``extra="forbid"`` — an unknown field is a validation error, never a silently
  dropped key: structured LLM output must fail the run, not be half-accepted;
- ``contract_name`` — stable identifier of the contract, equal to its file name
  in ``schemas/`` and to the ``contract`` value inside ``eval/golden/``;
- ``schema_version`` — version of the *contract pair* (input, layer) as a whole
  (``docs/data-lineage.md`` §3): it is bumped by a breaking change of the
  contract, while compatible additions do not bump it (ADR-007).

These are ``ClassVar``s: they are metadata about the contract, not payload, so
they never appear in ``model_json_schema()`` output or in serialized data.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base class of every phase-2 structured contract (task T00).

    Subclasses must set :attr:`contract_name`; ``extra="forbid"`` is inherited
    by all of them (Pydantic merges ``model_config`` from the base class).
    """

    model_config = ConfigDict(extra="forbid")

    #: Stable contract identifier; equals the JSON Schema file name in
    #: ``schemas/<contract_name>.json`` and the ``contract`` field of a golden
    #: chunk (``eval/golden``).
    contract_name: ClassVar[str]

    #: Integer version of the contract as a whole. ``1`` for the first
    #: accepted shape of every contract; a breaking change increments it and
    #: must be recorded in ``docs/tasklist-phase2.md`` and ADR-007.
    schema_version: ClassVar[int] = 1
