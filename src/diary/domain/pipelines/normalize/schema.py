"""Strict Pydantic schema of the normalization structured output (Task 13).

The model normalizes *only* what the pipeline sent it: one entry per original
export entry that carries normalizable text, in the original order. Every
returned record must echo the original entry's ``index`` (0-based position in
the export file), ``timestamp`` (the rendered ``YYYY-MM-DD HH:MM`` stamp) and
``kind`` — the completeness check in the pipeline then verifies that the
returned records match the sent ones one-to-one, so a result can never
silently drop, add, reorder or re-time an entry (AGENTS.md §15).

Audio entries without a transcription never reach the model (there is no text
to normalize); the backend re-inserts their ``[не расшифровано]`` marker at
render time from the immutable original parse.

Privacy contract: the schema carries diary-derived text only in
``normalized_text`` inside an in-memory Pydantic model — never in logs, error
summaries or persisted metadata.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The only timestamp shape an original export entry can have.
_TIMESTAMP_PATTERN = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"

#: Source kinds an entry may have (echoed from the layer-1 export).
EntryKind = Literal["text", "audio"]


class NormalizedEntry(BaseModel):
    """One normalized diary entry returned by the model.

    ``index``/``timestamp``/``kind`` must be copied from the corresponding
    input record (the pipeline fails the run when they do not match). Fields
    are strictly fixed — the schema forbids any additional key so the model
    cannot smuggle out-of-schema content into the artifact.
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    timestamp: str = Field(pattern=_TIMESTAMP_PATTERN)
    kind: EntryKind
    normalized_text: str


class NormalizeResult(BaseModel):
    """The whole structured output of one normalization chunk.

    ``entries`` keeps the input order; ``entry_count`` must equal the number
    of entries (the model is instructed to fill both, and the validator
    rejects an inconsistent response before any file is written).
    """

    model_config = ConfigDict(extra="forbid")

    entries: list[NormalizedEntry]
    entry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _entry_count_matches(self) -> "NormalizeResult":
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must equal the number of entries.")
        return self


__all__ = [
    "EntryKind",
    "NormalizeResult",
    "NormalizedEntry",
]
