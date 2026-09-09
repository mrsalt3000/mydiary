"""Normalization pipeline (layer 2; Task 13, architecture brief §10.1).

Responsibilities
================

* Consume an **original weekly export** (layer 1) resolved by database
  identity (``export.path``); the original file is never modified.
* Parse the layer-1 TXT into ordered entries
  (:mod:`diary.domain.pipelines.normalize.parser`) and normalize only entries
  that carry text (message bodies and raw STT). Audio entries the export
  marked ``[не расшифровано]`` never reach the model — there is nothing to
  normalize; their marker is re-inserted at render time.
* Build one prompt instance per chunk from the versioned prompt template
  ``prompts/normalize/v1.md`` plus the chunk's entries. Inputs longer than
  ``AppSettings.normalize_chunk_entries`` are split into deterministic
  entry-chunks (order preserved); each chunk becomes its own
  :class:`LLMRequest`, so the framework makes one provider call per chunk.
* Validate every chunk response against the strict :class:`NormalizeResult`
  schema (extra fields are rejected) and then a **completeness check**: the
  returned entries must equal the sent chunk one-to-one — same count, same
  indices, same order, same timestamps and kinds. Any mismatch is a permanent
  :class:`PipelineOutputError`; the run fails and no file is written.
* Render the final layer-2 TXT **from the validated structured data** (never
  from raw model text). The header embeds the lineage: layer label, week,
  original sequence ``NNNN``/revision ``rN``, derived version ``vM``, the
  original file name and the creation instant/timezone — consistent with the
  layer-1 format. Entries are rendered as ``[YYYY-MM-DD HH:MM] TEXT`` lines in
  the original order; untranscribed audio entries keep the
  ``[не расшифровано]`` marker.

Concurrency note: the framework executes one job at a time per worker process
and drives ``build_requests → provider calls → parse_responses → render``
synchronously for each run on the single registered instance. The pipeline
therefore keeps the per-run parse (:class:`_RunContext`) on the instance as
transient state between those calls; it is overwritten by the next run. A
future parallel executor must give each job its own pipeline instance.

Privacy: no diary content, prompts or model output is ever logged or stored
in metadata (AGENTS.md §7–8, §15); every error message is static.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from diary.core.models_config import TaskType
from diary.core.settings import PROJECT_ROOT, get_settings
from diary.db.models import Artifact, ArtifactLayer, PipelineType, WeeklyExport
from diary.domain.exports.service import export_filename
from diary.domain.exports.weeks import week_monday
from diary.domain.llm.models import LLMRequest, LLMResponse, utc_now
from diary.domain.pipelines.base import (
    Pipeline,
    PipelineInputError,
    PipelineOutputError,
    PipelinePermanentError,
)
from diary.domain.pipelines.normalize.parser import (
    OriginalDocument,
    OriginalParseError,
    ParsedEntry,
    parse_original_text,
)
from diary.domain.pipelines.normalize.schema import NormalizedEntry, NormalizeResult

#: Versioned prompt template of this pipeline (recorded in run provenance).
PROMPT_VERSION = "v1"
#: Prompt template location: ``prompts/normalize/v1.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "normalize" / f"{PROMPT_VERSION}.md"
#: Line that introduces the chunk's entry records inside a request.
ENTRIES_HEADER = "Входные записи (JSON, одна строка):"
#: Marker kept for audio entries with no successful transcription.
_UNTRANSCRIBED_MARKER = "[не расшифровано]"

#: Static, non-sensitive error summaries (AGENTS.md §8).
_INPUT_PARSE_MESSAGE = "The original export file cannot be parsed."
_OUTPUT_INVALID_MESSAGE = "The pipeline output did not pass structured validation."
_CHUNK_COUNT_MESSAGE = (
    "The number of normalization responses does not match the number of input chunks."
)
_COMPLETENESS_MESSAGE = "The normalization result dropped or added entries relative to the input."
_CHANGED_ENTRY_MESSAGE = "The normalization result changed an entry timestamp or source kind."
_PROMPT_MISSING_MESSAGE = "The normalization prompt template is unavailable."
_CONTEXT_MISSING_MESSAGE = "The normalization run context is missing."


@dataclass(frozen=True)
class _RunContext:
    """Transient per-run state shared between request building, response
    parsing and rendering (see the concurrency note in the module docstring)."""

    document: OriginalDocument
    content_entries: list[ParsedEntry]


def _text_lines(value: str) -> list[str]:
    """Split rendered text into lines (mirrors the layer-1 renderer)."""
    return value.rstrip("\n").split("\n")


def _chunk_content(entries: list[ParsedEntry], chunk_size: int) -> list[list[ParsedEntry]]:
    """Split content entries into deterministic, order-preserving chunks."""
    return [entries[start : start + chunk_size] for start in range(0, len(entries), chunk_size)]


def extract_json_object(text: str) -> object:
    """Extract one JSON object from a model response.

    The expected output is strict JSON, but some endpoints wrap it in
    Markdown fences or add prose; the extraction therefore first tries the raw
    text and then falls back to the outermost ``{...}`` span. Raises
    :class:`ValueError` (static message, no content) when no JSON object can
    be found.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model response contains no JSON object.") from None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("The model response contains no JSON object.") from exc


class NormalizePipeline(Pipeline[NormalizeResult]):
    """The real normalization pipeline (registered in Task 13).

    Registered for :attr:`~diary.db.models.PipelineType.normalize` in the
    pipeline registry by :mod:`.registration`; the worker registers it at
    startup so real ``normalize`` jobs no longer fail with "not registered".
    """

    pipeline_type = PipelineType.normalize
    layer = ArtifactLayer.normalized
    output_ext = ".txt"
    task_type = TaskType.normalization
    result_schema = NormalizeResult
    #: The layer-2 header embeds the export row and the derived version, so the
    #: framework calls :meth:`render_artifact` with that context.
    render_requires_context = True

    def __init__(self) -> None:
        self._context: _RunContext | None = None

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request(self, *, export: WeeklyExport, input_artifact: Artifact | None) -> LLMRequest:
        """Build one request for the whole input (single-chunk convenience)."""
        return self.build_requests(export=export, input_artifact=input_artifact)[0]

    def build_requests(
        self, *, export: WeeklyExport, input_artifact: Artifact | None
    ) -> list[LLMRequest]:
        """Build one :class:`LLMRequest` per entry-chunk of the original export.

        Raises :class:`PipelineInputError` when the input is not the original
        export itself (layers 3-4 are out of scope), when the file cannot be
        parsed, or when the export carries no normalizable entries.
        """
        if input_artifact is not None:
            raise PipelineInputError("Normalization always consumes the original export directly.")
        document = self._load_document(export)
        content_entries = [entry for entry in document.entries if entry.text is not None]
        if not content_entries:
            raise PipelineInputError("The original export contains no entries to normalize.")
        self._context = _RunContext(document=document, content_entries=content_entries)
        chunk_size = get_settings().normalize_chunk_entries
        prompt = self._load_prompt()
        return [
            self._build_chunk_request(prompt, chunk)
            for chunk in _chunk_content(content_entries, chunk_size)
        ]

    def _build_chunk_request(self, prompt: str, chunk: list[ParsedEntry]) -> LLMRequest:
        """One request containing exactly ``chunk``'s entries, in order."""
        records = [
            {
                "index": entry.index,
                "timestamp": entry.timestamp,
                "kind": entry.kind.value,
                "text": entry.text,
            }
            for entry in chunk
        ]
        content = (
            f"{prompt}\n\n"
            f"Всего записей в блоке ниже: {len(chunk)}.\n"
            f"{ENTRIES_HEADER}\n"
            f"{json.dumps(records, ensure_ascii=False)}"
        )
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    def _load_document(self, export: WeeklyExport) -> OriginalDocument:
        """Read and parse the original export file (resolved by DB identity)."""
        path: Path = get_settings().data_root / export.path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineInputError(_INPUT_PARSE_MESSAGE) from exc
        try:
            return parse_original_text(text)
        except OriginalParseError as exc:
            raise PipelineInputError(_INPUT_PARSE_MESSAGE) from exc

    def _load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    # ------------------------------------------------------------------
    # Structured-output parsing and completeness validation
    # ------------------------------------------------------------------

    def parse_result(self, response: LLMResponse) -> NormalizeResult:
        """Validate one chunk response against the strict result schema.

        This parses and schema-validates only; the framework path always goes
        through :meth:`parse_responses`, which additionally runs the
        completeness check against the chunk the response belongs to.
        """
        return self._parse_chunk_response(response)

    def parse_responses(self, responses: list[LLMResponse]) -> NormalizeResult:
        """Parse every chunk response and merge them in chunk order.

        Each response is validated against the strict schema and then against
        its chunk: same count, same indices in the same order, and matching
        timestamps/kinds. A mismatch raises :class:`PipelineOutputError` with a
        static message; the framework fails the run and writes no file.
        """
        context = self._context
        if context is None:
            raise PipelineOutputError(_CONTEXT_MISSING_MESSAGE)
        chunk_size = get_settings().normalize_chunk_entries
        chunks = _chunk_content(context.content_entries, chunk_size)
        if len(responses) != len(chunks):
            raise PipelineOutputError(_CHUNK_COUNT_MESSAGE)
        merged: list[NormalizedEntry] = []
        for response, expected in zip(responses, chunks):
            parsed = self._parse_chunk_response(response)
            self._assert_chunk_complete(parsed, expected)
            merged.extend(parsed.entries)
        return NormalizeResult(entries=merged, entry_count=len(merged))

    @staticmethod
    def _parse_chunk_response(response: LLMResponse) -> NormalizeResult:
        """Parse one provider response into a validated :class:`NormalizeResult`."""
        try:
            payload = extract_json_object(response.text)
        except ValueError as exc:
            raise PipelineOutputError(_OUTPUT_INVALID_MESSAGE) from exc
        try:
            return NormalizeResult.model_validate(payload)
        except ValidationError as exc:
            raise PipelineOutputError(_OUTPUT_INVALID_MESSAGE) from exc

    @staticmethod
    def _assert_chunk_complete(parsed: NormalizeResult, expected: list[ParsedEntry]) -> None:
        """Verify the chunk output equals the chunk input one-to-one.

        A dropped/added record, a reordered record, an invented index, a
        changed timestamp or a changed source kind all fail the run — the
        invariants of Task 13 allow none of them.
        """
        if parsed.entry_count != len(expected) or len(parsed.entries) != len(expected):
            raise PipelineOutputError(_COMPLETENESS_MESSAGE)
        for output, entry in zip(parsed.entries, expected):
            if output.index != entry.index:
                raise PipelineOutputError(_COMPLETENESS_MESSAGE)
            if output.timestamp != entry.timestamp or output.kind != entry.kind.value:
                raise PipelineOutputError(_CHANGED_ENTRY_MESSAGE)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, structured: NormalizeResult) -> str:
        """Plain-render guard (this pipeline always renders with run context).

        The framework calls :meth:`render_artifact` because
        ``render_requires_context`` is set; reaching this method means the
        framework contract changed and must fail loudly, not produce a
        header-less artifact.
        """
        raise PipelinePermanentError("The normalization pipeline must render with run context.")

    def render_artifact(
        self, structured: NormalizeResult, *, export: WeeklyExport, version: int
    ) -> str:
        """Render the layer-2 TXT from validated data plus the export row.

        The header carries the lineage (layer label, week, sequence, revision,
        derived version and the original file name) so the artifact is
        self-describing; entries are emitted in the original file order.
        Untranscribed audio entries (which never reached the model) keep the
        ``[не расшифровано]`` marker.
        """
        context = self._context
        if context is None:
            raise PipelineOutputError(_CONTEXT_MISSING_MESSAGE)
        document = context.document
        normalized_by_index = {entry.index: entry for entry in structured.entries}
        content_indices = {entry.index for entry in context.content_entries}
        if set(normalized_by_index) != content_indices:
            raise PipelineOutputError(_COMPLETENESS_MESSAGE)

        settings = get_settings()
        created_local = utc_now().astimezone(settings.tzinfo)
        sunday = export.week_end_date
        original_filename = export_filename(sunday, export.seq_number, export.revision)
        lines: list[str] = [
            "# Дневник — нормализованный",
            f"Неделя: {week_monday(sunday).isoformat()} — {sunday.isoformat()}",
            f"Дата воскресенья: {sunday.isoformat()}",
            f"Номер выгрузки: {export.seq_number:04d}",
            f"Ревизия: r{export.revision}",
            f"Версия: v{version}",
            f"Оригинал: {original_filename}",
            f"Создано: {created_local.isoformat(timespec='seconds')}",
            f"Timezone: {settings.timezone}",
            "",
            "---",
            "",
        ]
        for position, entry in enumerate(document.entries):
            if position:
                lines.append("")
            if entry.text is None:
                entry_text = _UNTRANSCRIBED_MARKER
            else:
                entry_text = normalized_by_index[entry.index].normalized_text
            lines.append(f"[{entry.timestamp}] TEXT")
            lines.extend(_text_lines(entry_text))
        return "\n".join(lines) + "\n"


__all__ = [
    "ENTRIES_HEADER",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "NormalizePipeline",
    "extract_json_object",
]
