"""Summarization pipeline (layer 3; Task 14, architecture brief §10.2).

Responsibilities
================

* Consume **only** a normalized artifact (layer 2) resolved by database
  identity. The original export (layer 1) is never read: the framework
  resolves the export's latest succeeded ``normalized`` artifact through the
  pipeline's ``input_layer`` declaration, and the pipeline raises a permanent,
  content-free error when the export has no normalized result yet
  ("run the upstream pipeline first").
* Read the layer-2 TXT and split it into deterministic, order-preserving
  entries (``[YYYY-MM-DD HH:MM] TEXT`` blocks). Below the configured context
  threshold the whole entry text goes to the model in **one single-pass
  request**; above it the entries are chunked deterministically (whole entries
  only, never split mid-entry; ``AppSettings.summarize_chunk_size`` bounds a
  chunk in characters) and **map-reduce** is used: one request per chunk
  (``Режим: частичный, блок i из n``), each answered with a partial
  :class:`SummaryResult` of the *same schema*.
* REDUCE is a **deterministic in-memory merge** inside
  :meth:`parse_responses`: the validated partial summaries are concatenated
  section by section in chunk order (exact duplicates removed) into one final
  :class:`SummaryResult`. No reduce LLM round-trip is needed and **no
  intermediate content is ever persisted**: partial summaries are transient
  run state (like normalization's chunk merging, Task 13) and never reach
  ``metadata_json``, run metadata, temp files or logs (AGENTS.md §7-8).
* Validate every response against the strict :class:`SummaryResult` schema
  (extra keys rejected) before anything is written; a malformed response
  fails the run permanently with no file.
* Render the final layer-3 Markdown **from the validated structured data**.
  The header embeds the lineage — week, sequence ``NNNN``, revision ``rN``,
  derived version ``vM`` and the exact normalized source file name — so the
  output always identifies its layer-2 source. Bullets are single-line
  (whitespace collapsed), so diary-derived text can never inject markdown
  structure.

Privacy: diary content and model output are in-memory request/structured
objects only; error messages are static (AGENTS.md §7-8, §15).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from diary.core.models_config import TaskType
from diary.core.settings import PROJECT_ROOT, get_settings
from diary.db.models import Artifact, ArtifactLayer, PipelineType, WeeklyExport
from diary.domain.exports.weeks import week_monday
from diary.domain.llm.models import LLMRequest, LLMResponse, utc_now
from diary.domain.pipelines.base import (
    Pipeline,
    PipelineInputError,
    PipelineOutputError,
    PipelinePermanentError,
)
from diary.domain.pipelines.normalize.pipeline import extract_json_object
from diary.domain.pipelines.summarize.schema import SummaryResult, WellbeingSection

#: Versioned prompt template of this pipeline (recorded in run provenance).
PROMPT_VERSION = "v1"
#: Prompt template location: ``prompts/summarize/v1.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "summarize" / f"{PROMPT_VERSION}.md"

#: Mode line of a single-pass request (the whole document follows below).
FINAL_MODE_LINE = "Режим: финальный (весь нормализованный документ)."
#: Prefix of a map (partial) mode line; the block number and total follow.
PARTIAL_MODE_PREFIX = "Режим: частичный, блок "

#: Static, non-sensitive error summaries (AGENTS.md §8).
_INPUT_READ_MESSAGE = "The normalized source file cannot be read."
_INPUT_PARSE_MESSAGE = "The normalized source cannot be parsed into entries."
_INPUT_EMPTY_MESSAGE = "The normalized source contains no entries to summarize."
_OUTPUT_INVALID_MESSAGE = "The pipeline output did not pass structured validation."
_RESPONSE_COUNT_MESSAGE = (
    "The number of summarization responses does not match the number of input chunks."
)
_PROMPT_MISSING_MESSAGE = "The summarization prompt template is unavailable."
_CONTEXT_MISSING_MESSAGE = "The summarization run context is missing."

#: Rendered marker for a section without source-grounded content.
_NO_DATA_MARKER = "— нет данных —"
#: Rendered marker for the wellbeing section without explicit source support.
_NO_WELLBEING_DATA_MARKER = "Явных данных о самочувствии и эмоциях в источнике нет."

#: Entry header line of a layer-2 document: ``[YYYY-MM-DD HH:MM] TEXT``.
_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] TEXT$")
#: Line that ends the layer-2 header block.
_HEADER_SEPARATOR = "---"


@dataclass(frozen=True)
class _RunContext:
    """Transient per-run state shared between request building, response
    parsing and rendering (see the concurrency note in the module docstring of
    the normalization pipeline, mirrored here)."""

    #: The exact normalized input artifact of this run (for the header link).
    input_artifact: Artifact
    #: Rendered request text per provider call, in call order.
    chunk_texts: list[str]


def split_normalized_entries(text: str) -> list[str]:
    """Split a layer-2 document into its ordered entry texts.

    The header block ends at the first ``---`` line; every following
    ``[YYYY-MM-DD HH:MM] TEXT`` line starts an entry that runs until the next
    entry header (or the end of the document). Interior blank lines and the
    trailing separator blank are handled so entries keep their content and
    re-joining with ``"\\n\\n"`` reproduces the source layout. Raises
    :class:`ValueError` (static message) when the document has no header
    separator or no entries.
    """
    lines = text.split("\n")
    try:
        separator_index = lines.index(_HEADER_SEPARATOR)
    except ValueError as exc:
        raise ValueError(_INPUT_PARSE_MESSAGE) from exc
    content = lines[separator_index + 1 :]
    starts = [index for index, line in enumerate(content) if _ENTRY_START_RE.match(line)]
    if not starts:
        raise ValueError(_INPUT_PARSE_MESSAGE)
    entries: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(content)
        entry = "\n".join(content[start:end]).rstrip("\n")
        entries.append(entry)
    return entries


def chunk_entries_by_chars(entries: list[str], chunk_size: int) -> list[list[str]]:
    """Split ``entries`` into deterministic, order-preserving character chunks.

    Entries are never split: an entry is appended to the current chunk while
    the chunk stays at most ``chunk_size`` characters (measured as
    ``len(entry) + 2`` for the ``"\\n\\n"`` re-joining separator); a single
    entry larger than ``chunk_size`` forms a chunk of its own. The result is
    deterministic for a given entry list.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for entry in entries:
        entry_size = len(entry) + 2
        if current and current_size + entry_size > chunk_size:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += entry_size
    if current:
        chunks.append(current)
    return chunks


def merge_partials(partials: list[SummaryResult]) -> SummaryResult:
    """Deterministic reduce: merge partial chunk summaries into one result.

    Each section is concatenated in chunk order with exact duplicates removed
    (a repeated phrasing across chunks must not be double-counted). The
    wellbeing section is supported when *any* partial reported explicit source
    support; without support its points are dropped. The merged object always
    satisfies the strict schema — partials were already validated.
    """
    if not partials:
        raise PipelineOutputError("The summarization run produced no partial results.")

    def merged_section(getter) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for partial in partials:
            for item in getter(partial):
                if item not in seen:
                    seen.add(item)
                    merged.append(item)
        return merged

    wellbeing_supported = any(partial.wellbeing.explicit_support for partial in partials)
    wellbeing_points = merged_section(lambda partial: partial.wellbeing.points)
    if not wellbeing_supported:
        wellbeing_points = []
    return SummaryResult(
        key_events=merged_section(lambda partial: partial.key_events),
        work_and_load=merged_section(lambda partial: partial.work_and_load),
        wellbeing=WellbeingSection(explicit_support=wellbeing_supported, points=wellbeing_points),
        supportive_factors=merged_section(lambda partial: partial.supportive_factors),
        open_questions=merged_section(lambda partial: partial.open_questions),
    )


class SummarizePipeline(Pipeline[SummaryResult]):
    """The real summarization pipeline (registered in Task 14).

    Registered for :attr:`~diary.db.models.PipelineType.summarize` in the
    pipeline registry by :mod:`.registration`; the worker registers it at
    startup so real ``summarize`` jobs no longer fail with "not registered".
    """

    pipeline_type = PipelineType.summarize
    layer = ArtifactLayer.summary
    output_ext = ".md"
    task_type = TaskType.summarization
    result_schema = SummaryResult
    #: The layer-3 header embeds the export row and the derived version, so the
    #: framework calls :meth:`render_artifact` with that context.
    render_requires_context = True
    #: Layer 3 is built only from layer 2 (normalized), never the original.
    input_layer = ArtifactLayer.normalized

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
        """Build one request for the whole document or one per entry-chunk.

        The input must be a normalized artifact (layer 2) — the framework
        resolves it for this pipeline. Documents whose entry text fits the
        configured context threshold produce a single single-pass request;
        longer documents are chunked deterministically and produce one partial
        (map) request per chunk, in order.
        """
        if input_artifact is None:
            raise PipelineInputError(
                "Summarization always consumes a normalized artifact (layer 2)."
            )
        text = self._load_normalized_text(input_artifact)
        try:
            entries = split_normalized_entries(text)
        except ValueError as exc:
            raise PipelineInputError(_INPUT_PARSE_MESSAGE) from exc
        if not entries:
            raise PipelineInputError(_INPUT_EMPTY_MESSAGE)

        settings = get_settings()
        total_size = sum(len(entry) + 2 for entry in entries)
        prompt = self._load_prompt()
        if total_size <= settings.summarize_mapreduce_threshold:
            document = "\n\n".join(entries)
            self._context = _RunContext(input_artifact=input_artifact, chunk_texts=[document])
            return [self._build_request(prompt, mode_line=FINAL_MODE_LINE, text=document)]

        chunk_size = settings.summarize_chunk_size
        chunks = chunk_entries_by_chars(entries, chunk_size)
        self._context = _RunContext(
            input_artifact=input_artifact,
            chunk_texts=["\n\n".join(chunk) for chunk in chunks],
        )
        return [
            self._build_request(
                prompt,
                mode_line=f"{PARTIAL_MODE_PREFIX}{index} из {len(chunks)}.",
                text=chunk_text,
            )
            for index, chunk_text in enumerate(self._context.chunk_texts, start=1)
        ]

    @staticmethod
    def _build_request(prompt: str, *, mode_line: str, text: str) -> LLMRequest:
        """One request: static prompt + mode line + the (chunk) entry text."""
        content = f"{prompt}\n\n{mode_line}\n\n{text}"
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    def _load_normalized_text(self, input_artifact: Artifact) -> str:
        """Read the normalized input file (resolved by DB identity)."""
        assert input_artifact.path is not None  # a layer-2 artifact is file-backed
        path: Path = get_settings().data_root / input_artifact.path
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineInputError(_INPUT_READ_MESSAGE) from exc

    def _load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    # ------------------------------------------------------------------
    # Structured-output parsing and deterministic reduce
    # ------------------------------------------------------------------

    def parse_result(self, response: LLMResponse) -> SummaryResult:
        """Validate one response against the strict summary schema."""
        return self._parse_response(response)

    def parse_responses(self, responses: list[LLMResponse]) -> SummaryResult:
        """Parse every response and reduce it to one validated result.

        A single-pass run yields exactly one response that is validated as the
        final result. A map-reduce run must return one response per built
        chunk (deterministic order); every response is validated as a partial
        :class:`SummaryResult` and the partials are merged deterministically
        in memory — nothing intermediate is persisted anywhere.
        """
        context = self._context
        if context is None:
            raise PipelineOutputError(_CONTEXT_MISSING_MESSAGE)
        if len(context.chunk_texts) == 1:
            if len(responses) != 1:
                raise PipelineOutputError(_RESPONSE_COUNT_MESSAGE)
            return self._parse_response(responses[0])
        if len(responses) != len(context.chunk_texts):
            raise PipelineOutputError(_RESPONSE_COUNT_MESSAGE)
        partials = [self._parse_response(response) for response in responses]
        return merge_partials(partials)

    @staticmethod
    def _parse_response(response: LLMResponse) -> SummaryResult:
        """Parse one provider response into a validated :class:`SummaryResult`."""
        try:
            payload = extract_json_object(response.text)
        except ValueError as exc:
            raise PipelineOutputError(_OUTPUT_INVALID_MESSAGE) from exc
        try:
            return SummaryResult.model_validate(payload)
        except ValidationError as exc:
            raise PipelineOutputError(_OUTPUT_INVALID_MESSAGE) from exc

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, structured: SummaryResult) -> str:
        """Plain-render guard (this pipeline always renders with run context).

        The framework calls :meth:`render_artifact` because
        ``render_requires_context`` is set; reaching this method means the
        framework contract changed and must fail loudly, not produce a
        header-less artifact.
        """
        raise PipelinePermanentError("The summarization pipeline must render with run context.")

    def render_artifact(
        self, structured: SummaryResult, *, export: WeeklyExport, version: int
    ) -> str:
        """Render the layer-3 Markdown from validated data plus the export row.

        The header embeds the week lineage and, critically, the exact
        normalized source file name, so the summary always identifies the
        layer-2 artifact it was built from (data-lineage.md §6).
        """
        context = self._context
        if context is None:
            raise PipelineOutputError(_CONTEXT_MISSING_MESSAGE)
        settings = get_settings()
        created_local = utc_now().astimezone(settings.tzinfo)
        sunday = export.week_end_date
        assert context.input_artifact.path is not None  # a layer-2 artifact is file-backed
        source_filename = context.input_artifact.path.rsplit("/", 1)[-1]
        lines: list[str] = [
            "# Саммари недели",
            f"Неделя: {week_monday(sunday).isoformat()} — {sunday.isoformat()}",
            f"Дата воскресенья: {sunday.isoformat()}",
            f"Номер выгрузки: {export.seq_number:04d}",
            f"Ревизия: r{export.revision}",
            f"Версия: v{version}",
            f"Источник (нормализованный): {source_filename}",
            f"Создано: {created_local.isoformat(timespec='seconds')}",
            f"Timezone: {settings.timezone}",
            "",
            "---",
            "",
        ]
        lines.extend(self._render_section("Ключевые события", structured.key_events))
        lines.extend(self._render_section("Работа и нагрузка", structured.work_and_load))
        if structured.wellbeing.explicit_support:
            lines.extend(self._render_section("Самочувствие и эмоции", structured.wellbeing.points))
        else:
            lines.extend(
                [
                    "## Самочувствие и эмоции",
                    "",
                    _NO_WELLBEING_DATA_MARKER,
                ]
            )
        lines.extend(self._render_section("Поддерживающие факторы", structured.supportive_factors))
        lines.extend(
            self._render_section("Открытые вопросы и неизвестное", structured.open_questions)
        )
        return "\n".join(lines).rstrip("\n") + "\n"

    @staticmethod
    def _render_section(title: str, items: list[str]) -> list[str]:
        """One ``##`` section: a single bullet per item, one line each.

        Item text is whitespace-collapsed so diary text can never smuggle
        blank lines or markdown structure into the rendered bullet list.
        """
        lines = [f"## {title}", ""]
        if items:
            for item in items:
                collapsed = " ".join(item.split())
                if collapsed:
                    lines.append(f"- {collapsed}")
            if len(lines) == 2:
                lines.append(_NO_DATA_MARKER)
        else:
            lines.append(_NO_DATA_MARKER)
        return lines


__all__ = [
    "FINAL_MODE_LINE",
    "PARTIAL_MODE_PREFIX",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SummarizePipeline",
    "chunk_entries_by_chars",
    "merge_partials",
    "split_normalized_entries",
]
