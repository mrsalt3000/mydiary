"""STT-correction pipeline (layer ``stt_corrected``; task T04).

Responsibilities
================

* Take **one raw entry** (:class:`~diary.domain.contracts.RawEntry`) — the
  message text or its raw STT transcript — and build exactly one
  :class:`LLMRequest` from the versioned prompt template
  ``prompts/stt_fix/v1.md`` plus that entry's text. The entry object is read
  only: nothing here rewrites the message, its audio or its raw transcript.
* Validate the completion against the T00 contract
  :class:`~diary.domain.contracts.SttCorrection` through the shared strict
  bridge (:mod:`diary.domain.llm.structured`): the response must be one JSON
  object whose ``uncertain`` fragments are verbatim substrings of its ``text``.
  Anything else is an :class:`~diary.domain.llm.errors.LLMStructuredOutputError`
  — a failed run with no stored artifact (AGENTS.md §15).
* Fill the **envelope** fields of the contract itself, never from the model
  (ADR-007 §7): ``raw_id`` is the identity of the entry that was corrected,
  ``model`` is the model of the resolved ``diary_stt`` profile and
  ``prompt_ver`` is this module's prompt version. A completion that reports its
  own ``model``/``prompt_ver`` cannot change the stored provenance — the
  bridge overwrites those keys with the backend values.

Storage is not this module's business: :mod:`.service` hands the validated
correction to the append-only result store of task T08 under the ``raw_id``
anchor. The layer renders no file: its result is stored inline as validated
JSON (ADR-008 §1), so ``output_ext`` exists only for registry consistency.

Privacy: the entry text is in-memory request/contract content only; it is never
logged, and every error summary is static (AGENTS.md §7-8).
"""

from __future__ import annotations

import json

from diary.core.models_config import OpenAICompatibleProfile
from diary.core.settings import PROJECT_ROOT
from diary.db.models import ArtifactLayer, PipelineType
from diary.domain.contracts import RawEntry, SttCorrection
from diary.domain.llm.models import LLMRequest, LLMResponse
from diary.domain.llm.structured import parse_contract_response
from diary.domain.pipelines.base import PipelinePermanentError, SecondFamilyLayer

#: Versioned prompt template of this layer (``prompts/stt_fix/v1.md``).
PROMPT_VERSION = "v1"
#: Prompt template location: ``prompts/stt_fix/v1.md``.
PROMPT_PATH = PROJECT_ROOT / "prompts" / "stt_fix" / f"{PROMPT_VERSION}.md"
#: Phase-2 profile field of ``configs/models.yaml`` this layer runs under.
PROFILE_NAME = "diary_stt"
#: Line that introduces the entry being corrected inside a request.
ENTRY_HEADER = "Входное сообщение (JSON, одна запись):"

#: Static, non-sensitive error summary (AGENTS.md §8).
_PROMPT_MISSING_MESSAGE = "The stt_fix prompt template is unavailable."


class SttFixPipeline(SecondFamilyLayer[SttCorrection]):
    """The T04 STT-correction layer (second anchor family, one message per run)."""

    pipeline_type = PipelineType.stt_corrected
    layer = ArtifactLayer.stt_corrected
    output_ext = ".txt"
    profile_name = PROFILE_NAME
    result_schema = SttCorrection

    def load_prompt(self) -> str:
        """The versioned prompt template text (static, no diary content)."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelinePermanentError(_PROMPT_MISSING_MESSAGE) from exc

    def build_request(self, *, entry: RawEntry) -> LLMRequest:
        """One request containing the prompt and exactly this entry's text.

        The entry is passed as a JSON object with a single ``text`` key: the
        model is asked to return the corrected text, so its identity is not
        part of the payload (the backend already knows it) and cannot be echoed
        back as a fact.
        """
        prompt = self.load_prompt()
        payload = json.dumps({"text": entry.text}, ensure_ascii=False)
        content = f"{prompt}\n\n{ENTRY_HEADER}\n{payload}"
        return LLMRequest.from_prompt(content, prompt_version=PROMPT_VERSION)

    def parse_result(
        self,
        response: LLMResponse,
        *,
        entry: RawEntry,
        profile: OpenAICompatibleProfile,
    ) -> SttCorrection:
        """Validate one completion into the :class:`SttCorrection` contract.

        The envelope is built here, from the backend's own knowledge: the
        corrected entry's identity and the resolved profile/prompt version.
        """
        envelope: dict[str, object] = {
            "raw_id": str(entry.id),
            "model": profile.model,
            "prompt_ver": PROMPT_VERSION,
        }
        return parse_contract_response(response, SttCorrection, envelope=envelope)


__all__ = [
    "ENTRY_HEADER",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SttFixPipeline",
]
