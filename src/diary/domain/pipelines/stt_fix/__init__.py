"""STT-correction layer — ``stt_corrected`` (phase 2, task T04).

The layer turns one raw diary message (a voice message with raw STT text, or a
text message with obvious typos) into the same text with recognition errors
fixed, and stores the validated result under the message anchor. Its
definition of done (``docs/tasklist-phase2.md`` §6, T04) is a pair of vetoes:

* **it adds no facts** — no name, number, event or intention that the input
  does not contain, and no reordering or dropping of what it does contain;
* **doubt is expressed only through ``uncertain[]``** — the corrected ``text``
  carries no marker of any kind (the literal ``[?]`` of the original brief is
  not part of the contract; ADR-007 §3). A fragment inside ``text`` that the
  correction could not resolve is copied verbatim into ``uncertain``; the
  contract rejects an entry that is not a substring of ``text``, so the two can
  never drift apart.

The message the layer reads is **immutable input**: the layer never writes to
``messages``/``transcriptions`` — its only write path is the append-only result
store of task T08 (``diary.db.results.save_layer_result``), which appends a new
version for the (message, layer) pair on every rerun (ADR-005).

Modules
=======

* :mod:`.pipeline` — :class:`SttFixPipeline`: the versioned prompt template, the
  request built from one raw entry, and the strict parse of the completion into
  the :class:`~diary.domain.contracts.SttCorrection` contract (envelope fields
  supplied by the backend);
* :mod:`.service` — :func:`correct_stt` / :func:`run_stt_fix`: one correction
  run over one entry, stored through ``save_layer_result``;
* :mod:`.registration` — guarded wiring of the layer into the pipeline
  registry, mirroring the normalization/summarization/wellbeing registrations.

Privacy: the message text enters an in-memory request/contract only. It is
never logged, and no error message, run metadata or stored payload contains
anything but the validated correction (AGENTS.md §7-8, §15).
"""

from diary.domain.pipelines.stt_fix.pipeline import (
    ENTRY_HEADER,
    PROFILE_NAME,
    PROMPT_PATH,
    PROMPT_VERSION,
    SttFixPipeline,
)
from diary.domain.pipelines.stt_fix.registration import ensure_stt_fix_pipeline_registered
from diary.domain.pipelines.stt_fix.service import (
    SttFixOutcome,
    correct_stt,
    run_stt_fix,
)

__all__ = [
    "ENTRY_HEADER",
    "PROFILE_NAME",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SttFixOutcome",
    "SttFixPipeline",
    "correct_stt",
    "ensure_stt_fix_pipeline_registered",
    "run_stt_fix",
]
