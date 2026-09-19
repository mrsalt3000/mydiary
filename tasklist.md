# Local Diary — Task List for Qwen Code

## Purpose

Build a local-first personal diary application in Python for WSL 2. The user can:

- Create and keep text diary messages.
- Record or upload voice messages.
- Start speech-to-text manually for each voice message.
- Store the chat history and metadata in SQLite.
- Export a calendar week into a structured TXT file.
- Normalize exported original texts with an LLM.
- Create concise weekly summaries from normalized texts.
- Run a concise, fact-grounded wellbeing reflection over normalized texts.
- Configure a different STT/LLM provider and model for each task, including local and commercial OpenAI-compatible endpoints.

This document is the execution contract for the coding agent. Tasks must be completed **strictly in order**.

---

## Mandatory agent workflow

### Work one task at a time

The agent must never begin a task until the preceding task is accepted by the user.

For every task, follow this exact loop:

1. **Read the task completely** and inspect the relevant existing repository files.
2. **Analyze** the task before making changes:
   - State the objective.
   - Identify affected modules, files, APIs, data models, tests, and documentation.
   - Identify dependencies on earlier tasks.
   - Identify technical risks, data/privacy implications, and assumptions.
   - State whether the task requires a migration, configuration change, dependency addition, or documentation update.
3. **Propose an implementation plan**:
   - Give a short, concrete list of intended edits.
   - Name files expected to be created or changed.
   - Name tests and validation commands to run.
   - Mention any decision that would materially affect architecture or user data.
4. **Stop and request explicit confirmation.**
   - Do not edit files.
   - Do not run code-generation commands that change repository state.
   - Do not install packages.
   - Do not create migrations.
5. Only after the user explicitly confirms, **implement exactly the approved plan**.
6. **Validate** the implementation:
   - Run the task-specific tests.
   - Run formatting/linting/type checks where applicable.
   - Report commands run and their results.
7. **Report completion**:
   - Summarize changed files.
   - Summarize behavior implemented.
   - State known limitations or deviations.
   - Confirm whether all acceptance criteria passed.
   - Do not begin the next task automatically.

### Confirmation template

Before implementation, the agent should use this form:

```markdown
## Task N — Analysis

### Objective
...

### Repository impact
- Files to create/change: ...
- Dependencies/configuration: ...
- Data/API impact: ...
- Risks and assumptions: ...

## Proposed plan
1. ...
2. ...
3. ...

## Validation
- `...`
- `...`

Proceed with this plan?
```

### Completion template

```markdown
## Task N — Complete

### Implemented
- ...

### Files changed
- `...`

### Validation
- `command` — passed/failed

### Notes
- ...

Acceptance criteria: passed / partially passed / failed.
```

---

## Global engineering rules

### Source of truth and data lineage

- SQLite is the source of truth for messages, audio metadata, transcription metadata, exports, pipeline runs, and jobs.
- Audio files are stored on the local filesystem; SQLite stores their metadata and paths.
- Original user text, original audio, and raw STT output are immutable source data.
- No LLM pipeline may overwrite original messages, audio, raw transcription, or original weekly exports.
- Every derived artifact must be traceable to its precise input artifact/export and processing run.
- Re-running an LLM pipeline creates a new result version. It does not overwrite an earlier derived result.

### Privacy and secrets

- Never put API keys in source code, Git-tracked YAML, SQLite, logs, test fixtures, screenshots, or error messages.
- Configuration files may reference a secret only by environment-variable name.
- Do not log diary text, prompts containing diary text, audio bytes, transcription text, or LLM responses at INFO/WARNING/ERROR level.
- An external LLM endpoint may receive diary content only when explicitly enabled in configuration and clearly disclosed in the UI/docs.
- Bind the application to localhost by default.

### Code quality

- Use Python 3.12+.
- Use `src/` layout.
- Use type hints for production Python code.
- Keep FastAPI routers thin; put application logic into services/domain modules.
- Keep database access in repositories or a defined persistence layer.
- Use Pydantic models for API input/output and structured LLM responses.
- Use Alembic for every schema change after the initial database migration setup.
- Add tests together with functional code.
- Keep test data synthetic; never commit real diary content or real audio.
- Use atomic writes for artifacts: temporary path, flush/fsync as applicable, then rename.

### Repository conventions

Recommended target structure:

```text
local-diary/
├── src/
│   └── diary/
│       ├── api/
│       ├── core/
│       ├── db/
│       ├── domain/
│       │   ├── audio/
│       │   ├── chat/
│       │   ├── exports/
│       │   ├── llm/
│       │   └── pipelines/
│       ├── workers/
│       ├── templates/
│       └── static/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── prompts/
│   ├── normalize/
│   ├── summarize/
│   └── wellbeing/
├── configs/
├── docs/
│   └── decisions/
├── skills/
├── data/
│   ├── db/
│   ├── audio/
│   ├── exports/
│   │   ├── 01_original/
│   │   ├── 02_normalized/
│   │   ├── 03_summaries/
│   │   └── 04_wellbeing/
│   ├── temp/
│   └── backups/
├── docker/
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── Makefile
└── README.md
```

### Definition of done for every task

A task is done only when all applicable conditions are met:

- The approved implementation plan is complete.
- New behavior has automated tests.
- Existing relevant tests still pass.
- Formatting, linting, and type checks pass when configured.
- Documentation/config examples are updated if behavior is user-visible or configurable.
- No secrets or diary content were introduced into Git-tracked files or logs.
- The completion report includes validation results.

---

## Product decisions fixed for MVP

These decisions are baseline requirements unless changed by a later explicit decision task.

- Single local user.
- Primary language: Russian; STT language is configurable and defaults to `ru`.
- Default application timezone: `Europe/Moscow`, configurable through `APP_TIMEZONE`.
- A weekly period is Monday 00:00:00 through Sunday 23:59:59.999999 in the configured timezone.
- The date included in a weekly file name is the Sunday date of that week.
- Empty weeks do not produce an export.
- Weekly export sequence numbers are assigned only to actual original exports and have no gaps.
- Original export sequence number is reused in names of downstream artifacts to make lineage obvious.
- A re-export of a previously exported week requires an explicit user action to create a new original revision; it must not silently overwrite a prior original export.
- LLM analysis and normalization are always user-triggered; no automatic cloud calls.
- Wellbeing analysis is a supportive reflection based only on documented facts. It is not diagnosis, treatment, or crisis care.

---

# Task sequence

## Task 01 — Initialize repository and engineering baseline

### Objective

Create a runnable, maintainable Python project skeleton using `src` layout and establish quality tooling.

### Scope

- Initialize `pyproject.toml`.
- Add Python package structure under `src/diary/`.
- Add FastAPI application entrypoint and `/health` endpoint.
- Add initial dependencies for FastAPI, Uvicorn, Pydantic Settings, SQLAlchemy, Alembic, Jinja2, HTTPX, PyYAML, and timezone support.
- Add developer dependencies: pytest, pytest-asyncio, Ruff, mypy or Pyright, pre-commit.
- Add initial tests for importability and `/health`.
- Add `.gitignore`, `.env.example`, `README.md`, `Makefile`, and pre-commit configuration.
- Ensure logs are structured and do not contain request bodies by default.

### Acceptance criteria

- `make dev` starts the web application in WSL 2.
- `GET /health` returns a successful response with a simple JSON payload.
- `make test` passes.
- `make lint` passes.
- The project imports from `src/` rather than the repository root.
- README documents the first local launch workflow.

### Expected validation

```bash
make test
make lint
python -m uvicorn diary.main:app --host 127.0.0.1 --port 8000
```

---

## Task 02 — Write architecture, privacy, and decision records

### Objective

Document the system before persistence and pipelines are implemented.

### Scope

- Create `docs/architecture.md`.
- Create `docs/privacy-and-safety.md`.
- Create initial ADRs in `docs/decisions/` for:
  - SQLite as source of truth.
  - Local filesystem for audio/artifacts.
  - SQLite-backed worker queue for MVP.
  - OpenAI-compatible abstraction for LLM providers.
  - Immutable original data and versioned derived artifacts.
- Document data flow from browser through SQLite/filesystem/worker to each of four layers.
- Document local versus external processing and explicit user consent implications.
- Document the limitations and safety position of wellbeing analysis.

### Acceptance criteria

- Documentation states exact definitions for original, normalized, summary, and wellbeing artifact layers.
- Documentation explains why original data is immutable.
- Documentation describes external-provider data exposure clearly.
- No implementation code is required beyond links or comments needed for docs.

### Expected validation

```bash
make lint
```

---

## Task 03 — Implement application settings and model configuration

### Objective

Create validated configuration for application behavior, storage paths, timezone, and per-task model profiles.

### Scope

- Implement application settings with Pydantic Settings.
- Support `.env` and a YAML model configuration file.
- Create `configs/app.example.yaml` and `configs/models.example.yaml`.
- Implement configuration models for these task types:
  - `transcription`
  - `normalization`
  - `summarization`
  - `wellbeing_analysis`
- Support a local `faster_whisper` transcription profile.
- Support an `openai_compatible` LLM profile with `base_url`, `model`, `api_key_env`, timeout, temperature, and token limits.
- Resolve secrets only from environment variables.
- Create a safe diagnostic endpoint or internal service that confirms profile availability without returning API keys.

### Acceptance criteria

- Four task profiles can use different models/providers.
- Missing required secrets fail with a clear, non-sensitive error.
- Configuration values are validated at startup or explicit validation time.
- No secret value appears in serialized config, logs, or API responses.
- Unit tests cover valid and invalid configurations.

### Expected validation

```bash
make test
make lint
```

---

## Task 04 — Create SQLite persistence foundation and migrations

### Objective

Create the SQLite database foundation, SQLAlchemy setup, Alembic setup, and core tables.

### Scope

Implement migrations and SQLAlchemy models for:

- `messages`
- `audio_files`
- `transcriptions`
- `weekly_exports`
- `processing_runs`
- `artifacts`
- `jobs`
- `model_configs` if persisted runtime profiles are required; otherwise document why profiles remain file-based for MVP.

Minimum design requirements:

- UUID or otherwise stable IDs.
- Timezone-aware timestamps stored consistently.
- Foreign keys and appropriate indexes.
- Status enums/check constraints where practical.
- Hash fields for source files/artifacts.
- JSON metadata field only for genuinely flexible metadata, not core relational fields.
- Repository interfaces/services for basic persistence operations.

### Acceptance criteria

- A fresh database can be created solely through Alembic migrations.
- Migration upgrade is repeatable on a fresh environment.
- Core entities can be inserted and read through repositories.
- Indexes exist for message time, job status, export week, and artifact lineage.
- Integration tests use an isolated test database.

### Expected validation

```bash
make migrate
make test
make lint
```

---

## Task 05 — Implement text diary messages and chat history API

### Objective

Provide the first complete user value path: create and read text diary messages stored in SQLite.

### Scope

- Define request/response Pydantic schemas.
- Implement message creation service for `kind=text`.
- Implement chronological history retrieval with pagination or cursor-based loading.
- Apply the configured application timezone when displaying timestamps.
- Define and implement a minimal deletion policy; use soft delete if messages must remain traceable.
- Add API routes under a versioned or clearly grouped router.
- Add tests for validation, ordering, timezone behavior, and persistence.

### Acceptance criteria

- A text message can be created through API.
- After restart, the message remains available.
- History is returned in deterministic chronological order.
- Empty message bodies and oversized payloads are rejected.
- Deleted messages follow the documented policy.

### Expected validation

```bash
make test
make lint
```

---

## Task 06 — Implement initial chat UI

### Objective

Create a minimal, usable local web interface for text diary messaging.

### Scope

- Add base template and chat page.
- Use Jinja2 with HTMX and minimal JavaScript where appropriate.
- Render chat history safely; escape all user text.
- Add text input and submit action.
- Display message timestamps in configured timezone.
- Add empty state, loading state, and user-friendly validation error state.
- Add progressive loading of older messages if history is paginated.
- Keep styling minimal, local, and dependency-light.

### Acceptance criteria

- A user can open the localhost page, add text, refresh, and still see messages.
- The user does not need to call API manually.
- HTML rendering is safe against script injection.
- Browser interaction is covered by route/template tests where practical.

### Expected validation

```bash
make test
make lint
```

---

## Task 07 — Implement audio ingestion and storage

### Objective

Allow the user to record/upload voice messages and persist audio locally without automatic transcription.

### Scope

- Add browser recording using `MediaRecorder`.
- Add file-upload fallback.
- Define accepted MIME types and configurable maximum upload size.
- Validate uploads server-side.
- Store audio under `data/audio/YYYY/MM/` using server-generated UUID names, never user-provided file names.
- Capture metadata: original MIME type, size, hash, storage path, creation time, and duration where reliable.
- Create an `audio_files` record and a linked `messages(kind=audio)` record.
- Add audio playback in chat.
- Set initial transcription status to `not_requested`.
- Implement safe cleanup when persistence fails after file write.

### Acceptance criteria

- Recorded/uploaded audio remains playable after browser refresh and application restart.
- Audio filename/path traversal attempts are impossible.
- No transcription is started automatically.
- Unsupported/oversized files produce clear UI/API errors.
- Tests cover upload validation and filesystem/database consistency.

### Expected validation

```bash
make test
make lint
```

---

## Task 08 — Implement durable SQLite-backed job queue and worker

### Objective

Create the background-task mechanism required by STT and LLM pipelines without introducing Redis/Celery in MVP.

### Scope

- Implement a durable `jobs` table and job repository.
- Define job types, initially including `transcribe_audio`.
- Define statuses: `pending`, `running`, `succeeded`, `failed`, `retrying`, `cancelled`.
- Implement atomic claim/lock logic suitable for one or more local workers.
- Implement configurable retry count and exponential backoff.
- Implement idempotency keys.
- Implement worker runner CLI/module.
- Add safe exception handling and non-sensitive error summaries.
- Add API/UI status polling support.
- Add Compose service definition only if Docker setup already exists; otherwise defer container wiring to Task 17.

### Acceptance criteria

- Jobs survive application/worker restart.
- A job cannot be processed concurrently by two workers.
- A failed transient job can retry according to policy.
- A permanently failed job exposes a status and safe error summary.
- Unit and integration tests cover claiming, retries, idempotency, and restart behavior.

### Expected validation

```bash
make test
make lint
python -m diary.workers.runner --help
```

---

## Task 09 — Implement local speech-to-text pipeline

### Objective

Perform manual local transcription of voice messages through a provider abstraction.

### Scope

- Define `TranscriptionProvider` protocol/interface.
- Implement a `FasterWhisperProvider` or equivalent local adapter.
- Read task-specific STT configuration from Task 03.
- Support configurable device (`cpu`/`cuda`), compute type, model, language, beam settings where applicable.
- Add a manual `Transcribe` action for one voice message.
- Create a `transcribe_audio` job rather than processing in the HTTP request.
- Persist raw output in `transcriptions`; do not overwrite original audio or message.
- Update transcription status lifecycle.
- Render raw transcription below the audio message.
- Add `Retranscribe` action that explicitly creates a new transcription run/result using the selected profile.
- Provide CPU fallback documentation.

### Acceptance criteria

- Clicking Transcribe queues work and does not block the browser request.
- The raw transcription is persisted and visible after reload.
- A failed transcription does not lose audio or corrupt message state.
- Retranscription does not erase prior raw result history.
- Tests use a fake provider and do not require a GPU/model download in CI.

### Expected validation

```bash
make test
make lint
```

---

## Task 10 — Define weekly export rules and implement original TXT export

### Objective

Create deterministic, versioned exports of diary messages for a selected calendar week.

### Scope

- Implement timezone-aware week-boundary calculation.
- Interpret each week as Monday through Sunday.
- Use Sunday date in filenames.
- Export all messages whose creation time belongs to the selected local week.
- Include text entries exactly as originally entered.
- Include audio metadata and raw STT text if a transcription exists.
- Explicitly mark audio entries that have not been transcribed.
- Implement original artifact naming:

```text
YYYY-MM-DD_original_NNNN_rN.txt
```

Where:

- `YYYY-MM-DD` is the Sunday date.
- `NNNN` is the global sequence number of actual original exports, with no gaps.
- `rN` is the revision for the same week/export lineage; first revision is `r1`.

- Create the first export only when the week is non-empty.
- Create a new original revision only through an explicit re-export/revision action.
- Store the artifact path, hash, week end date, sequence number, revision, and relationships in SQLite.
- Use atomic filesystem writes.
- Provide download route/UI.

### Required original TXT format

```text
# Дневник — оригинал
Неделя: YYYY-MM-DD — YYYY-MM-DD
Дата воскресенья: YYYY-MM-DD
Номер выгрузки: NNNN
Ревизия: rN
Создано: ISO-8601 timestamp
Timezone: IANA timezone

---

[YYYY-MM-DD HH:MM] TEXT
Original user text

[YYYY-MM-DD HH:MM] AUDIO
Файл: relative/safe/path
STT raw:
Raw transcription or [не расшифровано]
```

### Acceptance criteria

- Week boundaries are correct around Sunday/Monday and configured timezone.
- Empty weeks do not create a file or consume a sequence number.
- Gaps between active weeks do not create numbering gaps.
- Re-export creates an explicit new revision rather than overwriting earlier file.
- Output ordering is deterministic.
- Export file and database row are consistent after success/failure.
- Tests cover timezone, numbering, empty week, re-export revision, and absent STT.

### Expected validation

```bash
make test
make lint
```

---

## Task 11 — Implement reusable OpenAI-compatible LLM client

### Objective

Create one safe, configurable LLM abstraction for local and commercial model endpoints.

### Scope

- Define `LLMProvider` protocol/interface.
- Implement `OpenAICompatibleProvider` using HTTPX.
- Support endpoint base URL, model name, API key environment variable, timeout, retries, temperature, and max tokens.
- Resolve profiles by task type through Task 03 configuration.
- Define normalized request and response objects independent of provider implementation.
- Capture processing provenance: provider type, model, prompt version, timestamps, status, token usage when returned, input hash, and error class.
- Never persist API keys.
- Never log full diary content/prompt/completion.
- Add mock/fake LLM provider for tests.
- Validate network error, timeout, authentication error, rate-limit error, and malformed response behavior.

### Acceptance criteria

- Normalization, summarization, and wellbeing tasks can point to different OpenAI-compatible endpoints/models by configuration alone.
- API keys remain absent from logs, database, and API output.
- Failure modes produce safe job failures and are retryable according to policy.
- Tests run without calling a real external endpoint.

### Expected validation

```bash
make test
make lint
```

---

## Task 12 — Implement LLM pipeline framework and artifact versioning

### Objective

Create shared orchestration for derived-artifact pipelines before implementing individual prompts.

### Scope

- Define pipeline interface/base class.
- Create shared processing-run lifecycle: queued, running, succeeded, failed.
- Implement input artifact resolution by database identity rather than uncontrolled file globs.
- Implement derived artifact naming convention:

```text
YYYY-MM-DD_normalized_NNNN_rN_vM.txt
YYYY-MM-DD_summary_NNNN_rN_vM.md
YYYY-MM-DD_wellbeing_NNNN_rN_vM.md
```

Where:

- `NNNN` is the original export sequence number.
- `rN` is the original export revision.
- `vM` is the version of the derived result for that pipeline/input.

- Implement atomic derived artifact writing.
- Create `processing_runs` and `artifacts` records with source relationships and hashes.
- Implement pipeline task enqueueing and status retrieval.
- Enforce that source original files are never modified.
- Add generic UI/API endpoint to list pipeline runs/artifacts for an export.

### Acceptance criteria

- A pipeline result can always be traced to a source original export and exact processing run.
- Re-run creates `v2` rather than replacing `v1`.
- Invalid/missing input artifact fails safely without file changes.
- Tests cover lineage, artifact naming, version increments, and atomic-write failures.

### Expected validation

```bash
make test
make lint
```

---

## Task 13 — Implement normalization pipeline (layer 2)

### Objective

Produce a cleaned, consistent diary document from an original weekly export without altering factual meaning.

### Scope

- Create `prompts/normalize/v1.md`.
- Add a strict Pydantic schema for structured LLM output.
- Use the task-specific `normalization` LLM profile.
- Process original exports selected explicitly by artifact ID or user action.
- Require the model to preserve entry order, timestamps, source types, and all material facts.
- Permit only:
  - obvious STT correction;
  - removal of clear speech disfluencies and transcription artifacts;
  - punctuation/orthography normalization;
  - consistent formatting.
- Require uncertain phrases to remain unchanged or be marked `[неразборчиво]`; the LLM must not invent a correction.
- Let backend code render the final TXT from validated structured data.
- Validate that input entries are not silently dropped.
- Support chunking for long original exports while preserving order and lineage.
- Add one-file execution and batch processing of eligible original exports.
- Add UI view to compare original and normalized output.

### Normalization prompt invariants

The prompt must explicitly prohibit:

- adding facts, motivations, diagnoses, or emotional interpretations;
- deleting meaningful information;
- changing chronology/timestamps;
- literary rewriting of the user’s voice;
- claiming certainty when a STT phrase is ambiguous.

### Acceptance criteria

- The pipeline creates a second-layer TXT file linked to the exact original export.
- The original export remains byte-for-byte unchanged.
- LLM output must pass Pydantic validation before a file is written.
- Invalid output becomes a failed processing run, not a partial/corrupted artifact.
- Tests cover success, malformed JSON, missing entry, chunked input, and rerun versioning.

### Expected validation

```bash
make test
make lint
```

---

## Task 14 — Implement summarization pipeline (layer 3)

### Objective

Create a concise, factual weekly summary from a normalized document.

### Scope

- Create `prompts/summarize/v1.md`.
- Use only normalized artifacts as inputs.
- Define a strict structured summary schema with these sections:
  - key events;
  - work and load;
  - wellbeing/emotions only when explicitly supported by source;
  - supportive factors;
  - open questions/unknowns.
- Require source-grounded language; do not introduce facts not present in the normalized source.
- Implement single-pass processing for inputs below a configured context threshold.
- Implement map-reduce summarization above the threshold.
- Record intermediate map outputs as run metadata or controlled temporary artifacts, without exposing user content in logs.
- Render final result as Markdown in `03_summaries/`.
- Support user-triggered single and batch runs.
- Render/read/download summaries in the UI.

### Acceptance criteria

- Summary is generated only from a normalized input artifact.
- Long documents use deterministic chunking and final aggregation.
- The output identifies the exact normalized source artifact.
- Re-run creates a new summary version.
- Tests cover single-pass, map-reduce selection, malformed response, and lineage.

### Expected validation

```bash
make test
make lint
```

---

## Task 15 — Implement safe wellbeing reflection pipeline (layer 4)

### Objective

Generate a short, cautious reflection based on facts in normalized diary entries, with safety constraints and no medical diagnosis.

### Scope

- Create `prompts/wellbeing/v1.md`.
- Use only normalized artifacts as inputs.
- Define a strict structured schema with sections:
  - observed facts;
  - cautious interpretation;
  - one to three low-risk practical recommendations;
  - confidence and limitations.
- Explicitly distinguish quoted/observed facts from interpretation.
- Explicitly prohibit diagnoses, treatment plans, invented symptoms, certainty about causation, and unsupported claims.
- Ensure recommendations are concise, actionable, and tied to observed facts where possible.
- Implement a safety classifier/rule layer with levels:
  - `normal`;
  - `needs_support`;
  - `urgent`.
- Implement deterministic crisis fallback UI/content for `urgent`; it must not depend on unconstrained LLM generation.
- Require explicit user action and confirmation in the UI before the analysis job is created.
- Provide an application setting that disables wellbeing analysis entirely.
- Render final report as Markdown in `04_wellbeing/`.

### Required user-facing limitation

Every wellbeing report must state, in substance:

> This reflection is based only on the diary records available for this period. It is not a medical or psychological diagnosis and does not replace qualified professional help.

### Acceptance criteria

- No analysis is launched automatically.
- Result is explicitly grounded in observed diary facts.
- Output contains no diagnostic label or fabricated evidence.
- Urgent-risk flow shows deterministic escalation guidance and records safe minimal metadata.
- Pipeline can be disabled by configuration.
- Tests cover normal, needs-support, urgent, malformed LLM output, and disabled state.

### Expected validation

```bash
make test
make lint
```

---

## Task 16 — Build exports and processing management UI

### Objective

Expose the artifact workflow through a usable local interface.

### Scope

- Add an Exports page listing original weekly exports.
- Display Sunday date, week period, sequence number, revision, creation date, statuses, and available derived artifacts.
- Provide explicit buttons/actions for:
  - export week;
  - create new revision of original export;
  - normalize;
  - summarize;
  - wellbeing reflection;
  - retry failed job;
  - download an artifact.
- Provide a Processing page with current/recent jobs and safe error summaries.
- Provide a detail view for an export showing lineage across all layers.
- Add polling/refresh behavior for asynchronous statuses.
- Add clear consent warning before any configured external LLM processing.
- Add clear confirmation before wellbeing reflection.
- Do not display secrets, full internal stack traces, or raw provider error payloads.

### Acceptance criteria

- A user can complete the four-layer flow in a browser.
- Actions visibly show queued/running/succeeded/failed states.
- Original vs normalized and normalized vs summary/wellbeing relationships are understandable.
- All destructive or revision-creating actions require an appropriate explicit UI confirmation.

### Expected validation

```bash
make test
make lint
```

---

## Task 17 — Docker Compose and WSL 2 deployment

### Objective

Make local deployment reproducible in WSL 2 for CPU and optional NVIDIA GPU configurations.

### Scope

- Create application and worker Dockerfiles as appropriate.
- Create `docker-compose.yml` with web and worker services.
- Persist application data using named volumes or documented bind mounts.
- Keep application bind restricted to localhost by default.
- Add configurable connection path for a local LLM endpoint:
  - compose service name when model is in Compose;
  - `host.docker.internal` when model runs on host where applicable;
  - documented WSL-specific alternatives if needed.
- Add CPU profile/default.
- Add optional NVIDIA GPU profile for STT where host environment supports it.
- Add `.dockerignore`.
- Document setup for Docker Desktop/WSL integration, FFmpeg, optional NVIDIA/CUDA, and local model endpoint.
- Ensure database and artifact directories survive `docker compose down` unless volumes are intentionally removed.

### Acceptance criteria

- A documented command starts web and worker services.
- Data persists across container recreation.
- CPU-only setup works without GPU-specific dependencies.
- GPU use is opt-in and documented, not required for MVP.
- Compose configuration does not expose the service publicly by default.

### Expected validation

```bash
docker compose config
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:8000/health
```

---

## Task 18 — Backups, recovery, and local security hardening

### Objective

Protect the user’s personal diary data and provide a clear recovery path.

### Scope

- Implement a backup command/service that captures SQLite and selected data directories.
- Support backup modes:
  - database plus text artifacts;
  - full backup including audio.
- Use timestamped archive naming and integrity metadata.
- Document recovery procedure.
- Implement recovery as an explicit CLI command with a mandatory confirmation flag.
- Add retention/cleanup policy for `data/temp/` only; never automatically delete source audio or artifacts.
- Validate all user-controlled file identifiers/paths.
- Keep localhost-only defaults.
- Add optional simple local access protection/PIN only if it does not destabilize the MVP; otherwise document it as post-MVP.
- Add a privacy disclosure view that identifies whether each task profile is local or external.

### Acceptance criteria

- A backup can be created and inspected without modifying source data.
- Recovery process is documented and guarded against accidental overwrite.
- Temporary files are safely cleaned according to policy.
- Path traversal and unsafe file access attempts are rejected.
- Tests cover backup manifest creation and path validation.

### Expected validation

```bash
make test
make lint
```

---

## Task 19 — End-to-end tests, observability, and acceptance review

### Objective

Verify the MVP as a coherent local system and document its operational limits.

### Scope

- Add request IDs and job IDs to structured logs.
- Add `/ready` endpoint if dependency checks make it meaningful.
- Add safe operational metrics in logs: job duration, status, model identifier, token counts where available, file sizes, but never diary content.
- Create synthetic fixtures representing text messages, voice metadata, raw STT, and mock LLM results.
- Build end-to-end test flow:
  1. Create text message.
  2. Upload audio message.
  3. Queue and complete mocked transcription.
  4. Export week.
  5. Normalize export.
  6. Summarize normalized export.
  7. Run wellbeing reflection.
  8. Verify full artifact lineage.
- Test error cases for failed STT, failed LLM, malformed LLM JSON, duplicate worker claims, and re-export revision.
- Write `docs/known-limitations.md`.
- Update README with MVP capabilities, architecture overview, configuration, and operational commands.
- Perform and document a manual acceptance checklist for WSL 2.

### Acceptance criteria

- Entire four-layer workflow passes in automated test using mocks.
- Primary data remains intact after each simulated pipeline failure.
- Artifact lineage is verifiable from UI/API/database.
- Logs are useful without exposing diary content/secrets.
- Documentation is sufficient for a clean local installation.

### Expected validation

```bash
make test
make lint
make typecheck
```

---

# Post-MVP backlog

These items must not be started without a new explicit task list or user confirmation.

- SQLite FTS5 full-text search over messages and normalized artifacts.
- Manual tags for sleep, energy, work, family, health, and mood.
- Monthly and quarterly summaries derived from weekly summaries.
- Trend analysis with source references and uncertainty indicators.
- Encryption at rest for SQLite, audio, and artifact files.
- Local authentication/PIN and multi-device access.
- Native mobile application.
- Multi-user support.
- Speaker diarization.
- RAG/vector database semantic search across long diary history.
- Calendar integration.
- Notifications/reminders.
- Fine-tuning on diary data.
- Kubernetes, Redis, Celery, Kafka, or distributed processing.

Second-phase backlog (T00–T12: second anchor family, layer 5 `coach_note`, golden eval set) — [`docs/tasklist-phase2.md`](docs/tasklist-phase2.md); started only by an explicit user decision (see ADR-007).

---

# Final MVP acceptance checklist

Before declaring the project MVP-complete, verify all items below.

- [ ] Text diary messages are saved and restored from SQLite.
- [ ] Voice messages are recorded/uploaded and retained locally.
- [ ] Transcription starts only through an explicit user action.
- [ ] Raw STT remains distinct from original audio and later normalized text.
- [ ] Weekly original exports are deterministic, revisioned, and sequentially numbered without gaps for inactive weeks.
- [ ] Original exports are never changed by downstream processing.
- [ ] Normalization produces a separate, versioned second-layer artifact.
- [ ] Summarization consumes only normalized artifacts and produces a separate third-layer artifact.
- [ ] Wellbeing reflection consumes only normalized artifacts, is manual, concise, fact-grounded, and non-diagnostic.
- [ ] Urgent-risk handling uses deterministic safety behavior.
- [ ] Each AI task can use a separate configurable model/provider.
- [ ] API keys do not appear in source, logs, SQLite, or UI.
- [ ] The application runs locally in WSL 2, with reproducible setup documentation.
- [ ] Backups can be created and recovery is documented.
- [ ] Automated tests cover the end-to-end flow and failure safety.

