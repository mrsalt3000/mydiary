# Реестр архитектурных решений (ADR)

## Конвенции

- Каждое архитектурное решение фиксируется отдельным файлом `ADR-NNN-<slug>.md`.
- Формат обязателен:

```markdown
# ADR-NNN: Title

- Status: Accepted
- Date: YYYY-MM-DD

## Context
...

## Decision
...

## Consequences
...

## Alternatives considered
...
```

- Статусы: `Accepted` (принято), `Proposed` (предложено), `Deprecated`/`Superseded by ADR-NNN` (заменено). Новые решения начинаются как `Proposed` и получают `Accepted` после явного утверждения пользователем.
- Изменение принятого решения — только через новый ADR (`Superseded by …`), не правкой старого текста задним числом.
- Решение обязательно связывается с задачами `../tasklist.md`, где оно исполняется.
- Детальная архитектура: `../architecture-brief.md`.

## Индекс

| ADR | Решение | Статус | Дата |
|---|---|---|---|
| [ADR-001](ADR-001-sqlite-source-of-truth.md) | SQLite — локальный источник истины для метаданных и состояния | Accepted | 2026-09-09 |
| [ADR-002](ADR-002-local-files-for-audio-and-artifacts.md) | Локальная ФС для аудио и артефактов; в БД — метаданные | Accepted | 2026-09-09 |
| [ADR-003](ADR-003-sqlite-job-queue-for-mvp.md) | Durable-очередь задач на SQLite для MVP | Accepted | 2026-09-09 |
| [ADR-004](ADR-004-openai-compatible-llm-abstraction.md) | Провайдер-нейтральная OpenAI-совместимая LLM-абстракция | Accepted | 2026-09-09 |
| [ADR-005](ADR-005-immutable-originals-and-versioned-artifacts.md) | Первичные данные неизменяемы; производные артефакты версионируются | Accepted | 2026-09-09 |
| [ADR-006](ADR-006-wellbeing-safety-boundaries.md) | Wellbeing-вывод — поддерживающая рефлексия, не диагноз; детерминированная эскалация | Accepted | 2026-09-09 |
| [ADR-007](ADR-007-second-anchor-family-for-derivational-layers.md) | Вторая семья якорей происхождения для производных слоёв фазы 2 (вариант B) | Accepted | 2026-09-19 |
| [ADR-008](ADR-008-second-anchor-family-persistence.md) | Персистентность второй семьи якорей в `artifacts` (дельта схемы под вариант B) | Accepted | 2026-09-19 |
| [ADR-009](ADR-009-optional-reasoning-effort.md) | Опциональное поле `reasoning_effort` у LLM-профиля и запроса (T02b) | Accepted | 2026-09-19 |
