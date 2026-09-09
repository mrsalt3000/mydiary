# Skill: sqlite-data-model

## Purpose

Правила работы со слоем персистентности: SQLAlchemy-модели, Alembic-миграции, репозитории, транзакции, индексы, конвенции времени. SQLite — источник истины (ADR-001).

## Scope and boundaries

- Входит: модели таблиц (Task 04), миграции, репозитории/сервисы персистентности, настройка соединения (WAL/busy timeout), транзакционные границы.
- Вне границ: HTTP-слой (backend-fastapi), файловые операции артефактов (domain-сервисы, см. weekly-export/llm-pipelines).

## Mandatory rules

- Схема меняется только Alembic-миграцией; свежая БД создаётся только миграциями; повторяемость на чистом окружении обязательна.
- Модели: стабильные ID (UUID), timezone-aware временные метки единым способом (UTC в БД), FK и индексы (время сообщений, статус задач, неделя выгрузки, происхождение артефактов), enum/check-constraint'ы для статусов.
- Хэши (SHA-256) для файлов/артефактов — поля в БД; JSON-метаданные — только для гибких метаданных.
- Доступ из кода — через репозитории; сессии не используются вне слоя персистентности.
- `model_configs` в MVP не персистится: профили остаются файловыми (Task 03/04), решение документируется.
- Интеграционные тесты — на изолированной тестовой БД, не продовой.
- Миграции и тесты не содержат пользовательских данных и секретов.

## Expected module locations

- `../src/diary/db/` — модели, репозитории, engine/session, интеграция Alembic.
- Каталог миграций Alembic — определяется в Task 04 (ожидаемо внутри `src/diary/db/` или `migrations/`).
- Тесты: `../tests/integration/`.

## Quality and security checks

- Миграция upgrade/downgrade повторяема; репозитории покрыты интеграционными тестами.
- Конкурентный доступ web + worker к SQLite: WAL, busy timeout, транзакционные claim'ы (см. также Task 08) проверяются тестами.
- Временные метки и границы недель тестируются с фиксированным `APP_TIMEZONE`.

## Typical validation commands (целевые, после Task 01/04)

```bash
make migrate
make test
make lint
```

## Links

- ADR: `../docs/decisions/ADR-001.md`, `../docs/decisions/ADR-003.md`, `../docs/decisions/ADR-005.md`.
- Документация: `../docs/data-lineage.md`, `../docs/architecture-brief.md` §5.
- Задачи: `../tasklist.md` — Task 04 (фундамент и миграции), Task 05 (сообщения), Task 08 (jobs).
