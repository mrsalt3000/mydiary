# Журнал приёмки задач (acceptance log)

Ведётся **агентом-приёмщиком** после каждого цикла разработки в multi-agent режиме (решение пользователя от 2026-09-09; см. `docs/execution-protocol.md` §7).

## Правила ведения

- **Статусы**: `not_started` — задача не начата; `in_work` — разработчик реализует; `rework` — возвращена на доработку; `accepted` — приёмка пройдена.
- **Отправок на доработку**: число возвратов задачи разработчику (0 — принята с первой проверки).
- **Причины возвратов**: хронологический список `r1`, `r2`, … — краткая причина каждого возврата с указанием, какой критерий/проверка не прошла.
- Запись в файл — единственное изменение в репозитории, которое выполняет агент-приёмщик.
- Содержимое дневника и секреты в журнал не попадают; причины — технические формулировки.

## Реестр задач

| Задача | Статус | Отправок на доработку | Причины возвратов |
|---|---|---|---|
| Task 01 — Initialize repository and engineering baseline | accepted | 0 | — |
| Task 02 — Write architecture, privacy, and decision records | accepted | 0 | — |
| Task 03 — Implement application settings and model configuration | accepted | 0 | — |
| Task 04 — Create SQLite persistence foundation and migrations | accepted | 0 | — |
| Task 05 — Implement text diary messages and chat history API | accepted | 0 | — |
| Task 06 — Implement initial chat UI | accepted | 0 | — |
| Task 07 — Implement audio ingestion and storage | accepted | 0 | — |
| Task 08 — Implement durable SQLite-backed job queue and worker | accepted | 0 | — |
| Task 09 — Implement local speech-to-text pipeline | accepted | 0 | — |
| Task 10 — Define weekly export rules and implement original TXT export | accepted | 0 | — |
| Task 11 — Implement reusable OpenAI-compatible LLM client | accepted | 0 | — |
| Task 12 — Implement LLM pipeline framework and artifact versioning | accepted | 0 | — |
| Task 13 — Implement normalization pipeline (layer 2) | accepted | 0 | — |
| Task 14 — Implement summarization pipeline (layer 3) | accepted | 0 | — |
| Task 15 — Implement safe wellbeing reflection pipeline (layer 4) | accepted | 0 | — |
| Task 16 — Build exports and processing management UI | accepted | 1 | r1 |
| Task 17 — Docker Compose and WSL 2 deployment | accepted | 0 | — |
| Task 18 — Backups, recovery, and local security hardening | accepted | 0 | — |
| Task 19 — End-to-end tests, observability, and acceptance review | accepted | 0 | — |

## История возвратов

- **r1 (Task 16)** — Wellbeing-действие на страницах `/exports` и `/exports/{id}` неработоспособно из браузера: `static/js/exports.js` (`startPipeline`) отправляет на `POST /api/v1/pipelines/wellbeing/runs` тело `{"export_id": ...}` без поля `confirm`, которое схема `WellbeingRunCreate` требует как обязательное (`confirm: bool` без default). Сервер отвечает `422 Field required` (а не `409`, как декларирует docstring схемы), в клиентской карте ошибок wellbeing нет обработки 422 → пользователь всегда видит общий текст ошибки; полноценный четырёхслойный флоу (export → normalize → summarize → wellbeing) из UI Task 16 невыполним. Рабочий запуск существует только на wellbeing-странице (`app.js` шлёт `confirm: true`), а интеграционные тесты обходят JS прямым вызовом API с `confirm: true` и дефект не ловят. Ожидается: клиент после отметки чекбокса подтверждения шлёт `confirm: true` (и/или `confirm: bool = False` в схеме, чтобы отсутствие поля давало серверный 409), плюс тест тела запроса/отображения ошибки. Прочие критерии Task 16 (страницы, retry-семантика, экранирование, external-consent, поллинг, приватность, состояние репозитория) проверены и прошли.

## Фаза 2 (бэклог T00–T12) — реестр задач

| Задача | Статус | Отправок на доработку | Причины возвратов |
|---|---|---|---|
| T00 — Контракты + 10 golden-чанков RU | accepted | 0 | — |

## История приёмок фазы 2

- **T00 (2026-09-19)** — `accepted`, 0 отправок на доработку. Скоуп совпадает с утверждённым (6 изменённых + 9 групп новых файлов; лишних нет, `docs/tasklist-phase2.md` §6 фиксирует изменения первой фазы). Независимый повторный прогон: `make lint` (ruff check + format --check, 184 файла) — passed; `make typecheck` (mypy, 79 файлов) — passed; `make test` — 685 passed; `generate_contract_schemas.py --check` — passed (5 файлов); `pytest tests/unit/test_contracts_schemas.py tests/unit/test_golden_manifest.py -q` — 127 passed. Схемы воспроизведены во временный каталог и `diff` — байт-в-байт идентичны; негативная ветка `--check` (испорченная копия) — exit 1. Golden: ровно 10 уникальных чанков, 3/3/2/2, `synthetic: true`, `lang: "ru"`, `chunk_chars`/`chunk_tokens_est` сходятся с фактическим входом, `num_ctx` по правилу 8192/16384. Приватность: privacy-тест не скипается (сканирует `data/README.md` и `data/db/diary.db`, где 9 сообщений и 43 транскрипции) и падает на искусственно подсаженном пересечении во временном каталоге — проверка работоспособна. Решения 1–10 выполнены фактически. Иммутабельность: HEAD по-прежнему `1ac2d20`, ничего не застейджено, нет правок `pipelines/`, `db/`, `alembic/`, `configs/`, `prompts/`, `pyproject.toml`; новых зависимостей (включая `jsonschema`) нет; `configs/models.yaml` не коммитится (`.gitignore`). Валидаторы сверх буквы плана (`raw_entry` tz-aware/безопасный `audio_path`; непустые элементы; ISO-уникальные `daily_ids`) признаны допустимым кодированием требуемых инвариантов. Не блокирующие наблюдения: `experiment_7d` требует цифру «7» (ложное падение при честной формулировке словами); privacy-скан не покрывает `*.db-wal`/`*.db-shm`; ветка `num_ctx=16384` не представлена ни одним чанком (`chunks_planned == 1` у всех).
