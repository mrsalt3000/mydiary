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
| Task 15 — Implement safe wellbeing reflection pipeline (layer 4) | not_started | 0 | — |
| Task 16 — Build exports and processing management UI | not_started | 0 | — |
| Task 17 — Docker Compose and WSL 2 deployment | not_started | 0 | — |
| Task 18 — Backups, recovery, and local security hardening | not_started | 0 | — |
| Task 19 — End-to-end tests, observability, and acceptance review | not_started | 0 | — |

## История возвратов

(Пусто — заполняется агентами-приёмщиками при вердиктах `rework`.)
