# Skill: testing

## Purpose

Правила тестирования в этом репозитории: размещение тестов, синтетические данные, mock-провайдеры, интеграционные сеты, end-to-end путь, критерии приёмки. Стратегия: `../docs/testing-strategy.md`.

## Scope and boundaries

- Входит: unit/integration/e2e-тесты, фикстуры, mock/fake-провайдеры, команды валидации.
- Вне границ: реальные модели/GPU/сеть в обычных прогонах (запрещены); реальные данные дневника в тестах (запрещены).

## Mandatory rules

- Размещение: unit — `../tests/unit/`, integration — `../tests/integration/`, фикстуры — `../tests/fixtures/`; e2e-сценарий — по решению Task 19.
- Каждая задача добавляет тесты вместе с кодом (AGENTS.md §13).
- Данные — только синтетические; реальные тексты/аудио дневника в Git-отслеживаемых тестах запрещены.
- STT/LLM в тестах — mock/fake-провайдеры: успех, таймаут, auth/rate-limit/network-ошибка, malformed JSON/структура.
- Интеграционные тесты используют изолированную тестовую БД (не продовую); миграции проверяются на чистой БД.
- Тестируются инварианты: неизменность оригиналов (байт-в-байт), атомарность артефактов, отсутствие секретов/дневника в логах и ответах, детерминизм выгрузок.
- Результаты команд валидации сообщаются дословно (passed/failed); не запущенная проверка не выдаётся за успех.

## Expected module locations

- `../tests/unit/`, `../tests/integration/`, `../tests/fixtures/`; conftest/helpers — по мере необходимости в этих каталогах.

## Quality and security checks

- Покрытие критичных сценариев по задачам — см. `../docs/testing-strategy.md` §4.
- e2e (Task 19): текстовое сообщение → аудио → мок-транскрибация → выгрузка → нормализация → саммари → wellbeing → проверка трассировки; плюс сценарии ошибок (failed STT/LLM, malformed JSON, двойной claim, re-export revision).

## Typical validation commands (целевые, после Task 01)

```bash
make test
make lint
make typecheck
```

## Links

- Документация: `../docs/testing-strategy.md`; `../docs/architecture-brief.md` §13.
- Задачи: `../tasklist.md` — все задачи (тесты в каждой), Task 19 (e2e и приёмка).
