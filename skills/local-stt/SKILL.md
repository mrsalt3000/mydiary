# Skill: local-stt

## Purpose

Правила аудио-функциональности и локальной транскрипции: безопасная загрузка/хранение аудио, ручной запуск STT через durable-задачи, интерфейс провайдера транскрибации, происхождение raw-результатов, границы CPU/GPU.

## Scope and boundaries

- Входит: приём аудио (MediaRecorder/upload), валидация, хранение под серверными именами, метаданные, жизненный цикл статуса транскрипции, `TranscriptionProvider` и локальные адаптеры, транскрибация как задача.
- Вне границ: LLM-конвейеры (llm-pipelines), очередь как механизм (см. Task 08, sqlite-data-model), рендер в UI (frontend-htmx).

## Mandatory rules

- Транскрибация — только вручную (явная кнопка Transcribe/Retranscribe); автозапуска нет; HTTP-запрос не блокируется — создаётся durable job (`transcribe_audio`).
- Аудио: серверная валидация MIME/размера; пути `data/audio/YYYY/MM/<uuid>.<ext>`; пользовательские имена/пути запрещены (path traversal).
- Метаданные (MIME, размер, SHA-256, путь, время, длительность где надёжно) и `messages(kind=audio)` создаются согласованно; при сбое после записи файла — безопасная очистка.
- Первичный статус аудио — `not_requested`; при сбое транскрибации аудио и сообщение не теряются/не повреждаются; повторная транскрибация — новая запись результата, старая не стирается.
- Интерфейс `TranscriptionProvider`: профиль задачи (Task 03), параметры язык/device/compute/model/beam; по умолчанию `ru`, `cpu`; CUDA — опционально.
- raw-текст хранится отдельно (`transcriptions`), не заменяя аудио и текст сообщения; отображается под аудио-сообщением.
- В CI/тестах — fake-провайдер, без GPU и скачивания моделей.
- Временные/промежуточные файлы аудио — в `data/temp/`; содержимое аудио не логируется.

## Expected module locations

- `../src/diary/domain/audio/` — приём/хранение и абстракции транскрибации; `../src/diary/domain/chat/` — связь сообщений.
- `../src/diary/workers/` — исполнение задачи; `../src/diary/core/` — конфигурация STT.
- Таблицы `audio_files`, `transcriptions`, `jobs` — модель в `../src/diary/db/`.
- Тесты: `../tests/integration/` (upload/consistency), unit для статусов.

## Quality and security checks

- Тесты: валидация загрузки; консистентность файл↔БД; отсутствие автотранскрибации; сбой транскрибации не портит аудио/сообщение; история ретранскрибаций сохраняется.
- Проверка невозможности path traversal (передача пользовательских имён/путей отклоняется).

## Typical validation commands (целевые, после Task 01/09)

```bash
make test
make lint
python -m diary.workers.runner --help
```

## Links

- ADR: `../docs/decisions/ADR-001.md`, `../docs/decisions/ADR-002.md`, `../docs/decisions/ADR-003.md`.
- Документация: `../docs/architecture-brief.md` §8; `../docs/privacy-and-safety.md` §5.
- Задачи: `../tasklist.md` — Task 07 (приём/хранение), Task 08 (очередь), Task 09 (STT).
