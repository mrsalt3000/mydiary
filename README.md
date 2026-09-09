# Local Diary

Локальный (local-first) однопользовательский личный дневник для WSL 2: текстовые и голосовые сообщения, ручная транскрипция, еженедельные TXT-выгрузки и три производных слоя (нормализация, саммари, осторожная wellbeing-рефлексия) с независимой настройкой локальных или внешних моделей для каждой задачи.

## Статус репозитория

**Идёт последовательная реализация MVP (Task 01–19).**

Task 01 выполнен и принят (см. [`docs/acceptance-log.md`](docs/acceptance-log.md)): настроен Python-скелет (`src/diary/`) с FastAPI и `GET /health`, тесты, тулинг (pytest, Ruff, mypy, pre-commit), `Makefile`, `.gitignore`/`.env.example`. БД, выгрузки и LLM-конвейеры добавляются последующими задачами Task 02–19 строго по порядку.

## Документы

| Документ | Назначение |
|---|---|
| [`tasklist.md`](tasklist.md) | Авторитетный бэклог задач Task 01–19 и правила исполнения (не переписывается без запроса) |
| [`AGENTS.md`](AGENTS.md) | Операционный контракт для агентов: цикл работ, авторитетность, правила приватности/данных/кода |
| [`docs/architecture-brief.md`](docs/architecture-brief.md) | Полный архитектурный бриф: скоуп, топология, модули, данные, выгрузки, аудио/STT, LLM-конфигурация, конвейеры, надёжность, безопасность, тесты, вехи |
| [`docs/decisions/`](docs/decisions/README.md) | ADR-001…006: зафиксированные архитектурные решения |
| [`docs/execution-protocol.md`](docs/execution-protocol.md) | Протокол подтверждения задач, шаблоны, работа со сбоями валидации |
| [`docs/privacy-and-safety.md`](docs/privacy-and-safety.md) | Классификация данных, секреты, локальные/внешние провайдеры, логи, wellbeing-границы |
| [`docs/data-lineage.md`](docs/data-lineage.md) | Слои артефактов, ревизии/версии, хэши, имена файлов, прослеживаемость |
| [`docs/testing-strategy.md`](docs/testing-strategy.md) | Слои тестов, mock-провайдеры, фикстуры, команды |
| [`docs/wsl2-deployment-brief.md`](docs/wsl2-deployment-brief.md) | Допущения WSL 2, Docker/Compose (цель), GPU, локальные LLM-эндпоинты |
| [`docs/known-limitations.md`](docs/known-limitations.md) | Начальные ограничения и риски |
| [`skills/`](skills/README.md) | Исполнительные руководства по областям (9 навыков) |
| [`prompts/`](prompts/README.md) | Политика версионирования промптов (каталоги задач) |
| [`configs/`](configs/README.md) | Политика конфигурации (примеры, без секретов) |
| [`data/`](data/README.md) | Локальные/приватные данные: политика и структура каталогов |

## Структура каталогов

```text
.
├── tasklist.md          # бэклог задач (источник исполнения)
├── AGENTS.md            # контракт для агентов
├── README.md
├── docs/                # архитектура, ADR, политики
├── skills/              # SKILL.md по областям
├── prompts/             # версионируемые промпты (normalize/summarize/wellbeing)
├── configs/             # политика конфигурации (примеры)
├── src/                 # пакет src/diary/ (Task 01: FastAPI, /health)
├── tests/               # unit/ integration/ fixtures/ (Task 01: первые тесты)
├── data/                # приватные данные: db/ audio/ exports/ temp/ backups/
├── docker/              # будущие Dockerfile'ы (пусто)
└── scripts/             # будущие скрипты (пусто)
```

## Ближайшее действие

Идёт исполнение бэклога по порядку. Задачи ведутся автономно в multi-agent режиме (анализ и план утверждает ведущий; реализацию выполняет агент-разработчик; приёмку — агент-приёмщик с записью вердикта в `docs/acceptance-log.md`; коммиты ведущий выполняет после успешной приёмки), см. [`docs/execution-protocol.md`](docs/execution-protocol.md) §7. Текущий статус по задачам — в [`docs/acceptance-log.md`](docs/acceptance-log.md).

## Первый локальный запуск (WSL 2)

Предварительные условия: Python 3.12 (`~/.local/bin/python3.12`), `make`.

1. Установка окружения (создаёт `.venv/` и ставит пакет `diary` в режиме editable вместе с dev-зависимостями):

   ```bash
   make install
   ```

2. Запуск приложения (FastAPI + Uvicorn, localhost, авто-перезагрузка):

   ```bash
   make dev
   ```

3. Проверка работоспособности:

   ```bash
   curl -s http://127.0.0.1:8000/health
   # => {"status":"ok"}
   ```

   Интерактивная документация API: http://127.0.0.1:8000/docs (доступна только на localhost).

4. Качество кода:

   ```bash
   make test       # pytest (unit + integration)
   make lint       # ruff check + ruff format --check
   make typecheck  # mypy (проверка типов в src/)
   ```

5. (Опционально) git-хуки перед коммитами:

   ```bash
   .venv/bin/pre-commit install
   ```

Приватные данные (содержимое `data/`, файл `.env`) никогда не попадают в Git: см. `.gitignore`. Пример окружения — в `.env.example` (только синтетические значения).

## Транскрипция на CPU (локальный STT, Task 09)

Расшифровка голосовых сообщений выполняется **вручную** (кнопка «Расшифровать» / «Перерасшифровать» под сообщением), в фоне — durable-задачей воркера, и только на этой машине. Пакет `faster-whisper` — **опциональная** зависимость: он не ставится в обычном окружении и в CI.

1. Установить опциональный extra (native-зависимость + загрузка модели CTranslate2):

   ```bash
   .venv/bin/pip install -e ".[stt]"
   ```

2. Задать STT-профиль в конфигурации моделей: скопировать `configs/models.example.yaml` → `configs/models.yaml` (файл не коммитится) и при необходимости указать модель, язык, устройство и тип вычислений. Профиль `transcription` для CPU по умолчанию:

   ```yaml
   profiles:
     transcription:
       provider: faster_whisper
       model: small          # имя/размер модели Whisper
       language: ru          # язык распознавания (по умолчанию ru)
       device: cpu           # cpu или cuda (cuda — опционально, см. docs/wsl2-deployment-brief.md)
       compute_type: int8    # int8/int8_float16/int8_float32/float16/float32
       beam_size: 5
   ```

   Путь к файлу конфигурации по умолчанию — `configs/models.yaml`; переопределяется переменной окружения `MODELS_CONFIG`.

3. Запустить воркер (отдельный процесс; задачи переживают рестарт):

   ```bash
   .venv/bin/python -m diary.workers.runner --poll-interval 2
   ```

   или обработать все ожидающие задачи однократно: `--once`.

4. Нажать «Расшифровать» у голосового сообщения в чате. При первом использовании faster-whisper скачает выбранную модель в локальный кэш (разовая операция, дальше всё работает офлайн). raw-текст сохраняется в `transcriptions` и не перезаписывает аудио; повторная расшифровка добавляет новую версию, старая остаётся в истории.

Примечания: CPU-fallback — устройство `cpu` с `compute_type: int8` (значения по умолчанию); GPU (CUDA под WSL 2) — опциональная настройка вне требований MVP. Если `faster-whisper` не установлен, задача расшифровки завершится с понятной ошибкой, не затрагивая аудио и сообщение.
