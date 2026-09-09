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
├── docker-compose.yml   # Compose: web + worker (+ опциональные STT/GPU-воркеры)
├── docker/              # Dockerfile (Task 17)
├── .dockerignore        # исключения build-контекста (Task 17)
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

## Запуск в Docker (WSL 2, Task 17)

Альтернативный способ запуска — контейнеры через Docker Desktop с WSL 2 integration (или Docker Engine внутри WSL 2). Образ один (`docker/Dockerfile`): из него запускаются `web`, `worker` и одноразовая миграция; опциональные STT-воркеры собираются из того же Dockerfile с build-флагами. Приложение публикуется только на `127.0.0.1:8000`, наружу ничего не открыто.

### Предварительные условия

- Docker Desktop (WSL 2 backend) или Docker Engine внутри WSL 2; `docker compose` доступен.
- Репозиторий и его `data/` — на файловой системе WSL (ext4), не на `/mnt/c` (производительность SQLite).
- UID владельца данных = 1000 (стандартный первый пользователь WSL 2). Если ваш UID другой — см. «UID/GID владельца» ниже.
- LLM-задачи (нормализация/саммаризация/wellbeing) требуют реального `configs/models.yaml` (см. пример `configs/models.example.yaml`); файл подключается в контейнер bind-mount'ом и в образ не зашивается.

### Быстрый старт (CPU, без локальной транскрипции)

```bash
# в корне репозитория
docker compose up -d --build
docker compose ps                 # migrate (exited 0) + web + worker
curl -s http://127.0.0.1:8000/health   # => {"status":"ok"}
```

- Схема БД применяется одноразовым сервисом `migrate` (`alembic upgrade head`) перед стартом `web`/`worker`; повторный запуск — no-op.
- Веб-интерфейс: http://127.0.0.1:8000 (только localhost), API-документация: http://127.0.0.1:8000/docs.
- Задачи очереди без STT (нормализация/саммаризация/wellbeing) выполняет базовый `worker`.

### Данные и их сохранность

- `./data` (SQLite `data/db/diary.db`, аудио, выгрузки, temp, backups) смонтирован в `/app/data` bind-mount'ом. Данные переживают `docker compose down` и пересоздание контейнеров; удаляются только при явном `docker compose down -v` (не рекомендуется) или `rm -rf data`.
- `./configs` смонтирован в `/app/configs` read-only: рабочие профили моделей задаются на хосте в `configs/models.yaml` и не входят в образ.
- Секреты — только в локальном `.env` на хосте (файл, если есть, передаётся в контейнеры через `env_file`; значения в репозиторий не попадают). Если `.env` нет — работают встроенные значения по умолчанию.

### Транскрипция в Docker

Базовый CPU-образ не содержит `faster-whisper` (опциональный extra, политика проекта). Запустите дополнительный воркер с STT:

- **CPU-транскрипция** (профиль `stt`): соберите воркер с `INSTALL_STT=1` и укажите в `configs/models.yaml` профиль `transcription` с `device: cpu` (по умолчанию так и есть):

  ```bash
  docker compose --profile stt up -d --build web worker-stt
  ```

  Воркер `worker-stt` (образ `diary:local-stt`) выполняет задачи `transcribe_audio` на CPU; модель faster-whisper при первом использовании скачивается в именованный volume `local-diary_whisper-cache` и переживает пересоздание контейнеров.

- **GPU-транскрипция** (профиль `gpu`, опционально): требуется NVIDIA driver + CUDA в WSL 2 и nvidia-container-toolkit (Docker Desktop включает GPU support для WSL 2). Сборка добавляет CUDA-библиотеки (cuBLAS/cuDNN) и запрашивает GPU:

  ```bash
  docker compose --profile gpu up -d --build web worker-gpu
  ```

  Затем в `configs/models.yaml` для профиля `transcription` укажите `device: cuda` (и при необходимости `compute_type`, например `float16`). GPU запрашивается только у сервиса `worker-gpu` через `deploy.resources.reservations.devices`; обычные сервисы GPU не используют.

  Примечание: воркеры с STT и без него могут работать одновременно — очередь рассчитана на несколько воркеров; чтобы не тратить ресурсы, останавливайте неиспользуемый (`docker compose stop worker`).

### Подключение локальной LLM-модели

Адрес эндпоинта задаётся в `configs/models.yaml` (поле `base_url` профилей `normalization`/`summarization`/`wellbeing_analysis`) — вне кода и вне образа. Варианты из контейнера:

1. **Модель как compose-сервис** (например Ollama в том же Compose-проекте): `base_url: http://<имя-сервиса>:11434/v1` — контейнеры одного compose-проекта видят друг друга по имени сервиса.
2. **Модель на хосте Windows/WSL** (запущена вне Docker): `base_url: http://host.docker.internal:PORT/v1`. Docker Desktop для WSL 2 резолвит `host.docker.internal` в адрес хоста. Для чистого Docker Engine в WSL 2 добавьте в секцию сервиса `extra_hosts: ["host.docker.internal:host-gateway"]`.
3. **Внешний провайдер**: обычный `https://…` URL; при этом в профиле должен стоять `external: true` и требуемая переменная `*_API_KEY` в `.env` (см. политику приватности).

Для вариантов 2–3 на CPU-контейнерах без GPU ничего дополнительно ставить не нужно.

### UID/GID владельца данных

Контейнеры работают от не-root пользователя `app` (UID/GID 1000 в образе; в compose `user: "${APP_UID:-1000}:${APP_GID:-1000}"`). Если ваш пользователь WSL имеет другой UID:

```bash
echo -e "APP_UID=$(id -u)\nAPP_GID=$(id -g)" >> .env
```

и пересоберите образ с теми же значениями (`docker compose build --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g)`). Кэш STT-моделей (volume `local-diary_whisper-cache`, `/app/.hf-cache`) при смене UID один раз поправьте: `docker compose run --rm -u root worker-stt chown -R "${APP_UID:-1000}:${APP_GID:-1000}" /app/.hf-cache`.

### Полезные команды

```bash
docker compose logs -f web        # логи веб-приложения
docker compose logs -f worker-stt # логи STT-воркера
docker compose down               # остановить (данные ./data сохраняются)
docker compose down -v            # остановить и удалить volume whisper-cache
docker compose build              # пересобрать образы без запуска
```

Образ не содержит пользовательских данных, `.env` и реального `configs/models.yaml`. Подробности окружения и допущений — в `docs/wsl2-deployment-brief.md`.

## Бэкапы и восстановление (Task 18)

Бэкапы создаются и восстанавливаются **только явными командами** через CLI
(`python -m diary.cli`). Создание бэкапа никогда не изменяет исходные данные
(только чтение); восстановление — необратимая операция и требует
обязательного флага подтверждения.

### Создание бэкапа

```bash
# «text»: SQLite (data/db/diary.db) + текстовые артефакты выгрузок
#         (data/exports/01_original … 04_wellbeing)
.venv/bin/python -m diary.cli backup --mode text

# «full»: text + аудио (data/audio)
.venv/bin/python -m diary.cli backup --mode full
```

- Архив создаётся в `data/backups/YYYYmmdd-HHMMSS_<mode>.tar.gz`.
- SQLite снимается консистентным онлайн-снапшотом (без остановки
  приложения); вместе с файлами в архив кладётся JSON-манифест
  (`manifest.json`) с путями относительно корня данных, SHA-256, размерами
  и временем — он описывает ровно то, что будет восстанавливаться.
- Перед бэкапом можно почистить временный каталог:

  ```bash
  .venv/bin/python -m diary.cli backup --mode full --clean-temp
  ```

  `--clean-temp` удаляет **только файлы внутри `data/temp/`**, старше
  порога `APP_TEMP_RETENTION_S` (по умолчанию 86400 с = 1 сутки). Аудио,
  выгрузки, бэкапы и база данных автоматически не удаляются никогда.

### Проверка целостности

Архив можно проверить без распаковки (читается архив, валиден манифест,
хэши/размеры файлов совпадают):

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from diary.domain.backup import verify_backup
latest = sorted(Path("data/backups").glob("*.tar.gz"))[-1]
print(verify_backup(latest))
PY
```

### Восстановление (recovery)

```bash
# Без --confirm команда ОТКАЖЕТСЯ выполнять восстановление:
.venv/bin/python -m diary.cli recover --archive data/backups/20260909-101530_full.tar.gz
# => error: Refusing to restore without explicit confirmation. ...

# С явным подтверждением:
.venv/bin/python -m diary.cli recover --archive data/backups/20260909-101530_full.tar.gz --confirm
```

Что делает recovery:

1. Архив проходит полную проверку целостности (манифест, безопасность
   путей, хэши); при любом нарушении восстановление отклоняется.
2. Перезаписываются **только файлы, перечисленные в манифесте**, строго
   внутри корня данных (пути с `..`, абсолютные пути и выходы через
   симлинки отклоняются).
3. Файлы, которых нет в архиве, не удаляются и не изменяются.

Перед восстановлением остановите приложение и воркеры (активная SQLite-БД
не должна перезаписываться «на лету»), при необходимости сделайте свежий
бэкап. Рекомендуется держать `db/`, `audio/`, `exports/`, `temp/`,
`backups/` внутри единого `APP_DATA_ROOT` — тогда пути архива совпадают с
раскладкой данных.

### PIN / локальная защита

Простая локальная защита PIN-кодом и шифрование данных at rest — **post-MVP**
(см. бэклог в `tasklist.md`): в MVP доступ к дневнику защищён тем, что
приложение слушает только localhost, а секреты и содержимое дневника не
попадают в Git и логи. Страница «Приватность» (`GET /privacy`, ссылка в
навигации) показывает для каждой задачи (транскрипция/нормализация/
саммаризация/wellbeing), локально или внешне выполняется обработка, какие
модель/провайдер настроены и настроен ли ключ (без значений ключей).


