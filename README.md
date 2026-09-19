# Local Diary

Локальный (local-first) однопользовательский личный дневник для WSL 2: текстовые и голосовые сообщения с ручной транскрипцией, еженедельные TXT-выгрузки и три производных слоя — нормализация (слой 2), саммаризация (слой 3) и осторожная wellbeing-рефлексия (слой 4). Каждая задача (транскрипция/нормализация/саммаризация/wellbeing) настраивает собственный провайдер/модель — локальные или внешние OpenAI-совместимые эндпоинты.

**Статус: MVP (Task 01–19) реализован.** Реестр приёмок — [`docs/acceptance-log.md`](docs/acceptance-log.md); реальные ограничения — [`docs/known-limitations.md`](docs/known-limitations.md).

## Возможности MVP

- Текстовые сообщения в SQLite с пагинацией и мягким удалением.
- Запись/загрузка голосовых сообщений (аудио хранится локально, пути генерируются сервером).
- Ручная транскрипция (кнопка «Расшифровать») через durable-очередь на SQLite; raw-STT хранится отдельно от аудио, ретранскрибация добавляет версии.
- Недельные выгрузки (понедельник–воскресенье, воскресенье в имени файла), нумерация `NNNN` без пропусков и ревизии `rN` через явный re-export.
- Конвейеры с происхождением данных: оригинал → нормализация → саммари/wellbeing, каждый артефакт привязан к точному входу и прогону (ADRs, `docs/data-lineage.md`).
- Web UI на localhost (чат, выгрузки/детали происхождения, processing, страница приватности) + JSON API `/api/v1`.
- Бэкапы и восстановление через CLI; Docker Compose (web + worker, опциональный STT/GPU).
- Наблюдаемость: request id в структурированных логах, `GET /ready`, безопасные метрики job'ов в логах — без содержимого дневника.

## Быстрый старт (WSL 2)

Предварительные условия: Python 3.12 (`~/.local/bin/python3.12`), `make`. Данные репозитория и `data/` — на файловой системе WSL (ext4), не на `/mnt/c`.

```bash
make install                 # .venv + editable-пакет с dev-зависимостями
cp .env.example .env         # синтетические значения; APP_TIMEZONE по умолчанию Europe/Moscow
make migrate                 # alembic upgrade head → data/db/diary.db
make dev                     # web на http://127.0.0.1:8000 (localhost)
```

В отдельном терминале запустите воркер (выполняет задачи очереди):

```bash
.venv/bin/python -m diary.workers.runner --poll-interval 2
```

Проверки:

```bash
curl -s http://127.0.0.1:8000/health   # {"status":"ok"}
curl -s http://127.0.0.1:8000/ready    # {"status":"ready"} — SQLite доступна; 503, если нет
```

Качество кода: `make test`, `make lint`, `make typecheck`. Перед коммитами: `.venv/bin/pre-commit install`.

### Запуск с GPU-транскрипцией (опционально)

Транскрипцию можно выполнять на видеокарте. На проверенной конфигурации с 6 ГБ VRAM это позволяет держать `large-v3` примерно за то же время, какое CPU тратил на `small` (разброс 0.7–1.6×), а `large-v3-turbo` — примерно на порядок быстрее CPU. Окружение для этого собирается из нескольких неочевидных частей, поэтому используйте скрипт:

```bash
scripts/run-local.sh web       # uvicorn на 127.0.0.1:8000
scripts/run-local.sh worker    # воркер очереди
```

Он делает три вещи, которые при ручном запуске легко упустить: экспортирует `.env` в окружение процесса (иначе `*_API_KEY` не резолвятся — `load_dotenv` в проекте нет), добавляет в `LD_LIBRARY_PATH` каталог с `libcublas.so.12`, и включает `HF_HUB_OFFLINE=1`, чтобы каждая задача не проверяла ревизию модели в HuggingFace. Подробности — `scripts/run-local.sh --help`.

Предварительные условия для GPU: профиль `transcription` в `configs/models.yaml` с `device: cuda` и подходящим `compute_type` — `int8_float16` укладывается в 6 ГБ, `float16` быстрее, но требует ~5.8 ГБ VRAM и оставляет слишком мало запаса, — а также резолвящийся `libcublas.so.12`. ctranslate2 собран под CUDA 12, тогда как в системе может стоять CUDA 13 — тогда достаточно симлинка:

```bash
mkdir -p ~/.local/lib/ct2-cuda
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13 \
       ~/.local/lib/ct2-cuda/libcublas.so.12
```

Откат на CPU: верните в профиле `device: cpu` и `compute_type: int8`. Автоматического фолбэка нет — при занятой видеокарте или нехватке VRAM задача расшифровки завершится ошибкой.

## Конфигурация

- **Окружение** — `.env` (git-ignored), переменные с префиксом `APP_*`. Все параметры описаны в `.env.example`: таймзона, лимиты сообщений/аудио, политика очереди (retry/backoff/stale), размеры чанков нормализации и саммаризации, `APP_WELLBEING_ENABLED`, пути каталогов данных, `APP_TEMP_RETENTION_S`.
- **Модели** — `configs/models.yaml` (git-ignored; пример `configs/models.example.yaml`, путь переопределяется `MODELS_CONFIG`). Четыре профиля: `transcription` (`faster_whisper`) и `normalization`/`summarization`/`wellbeing_analysis` (`openai_compatible` с `base_url`, `model`, `timeout_s`, `temperature`, `max_tokens`, `external`).
- **Секреты** — только переменные окружения, на которые профиль ссылается по имени (`api_key_env`). Значения ключей не хранятся и не логируются; внешние эндпоинты получают содержимое только при `external: true` и явном согласии в UI (страница `/privacy` показывает локальность каждой задачи).
- Смена настроек требует перезапуска процессов (settings кэшируются при старте).

## Наблюдаемость и логи

Формат строки лога (web и воркер):

```
2026-09-09 12:00:00,000 | INFO | diary.api | request_id=<32 hex> | http_request method=GET path=/health status=200 duration_ms=1.2
```

- Каждый HTTP-запрос получает request id (промежуточный слой приложения); логируется только method/path/status/длительность.
- При завершении job воркер логирует безопасные метрики: `job`, `type`, `status`, `attempts`, `duration_ms`, а при наличии — `model`, `provider`, `prompt_version`, `tokens_input/output`, `artifact_bytes`. Содержимое дневника, промпты, транскрипции и секреты в логах отсутствуют (проверяется автотестами).
- `GET /ready` проверяет доступность SQLite (не состояние миграций и не наличие профилей — для этого страницы `/processing` и `/privacy`).

## Транскрипция на CPU (локальный STT)

`faster-whisper` — опциональный extra (не ставится по умолчанию/в CI):

```bash
.venv/bin/pip install -e ".[stt]"
```

В `configs/models.yaml` профиль `transcription` по умолчанию: `device: cpu`, `compute_type: int8`, `language: ru` (см. `configs/models.example.yaml`). Кнопка «Расшифровать» у аудио-сообщения ставит задачу воркеру; модель скачивается в локальный кэш при первом использовании. Если extra не установлен — понятная ошибка, аудио не теряется. GPU/CUDA — опционально (`docs/wsl2-deployment-brief.md`).

## Бэкапы и восстановление (CLI)

```bash
.venv/bin/python -m diary.cli backup --mode text      # SQLite + текстовые артефакты
.venv/bin/python -m diary.cli backup --mode full --clean-temp   # + аудио; чистит только data/temp/
.venv/bin/python -m diary.cli recover --archive data/backups/<файл>.tar.gz          # отказ без --confirm
.venv/bin/python -m diary.cli recover --archive data/backups/<файл>.tar.gz --confirm
```

Архивы — в `data/backups/`, с JSON-манифестом (пути, SHA-256, размеры). Проверка целостности архива без распаковки:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from diary.domain.backup import verify_backup
latest = sorted(Path("data/backups").glob("*.tar.gz"))[-1]
print(verify_backup(latest))
PY
```

Перед восстановлением остановите web/воркеры; восстановление перезаписывает только файлы из манифеста и нетранзакционно (пофайлово) — перед ним рекомендуется свежий бэкап (`docs/known-limitations.md` §4).

## Docker (WSL 2)

Альтернативный запуск контейнерами (данные — bind-mount `./data`, конфиг — `./configs` read-only, секреты — `.env` через `env_file`; наружу публикуется только `127.0.0.1:8000`):

```bash
docker compose up -d --build
docker compose ps                 # migrate (exited 0) + web + worker
curl -s http://127.0.0.1:8000/health
```

- CPU-транскрипция: `docker compose --profile stt up -d --build web worker-stt` (образ собирается с `INSTALL_STT=1`; кэш моделей — volume `local-diary_whisper-cache`).
- GPU-транскрипция (опционально): `docker compose --profile gpu up -d --build web worker-gpu` + профиль `device: cuda` (нужны NVIDIA driver/CUDA в WSL 2 и nvidia-container-toolkit).
- Локальная LLM-модель: `base_url` = имя compose-сервиса (`http://<service>:порт/v1`) или `http://host.docker.internal:порт/v1`.
- UID владельца данных = 1000 по умолчанию; при другом UID задайте `APP_UID`/`APP_GID` в `.env` и пересоберите с теми же `--build-arg`.
- `docker compose down` сохраняет `./data`; `down -v` удаляет volume whisper-cache.

Полезные команды: `docker compose logs -f web`, `docker compose logs -f worker-stt`, `docker compose build`.

## Контракты и golden-набор (фаза 2)

Вторая фаза ([`docs/tasklist-phase2.md`](docs/tasklist-phase2.md), T00–T12; решение — [ADR-007](docs/decisions/ADR-007-second-anchor-family-for-derivational-layers.md)) вводит структурированные контракты производных слоёв — `raw_entry`, `stt_corrected`, `daily_summary`, `weekly_summary`, `coach_note` — и синтетический golden-набор для проверки качества:

- модели контрактов: [`src/diary/domain/contracts/`](src/diary/domain/contracts) (Pydantic, `extra="forbid"`; JSON Schema выводятся из них);
- [`schemas/`](schemas/README.md) — **производные** JSON Schema: правятся не руками, а правкой модели + регенерацией;
- [`eval/golden/`](eval/golden/README.md) — 10 синтетических RU-чанков (3/3/2/2) и манифест с порогами (8/10) и лимитами латентности T10 (правка < 5 с, день < 15 с).

```bash
make contracts                                                  # перезаписать schemas/*.json из моделей
.venv/bin/python scripts/generate_contract_schemas.py --check   # проверить расхождение, ничего не записывая
```

Это подготовленный, но ещё не подключённый слой: рантайм MVP (слои 1–4) работает как принято и модулей `domain/contracts` не использует — их подключают задачи T01–T12 (персистентность второй семьи якорей — только T08, `artifacts.export_id NOT NULL` снимается там же).

## Руководства и документация

| Документ | Назначение |
|---|---|
| [`tasklist.md`](tasklist.md) | Авторитетный бэклог Task 01–19 (источник исполнения; не переписывается без запроса) |
| [`AGENTS.md`](AGENTS.md) | Операционный контракт для агентов: цикл работ, приватность, DoD |
| [`docs/architecture-brief.md`](docs/architecture-brief.md) | Полный архитектурный бриф (модули, данные, конвейеры, тесты, вехи) |
| [`docs/decisions/`](docs/decisions/README.md) | ADR-001…006 |
| [`docs/data-lineage.md`](docs/data-lineage.md) | Слои артефактов, ревизии/версии, хэши, имена, прослеживаемость |
| [`docs/privacy-and-safety.md`](docs/privacy-and-safety.md) | Классификация данных, секреты, провайдеры, логи, wellbeing-границы |
| [`docs/testing-strategy.md`](docs/testing-strategy.md) | Слои тестов, mock-провайдеры, фикстуры, e2e |
| [`docs/known-limitations.md`](docs/known-limitations.md) | Актуальные ограничения MVP (Task 19) |
| [`docs/manual-acceptance-checklist.md`](docs/manual-acceptance-checklist.md) | Пошаговый ручной сценарий приёмки MVP на WSL 2 |
| [`docs/wsl2-deployment-brief.md`](docs/wsl2-deployment-brief.md) | Допущения WSL 2, Docker/Compose, GPU, локальные LLM-эндпоинты |
| [`docs/execution-protocol.md`](docs/execution-protocol.md) | Протокол multi-agent исполнения задач |
| [`skills/`](skills/README.md) | Исполнительные руководства по областям |
| [`prompts/`](prompts/README.md), [`configs/`](configs/README.md), [`data/`](data/README.md) | Политики промптов, конфигурации и приватных данных |

## PIN / локальная защита

PIN-код и шифрование at rest — **post-MVP** (см. бэклог `tasklist.md`): доступ защищён привязкой к localhost и тем, что секреты и содержимое дневника не попадают в Git и логи. Страница «Приватность» (`GET /privacy`) показывает для каждой задачи провайдера/модель/endpoint host и статус ключа — без значений.
