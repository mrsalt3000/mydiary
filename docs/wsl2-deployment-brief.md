# Бриф развёртывания под WSL 2

- Архитектурный контекст: `architecture-brief.md`.
- Исполнение: Task 17 (Docker Compose), Task 09 (STT CPU/GPU), Task 18 (бэкапы).
- Решение по очереди/архитектуре: `decisions/ADR-001…003`.

## 1. Допущения о целевой среде

- Хост: Windows с WSL 2; дистрибутив Linux внутри WSL 2 — целевая среда запуска.
- Запуск MVP возможен: (а) напрямую в WSL 2 (python/venv, `make dev`), (б) через Docker Desktop c WSL 2 integration (Compose, web + worker).
- Один локальный пользователь; приложение слушает localhost по умолчанию.
- Проксирование/публикация наружу — вне скоупа MVP.

## 2. Особенности файловой системы

- Код и данные желательно держать внутри файловой системы Linux WSL (ext4), а не в `/mnt/c/...` — производительность и надёжность (тип файла, права, блокировки SQLite).
- SQLite в WSL 2: файл БД — на ext4; учитывать WAL и busy timeout (Task 04).
- Пути приложения конфигурируются; `data/` — единое дерево для web и воркера (общий volume в Compose).

## 3. Docker Compose (цель Task 17)

- Сервисы: `web` (FastAPI/uvicorn) и `worker` (durable runner). Общий код-образ или раздельные Dockerfile — по результатам Task 17.
- Данные переживают `docker compose down`: именованные volume'ы или документированные bind mounts для `data/db`, `data/audio`, `data/exports`, `data/backups`.
- Публикация порта ограничена `127.0.0.1:8000:8000`; worker порты не публикует.
- `.dockerignore` исключает `data/`, `.env`, кэши, виртуальные окружения.
- Профиль по умолчанию — CPU. GPU-профиль — опциональный (см. §5).

## 4. Локальный LLM-эндпоинт

- Если модель запускается в Compose — адрес по имени сервиса (например `http://llm:11434/v1`).
- Если модель запущена на хосте Windows/WSL — `http://host.docker.internal:PORT/v1` (WSL 2 / Docker Desktop поддерживают `host.docker.internal`); альтернативы WSL-специфичные документируются в Task 17.
- Адреса задаются конфигурацией профилей (Task 03), не зашиваются в код.

## 5. FFmpeg, GPU и системные зависимости

- FFmpeg требуется для аудио (запись/конвертация при необходимости, Task 07) — устанавливается в окружении/образе. **Для распознавания он не нужен:** и faster-whisper, и второй STT-бэкенд декодируют через PyAV, который несёт библиотеки FFmpeg в своём колесе (проверено в T42 прогоном с пустым `PATH` — `stt-backends-analysis.md` §2, §5).
- STT (faster-whisper): CPU-режим — базовая конфигурация по умолчанию (compute_type `int8` и т.п., Task 09).
- **Второй STT-бэкенд (GigaAM-v2 CTC+LM, T45) в образ не входит и ставится вне репозитория.** Так решено замером (сайдкар вместо extra: `unload()` не возвращает RAM, а `trust_remote_code` изолируется процессом — `stt-backends-analysis.md` §10, `ADR-017` §7). На хосте WSL 2 это две команды и три системных требования:
  - `scripts/setup-gigaam-venv.sh` — отдельный venv сайдкара (по умолчанию `~/.local/share/mydiary/gigaam/venv`, **6.5 GB** на диске). Тянет **системный тулчейн `cmake`/`g++`/`make`**: `kenlm 0.3.0` существует только как sdist и собирается из исходников (измерено: сборка 38.33 с, `stt-backends-analysis.md` §4.3). Скрипт проверяет тулчейн **до** установки и говорит, чего не хватает.
  - `scripts/fetch-gigaam-model.sh` — веса закреплённой ревизии (2.64 GB) в `data/models/gigaam` (или `APP_GIGAAM_MODEL_DIR`), с проверкой sha256. Качает **тот же `huggingface_hub`, которым сайдкар читает кэш** (`snapshot_download` интерпретатором сайдкара), поэтому порядок обратный не работает: сначала venv, потом веса. Готовые файлы проверяются и пропускаются, так что повторный запуск ничего не качает.
  - Приложению указывается интерпретатор: `APP_GIGAAM_PYTHON=<…>/venv/bin/python`. Venv приложения при этом не меняется вовсе — в нём ни `torch`, ни `transformers`, ни `kenlm`, ни `pyctcdecode` (доказательство фактом — `stt-backends-analysis.md` §4.4), и CI остаётся без тяжёлых зависимостей.
  - **В Docker второй бэкенд не разворачивается**: сайдкар-venv и веса лежат вне репозитория и вне образа, поэтому бэкенд доступен при локальном запуске (WSL 2), а не в Compose. Это ограничение названо, а не спрятано; включение его в образ — отдельная задача (образ вырос бы на ~6.5 GB и потребовал бы тулчейн на этапе сборки).
- CUDA в WSL 2 — опционально: NVIDIA driver на хосте + WSL CUDA; образ/профиль с GPU включается только при наличии окружения; GPU не требуется для MVP. Второму бэкенду GPU тоже не обязателен: `device: cpu` **проверен живьём** (T45, 2026-09-24) — работает, но на порядок медленнее (измерение и условия — `stt-backends-analysis.md` §12), поэтому это отладочный режим для машины без GPU, а не рабочий.
- Точные системные пакеты и версии фиксируются в Task 07/09/17 после проверки совместимости.

## 6. Переменные окружения

- Пример — `.env.example` (Task 01); значения секретов — только в окружении (не в файлах репозитория).
- Ключевые переменные (ожидаемые): `APP_TIMEZONE` (по умолчанию `Europe/Moscow`), пути данных, имена переменных секретов профилей (`*_API_KEY`), порт/привязка.
- Второй STT-бэкенд: `APP_GIGAAM_PYTHON` (интерпретатор сайдкара), `APP_GIGAAM_WORKER_SCRIPT` (по умолчанию `<repo>/scripts/gigaam-stt-worker.py`), `APP_GIGAAM_MODEL_DIR` (каталог весов; по умолчанию `<data root>/models/gigaam`) и `APP_GIGAAM_WORKER_TIMEOUT_S` — все описаны в `.env.example`. Бэкенд они **не** переключают: за это отвечает `provider:` в профиле `transcription`.
- Compose передаёт окружение в сервисы без публикации секретов в Git.

## 7. Верификация развёртывания

Ожидаемые команды после Task 17 (приведены как целевые):

```bash
docker compose config
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:8000/health
```

Проверки: данные сохраняются между пересозданиями контейнеров; CPU-сетап работает без GPU-зависимостей; наружу сервис не опубликован по умолчанию.

## 8. Итоговое состояние после Task 17

Реализовано (см. также секцию README «Запуск в Docker»):

- `docker/Dockerfile` — один образ для `web`, `worker` и одноразовой миграции; опциональные build-флаги `INSTALL_STT=1` (extra `stt`, CPU) и `ENABLE_CUDA=1` (NVIDIA cuBLAS/cuDNN через pip, `LD_LIBRARY_PATH`). Не-root пользователь `app` (UID/GID 1000, переопределяются build-аргументами). Пользовательские данные/секреты/`.env`/реальный `configs/models.yaml` в образ не попадают.
- `docker-compose.yml` — сервисы `migrate` (one-shot `alembic upgrade head`), `web` (uvicorn, публикация только `127.0.0.1:8000`), `worker` (CPU, без STT-extra), `worker-stt` (профиль `stt`, CPU-транскрипция), `worker-gpu` (профиль `gpu`, запрос GPU через `deploy.resources.reservations.devices`). Данные — bind-mount `./data:/app/data` (+ `./configs` read-only); кэш моделей — именованный volume. `depends_on` на успешное завершение `migrate`.
- `.dockerignore` — исключены `.git`, `.venv`, `data/`, `.env*`, кэши, тесты/доки; оставлены `src/`, `alembic/`, `prompts/`, `pyproject.toml`, `alembic.ini`, `README.md`.
- Проверенные факты окружения: FFmpeg в рантайме не нужен — приложение не транскодирует аудио (длительность остаётся NULL), а faster-whisper декодирует через PyAV, который поставляет собственные библиотеки FFmpeg. Все зависимости имеют manylinux-колёса под CPython 3.12, поэтому toolchain (gcc/build-essential) в образе не ставится. `tzdata` в образе обязателен для IANA-зон (например `Europe/Moscow`).
- Локальный LLM-эндпоинт из контейнера: имя compose-сервиса (`http://<service>:port/v1`), `host.docker.internal` (Docker Desktop; для Docker Engine в WSL 2 нужен `extra_hosts: ["host.docker.internal:host-gateway"]`) или внешний `https://…` с `external: true`. Адреса задаются в `configs/models.yaml`, не в коде.
- GPU: опциональный профиль `gpu`; требует NVIDIA driver/CUDA в WSL 2 и nvidia-container-toolkit. По умолчанию сервисы запускаются без GPU и без STT-зависимостей.
- Верификация (Task 17): `docker compose config` — валиден для default/`stt`/`gpu` профилей; тесты/линт/типы проекта — см. отчёт задачи.
