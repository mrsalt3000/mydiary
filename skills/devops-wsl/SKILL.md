# Skill: devops-wsl

## Purpose

Правила окружения и развёртывания: WSL 2, будущий Docker/Compose, volume'ы, localhost, FFmpeg, опциональный GPU, переменные окружения. Детали допущений: `../docs/wsl2-deployment-brief.md`.

## Scope and boundaries

- Входит: настройка окружения разработки (Task 01), системные зависимости (FFmpeg, STT), Compose-пакетирование web+worker (Task 17), бэкапы/recovery (Task 18).
- Вне границ: публикация наружу, прод-инфраструктура, распределённые системы (вне MVP).

## Mandatory rules

- Привязка к localhost по умолчанию (`127.0.0.1`); worker порты не публикует.
- Данные — внутри Linux-ФС WSL 2 (не `/mnt/c` для активных данных): производительность и корректность SQLite.
- Данные переживают `docker compose down`: именованные volume'ы/документированные bind mounts для `data/`; в `data/` не должно быть реального содержимого в Git.
- `.env` и секреты — только в окружении; `.env.example` без значений (Task 01); секреты не попадают в Compose-файлы репозитория.
- FFmpeg — системная зависимость аудио (Task 07/17); фиксируется в образе/документации.
- GPU (CUDA) — опциональный профиль (STT): включается только при наличии окружения NVIDIA в WSL 2; CPU — профиль по умолчанию; GPU не требуется для MVP.
- Локальный LLM-эндпоинт: имя compose-сервиса или `host.docker.internal` (документированные WSL-альтернативы) — через конфигурацию профилей, не в коде.
- `.dockerignore` исключает `data/`, `.env*`, кэши, окружения.
- Бэкап: два режима (БД+текст, полный с аудио); recovery — явная команда с обязательным подтверждением; очистка — только `data/temp/` (Task 18).

## Expected module locations

- Корень: `docker/` (Dockerfile'ы по задачам), `docker-compose.yml` (Task 17), `.dockerignore`, `Makefile` (Task 01).
- Скрипты администрирования — `../scripts/` (по мере появления задач, например backup/recovery CLI в Task 18).
- Документация: `../docs/wsl2-deployment-brief.md`.

## Quality and security checks

- Команды верификации после Task 17: `docker compose config`, `docker compose up --build -d`, `docker compose ps`, `curl http://127.0.0.1:8000/health`.
- Проверки: данные сохраняются между пересозданиями; CPU-сетап работает без GPU-зависимостей; сервисы не опубликованы наружу по умолчанию; секретов в Git нет.

## Typical validation commands (целевые, после Task 01/17)

```bash
make test
make lint
docker compose config
```

## Links

- ADR: `../docs/decisions/ADR-001.md`, `../docs/decisions/ADR-003.md`.
- Документация: `../docs/wsl2-deployment-brief.md`; `../docs/architecture-brief.md` §3, §12; `../docs/privacy-and-safety.md`.
- Задачи: `../tasklist.md` — Task 01 (окружение), Task 07/09 (FFmpeg/STT), Task 17 (Compose), Task 18 (бэкапы).
