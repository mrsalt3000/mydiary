#!/usr/bin/env bash
#
# scripts/run-local.sh — запуск web или worker с правильно подготовленным окружением.
#
# Зачем скрипт, если есть `make dev`: три вещи, которые при ручном запуске
# легко забыть, а последствия неочевидны.
#
#   1. Значения из .env НЕ попадают в os.environ — в проекте нет load_dotenv,
#      .env читает только pydantic-settings, а профили LLM резолвят ключ через
#      os.environ[api_key_env]. Без `set -a; . ./.env; set +a` ключи *_API_KEY
#      остаются невидимыми, и любой конвейер падает на первом запросе.
#   2. ctranslate2 4.8.2 собран под CUDA 12 и ищет libcublas.so.12, тогда как в
#      системе CUDA 13. Каталог с симлинком-шимом надо добавить в
#      LD_LIBRARY_PATH, иначе транскрипция на GPU падает с
#      "Library libcublas.so.12 is not found or cannot be loaded".
#   3. Модель STT уже в локальном кэше, поэтому HF_HUB_OFFLINE=1 убирает два
#      запроса к huggingface.co на каждую задачу (проверка ревизии модели).
#      Перед сменой модели запустите один раз с HF_HUB_OFFLINE=0.
#
# Использование:
#   scripts/run-local.sh web [аргументы...]      # uvicorn на 127.0.0.1:8000
#   scripts/run-local.sh worker [аргументы...]   # воркер очереди задач
#   scripts/run-local.sh cli [аргументы...]      # python -m diary.cli (day/week/coach/backup)
#   scripts/run-local.sh nli [аргументы...]      # python -m diary.cli nli (отчёт NLI по сохранённой сводке дня)
#   scripts/run-local.sh --help
#
# Аргументы после режима ЗАМЕНЯЮТ аргументы по умолчанию (web: --host 127.0.0.1
# --port 8000 --reload; worker: --poll-interval 2; cli и nli: ничего — подкоманду
# и её флаги передавайте явно, например `cli day --date 2026-09-14`,
# `nli --date 2026-09-14`). Режим cli — это
# «один вход» в пайплайн фазы 2 с этой машины (T11): он готовит то же
# окружение, что web/worker (.env в окружение, шим libcublas, HF_HUB_OFFLINE),
# без которого профили LLM не найдут ключ, а STT на GPU — библиотеку.
#
# Режим nli (T27) — инструмент владельца данных: NLI-проверка **сохранённой**
# дневной сводки против источника, отчёт без текста, база открывается только на
# чтение. Это не гейт: ничего не пишется и ничего не блокируется. Сам NLI считает
# на CPU (onnxruntime), GPU и шим libcublas ему не нужны, HF_HUB_OFFLINE ни на что
# не влияет; окружение всё равно готовится то же, что у других режимов, — чтобы
# результат не зависел от способа запуска.
#
# Переопределяемые переменные окружения:
#   CT2_CUDA_SHIM_DIR  каталог с libcublas.so.12 (по умолчанию ~/.local/lib/ct2-cuda)
#   HF_HUB_OFFLINE     по умолчанию 1; поставьте 0, чтобы разрешить скачивание моделей

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

CT2_CUDA_SHIM_DIR="${CT2_CUDA_SHIM_DIR:-$HOME/.local/lib/ct2-cuda}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
VENV_UVICORN="$REPO_ROOT/.venv/bin/uvicorn"

usage() {
    awk 'NR < 3 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

mode="${1:-}"
if [ -z "$mode" ] || [ "$mode" = "-h" ] || [ "$mode" = "--help" ]; then
    usage
    [ -z "$mode" ] && exit 2
    exit 0
fi
shift

case "$mode" in
    web | worker | cli | nli) ;;
    *)
        printf 'run-local.sh: неизвестный режим %s (ожидается web, worker, cli или nli)\n' "$mode" >&2
        exit 2
        ;;
esac

if [ ! -f "$REPO_ROOT/.env" ]; then
    printf 'run-local.sh: нет %s/.env — создайте из .env.example:\n' "$REPO_ROOT" >&2
    printf '    cp .env.example .env\n' >&2
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    printf 'run-local.sh: нет %s — выполните `make install`\n' "$VENV_PYTHON" >&2
    exit 1
fi

# 1. .env → окружение процесса (иначе *_API_KEY не резолвятся).
set -a
# shellcheck source=/dev/null
. "$REPO_ROOT/.env"
set +a

# 2. Каталог-шим для libcublas.so.12 → libcublas.so.13.
if [ -e "$CT2_CUDA_SHIM_DIR/libcublas.so.12" ]; then
    LD_LIBRARY_PATH="$CT2_CUDA_SHIM_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
    cuda_note="шим найден, GPU-транскрипция доступна"
else
    cuda_note="шим НЕ найден в $CT2_CUDA_SHIM_DIR — при device: cuda задача упадёт"
fi

# 3. Офлайн-режим HF Hub для закэшированной модели.
export HF_HUB_OFFLINE

cd "$REPO_ROOT"

case "$mode" in
    web)
        if [ "$#" -eq 0 ]; then
            set -- --host 127.0.0.1 --port 8000 --reload
        fi
        printf 'run-local.sh: web | %s | HF_HUB_OFFLINE=%s\n' "$cuda_note" "$HF_HUB_OFFLINE" >&2
        exec "$VENV_UVICORN" diary.main:app "$@"
        ;;
    worker)
        if [ "$#" -eq 0 ]; then
            set -- --poll-interval 2
        fi
        printf 'run-local.sh: worker | %s | HF_HUB_OFFLINE=%s\n' "$cuda_note" "$HF_HUB_OFFLINE" >&2
        exec "$VENV_PYTHON" -m diary.workers.runner "$@"
        ;;
    cli)
        # Подкоманда обязательна (day/week/coach/backup/recover) — без неё
        # аргументов нет, и argparse сам объяснит, что нужно.
        printf 'run-local.sh: cli | %s | HF_HUB_OFFLINE=%s\n' "$cuda_note" "$HF_HUB_OFFLINE" >&2
        exec "$VENV_PYTHON" -m diary.cli "$@"
        ;;
    nli)
        # Отдельный режим ради подготовленного окружения и понятной подсказки в
        # --help; подкоманда nli у diary.cli делает саму работу (T27).
        printf 'run-local.sh: nli | %s | HF_HUB_OFFLINE=%s\n' "$cuda_note" "$HF_HUB_OFFLINE" >&2
        exec "$VENV_PYTHON" -m diary.cli nli "$@"
        ;;
esac
