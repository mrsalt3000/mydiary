#!/usr/bin/env bash
#
# scripts/serve-ollama.sh — рантайм Ollama для слоёв фазы 2 (T01, роль A1).
#
# Держит рантайм LLM целиком вне репозитория и без root: бинарник —
# ~/.local/bin/ollama, модели — OLLAMA_MODELS, лог и pidfile —
# ~/.local/state/mydiary/. Внутрь репозитория не пишется ни бинарников,
# ни логов, ни моделей (AGENTS.md §7–§8).
#
# Контекст модели (num_ctx 8192) задаётся только в ollama/Modelfile: серверные
# переменные ниже влияют на скорость и расход VRAM, но не на размер контекста.
# OLLAMA_MAX_LOADED_MODELS=1 и OLLAMA_NUM_PARALLEL=1 — GPU-серийность:
# STT и LLM не работают одновременно (docs/tasklist-phase2.md §9.10).
#
# Использование:
#   scripts/serve-ollama.sh install [<tag>]    # скачать релиз (по умолчанию v0.34.2) в ~/.local
#   scripts/serve-ollama.sh start              # поднять serve в фоне (идемпотентно)
#   scripts/serve-ollama.sh status             # health + `ollama ps`
#   scripts/serve-ollama.sh stop               # остановить serve по pidfile
#   scripts/serve-ollama.sh unload <model>     # выгрузить модель из VRAM перед ASR-джобой
#   scripts/serve-ollama.sh pull <model>       # скачать модель в OLLAMA_MODELS
#   scripts/serve-ollama.sh create <name>      # создать тег из ollama/Modelfile
#   scripts/serve-ollama.sh --help
#
# Переопределяемые переменные окружения:
#   OLLAMA_MODELS  каталог моделей Ollama (по умолчанию ~/.ollama/models)
#   PYTHON_BIN     интерпретатор для распаковки релиза (по умолчанию /usr/bin/python3)
#   TMPDIR         каталог для временного архива при install

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# --- абсолютные пути: ни PATH, ни dot-файлы здесь не участвуют ---------------
OLLAMA_BIN="$HOME/.local/bin/ollama"
OLLAMA_PREFIX="$HOME/.local"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
STATE_DIR="$HOME/.local/state/mydiary"
LOG="$STATE_DIR/ollama-serve.log"
PIDFILE="$STATE_DIR/ollama-serve.pid"
MODELFILE="$REPO_ROOT/ollama/Modelfile"

# --- рантайм-конфигурация ---------------------------------------------------
OLLAMA_HOST="127.0.0.1:11434" # только loopback: наружу сервер не публикуется
HEALTH_URL="http://$OLLAMA_HOST/api/version"
HEALTH_TIMEOUT_S=30
RELEASE_TAG_DEFAULT="v0.34.2"
RELEASE_ASSET="ollama-linux-amd64.tar.zst"

export OLLAMA_HOST
export OLLAMA_MODELS
export OLLAMA_NO_CLOUD=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_KEEP_ALIVE=5m

usage() {
    awk 'NR < 3 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

die() {
    printf 'serve-ollama.sh: %s\n' "$*" >&2
    exit 1
}

require_curl() {
    command -v curl >/dev/null 2>&1 || die 'нет curl — он нужен для health-проверки и скачивания релиза'
}

require_binary() {
    [ -x "$OLLAMA_BIN" ] || die "нет $OLLAMA_BIN — выполните: scripts/serve-ollama.sh install"
}

health() {
    curl -sf --max-time 3 "$HEALTH_URL" >/dev/null 2>&1
}

wait_for_health() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
    while [ "$SECONDS" -lt "$deadline" ]; do
        health && return 0
        sleep 1
    done
    return 1
}

# Жив ли процесс и правда ли это ollama (страховка от устаревшего pidfile).
is_ollama_pid() {
    local pid="${1:-}"
    [ -n "$pid" ] || return 1
    [ -r "/proc/$pid/cmdline" ] || return 1
    tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q 'ollama'
}

# Фолбэк распаковки, когда в системе нет бинарника zstd: tarfile умеет режим
# r:zst с Python 3.14, а extractall(filter="data") сохраняет бит 0o755 на
# бинарнике и отбрасывает опасные элементы архива (пути наружу, спецфайлы).
python_extract_zst() {
    local archive="$1" dest="$2" python_bin="" candidate
    for candidate in "${PYTHON_BIN:-}" /usr/bin/python3 "$(command -v python3 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            python_bin="$candidate"
            break
        fi
    done
    [ -n "$python_bin" ] || die "нет zstd и нет python3 — распакуйте $archive вручную в $dest"
    printf 'serve-ollama.sh: zstd нет — распаковка через %s\n' "$python_bin" >&2
    "$python_bin" - "$archive" "$dest" <<'PY' || die "python3 не распаковал $archive — поставьте пакет zstd"
import sys
import tarfile

archive, dest = sys.argv[1], sys.argv[2]
with tarfile.open(archive, "r:zst") as tar:
    tar.extractall(dest, filter="data")
PY
}

cmd_install() {
    require_curl
    local tag="${1:-$RELEASE_TAG_DEFAULT}"
    local url="https://github.com/ollama/ollama/releases/download/$tag/$RELEASE_ASSET"
    local tmp_dir
    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/serve-ollama-install.XXXXXX")"

    printf 'serve-ollama.sh: скачиваю %s\n' "$url" >&2
    if ! curl -fL --retry 3 --retry-delay 2 --output "$tmp_dir/$RELEASE_ASSET" "$url"; then
        die "не скачался $url (проверьте имя тега)"
    fi

    mkdir -p "$OLLAMA_PREFIX"
    if command -v zstd >/dev/null 2>&1; then
        printf 'serve-ollama.sh: распаковка (tar --zstd) в %s\n' "$OLLAMA_PREFIX" >&2
        tar --zstd -xf "$tmp_dir/$RELEASE_ASSET" -C "$OLLAMA_PREFIX"
    else
        python_extract_zst "$tmp_dir/$RELEASE_ASSET" "$OLLAMA_PREFIX"
    fi
    rm -rf "$tmp_dir"

    [ -x "$OLLAMA_BIN" ] || die "распаковка не дала исполняемый $OLLAMA_BIN"
    printf 'serve-ollama.sh: установлено в %s\n' "$OLLAMA_PREFIX" >&2
    "$OLLAMA_BIN" --version
}

cmd_start() {
    require_curl
    if health; then
        printf 'serve-ollama.sh: сервер уже отвечает на %s — переиспользую\n' "$HEALTH_URL" >&2
        return 0
    fi
    require_binary

    if [ -f "$PIDFILE" ] && is_ollama_pid "$(cat "$PIDFILE")"; then
        die "pid $(cat "$PIDFILE") жив, но $HEALTH_URL не отвечает — разбирайтесь по логу $LOG"
    fi

    mkdir -p "$STATE_DIR" "$OLLAMA_MODELS"
    printf 'serve-ollama.sh: запускаю %s serve, лог %s\n' "$OLLAMA_BIN" "$LOG" >&2
    setsid nohup "$OLLAMA_BIN" serve >>"$LOG" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" >"$PIDFILE"

    if ! wait_for_health; then
        die "сервер не поднялся за ${HEALTH_TIMEOUT_S} с (pid $pid) — смотрите $LOG"
    fi
    printf 'serve-ollama.sh: сервер поднят (pid %s)\n' "$pid" >&2
}

cmd_status() {
    require_curl
    require_binary
    if ! health; then
        printf 'serve-ollama.sh: %s не отвечает (pidfile %s) — запустите start\n' "$HEALTH_URL" "$PIDFILE" >&2
        return 1
    fi
    printf 'serve-ollama.sh: %s отвечает (pidfile %s)\n' "$HEALTH_URL" "$PIDFILE" >&2
    "$OLLAMA_BIN" ps
}

cmd_stop() {
    require_curl
    if [ ! -f "$PIDFILE" ]; then
        if health; then
            die "сервер отвечает на $HEALTH_URL, но $PIDFILE нет — он поднят не этим скриптом, остановите вручную"
        fi
        printf 'serve-ollama.sh: сервер не запущен\n' >&2
        return 0
    fi

    local pid
    pid="$(cat "$PIDFILE")"
    rm -f "$PIDFILE"

    if ! is_ollama_pid "$pid"; then
        printf 'serve-ollama.sh: процесса %s нет (pidfile устарел)\n' "$pid" >&2
        return 0
    fi

    kill "$pid" 2>/dev/null || true
    local _attempt
    for _attempt in 1 2 3 4 5 6 7 8 9 10; do
        is_ollama_pid "$pid" || break
        sleep 1
    done
    if is_ollama_pid "$pid"; then
        printf 'serve-ollama.sh: pid %s не завершился за 10 с — SIGKILL\n' "$pid" >&2
        kill -9 "$pid" 2>/dev/null || true
    fi
    printf 'serve-ollama.sh: остановлен (pid %s)\n' "$pid" >&2
}

cmd_unload() {
    local model="${1:-}"
    [ -n "$model" ] || die 'нужно имя модели: scripts/serve-ollama.sh unload <model>'
    require_binary
    "$OLLAMA_BIN" stop "$model"
}

cmd_pull() {
    local model="${1:-}"
    [ -n "$model" ] || die 'нужно имя модели: scripts/serve-ollama.sh pull <model>'
    require_binary
    "$OLLAMA_BIN" pull "$model"
}

cmd_create() {
    local name="${1:-}"
    [ -n "$name" ] || die 'нужно имя тега: scripts/serve-ollama.sh create <name>'
    require_binary
    [ -f "$MODELFILE" ] || die "нет $MODELFILE"
    (cd "$REPO_ROOT" && "$OLLAMA_BIN" create "$name" -f ollama/Modelfile)
}

cmd="${1:-}"
if [ -z "$cmd" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ]; then
    usage
    [ -z "$cmd" ] && exit 2
    exit 0
fi
shift

case "$cmd" in
    install) cmd_install "$@" ;;
    start) cmd_start ;;
    status) cmd_status ;;
    stop) cmd_stop ;;
    unload) cmd_unload "$@" ;;
    pull) cmd_pull "$@" ;;
    create) cmd_create "$@" ;;
    *)
        printf 'serve-ollama.sh: неизвестная команда %s\n\n' "$cmd" >&2
        usage >&2
        exit 2
        ;;
esac
