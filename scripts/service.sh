#!/usr/bin/env bash
#
# scripts/service.sh — поднять, остановить и проверить весь локальный стек дневника.
#
# Зачем скрипт. Рабочий стек — три процесса, и два из них нельзя поднять
# «как по учебнику», а ошибки тут молчаливые:
#   * Ollama — с OLLAMA_KEEP_ALIVE=0: режим 5m (режим serve-ollama.sh) возвращает
#     перехват CPU-тегом недели, замер T29 — `cli coach` 10.02 с → 116.26 с;
#   * веб — без --reload: у перезагрузчика uvicorn отдельный процесс, и сторож по
#     cmdline его не видит (там `spawn_main`, а не `uvicorn diary.main:app`),
#     поэтому такого держателя порта `status` и `stop` опознают по занятому порту;
#   * оба приложения — только через scripts/run-local.sh: он экспортирует .env
#     (в проекте нет load_dotenv, без этого профили LLM не видят *_API_KEY) и
#     ставит шим libcublas для GPU-Whisper.
#
# Скрипт поднимает всё это одной командой, пишет pid-файлы и логи в
# ~/.local/state/mydiary и умеет показать состояние (status). Остановка симметрична
# запуску: компонент, поднятый вне скрипта (вручную, `make dev`), тоже гасится, а если
# его не видно по имени процесса (перезагрузчик uvicorn) — порт освобождает уборка по
# порту; иначе `stop` оставлял бы порт занятым. Сторож по cmdline сверяет ещё и argv[0],
# поэтому чужая оболочка, у которой шаблон просто стоит в командной строке, не попадает
# ни в отчёт, ни под kill. Pid-файл Ollama назван так же, как в serve-ollama.sh, поэтому
# двум скриптам виден один и тот же сервер: `serve-ollama.sh stop` тоже его остановит.
#
# Использование:
#   scripts/service.sh start     # ollama + web (:8000) + worker; уже поднятое переиспользует
#   scripts/service.sh stop      # stop в порядке, обратном запуску; ждёт завершения, затем SIGKILL
#   scripts/service.sh restart
#   scripts/service.sh status    # pid и живость каждого компонента (health — у Ollama и веба); поднятое вне скрипта помечает
#   scripts/service.sh --help
#
# С Windows это запускают обёртки scripts/windows/start-diary.bat и
# stop-diary.bat (или их копии на рабочем столе): они вызывают
# `wsl.exe -d Ubuntu -e bash -lc "cd <repo> && scripts/service.sh start"`.
#
# Переопределяемые переменные окружения:
#   DIARY_STATE_DIR   каталог pid-файлов и логов (по умолчанию ~/.local/state/mydiary)
#   DIARY_WEB_PORT    порт веба (по умолчанию 8000)
#   OLLAMA_BIN        путь к бинарю Ollama (по умолчанию ~/.local/bin/ollama)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

STATE_DIR="${DIARY_STATE_DIR:-$HOME/.local/state/mydiary}"
WEB_PORT="${DIARY_WEB_PORT:-8000}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/bin/ollama}"

OLLAMA_PORT=11434
OLLAMA_HOST_VALUE="127.0.0.1:$OLLAMA_PORT"
OLLAMA_URL="http://$OLLAMA_HOST_VALUE/api/version"
WEB_URL="http://127.0.0.1:$WEB_PORT/"

OLLAMA_PIDFILE="$STATE_DIR/ollama-serve.pid"
OLLAMA_LOG="$STATE_DIR/ollama-serve.log"
WEB_PIDFILE="$STATE_DIR/diary-web.pid"
WEB_LOG="$STATE_DIR/diary-web.log"
WORKER_PIDFILE="$STATE_DIR/diary-worker.pid"
WORKER_LOG="$STATE_DIR/diary-worker.log"

OLLAMA_MARKER="ollama serve"
WEB_MARKER="uvicorn diary.main:app"
WORKER_MARKER="diary.workers.runner"

# Сторож по cmdline сверяет ещё и исполняемый файл процесса: шаблон встречается и в
# командной строке чужой оболочки (например, `bash -c 'pgrep -f "uvicorn diary.main:app"'`),
# и без этой сверки `stop` погасил бы такую оболочку. Приложения запускаются через
# run-local.sh, который делает exec в интерпретатор .venv, поэтому их argv[0] — python.
OLLAMA_EXE="ollama"
APP_EXE="python"

# Сколько ждать готовности / завершения, секунды.
READY_TIMEOUT_S=30
STOP_TIMEOUT_S=15

die() {
    printf 'service.sh: %s\n' "$*" >&2
    exit 1
}

note() {
    printf 'service.sh: %s\n' "$*" >&2
}

# Жив ли процесс и правда ли это тот самый процесс (страховка от устаревшего pidfile).
# `exe` — ожидаемое имя исполняемого файла (argv[0]); пустое значение отключает сверку.
is_our_pid() {
    local pid="${1:-}" marker="$2" exe="${3:-}" cmdline
    [ -n "$pid" ] || return 1
    [ -r "/proc/$pid/cmdline" ] || return 1
    cmdline="$(cmdline_of "$pid")"
    case "$cmdline" in
        *"$marker"*) ;;
        *) return 1 ;;
    esac
    [ -n "$exe" ] || return 0
    case "$(basename -- "${cmdline%% *}")" in
        "$exe"*) return 0 ;;
        *) return 1 ;;
    esac
}

pid_of_file() {
    local pidfile="$1" marker="$2" exe="${3:-}" pid
    [ -f "$pidfile" ] || return 1
    pid="$(cat "$pidfile")"
    is_our_pid "$pid" "$marker" "$exe" || return 1
    printf '%s' "$pid"
}

# Процессы, поднятые вне этого скрипта (ручная строка из §2 хендоффа, `make dev` —
# обычные случаи). Свой собственный процесс из выдачи исключаем.
pids_by_cmdline() {
    local marker="$1" exe="${2:-}" pid
    for pid in $(pgrep -f -- "$marker" || true); do
        [ "$pid" = "$$" ] && continue
        [ "$pid" = "$PPID" ] && continue
        is_our_pid "$pid" "$marker" "$exe" || continue
        printf '%s\n' "$pid"
    done
}

cmdline_of() {
    tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null || true
}

# Кто держит порт. Свои процессы видны и без root; перезагрузчик uvicorn попадает
# сюда, хотя по cmdline не опознаётся.
port_holder_pids() {
    ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
}

# Компонент жив, если его нашёл pid-файл, сторож по cmdline или — для веба и Ollama —
# занятый порт: у перезагрузчика uvicorn cmdline не опознаётся (`spawn_main`), и без
# проверки порта он оставался бы для status невидимым, а порт занятым. Держатель порта
# опознаётся портом, а не cmdline, поэтому сверка argv[0] к нему не применяется.
component_pids() {
    local marker="$1" exe="${2:-}" port="${3:-}" pid
    {
        pids_by_cmdline "$marker" "$exe"
        [ -n "$port" ] && port_holder_pids "$port"
    } | sort -u
}

# Что именно переиспользуется, если компонент уже поднят: без этого «переиспользую»
# и «не запущен» в одном выводе противоречили бы друг другу.
holders_note() {
    local marker="$1" exe="${2:-}" port="${3:-}" pid out=""
    while read -r pid; do
        [ -n "$pid" ] && out="$out $pid"
    done < <(component_pids "$marker" "$exe" "$port")
    if [ -n "$out" ]; then
        printf ' (pid%s)' "$out"
    fi
    return 0
}

# Порт обязан быть свободен после stop: это последняя проверка фактом, после того как
# компоненты погашены выше.
free_port() {
    local port="$1" name="$2" pid deadline
    local -a pids=()
    while read -r pid; do
        [ -n "$pid" ] && pids+=("$pid")
    done < <(port_holder_pids "$port")
    [ "${#pids[@]}" -eq 0 ] && return 0
    for pid in "${pids[@]}"; do
        note "$name: порт $port держит pid $pid ($(cmdline_of "$pid")) — останавливаю"
    done
    kill "${pids[@]}" 2>/dev/null || true
    deadline=$((SECONDS + STOP_TIMEOUT_S))
    while [ "$SECONDS" -lt "$deadline" ]; do
        [ -z "$(port_holder_pids "$port")" ] && return 0
        sleep 0.3
    done
    while read -r pid; do
        [ -n "$pid" ] || continue
        note "pid $pid не освободил порт $port за ${STOP_TIMEOUT_S} с — SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    done < <(port_holder_pids "$port")
}

health_ok() {
    curl -sf --max-time 3 "$1" >/dev/null 2>&1
}

wait_for_health() {
    local url="$1" name="$2" deadline=$((SECONDS + READY_TIMEOUT_S))
    while [ "$SECONDS" -lt "$deadline" ]; do
        health_ok "$url" && return 0
        sleep 0.5
    done
    note "$name не отвечает на $url за ${READY_TIMEOUT_S} с"
    return 1
}

# Запуск отдельным сеансом (setsid) с отвязкой от терминала (nohup): процесс обязан
# пережить выход и этого скрипта, и обёртки wsl.exe, через которую его запускают с Windows.
# Pid пишет сам потомок — так он указывает на процесс, который в итоге и работает
# (run-local.sh делает exec), а не на промежуточный setsid.
launch_detached() {
    local pidfile="$1" logfile="$2" stdout_note="$3"
    shift 3
    : >"$logfile"
    printf '%s\n' "$stdout_note" >>"$logfile"
    # Убираем прошлый pid-файл: ожидание ниже считает готовностью непустой файл,
    # а устаревший pid в нём сделал бы проверку живого процесса ложной.
    rm -f "$pidfile"
    setsid nohup bash -c 'printf "%s\n" "$$" >"$1"; shift; exec "$@"' _ "$pidfile" "$@" \
        >>"$logfile" 2>&1 </dev/null &
    local deadline=$((SECONDS + READY_TIMEOUT_S))
    while [ "$SECONDS" -lt "$deadline" ]; do
        [ -s "$pidfile" ] && return 0
        sleep 0.2
    done
    return 1
}

start_ollama() {
    if health_ok "$OLLAMA_URL"; then
        note "Ollama уже отвечает на $OLLAMA_URL — переиспользую$(holders_note "$OLLAMA_MARKER" "$OLLAMA_EXE" "$OLLAMA_PORT")"
        return 0
    fi
    local stale
    if stale="$(pid_of_file "$OLLAMA_PIDFILE" "$OLLAMA_MARKER" "$OLLAMA_EXE")"; then
        die "pid $stale жив, но $OLLAMA_URL не отвечает — разбирайтесь по логу $OLLAMA_LOG"
    fi
    [ -x "$OLLAMA_BIN" ] || die "не найден исполняемый $OLLAMA_BIN"
    mkdir -p "$STATE_DIR"
    launch_detached "$OLLAMA_PIDFILE" "$OLLAMA_LOG" \
        "service.sh: ollama serve | OLLAMA_KEEP_ALIVE=0 | лог $OLLAMA_LOG" \
        env "OLLAMA_HOST=$OLLAMA_HOST_VALUE" OLLAMA_KEEP_ALIVE=0 OLLAMA_MAX_LOADED_MODELS=2 "$OLLAMA_BIN" serve \
        || die "не удалось записать pid в $OLLAMA_PIDFILE"
    wait_for_health "$OLLAMA_URL" "Ollama" || die "сервер не поднялся — смотрите $OLLAMA_LOG"
    note "Ollama поднят (pid $(cat "$OLLAMA_PIDFILE"), OLLAMA_KEEP_ALIVE=0)"
}

start_web() {
    if health_ok "$WEB_URL"; then
        note "веб уже отвечает на $WEB_URL — переиспользую$(holders_note "$WEB_MARKER" "$APP_EXE" "$WEB_PORT")"
        return 0
    fi
    local stale
    if stale="$(pid_of_file "$WEB_PIDFILE" "$WEB_MARKER" "$APP_EXE")"; then
        die "pid $stale жив, но $WEB_URL не отвечает — смотрите $WEB_LOG"
    fi
    launch_detached "$WEB_PIDFILE" "$WEB_LOG" \
        "service.sh: web | без --reload | лог $WEB_LOG" \
        "$REPO_ROOT/scripts/run-local.sh" web --host 127.0.0.1 --port "$WEB_PORT" \
        || die "не удалось записать pid в $WEB_PIDFILE"
    wait_for_health "$WEB_URL" "веб" || die "сервер не поднялся — смотрите $WEB_LOG"
    note "веб поднят (pid $(cat "$WEB_PIDFILE"), $WEB_URL)"
}

start_worker() {
    local existing
    if existing="$(pid_of_file "$WORKER_PIDFILE" "$WORKER_MARKER" "$APP_EXE")"; then
        note "воркер уже работает (pid $existing) — переиспользую"
        return 0
    fi
    launch_detached "$WORKER_PIDFILE" "$WORKER_LOG" \
        "service.sh: worker | лог $WORKER_LOG" \
        "$REPO_ROOT/scripts/run-local.sh" worker \
        || die "не удалось записать pid в $WORKER_PIDFILE"
    local pid
    pid="$(cat "$WORKER_PIDFILE")"
    sleep 2 # intentional-sleep: воркер не имеет health-эндпоинта, даём ему дойти до цикла опроса
    if ! is_our_pid "$pid" "$WORKER_MARKER" "$APP_EXE"; then
        tail -n 5 "$WORKER_LOG" >&2 || true
        die "воркер завершился сразу после старта — смотрите $WORKER_LOG"
    fi
    grep -q 'started poll_interval' "$WORKER_LOG" \
        || note "воркер жив (pid $pid), но строки о старте опроса в логе ещё нет — смотрите $WORKER_LOG"
    note "воркер поднят (pid $pid)"
}

# Остановка компонента. Если pid-файла нет (процесс подняли вручную, `make dev`,
# строкой из хендоффа), компонент всё равно гасится — иначе `stop` оставлял бы
# живым процесс, о котором этот скрипт не знает, и порт 8000 оставался бы занят.
stop_component() {
    local pidfile="$1" marker="$2" name="$3" exe="${4:-}" port="${5:-}" pid how deadline
    local -a pids=()
    if pid="$(pid_of_file "$pidfile" "$marker" "$exe")"; then
        note "останавливаю $name (pid $pid)"
        kill "$pid" 2>/dev/null || true
    else
        rm -f "$pidfile"
        while read -r pid; do
            [ -n "$pid" ] && pids+=("$pid")
        done < <(component_pids "$marker" "$exe" "$port")
        if [ "${#pids[@]}" -eq 0 ]; then
            note "$name не запущен"
            return 0
        fi
        for pid in "${pids[@]}"; do
            how="по cmdline"
            if [ -n "$port" ] && port_holder_pids "$port" | grep -qx "$pid"; then
                how="держит порт $port"
            fi
            note "$name поднят вне этого скрипта — останавливаю pid $pid ($how): $(cmdline_of "$pid")"
        done
        kill "${pids[@]}" 2>/dev/null || true
    fi
    # Ждём, пока не останется ни процессов по cmdline, ни держателя порта.
    deadline=$((SECONDS + STOP_TIMEOUT_S))
    while [ "$SECONDS" -lt "$deadline" ]; do
        [ -z "$(component_pids "$marker" "$exe" "$port")" ] && break
        sleep 0.3
    done
    while read -r pid; do
        [ -n "$pid" ] || continue
        note "pid $pid не завершился за ${STOP_TIMEOUT_S} с — SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    done < <(component_pids "$marker" "$exe" "$port")
    rm -f "$pidfile"
}

cmd_start() {
    mkdir -p "$STATE_DIR"
    start_ollama
    start_web
    start_worker
    printf 'service.sh: стек готов — %s\n' "$WEB_URL" >&2
    cmd_status
}

cmd_stop() {
    stop_component "$WORKER_PIDFILE" "$WORKER_MARKER" "воркер" "$APP_EXE"
    stop_component "$WEB_PIDFILE" "$WEB_MARKER" "веб" "$APP_EXE" "$WEB_PORT"
    free_port "$WEB_PORT" "веб"
    stop_component "$OLLAMA_PIDFILE" "$OLLAMA_MARKER" "Ollama" "$OLLAMA_EXE" "$OLLAMA_PORT"
    free_port "$OLLAMA_PORT" "Ollama"
    note "стек остановлен"
}

cmd_restart() {
    cmd_stop
    cmd_start
}

# Порядок аргументов: имя, pid-файл, шаблон cmdline, ожидаемый argv[0], URL health (пусто —
# нет проверки), порт (пусто — компонент порт не слушает).
report_component() {
    local name="$1" pidfile="$2" marker="$3" exe="${4:-}" url="${5:-}" port="${6:-}" pid state
    local -a pids=()
    if pid="$(pid_of_file "$pidfile" "$marker" "$exe")"; then
        state="работает (pid $pid)"
    else
        while read -r pid; do
            [ -n "$pid" ] && pids+=("$pid")
        done < <(component_pids "$marker" "$exe" "$port")
        if [ "${#pids[@]}" -eq 0 ]; then
            printf '  %s: не запущен\n' "$name"
            return 0
        fi
        state="работает (pid ${pids[*]}), поднят вне этого скрипта"
    fi
    if [ -n "$url" ]; then
        if health_ok "$url"; then
            state="$state, $url отвечает"
        else
            state="$state, но $url НЕ отвечает"
        fi
    fi
    printf '  %s: %s\n' "$name" "$state"
}

cmd_status() {
    report_component "Ollama" "$OLLAMA_PIDFILE" "$OLLAMA_MARKER" "$OLLAMA_EXE" "$OLLAMA_URL" "$OLLAMA_PORT"
    report_component "веб" "$WEB_PIDFILE" "$WEB_MARKER" "$APP_EXE" "$WEB_URL" "$WEB_PORT"
    report_component "воркер" "$WORKER_PIDFILE" "$WORKER_MARKER" "$APP_EXE"
}

usage() {
    printf 'Использование: scripts/service.sh {start|stop|restart|status}\n' >&2
}

main() {
    case "${1:-}" in
        start) cmd_start ;;
        stop) cmd_stop ;;
        restart) cmd_restart ;;
        status) cmd_status ;;
        --help | -h | help | "") usage; [ -n "${1:-}" ] || exit 2 ;;
        *) usage; die "неизвестное действие: $1" ;;
    esac
}

main "$@"
