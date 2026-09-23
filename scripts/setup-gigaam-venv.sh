#!/usr/bin/env bash
#
# scripts/setup-gigaam-venv.sh — собрать рантайм второго STT-бэкенда (GigaAM-v2 CTC+LM, T45).
#
# Зачем отдельный venv, а не extra в приложении. Способ установки выбран замером
# спайка T42 и записан решением: `docs/stt-backends-analysis.md` §10, `ADR-017` §7.
# `unload()` возвращает VRAM, но не RAM (пик контура — `VmHWM` 4553.2 MiB), поэтому
# «выгрузка» — это выход процесса-сайдкара; torch/CUDA/KenLM живут в своём
# интерпретаторе вне репозитория, а venv приложения (numpy 2.5.3, 63 пакета) не
# меняется вовсе. Поэтому в `pyproject.toml` нет extra `stt-gigaam`: extra ставится
# именно в venv приложения и понизил бы его numpy (метаданные `pyctcdecode 0.5.0`
# требуют `numpy<2`). Пин — не extra, пин — этот скрипт.
#
# Использование:
#   scripts/setup-gigaam-venv.sh                 # собрать venv
#   scripts/setup-gigaam-venv.sh --check         # проверить уже собранное, без сети
#   scripts/setup-gigaam-venv.sh --require-cuda  # считать отсутствие GPU ошибкой
#   scripts/setup-gigaam-venv.sh --prefix DIR    # куда ставить (по умолчанию ниже)
#   scripts/setup-gigaam-venv.sh --python PY     # базовый интерпретатор для venv
#
# Куда ставится: `$GIGAAM_VENV_PREFIX` (переменная только этого скрипта, приложение её
# не читает), по умолчанию `~/.local/share/mydiary/gigaam`; сам venv — `<prefix>/venv`.
# Префикс внутри репозитория отвергается: рантайм сайдкара в репозитории не живёт.
# Веса качает **отдельный** скрипт — `scripts/fetch-gigaam-model.sh`; этот скрипт
# сеть не трогает вообще (только установка пакетов), весов не скачивает и по умолчанию
# ничего не пишет внутрь репозитория.
#
# Что должен знать пользователь (и что не прячется):
#   * `kenlm 0.3.0` существует **только sdist** — это C++/cmake-сборка, поэтому нужен
#     системный тулчейн `cmake`, `g++`, `make` (измерено: `cmake 4.2.3`, `g++ 15.2.0`,
#     сборка 38.33 с; `docs/stt-backends-analysis.md` §4.3). Скрипт проверяет тулчейн
#     **до** установки и говорит, чего не хватает, вместо падения внутри pip;
#   * `pyctcdecode 0.5.0` ставится с `--no-deps`: его метаданные требуют `numpy<2`, а
#     работает он на numpy 2 (проверено всеми прогонами T42, §4.2/§11). Без `--no-deps`
#     резолвер понизил бы numpy — то есть поставил бы в сайдкар другую арифметику,
#     чем измеренная;
#   * место на диске — **6.5 GB** (измерено `du -sh` спайк-venv: `nvidia/*` 3.6 GB,
#     `torch/` 1.7 GB, `triton/` 892 MB; §4.5). Это плата за изоляцию, названная в §10.
#
# Системный `ffmpeg` не нужен: аудио декодирует сайдкар через PyAV, который несёт
# библиотеки FFmpeg в своём колесе (§5, проверено прогоном с пустым PATH).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Пины — измеренный набор T42 §4.1 (2026-09-23). Не «примерно совместимые», а те
# самые версии, на которых сняты числа VRAM/латентности и на которых работает
# `pyctcdecode` при numpy 2. Точка пина — этот файл; `configs/models.example.yaml`
# пинит совместимую **ревизию** весов (и то же делает `fetch-gigaam-model.sh`).
# ---------------------------------------------------------------------------
TORCH_PINS=(torch==2.14.0+cu126 torchaudio==2.11.0+cu126)
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
HF_PINS=(transformers==5.17.0 safetensors==0.8.0 av==18.1.0 numpy==2.5.3 huggingface_hub==1.32.0)
LM_PINS=(kenlm==0.3.0 pyctcdecode==0.5.0 pygtrie==2.6.2)
#: Версия Python, на которой набор измерен (спайк — 3.12.14). Другая версия — это
#: другой набор колёс (и другая сборка `kenlm`), поэтому она требует осознанной правки.
PYTHON_VERSION="3.12"

#: Префикс провижининга по умолчанию. Тот же путь знает
#: `scripts/fetch-gigaam-model.sh` — она ищет там интерпретатор; тест сверяет два
#: скрипта между собой, чтобы пути не разъехались.
DEFAULT_PREFIX="$HOME/.local/share/mydiary/gigaam"
PREFIX="${GIGAAM_VENV_PREFIX:-$DEFAULT_PREFIX}"
PYTHON=""
MODE="install"
REQUIRE_CUDA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      MODE="check"
      shift
      ;;
    --require-cuda)
      REQUIRE_CUDA=1
      shift
      ;;
    --prefix)
      PREFIX="${2:?--prefix требует каталог}"
      shift 2
      ;;
    --prefix=*)
      PREFIX="${1#--prefix=}"
      shift
      ;;
    --python)
      PYTHON="${2:?--python требует интерпретатор}"
      shift 2
      ;;
    --python=*)
      PYTHON="${1#--python=}"
      shift
      ;;
    -h|--help)
      sed -n '2,45p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "неизвестный аргумент: $1" >&2
      exit 2
      ;;
  esac
done

if command -v realpath >/dev/null 2>&1; then
  PREFIX="$(realpath -m "$PREFIX")"
fi
case "$PREFIX" in
  "$ROOT"|"$ROOT"/*)
    echo "отказ: префикс внутри репозитория ($PREFIX)." >&2
    echo "Рантайм сайдкара ставится вне репозитория — см. scripts/setup-gigaam-venv.sh." >&2
    exit 2
    ;;
esac

VENV="$PREFIX/venv"
VENV_PY="$VENV/bin/python"
ALL_PINS=("${TORCH_PINS[@]}" "${HF_PINS[@]}" "${LM_PINS[@]}")

# verify — сверить установленные версии с пинами и показать устройство исполнения.
# Пины передаются аргументами, а не перепечатываются здесь: у них один источник (выше).
verify() {
  local extra=()
  [[ "$REQUIRE_CUDA" == "1" ]] && extra=("--require-cuda")
  "$VENV_PY" - "${extra[@]}" "${ALL_PINS[@]}" <<'PY'
import sys

import torch
from importlib.metadata import version

arguments = sys.argv[1:]
require_cuda = "--require-cuda" in arguments
pins = [argument for argument in arguments if argument != "--require-cuda"]

mismatches = []
for pin in pins:
    name, expected = pin.split("==", 1)
    try:
        actual = version(name)
    except Exception:  # not installed at all
        actual = None
    if actual != expected:
        mismatches.append(f"{name}: ожидалось {expected}, получено {actual}")
    else:
        print(f"ok   {name}=={actual}")

cuda = torch.cuda.is_available()
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {cuda}")
if cuda:
    properties = torch.cuda.get_device_properties(0)
    print(f"device: {torch.cuda.get_device_name(0)}  (VRAM {properties.total_memory // (1024 * 1024)} MiB)")

if mismatches:
    for mismatch in mismatches:
        print(f"НЕ СОВПАЛО: {mismatch}", file=sys.stderr)
    sys.exit(1)
if require_cuda and not cuda:
    print(
        "GPU не виден этому интерпретатору (--require-cuda): нужны NVIDIA-драйвер в WSL 2 "
        "и `device: cuda`; профиль с `device: cpu` этот отказ не блокирует.",
        file=sys.stderr,
    )
    sys.exit(1)
sys.exit(0)
PY
}

if [[ "$MODE" == "check" ]]; then
  if [[ ! -x "$VENV_PY" ]]; then
    echo "нет интерпретатора: $VENV_PY" >&2
    echo "собрать: scripts/setup-gigaam-venv.sh" >&2
    exit 1
  fi
  echo "интерпретатор сайдкара: $VENV_PY"
  verify
  exit $?
fi

# ---------------------------------------------------------------------------
# Установка
# ---------------------------------------------------------------------------
for tool in cmake g++ make; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "нет системного тулчейна: не найден $tool" >&2
    echo "kenlm 0.3.0 собирается из sdist (C++/cmake) — установите cmake, g++, make." >&2
    exit 1
  fi
done

if [[ -z "$PYTHON" ]]; then
  if command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
    PYTHON="python$PYTHON_VERSION"
  else
    PYTHON="python3"
  fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "базовый интерпретатор не найден: $PYTHON" >&2
  exit 1
fi
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)' || {
  echo "$PYTHON старее Python 3.12" >&2
  exit 1
}

#: Пакеты ставит `uv`, если он есть (измеренный путь T42 — `uv 0.12.7`, и в T45 живьём
#: проверен именно он); без `uv` — тот же набор пинов через `python -m venv` + `pip`.
#: Путь `pip` в T45 живьём не выполнялся: это резервный вариант, а не измеренный.
INSTALLER="pip"
if command -v uv >/dev/null 2>&1; then
  INSTALLER="uv"
fi

echo "префикс: $PREFIX"
echo "venv:    $VENV"
echo "питон:   $PYTHON ($PYTHON_VERSION+)"
echo "пины:    ${ALL_PINS[*]}"

if [[ "$INSTALLER" == "uv" ]]; then
  if [[ ! -x "$VENV_PY" ]]; then
    uv venv "$VENV" --python "$PYTHON"
  fi
  install() { uv pip install --python "$VENV_PY" "$@"; }
else
  if [[ ! -x "$VENV_PY" ]]; then
    "$PYTHON" -m venv "$VENV"
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
  fi
  install() { "$VENV_PY" -m pip install "$@"; }
fi

# 1. torch/torchaudio — с индекса PyTorch (в PyPI колеса помечены иначе: локальная
#    метка `+cu126` берётся оттуда, §4.2).
echo "== torch/torchaudio ($TORCH_INDEX)"
install --index-url "$TORCH_INDEX" "${TORCH_PINS[@]}"

# 2. HF-обвязка.
echo "== обвязка Hugging Face"
install --index-url https://pypi.org/simple "${HF_PINS[@]}"

# 3. LM-контур — обязательно без зависимостей (см. шапку): иначе numpy понижается.
echo "== LM-контур (--no-deps)"
install --index-url https://pypi.org/simple --no-deps "${LM_PINS[@]}"

echo "== проверка"
verify
echo
echo "готово. Укажите приложению интерпретатор сайдкара:"
echo "  APP_GIGAAM_PYTHON=$VENV_PY"
echo "Веса (2.64 GB) качает отдельный скрипт: scripts/fetch-gigaam-model.sh"
