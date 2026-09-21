#!/usr/bin/env bash
#
# scripts/fetch-nli-model.sh — скачать веса NLI-судьи для проверки сводок (T24a).
#
# Инструмент необязательный, поэтому веса лежат вне репозитория и скачиваются
# отдельно: `data/models/nli/` по умолчанию (переопределяется APP_NLI_MODEL_DIR).
# Файлы пинуются по sha256 — тот же принцип, что и у закреплённого релиза Ollama:
# подменённый или недокачанный файл отвергается, а не «работает как-то иначе».
#
# Модель: rubert-base-cased-nli-threeway (русский трёхзначный NLI: entailment /
# contradiction / neutral), ONNX-конверсия metalmon80/rubert-nli-threeway-onnx.
# Замеры, обоснование выбора и границы применимости — docs/nli-validation-analysis.md.
#
# Использование:
#   scripts/fetch-nli-model.sh            # в APP_NLI_MODEL_DIR или data/models/nli
#   scripts/fetch-nli-model.sh --fp16     # лёгкий вариант (356 МБ вместо 712 МБ)
#   scripts/fetch-nli-model.sh --check    # только проверить уже скачанное
#
# Скрипт ничего не пишет внутрь репозитория, кроме самого каталога весов,
# который не попадает в git (data/ — приватный каталог приложения).

set -euo pipefail

REPO="metalmon80/rubert-nli-threeway-onnx"
REVISION="main"
MODEL_FP32="model.onnx"
MODEL_FP16="model.fp16.onnx"
SHA_FP32="ee870cefeddcc1b79ed1eaf8c9cf3ec331d92f0d984945ab6aa32d017aac6c2c"
SHA_FP16="e838a882102d328e74eef45b52700755ff63545cb39ad132b717c44ce714a48e"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${APP_NLI_MODEL_DIR:-$ROOT/data/models/nli}"
MODE="fp32"

for argument in "$@"; do
  case "$argument" in
    --fp16) MODE="fp16" ;;
    --check) MODE="check" ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "неизвестный аргумент: $argument" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  fp16) MODEL_FILE="$MODEL_FP16"; MODEL_SHA="$SHA_FP16" ;;
  *) MODEL_FILE="$MODEL_FP32"; MODEL_SHA="$SHA_FP32" ;;
esac

verify() {
  local file="$1" expected="$2"
  local actual
  actual="$(sha256sum "$file" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 не совпал для $file" >&2
    echo "  ожидалось: $expected" >&2
    echo "  получено:  $actual" >&2
    return 1
  fi
  return 0
}

if [[ "$MODE" == "check" ]]; then
  status=0
  for name in "$MODEL_FP32:$SHA_FP32" "$MODEL_FP16:$SHA_FP16"; do
    file="${name%%:*}"; sha="${name##*:}"
    if [[ -f "$TARGET/$file" ]]; then
      if verify "$TARGET/$file" "$sha"; then echo "ok: $file"; else status=1; fi
    else
      echo "нет: $file"
    fi
  done
  [[ -f "$TARGET/tokenizer.json" ]] && echo "ok: tokenizer.json" || { echo "нет: tokenizer.json"; status=1; }
  echo "каталог: $TARGET"
  exit "$status"
fi

mkdir -p "$TARGET"
echo "каталог весов: $TARGET"
echo "режим: $MODE"

for file in "$MODEL_FILE" tokenizer.json config.json; do
  if [[ -f "$TARGET/$file" ]]; then
    echo "уже есть: $file"
  else
    echo "скачиваю: $file"
    curl -fL --progress-bar -o "$TARGET/$file" \
      "https://huggingface.co/$REPO/resolve/$REVISION/$file"
  fi
done

if ! verify "$TARGET/$MODEL_FILE" "$MODEL_SHA"; then
  rm -f "$TARGET/$MODEL_FILE"
  echo "файл удалён; повторите загрузку" >&2
  exit 1
fi

echo "готово: $TARGET/$MODEL_FILE (sha256 совпал)"
echo "проверить: scripts/fetch-nli-model.sh --check"
