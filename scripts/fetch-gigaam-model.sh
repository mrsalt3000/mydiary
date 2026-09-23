#!/usr/bin/env bash
#
# scripts/fetch-gigaam-model.sh — скачать веса второго STT-бэкенда GigaAM-v2 CTC+LM (T45).
#
# Веса не лежат в репозитории (`data/**` в `.gitignore`) и качаются отдельным шагом —
# так же, как веса NLI (`scripts/fetch-nli-model.sh`). Приложение их **никогда** не
# качает само: сайдкар запускается с `HF_HUB_OFFLINE=1`
# (`src/diary/domain/transcription/gigaam.py`), поэтому промах кэша — это понятный
# отказ «рантайм не провижинен», а не многогигабайтная загрузка в момент нажатия
# «Расшифровать» (ADR-017 §4, DoD 2 задачи T45).
#
# Куда качается: `$APP_GIGAAM_MODEL_DIR`, по умолчанию `data/models/gigaam`. Скрипт
# наполняет **отдельный HF-кэш** внутри этого каталога — `$TARGET/hub`, — который не
# смешивается с личным `~/.cache/huggingface` пользователя. Приложение указывает
# сайдкару именно на `$TARGET` (`HF_HOME` ребёнка = `APP_GIGAAM_MODEL_DIR`), поэтому
# распознавание читает ровно те файлы, что положил этот скрипт.
#
# Кто именно качает — **huggingface_hub.snapshot_download** интерпретатором сайдкара.
# Это не деталь реализации, а измеренный факт (2026-09-24): обёртка грузит языковую
# модель через `AutoProcessor.from_pretrained` → `pyctcdecode.load_from_hf_hub` →
# `snapshot_download`, а тот в офлайне требует **кэш, записанный им самим** —
# перечень файлов коммита (`trees/<commit>.json`) и `blobs/`. Снапшот, собранный
# вручную (curl'ом в `snapshots/<revision>/`), этому требованию не отвечает: пофайловый
# `hf_hub_download`, `AutoConfig` и `AutoTokenizer` из него читаются, а `AutoProcessor`
# уходит в API и падает `OfflineModeIsEnabled` — то есть распознавание не работает,
# хотя файлы на месте и sha256 совпадают. Поэтому качает та же библиотека, которая
# читает: раскладка кэша не может разойтись с читателем.
#
# Порядок провижининга: сначала `scripts/setup-gigaam-venv.sh` (нужен интерпретатор),
# потом этот скрипт. Интерпретатор берётся из `APP_GIGAAM_PYTHON`, иначе — путь по
# умолчанию (тот же, что печатает скрипт сборки venv).
#
# Использование:
#   scripts/fetch-gigaam-model.sh            # скачать (повторный запуск ничего не качает)
#   scripts/fetch-gigaam-model.sh --check    # только проверить уже скачанное: без сети и без интерпретатора
#
# Ревизия и sha256 — пин, а не «последняя версия»: обёртка исполняется локально через
# `trust_remote_code`, поэтому непиннутая ревизия означала бы исполнение неизвестного
# кода (`docs/stt-backends-analysis.md` §11, риск 1). Ревизия ниже — та самая, на
# которой сняты замеры T42, и то же значение стоит в `revision:` обоих примеров
# конфига; тест сверяет их между собой, чтобы пины не разъехались.
#
# Объём — 2 637 136 001 B (2.46 GiB) по файлам снапшота (измерено T42 §4.5); из них
# 2.14 GB — языковая модель KenLM, которая живёт на CPU и в VRAM не уходит (§9).
# Скрипт ничего не пишет внутрь репозитория, кроме самого каталога весов, который в
# git не попадает.

set -euo pipefail

MODEL_ID="waveletdeboshir/gigaam-ctc-with-lm"
REVISION="07d97cf8a4f178b68679cafae0600c8487877301"

#: Префикс провижининга по умолчанию — **тот же**, что у `scripts/setup-gigaam-venv.sh`
#: (тест сверяет два скрипта между собой). Свой префикс: `APP_GIGAAM_PYTHON` здесь,
#: `GIGAAM_VENV_PREFIX` — там.
DEFAULT_PREFIX="$HOME/.local/share/mydiary/gigaam"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${APP_GIGAAM_MODEL_DIR:-$ROOT/data/models/gigaam}"
SNAPSHOT="$TARGET/hub/models--${MODEL_ID//\//--}/snapshots/$REVISION"
MODE="fetch"

# path|bytes|sha256 — файлы снапшота закреплённой ревизии.
# Три крупных файла: числа из `docs/stt-backends-analysis.md` §4.5 (измерено
# 2026-09-23). Остальные файлы снапшота: те же 38 674 B суммарно (§4.5), размеры и
# sha256 измерены 2026-09-24 с того же снапшота той же ревизии — пинятся здесь, чтобы
# подмена мелкого файла (config.json, `gigaam_transformers.py`, словарь) не прошла
# незамеченной: именно эти файлы и исполняются через `trust_remote_code`.
FILES=(
  "model.safetensors|465343856|c7bd7e277cb601bf55036251be654dd374c455313edefaa69d32e6ec1f9c7161"
  "language_model/ru_3gram.bin|2142431571|9c2a8143d33c234f881f0f7072bd0ed12c5dd2d697328410c317ed50892d70ee"
  "language_model/unigrams.txt|29321792|6c5edc3b6134e2a64b04696a1c00abccb660a86776b8016c43108f80f35d2fae"
  "language_model/attrs.json|108|0172bab6b9ebe5bccb54ffc3ca736ee02b0dbc954afd25c12e3f20ea0e5b7f7e"
  "config.json|1347|7ef5a5d044b8e0c7fc3c97baa3a55711e9d0435b9491dca88abb4ce9faf6433f"
  "preprocessor_config.json|546|4aac1fdb0120710fd95ac295a5c675726ba25c7e50d7015096a4d944905e6d3e"
  "tokenizer_config.json|598|a8d9a205e302fdfd16ede0aeb4bab65a83b83b1e97b2ba34f363d3660eb6bc33"
  "special_tokens_map.json|41|a127ef0131bf1129d2d73c490ba145ff981c92bcc93bbf0061ad4b5988ca179c"
  "added_tokens.json|30|237497ffd7dd9cbbc03c5174d60a3295771af22e19b36d251986d83c63be5870"
  "alphabet.json|368|daf8fcfdd001ec690fd0bb73b07b7e1a7ecae7260040bfcf2c32d69a2be790ec"
  "vocab.json|398|d89f7341db18ca4f9ad96a0d96629658e2e99dbba78ad63cda3fd697d1592cd1"
  "gigaam_transformers.py|10793|ee5d61bed5f8d5dde814ead3bcae68bdfb629c56310320005aa701ce43726f3e"
  "encoder.py|19956|1cd0a2bc1b8a97631d2bc196c6108a0b013e8552939f2ef4f051160319221dae"
  "README.md|3014|25fa0cd70999107a0cdbcbcdee89f112db8d16c4459ddc6701a487564309367b"
  ".gitattributes|1583|207a6798bef5d0db744846ecede50897b013b260d276f121e648ebee8e33ce94"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      MODE="check"
      shift
      ;;
    -h|--help)
      sed -n '2,50p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "неизвестный аргумент: $1" >&2
      exit 2
      ;;
  esac
done

verify() {
  # verify FILE EXPECTED_BYTES EXPECTED_SHA256 — размер и sha256, оба названы.
  local file="$1" expected_bytes="$2" expected_sha="$3"
  local actual_sha actual_bytes
  actual_sha="$(sha256sum "$file" | cut -d' ' -f1)"
  actual_bytes="$(stat -Lc %s "$file")"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "sha256 не совпал для $file" >&2
    echo "  ожидалось: $expected_sha" >&2
    echo "  получено:  $actual_sha" >&2
    return 1
  fi
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "размер не совпал для $file" >&2
    echo "  ожидалось: $expected_bytes байт" >&2
    echo "  получено:  $actual_bytes байт" >&2
    return 1
  fi
  return 0
}

if [[ "$MODE" == "check" ]]; then
  status=0
  total=0
  for entry in "${FILES[@]}"; do
    IFS='|' read -r path expected_bytes expected_sha <<<"$entry"
    file="$SNAPSHOT/$path"
    if [[ -f "$file" ]]; then
      if verify "$file" "$expected_bytes" "$expected_sha"; then
        echo "ok: $path"
        total=$((total + expected_bytes))
      else
        status=1
      fi
    else
      echo "нет: $path"
      status=1
    fi
  done
  echo "каталог весов: $TARGET"
  if [[ "$status" == "0" ]]; then
    echo "снапшот на месте: ${total} B"
  else
    echo "скачать: scripts/fetch-gigaam-model.sh" >&2
  fi
  exit "$status"
fi

# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
PYTHON="${APP_GIGAAM_PYTHON:-$DEFAULT_PREFIX/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "нет интерпретатора сайдкара: $PYTHON" >&2
  echo "Сначала соберите его: scripts/setup-gigaam-venv.sh" >&2
  echo "(или укажите существующий: APP_GIGAAM_PYTHON=<…>/venv/bin/python)" >&2
  exit 2
fi

mkdir -p "$TARGET"
echo "каталог весов: $TARGET"
echo "ревизия:       $REVISION ($MODEL_ID)"
echo "интерпретатор: $PYTHON"

# `HF_HUB_OFFLINE=0` — потому что это и есть явный сетевой шаг провижининга: офлайн-пин
# относится к распознаванию, а не к скачиванию весов. `HF_HUB_DISABLE_TELEMETRY=1` —
# та же приватностная настройка, что у сайдкара. `max_workers=4` — умеренная
# параллельность: соединения к CDN иногда принимаются и «замирают» (см. ниже), и
# четыре потока переносят это легче, чем восемь.
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 "$PYTHON" - "$MODEL_ID" "$REVISION" "$TARGET/hub" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, revision, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    cache_dir=cache_dir,
    max_workers=4,
)
print(f"снапшот: {path}")
PY

echo "== проверка пинов (sha256 + размер)"
for entry in "${FILES[@]}"; do
  IFS='|' read -r path expected_bytes expected_sha <<<"$entry"
  file="$SNAPSHOT/$path"
  if [[ ! -f "$file" ]]; then
    echo "нет: $path" >&2
    echo "загрузка неполна; повторите запуск" >&2
    exit 1
  fi
  verify "$file" "$expected_bytes" "$expected_sha" || {
    echo "файл не совпал с пином; повторите загрузку" >&2
    exit 1
  }
done

echo "готово: $SNAPSHOT"
echo "проверить:  scripts/fetch-gigaam-model.sh --check"
echo "приложению: APP_GIGAAM_MODEL_DIR=${APP_GIGAAM_MODEL_DIR:-<по умолчанию>} (каталог весов)"
echo "профиль transcription в configs/models.yaml должен нести revision: $REVISION"
