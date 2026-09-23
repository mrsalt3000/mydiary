Контекст
Сейчас STT — faster-whisper large-v3, INT8 / int8_float16, language=ru. Целевой вариант по качеству русского — GigaAM-v2 CTC+LM. Машина: WSL2, RTX 3060 6 ГБ VRAM, 32 ГБ RAM. STT и LLM в VRAM одновременно не удерживаются; после транскрипта модель выгружается. Сырое аудио и raw неизменяемы, результат STT пишется отдельной версией.

GigaAM-v2 CTC+LM
Официальный gigaam.load_model("v2_ctc"|"ctc") — greedy CTC без ngram LM, клип до ~25 с. Это не CTC+LM. 

CTC+LM — Hugging Face waveletdeboshir/gigaam-ctc-with-lm: GigaAM-v2-CTC, beamsearch, внешняя ngram LM, MIT, unofficial wrapper над salute-developers/GigaAM. Зависимости: torch, torchaudio, transformers, kenlm, pyctcdecode. Декод: processor.batch_decode(logits, beam_width=64, alpha=0.5, beta=0.5). Аудио: mono 16 kHz.

Рядом существуют v3, RNNT, e2e, onnx-asr, Parakeet — это другие модели, не этот контур. pyannote для longform в официальном GigaAM требует HF token и gated-модели.

Контракт
Вход: путь к wav/ogg/mp3/m4a.

Выход: text, segments[{start,end,text}], language="ru", backend (gigaam2_ctc_lm | faster_whisper), model, duration_s, extra. Если LM не даёт таймкоды, допустим один сегмент на файл. Повторный прогон raw не меняет.

Интерфейс бэкенда: transcribe(path) -> SttResult, unload(). Оркестратор ходит в factory по конфигу, без прямого знания Whisper/GigaAM.

Конфиг:

text
stt:
  backend: gigaam2_ctc_lm          # gigaam2_ctc_lm | faster_whisper
  device: cuda                     # cuda | cpu
  unload_after: true
  sample_rate: 16000
  faster_whisper:
    model: large-v3
    compute_type: int8_float16
    language: ru
    vad_filter: true
    beam_size: 5
  gigaam2_ctc_lm:
    model_id: waveletdeboshir/gigaam-ctc-with-lm
    revision: pin SHA or tag
    trust_remote_code: true
    beam_width: 64
    alpha: 0.5
    beta: 0.5
    max_chunk_s: 20
    chunk_overlap_s: 0.5
Невалидный backend — явная ошибка, не тихий fallback. Переключение — конфиг или STT_BACKEND, без правки кода. Модель на диске кэшируется; в памяти один инстанс до unload. Логиты на GPU, decode LM на CPU. Клип длиннее max_chunk_s режется по тишине (silero-vad / ffmpeg silencedetect), чанки ≤ 20 с, overlap 0.5 с, склейка без дубля на шве.

Ограничения
Пик VRAM STT < 4 ГБ. CUDA OOM на GigaAM не переключает на Whisper сам. CPU-режим есть для отладки. faster-whisper остаётся вторым бэкендом. Default после перехода — gigaam2_ctc_lm. Текст дневника в логи не пишется. HF token для этого wrapper не обязателен; если появится — только env.

Ориентир смоука: оба backend на одном wav; файл > 25 с проходит; после unload GPU-память инференса почти нулевая; короткий RU-ролик и ролик > 40 с. В отчёте достаточно: backend, длительность, время инференса, пик VRAM, первые 200 символов.
