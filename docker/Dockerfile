# syntax=docker/dockerfile:1
# Local Diary application image (Task 17, docker-compose.yml).
#
# One image serves both Compose services:
#   * web    — uvicorn (diary.main:app)
#   * worker — python -m diary.workers.runner
# plus, with opt-in build flags, STT-capable worker images:
#   * INSTALL_STT=1  — also installs the optional `stt` extra
#                      (faster-whisper). CPU transcription works with this
#                      flag alone (device: cpu in configs/models.yaml).
#   * ENABLE_CUDA=1  — additionally installs NVIDIA cuBLAS/cuDNN pip packages
#                      and points LD_LIBRARY_PATH at them. Requires
#                      INSTALL_STT=1 and, at run time, a working NVIDIA driver
#                      with the nvidia container runtime (WSL 2). Used by the
#                      `gpu` compose profile (device: cuda in models.yaml).
#
# Verified decisions for the current dependency set (Task 17):
#   * No build toolchain (gcc/build-essential): every runtime dependency ships
#     a manylinux wheel for CPython 3.12 (pydantic-core, SQLAlchemy,
#     uvicorn[standard], PyAV, ctranslate2). Add toolchain packages only if a
#     future dependency lacks prebuilt wheels.
#   * No system FFmpeg: the application never transcodes audio at runtime — it
#     stores and serves uploads as-is (duration stays NULL by design; see
#     src/diary/domain/audio/service.py) — and faster-whisper decodes audio
#     through PyAV, which bundles its own FFmpeg libraries.
#   * tzdata is installed so IANA zones other than UTC (e.g. Europe/Moscow)
#     resolve inside the container (Python zoneinfo).
#   * The image is non-root (USER app) and carries no user data, secrets,
#     .env or a real configs/models.yaml: those arrive via bind mounts and
#     environment at run time (AGENTS.md §7-8).

FROM python:3.12-slim

# HF_HOME: faster-whisper / huggingface_hub model cache. Must be writable by
# the runtime user; compose mounts a named volume at /app/.hf-cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache

# Directories of the NVIDIA pip wheels installed only in ENABLE_CUDA=1 images.
# Nonexistent entries in LD_LIBRARY_PATH are ignored by the dynamic loader, so
# this variable is harmless on CPU-only images (which never install the wheels).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

WORKDIR /app

# System packages. curl is used by the web healthcheck; tzdata provides IANA
# timezones; libstdc++6/libgomp1 cover native runtime libs of optional STT
# wheels (ctranslate2) and are no-ops if already present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tzdata \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Python dependency layer — rebuilds only when pyproject.toml/README.md or
# src/ change. Editable install keeps `import diary` resolving to /app/src, so
# PROJECT_ROOT (diary.core.settings) stays /app and all default data paths
# point at /app/data (bind-mounted in compose).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Optional speech-to-text runtime (faster-whisper). Off by default: the base
# CPU image has no GPU- or STT-specific dependencies.
ARG INSTALL_STT=0
RUN if [ "${INSTALL_STT}" = "1" ]; then \
        pip install --no-cache-dir -e ".[stt]"; \
    fi

# Optional NVIDIA cuBLAS/cuDNN for GPU transcription (CUDA 12 / cuDNN 9,
# matching current ctranslate2 wheels — see faster-whisper README, "GPU").
# The NVIDIA driver itself is mounted by the container runtime, not installed
# here.
ARG ENABLE_CUDA=0
RUN if [ "${ENABLE_CUDA}" = "1" ]; then \
        pip install --no-cache-dir "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"; \
    fi

# Migration sources and LLM prompt templates (loaded from /app at run time).
COPY alembic.ini ./
COPY alembic ./alembic
COPY prompts ./prompts

# Non-root runtime user. Default UID/GID 1000 match the standard first WSL 2
# user so the ./data bind mount (owned by the host user) is writable without
# extra setup. Build with --build-arg APP_UID/APP_GID and set the same values
# in compose (APP_UID/APP_GID in .env) when the host UID differs.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/app --shell /bin/bash app \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /app/.hf-cache

USER app

EXPOSE 8000

# Default command: the web application. Compose overrides this for the worker
# (python -m diary.workers.runner ...) and the one-shot migration step
# (alembic upgrade head). Inside the container uvicorn binds 0.0.0.0; compose
# publishes only 127.0.0.1:8000 on the host, so the app stays localhost-only.
CMD ["python", "-m", "uvicorn", "diary.main:app", "--host", "0.0.0.0", "--port", "8000"]
