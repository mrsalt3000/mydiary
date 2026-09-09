# Local Diary — developer task runner (Task 01)

PYTHON    ?= /home/mrsalt/.local/bin/python3.12
VENV      ?= .venv
VENV_PY   := $(VENV)/bin/python
VENV_PIP  := $(VENV)/bin/pip
VENV_UVICORN := $(VENV)/bin/uvicorn
VENV_PYTEST  := $(VENV)/bin/pytest
VENV_RUFF    := $(VENV)/bin/ruff
VENV_MYPY    := $(VENV)/bin/mypy

.PHONY: install dev test lint typecheck format

## install — create .venv on Python 3.12 and install the package with dev extras
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install -e ".[dev]"

## dev — run the application locally with auto-reload
dev:
	$(VENV_UVICORN) diary.main:app --host 127.0.0.1 --port 8000 --reload

## test — run the test suite
test:
	$(VENV_PYTEST)

## lint — run ruff checks and verify formatting
lint:
	$(VENV_RUFF) check .
	$(VENV_RUFF) format --check .

## typecheck — run mypy over the package sources
typecheck:
	$(VENV_MYPY)

## format — apply ruff formatting in place
format:
	$(VENV_RUFF) format .
