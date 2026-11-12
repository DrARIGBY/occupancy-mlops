# occupancy-mlops - local commands. Paths and ports are variables, e.g. `make serve PORT=8080`.
VENV      ?= .venv
PYTHON_VERSION ?= 3.12
DATA_DIR  ?= data/raw
MODEL_DIR ?= models
HOST      ?= 127.0.0.1
PORT      ?= 8000
N_ITER    ?= 10
REPLAY_LIMIT ?= 100

# Windows venvs keep executables in Scripts/, POSIX ones in bin/.
ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV)/Scripts
else
  VENV_BIN := $(VENV)/bin
endif
PY := $(VENV_BIN)/python

.PHONY: setup train test serve replay ci-local

setup:
	uv venv $(VENV) --python $(PYTHON_VERSION) --allow-existing
	uv pip install --python $(VENV) -e '.[dev]'

# Task 1: training pipeline -> models/model.joblib + models/metadata.json
train:
	$(PY) -m occupancy.train --data-dir $(DATA_DIR) --out $(MODEL_DIR) --n-iter $(N_ITER)

test:
	$(PY) -m pytest

# Task 2: event-driven endpoint, and (in a second terminal) the event source
serve:
	@echo "Open http://$(HOST):$(PORT)/docs to try the endpoints (POST /events, GET /health)."
	@echo "Send the test week from a second terminal: make replay"
	OCC_MODEL_DIR=$(MODEL_DIR) $(PY) -m uvicorn occupancy.service:app --host $(HOST) --port $(PORT)

replay:
	OCC_DATA_DIR=$(DATA_DIR) $(PY) -m occupancy.replay --url http://$(HOST):$(PORT) --limit $(REPLAY_LIMIT)

# Task 3: test -> build -> smoke-test -> push (only if PUSH_REGISTRY is set)
ci-local:
	PYTHON=$(PY) N_ITER=$(N_ITER) bash Scripts/ci_local.sh
