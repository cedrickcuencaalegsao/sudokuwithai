# sudoku-onnx-solver
#
#   make setup                 create .venv and install dependencies
#   make run                   random puzzle, solved with $(MODEL)
#   make run CLUES=28 SEED=7   harder, reproducible puzzle
#   make animate               redraw the grids after every placed digit
#   make puzzle P="53..7...."  solve your own 81-char puzzle
#   make dummy                 smoke test with a random-weight model
#   make clean                 remove .venv and caches
#
# Extra flags go in ARGS, e.g.  make run ARGS="--layout channel-major"

MODEL ?= sudoku_model_3.onnx
CLUES ?= 32
SEED  ?=
ARGS  ?=

# First Python in this list that exists. onnxruntime has no wheels for very new releases.
PYTHON ?= $(shell command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

SEED_FLAG := $(if $(SEED),--seed $(SEED),)

.DEFAULT_GOAL := help
.PHONY: help setup run animate puzzle dummy clean

help:
	@echo "make setup     create .venv and install dependencies"
	@echo "make run       generate a puzzle and solve it with MODEL (default: $(MODEL))"
	@echo "make animate   same, redrawing the grids after every digit"
	@echo "make puzzle P=<81 chars>   solve your own puzzle"
	@echo "make dummy     smoke test with a random-weight model"
	@echo "make clean     remove .venv and caches"
	@echo ""
	@echo "Variables: MODEL=path.onnx CLUES=32 SEED=7 ARGS=\"--input-encoding scaled\""

$(PY): requirements.txt
	@test -n "$(PYTHON)" || (echo "No Python found. Install one: brew install python@3.12" && exit 1)
	@echo "Using $$($(PYTHON) --version) at $(PYTHON)"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@touch $(PY)

setup: $(PY)

run: $(PY)
	$(PY) main.py --model "$(MODEL)" --clues $(CLUES) $(SEED_FLAG) $(ARGS)

animate: $(PY)
	$(PY) main.py --model "$(MODEL)" --clues $(CLUES) $(SEED_FLAG) --animate $(ARGS)

puzzle: $(PY)
	@test -n "$(P)" || (echo 'Usage: make puzzle P="53..7....6..195...."' && exit 1)
	$(PY) main.py --model "$(MODEL)" --puzzle "$(P)" $(ARGS)

dummy: $(PY)
	$(PY) make_dummy_model.py --out dummy.onnx --in-size 810 --out-size 729
	$(PY) main.py --model dummy.onnx --clues $(CLUES) $(SEED_FLAG) $(ARGS)

clean:
	rm -rf $(VENV) __pycache__ dummy.onnx
