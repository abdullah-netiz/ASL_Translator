# Shortcuts: extract, train, eval, serve, test, qc
VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PYTEST ?= $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
UVICORN ?= $(if $(wildcard $(VENV)/bin/uvicorn),$(VENV)/bin/uvicorn,uvicorn)

.PHONY: help manifest extract qc train eval serve test

help:
	@echo "Usage:"
	@echo "  make manifest - Reconcile CSV and videos to build manifests (Gate 0)"
	@echo "  make extract  - Run landmark extraction pipeline (Phase 1)"
	@echo "  make qc       - Generate Phase 1 QC markdown report"
	@echo "  make test     - Run unit test suite"
	@echo "  make train    - Run model training"
	@echo "  make eval     - Run evaluation on validation/test sets"
	@echo "  make serve    - Start FastAPI + WebSocket server"

manifest:
	$(PYTHON) scripts/build_manifests.py --config configs/data.yaml --split all

extract:
	$(PYTHON) -m src.extract.run_extraction --config configs/data.yaml --config-lm configs/landmarks.yaml --split all

qc:
	$(PYTHON) -m src.extract.qc --config configs/data.yaml --out reports/phase1_extraction_qc.md

train:
	$(PYTHON) -m src.train.train

eval:
	$(PYTHON) -m src.eval.evaluate

serve:
	$(UVICORN) src.serve.server:app --reload

test:
	$(PYTEST) tests/
