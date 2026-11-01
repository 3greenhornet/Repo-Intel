.PHONY: help install demo build ask eval test train api mcp clean compile-deps upgrade-deps check-python

PYTHON ?= python3

help:
	@echo "RepoIntel AI — make targets"
	@echo "  make demo    — create fixture repo + build knowledge base"
	@echo "  make api     — start FastAPI server + dashboard on :8000"
	@echo "  make ask Q='why is login slow?'  — ask a question (CLI)"
	@echo "  make eval    — run Phase 3 evaluation (4 baselines)"
	@echo "  make test    — run unit + integration tests"
	@echo "  make train   — train quality-prediction ML models"
	@echo "  make compile-deps  — regenerate hash-locked requirements.txt"
	@echo "  make clean   — remove data/ artifacts"

# FIX-020: fail fast when the interpreter is older than the declared minimum
check-python:
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required'"

demo: check-python
	$(PYTHON) scripts/create_demo_repo.py data/repos/demo-web-shop
	$(PYTHON) scripts/build_knowledge.py data/repos/demo-web-shop --no-github

build: check-python
	$(PYTHON) scripts/build_knowledge.py $(REPO) $(if $(BRANCH),--branch $(BRANCH),)

ask: check-python
	$(PYTHON) scripts/ask.py $(REPO) "$(Q)"

eval: check-python
	$(PYTHON) scripts/run_eval.py

train: check-python
	$(PYTHON) scripts/train_quality_models.py

# FIX-021: lock file generation — edit requirements.in, then run this
compile-deps:
	pip-compile --generate-hashes --output-file backend/requirements.txt backend/requirements.in

upgrade-deps:
	pip-compile --upgrade --generate-hashes --output-file backend/requirements.txt backend/requirements.in

test: check-python
	cd backend && $(PYTHON) -m pytest tests/unit tests/integration -v

api: check-python
	cd backend && $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

mcp: check-python
	cd backend && $(PYTHON) -m app.core.mcp.server

clean:
	rm -rf data/faiss data/*.pkl data/*.db data/*.db-* data/evaluation_results.json data/eval_abc.json
