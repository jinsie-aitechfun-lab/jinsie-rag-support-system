SHELL := /bin/bash

# If you already activated (py310), prefer direct python for stable --reload logs.
# Keep conda-run versions as fallbacks.
PYTHON := python
UVICORN := python -m uvicorn

PYTHON_CONDA := conda run -n py310 python
UVICORN_CONDA := conda run -n py310 python -m uvicorn

PORT ?= 8001
APP ?= app.main:app

SAMPLES_DIR ?= docs/samples
SAMPLES_REQ ?= $(SAMPLES_DIR)/rag_keyword_request.json
SAMPLES_RES ?= $(SAMPLES_DIR)/rag_keyword_response.json
SAMPLES_RAW ?= $(SAMPLES_DIR)/rag_keyword_response.raw.txt

# RAG retrieval mode: keyword | vector
MODE ?= keyword

.PHONY: help run run-conda health rag samples-rag ps kill

help:
	@echo "Targets:"
	@echo "  make run                       - start dev server on :$(PORT) (prefer activated env)"
	@echo "  make run-conda                 - start dev server via conda run on :$(PORT)"
	@echo "  make health                    - curl /health"
	@echo "  make rag Q='..'                - call /v1/rag/run (default MODE=keyword)"
	@echo "  make rag Q='..' MODE=vector    - call /v1/rag/run with vector retrieval"
	@echo "  make samples-rag MODE=vector   - generate docs/samples rag response from request json (override mode)"
	@echo "  make ps                        - show process listening on :$(PORT)"
	@echo "  make kill                      - stop process on :$(PORT) (TERM -> KILL)"

run:
	$(UVICORN) $(APP) --reload --port $(PORT)

run-conda:
	$(UVICORN_CONDA) $(APP) --reload --port $(PORT)

health:
	@curl -s http://127.0.0.1:$(PORT)/health && echo

rag:
	@if [ -z "$(Q)" ]; then echo "Usage: make rag Q='你的问题' [MODE=keyword|vector]"; exit 1; fi
	@curl -s -X POST http://127.0.0.1:$(PORT)/v1/rag/run \
	  -H 'Content-Type: application/json' \
	  -d "{\"query\":\"$(Q)\",\"retrieval_mode\":\"$(MODE)\"}" && echo

samples-rag:
	@test -f $(SAMPLES_REQ) || (echo "missing: $(SAMPLES_REQ)"; exit 1)
	@mkdir -p $(SAMPLES_DIR)
	@echo "POST http://127.0.0.1:$(PORT)/v1/rag/run < $(SAMPLES_REQ) (MODE=$(MODE))"
	@curl -sS -D - -o $(SAMPLES_RAW) -X POST http://127.0.0.1:$(PORT)/v1/rag/run \
	  -H 'Content-Type: application/json' \
	  -d @$(SAMPLES_REQ) | sed -n '1,20p'
	@echo
	@echo "--- raw body (head 40 lines) ---"
	@sed -n '1,40p' $(SAMPLES_RAW) || true
	@echo
	@$(PYTHON) -c "import json; p='$(SAMPLES_RAW)'; s=open(p,'r',encoding='utf-8',errors='replace').read().strip(); json.loads(s); print('[OK] json detected')" \
	  && $(PYTHON) -m json.tool "$(SAMPLES_RAW)" > "$(SAMPLES_RES)" \
	  && echo "[OK] wrote $(SAMPLES_RES)" \
	  || (echo "[ERR] response is not valid JSON, kept raw at $(SAMPLES_RAW)"; exit 1)

ps:
	@lsof -nP -iTCP:$(PORT) -sTCP:LISTEN || true

kill:
	@PIDS="$$(lsof -t -iTCP:$(PORT) -sTCP:LISTEN)"; \
	if [ -z "$$PIDS" ]; then echo "[OK] no process on :$(PORT)"; exit 0; fi; \
	echo "[INFO] stopping: $$PIDS"; \
	kill $$PIDS 2>/dev/null || true; \
	sleep 0.3; \
	PIDS2="$$(lsof -t -iTCP:$(PORT) -sTCP:LISTEN)"; \
	if [ -n "$$PIDS2" ]; then echo "[WARN] still alive, force killing: $$PIDS2"; kill -9 $$PIDS2 2>/dev/null || true; fi
