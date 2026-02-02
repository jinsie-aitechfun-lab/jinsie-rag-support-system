SHELL := /bin/bash
PYTHON := conda run -n py310 python
UVICORN := conda run -n py310 python -m uvicorn

PORT ?= 8001
APP ?= app.main:app

SAMPLES_DIR ?= docs/samples
SAMPLES_REQ ?= $(SAMPLES_DIR)/rag_keyword_request.json
SAMPLES_RES ?= $(SAMPLES_DIR)/rag_keyword_response.json
SAMPLES_RAW ?= $(SAMPLES_DIR)/rag_keyword_response.raw.txt

.PHONY: help run health rag samples-rag ps kill

help:
	@echo "Targets:"
	@echo "  make run              - start dev server on :$(PORT)"
	@echo "  make health           - curl /health"
	@echo "  make rag Q='..'       - call /v1/rag/run"
	@echo "  make samples-rag      - generate docs/samples rag response from request json"
	@echo "  make ps               - show process listening on :$(PORT)"
	@echo "  make kill             - kill process on :$(PORT)"

run:
	$(UVICORN) $(APP) --reload --port $(PORT)

health:
	@curl -s http://127.0.0.1:$(PORT)/health && echo

rag:
	@if [ -z "$(Q)" ]; then echo "Usage: make rag Q='你的问题'"; exit 1; fi
	@curl -s -X POST http://127.0.0.1:$(PORT)/v1/rag/run \
	  -H 'Content-Type: application/json' \
	  -d "{\"query\":\"$(Q)\"}" && echo

samples-rag:
	@test -f $(SAMPLES_REQ) || (echo "missing: $(SAMPLES_REQ)"; exit 1)
	@mkdir -p $(SAMPLES_DIR)
	@echo "POST http://127.0.0.1:$(PORT)/v1/rag/run < $(SAMPLES_REQ)"
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
	@lsof -t -iTCP:$(PORT) -sTCP:LISTEN | xargs -r kill -9
