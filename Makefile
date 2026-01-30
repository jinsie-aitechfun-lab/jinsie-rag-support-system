SHELL := /bin/bash
PYTHON := conda run -n py310 python
UVICORN := conda run -n py310 python -m uvicorn

PORT ?= 8001
APP ?= app.main:app

.PHONY: help run health rag ps kill

help:
	@echo "Targets:"
	@echo "  make run        - start dev server on :$(PORT)"
	@echo "  make health     - curl /health"
	@echo "  make rag Q='..' - call /v1/rag/run"
	@echo "  make ps         - show process listening on :$(PORT)"
	@echo "  make kill       - kill process on :$(PORT)"

run:
	$(UVICORN) $(APP) --reload --port $(PORT)

health:
	@curl -s http://127.0.0.1:$(PORT)/health && echo

rag:
	@if [ -z "$(Q)" ]; then echo "Usage: make rag Q='你的问题'"; exit 1; fi
	@curl -s -X POST http://127.0.0.1:$(PORT)/v1/rag/run \
	  -H 'Content-Type: application/json' \
	  -d "{\"query\":\"$(Q)\"}" && echo

ps:
	@lsof -nP -iTCP:$(PORT) -sTCP:LISTEN || true

kill:
	@lsof -t -iTCP:$(PORT) -sTCP:LISTEN | xargs -r kill -9
