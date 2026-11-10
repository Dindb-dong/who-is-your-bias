.PHONY: setup dev build test evaluate serve
setup:
	uv sync --extra dev
	npm ci
dev:
	uv run python scripts/dev.py
build:
	npm run build
test:
	uv run pytest
	npm run check
evaluate:
	uv run bias evaluate --fail-on-regression --output reports/local/retrieval.json
serve:
	uv run bias serve

