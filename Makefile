.PHONY: setup dev build test test-e2e evaluate serve
setup:
	uv sync --locked --extra dev
	npm ci
	npm ci --prefix tools/avatar --ignore-scripts
dev:
	uv run python scripts/dev.py
build:
	npm run build
test: evaluate
	uv run pytest
	uv run bias validate
	npm run check
	node --test tools/avatar_audition.test.mjs
test-e2e:
	npm run test:e2e
evaluate:
	uv run bias --corpus data/personas/elon-musk evaluate --fail-on-regression --output reports/local/retrieval.json
serve:
	uv run bias serve

