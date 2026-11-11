.PHONY: setup help lint format test check search-up search-down search-check api worker ingestion-check workflow-check snapshot-check ragas-check enrichment-check reranking-check comparison-check admin-check

setup:
	uv sync --locked

help:
	uv run --locked s3-rag --help

lint:
	uv run --locked ruff check rag tests
	uv run --locked ruff format --check rag tests

format:
	uv run --locked ruff format rag tests

test:
	uv run --locked pytest

check: lint test

search-up:
	docker compose up -d --wait --wait-timeout 180 opensearch

search-down:
	docker compose down

search-check:
	uv run --locked s3-rag --config config.toml search check

api:
	uv run --locked uvicorn rag.api.app:app --host 127.0.0.1 --port 8000 --reload --reload-dir rag --no-access-log

worker:
	uv run --locked s3-rag --config config.toml worker

ingestion-check:
	uv run --locked python tests/check_local_ingestion.py

workflow-check:
	uv run --locked python tests/check_local_workflow.py

snapshot-check:
	uv run --locked python tests/check_local_snapshots.py

ragas-check:
	RAGAS_DO_NOT_TRACK=true uv run --locked --group eval python tests/check_ragas.py

enrichment-check:
	uv run --locked python tests/check_local_enrichment.py

reranking-check:
	uv run --locked python tests/check_local_reranking.py

comparison-check:
	uv run --locked python tests/check_local_comparison.py

admin-check:
	uv run --locked python tests/check_local_admin.py
