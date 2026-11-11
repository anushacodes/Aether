.PHONY: setup help lint format test check

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
