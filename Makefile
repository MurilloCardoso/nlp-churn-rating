.PHONY: install lint test train run

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest

train:
	uv run python -m src.main

run:
	uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
