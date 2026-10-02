.PHONY: setup check format test smoke build run

setup:
	uv sync --frozen

check:
	uv run --frozen python scripts/check.py
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format:
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

test:
	uv run --frozen pytest

smoke:
	uv run --frozen pytest -m smoke

build:
	uv build

run:
	uv run --frozen hm-build-notebook
