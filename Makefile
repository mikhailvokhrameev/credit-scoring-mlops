.PHONY: check lint test smoke

check: lint test

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

smoke:
	uv run pytest -q -m smoke
