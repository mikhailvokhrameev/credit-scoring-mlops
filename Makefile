.PHONY: check lint test smoke notebooks

check: lint test

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

smoke:
	uv run pytest -q -m smoke

notebooks:
	uv run --group notebooks jupyter nbconvert --to notebook --execute notebooks/04_interpretation.ipynb --output-dir $$(mktemp -d)
