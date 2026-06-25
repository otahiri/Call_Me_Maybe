.PHONY: all install debug clean lint test
ARGS=
all: run
install:
	uv sync

run:
	uv run python3 -m src $(ARGS)

debug: 
	uv run python3 -m pdb src/main.py

clean: 
	rm -rf __pycache__ .mypy_cache tests/__pycache__  tests/.mypy_cache src/__pycache__  src/.mypy_cache

lint: 
	uv run flake8 src tests
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	uv run python3 -m tests
