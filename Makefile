install:
	uv sync

run:
	uv run python3 main.py

debug: 
	uv run python3 -m pdb main.py

clean: 
	rm -rf __pycache__ .mypy_cache

lint: 
	uv run flake8 . --exclude .venv
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
