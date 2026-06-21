export UV_CACHE_DIR="/home/otahiri-/goinfre/uv-cache"
export UV_PROJECT_ENVIRONMENT="/home/otahiri-/goinfre/call_me_maybe_env"
export HF_HOME="/home/otahiri-/goinfre/hg_cache"


all: run
install:

	@HF_HOME="/home/otahiri-/goinfre/hg_cache" UV_CACHE_DIR="/home/otahiri-/goinfre/uv-cache" UV_PROJECT_ENVIRONMENT="/home/otahiri-/goinfre/call_me_maybe_env" uv sync

run:
	@HF_HOME="/home/otahiri-/goinfre/hg_cache" UV_CACHE_DIR="/home/otahiri-/goinfre/uv-cache" UV_PROJECT_ENVIRONMENT="/home/otahiri-/goinfre/call_me_maybe_env" uv run python3 -m src

debug: 
	@HF_HOME="/home/otahiri-/goinfre/hg_cache" UV_CACHE_DIR="/home/otahiri-/goinfre/uv-cache" UV_PROJECT_ENVIRONMENT="/home/otahiri-/goinfre/call_me_maybe_env" uv run python3 -m pdb src/main.py

clean: 
	rm -rf __pycache__ .mypy_cache

lint: 
	UV_CACHE_DIR="/home/otahiri-/goinfre/uv-cache" UV_PROJECT_ENVIRONMENT="/home/otahiri-/goinfre/call_me_maybe_env" uv run flake8 . --exclude .venv
	UV_CACHE_DIR="/home/otahiri-/goinfre/uv-cache" UV_PROJECT_ENVIRONMENT="/home/otahiri-/goinfre/call_me_maybe_env" uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
