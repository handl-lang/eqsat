#!/usr/bin/env bash
# scripts/gates.sh — the bupha gate for handl-lang/eqsat.
#
# The same ladder as .github/workflows/ci.yml, run natively inside the
# pushed sha's worktree (no containers): lint, format, types, dead code,
# the import-layer contract, then the full suite under the coverage
# floor pinned in pyproject.toml. Cheap-first so a red gate fails fast.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
    uv sync --group dev
fi

.venv/bin/ruff check packages
.venv/bin/ruff format --check packages
.venv/bin/ty check
.venv/bin/vulture packages
.venv/bin/lint-imports
.venv/bin/coverage run -m pytest -q
.venv/bin/coverage report -m
