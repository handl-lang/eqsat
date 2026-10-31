#!/usr/bin/env bash
# scripts/gates.sh — the bupha gate for handl-lang/eqsat.
#
# The same ladder as .github/workflows/ci.yml, run natively inside the
# pushed sha's worktree (no containers): lint, format, types, dead code,
# the import-layer contract, then the suite under the coverage floor
# pinned in pyproject.toml. Cheap-first so a red gate fails fast.
#
# Two lanes (same model as graphopt's gates.sh):
#   MLOPT_GATES_LANE=dev  (default) — `-m "not slow"`: the two
#       `test_ported_square_rsqrt_laws` saturations are hours-scale
#       under the current ALL_RULES bank (one `div_one_sqrt` spelling
#       observed >10 min mid-enumeration; a bupha run pegged a core
#       for >4 h without finishing). The push gate cannot wait on
#       them, so the default lane deselects them; the 74% floor still
#       holds (75% measured on the fast lane, 2026-10-31).
#   MLOPT_GATES_LANE=full — everything, once, under coverage. Run by
#       hand before release; do not point bupha at it on this host.
set -euo pipefail
cd "$(dirname "$0")/.."

LANE="${MLOPT_GATES_LANE:-dev}"   # dev | full
PYTEST_M=()
if [[ "$LANE" == "dev" ]]; then
    PYTEST_M=(-m "not slow")
elif [[ "$LANE" != "full" ]]; then
    echo "unknown MLOPT_GATES_LANE=$LANE (dev|full)" >&2
    exit 2
fi

if [ ! -x .venv/bin/python ]; then
    uv sync --group dev
fi

.venv/bin/ruff check packages
.venv/bin/ruff format --check packages
.venv/bin/ty check
.venv/bin/vulture packages
.venv/bin/lint-imports
.venv/bin/coverage run -m pytest -q "${PYTEST_M[@]}"
.venv/bin/coverage report -m
