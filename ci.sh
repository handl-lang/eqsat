#!/usr/bin/env bash
# bupha entry point (runner.kind = "script", cmd = "./ci.sh").
# The gate itself lives in scripts/gates.sh (lint → format → ty →
# vulture → import-linter → suite under the coverage floor; MLOPT_GATES_LANE
# selects the dev/full lane) so it can be run by hand.
set -euo pipefail

exec bash scripts/gates.sh "$@"
