# eqsat

The eqsat substrate for the mlopt game — extracted from
`handl-lang/graphopt` (plan 0032 phase 4).

- `packages/mlopt-core` — the torch-free semantic engine: term IR
  (`mlopt_core.ir`), the e-graph (`mlopt_core.egraph` — `EGraph`,
  `run`, `extract_best`, `certificate`/`verify_certificate`), the law
  bank + machinery (`mlopt_core.laws` — `DEFAULT`, `ALL_RULES`,
  `provenance`, `cond`/`tensor`/`minted`), the cost algebra
  (`mlopt_core.cost` — `dag_cost` and the shipped models), attrs,
  typing, and the ports (`mlopt_core.ports` — `Sink`, `Source`,
  `Engine`, `Composer`, `Executor`, …).
- `packages/mlopt-orchestrator` — the backend-neutral pipeline
  (`mlopt_orchestrator` — `search`, `lower`, `Optimizer`, strategies
  `Monolithic`/`Compositional`/`Autotuned`, criteria family).  Renames
  to `pipeline` in a later phase; the name is kept for now.
- `packages/mlopt-native` — the optional PyO3/Rust search engine
  behind `engine=` (`mlopt_native.NativeEngine`).  Excluded from the
  uv workspace; build with `maturin develop --release` from
  `packages/mlopt-native`.

Downstream: `handl-lang/graphopt` (tensor domain) and
`handl-lang/kernelopt` (codegen domain) consume these as pinned
git+ssh deps on `packages/` subdirectories.  `handl-lang/gamekit` is
the orthogonal game layer (`gamekit ⊥ eqsat` — the game kernel
imports nothing from here).

## Develop

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check packages
.venv/bin/ruff format --check packages
.venv/bin/ty check
.venv/bin/lint-imports
```

The coverage floor (`[tool.coverage.report] fail_under`) is
re-measured for this repo's suite — 56% at the extraction scaffold,
75% after the substrate-pure tests were ported back from graphopt's
domain test files (`tests/test_ported_*`, plus the duck-typed
`tests/test_value_passes.py` / `tests/test_profile.py`) — tighten,
never lower.  The suite includes two `slow`-marked e-graph
saturations (~17 min); iterate with `-m "not slow"` (~35 s).
