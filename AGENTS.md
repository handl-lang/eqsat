# Agent guide — eqsat

The eqsat substrate for the mlopt game, extracted from
`handl-lang/graphopt` (plan 0032 phase 4).  Python 3.13, stdlib-only
shipped code: `mlopt_core` is the zero-dependency engine and
`mlopt_orchestrator` depends on it alone.  Downstream domain repos
(`graphopt`, `kernelopt`) consume this repo as pinned git+ssh deps on
`packages/` subdirectories — a breaking change here breaks every
downstream `uv sync`, so the frozen surface in
`graphopt:docs/public_api.md` is the contract.

## Layout

- `packages/mlopt-core` — term IR, e-graph, certificates, the law
  bank + machinery (gauntlet, provenance, `cond`/`tensor`/`minted`),
  the cost algebra, attrs/typing, ports.  The whole package travels:
  bank-vs-machinery is a later sub-package split.
- `packages/mlopt-orchestrator` — the backend-neutral pipeline
  (`search`/`lower`/`Optimizer`/strategies/criteria).  Renames to
  `pipeline` later; name kept for now.
- `packages/mlopt-native` — the optional Rust engine behind
  `engine=` (maturin/PyO3; excluded from the uv workspace).
- `tests/` — the substrate suite (the 39 files whose imports stay
  inside the substrate + hypothesis; domain-bound test functions were
  carved back into `graphopt:tests/test_substrate_domain_edges.py`).

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q          # the suite
.venv/bin/ruff check packages                  # lint (shipped source)
.venv/bin/ruff format --check packages         # formatting
.venv/bin/ty check                             # typecheck — 0 errors
.venv/bin/lint-imports                         # hexagonal contracts
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

- The import-linter contracts are the intra-repo slice of the parent
  monorepo's: core is torch-free (and substrate-top), the orchestrator
  imports no backend, layers `orchestrator > (native) > core`.
- `mlopt_native` is opt-in: `test_native_engine.py` self-skips when it
  isn't built (`maturin develop --release` in `packages/mlopt-native`).
- Coverage floor re-measured post-extraction — see
  `[tool.coverage.report] fail_under` in `pyproject.toml`; tighten,
  never lower.
