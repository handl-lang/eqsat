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
- `tests/` — the substrate suite (121 test files whose imports stay
  inside the substrate + hypothesis; domain-bound test functions were
  carved back into `graphopt:tests/test_substrate_domain_edges.py`).
  `tests/test_ported_<stem>.py` are the substrate-pure functions
  ported *back* from `graphopt:tests/test_<stem>.py` — each asserts
  only on `mlopt_core`/`mlopt_orchestrator` + stdlib/pytest/hypothesis
  and travelled to the domain repo only because sibling tests in the
  same file needed domain packages.  `tests/test_value_passes.py` +
  `tests/faketensor.py` drive the value-analysis passes (param
  sharing, weight specials, low-rank factorisation, head sharing)
  through a duck-typed tensor; `tests/test_profile.py` pins the
  torch-free half of target-profile calibration.
  `test_ported_square_rsqrt_laws.py` carries the two `slow`-marked
  e-graph saturations — hours-scale under the current `ALL_RULES`
  bank (a single `div_one_sqrt` spelling was observed >10 min inside
  one `apply_rule` enumeration; a CI run pegged a core >4 h without
  finishing — see `project/` I-002): iterate with `-m "not slow"`.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q          # the suite (hours-scale —
                                              #  see the `slow` note above)
.venv/bin/python -m pytest tests/ -q -m "not slow"   # the fast lane (~35 s)
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

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
