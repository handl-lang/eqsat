# eqsat

The eqsat substrate for the mlopt game — Python 3.13, stdlib-only
shipped code: `mlopt_core` is the zero-dependency engine and
`mlopt_orchestrator` depends on it alone.  Downstream domain repos
(`graphopt`, `kernelopt`) consume these as pinned git deps on
`packages/` subdirectories.

## Packages

- `packages/mlopt-core` — the torch-free semantic engine: term IR
  (`mlopt_core.ir`), the e-graph (`mlopt_core.egraph` — `EGraph`,
  `run`, `extract_best`, `certificate`/`verify_certificate`), the law
  bank + machinery (`mlopt_core.laws` — `DEFAULT`, `ALL_RULES`,
  `provenance`, `cond`/`tensor`/`minted`), the minted-record store
  (`mlopt_core.laws.store` — the shipped `minted.db`, tiers 0–4) and
  its auto-admission path (`mlopt_core.laws.minted.admit_prim_record`
  persists a gauntlet-cleared prim into `store.AUTO_TIER`;
  `mlopt_core.laws.provenance.AUTO_ADMITTED` classifies those names
  into `MACHINE_SHIPPED`), the cost algebra
  (`mlopt_core.cost` — `dag_cost` and the shipped models), attrs,
  typing, and the ports (`mlopt_core.ports` — `Sink`, `Source`,
  `Engine`, `Composer`, `Executor`, …).
- `packages/mlopt-orchestrator` — the backend-neutral pipeline
  (`mlopt_orchestrator` — `search`, `lower`, `Optimizer`, strategies
  `Monolithic`/`Compositional`/`Autotuned`, criteria family).  Renames
  to `pipeline` in a later phase; the name is kept for now.  The
  default `engine=` is `"eqsat"` — the engine-eqsat (Rust) search
  behind the `eqsat-run` CLI bridge (`mlopt_orchestrator.eqsat`,
  WP4 parity sweep v4 GO); `engine="python-egraph"` opts back into
  the reference `EGraph`, and a missing `eqsat-run` binary is a
  named `EqsatUnavailableError`, never a silent fallback.
- `packages/mlopt-native` — the optional PyO3/Rust search engine
  behind `engine=` (`mlopt_native.NativeEngine`).  Excluded from the
  uv workspace; build with `maturin develop --release` from
  `packages/mlopt-native`.

`handl-lang/gamekit` is the orthogonal game layer (`gamekit ⊥ eqsat`
— the game kernel imports nothing from here).

## Develop

```sh
uv sync --group dev
```

## Verify

```sh
uv run scripts/gate.py --coverage   # pytest → ruff → ty → lint-imports → coverage floor
```

`scripts/gate.py` is a stdlib runner over the dev toolchain;
`scripts/gates.sh` remains the repo's full local gate.  The coverage floor
(`[tool.coverage.report] fail_under = 74`) is pinned for
hypothesis-order slack — tighten, never lower.  The suite includes
two `slow`-marked e-graph saturations (hours-scale under the current
`ALL_RULES`); iterate with `-m "not slow"` (~35 s).

## License

MIT.

## Documentation

<https://handl-lang.github.io/docs/eqsat/>
