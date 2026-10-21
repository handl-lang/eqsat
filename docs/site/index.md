# eqsat — the equality-saturation substrate

<!-- provenance: docs: + eqsat:README.md@e0633d5 -->

**eqsat** is the substrate the mlopt game is played on: the torch-free
semantic engine extracted from `graphopt` at the plan-0032 phase-4 repo
split (`research-mlopt` → `graphopt` → three public repos). It owns the
*representation and the machinery* — term IR, e-graph, the law bank and
its admission machinery, the cost algebra, and the ports — plus the
backend-neutral pipeline that runs search over them. Domain repos bring
the backends and the boards; eqsat brings everything that must not know
them.

## What's inside

| package | role |
|---|---|
| `mlopt-core` | term IR (`mlopt_core.ir`), the e-graph (`mlopt_core.egraph` — `EGraph`, `run`, `extract_best`, `certificate`/`verify_certificate`), the law bank + machinery (`mlopt_core.laws` — `DEFAULT`, `ALL_RULES`, `provenance`, `cond`/`tensor`/`minted`), the cost algebra (`mlopt_core.cost` — `dag_cost` + the shipped models), attrs, typing, and the ports (`mlopt_core.ports` — `Sink`, `Source`, `Engine`, `Composer`, `Executor`, …). Stdlib-only. |
| `mlopt-orchestrator` | the backend-neutral pipeline (`search`, `lower`, `Optimizer`, strategies `Monolithic`/`Compositional`/`Autotuned`, the criteria family). Depends on `mlopt-core` alone. |
| `mlopt-native` | the optional PyO3/Rust search engine behind `engine=` (`mlopt_native.NativeEngine`). Excluded from the uv workspace — build with `maturin develop --release`. |

## How it's consumed

`graphopt` pins the packages as git dependencies on `packages/`
subdirectories — `uv sync` in a `graphopt` clone resolves them
automatically; there is nothing separate to install. Working on the
substrate itself:

```sh
git clone https://github.com/handl-lang/eqsat
cd eqsat
uv sync --group dev
.venv/bin/python -m pytest tests/ -q     # the substrate suite
```

## Read next

- [The e-graph and `dag_cost`](./explanation/egraph) — saturation as a
  closure, extraction, honest pricing.
- [Certificates](./explanation/certificates) — the replayable derivation
  that is the claim.
- [The ports](./explanation/ports) — `Source`/`Sink`/`Engine`: how
  backends plug into a substrate that never imports them.
- [The frozen API slice](./reference/frozen-api) — the substrate rows of
  the phase-1 contract.
- The whole-system account stays with graphopt:
  [how mlopt works](/graphopt/explanation/mechanism).
- Source: [`handl-lang/eqsat`](https://github.com/handl-lang/eqsat).
