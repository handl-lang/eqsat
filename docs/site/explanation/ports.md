# The ports — where backends meet the substrate

<!-- provenance: eqsat:packages/mlopt-core/src/mlopt_core/ports.py@e0633d5 -->

eqsat never imports a backend, and a backend never sees the search. They
meet through `mlopt_core.ports`: a set of `Protocol` shapes each side
implements or consumes. This is the seam the repo split cut along —
`graphopt`'s `mlopt_torch`/`mlopt_onnx` adapters are *implementations*
of these ports, not parts of the engine.

## The contract surfaces

- **`Source`** — lifts a domain object (a `torch.nn.Module`, an ONNX
  graph) into `mlopt_core` terms. The substrate sees only terms.
- **`Sink`** — the way back down: `lower` emits a runnable artifact,
  `verify` checks it against the source, `executors` lists what can run
  it — all on top of **`Capabilities`**'s `supported_ops`/`ops`, the
  hard feasibility declaration.
- **`Engine`** — the `engine=` contract on `search`: an alternate
  saturation engine (the PyO3 `mlopt_native.NativeEngine`). Engines are
  **never auto-detected** — an `Engine`-port value is passed explicitly
  and `stats["engine"]` records which ran (`"python"` reference default,
  `"native"` opt-in). The port covers *search* only — certificates and
  the non-local passes are Python-engine capabilities.
- **`Runner` / `Executor` / `ExecutorSpec` / `Composer`** — delivery:
  how a lowered artifact is executed, timed, and chained.
- **`Policy`** — *orders* the saturation schedule (which rules to try
  first). Every rule still fires, so the fixed point and the certificate
  are unchanged — a policy can never decide equivalence.
- **`Criterion` / `CostFn` / `Strategy` / `TaskMetric`** — what
  "cheapest" means, and the `Monolithic`/`Compositional`/`Autotuned`
  strategy shapes.
- **The rest of the port vocabulary** — `Meter`, `Verifier`,
  `VerifyResult`, `PerformanceModel`, `Profiler`, `OpRegistry`,
  `RuleSetLike`, `RuleLike`, `RuleSetProvider`, `Binding`,
  `TimingResult`, `signature_conforms`.

## The boundary honesty

The port binds *shapes*, never *contents*: `DOMAINS`, `ALL_RULES`,
`MACHINE_SHIPPED` rows, and the `lawdata` tables are data — the
registries' names are frozen, the rows are each domain repo's business.
Compat re-exports (`egraph._term_match`, `cost._memo_dispatch`, …) stay
internal even where reachable — "reachable" is not "public".

The frozen names are the [substrate slice of the
manifest](../reference/frozen-api); the domain-side implementations are
documented under [graphopt](/graphopt/).
