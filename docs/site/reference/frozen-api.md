::: warning Vendored slice — do not edit here
Synced from `graphopt:docs/public_api.md` @ `fb00089` — the phase-1
frozen manifest (`API_VERSION = "0.1.0"`). This page is the **eqsat
slice**: the substrate rows of the contract, landed at the plan-0032
phase-4 split. The manifest upstream carries all three slices; the
game rows are [gamekit's](/gamekit/reference/frozen-api) and the
domain rows are [graphopt's](/graphopt/reference/frozen-api).
:::

# The frozen public API — the eqsat (substrate) slice

> **Version:** `from mlopt_core import API_VERSION`. The `0.x` series
> covered the monorepo contract; with the split landed, each repo
> versions its own slice — the rows below are the ones
> `handl-lang/eqsat` must not break.

* **Frozen** — a name in the table below. Renaming, retyping, or
  re-homing it without an `API_VERSION` major bump is a contract break.
* **Public, unfrozen** — any other name in a module's `__all__`.
* **Internal** — everything else, including underscore compat
  re-exports (`egraph._term_match`, `cost._memo_dispatch`, …). Reachable
  is not public.

## Substrate contracts

| Contract | Canonical location | Frozen surface |
|---|---|---|
| Term IR | `mlopt_core.ir` | `IR`, `Op` (incl. `Op.make`), `Var`, `Param`, `Const`, `TensorType`, `op_repr`, `op_repr_dag`, `term_to_data`, `term_from_data` |
| E-graph engine | `mlopt_core.egraph` | `EGraph`, `ENode`, `EClass`, `UnionFind`, `Rewrite` |
| `EGraph` protocol | `mlopt_core.egraph.EGraph` | `add_term(term) -> eid`, `find(eid)`, `rebuild(classes=None)`, `extract_best(eid, cost_fn, **kw)`, `certificate(src, dst, root_eid=...)`, **`run(rules, root_eid, *, max_iterations=100, max_nodes=100_000, rule_budgets=None, stop="fixed_point", patience=3, cost_fn=None, policy=None) -> stats`** — `stats` carries `stop`, `n_enodes`, `n_classes`, `rule_fires`, `rule_budgets` |
| Certificates | `mlopt_core.egraph` | `Certificate`, `CertStep`, `ProofEdge`, `CertificateVerificationError`, `cert_to_data`, `cert_from_data`, **`verify_certificate(term, cert, *, strict=False)`** — the standalone derivation replay |
| DAG pricing | `mlopt_core.cost` | **`dag_cost(term, cost_fn, memo=None)`** — the shared-subtree-honest price; plus the shipped models `count_cost`, `flops_cost`, `param_bytes_cost`, `param_bytes_cost_for`, `backend_cost`, `executor_cost_for`, `fused_cost_for`, `lowering_aware_cost_for`, `depth_cost`, `depth_cost_for`, `roofline_cost`, `roofline_cost_for`, `launch_aware_cost`, `fusion_regions`, `fusion_member_key`, `CostModel`, `LOWERINGS` |
| Ports | `mlopt_core.ports` | **`Sink`** (`lower`/`verify`/`executors` on top of **`Capabilities`**'s `supported_ops`/`ops`), **`Source`**, **`Engine`** (the `engine=` contract below), `Composer`, `Executor`, `ExecutorSpec`, `Runner`, `Meter`, `Verifier`, `VerifyResult`, `CostFn`, `Criterion`, `Policy`, `Strategy`, `TaskMetric`, `PerformanceModel`, `Profiler`, `OpRegistry`, `RuleSetLike`, `RuleLike`, `RuleSetProvider`, `Binding`, `TimingResult`, `signature_conforms` |
| Law machinery | `mlopt_core.laws` | `RuleSet`, `preset`, `DEFAULT`, `FULL`, `SYMMETRY`, `ALL_RULES`, `tags`, `provenance` — the *names* are frozen; the **bank contents** (`ALL_RULES` membership, `MACHINE_SHIPPED` rows) are each domain repo's data, not the substrate's contract |
| Provenance registry | `mlopt_core.laws.provenance` | `MACHINE_SHIPPED`, `MACHINE_STORED`, `MACHINE_LAWS`, `law_provenance(rule) -> "machine" \| "human"` |
| The pipeline | `mlopt_orchestrator` | `Optimizer`, **`search(model, x, *, source, capabilities=None, criteria=None, cost_fn=None, rules=None, engine=None, policy=None, …) -> SearchResult`**, `lower`, `SearchResult`, `LowerResult`, `Backend`, `discover_alternatives`, `DEFAULT_RULES`, `OptimizationResourceError`, strategies `Monolithic`/`Compositional`/`Autotuned`, runners `IdentityRunner`/`ChainedRunner`/`runner_candidate`, criteria family (`Criterion`, `Criteria`, `Blend`, `criteria_cost`, `peak_bytes_cost`, `LatencyCriterion`, `MemoryCriterion`, `FlopsCriterion`, `DepthCriterion`, `CompiledCriterion`, `PredictedCriterion`), `delivered_cost_for`, `structural_key` |
| `engine=` | `search`/`Optimizer` kwarg | An `Engine`-port value — instance, class, or zero-arg factory. Engines are **never auto-detected**; `stats["engine"]` records which ran (`"python"` reference default, `"native"` for `mlopt_native.NativeEngine`). The port covers the *search* only — certificates and the non-local passes are Python-engine capabilities |

## Boundary notes

- **`mlopt_native`** is an opt-in engine behind `engine=`, not a frozen
  import path: it is excluded from the uv workspace, so the contract
  binds the `Engine` *port*, not the package.
- **Registry contents vs registry shape**: `ALL_RULES` /
  `MACHINE_SHIPPED` contents are *data* — the names and access
  functions are frozen, the rows are each consuming repo's business.

The full manifest — including the domain-side rows — is
[`docs/public_api.md`](https://github.com/handl-lang/graphopt/blob/main/docs/public_api.md)
upstream.
