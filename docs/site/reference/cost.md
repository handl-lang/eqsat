# The cost algebra — signature reference

<!-- provenance: eqsat:packages/mlopt-core/src/mlopt_core/cost@2d6e4eb + graphopt:docs/public_api.md@ed89ead -->

Signature-level reference for `mlopt_core.cost`. The conceptual
account — why the substrate ships *three* trust levels of cost — is
[the three costs](/graphopt/explanation/three-costs); the frozen-name
list is [the manifest slice](./frozen-api).

## `dag_cost` — the shared-subtree-honest price

```python
dag_cost(term, cost_fn, memo=None) -> float
```

Walks the program *graph* — a shared subtree is paid once, so a DAG is
never billed as a tree. `cost_fn` is any `(term, memo=None) -> float`
callable (the `CostFn` port convention); `memo` threads the
shared-subterm cache. This is the **ranking** price: static weights, no
tensors executed, no claim.

## The shipped models

| family | names | shape |
|---|---|---|
| counting | `count_cost(term, memo=None)`, `flops_cost(term, memo=None)` | bare cost fns |
| parameters | `param_bytes_cost(term, memo=None)`, `param_bytes_cost_for(source_tensors=None, by_bytes=False)` | storage pricing; the `_for` variant binds the source tensors |
| backend-aware | `backend_cost(cost_fn, supported_ops) -> CostFn` | wraps a base fn — members using ops outside `supported_ops` price `+inf`, so search never commits to a form the sink cannot lower |
| executor | `executor_cost_for(profile=None, *, lowering="generic", base="roofline") -> CostFn` | per-executor pricing, calibrated to a target profile |
| fused | `fused_cost_for(profile=None) -> CostFn` | fusion-aware pricing |
| lowering-aware | `lowering_aware_cost_for(profile=None, *, lowerings=LOWERINGS, base="roofline") -> CostFn` | prices across the `LOWERINGS` family (`("generic", "batched_scan", "compiled")`) |
| depth / roofline | `depth_cost`, `depth_cost_for(profile=None)`; `roofline_cost`, `roofline_cost_for(profile=None, *, peak_flops=None, peak_bw=None, launch_s=None)` | critical-path and roofline pricing |
| launch | `launch_aware_cost(term, memo=None)` | launch-overhead pricing |
| fusion regions | `fusion_regions(term, memo=None) -> tuple[frozenset, ...]`, `fusion_member_key(term, memo=None) -> tuple[int, int]` | the fusible-partition analysis the fused models price over |
| configurable | `CostModel(op_weights=None, weight_coeff=1.0, matmul_coeff=2.0)` | a callable weighted-FLOPs model for `EGraph.extract_best` |

The `_for` family are *factories* — they return a `CostFn` bound to a
target profile or lowering; the bare names are ready-made cost fns.
A `profile` is a `mlopt_core.profile.TargetProfile`-like object or
dict; `None` is the built-in profile.

## What a cost can and cannot say

`dag_cost` ranking says "probably cheaper" — it has no notion of
memory traffic, kernel fusion, or launch overhead (a fused `sdpa`
looks expensive on paper and is fast under Inductor). Measured pay and
the certificate are the domain side's job — see
[the three costs](/graphopt/explanation/three-costs).

Source:
[`mlopt_core/cost/`](https://github.com/handl-lang/eqsat/tree/main/packages/mlopt-core/src/mlopt_core/cost).
