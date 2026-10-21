# The e-graph and `dag_cost`

<!-- provenance: eqsat:packages/mlopt-core/src/mlopt_core/{egraph,cost}@e0633d5 — summarized from the source + the frozen manifest -->

The e-graph is the substrate's central object: it holds *every*
equivalent spelling of a program at once, so search becomes "intern the
terms, saturate under the laws, extract the cheapest member" rather than
a walk over rewrite sequences.

## Terms in

`mlopt_core.ir` is the term language — `IR`, `Op`, `Var`, `Param`,
`Const`, `TensorType`. `EGraph.add_term(term)` interns a term and
returns its e-class id; an `ENode` is an operator plus children e-class
ids, hash-consed so identical subterms share storage, and a `UnionFind`
merges the classes a rewrite proves equal.

## Saturation is a closure, not a search

```python
stats = eg.run(rules, root_eid,
               max_iterations=100, max_nodes=100_000,
               rule_budgets=None, stop="fixed_point",
               patience=3, cost_fn=None, policy=None)
```

`run` fires the rule set to a fixed point — the e-graph *is* the set of
consequences, not a search over them. The controls are honest bounds:
`max_iterations`/`max_nodes`/`rule_budgets` cap the work, `stop`/`patience`
pick the termination mode, and `stats` reports `stop`, `n_enodes`,
`n_classes`, `rule_fires`. A bound can only ever *refuse* a claim —
resource exhaustion reports unfinished, never a fabricated merge.

`extract_best(eid, cost_fn, **kw)` then picks the cheapest member of the
saturated class under a cost function.

## `dag_cost` — the guess that ranks

`dag_cost(term, cost_fn, memo=None)` prices a program graph *honestly*:
shared subtrees are paid once, so a DAG is not billed as a tree. It sums
static weights — FLOPs per op, parameter bytes, node counts — with no
tensors ever executing. The shipped models cover the axes the
[three costs](/graphopt/explanation/three-costs) distinguish:

| family | models |
|---|---|
| counting | `count_cost`, `flops_cost`, `param_bytes_cost(_for)` |
| backend-aware | `backend_cost`, `executor_cost_for`, `fused_cost_for`, `lowering_aware_cost_for` |
| shape | `depth_cost(_for)`, `roofline_cost(_for)`, `launch_aware_cost` |
| fusion | `fusion_regions`, `fusion_member_key`, `CostModel`, `LOWERINGS` |

`dag_cost` has no notion of memory traffic, kernel fusion, or launch
overhead — a fused `sdpa` looks expensive on paper and is fast under
Inductor — so it is used for *ranking* candidates, never for *claims*.
The measured lanes (fixtures, verify) are the graphopt side's job; see
[the three costs](/graphopt/explanation/three-costs).

## Where it lives

`eqsat/packages/mlopt-core/src/mlopt_core/egraph/` (`core`, `match`,
`extract`, `terms`, `types`, `proof`, `certs`) and
`.../cost/` (`basic`, `backend`, `executor`, `fusion`, `params`,
`roofline`). The frozen surface is the
[eqsat slice of the manifest](../reference/frozen-api); the Rust engine
behind `engine=` is `mlopt-native` in the same repo.
