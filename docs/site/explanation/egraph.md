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

## Congruence closes at `rebuild`, not mid-saturation

A rewrite merge is not the only way two e-classes become equal:
enodes that are identical once their children are canonicalised must
share a class (congruence). `EGraph` maintains the machinery
incrementally — `_cong_owner` / `_cong_pend`, the lever-2c worklist —
but drains it only in `_close_congruence`, which runs on the
*unrestricted* `rebuild()`. Inside `run`, each iteration calls
`rebuild(dirty_frontier)`, which canonicalises children but does NOT
close congruence; the unrestricted `rebuild()` at the end of `run`
(and explicit `eg.rebuild()` calls between passes, as the
orchestrator makes) performs the closure.

Consequence: the per-iteration fixed point is **not**
congruence-closed. Two classes that only become structurally identical
through merges made late in a run stay apart while the remaining rules
fire, so a rewrite chain that needs the merged class (e.g. a repeated-
metavariable LHS whose two occurrences sit in congruent-but-unmerged
classes) can be missed until a later `run`. This is a completeness
gap, not a soundness one: congruence merges are always valid, they are
never skipped — the final `rebuild()` still performs them, and merged
classes carry the union of both member sets into extraction. Closing
congruence inside the loop would grow the fixed point (more members,
more rewrites reachable) at the cost of re-firing rules on merged
classes; it is a deliberate deferred-closure trade-off, matching how
`rebuild(classes=...)` documents the restricted pass.

## Cost memos are bounded

`extract_best` keeps one content-keyed cost memo per cost function on
the e-graph (`_cost_memo_for`), shared across every extraction pass so
repeat pricing is a memo hit. Both levels are bounded `_CostMemo`
maps — 100k entries per cost model, 1,024 distinct cost fns per
e-graph — with LRU-ish eviction. Eviction is semantics-preserving:
every entry is a pure function of `(cost_fn, term)` and terms are
interned, so a dropped entry is recomputed identically on the next
miss; the bound only caps memory on long discovery loops. The same
bounded-memo pattern caps the law-side shape memos
(`mlopt_core.laws.base._SHAPE_MEMO`, `laws.cond._MEMO`) — module-global
shape caches shared by every check hook, bounded at 100k entries for
the same recompute-only eviction reason.

## Rule sets compose — and scope — by name

`run`'s `rules` argument is a `mlopt_core.laws.RuleSet`: a named,
frozen collection with set algebra — `A + B` unions, `A - B`
subtracts, `A & B` intersects, all by *rule name*, with
same-name/different-rule collisions raising rather than silently
shadowing. Presets (`DEFAULT`, `FULL`, `SYMMETRY`, `CARRIER_SEARCH`,
…) are themselves compositions, and the orchestrator layers carrier
banks on top the same way.

`RuleSet.scoped(...)` derives a view without touching the bank:
`include`/`exclude` select members, `rename` retitles them
(priorities follow), and `predicate=` / `op=`+`metavar=` conjoin a
side condition into every member's `check` — the scope sees the same
resolved binding the rule's own check does, so "only fire when the
bound term's `.op` is `linear`" is `scoped(op="linear",
metavar="t")`. Scoped members are new `Rewrite` objects (the scope is
a closure, so scoped sets are pipeline-local, not
`laws.serialize`-portable) and collide with their unscoped twins on
union — `rename=` namespaces a scoped copy meant to coexist. Scope is
binding-level: there is no "only at the e-class root" — the match
enumeration does not surface position to `check`.

## Where it lives

`eqsat/packages/mlopt-core/src/mlopt_core/egraph/` (`core`, `match`,
`extract`, `terms`, `types`, `proof`, `certs`) and
`.../cost/` (`basic`, `backend`, `executor`, `fusion`, `params`,
`roofline`). The frozen surface is the
[eqsat slice of the manifest](../reference/frozen-api). `EGraph` is
the reference engine and the `engine="python-egraph"` opt-out; the
*default* `engine=` interpretation is `"eqsat"` — the engine-eqsat
(Rust) search behind `mlopt_orchestrator.eqsat`'s `eqsat-run` CLI
bridge (the WP4 parity sweep v4 GO), with `mlopt-native` the other
opt-in Rust engine in this repo.
