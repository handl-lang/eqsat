# The e-graph — signature reference

<!-- provenance: eqsat:packages/mlopt-core/src/mlopt_core/egraph@2d6e4eb + graphopt:docs/public_api.md@ed89ead -->

Signature-level reference for the saturation core. The conceptual
account is [the e-graph and `dag_cost`](../explanation/egraph); the
frozen-name list is [the manifest slice](./frozen-api).

## `EGraph`

```python
eg = EGraph()
eid  = eg.add_term(term)                     # intern; -> root e-class id
eid  = eg.find(eid)                          # canonical e-class id
ok   = eg.rebuild(classes=None)              # canonicalise (+congruence
                                             #  on an unrestricted pass)
stats = eg.run(rules, root_eid,
               max_iterations=100, max_nodes=100_000,
               rule_budgets=None, stop="fixed_point",
               patience=3, cost_fn=None, policy=None)
term = eg.extract_best(eid, cost_fn, **kw)   # min-cost member
cert = eg.certificate(src, dst=None, *, root_eid=None, cost_fn=None)
```

| member | contract |
|---|---|
| `add_term(term)` | intern the term's DAG — every enode registered, shared subtrees shared; returns the root's e-class id |
| `find(eid)` | union-find lookup — the canonical e-class |
| `rebuild(classes=None)` | restore the congruence invariant after merges; `classes=None` is the unrestricted pass |
| `run(rules, root_eid, ...)` | equality saturation to the stop condition; returns the run-record `stats` dict (`stop`, `iterations`, `n_enodes`, `n_classes`, `rule_fires`, `rule_budgets`, `budget_suspended`). Bounds only ever *refuse* — exhaustion reports unfinished, never a fabricated merge |
| `extract_best(eid, cost_fn, **kw)` | greedy minimum-cost extraction; `overrides`/`bans`/`fusion_epsilon` ride in `**kw` |
| `certificate(src, dst, *, root_eid)` | the derivation the saturated graph witnesses between two terms — a replayable `Certificate` |
| `n_enodes` / `n_classes` / `rule_fires` | the run-record properties (`{rule_name: merge_count}` for the last run) |

`ENode` is an operator plus children e-class ids, hash-consed;
`EClass` is a congruence class of enodes; `UnionFind` merges classes a
rewrite proves equal.

## `Rewrite` — a law as a record

```python
Rewrite(
    name: str,
    lhs: Any,            # pattern term — leaf *strings* are metavars
    rhs: Any,
    law: str = "",       # provenance text
    check: Callable[[dict], bool] | None = None,   # side condition over
                                                   #  resolved bindings
    derive: Callable[[dict], dict | None] | None = None,  # computed attrs
    tags: frozenset[str] = frozenset(),  # mlopt_core.laws.tags constants
    error_bound: float | None = None,    # certified-approximation axis
    bound_norm: str = "spectral",
    derivation: tuple[str, ...] = (),    # axiom/lemma bookkeeping
    cond: Any = None,                    # declarative side condition
    dspec: Any = None,                   # declarative derive spec
)
```

`check`/`derive` are the procedural hooks; `cond`/`dspec` are the same
contracts as *pure data* (a `mlopt_core.laws.cond` tuple tree /
`{NAME: expr}` map), folded into `check`/`derive` at construction so
every evaluation site honours the conjunction through the single
`rule.check` convention. `derive` returning `None` vetoes the rewrite.
`error_bound`/`bound_norm` mark an ε-law — certificates aggregate the
per-step bounds conservatively.

## Certificates

```python
from mlopt_core.egraph import (
    Certificate, CertStep, ProofEdge, CertificateVerificationError,
    cert_to_data, cert_from_data, verify_certificate,
)

data = cert_to_data(cert)                 # -> dict — the transport form
cert = cert_from_data(data, rules)        # rehydrate against a rule set
out  = verify_certificate(src_term, cert, strict=False)
```

| name | role |
|---|---|
| `Certificate` | `src`/`dst`/`root_eid`/`steps`/`rules`/`stats` — the derivation |
| `CertStep` | `rule`/`path`/`lhs`/`rhs`/`bindings`/`egraph_dependent`/`note` |
| `ProofEdge` | `rule`/`a`/`b`/`subst`/`note` — the merge log |
| `verify_certificate(term, cert, *, strict=False)` | standalone replay — a certificate that does not replay raises `CertificateVerificationError` |
| `cert_to_data` / `cert_from_data` | the wire form, so a certificate crosses a cache or a backend boundary |

See [certificates](../explanation/certificates) for why the derivation —
not the timing — is the claim.

Source:
[`mlopt_core/egraph/`](https://github.com/handl-lang/eqsat/tree/main/packages/mlopt-core/src/mlopt_core/egraph).
