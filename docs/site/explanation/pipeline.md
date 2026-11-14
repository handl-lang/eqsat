# The pipeline — `mlopt_orchestrator.optimize` and its modules

<!-- provenance: eqsat:packages/mlopt-orchestrator/src/mlopt_orchestrator/optimize.py -->

`optimize.py` used to be a ~4,000-line god-module holding the whole
orchestrator. It is now the *entry-point* module — the verbs, the
strategies, `Optimizer`, and the rule-set resolution — with the
implementation split across sibling modules:

| module | owns |
|---|---|
| `resources.py` | `max_enodes`/`max_memory_mb` caps, `OptimizationResourceError`, the OOM → resource-error translation (`_oom_to_resource_error`, `_looks_like_oom`, `_check_resources`, `_current_memory_mb`). |
| `pricing.py` | the delivered-price extraction models — `_default_cost_fn`, `delivered_cost_for`/`_delivered_cost` (measured-feedback corrected), `_carrier_upgrade` (the whole-spine batched-executor win additive extraction can't price), `_upgrade_pricing`, and the `_lower_extracted`/`_route_spec` executor-table routing. |
| `bounds.py` | the `error_budget` machinery — the bound ledger (`_bound_gate`, `_bounded_term`'s ban loop), weight-space → output-space propagation (`_propagate_bounds`, `_entry_amp`, `_row_norm`), and the honest verify gates (`_bound_verify`, `_block_bound_gate`, `_bounded_gate`, `_BoundedVerify`). |
| `taskgate.py` | the `task=`/`task_tol=` contract — `_task_contract`/`_resolve_task`, the task-distance verify (`_task_verify`, `_block_verify_task`), the operative-gate composition (`_gate_delivery`, `_block_gate`), and the `verify_metric`/`accepted_by` stats labels. |
| `cache.py` | the structural-signature search cache — `structural_key`/`_canonical` (leaf-name-modulo canonicalisation), the replay machinery (`_cache_entry`, `_cache_replay`, `_replay_derived`, `_ties_hold`, `_remap_term`, `_derive_dedup`, `_dedup_blocks`, `_leaf_equal`, `_param_*` helpers, `_block_key`). |
| `lifts.py` | the non-local passes — `_pairing_and_lifts` (paired linear/conv extraction groups) and the detect-gated lifts (`_carrier_lifts`, `_headshare_lifts`, `_factor_lifts`, `_special_lifts`). |
| `compositional.py` | `_optimize_compositional` — the per-block driver behind the `Compositional` strategy — plus the fail-closed graft rollback (`_e2e_or_rollback`, `_roll_back_delivery`). |

`optimize.py` itself keeps: the resource-bound *decorator* usage on
the verbs, `_resolve_cost_fn`/`_resolve_rules`/`_resolve_engine`/
`default_rules`/`__getattr__` (the lazily-composed `DEFAULT_RULES` —
its module-global `_DEFAULT_RULES` cache is patch-point), 
`_select_best_term` (greedy → paired → carrier → causal → bound
gate), `_policy_kwargs`, `search`, `lower`,
`discover_alternatives`, `ir_to_string`/`term_cost`, the
`_SEARCH_KW`/`_LOWER_KW` keyword partition, the three strategy
classes, `Optimizer`, and the `_search`/`_lower` aliases its methods
bind.

## The compat contract

The split is **internal**: `from mlopt_orchestrator.optimize import
X` resolves every name the monolith used to define — the public
verbs/results/strategies *and* the private helpers tests and sibling
modules reach for (`_lower_extracted`, `_e2e_or_rollback`,
`_block_bound_gate`, `_bounded_gate`, `_rel_bound`,
`_propagate_bounds`, `_bound_verify`, `_carrier_plans`,
`_carrier_lifts`, `_task_contract`, `_accepted_by`, `_verify_metric`,
`_policy_kwargs`, `_check_resources`, `_current_memory_mb`,
`_looks_like_oom`, `_oom_to_resource_error`, the whole
`_cache_*`/`_canon_*`/`_param_*`/`_bound_*`/`_block_*`/`_task_*`
families). `optimize.py` re-exports them with explicit
`from ... import n as n` blocks; the file carries a file-level
`# ruff: noqa: I001` because isort's default would explode each
re-export into its own statement.

Two monkeypatch-sensitive points drove the granularity:

- `tests/test_backend_neutral.py` does
  `monkeypatch.setattr(O, "_DEFAULT_RULES", None)` — so
  `_DEFAULT_RULES`, `default_rules`, `__getattr__` and
  `_resolve_rules` stayed in `optimize.py` where the patched global
  is actually read.
- Sibling modules import helpers *through* `optimize`
  (`autotune` → `Optimizer`/`_lower_extracted`, `runners` →
  `_lower_extracted`, `diagram`/`morphisms.reify` →
  `_e2e_or_rollback`/`_block_bound_gate`/`_bounded_gate`/`_rel_bound`)
  — all satisfied by the re-export surface, and none of the new
  modules import `optimize` at top level (the dependency direction
  is one-way: `optimize` → siblings; `compositional` reaches back
  only under `TYPE_CHECKING`).
