# The ports — signature reference

<!-- provenance: eqsat:packages/mlopt-core/src/mlopt_core/ports.py@2d6e4eb + graphopt:docs/public_api.md@ed89ead -->

Signature-level reference for `mlopt_core.ports` — the `Protocol`
shapes where backends meet a substrate that never imports them. The
conceptual account is [the ports](../explanation/ports); the
frozen-name list is [the manifest slice](./frozen-api).

## The graph boundary — `Source` / `Capabilities` / `Sink`

```python
class Source(Protocol):
    def to_ir(self, model, example_inputs) -> tuple[IR, dict[str, Any]]:
        """Export `model` to IR plus its leaf values (params/buffers
        keyed by IR param name)."""

class Capabilities(Protocol):
    @property
    def supported_ops(self) -> frozenset[str]:
        """The op names the backend can lower — bounds the reachable
        equivalence class (out-of-set members price +inf)."""
    @property
    def ops(self) -> OpRegistry:
        """The lowering registry compile-time folds dispatch through."""

class Sink(Capabilities, Protocol):
    def lower(self, ir, params=None) -> Executor:
        """Materialise a runnable executor from `ir` and leaves."""
    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None
               ) -> VerifyResult:
        """Run `ref` and `opt` on `inputs`; return the report."""
    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """Named executor specs (mapping order is routing order;
        may be empty)."""
```

`search` consumes only `Capabilities` — a backend that wants to
*price* a search implements just the op surface; `Sink` adds
materialisation and verification.

## `Engine` — the `engine=` port

```python
class Engine(Protocol):
    def add_term(self, term, _memo=None, provenance="input") -> int: ...
    def find(self, eid) -> int: ...
    def rebuild(self, classes=None) -> bool: ...
    def run(self, rules, root_eid, max_iterations=100,
            max_nodes=100_000, rule_budgets=None,
            stop="fixed_point", patience=3, cost_fn=None
            ) -> dict[str, Any]: ...
    def extract_best(self, eid, cost_fn, **kw): ...
    n_enodes: int;  n_classes: int;  rule_fires: dict[str, int]
```

An `engine=` value is an instance, class, or zero-arg factory.
Engines are **never auto-detected**; `stats["engine"]` records which
ran (`"python"` reference default — `EGraph` itself is a conforming
`Engine` — `"native"` for `mlopt_native.NativeEngine`). An engine may
also accept `policy=` on `run` (reorders, never drops). The port
covers *search* only — proof machinery and the non-local passes are
Python-engine capabilities.

## Delivery & measurement

| port | member | contract |
|---|---|---|
| `Executor` | `forward(*xs) -> Any` | the runnable `lower` returns — backend-neutral (`xs`/`return` are `Any`); `PlannedExecutor`/`BatchedExecutor` refine it (`_plan`, `is_batched`) |
| `ExecutorSpec` | `name`/`lower`/`accepts`/`engaged`/`carrier` | one row of `Sink.executors` — the term-level `accepts` probe and module-level `engaged` probe drive routing |
| `Runner` | `name`; `apply(module, example_input, stats) -> module` | delivery-stage transform — wraps the lowered module, records `stats["compiled"]`-style keys; `ChainedRunner` composes left-to-right |
| `Meter` | `time(runnable, inputs, *, warmup=5, n_calls=30, timeout_s=None) -> TimingResult` | the `Autotuned` strategy's timing seam; median + IQR, classified failures |
| `Verifier` | `verify(ref, opt, inputs, *, rtol, atol) -> VerifyResult` | module-level equivalence (the torch impl delegates to `report.verify_module`) |
| `Composer` | `blocks`/`capture_inputs`/`graft`/`clone_sharing`/`boundary`/`perturbed` | the structural port `Compositional` needs — block picking, input capture, graft-back |

## The value & move ports

| port | member | contract |
|---|---|---|
| `CostFn` | `(term, memo=None) -> float` | the pricing convention every cost model follows |
| `Criterion` | `name`; `cost_fn(profile=None) -> CostFn` | one selection axis, calibrated to a target profile |
| `Policy` | `name`; `choose(state, actions) -> action` | *orders* the saturation schedule — every rule still fires, so a policy can never decide equivalence |
| `Strategy` | the whole-pipeline policy shape | `Monolithic`/`Compositional`/`Autotuned` in `mlopt_orchestrator` |
| `TaskMetric` | the task-aware correctness metric | pairs with `task_tol` on `lower`/`verify` |
| `PerformanceModel` | `name`; `predict(features, hardware=None) -> float` | predicted seconds — ranks, never prunes |

## The remaining vocabulary

`OpRegistry`, `RuleSetLike` (the structural twin of `egraph.Rewrite`'s
read surface — `name`/`lhs`/`rhs`/`law`/`check`/`derive`/
`error_bound`/`bound_norm`), `RuleLike`, `RuleSetProvider`,
`Binding`, `TimingResult`, `Profiler`, `VerifyResult`.

## `signature_conforms`

```python
signature_conforms(fn, proto, *, strict=False) -> bool
```

The companion to `isinstance` on `runtime_checkable` protocols:
`isinstance` checks member *presence*, `signature_conforms` checks a
callable's *shape* against a port's `__call__` — the check that keeps
a duck-typed `CostFn`/`Runner` honest.

Source:
[`mlopt_core/ports.py`](https://github.com/handl-lang/eqsat/blob/main/packages/mlopt-core/src/mlopt_core/ports.py).
