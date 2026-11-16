# ruff: noqa: I001
"""Top-level optimization pipeline — backend-neutral (plan 0007).

This module implements the four-phase killer experiment:

Phase 1 — Equivalence:   model → IR (through the ``source`` port)
Phase 2 — Search:        IR → e-graph → equality saturation → best term
Phase 3 — Lower:          best term → runnable (through the ``sink``
                        port's executor routing)
Phase 4 — Compare:        benchmark vs. vanilla TorchInductor

The public surface is the *verb* pair plus the configured entry
object (plan 0006), and the orchestrator is backend-neutral
(plan 0007): it orchestrates ANY backend through the
:class:`mlopt_core.ports` protocols and imports no torch.

* :func:`search` — phases 1+2: ``model -> SearchResult`` (the IR,
  saturated e-graph, extracted term, leaf values and search-record
  stats, all inspectable).
* :func:`lower` — phase 3 (+verify): ``SearchResult -> LowerResult``;
  re-lowering the same result under different runners delivers
  different executables from ONE search.  ``verify=True`` (the
  default) is substrate-enforced — a failed or missing verify report
  raises :class:`VerifyDeclinedError` instead of returning an
  unverified module.
* :class:`Optimizer` — the configured entry point: an explicit
  :class:`~mlopt_core.pipeline.Backend` (or explicit
  ``source``/``sink``/``composer``/``meter`` ports — no default
  backend) plus ``criteria``/``runner`` delivery defaults;
  ``.search`` / ``.lower`` / ``.optimize`` / ``.discover``.
* :class:`Monolithic` / :class:`Compositional` / :class:`Autotuned` —
  the :class:`~mlopt_core.ports.Strategy` seam behind
  ``Optimizer.optimize(..., strategy=...)``.

Backend specifics live on the adapter side: carrier executors arrive
through ``sink.executors``, causal-mask specialization through the
sink's optional ``specialize_causal`` hook, per-block structural
machinery through the ``composer`` port, and wall-clock timing
through the ``meter`` port.  ``Optimizer(backend=TorchBackend())``
is the supported torch spelling — no default is assumed anywhere in
this package.

The implementation is split across sibling modules — ``resources``
(the caps / OOM translation), ``pricing`` (delivered-cost models +
executor routing), ``bounds`` (the ``error_budget`` ledger and
bound gates), ``taskgate`` (the ``task=`` verify contract),
``cache`` (the structural-signature search cache), ``lifts`` (the
non-local pairing/lift passes) and ``compositional`` (the per-block
driver).  Every name they define stays importable from this module
— the split is internal, the surface is frozen.  The
``from ... import n as n`` re-export blocks below are deliberate:
isort's default would explode each into its own statement, so the
file carries a file-level ``I001`` exemption (the blocks are
hand-sorted).
"""

from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from mlopt_core.cost import backend_cost, dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, op_repr
from mlopt_core.laws import (
    RuleSet,
    preset,
    tags as _law_tags,
)
from mlopt_core.pipeline import (
    Backend,
    LowerResult,
    PipelineStats,
    SearchResult,
)
from mlopt_core.ports import (
    Capabilities,
    Composer,
    CostFn,
    Meter,
    Runner,
    Sink,
    Source,
    Strategy,
    TaskMetric,
    VerifyOracle,
)

from mlopt_orchestrator.bounds import (
    ErrorBoundEntry as ErrorBoundEntry,
    _ban_bound_members as _ban_bound_members,
    _block_bound_gate as _block_bound_gate,
    _bound_gate as _bound_gate,
    _bound_ledger as _bound_ledger,
    _bound_p as _bound_p,
    _bound_record as _bound_record,
    _bound_rules as _bound_rules,
    _bound_verify as _bound_verify,
    _bounded_gate as _bounded_gate,
    _bounded_term as _bounded_term,
    _BoundedVerify as _BoundedVerify,
    _check_budget_engine as _check_budget_engine,
    _delivered_bound as _delivered_bound,
    _entry_amp as _entry_amp,
    _error_bound_entries as _error_bound_entries,
    _propagate_bounds as _propagate_bounds,
    _rel_bound as _rel_bound,
    _row_norm as _row_norm,
    _site_data_term as _site_data_term,
    _site_eval as _site_eval,
    _subterms as _subterms,
    _term_params as _term_params,
)
from mlopt_orchestrator.cache import (
    _block_key as _block_key,
    _cache_entry as _cache_entry,
    _cache_replay as _cache_replay,
    _canon_attr as _canon_attr,
    _canon_leaf as _canon_leaf,
    _canon_term as _canon_term,
    _canonical as _canonical,
    _dedup_blocks as _dedup_blocks,
    _derive_dedup as _derive_dedup,
    _leaf_equal as _leaf_equal,
    _param_leaves as _param_leaves,
    _param_order as _param_order,
    _param_shapes_ok as _param_shapes_ok,
    _remap_term as _remap_term,
    _replay_derived as _replay_derived,
    _term_param_names as _term_param_names,
    _ties_hold as _ties_hold,
    structural_key as structural_key,
)
from mlopt_orchestrator.carriers import get_carriers
from mlopt_orchestrator.compositional import (
    _e2e_or_rollback as _e2e_or_rollback,
    _optimize_compositional as _optimize_compositional,
    _roll_back_delivery as _roll_back_delivery,
)
from mlopt_orchestrator.criteria import (
    Criteria,
    Criterion,
    criteria_cost,
)
from mlopt_orchestrator.lifts import (
    _carrier_lifts as _carrier_lifts,
    _factor_lifts as _factor_lifts,
    _headshare_lifts as _headshare_lifts,
    _pairing_and_lifts as _pairing_and_lifts,
    _special_lifts as _special_lifts,
)
from mlopt_orchestrator.pricing import (
    _DELIVERED_CANDIDATES as _DELIVERED_CANDIDATES,
    _carrier_plans as _carrier_plans,
    _carrier_upgrade as _carrier_upgrade,
    _corrected_delivered as _corrected_delivered,
    _default_cost_fn as _default_cost_fn,
    _delivered_cost as _delivered_cost,
    _lower_extracted as _lower_extracted,
    _route_spec as _route_spec,
    _upgrade_pricing as _upgrade_pricing,
    delivered_cost_for as delivered_cost_for,
)
from mlopt_orchestrator.resources import (
    OptimizationResourceError as OptimizationResourceError,
    _check_resources as _check_resources,
    _current_memory_mb as _current_memory_mb,
    _looks_like_oom as _looks_like_oom,
    _oom_to_resource_error as _oom_to_resource_error,
)
from mlopt_orchestrator.runners import IdentityRunner
from mlopt_orchestrator.taskgate import (
    _accepted_by as _accepted_by,
    _block_decline as _block_decline,
    _block_gate as _block_gate,
    _block_verify_task as _block_verify_task,
    _contract_tol as _contract_tol,
    _gate_delivery as _gate_delivery,
    _resolve_task as _resolve_task,
    _require_verified_delivery as _require_verified_delivery,
    _run_exec as _run_exec,
    _task_contract as _task_contract,
    _task_declared as _task_declared,
    _task_verify as _task_verify,
    _TaskVerify as _TaskVerify,
    _verify_metric as _verify_metric,
    VerifyDeclinedError as VerifyDeclinedError,
)

#: Bounded saturation: the rules tagged
#: :data:`mlopt_core.laws.tags.EXPANSIVE` — the pure-symmetry monoid
#: laws enumerating every bracketing/ordering of a summation
#: (Catalan-scale on the residual accumulator), the scale-hoist laws
#: pairing every scale member with every linear, and the
#: distribute/factor/naturality/assoc algebra generating cross-product
#: closures (enodes grew 337 → 40k in four iterations on a
#: DeepParallel stack) — run under a per-rule enode budget, which
#: truncates the reordering closure but leaves every content-bearing
#: rewrite at the exact fixed point.  Structural fusions
#: (qkv/swiglu/sdpa folds, gqa_absorb) and the simplification
#: singletons stay unbudgeted: their matches are pattern-specific,
#: not closure-generating.  Measured on stacked ParallelBlocks (the
#: model that motivated ``optimize_compositional``): identical
#: extracted cost at every budget ≥ 512 while saturation drops from
#: minutes to ~1s.
logger = logging.getLogger("mlopt_orchestrator.optimize")


# ---------------------------------------------------------------------------
#  The verbs — search (phases 1+2) and lower (phase 3 + verify)
# ---------------------------------------------------------------------------


def _resolve_cost_fn(
    cost_fn: CostFn | None,
    criteria: Any,
    capabilities: Capabilities | None,
) -> tuple[CostFn, Any]:
    """Selection-model precedence: ``cost_fn`` > ``criteria`` > default.

    Returns ``(cost_fn, criteria_used)`` — the normalised-blend marker
    ``criteria_cost`` sets is read BEFORE the ``backend_cost`` wrap,
    which propagates only the billing markers.  With ``capabilities``
    given, members using an op the backend cannot lower price at
    ``+inf``, so extraction never commits to one; omitted, pricing is
    backend-agnostic.
    """
    if cost_fn is None:
        cost_fn = (
            criteria_cost(criteria)
            if criteria is not None
            else _default_cost_fn()
        )
    criteria_used = getattr(cost_fn, "criteria", None)
    if capabilities is not None:
        cost_fn = backend_cost(cost_fn, capabilities.supported_ops)
    return cost_fn, criteria_used


#: The composed default rule set — ``mlopt_core.laws.DEFAULT`` plus
#: the carrier-package ``CARRIERS`` preset.  Composed **once,
#: lazily**: the carrier preset arrives through
#: :mod:`mlopt_orchestrator.carriers` (the carrier package registers
#: it), so the orchestrator imports no backend; a process without
#: carriers simply contributes the core default.  Subsumed- and
#: symmetry-tagged
#: rules are excluded by the *preset*, not by a filter in this module
#: — the ``_SUBSUMED`` hidden list and the ``ruleset: str`` switch
#: are gone (plan 0009).
_DEFAULT_RULES: RuleSet | None = None


def default_rules() -> RuleSet:
    """Return the pipeline's composed default rule set (``DEFAULT_RULES``)."""
    global _DEFAULT_RULES
    if _DEFAULT_RULES is None:
        from mlopt_core import laws

        rs = laws.DEFAULT
        m = get_carriers()
        if m is not None:
            rs = rs + m.preset()
        _DEFAULT_RULES = replace(
            rs,
            name="default_rules",
            description=(
                "the pipeline default — core DEFAULT + the "
                "carrier-package CARRIERS preset"
            ),
        )
    return _DEFAULT_RULES


_BF16_SAFE: RuleSet | None = None


def bf16_safe_rules() -> RuleSet:
    """Return the composed default minus the sdpa-surface rules.

    The ``rules="bf16_safe"`` resolution: :func:`default_rules` (core
    ``DEFAULT`` + the registered carrier preset) filtered by
    :func:`mlopt_core.laws.sdpa_surface` — every rule whose lhs or rhs
    pattern mentions the fused ``sdpa`` op.  Those rules are exact
    laws but precision-sensitive at reduced precision: the verify
    reference lowers the *spelled* exported IR while the ``sdpa``
    kernel keeps softmax/accumulation in its internal dtype, and the
    gap compounds with depth (measured: whisper-turbo bf16, 32
    layers, rel 0.418 vs the 0.0625 floor —
    ``graphopt:docs/reports/turbo_bf16_delivery.md``).  The filter is
    pattern-level, so carrier-package producers/decomposers
    (``attn_sdpa_*`` / ``sdpa_cat_*`` / …) are excluded without a
    name registry.  fp32/fp64 callers keep :func:`default_rules`.
    """
    global _BF16_SAFE
    if _BF16_SAFE is None:
        from mlopt_core.laws.ruleset import sdpa_surface

        base = default_rules()
        kept = tuple(r for r in base if not sdpa_surface(r))
        names = {r.name for r in kept}
        _BF16_SAFE = RuleSet(
            "bf16_safe",
            kept,
            {k: v for k, v in base.priorities.items() if k in names},
            description=(
                "default_rules minus the sdpa-surface rules — the "
                "bf16/fp16-safe delivery set under verify=True "
                "(the spelled-IR reference vs the fused kernel's "
                "internal precision compounds with depth)"
            ),
        )
    return _BF16_SAFE


def __getattr__(name: str) -> Any:
    """Lazily materialise ``DEFAULT_RULES`` (carrier composition)."""
    if name == "DEFAULT_RULES":
        return default_rules()
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def _resolve_rules(rules: Any) -> RuleSet:
    """Normalise a ``rules`` argument to a :class:`RuleSet`.

    ``None`` → :data:`DEFAULT_RULES` (the composed default above).
    A string resolves to a named preset — the orchestrator-level
    names (``default`` / ``bf16_safe`` / ``carrier_search`` / ``xc``)
    first, then :func:`mlopt_core.laws.preset`; anything else
    iterable becomes
    an anonymous set.  The old ``ruleset: str`` switch and the hidden
    ``_SUBSUMED`` exclusion are gone — the preset itself says what
    is in.
    """
    if rules is None:
        return default_rules()
    if isinstance(rules, RuleSet):
        return rules
    if isinstance(rules, str):
        if rules == "carrier_search":
            from mlopt_orchestrator.regime import _carrier_search

            return _carrier_search()
        if rules == "xc":
            from mlopt_orchestrator.regime import _xc_rules

            return _xc_rules()
        if rules in ("default", "default_rules"):
            return default_rules()
        if rules in ("bf16_safe", "bf16-safe"):
            return bf16_safe_rules()
        return preset(rules)
    return RuleSet("custom", tuple(rules))


#: The named engine spellings ``_resolve_engine`` accepts.  ``None``
#: resolves to :data:`_DEFAULT_ENGINE_NAME` — the engine-eqsat (Rust)
#: interpretation since the WP4 parity sweep v4 GO
#: (``host:project/reports/wp4_parity_sweep_v4.md``); the Python
#: reference engine stays one explicit name away.
_DEFAULT_ENGINE_NAME = "eqsat"
_ENGINE_NAMES = ("eqsat", "python-egraph", "python", "egraph")


def _resolve_engine(engine: Any) -> Any:
    """Materialise the saturation engine for :func:`search`.

    ``None`` → :data:`_DEFAULT_ENGINE_NAME` — ``"eqsat"``, the
    engine-eqsat (Rust) search interpretation behind the same
    :class:`~mlopt_core.ports.Engine` port (ADR-0174/0178; the WP4
    parity sweep v4 GO flipped the default).  A missing ``eqsat-run``
    binary is a named :class:`~mlopt_orchestrator.eqsat.EqsatUnavailableError`
    at construction — never a silent fallback; pass
    ``engine="python-egraph"`` for the reference engine.  The string
    names ``"eqsat"`` / ``"python-egraph"`` (aliases ``"python"`` /
    ``"egraph"``) select the built-in engines; a class or zero-arg
    factory is called to produce the engine; an engine instance —
    anything carrying the :class:`~mlopt_core.ports.Engine` surface,
    duck-typed on ``add_term`` — is used directly.  Engines are
    **never auto-detected**: installing ``mlopt_native`` changes
    nothing until ``engine=`` is passed.
    """
    if engine is None:
        engine = _DEFAULT_ENGINE_NAME
    if isinstance(engine, str):
        if engine == "eqsat":
            # Resolved lazily so importing the orchestrator never
            # requires the eqsat-run binary — construction is where
            # the binary resolves (and declines by name).
            from mlopt_orchestrator.eqsat import EqsatEngine

            return EqsatEngine()
        if engine in ("python-egraph", "python", "egraph"):
            return EGraph()
        raise ValueError(
            f"unknown engine name {engine!r} — expected one of "
            f"{_ENGINE_NAMES} or an Engine instance/class/factory"
        )
    if isinstance(engine, type) or not hasattr(engine, "add_term"):
        engine = engine()
    return engine


def _select_best_term(
    eg: EGraph,
    root_eid: int,
    cost_fn: CostFn,
    stats: dict[str, Any],
    *,
    groups: list,
    fusion_epsilon: float,
    delivers_compiled: bool,
    specialize_causal: bool,
    capabilities: Capabilities | None,
    source_tensors: dict,
    error_budget: float | None = None,
    src: Any = None,
    x: Any = None,
) -> Any:
    """Extract the search's term: greedy -> paired -> carrier -> causal.

    * ``extract_best`` — the additive DAG-cost minimum under
      ``cost_fn`` (``fusion_epsilon`` arms the near-tie fusion band);
    * paired extraction — with pairing groups present, force every
      paired member to its split enode and compare true DAG costs
      (one shared memo prices the baseline once);
    * :func:`_carrier_upgrade` — the whole-spine batched-executor win
      the additive decomposition can't price, billed under the
      intended delivery (``delivers_compiled``) and corrected by the
      measured-feedback data a profile-carrying ``cost_fn`` supplies
      (the ``profile`` marker + this input's ``shape_bucket`` — ``x``
      keys the corrections);
    * the causal-mask const fold — the opt-out specialization, run
      through the capabilities object's optional
      ``specialize_causal(term, params, memo) -> term`` hook (a
      ``Sink`` carries it for the torch backend): no hook, no fold;
    * the error-budget gate (``error_budget``, plan 0012) — the
      chosen term's bound ledger (the ``src`` -> ``best_term``
      certificate plus delivered-member fingerprints, since a
      budget-exhausted derivation undercounts) must total at most
      ``error_budget``; a violation re-extracts through
      :func:`_bounded_term`'s ban loop.  Every bound-carrying member
      the accepted term delivers lands in ``stats["error_bounds"]``
      — never silent.
    """
    best_term = eg.extract_best(
        root_eid, cost_fn, fusion_epsilon=fusion_epsilon
    )
    if groups:
        # Coordinated extraction: force every paired member to its
        # split enode AND steer consumers through the shared GEMM.
        # Compare true DAG costs — forcing loses if a group is only
        # partially reachable or a bypassing alternative was already
        # cheaper.  One shared memo across both calls: a forced term
        # shares most subterms with the best term, and the baseline
        # price is computed once, not per candidate.
        _dc_memo: dict = {}
        _best_dag = dag_cost(best_term, cost_fn, memo=_dc_memo)
        forced = eg.extract_paired(root_eid, cost_fn, groups)
        # Honest-decline bookkeeping: an un-extractable forced term
        # prices at +inf, so the same comparison decides and the stats
        # record the verdict with the cost delta.
        _forced_dag = (
            dag_cost(forced, cost_fn, memo=_dc_memo)
            if forced is not None
            else float("inf")
        )
        if _forced_dag <= _best_dag:
            best_term = forced
            stats["paired_extract"] = True
        else:
            stats["paired_extract"] = False
            stats["paired_delta"] = _forced_dag - _best_dag
    # Coordinated carrier selection: a batched-executor win is a
    # whole-spine property the additive extraction can't price.  The
    # delivered-price comparison runs under the profile the selection
    # model carries (the ``profile`` marker ``backend_cost`` forwards
    # — ``executor_cost_for(profile, ...)`` and criteria blends set
    # it): a measured-calibrated cost_fn corrects the
    # batched-vs-generic routing; no marker keeps the built-in
    # constants.  ``bucket`` keys the profile's measured corrections
    # to this input's shape class.
    profile, bucket = _upgrade_pricing(cost_fn, x)
    best_term = _carrier_upgrade(
        eg,
        root_eid,
        best_term,
        cost_fn,
        profile=profile,
        compiled=delivers_compiled,
        bucket=bucket,
    )
    # Causal specialization: a param-only attn_mask that evaluates to a
    # lower-triangular keep-mask is is_causal=True — no mask op at all.
    # The fold is backend machinery (it evaluates parameter subtrees in
    # the backend's own runtime) — the capabilities port surfaces it
    # via the optional ``specialize_causal`` hook; a backend without
    # one simply skips the pass.
    if specialize_causal and capabilities is not None:
        fold = getattr(capabilities, "specialize_causal", None)
        if fold is not None:
            _cm: dict = {}
            pre_fold = best_term
            best_term = fold(best_term, source_tensors, _cm)
            if _cm.get("_hit"):
                stats["causal_specialized"] = True
                # The compositional cache stores the pre-fold term and
                # re-runs the fold on each hit block's own values —
                # the fold's verdict is value-dependent (the mask must
                # evaluate to the causal triangle), so the cached form
                # is the one BEFORE this rewrite.
                stats["pre_causal_term"] = pre_fold
    # The bound gate: only members whose delivered bound accumulates
    # to at most ``error_budget`` are acceptable — see ``_bound_gate``.
    best_term = _bound_gate(
        eg, root_eid, src, cost_fn, stats, best_term, error_budget
    )
    return best_term


def _check_detect_engine(
    eg: Any,
    detect_factors: bool,
    detect_specials: bool,
    detect_headshare: bool,
) -> None:
    """Refuse ``detect_*`` opt-ins on engines without witnessed unions.

    The passes offer witnessed members through the proof-carrying
    ``union(witness=...)`` surface — Python-``EGraph`` machinery the
    ``Engine`` port does not require.  Like ``error_budget``, an
    explicit opt-in that cannot run fails loudly rather than
    silently skipping the pass.
    """
    if isinstance(eg, EGraph):
        return
    asked = [
        n
        for n, on in (
            ("detect_factors", detect_factors),
            ("detect_specials", detect_specials),
            ("detect_headshare", detect_headshare),
        )
        if on
    ]
    if asked:
        raise TypeError(
            f"{', '.join(asked)} requires the reference EGraph engine "
            "— the detect passes offer witnessed members through "
            f"union(witness=...) (got {type(eg).__name__}; pass "
            "engine='python-egraph' to opt out of the eqsat default)"
        )


def _policy_kwargs(policy: Any, engine: Any) -> dict[str, Any]:
    """Return the engine kwargs a policy implies.

    The ``Engine`` port says an engine *may* accept ``policy``, so a
    caller cannot tell from the port alone.  A policy is therefore
    passed only to an engine whose ``run`` declares the parameter;
    anything else gets a clear error naming the engine, rather than a
    bare ``TypeError`` from a missing keyword deep inside the call.
    """
    if policy is None:
        return {}
    try:
        accepts = "policy" in inspect.signature(engine.run).parameters
    except (TypeError, ValueError):
        # A run() with no introspectable signature (an exotic callable)
        # cannot be confirmed to take a policy.
        accepts = False
    if not accepts:
        raise TypeError(
            f"engine {type(engine).__name__} does not accept a policy: "
            "its run() has no 'policy' parameter"
        )
    return {"policy": policy}


@_oom_to_resource_error
def search(
    model: Any,
    x: Any,
    *,
    source: Source,
    capabilities: Capabilities | None = None,
    criteria: (
        dict[str, float] | Criteria | Criterion | list | tuple | None
    ) = None,
    cost_fn: CostFn | None = None,
    rules: RuleSet | str | Iterable | None = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    symmetry_budget: int | None = 2048,
    max_memory_mb: float | None = None,
    specialize_causal: bool = True,
    fusion_epsilon: float = 0.0,
    delivers_compiled: bool = False,
    meter: Meter | None = None,
    verbose: bool = False,
    stop: str = "fixed_point",
    patience: int = 3,
    engine: Any = None,
    detect_factors: bool = False,
    detect_specials: bool = False,
    detect_headshare: bool = False,
    error_budget: float | None = None,
    task: TaskMetric | None = None,
    task_tol: float | None = None,
    policy: Any = None,
    corpus_probe: Callable[[str, Any, Any], Any] | None = None,
    device_policy: Any = None,
) -> SearchResult:
    """Run the search phase: ``model -> SearchResult``.

    Export → e-graph → bounded equality saturation → non-local passes
    → extraction.  The result carries everything the lower phase and
    an interactive caller need: the exported ``IR`` and leaf values,
    the saturated ``EGraph`` and root e-class, the extracted ``term``,
    the pricing actually used, and the search-record ``stats``.

    Parameters
    ----------
    model
        The model to optimize — the type only ``source`` interprets.
    x
        An example input for tracing (tensor or positional-args tuple).
    source : Source
        The graph-source port (``model -> (IR, leaves)``) — REQUIRED.
        There is no assumed frontend; choosing torch means importing
        ``kit_torch`` and passing ``TorchSource``.
    capabilities : Capabilities, optional
        The backend's op surface — ``supported_ops`` bounds extraction
        to forms the backend can lower (``backend_cost`` pricing), and
        its optional ``specialize_causal`` hook supplies the causal
        const fold's evaluator.  Omitted: pricing is backend-agnostic
        and the causal fold is skipped.  A :class:`Sink` satisfies it
        (every sink is a ``Capabilities``).
    cost_fn : CostFn, optional
        Term-extraction pricing; ``None`` falls to ``criteria`` then
        the executor-aware default.
    criteria : dict, Criterion, Criteria, or sequence, optional
        Selection axes blended into the extraction model — see
        ``optimize_model``.
    rules : RuleSet, str, or iterable, optional
        The saturation rule set — a first-class
        :class:`~mlopt_core.laws.RuleSet`, a preset *name* (resolved
        by :func:`mlopt_core.laws.preset` plus the orchestrator-level
        ``"default"`` / ``"bf16_safe"`` / ``"carrier_search"`` /
        ``"xc"``), or a plain
        iterable of rewrites.  ``None`` resolves to
        :data:`DEFAULT_RULES` — ``core.DEFAULT + carriers.CARRIERS``.
        Subsumed/symmetry exclusion is the preset's business: pass
        ``laws.DEFAULT + laws.SYMMETRY`` (or ``laws.FULL``) to opt
        into the symmetry generators.
    max_iterations : int
        Maximum equality-saturation iterations.
    max_enodes : int, optional
        E-node bound; crossing it raises
        :class:`OptimizationResourceError`.  ``None`` disables.
    max_memory_mb : float, optional
        Process memory bound in MiB.  ``None`` disables.
    symmetry_budget : int, optional
        Per-rule enode budget for the expansive rules; ``None`` is
        unbounded saturation.
    fusion_epsilon : float, default 0.0
        Near-tie fusion-preferred extraction band — see
        ``optimize_model``.
    specialize_causal : bool, default True
        The causal-mask const fold is an opt-out search pass: a
        param-only ``sdpa`` mask that evaluates to the lower-triangular
        keep-mask becomes ``is_causal=True``.
    delivers_compiled : bool, default False
        Delivery hint for carrier selection only: when the eventual
        delivery will be ``torch.compile``-wrapped, the carrier upgrade
        bills terms under the fusion-region (``lowering="compiled"``)
        price.  The runner itself is a ``lower`` concern.
    meter : Meter, optional
        The backend's timing port — consulted only for the optional
        device-memory read the ``max_memory_mb`` bound needs (host
        RSS is read portably).  ``None`` counts host bytes only.
    stop : {"fixed_point", "improving"}, default "fixed_point"
        Saturation stop policy (plan 0010, lever 1d).
        ``"fixed_point"`` runs to the exact fixed point — the
        historical behaviour and the default.  ``"improving"`` is an
        **opt-in heuristic**: the best term is re-extracted after
        every iteration and saturation stops once its cost has not
        strictly improved for ``patience`` consecutive iterations —
        most wins land in the first few iterations, but it can stop
        before the true optimum.  The outcome is recorded in
        ``stats["stop"]`` and ``stats["improved"]``.
    patience : int, default 3
        Consecutive non-improving iterations tolerated under
        ``stop="improving"``.
    engine : Engine, optional
        The saturation engine — an instance, class, or zero-arg
        factory conforming to :class:`mlopt_core.ports.Engine`, or a
        built-in name: ``"eqsat"`` (the engine-eqsat Rust
        interpretation, :class:`mlopt_orchestrator.eqsat.EqsatEngine`,
        ADR-0178) or ``"python-egraph"`` (the pure-Python reference
        ``EGraph``; aliases ``"python"`` / ``"egraph"``).  ``None``
        resolves to ``"eqsat"`` — the WP4 parity sweep v4 GO made it
        the default; a missing ``eqsat-run`` binary declines loudly
        with :class:`~mlopt_orchestrator.eqsat.EqsatUnavailableError`
        (pass ``engine="python-egraph"`` to opt out).  The optional
        native accelerator is ``mlopt_native.NativeEngine`` — engines
        are never auto-detected, and ``stats["engine"]`` records
        which ran.  The engine port covers the *search* only: the
        non-local pairing/lift passes (which mutate through the
        proof-carrying ``union(witness=...)`` surface) run only on
        the Python engine and are skipped otherwise; certificates
        likewise stay a Python-engine concern.
    detect_factors : bool, default False
        Opt-in low-rank detection pass
        (:func:`mlopt_core.laws.factored.offer_low_rank_factors`):
        weight parameters whose *stored values* factor at rank ``r``
        below the ``r·(i+o) < i·o`` break-even get a witnessed
        factored member (``x@A@B`` / ``linear(linear(x,A),B)``) in
        their consumer's e-class — a dense-but-low-rank ``Linear``
        then extracts as the two-GEMM chain when the cost model
        prefers it.  Spelled factorisations (LoRA ``lora_A`` /
        ``lora_B`` pairs, ``x@(A@B)`` terms) need no detection — the
        ``assoc_*`` laws already derive both directions and the cost
        model picks; this flag covers weights whose low rank is only
        visible in the values.
    detect_specials : bool, default False
        Opt-in structurally-special weight pass
        (:func:`mlopt_core.laws.specials.offer_weight_specials`):
        weight parameters whose *stored values* are identity,
        diagonal, zero, block-diagonal, or carry bitwise-dead input
        slices / bitwise-duplicate output slices get witnessed
        **exact** members (bound 0) — skip/pointwise-mul, narrowed
        projections plus gathers, per-block splits — in their
        consumer's e-class.  Unlike ``detect_factors`` nothing here
        is approximate: a slice is dead iff every value is exactly
        0.0, and duplicates dedupe bitwise.  See ``error_budget`` for
        the certified-approximate extension.
    detect_headshare : bool, default False
        Opt-in shared-head detection pass
        (:func:`mlopt_core.laws.headshare.share_duplicate_attention_heads`):
        ``sdpa`` sites whose per-head (Wq, Wk, Wv) weight blocks are
        *bitwise-equal* under a resolved head structure get a
        witnessed gather-``sdpa``-gather member computing each unique
        head once — exact (bound 0), competing through the normal
        extraction cost.  Applicability is deliberately narrow:
        shared-head architectures, quantization-induced ties, GQA
        kv replication, post-``WeightTie`` merges — ordinary trained
        weights almost never tie bitwise, in which case nothing fires.
    error_budget : float, optional
        Opt-in certified-approximation budget (plan 0012).  ``None``
        (the default) keeps the search exact — no bounded member is
        offered and no bound gate runs.  When a float is given:

        * ``detect_specials``'s pass additionally offers the bounded
          ``elide_bounded`` / ``zero_bounded`` members of
          :func:`mlopt_core.laws.specials.offer_weight_specials`
          (each with a measured ``error_bound ≤ error_budget``);
        * extraction accepts a term only when its ledgered bound —
          the certificate's accumulated ``error_bound`` plus any
          bound member the delivered term provably uses — stays
          within the budget (a derivation that exhausts its budget
          records ``egraph_dependent`` stubs which undercount, so
          delivered members are also fingerprinted directly);
        * the ledger is never silent: ``stats["error_budget"]``
          echoes the request, ``stats["error_bound_total"]`` is the
          accepted certificate bound (weight space), and
          ``stats["error_bounds"]`` lists every bound-carrying member
          used (``rule`` / ``law`` / ``bound`` / ``norm``); :func:`lower`
          propagates each ``bound`` to an ``output_bound`` on the
          verify input (:func:`_propagate_bounds`) and fills in
          ``measured_max_rel``.

        Requires the reference ``EGraph`` at ``truncation_level >= 2``
        (the bound ledger reads certificates and rule-application
        witnesses); anything else raises :class:`TypeError`.
    task : TaskMetric, optional
        Opt-in task-level equivalence contract (plan 0015) — a
        :class:`~mlopt_core.ports.TaskMetric` value such as
        :class:`~mlopt_core.metrics.TopKAgreement` or
        :class:`~mlopt_core.metrics.ArgmaxStability`.  The search is
        unchanged — extraction still picks the cheapest member under
        ``cost_fn`` — but the contract is *recorded*: it rides the
        result so :func:`lower` gates the delivered module's verify on
        ``task.distance(ref_out, opt_out) <= tolerance`` evaluated on
        the verify input, and ``stats["task"]`` names the metric and
        tolerance (distance/verdict land at verify time).  The
        certificate is then calibration-conditioned on the verify
        input — distribution shift is the honest caveat.
    task_tol : float, optional
        Override the metric's own ``tolerance``; required when the
        metric declares none.
    policy : Policy, optional
        An in-search action ordering consulted once per saturation
        iteration (``EGraph.run(..., policy=...)``).  It may only
        *reorder* the rules, so the fixed point and the certificate are
        unchanged; the name is recorded in ``stats["policy"]``.
    corpus_probe : callable, optional
        An injected per-model corpus/coverage probe — the
        backend-neutral seam downstream domain repos use (graphopt's
        ``kit_discovery.model_corpus`` is the intended callable;
        this package never imports it).  Invoked ONCE, right after
        export, as ``corpus_probe(name, model, x)`` where ``name`` is
        the model's ``__name__`` or class name; whatever it returns
        is recorded verbatim under ``stats["corpus_probe"]`` (it
        flows through ``lower`` into the delivery stats).  The probe
        owns its own side effects — including any dtype cast its
        intake performs on ``model`` — so a caller wiring a mutating
        probe should wrap it to restore what it changed.  ``None``
        (the default) runs nothing: the search is byte-for-byte the
        historical path.
    device_policy : DevicePolicy, optional
        The placement/refusal policy (Phase B) — a
        :class:`~mlopt_core.ports.DevicePolicy`-shaped value such as
        ``kit_torch.device.DevicePolicy``.  Invoked ONCE, before
        export, as ``device_policy.resolve(model, x)``: the resolved
        placement is recorded under ``stats["device"]``
        (``str()``-rendered — the substrate stores no device type)
        and any refusal the policy raises propagates — a mixed or
        distributed placement fails loudly before the e-graph is
        built, never delivers a silently wrong-device module.
        ``None`` (the default) runs the historical unpoliced path.
        The delivery side re-checks through ``lower``'s
        ``device_policy`` argument.
    verbose : bool
        Print progress.

    Returns
    -------
    SearchResult
        ``ir`` / ``eg`` / ``root_eid`` / ``term`` / ``param_values`` /
        ``stats`` (the search record) / ``cost_fn`` / ``source`` /
        ``model`` (the verify reference ``lower`` uses).

    """
    cost_fn, criteria_used = _resolve_cost_fn(
        cost_fn, criteria, capabilities
    )
    contract = _task_contract(task, task_tol)

    # -- Optional device/placement policy ------------------------------
    # Phase B: runs BEFORE export so a refusal (mixed devices,
    # distributed wrappers, sharded params — whatever the backend's
    # policy names) fails before the e-graph is built.  The resolved
    # placement lands in ``stats["device"]`` below; the substrate
    # never interprets it.
    device_resolved: Any = None
    if device_policy is not None:
        device_resolved = device_policy.resolve(model, x)

    # Recursive walks (extraction, member resolution) descend the
    # e-class DAG, whose depth grows with the saturation closure —
    # thousands of levels on deep stacks.
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    # -- Phase 1: Export to IR -------------------------------------------
    if verbose:
        print(
            f"[Phase 1] Exporting {model.__class__.__name__} to IR..."
        )
    ir, source_tensors = source.to_ir(model, x)
    if verbose:
        print(f"  IR root: {op_repr(ir.root)}")
        print(f"  Inputs:  {[str(v) for v in ir.inputs]}")
        print(f"  Params:  {list(ir.params.keys())}")

    # -- Optional caller-injected corpus probe -------------------------
    # Runs after export (the probe sees the same lifted boundary every
    # corpus term crossed) and before saturation, so its cost never
    # hides inside the search record's timings.  The report lands in
    # ``stats["corpus_probe"]`` below; exceptions propagate — an
    # opt-in diagnostic fails loudly, never silently.
    probe_report: Any = None
    if corpus_probe is not None:
        probe_report = corpus_probe(
            getattr(model, "__name__", None) or type(model).__name__,
            model,
            x,
        )

    # -- Phase 2: Build e-graph and saturate -----------------------------
    if verbose:
        print(
            "[Phase 2] Building e-graph and running equality "
            "saturation..."
        )
    eg = _resolve_engine(engine)
    _check_budget_engine(eg, error_budget)
    _check_detect_engine(
        eg, detect_factors, detect_specials, detect_headshare
    )
    root_eid = eg.add_term(ir.root)
    rules = _resolve_rules(rules)
    if verbose:
        print(f"  Rules: {[r.name for r in rules]}")

    # Bounded-saturation budget for the EXPANSIVE-tagged rules (the
    # tag replaced the ``_EXPANSIVE_RULES`` name list); enforced
    # inside the matcher so a giant e-class cannot spend the whole
    # budget in one enumeration.
    rule_budgets = (
        {
            r.name: symmetry_budget
            for r in rules.tagged(_law_tags.EXPANSIVE)
        }
        if symmetry_budget is not None
        else None
    )
    # ``None`` = unbounded: the run loop wants a concrete watermark.
    run_cap = max_enodes if max_enodes is not None else sys.maxsize

    run_kw = _policy_kwargs(policy, eg)
    # The pipeline stats record — a ``PipelineStats`` (dict subclass)
    # either engine's plain ``run`` dict normalises into; every
    # downstream writer keeps mutating it as a dict.
    stats: PipelineStats = PipelineStats(
        eg.run(
            rules,
            root_eid,
            max_iterations=max_iterations,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
            stop=stop,
            patience=patience,
            cost_fn=cost_fn,
            **run_kw,
        )
    )
    # Which saturation core ran — the reference engine needs no marker,
    # engines declare ``engine_name`` ("native", ...).
    stats["engine"] = getattr(eg, "engine_name", "python")
    _check_resources(eg, max_enodes, max_memory_mb, meter)

    if isinstance(eg, EGraph):
        groups = _pairing_and_lifts(
            eg,
            rules,
            root_eid,
            stats,
            run_cap,
            rule_budgets,
            max_enodes,
            max_memory_mb,
            source_tensors,
            meter,
            detect_factors,
            detect_specials,
            detect_headshare,
            error_budget,
        )
    else:
        # Non-local passes (pairing + carrier lifts) offer members
        # through the proof-carrying ``union(witness=...)`` surface —
        # a Python-engine capability the ``Engine`` port does not
        # require.  A run needing them (or certificates) uses the
        # Python engine; see the ``engine`` docstring.
        groups = []
        stats["nonlocal_passes"] = "skipped (engine is not an EGraph)"

    stats["rule_fires"] = dict(eg.rule_fires)
    stats["criteria"] = criteria_used
    if probe_report is not None:
        stats["corpus_probe"] = probe_report
    if device_policy is not None:
        stats["device"] = str(device_resolved)
    if fusion_epsilon:
        stats["fusion_epsilon"] = fusion_epsilon
    _task_declared(stats, contract)
    if verbose:
        print(f"  E-graph: {stats}")

    # -- Extract best term -----------------------------------------------
    best_term = _select_best_term(
        eg,
        root_eid,
        cost_fn,
        stats,
        groups=groups,
        fusion_epsilon=fusion_epsilon,
        delivers_compiled=delivers_compiled,
        specialize_causal=specialize_causal,
        capabilities=capabilities,
        source_tensors=source_tensors,
        error_budget=error_budget,
        src=ir.root,
        x=x,
    )

    if verbose:
        print(f"  Best term: {op_repr(best_term)}")
        print(f"  Cost: {cost_fn(best_term):.2f} FLOPs (est.)")

    # Final watermark before the lower phase materialises parameters —
    # the phase that turns graph choices into real tensor bytes.
    _check_resources(eg, max_enodes, max_memory_mb, meter)

    return SearchResult(
        ir=ir,
        eg=eg,
        root_eid=root_eid,
        term=best_term,
        param_values=source_tensors,
        stats=stats,
        cost_fn=cost_fn,
        source=source,
        model=model,
        task=task,
        task_tol=_contract_tol(contract),
    )


def _oracle_env(ir: IR, param_values: dict, x: Any) -> dict:
    """Leaf bindings for a :class:`VerifyOracle` run.

    ``param_values`` copies verbatim — its keys are the IR's
    ``Param.name`` leaf names, the same contract ``sink.lower``
    consumes.  The program inputs bind from ``x``: a ``Mapping``
    binds by leaf name; a tuple pairs positionally with
    ``ir.inputs``; any other value binds a single-input program's
    one ``Var``.  A count mismatch is a loud ``TypeError`` — the
    oracle never guesses which input is which.
    """
    env = dict(param_values)
    inputs = list(ir.inputs)
    if isinstance(x, Mapping):
        env.update(x)
        return env
    xs = () if x is None else (x if isinstance(x, tuple) else (x,))
    if len(xs) != len(inputs):
        raise TypeError(
            f"verify oracle: IR declares {len(inputs)} inputs but "
            f"x supplies {len(xs)} values"
        )
    for v, val in zip(inputs, xs, strict=True):
        env[v.name] = val
    return env


@_oom_to_resource_error
def lower(
    result: SearchResult,
    x: Any,
    *,
    sink: Sink,
    runner: Runner | None = None,
    verify: bool = True,
    verify_oracle: VerifyOracle | None = None,
    rtol: float = 1e-4,
    atol: float | None = None,
    task: TaskMetric | None = None,
    task_tol: float | None = None,
    device_policy: Any = None,
    verbose: bool = False,
) -> LowerResult:
    """Run the lower phase: ``SearchResult -> LowerResult``.

    Executor routing (``_lower_extracted``): a term the sink's
    executor table claims — the first carrier ``ExecutorSpec`` whose
    ``accepts`` probe holds — goes to that level-batched executor;
    everything else to ``sink.lower``.  The delivery runner then
    decides HOW the routed executor ships — identity
    (``IdentityRunner`` — the default) or a backend-provided
    transform like ``kit_torch.runners.TorchCompileRunner`` /
    ``kit_cuda.CudaGraphRunner`` / a ``ChainedRunner``
    composition.
    ``stats`` is a FRESH ``PipelineStats`` — the search record plus
    the lowering keys; :attr:`SearchResult.stats` is never mutated —
    so one search can feed many deliveries.

    Parameters
    ----------
    result : SearchResult
        What :func:`search` (or ``Optimizer.search``) produced.
    x
        The example input — the runner needs it (compile probes,
        graph capture) and ``sink.verify`` runs on it.
    sink : Sink
        The graph-sink port — REQUIRED.  There is no assumed backend.
    runner : Runner, optional
        The delivery transform; ``None`` ships as lowered.
    verify : bool, default True
        Run the sink's equivalence gate.  The reference is a fresh
        lowering of ``result.ir`` — the un-optimized program the
        search exported — via ``sink.lower``: the verify contract is
        "the delivery preserves the IR the search certified", stated
        in the sink's own runtime so a non-torch backend works
        unchanged.  The report lands in ``LowerResult.verified``.
        Passing a :class:`~mlopt_core.ports.VerifyOracle` here is
        shorthand for ``verify=True, verify_oracle=oracle``.
        The gate is enforced, not advisory: a failed operative
        verdict (pointwise/bound, or the task distance under a
        ``task=`` contract) — or a missing report — raises
        :class:`VerifyDeclinedError` carrying the declined
        ``LowerResult`` as ``.result``.  ``verify=False`` is the
        explicit opt-out: no gate runs and ``verified`` is ``None``.
    verify_oracle : VerifyOracle, optional
        Opt-in term-level verify gate (plan 0037/0039) for backends
        whose artifacts have no torch runtime to verify in.  When
        given, the pointwise equivalence check runs through the
        oracle instead of ``sink.verify``: the oracle evaluates
        ``result.ir.root`` (the source program) and ``result.term``
        (the extracted program) under leaf bindings built from ``x``
        and ``result.param_values`` (:func:`_oracle_env`) and its
        ``compare`` produces the report — bound propagation and the
        task gate are unchanged.  The oracle observes *terms*, not
        the delivered module: it certifies the search's term-level
        equivalence, so lowering correctness remains the sink's (or
        the artifact's own) concern.  ``None`` keeps the
        ``sink.verify`` path — zero behaviour change.
    rtol, atol
        The equivalence tolerances, forwarded to ``sink.verify``.
        When the search ran under ``error_budget`` and accepted
        bounded members (``stats["error_bound_total"]``), the bound
        replaces the relative tolerance: the certificate's bound is
        in *weight* space, so it is first propagated to output units
        (:func:`_propagate_bounds` — ``bound · site-input norm`` per
        entry, evaluated on ``x``), the sink verify runs at
        ``rtol=inf``, and the *measured* ``max|Δy|`` must stay within
        the propagated ``stats["error_bound_output"]`` — a violation
        declines the delivery (``verified.passed`` False,
        ``stats["error_bounds_honored"]`` False), even when the
        relative difference was small.  Each
        ``stats["error_bounds"]`` entry records its propagated
        ``output_bound`` and the measured ``max_rel`` — the ledger is
        never silent.
    task : TaskMetric, optional
        Opt-in task-level gate (plan 0015) — a
        :class:`~mlopt_core.ports.TaskMetric` value.  When set, the
        verify *verdict* is the task distance: the delivered module
        and the fresh reference lowering run on ``x`` and
        ``verified.passed`` becomes
        ``task.distance(ref_out, opt_out) <= tolerance`` — the
        certificate is calibration-conditioned on ``x``.  The
        pointwise measurements are still reported
        (``verified.max_abs`` / ``verified.max_rel``), the bound
        ledger still populates, and a task-accepted bounded member
        records ``accepted_by="task"`` — never silent.  ``None``
        inherits the contract recorded by :func:`search`
        (``result.task`` / ``result.task_tol``).
    task_tol : float, optional
        Tolerance override; defaults to the result's recorded
        tolerance, then the metric's own ``tolerance``.
    device_policy : DevicePolicy, optional
        The delivery-side placement/refusal policy (Phase B) — a
        :class:`~mlopt_core.ports.DevicePolicy`-shaped value.  After
        the runner produces the delivered module the policy's
        ``check_module`` hook (when it provides one) runs on it —
        DDP/FSDP/TP wrappers, collective bindings and sharded params
        raise rather than deliver — and ``resolve`` answers the
        delivery device under ``stats["delivery_device"]``
        (``str()``-rendered).  ``None`` skips both; pass the same
        policy :func:`search` ran for end-to-end coverage.
    verbose : bool
        Print progress.

    """
    if runner is None:
        runner = IdentityRunner()
    # ``verify=<oracle>`` shorthand — an oracle passed in the verify
    # slot IS the gate (and implies verify=True).
    if verify_oracle is None and isinstance(verify, VerifyOracle):
        verify_oracle = verify
        verify = True
    stats: PipelineStats = PipelineStats(result.stats)
    # Task contract precedence — explicit ``task=`` wins; otherwise
    # the search's recorded contract governs (see ``_resolve_task``).
    task, contract = _resolve_task(task, task_tol, result)
    if verbose:
        print("[Phase 3] Lowering optimized IR to torch module...")
    params = dict(result.param_values)
    optimized_ir = IR(
        root=result.term,
        inputs=result.ir.inputs,
        input_names=result.ir.input_names,
        params=result.ir.params,
    )
    optimized_module = _lower_extracted(
        result.term, optimized_ir, params, sink
    )
    stats["lowering"] = (
        "batched"
        if getattr(optimized_module, "is_batched", False)
        else "generic"
    )
    # Delivery: the runner decides how the routed executor ships.
    # stats["runner"] records the name (member-name list for a chain);
    # runners write their own outcome keys (stats["compiled"],
    # stats["cuda_graph"]).
    runner_names = getattr(runner, "names", None)
    stats["runner"] = (
        list(runner_names)
        if runner_names is not None
        else getattr(runner, "name", type(runner).__name__)
    )
    optimized_module = runner.apply(optimized_module, x, stats)

    # -- Optional device/placement policy (delivery side) -------------
    # The search checked the SOURCE placement; here the DELIVERED
    # module is checked — a runner may have wrapped or moved it, and
    # the policy's refusal must see what actually ships.  ``check``
    # is duck-typed: a resolve-only policy stays valid.
    if device_policy is not None:
        check = getattr(device_policy, "check_module", None)
        if callable(check):
            check(optimized_module)
        stats["delivery_device"] = str(
            device_policy.resolve(optimized_module, x)
        )

    # Verify semantic equivalence — the delivered module against the
    # un-optimized IR lowered by the same sink (backend-neutral: the
    # reference is a runnable in the sink's runtime, which the
    # source-side model need not be).
    verified = None
    if verify:
        if verbose:
            print("[Verify] Checking output equivalence...")
        ref = sink.lower(result.ir, params)
        bound_total = float(stats.get("error_bound_total", 0.0))
        # The certified bound is in weight space; the gate compares the
        # measured output error against the *propagated* bound (same
        # ``max|Δy|`` units).  Under a bound the sink's relative
        # tolerance opens (rtol=inf): the bound replaces it — honoring
        # it implies ``max_rel ≤ bound`` restated in rel units, so
        # ``max(rtol, bound)`` semantics are preserved.
        if verify_oracle is None:
            report, output_bound = _block_bound_gate(
                sink,
                ref,
                optimized_module,
                x,
                rtol,
                stats,
                result,
                atol,
            )
        else:
            # Term-level oracle gate: evaluate the source IR root and
            # the extracted term under one leaf environment; the
            # oracle's ``compare`` is the pointwise report.  Bound
            # propagation and the task gate downstream are unchanged.
            env = _oracle_env(result.ir, params, x)
            oref = verify_oracle.run(result.ir.root, env)
            oout = verify_oracle.run(result.term, env)
            report, output_bound = _block_bound_gate(
                sink,
                ref,
                optimized_module,
                x,
                rtol,
                stats,
                result,
                atol,
                verify_fn=lambda rt: verify_oracle.compare(
                    oref, oout, rtol=rt, atol=atol
                ),
            )
        verified = _gate_delivery(
            stats,
            report,
            bound_total,
            output_bound,
            task,
            contract,
            ref,
            optimized_module,
            x,
        )
        if verbose:
            print(f"  Max abs diff:  {verified.max_abs:.6e}")
            print(f"  Max rel diff:  {verified.max_rel:.6e}")
            if verified.passed:
                print("  ✓ Semantically equivalent (within tolerance)")
            else:
                print("  ✗ WARNING: large difference detected!")

    delivered = LowerResult(
        module=optimized_module, stats=stats, verified=verified
    )
    if verify:
        # The verify gate is substrate-enforced: a failed operative
        # verdict — or a missing report — refuses the delivery with
        # VerifyDeclinedError carrying the declined result.  Shipping
        # unverified takes an explicit act: ``verify=False`` (no gate,
        # ``verified=None``) or catching the decline and using
        # ``exc.result``.
        _require_verified_delivery(delivered)
    return delivered


# ---------------------------------------------------------------------------
#  The strategy seam — Optimizer + Monolithic / Compositional / Autotuned
# ---------------------------------------------------------------------------

#: Keyword names each phase verb accepts — ``Monolithic`` partitions
#: ``optimize(..., **kw)`` between them (``verbose`` reaches both).
_SEARCH_KW = frozenset(inspect.signature(search).parameters) - {
    "model",
    "x",
    "source",
}
_LOWER_KW = frozenset(inspect.signature(lower).parameters) - {
    "result",
    "x",
    "sink",
}


class Monolithic:
    """The default strategy: ``lower ∘ search`` in one pass.

    One whole-model search, one delivery — what ``optimize_model``
    always did.  ``optimize(..., **kw)`` keywords partition by phase:
    search knobs (``rules``, ``max_iterations``, ``max_enodes``,
    ``cost_fn``, ``symmetry_budget``, ``specialize_causal``,
    ``fusion_epsilon``, ``delivers_compiled``, ``capabilities``,
    ``criteria``, ``meter``) reach the optimizer's ``.search``; lower
    knobs (``runner``, ``verify``, ``rtol``, ``atol``) reach
    ``.lower``; ``verbose`` reaches both.  Unknown names raise
    ``TypeError``.
    """

    name = "monolithic"

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Run search, then lower — the single-shot pipeline."""
        unknown = sorted(set(kw) - _SEARCH_KW - _LOWER_KW)
        if unknown:
            raise TypeError(
                f"optimize() got unexpected keywords: {unknown}"
            )
        s_kw = {k: v for k, v in kw.items() if k in _SEARCH_KW}
        l_kw = {k: v for k, v in kw.items() if k in _LOWER_KW}
        # A runner chosen at call time also hints the search: the
        # carrier upgrade prices delivered cost under the
        # fusion-region model when the delivery will be compiled.
        runner = l_kw.get("runner")
        if runner is not None:
            s_kw.setdefault(
                "delivers_compiled",
                bool(getattr(runner, "delivers_compiled", False)),
            )
        result = optimizer.search(model, x, **s_kw)
        return optimizer.lower(result, x, **l_kw)


class Compositional:
    """Per-block search+lower, then recompose — composer-driven.

    The strategy view of ``optimize_compositional``: selects blocks
    through the optimizer's :class:`~mlopt_core.ports.Composer`
    port, optimizes each on its captured input, runs the pairwise
    cross-block pass and grafts the results into a parameter-sharing
    clone — every backend-native step delegated to the composer, so
    the strategy itself is backend-neutral.

    ``block_pred`` / ``verify_tol`` / ``max_cross_pairs`` /
    ``cache`` are strategy configuration; the per-block search knobs
    (``rules``, ``max_iterations``, ``cost_fn``, ``max_enodes``,
    ``max_memory_mb``, ``verbose``) ride in ``**kw``.

    ``cache`` controls the structural-signature search cache:
    transformer-style stacks repeat ONE block structure N times with
    different parameters, and the second..Nth searches are redundant.
    ``None``/``True`` (the default) enables a fresh per-run dict;
    ``False`` disables; an explicit dict is used as-is — pass a
    caller-owned mapping to reuse search results across models.
    Replays re-leaf the cached term under the hit block's own names
    and values, re-run the value-dependent folds, and verify
    numerically before grafting; a replay that cannot honour the
    template's assumptions declines to the ordinary per-block search.
    Hits/misses land in ``stats["cache"]``.
    """

    name = "compositional"

    def __init__(
        self,
        *,
        block_pred: Callable | None = None,
        verify_tol: float = 1e-4,
        max_cross_pairs: int = 8,
        cache: dict | bool | None = None,
    ) -> None:
        """Store the strategy configuration."""
        self.block_pred = block_pred
        self.verify_tol = verify_tol
        self.max_cross_pairs = max_cross_pairs
        self.cache = cache

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Run the per-block pipeline through the optimizer's ports."""
        if self.cache is False:
            cache = None
        elif self.cache is None or self.cache is True:
            cache = {}
        else:
            cache = self.cache
        mod, stats = _optimize_compositional(
            model,
            x,
            optimizer=optimizer,
            block_pred=self.block_pred,
            verify_tol=self.verify_tol,
            max_cross_pairs=self.max_cross_pairs,
            cache=cache,
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


class Autotuned:
    """Measured autotune: one search, N verified timed deliveries.

    The strategy view of ``optimize_model_autotuned`` — re-lowers
    the extracted term through each candidate lowering path, verifies
    every candidate against the model, times the survivors through
    the optimizer's :class:`~mlopt_core.ports.Meter` port and ships
    the measured winner.

    ``candidates`` are names resolved against the backend's
    :attr:`~mlopt_core.ports.Sink.executors` table (``"generic"``
    plus every executor name — ``"batched"`` is the routing
    pseudo-name for the pipeline's own delivery) and
    :data:`mlopt_orchestrator.autotune.CANDIDATE_BUILDERS`
    (``"eager"``), or ``(name, builder)`` tuples; backend-provided
    builders arrive through ``builders=`` (the torch wrapper maps
    ``"torch_compile"`` / ``"torch_compile_generic"`` /
    ``"cuda_graph"`` via ``kit_torch.autotune.TORCH_BUILDERS``).  The
    remaining fields are the timing / budget / verify knobs.
    ``optimize`` kwargs (``rules``, ``max_iterations``, ``runner``,
    …) forward to the underlying pipeline call; an explicit
    ``verbose=`` keyword overrides the strategy field.
    """

    name = "autotuned"

    def __init__(
        self,
        candidates: Any = ("generic", "batched", "torch_compile"),
        *,
        budget_s: float | None = None,
        n_calls: int = 30,
        warmup: int = 5,
        rtol: float = 1e-4,
        atol: float | None = None,
        profile: Any = None,
        verbose: bool = False,
        builders: dict[str, Any] | None = None,
    ) -> None:
        """Store the autotune configuration."""
        self.candidates = candidates
        self.budget_s = budget_s
        self.n_calls = n_calls
        self.warmup = warmup
        self.rtol = rtol
        self.atol = atol
        self.profile = profile
        self.verbose = verbose
        #: Backend-provided candidate builders — names the
        #: orchestrator-side ``CANDIDATE_BUILDERS`` map does not know
        #: (e.g. the torch wrapper's ``TORCH_BUILDERS``).  Merged over
        #: the neutral table; ``None`` is backend-agnostic autotuning.
        self.builders = builders

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Search once via the ports, then time each candidate."""
        # Local import: autotune imports this module at top level, so
        # the reverse edge must defer to call time (the same pattern
        # runners.runner_candidate documents).
        from mlopt_orchestrator.autotune import _autotuned_impl

        mod, stats = _autotuned_impl(
            model,
            x,
            candidates=self.candidates,
            builders=self.builders,
            budget_s=self.budget_s,
            n_calls=self.n_calls,
            warmup=self.warmup,
            rtol=self.rtol,
            atol=self.atol,
            optimizer=optimizer,
            profile=self.profile,
            verbose=kw.pop("verbose", self.verbose),
            **kw,
        )
        return LowerResult(module=mod, stats=stats)


@dataclass
class Optimizer:
    """The configured entry point — explicit ports, no assumed backend.

    The pipeline runs through four ports — ``source`` (``model ->
    (IR, leaves)``), ``sink`` (``IR -> runnable`` + verify +
    executors), ``composer`` (per-block structural machinery) and
    ``meter`` (timing) — bundled as an immutable
    :class:`~mlopt_core.pipeline.Backend` or given individually.
    Resolution is ``backend`` first, then the explicit arguments
    override each port it carries; every port is required — there is
    NO default backend (choosing torch means importing
    ``kit_torch.backend.TorchBackend``).  ``criteria`` is the
    default selection blend; ``runner`` the default delivery
    (``IdentityRunner`` — the no-op, backend-neutral shipping step).

    The phase verbs mirror the module-level functions with the
    optimizer's ports wired in; ``optimize`` composes them through a
    :class:`~mlopt_core.ports.Strategy`::

        opt = Optimizer(backend=TorchBackend())
        mod, stats = opt.optimize(model, x)
        res = opt.search(model, x)          # the inspectable mid-state
        fast = opt.lower(res, x, runner=TorchCompileRunner())

    """

    source: Source | None = None
    sink: Sink | None = None
    composer: Composer | None = None
    meter: Meter | None = None
    backend: Backend | None = None
    criteria: (
        dict[str, float] | Criteria | Criterion | list | tuple | None
    ) = None
    runner: Runner = field(default_factory=IdentityRunner)
    #: The optimizer's default saturation rule set — ``None`` resolves
    #: to the composed :data:`DEFAULT_RULES` at search time.
    rules: RuleSet | str | Iterable | None = None
    #: The default saturation engine — ``None`` resolves to
    #: ``"eqsat"`` (engine-eqsat, the WP4-v4 default);
    #: ``"python-egraph"`` selects the pure-Python reference
    #: ``EGraph``.  An instance, class, zero-arg factory, or engine
    #: name conforming to :class:`mlopt_core.ports.Engine` (e.g.
    #: ``mlopt_native.NativeEngine`` / ``EqsatEngine``) selects
    #: explicitly.  Never auto-detected.
    engine: Any = None
    #: The default placement/refusal policy (Phase B) — a
    #: :class:`~mlopt_core.ports.DevicePolicy`-shaped value.  ``None``
    #: resolves from ``backend.device_policy`` when the backend
    #: carries one; ``search``/``lower``/``optimize`` forward it
    #: unless a call overrides it.  ``None`` end-to-end runs the
    #: historical unpoliced path.
    device_policy: Any = None

    def __post_init__(self) -> None:
        """Resolve ports — explicit args override the backend's.

        ``source`` and ``sink`` are REQUIRED — from the backend or
        explicitly.  ``composer``/``meter`` are the optional ports the
        :class:`Compositional` / :class:`Autotuned` strategies need;
        a strategy raises a clear error when its port is missing, so
        a bare ``Optimizer(source=..., sink=...)`` still runs
        :class:`Monolithic` — the historical minimal form.
        """
        if self.backend is not None:
            if self.source is None:
                object.__setattr__(self, "source", self.backend.source)
            if self.sink is None:
                object.__setattr__(self, "sink", self.backend.sink)
            if self.composer is None:
                object.__setattr__(
                    self, "composer", self.backend.composer
                )
            if self.meter is None:
                object.__setattr__(self, "meter", self.backend.meter)
            # Phase B: the backend may carry a placement policy; an
            # explicit ``device_policy=`` always wins (same
            # precedence as the four ports).
            if self.device_policy is None:
                object.__setattr__(
                    self,
                    "device_policy",
                    getattr(self.backend, "device_policy", None),
                )
        missing = [
            n for n in ("source", "sink") if getattr(self, n) is None
        ]
        if missing:
            raise TypeError(
                "Optimizer requires source/sink — pass "
                "backend=Backend(...) or explicit "
                "source=/sink= ports; composer=/meter= are needed "
                f"only by the per-block/measured strategies "
                f"(missing: {', '.join(missing)})"
            )

    def search(self, model: Any, x: Any, **kw: Any) -> SearchResult:
        """Run :func:`search` through this optimizer's ports.

        ``capabilities`` defaults to the sink (a ``Sink`` is a
        ``Capabilities``); ``criteria`` to the configured blend;
        ``delivers_compiled`` to the runner's marker; ``meter`` to the
        configured meter.  Keyword arguments override each default
        outright.
        """
        kw.setdefault("source", self.source)
        kw.setdefault("capabilities", self.sink)
        kw.setdefault("criteria", self.criteria)
        kw.setdefault("rules", self.rules)
        kw.setdefault(
            "delivers_compiled",
            bool(getattr(self.runner, "delivers_compiled", False)),
        )
        kw.setdefault("meter", self.meter)
        kw.setdefault("engine", self.engine)
        kw.setdefault("device_policy", self.device_policy)
        return _search(model, x, **kw)

    def lower(
        self, result: SearchResult, x: Any, **kw: Any
    ) -> LowerResult:
        """Run :func:`lower` through this optimizer's sink.

        ``runner=None`` means the configured default, not the
        identity — pass ``runner=IdentityRunner()`` explicitly to
        override a configured delivery.
        """
        kw.setdefault("sink", self.sink)
        if kw.get("runner") is None:
            kw["runner"] = self.runner
        kw.setdefault("device_policy", self.device_policy)
        return _lower(result, x, **kw)

    def optimize(
        self,
        model: Any,
        x: Any,
        *,
        strategy: Strategy | None = None,
        **kw: Any,
    ) -> LowerResult:
        """Run *strategy* (default :class:`Monolithic`) end to end.

        The result unpacks as ``(module, stats)`` — ``mod, stats =
        opt.optimize(model, x)`` mirrors the historical tuple.
        """
        strat = strategy if strategy is not None else Monolithic()
        return strat.run(model, x, optimizer=self, **kw)

    def discover(self, model: Any, x: Any, **kw: Any) -> SearchResult:
        """Run the discovery view — :func:`search` with frontier defaults.

        ``max_iterations=6`` unless the caller says otherwise (the
        ``rules`` default is the composed :data:`DEFAULT_RULES`);
        the :class:`SearchResult` carries the
        whole inspectable mid-state (``.alternatives()`` /
        ``.certificate()`` / ``.eg`` / ``.stats``).
        """
        kw.setdefault("max_iterations", 6)
        # Frontier inspection is proof-carrying EGraph machinery —
        # an unconfigured optimizer pins the reference engine
        # (mirrors ``discover_alternatives``); an explicit
        # ``Optimizer(engine=...)`` or call keyword still wins.
        if self.engine is None:
            kw.setdefault("engine", "python-egraph")
        return self.search(model, x, **kw)


# ---------------------------------------------------------------------------
#  The discover verb and the compositional driver
# ---------------------------------------------------------------------------


#: The verbs bound to module level — ``Optimizer``'s methods alias the
#: module functions so the method bodies can't be shadowed by the
#: same-named methods.
_search = search
_lower = lower


@_oom_to_resource_error
def discover_alternatives(
    model: Any,
    x: Any,
    *,
    source: Source,
    capabilities: Capabilities | None = None,
    rules: RuleSet | str | Iterable | None = None,
    max_iterations: int = 6,
    cost_fn: CostFn | None = None,
    engine: Any = "python-egraph",
    device_policy: Any = None,
) -> SearchResult:
    """Enumerate the frontier — return the :class:`SearchResult`.

    The discovery-engine view: the same export → e-graph → saturation
    → pairing pipeline as :func:`search` with frontier-oriented
    defaults (``max_iterations=6``; ``rules=None`` resolves to the
    composed :data:`DEFAULT_RULES`) — and
    the result is the inspectable mid-state, not a fixed report:

    * ``res.alternatives(top_k)`` — the top-k cheapest distinct
      members of the root class under the result's pricing
      (``cost_fn`` — or its default — is backend-relative when
      ``capabilities`` is given, so the frontier only lists forms the
      backend can lower);
    * ``res.eg.diverse_classes()`` — the e-classes holding
      structurally distinct equivalent terms, where emergent
      compositions hide;
    * ``res.stats["rule_fires"]`` — the provenance: which generic
      laws actually fired;
    * ``res.term`` / ``res.eg`` / ``res.stats`` — the extraction and
      its record.

    Human inspection of this frontier is how level-3 candidates —
    emergent compositions of known laws — are found.  ``source`` is
    required (no assumed frontend); ``capabilities`` supplies
    backend-relative pricing and the const-fold registry.

    ``rules`` follows :func:`search`'s contract — a ``RuleSet``, a
    preset name, a plain iterable, or ``None`` for the composed
    default.  The pairing-subsumed fusion rules are excluded by the
    presets, never by a filter inside the pipeline.

    ``engine`` defaults to ``"python-egraph"`` — deliberately *not*
    the ``search`` default: the frontier surface this verb exists for
    (``res.alternatives()``, ``res.certificate()``,
    ``res.eg.diverse_classes()``) is proof-carrying ``EGraph``
    machinery the ``Engine`` port does not require.  Passing another
    engine still runs the search, but those methods decline on it.
    """
    return search(
        model,
        x,
        source=source,
        capabilities=capabilities,
        cost_fn=cost_fn,
        rules=rules,
        max_iterations=max_iterations,
        engine=engine,
        device_policy=device_policy,
    )


def ir_to_string(term: Any) -> str:
    """Pretty-print an IR term as an S-expression."""
    return op_repr(term)


def term_cost(term: Any, cost_fn: CostFn | None = None) -> float:
    """Compute the cost of a term using the given cost function."""
    if cost_fn is None:
        cost_fn = flops_cost
    return cost_fn(term)
