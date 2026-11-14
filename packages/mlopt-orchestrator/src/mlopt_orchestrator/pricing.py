"""Delivered-price extraction models and executor routing.

The selection model that bills each term under the executor it
would actually get: ``_default_cost_fn`` (the generic-eval default),
the measured-feedback-corrected :func:`delivered_cost_for` /
:func:`_delivered_cost` pair, the whole-spine
:func:`_carrier_upgrade` re-examination additive extraction cannot
price, and the ``_lower_extracted`` / ``_route_spec`` executor
table lookup ``lower`` delivers through.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from mlopt_core.cost import _CostMarkers, executor_cost_for
from mlopt_core.ir import Op
from mlopt_core.ports import CostFn, ExecutorSpec, Sink
from mlopt_core.profile import corrected_price_ns, shape_bucket

from mlopt_orchestrator.carriers import get_carriers


def _default_cost_fn() -> CostFn:
    """Return the default extraction model.

    Roofline pricing plus the executor overhead of the lowering this
    pipeline delivers — ``"generic"`` per-node eval or
    Roofline + the additive generic-dispatch overhead (solver ops
    surcharged — the fidelity sweep caught a ``trace`` resolvent priced
    ~free hiding a 14.5 s ``linalg.solve``).  The level-batched carrier
    executors are *not* priced into selection: their cost is a
    whole-spine property, which ``extract_best``'s additive local-cost
    decomposition can't express — instead the lowering routes apply
    roots to them after extraction (``stats["lowering"]``), so a
    selected carrier member gets its fast executor for free.
    """
    return executor_cost_for(lowering="generic")


def _carrier_plans() -> dict[str, Callable]:
    """Carrier-apply root ops → (batched plan builder).

    A term rooted at one of these lowers through the level-batched
    executor.  The builders arrive through
    :mod:`mlopt_orchestrator.carriers` (the carrier package registers
    them), so the orchestrator never imports a backend; a process
    without carriers simply yields an empty map (no carrier upgrades).
    """
    m = get_carriers()
    return {} if m is None else m.plans()


#: Measured-feedback candidate names each delivered route consults,
#: in preference order.  The ``record_measured`` write-back contract
#: (the ``Autotuned`` strategy's ``profile=`` channel) records under
#: candidate names: ``"generic"`` is the serial IRModule evaluator,
#: ``"batched"`` the pipeline's own routed carrier delivery, and the
#: compiled deliveries record under their builder names.  The
#: ``executor_cost_for`` *lowering* names are accepted aliases too,
#: so a table keyed by lowering (e.g. a hand-built or
#: ``bench/drivers/probes/executor_cost_probe.py --emit-profile`` profile) applies.
_DELIVERED_CANDIDATES: dict[str, tuple[str, ...]] = {
    "generic": ("generic",),
    "batched_scan": ("batched", "batched_scan"),
    "compiled": ("torch_compile", "cuda_graph", "compiled"),
}


def _corrected_delivered(
    profile: Any,
    lowering: str,
    bucket: str | None,
    model_ns: float,
) -> float:
    """Apply the profile's measured correction to a delivered price.

    The measured-feedback consumption contract, one hop:
    :func:`~mlopt_core.profile.corrected_price_ns` — the learned
    ``corrections`` factor first, else a ``measured_ns`` residual or
    median substitution — consulted under each candidate name
    :data:`_DELIVERED_CANDIDATES` lists for *lowering*, first hit
    wins.  No profile, no bucket, or no recorded correction returns
    *model_ns* unchanged — the pure-model price is the default.
    """
    if profile is None or bucket is None:
        return model_ns
    for cand in _DELIVERED_CANDIDATES[lowering]:
        priced = corrected_price_ns(profile, cand, bucket, model_ns)
        if priced is not None and priced != model_ns:
            return priced
    return model_ns


def _delivered_cost(
    term: Any,
    profile: Any = None,
    compiled: bool = False,
    bucket: str | None = None,
    memo: dict | None = None,
) -> float:
    """Price a term under the lowering it would actually get.

    The batched carrier executor for plannable apply roots, generic
    eval otherwise (solver ops surcharged).  With ``compiled=True``
    the delivered module is torch.compile-wrapped whichever route
    ran, so the fusion-region model prices every term.

    ``profile`` calibrates the executor model's constants — and when
    it also carries measured-feedback data (``measured_ns`` residuals
    or learned ``corrections`` factors keyed by ``(candidate,
    bucket)`` — the contract :func:`~mlopt_core.profile.record_measured`
    writes), the price is corrected by
    :func:`_corrected_delivered`.  ``bucket`` is the
    :func:`~mlopt_core.profile.shape_bucket` key the corrections
    were recorded under; ``None`` keeps the pure-model price.
    ``memo`` is the shared extraction memo
    :func:`delivered_cost_for` threads through — callers pricing
    many terms over a shared-subterm DAG pass one dict so the
    per-lowering prices memoize across calls.
    """
    lowering = "generic"
    if compiled:
        lowering = "compiled"
    elif isinstance(term, Op):
        plan = _carrier_plans().get(term.op)
        if plan is not None and plan(term) is not None:
            lowering = "batched_scan"
    model_ns = executor_cost_for(profile, lowering=lowering)(term, memo)
    return _corrected_delivered(profile, lowering, bucket, model_ns)


def delivered_cost_for(
    profile: Any = None,
    *,
    x: Any = None,
    bucket: str | None = None,
    compiled: bool = False,
) -> CostFn:
    """Bill each term at its DELIVERED price — extraction cost fn.

    The selection model the executor-pricing fix needs as a public
    path: :func:`_delivered_cost` lifted into a ``cost_fn`` —
    carrier-apply-rooted, plannable terms bill under
    ``lowering="batched_scan"``, everything else under ``"generic"``
    (``compiled=True`` bills every term under the fusion-region
    model).  Unlike the additive
    :func:`~mlopt_core.cost.executor_cost_for` default, extraction
    sees the batched-vs-generic executor delta, so a term that is
    cheaper *delivered* can win greedy extraction directly — the
    mechanism ``bench/drivers/probes/executor_cost_probe.py`` demonstrates with its
    tools-level ``_delivered_cost_fn``.

    ``profile`` — a measured
    :class:`~mlopt_core.profile.TargetProfile` (produce one with
    ``scripts/calibrate_profile.py``) — supplies both the executor
    constants and the measured-feedback corrections
    :func:`_delivered_cost` consumes through
    :func:`~mlopt_core.profile.corrected_price_ns`: a learned
    ``corrections`` factor or a ``measured_ns`` residual recorded
    under (route candidate, ``bucket``).  ``bucket`` is the
    :func:`~mlopt_core.profile.shape_bucket` key the corrections
    were recorded under — pass ``x=`` the search's example input to
    derive it (an explicit ``bucket`` wins).  ``bucket=None`` keeps
    the pure-model delivered prices — a delivered-aware but
    *uncalibrated* comparison (the probe's ``cf-modeled`` arm).
    The closure also carries the ``profile`` marker
    :func:`~mlopt_core.cost.backend_cost` forwards into
    ``_select_best_term``, so the same table corrects
    ``_carrier_upgrade``'s comparison — under this input's OWN
    ``shape_bucket`` — selection and upgrade priced off one
    calibration.

    Non-additive at carrier roots (the batched-scan price is a
    whole-spine property): as an ``extract_best`` model it is
    approximate — the documented caveat it shares with
    :func:`~mlopt_core.cost.fused_cost_for` and
    :func:`~mlopt_core.cost.lowering_aware_cost_for`.  Carrier
    routing resolves at call time: built before the carrier package
    registers, every term bills generic.
    """
    if bucket is None and x is not None:
        bucket = shape_bucket(x)

    def cost(term: Any, memo: dict | None = None) -> float:
        return _delivered_cost(term, profile, compiled, bucket, memo)

    cost.__name__ = "delivered_cost_for"
    cast(_CostMarkers, cost).profile = profile
    return cost


def _carrier_upgrade(
    eg: Any,
    root_eid: int,
    best_term: Any,
    cost_fn: CostFn,
    profile: Any = None,
    compiled: bool = False,
    bucket: str | None = None,
) -> Any:
    """Coordinated carrier selection.

    ``extract_best``'s additive local-cost decomposition can't price
    the batched-scan win (a whole-spine property: log(T) batched
    levels vs T serial leaves), so apply-rooted members lose to terms
    that merely count cheaper — even though they run ~6x faster when
    lowered.  Re-examine the root eclass: force-extract each
    carrier-apply enode and compare *delivered* prices — each term
    billed under the executor it would route to.  Swap only when the
    carrier member is cheaper AND its batched plan exists (else the
    module degrades to serial eval and the price lied).

    ``profile`` / ``bucket`` ride into :func:`_delivered_cost`: a
    measured-calibrated profile corrects the comparison (a batched
    delivery measured slower than generic stops winning it); no
    profile keeps the built-in constants.

    An engine without a materialised ``_classes`` view (the
    :class:`~mlopt_core.ports.Engine` port does not require one)
    skips the pass — greedy extraction already ran.
    """
    eclasses = getattr(eg, "_classes", None)
    if eclasses is None:
        return best_term
    plans = _carrier_plans()
    cid = eg.find(root_eid)
    carriers = [n for n in eclasses[cid].nodes if n.op in plans]
    if not carriers:
        return best_term
    best_price = _delivered_cost(best_term, profile, compiled, bucket)
    for node in carriers:
        cand = eg.extract_best(root_eid, cost_fn, overrides={cid: node})
        if cand is None:
            continue
        price = _delivered_cost(cand, profile, compiled, bucket)
        if price < best_price:
            best_term, best_price = cand, price
    return best_term


def _lower_extracted(
    best_term: Any,
    optimized_ir: Any,
    source_tensors: dict | None,
    sink: Sink,
) -> Any:
    """Route the extracted term through the sink's executor table.

    Executor routing: term-level cost is blind to the lowering — a
    carrier-apply term evaluated by the generic evaluator runs its
    leaves one-by-one (~6-30x slower than the level-batched schedule
    it was priced for).  Route the term to the first *carrier*
    :class:`~mlopt_core.ports.ExecutorSpec` of
    :attr:`~mlopt_core.ports.Sink.executors` whose ``accepts`` probe
    holds (mapping order is routing order); carrier-agnostic sinks
    and non-carrier terms fall through to ``sink.lower`` — the
    backend-neutral default.  A declined carrier plan degrades to the
    generic path inside each builder rather than failing.
    """
    spec = _route_spec(best_term, sink)
    if spec is not None:
        return spec.lower(optimized_ir, source_tensors)
    return sink.lower(optimized_ir, source_tensors)


def _route_spec(best_term: Any, sink: Sink) -> ExecutorSpec | None:
    """First carrier executor spec accepting *best_term*, else None."""
    table = getattr(sink, "executors", None) or {}
    return next(
        (
            s
            for s in table.values()
            if s.carrier is not None and s.accepts(best_term)
        ),
        None,
    )


def _upgrade_pricing(cost_fn: CostFn, x: Any) -> tuple[Any, str | None]:
    """(profile, bucket) the carrier upgrade's delivered pricing needs.

    The ``profile`` marker the selection model carries
    (``backend_cost`` forwards it from ``executor_cost_for(profile,
    ...)`` and criteria blends), and — only when a profile and an
    example input are both present — the
    :func:`~mlopt_core.profile.shape_bucket` key the profile's
    measured corrections were recorded under.  ``(profile, None)``
    keeps the pure-model delivered prices; ``(None, None)`` the
    built-in constants.
    """
    profile = getattr(cost_fn, "profile", None)
    if profile is None or x is None:
        return profile, None
    return profile, shape_bucket(x)
