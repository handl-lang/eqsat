"""Certified bounded-error offers — the ``error_budget`` machinery.

Extraction-side bound ledger (:func:`_bound_gate` /
:func:`_bounded_term`'s ban loop over :func:`eg.extract_best`),
the weight-space -> output-space propagation
(:func:`_propagate_bounds` — ``bound . site-input row norm`` per
ledger entry, evaluated on the verify input) and the honest verify
gates (:func:`_bound_verify` / :func:`_block_bound_gate` /
:func:`_bounded_gate`) ``lower`` and the compositional block gate
consume.  ``None`` budget is the exact path — every entry point
here is a pass-through then.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Op, Param
from mlopt_core.ports import CostFn, Sink

from mlopt_orchestrator.resources import OptimizationResourceError

# ---------------------------------------------------------------------------
#  Certified bounded-error offers (plan 0012) — the budget plumbing
# ---------------------------------------------------------------------------


def _check_budget_engine(eg: Any, error_budget: float | None) -> None:
    """Bounded extraction needs proof witnesses — the reference EGraph.

    No-op when ``error_budget`` is ``None``.  A non-``EGraph`` engine
    lacks ``certificate`` / ``extract_best_bounded`` (the
    :class:`~mlopt_core.ports.Engine` port does not require them),
    and a level-1 e-graph records no merge witnesses — in either case
    the bound gate would silently pass every bounded member.  Fail
    loudly instead.
    """
    if error_budget is None:
        return
    if not isinstance(eg, EGraph):
        raise TypeError(
            "error_budget requires the reference EGraph engine — "
            "bounded offers and their certificates are Python-engine "
            f"machinery (got {type(eg).__name__})"
        )
    if eg.truncation_level < 2:
        raise TypeError(
            "error_budget requires proof witnesses — the engine was "
            "built at truncation_level < 2"
        )


def _subterms(term: Any) -> Iterable:
    """Yield *term* and every subterm, pre-order."""
    stack = [term]
    while stack:
        t = stack.pop()
        yield t
        if isinstance(t, Op):
            stack.extend(t.args)


def _term_params(term: Any) -> set[str]:
    """Names of every ``Param`` leaf *term* carries."""
    return {t.name for t in _subterms(term) if isinstance(t, Param)}


def _bound_rules(eg: EGraph) -> dict[str, Any]:
    """Return the bound-carrying rewrites this run registered."""
    return {n: r for n, r in eg._rule_objs.items() if r.error_bound}


def _delivered_bound(
    eg: EGraph, term: Any, bound: dict[str, Any]
) -> set[str]:
    """Bound-carrying rule names whose offered member *term* delivers.

    A bound offer survives extraction under two fingerprints, both
    robust to saturation rewriting the offered member after it was
    registered:

    * *verbatim delivery* — the member's root enode is registered to
      the bound witness's synthetic application (``_enode_app``), so
      a term locating to it used the offer directly;
    * *derived params* — bounded members introduce ``Param`` leaves
      the exact side lacks (``__bl`` / ``__lr`` derived weights);
      leaves survive rewriting, so a term still carrying one is a
      rewritten delivery of the member.
    """
    used_params = _term_params(term)
    delivered = {
        nm
        for nm, r in bound.items()
        if (_term_params(r.rhs) - _term_params(r.lhs)) & used_params
    }
    # Enodes registered to bound-carrying applications: a verbatim
    # delivered member locates to one of these.
    tainted: dict[Any, str] = {
        en: eg._applications[ai]["rule"]
        for en, ai in eg._enode_app.items()
        if eg._applications[ai]["rule"] in bound
    }
    for sub in _subterms(term):
        _ce, en = eg._locate(sub)
        if rn := tainted.get(en):
            delivered.add(rn)
    return delivered


def _bound_ledger(
    eg: EGraph, cert: Any, term: Any
) -> tuple[list[dict], float]:
    """Compute the bound ledger of a delivered term — entries + total.

    The certificate is the primary source: each replayed bound step
    contributes its rule's declared ``error_bound``.  But a derivation
    that exhausts its budget records ``egraph_dependent`` stubs and
    any bound merge inside such a gap contributes *nothing* to
    ``cert.error_bound`` — a silent undercount.  :func:`_delivered_bound`
    therefore re-scans the term itself for bound-member fingerprints;
    a detected-but-uncertified rule is appended and its bound added,
    so the ledger is never silently empty on an approximate term.
    """
    entries = _error_bound_entries(cert)
    bound = _bound_rules(eg)
    if not bound:
        return entries, cert.error_bound
    covered = {e["rule"] for e in entries}
    extra = sorted(_delivered_bound(eg, term, bound) - covered)
    for nm in extra:
        r = bound[nm]
        entries.append(
            {
                "rule": nm,
                "law": r.law,
                "bound": r.error_bound,
                "norm": r.bound_norm,
                "measured_max_rel": None,
            }
        )
    return entries, cert.error_bound + sum(
        bound[nm].error_bound for nm in extra
    )


def _ban_bound_members(
    eg: EGraph, term: Any, bound: dict[str, Any], bans: dict[int, set]
) -> bool:
    """Exclude every bound member *term* delivers; True on new bans.

    For each delivered bound rule this bans (a) the offered member's
    root enode, (b) its derived ``Param`` leaf enodes — rewritten
    spellings keep the leaf, so banning it kills every variant — and
    (c) the enode the term actually picked inside the offered class,
    which covers bound members carrying no derived param at all
    (``zero_bounded``).  Each new ban strictly shrinks the candidate
    space, so a caller looping on this either fits the budget or
    stalls — reported by the ``False`` return.
    """
    delivered = _delivered_bound(eg, term, bound)
    if not delivered:
        return False
    want: set = set()
    derived: list = []
    for nm in delivered:
        r = bound[nm]
        ce, _en = eg._locate(r.rhs)
        want.add(ce)
        keep = _term_params(r.lhs)
        derived += [
            sub
            for sub in _subterms(r.rhs)
            if isinstance(sub, Param) and sub.name not in keep
        ]
    progress = False

    def _ban(ce: Any, en: Any) -> None:
        nonlocal progress
        slot = bans.setdefault(ce, set())
        if en not in slot:
            slot.add(en)
            progress = True

    for sub in _subterms(term):
        ce, en = eg._locate(sub)
        if ce in want:
            _ban(ce, en)
    for p in derived:
        _ban(*eg._locate(p))
    return progress


def _bounded_term(
    eg: EGraph,
    root_eid: int,
    src: Any,
    cost_fn: CostFn,
    error_budget: float,
) -> tuple[Any, list[dict], float]:
    """Cheapest member whose delivered bound fits ``error_budget``.

    Iterates ``extract_best`` under accumulating enode bans: each
    round's ledger — the certificate plus the delivered-member
    fingerprints — either fits the budget (done) or names the bound
    members to exclude for the next round.  The original member is
    never bound-carried, so a real search root always satisfies the
    budget; ``None``/stall are reachable only for a degenerate
    fully-cyclic class or a bound member no fingerprint can name.
    """
    bans: dict[int, set] = {}
    bound = _bound_rules(eg)
    while True:
        term = eg.extract_best(root_eid, cost_fn, bans=bans)
        if term is None:
            raise OptimizationResourceError(
                f"error_budget={error_budget}: no member of the root "
                "e-class certifies within the budget"
            )
        cert = eg.certificate(src, term, root_eid=root_eid)
        entries, total = _bound_ledger(eg, cert, term)
        if total <= error_budget:
            return term, entries, total
        if not _ban_bound_members(eg, term, bound, bans):
            raise OptimizationResourceError(
                f"error_budget={error_budget}: the cheapest member "
                "relies on a bound offer that cannot be excluded"
            )


def _bound_gate(
    eg: EGraph,
    root_eid: int,
    src: Any,
    cost_fn: CostFn,
    stats: dict[str, Any],
    best_term: Any,
    error_budget: float | None,
) -> Any:
    """Apply the error-budget ledger gate to the extracted term.

    Pass-through when ``error_budget`` is ``None``.  Otherwise ledger
    the winner — the certificate plus the delivered-member
    fingerprints (a derivation that exhausted its budget undercounts)
    — and re-extract through :func:`_bounded_term`'s ban loop when
    the total exceeds the budget.  ``stats`` always records the
    request and the accepted bound — never silent.
    """
    if error_budget is None:
        return best_term
    cert = eg.certificate(src, best_term, root_eid=root_eid)
    entries, bound_total = _bound_ledger(eg, cert, best_term)
    if bound_total > error_budget:
        best_term, entries, bound_total = _bounded_term(
            eg, root_eid, src, cost_fn, error_budget
        )
    stats["error_budget"] = error_budget
    stats["error_bound_total"] = bound_total
    stats["error_bounds"] = entries
    return best_term


def _error_bound_entries(cert: Any) -> list[dict]:
    """One record per bound-carrying step of the accepted derivation.

    ``measured_max_rel`` stays ``None`` until ``lower``'s verify fills
    it with the delivered-vs-original measurement — the ledger is
    surfaced either way, never silent.
    """
    return [
        {
            "rule": s.rule,
            "law": r.law,
            "bound": r.error_bound,
            "norm": r.bound_norm,
            "measured_max_rel": None,
        }
        for s in cert.steps
        if (r := cert.rules.get(s.rule)) is not None and r.error_bound
    ]


# ---------------------------------------------------------------------------
#  Bound propagation — bridging the weight-space / output-space units
# ---------------------------------------------------------------------------
#
# The ledger's ``bound`` is certified in *weight* space (``max_abs`` =
# ``max|ΔW|`` for the specials' bounded members, ``frobenius`` for the
# low-rank residual).  The verify report measures *output* space —
# ``max|Δy|`` — and a site ``y = x·W`` amplifies the weight bound by
# the input's contraction norm.  Comparing the two directly is a unit
# bug (the bounded_e2e sweep measured ~2-8x output amplification on a
# real head, declining every delivery).  ``_propagate_bounds`` bridges
# the units honestly: per output element
# ``|Δy_j| = |Σ_k x_k·ΔW_jk| ≤ bound · max_i‖x_i‖_p``, so each entry's
# output bound is ``bound · amp`` with ``amp`` the *site* input's max
# row norm — evaluated, not guessed (the witness LHS is the original
# site expression; its data subterm is lowered through the sink and
# run on the verify input).


def _bound_p(bound_norm: str) -> float | None:
    """Row-norm exponent a bound norm propagates through, or ``None``.

    A per-element (``max_abs``) bound gives ``|Δy_j| ≤
    Σ_k|x_k|·max|ΔW|`` — the row-L1 factor.  Frobenius and spectral
    bounds give ``|Δy_j| ≤ ‖x_row‖_2·‖ΔW‖`` — the row-L2 factor.  Any
    other norm has no stated propagation rule here: ``None`` makes the
    gate fail closed rather than guess a bridge.
    """
    return {"max_abs": 1.0, "frobenius": 2.0, "spectral": 2.0}.get(
        bound_norm
    )


def _site_data_term(rule: Any) -> Any | None:
    """Return the data-side subterm of a bound witness's LHS.

    Bounded offers witness ``site -> offered`` where ``site`` is the
    original ``linear``/``matmul`` expression — its first argument is
    the site input.  A non-projection LHS (e.g. a morphism-level joint
    term) carries no isolable site.
    """
    lhs = getattr(rule, "lhs", None)
    if (
        isinstance(lhs, Op)
        and lhs.op in ("linear", "matmul")
        and lhs.args
    ):
        return lhs.args[0]
    return None


def _site_eval(
    sink: Sink, ir: IR, params: dict, term: Any, args: tuple
) -> Any:
    """Evaluate one subterm: lower it through the sink, run on ``args``.

    The subterm's ``Var`` leaves bind positionally to the module's own
    inputs, so running it on the verify input yields the real site
    activation — the operand the bounded weight actually contracts.
    """
    sub = IR(
        root=term,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=dict(ir.params),
    )
    mod = sink.lower(sub, params)
    return mod.forward(*args)


def _row_norm(t: Any, p: float) -> float | None:
    """``max_i ‖t_i‖_p`` over the last axis — the contraction factor.

    Duck-typed tensor ops (``abs`` / ``sum`` / ``max``), the same
    convention as the pairing/factored passes — the orchestrator names
    no tensor library.  ``None`` when the value cannot supply them.
    """
    abs_ = getattr(t, "abs", None)
    if not callable(abs_):
        return None
    try:
        a = abs_()
        v = a.sum(-1) if p == 1.0 else (a * a).sum(-1) ** 0.5
        return float(v.max())
    except Exception:
        return None


def _entry_amp(
    rule: Any,
    p: float,
    sink: Sink,
    ir: IR,
    params: dict,
    args: tuple,
) -> tuple[float | None, str]:
    """One ledger entry's site-input amplification, with provenance.

    Primary: evaluate the bound rule's witness-LHS data subterm on the
    verify args — the real site input — and take its row norm.
    Fallback: the module input's own row norm when the site input
    cannot be isolated or evaluated (``"block_input"`` — exact for a
    block that IS its projection site, a documented estimate deeper
    inside one).  ``(None, "none")`` when neither measures.
    """
    site = _site_data_term(rule) if rule is not None else None
    if site is not None:
        with contextlib.suppress(Exception):
            amp = _row_norm(_site_eval(sink, ir, params, site, args), p)
            if amp is not None:
                return amp, "site"
    amps = [n for n in (_row_norm(a, p) for a in args) if n is not None]
    return (max(amps), "block_input") if amps else (None, "none")


def _propagate_bounds(
    stats: dict[str, Any],
    eg: Any,
    ir: IR,
    params: dict,
    sink: Sink,
    args: tuple,
) -> float | None:
    """Propagate the ledger's weight-space bounds into output units.

    Each ``stats["error_bounds"]`` entry certifies ``‖ΔW‖ ≤ bound``
    in its ``norm``; at a site ``y = x_s·W`` the delivered output
    moves by at most ``bound · max_i‖(x_s)_i‖_p`` (``p`` from
    :func:`_bound_p`) — so the total is in the same ``max|Δy|`` units
    the verify report measures, not the weight-space units of the
    certificate.  ``_entry_amp`` prices each entry's site input;
    every entry records ``site_input_norm`` / ``site_input``
    (``"site"`` / ``"block_input"`` / ``"none"``) and the propagated
    ``output_bound`` — the ledger stays honest about which input
    priced it.

    Returns the summed output bound (absolute ``max|Δy|`` units),
    stored in ``stats["error_bound_output"]``, or ``None`` when some
    entry's norm has no propagation rule or no input norm could be
    measured — the gate then fails closed.
    """
    entries = stats.get("error_bounds")
    if not entries:
        return 0.0
    rules = _bound_rules(eg) if isinstance(eg, EGraph) else {}
    out: list[dict] = []
    total = 0.0
    unpropagated = False
    for e in entries:
        e2 = dict(e)
        amp, src = (
            _entry_amp(rules.get(e["rule"]), p, sink, ir, params, args)
            if (p := _bound_p(str(e.get("norm") or ""))) is not None
            else (None, "none")
        )
        e2["site_input_norm"] = amp
        e2["site_input"] = src
        e2["output_bound"] = (
            float(e["bound"]) * amp if amp is not None else None
        )
        if e2["output_bound"] is None:
            unpropagated = True
        else:
            total += e2["output_bound"]
        out.append(e2)
    stats["error_bounds"] = out
    stats["error_bound_output"] = None if unpropagated else total
    return stats["error_bound_output"]


def _bound_record(
    stats: dict[str, Any], report: Any, output_bound: float | None
) -> bool:
    """Fill the ledger's measured/propagated fields; return honored.

    The verify report measures ``max|Δy|`` — so the honored check is
    ``report.max_abs ≤ output_bound``, output bound against output
    measurement in the same units.  ``output_bound_rel`` restates each
    entry's bound in the report's relative units for the record
    (``max_rel = max_abs / (max|ref| + floor)``, so the denominator is
    ``max_abs / max_rel``; ``None`` when the measurement was exact).
    """
    denom = (
        report.max_abs / report.max_rel if report.max_rel > 0 else None
    )
    stats["error_bounds"] = [
        {
            **e,
            "measured_max_rel": report.max_rel,
            "output_bound_rel": (
                e["output_bound"] / denom
                if e.get("output_bound") is not None
                and denom is not None
                else None
            ),
        }
        for e in stats.get("error_bounds", [])
    ]
    honored = bool(
        output_bound is not None and report.max_abs <= output_bound
    )
    stats["error_bounds_honored"] = honored
    return honored


def _rel_bound(output_bound: float | None, report: Any) -> float:
    """Restate an absolute output bound in the report's rel units.

    For messages and records only — the gate itself compares in
    absolute units.  ``nan`` when the bound could not be propagated
    (fail-closed), ``inf`` when the measurement was exactly zero.
    """
    if output_bound is None:
        return float("nan")
    if report.max_rel <= 0 or report.max_abs <= 0:
        return float("inf")
    return output_bound * report.max_rel / report.max_abs


def _bounded_gate(report: Any, bound: float | None) -> bool:
    """Return the verify predicate under an accepted output bound.

    ``bound`` is the propagated OUTPUT bound (absolute ``max|Δy|``
    units, from :func:`_propagate_bounds`) — never the weight-space
    certificate, which is in different units and must not gate the
    measured error directly.  ``report.passed`` is the sink's
    tolerance gate (run with ``rtol=inf`` on bounded deliveries — the
    bound replaces the relative tolerance); the honest bound gate
    additionally requires the measured absolute error to stay within
    the propagated bound.  ``bound=None`` means the bound could not be
    propagated — the gate fails closed; ``bound=0`` means no bound
    members were delivered — the gate is the sink's verdict alone.
    """
    return bool(report.passed) and (
        bound == 0.0 or (bound is not None and report.max_abs <= bound)
    )


def _block_bound_gate(
    sink: Sink,
    block: Any,
    mod: Any,
    inputs: Any,
    verify_tol: float,
    st: dict[str, Any],
    res: Any,
    atol: float | None = None,
) -> tuple[Any, float | None]:
    """Verify one delivered module, propagating any bound first.

    Returns ``(report, output_bound)`` for :func:`_bounded_gate`.
    With no delivered bound members the plain ``verify_tol`` verify
    runs and ``output_bound`` is 0.  With them, the ledger's
    weight-space bound propagates to output units on the block's
    captured inputs (:func:`_propagate_bounds`), the sink verify runs
    at ``rtol=inf`` (the propagated bound replaces the relative
    tolerance), and the ledger gains its measured/propagated fields
    via :func:`_bound_record` — ``output_bound`` is ``None`` when the
    bound could not be propagated, which fails the gate closed.
    ``inputs`` is a tensor or positional-args tuple.
    """
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    bound = float(st.get("error_bound_total", 0.0))
    if bound <= 0.0:
        return (
            sink.verify(block, mod, args, rtol=verify_tol, atol=atol),
            0.0,
        )
    out_b = _propagate_bounds(
        st, res.eg, res.ir, dict(res.param_values), sink, args
    )
    report = sink.verify(block, mod, args, rtol=float("inf"), atol=atol)
    _bound_record(st, report, out_b)
    return report, out_b


def _bound_verify(
    stats: dict[str, Any],
    report: Any,
    bound_total: float,
    output_bound: float | None,
) -> Any:
    """Apply the honest bound gate to a verify report.

    No-op when ``bound_total`` is 0 (the accepted term delivers no
    bound members).  Otherwise each ``stats["error_bounds"]`` entry
    records the measured value and its propagated bound — the ledger
    is never silent — and ``stats["error_bounds_honored"]`` flags the
    verdict: a measured output error exceeding the propagated bound,
    a bound that could not be propagated (``output_bound=None``), or
    a failing sink report substitutes a ``passed=False`` report, even
    when the (unbounded) tolerance alone passed.
    """
    if bound_total <= 0.0:
        return report
    honored = _bound_record(stats, report, output_bound)
    if not honored or not report.passed:
        return _BoundedVerify(
            max_abs=report.max_abs, max_rel=report.max_rel
        )
    return report


@dataclass(frozen=True)
class _BoundedVerify:
    """A ``VerifyResult``-shaped report forced to ``passed=False``.

    ``lower`` substitutes this when a delivered module's measured
    error exceeds the certified bound its accepted members claimed —
    the bound gate is stricter than the verify tolerance.
    """

    max_abs: float
    max_rel: float
    passed: bool = False
