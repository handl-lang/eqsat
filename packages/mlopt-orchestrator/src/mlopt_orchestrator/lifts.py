"""Non-local passes — pairing and the value-dependent lifts.

:func:`_pairing_and_lifts` runs the diagram-level product law
(:func:`mlopt_core.laws.pair_shared_input_linears` /
``pair_shared_input_convs``) plus the opt-in non-local lifts —
exact weight tying, the ``__heads`` slice dedup, and the
detect-gated low-rank / weight-specials / head-share offers — each
witnessed so certificates stay replayable, with a brief
re-saturation between passes.  Returns the pairing-groups list the
coordinated extraction in ``_select_best_term`` needs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.laws import (
    pair_shared_input_convs,
    pair_shared_input_linears,
    share_duplicate_attention_heads,
    share_duplicate_param_slices,
    share_duplicate_params,
)

from mlopt_orchestrator.carriers import get_carriers
from mlopt_orchestrator.resources import _check_resources


def _pairing_and_lifts(
    eg: EGraph,
    rules: Iterable,
    root_eid: int,
    stats: dict[str, Any],
    run_cap: int,
    rule_budgets: dict[str, int] | None,
    max_enodes: int | None,
    max_memory_mb: float | None,
    source_tensors: dict,
    meter: Any = None,
    detect_factors: bool = False,
    detect_specials: bool = False,
    detect_headshare: bool = False,
    error_budget: float | None = None,
) -> list:
    """Non-local passes with a brief re-saturation between them.

    The diagram-level product law pairs every linear sharing an input
    into one GEMM + split views (no consumer pattern needed); then the
    non-local lifts — unrolled recurrences -> ``trace(F)``, stacks of
    same-state carrier applications -> one application, whole om trees
    over scanned values -> the deferred omd carrier, exact weight
    tying (duplicate Param leaves share one class), and — opt-in via
    ``detect_factors`` — the low-rank factored-parameter offers of
    :func:`mlopt_core.laws.factored.offer_low_rank_factors`, plus —
    opt-in via ``detect_specials`` — the structurally-special
    weight offers of
    :func:`mlopt_core.laws.specials.offer_weight_specials` (exact —
    or, when ``error_budget`` is set, additionally its bounded
    near-dead / near-duplicate elision members).  All witnessed so
    certificates stay replayable.

    Returns the pairing-groups list — the coordinated (paired)
    extraction in :func:`_select_best_term` needs it.
    """
    groups = pair_shared_input_linears(eg) + pair_shared_input_convs(eg)
    if groups:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb, meter)
        stats["pairing_groups"] = len(groups)
        # brief second saturation so other rules see the new enodes
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb, meter)

    lifts = _carrier_lifts(
        eg,
        source_tensors,
        stats,
        detect_factors,
        detect_specials,
        detect_headshare,
        error_budget,
    )
    if lifts:
        eg.rebuild()
        _check_resources(eg, max_enodes, max_memory_mb, meter)
        stats["nonlocal_lifts"] = len(lifts)
        # Provenance the compositional structural cache replays
        # against: exact-tie clusters (``share_duplicate_params``
        # returns ``list[list[str]]`` of param names) and slice-dedup
        # derivations (``share_duplicate_param_slices`` returns dicts
        # carrying ``dedup_param``).  A replayed term that dropped a
        # tied name or references a derived ``__heads`` stack is only
        # valid for a block whose own values reproduce the same
        # sharing structure — the records below are what a cache hit
        # re-checks / re-derives.
        ties = [
            r
            for r in lifts
            if isinstance(r, list)
            and all(isinstance(n, str) for n in r)
        ]
        derived = {
            r["dedup_param"]: {
                "base": r["param"],
                "heads": r["heads"],
                "imap": tuple(r["index_map"]),
            }
            for r in lifts
            if isinstance(r, dict) and "dedup_param" in r
        }
        # HeadShare sites carry the same replay burden: a cached term
        # whose sdpa gathers rely on head equality is valid only for
        # blocks whose own weights reproduce it — recorded as recheck
        # recipes for ``_cache_replay``'s ``_headshare_holds`` gate.
        headshare = [
            r["recheck"]
            for r in lifts
            if isinstance(r, dict) and "recheck" in r
        ]
        stats["param_sharing"] = {
            "ties": ties,
            "derived": derived,
            "headshare": headshare,
        }
        eg.run(
            rules,
            root_eid,
            max_iterations=5,
            max_nodes=run_cap,
            rule_budgets=rule_budgets,
        )
        _check_resources(eg, max_enodes, max_memory_mb, meter)
    return groups


def _carrier_lifts(
    eg: EGraph,
    source_tensors: dict,
    stats: dict[str, Any],
    detect_factors: bool,
    detect_specials: bool = False,
    detect_headshare: bool = False,
    error_budget: float | None = None,
) -> list:
    """Run the non-local carrier/tying lifts.

    The carrier lifts arrive through
    :mod:`mlopt_orchestrator.carriers`; the weight-tying lifts are
    core.  A process without carriers contributes only the tying
    passes.  ``detect_factors``
    arms the opt-in low-rank detection offers of
    :func:`mlopt_core.laws.factored.offer_low_rank_factors`;
    ``detect_specials`` arms the opt-in weight-structure offers of
    :func:`mlopt_core.laws.specials.offer_weight_specials` — exact
    by default, additionally bounded when ``error_budget`` is set;
    ``detect_headshare`` arms the opt-in bitwise-equal-head compute
    sharing of
    :func:`mlopt_core.laws.headshare.share_duplicate_attention_heads`.
    """
    m = get_carriers()
    carrier: list = [] if m is None else m.lifts(eg)
    return (
        carrier
        + share_duplicate_params(eg, source_tensors)
        + share_duplicate_param_slices(eg, source_tensors)
        + _headshare_lifts(eg, source_tensors, detect_headshare)
        + _factor_lifts(eg, source_tensors, stats, detect_factors)
        + _special_lifts(
            eg, source_tensors, stats, detect_specials, error_budget
        )
    )


def _headshare_lifts(
    eg: EGraph, source_tensors: dict, detect_headshare: bool
) -> list:
    """Opt-in bitwise-equal-head sharing offers (``detect_headshare``).

    The detection pass itself is the branch — when off this returns
    ``[]`` and the graph never sees the gather-``sdpa``-gather member.
    When on, each ``sdpa`` site whose (Wq, Wk, Wv) head blocks are
    bitwise-equal gets a witnessed deduplicated alternative; the offer
    is exact by construction and competes on cost like any other.
    """
    if not detect_headshare:
        return []
    return share_duplicate_attention_heads(eg, source_tensors)


def _factor_lifts(
    eg: EGraph,
    source_tensors: dict,
    stats: dict[str, Any],
    detect_factors: bool,
) -> list:
    """Opt-in low-rank factored-parameter offers (``detect_factors``).

    The detection pass itself is the branch — when off this returns
    ``[]`` without touching the graph; when on, each certified
    offer lands in ``stats["low_rank_factors"]`` (minus the e-class
    id, which means nothing outside this run).
    """
    if not detect_factors:
        return []
    from mlopt_core.laws.factored import offer_low_rank_factors

    offers = offer_low_rank_factors(eg, source_tensors)
    if offers:
        stats["low_rank_factors"] = [
            {k: v for k, v in r.items() if k != "eid"} for r in offers
        ]
    return offers


def _special_lifts(
    eg: EGraph,
    source_tensors: dict,
    stats: dict[str, Any],
    detect_specials: bool,
    error_budget: float | None = None,
) -> list:
    """Opt-in structurally-special weight offers (``detect_specials``).

    The detection pass itself is the branch — when off this returns
    ``[]`` without touching the graph; when on, each certified offer
    (identity/diagonal/zero/elide/block-diag members, all
    ``error_bound=0``) lands in ``stats["weight_specials"]`` (minus
    the e-class id, which means nothing outside this run).  With
    ``error_budget`` set, the pass additionally offers the bounded
    ``elide_bounded`` / ``zero_bounded`` members — certified
    approximate, each carrying its measured ``error_bound ≤
    error_budget``; the selection gate downstream enforces the
    accumulated bound.
    """
    if not detect_specials:
        return []
    from mlopt_core.laws.specials import offer_weight_specials

    offers = offer_weight_specials(
        eg, source_tensors, budget=error_budget
    )
    if offers:
        stats["weight_specials"] = [
            {k: v for k, v in r.items() if k != "eid"} for r in offers
        ]
    return offers
