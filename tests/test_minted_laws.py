"""The shipped minted-composite class — ``conv_bn_fold``, verbatim.

``mlopt_core.laws.minted`` embeds the stored-closure records the
conv+BN eval-mode fold minted as (the ``syn_*`` chain of
``bench/drivers/probes/conv_fold_premise_pay_probe.py``) and rebuilds the
transported guard at import.  These tests pin that the *shipped*
object is the measured one — same guarded region, same fired instance,
same extraction delta — and that the materializer refuses to degrade
into a weaker admit.
"""

from __future__ import annotations

from typing import Any

import pytest
from mlopt_core import meta
from mlopt_core.cost import count_cost, dag_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var, op_repr
from mlopt_core.laws import (
    ALL_RULES,
    BATCH_NORM_RULES,
    CONV_BN_FOLD,
    CONV_FOLD_RULES,
    DEFAULT,
    FUSION,
    MINTED_RULES,
    PRIM_FOLDS,
    PRIM_PREMISES,
    law_provenance,
    minted,
    tags,
)
from mlopt_core.laws.serialize import alpha_key, missing_hooks


def _T(*shape: int) -> TensorType:
    return TensorType(tuple(shape))


def _spelled(cout: int = 6, bn_c: int | None = None) -> Op:
    """The eval-mode ``BatchNorm2d(Conv2d(x))`` spelling the mint saw."""
    bc = cout if bn_c is None else bn_c
    return Op.make(
        "batch_norm",
        Op.make(
            "conv2d",
            Var("x", _T(1, 4, 8, 8)),
            Param("W", _T(cout, 4, 3, 3)),
            Param("b", _T(cout)),
            stride=(1, 1),
            padding=(1, 1),
        ),
        Param("g", _T(bc)),
        Param("be", _T(bc)),
        Param("mu", _T(bc)),
        Param("var", _T(bc)),
        training=False,
        momentum=0.1,
        eps=1e-05,
        cudnn_enabled=False,
    )


# ---------------------------------------------------------------------------
#  membership + provenance
# ---------------------------------------------------------------------------


def test_conv_bn_fold_ships_in_default_and_classifies_machine():
    assert CONV_BN_FOLD in ALL_RULES
    assert [CONV_BN_FOLD, *PRIM_FOLDS] == MINTED_RULES
    assert CONV_BN_FOLD in DEFAULT.rules
    assert CONV_BN_FOLD in FUSION.rules
    assert law_provenance(CONV_BN_FOLD) == "machine"
    # ``axiom`` by annotation — the shipped-names ``derivation`` field
    # cannot name the unshipped ``syn_*`` intermediates the true
    # derivation runs through; the honest lineage is ``parents``:
    assert CONV_BN_FOLD.derivation == ()
    assert CONV_BN_FOLD.parents == (
        "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6"
        "__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209",
        "conv_channel_shift_sp",
    )
    assert CONV_BN_FOLD.guard_pats is not None
    # The transported class: procedural hooks flagged honestly, the
    # guard *spec* carried as data (guard_pats + parents).
    assert missing_hooks(CONV_BN_FOLD) == ("check", "derive")


def test_shipped_pattern_is_the_embedded_record():
    """The shipped 2-cell is byte-identical to the stored mint."""
    root = minted._RECORDS[
        "syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6"
        "__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209"
        "__conv_channel_shift_sp_896"
    ]
    from mlopt_core.laws.serialize import object_from_data

    bare = object_from_data(root)
    assert alpha_key(CONV_BN_FOLD.lhs, CONV_BN_FOLD.rhs) == alpha_key(
        bare.lhs, bare.rhs
    )


# ---------------------------------------------------------------------------
#  firing — the measured region, torch-free
# ---------------------------------------------------------------------------


def test_conv_bn_fold_fires_on_spelled_eval_site():
    out = meta.apply_rewrite_at(CONV_BN_FOLD, _spelled(), ())
    assert out is not None
    assert out.op == "conv2d"
    rep = op_repr(out)
    assert "batch_norm" not in rep
    # the folded scale chain materialized through the derive
    assert "rsqrt" in rep
    assert "shape=(6, 1, 1, 1)" in rep  # WV — the weight-side view
    assert "shape=(1, 6, 1, 1)" in rep  # CH — the channel view


def test_conv_bn_fold_declines_off_region():
    # a channel count disagreeing with the weight's cout — the
    # transported guard's derive chain declines
    assert (
        meta.apply_rewrite_at(CONV_BN_FOLD, _spelled(bn_c=3), ())
        is None
    )
    # a non-default eps spelling is outside the minted region (the
    # LHS carries the seed's literal attrs)
    off = Op.make(
        "batch_norm",
        Op.make(
            "conv2d",
            Var("x", _T(1, 4, 8, 8)),
            Param("W", _T(6, 4, 3, 3)),
            Param("b", _T(6)),
            stride=(1, 1),
            padding=(1, 1),
        ),
        Param("g", _T(6)),
        Param("be", _T(6)),
        Param("mu", _T(6)),
        Param("var", _T(6)),
        training=False,
        momentum=0.1,
        eps=1e-03,
        cudnn_enabled=False,
    )
    assert meta.apply_rewrite_at(CONV_BN_FOLD, off, ()) is None


def test_conv_bn_fold_extraction_pays():
    term = _spelled()
    eg = EGraph()
    root = eg.add_term(term)
    eg.run([CONV_BN_FOLD], root, max_iterations=4, max_nodes=20_000)
    best = eg.extract_best(root, count_cost)
    assert best is not None and best != term
    assert dag_cost(best, count_cost) < dag_cost(term, count_cost)


# ---------------------------------------------------------------------------
#  the materializer contract — loud, never a silent weaker admit
# ---------------------------------------------------------------------------


def test_materialize_refuses_unresolvable_parent():
    # a syn record whose leaf parents resolve to nothing must raise —
    # shipping a guard-less rebuild would be the weakened admit
    with pytest.raises(ValueError, match="parent"):
        minted.materialize(
            "syn_batch_norm_expand__conv_channel_shift_sub_sp_6", {}
        )
    with pytest.raises(ValueError, match="parent"):
        minted.materialize("no_such_mint", {})


def test_materialize_refuses_partial_lib():
    # only one of the two leaf parents resolvable — still a raise
    lib = {r.name: r for r in BATCH_NORM_RULES}
    with pytest.raises(ValueError, match="parent"):
        minted.materialize(
            "syn_batch_norm_expand__conv_channel_shift_sub_sp_6", lib
        )


def test_materialize_intermediate_rebuilds_with_guard():
    lib = {r.name: r for r in [*BATCH_NORM_RULES, *CONV_FOLD_RULES]}
    inner = minted.materialize(
        "syn_batch_norm_expand__conv_channel_shift_sub_sp_6", lib
    )
    assert inner.name == (
        "syn_batch_norm_expand__conv_channel_shift_sub_sp_6"
    )
    assert inner.check is not None
    assert inner.parents == (
        "batch_norm_expand",
        "conv_channel_shift_sub_sp",
    )
    assert meta.provenance(inner) == inner.parents


# ---------------------------------------------------------------------------
#  store round-trip — the shipped object re-encodes its guard spec
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  the prim family — ``prim:synop_*`` folds, laws-only in DEFAULT
# ---------------------------------------------------------------------------


def _prim_bound(fold: Any) -> Any:
    """Instantiate a prim fold's LHS pattern with concrete leaves.

    Bind every leaf metavar to a same-shaped ``Var`` and every
    ``$attr:`` metavar to a concrete ``-1`` — a spelled site the fold
    can fire on (torch-free: the pattern IS the data; the folds carry
    no check/derive hooks to veto it).
    """
    from mlopt_core.meta import instantiate_pattern, pattern_metavars

    subst = {
        mv: (-1 if mv.startswith("$attr:") else Var(mv, _T(2, 4)))
        for mv in pattern_metavars(fold.lhs)
    }
    return instantiate_pattern(fold.lhs, subst)


def test_prim_records_are_frozen_and_honest():
    recs = minted.prim_records()
    assert len(recs) == 21 == len(minted.SYNOP_SHIPPED)
    for name, rec in recs.items():
        assert name.startswith("synop_")
        assert rec["fold"]["name"] == f"prim:{name}"
        # the mint's own measurements ride the record — all sites
        # equal on the fp64 denotation check, >=2 bound modules
        assert rec["sites"] == rec["equal_sites"] >= 1
        assert len(rec["mods"]) >= 2
        assert rec["alpha"].startswith("(")
        assert rec["handler"]["verdict"] in ("minted", "refused")


def test_prim_folds_ship_in_default_and_classify_machine():
    assert len(PRIM_FOLDS) == 21
    by_op = {f.rhs.op: f for f in PRIM_FOLDS}
    assert set(by_op) == set(minted.SYNOP_SHIPPED)
    for f in PRIM_FOLDS:
        assert f.name.startswith("prim:synop_")
        assert f in DEFAULT.rules and f in FUSION.rules
        assert f in ALL_RULES
        assert law_provenance(f) == "machine"
        assert law_provenance(minted.expand_name(f.rhs.op)) == "machine"
        assert tags.FUSION in f.tags
        # pure pattern data — the full-data census majority
        assert missing_hooks(f) == ()
        assert f.derivation == ()  # axiom by convention
        premise = getattr(f, "premises", ())
        assert len(premise) == 1
        assert premise[0].lhs == f.rhs and premise[0].rhs == f.lhs
    # the premises are cert axioms, never saturation rules
    assert len(PRIM_PREMISES) == 21
    for p in PRIM_PREMISES:
        assert p.name.endswith(".expand")
        assert p not in DEFAULT.rules and p not in ALL_RULES
        assert tags.SIMPLIFICATION in p.tags


def test_prim_folds_are_interned_and_declare_ops():
    from mlopt_core.ir import generator

    assert minted.prim_folds()[0] is minted.prim_folds()[0]
    for f in PRIM_FOLDS:
        g = generator(f.rhs.op)
        assert g is not None
        assert g.n_in == len(f.rhs.args) and g.n_out == 1
        assert "synthesized primitive" in g.law


def test_prim_fold_fires_and_extracts_on_spelled_site():
    """A spelled site folds to the ``synop_*`` call and it wins."""
    from mlopt_core.egraph.terms import _term_match

    for fold in PRIM_FOLDS:
        site = _prim_bound(fold)
        m = _term_match(fold.lhs, site)
        assert m is not None, fold.name
        eg = EGraph()
        root = eg.add_term(site)
        eg.run([fold], root, max_iterations=4, max_nodes=20_000)
        ops = {n.op for ec in eg._classes.values() for n in ec.nodes}
        assert fold.rhs.op in ops, fold.name
        best = eg.extract_best(root, count_cost)
        ops_best = set()
        stack = [best]
        while stack:
            t = stack.pop()
            if isinstance(t, Op):
                ops_best.add(t.op)
                stack.extend(t.args)
        assert fold.rhs.op in ops_best, fold.name
        assert dag_cost(best, count_cost) <= dag_cost(site, count_cost)


def test_prim_extraction_honesty_unknown_op_pricing():
    """The ``synop_*`` member prices mid-range — never 0, never +inf
    by default — and the ``supported_ops`` bound gates delivery."""
    from mlopt_core.cost.backend import backend_cost
    from mlopt_core.cost.basic import flops_cost

    fold = PRIM_FOLDS[0]
    call = Op.make(
        fold.rhs.op,
        *[Var(f"v{i}", _T(2, 4)) for i in range(len(fold.rhs.args))],
    )
    # an unknown-to-the-sink op can never be committed to (+inf);
    # a supported one prices like one fused elementwise kernel —
    # the honest middle, not 0 and not unreachable.
    unbound = backend_cost(count_cost, frozenset())
    bound = backend_cost(count_cost, {fold.rhs.op})
    assert unbound(call) == float("inf")
    assert bound(call) == 1.0
    assert flops_cost(call) > 0.0
