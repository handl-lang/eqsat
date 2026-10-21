"""Substrate-pure tests ported from ``graphopt:tests/test_square_rsqrt_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import pytest

from mlopt_core.egraph import EGraph, verify_certificate

from mlopt_core.ir import Const, IR, Op, Param, TensorType, Var

from mlopt_core.laws import (
    ALL_RULES,
    DEFAULT,
    DIV_SQRT_TO_RSQRT,
    MUL_SQUARE,
    POW_TO_RSQRT,
    RECIP_SQRT_TO_RSQRT,
    SIMPLIFICATION_RULES,
    SQUARE_EXPAND,
    tags,
)

from mlopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)


def _mean(x: Op | Var) -> Op:
    """``mean(x, dim=(-1,), keepdim=True)`` — the rms reduce."""
    return Op.make("mean", x, dim=(-1,), keepdim=True)

def _rms_reduces(u):
    """Every recorded-miss spelling of ``rsqrt(mean(u²) + eps)`` —
    plus the canonical spelling (the control: it folds with no
    bridges) and the doubly-missed spelling (``x·x`` AND ``1/√``)."""
    mean_pow = _mean(Op.make("pow", u, Const(2)))
    mean_mul = _mean(Op.make("mul", u, u))
    eps_pow = Op.make("add", mean_pow, Const(1e-6))
    eps_mul = Op.make("add", mean_mul, Const(1e-6))
    return {
        "canonical_rsqrt": Op.make("rsqrt", eps_pow),
        "mul_spelled_x2": Op.make("rsqrt", eps_mul),
        "div_one_sqrt": Op.make(
            "div", Const(1), Op.make("sqrt", eps_pow)
        ),
        "recip_sqrt": Op.make("reciprocal", Op.make("sqrt", eps_pow)),
        "pow_neghalf": Op.make("pow", eps_pow, Const(-0.5)),
        "both_missed": Op.make(
            "div", Const(1), Op.make("sqrt", eps_mul)
        ),
    }

def test_bridge_laws_registered_and_tagged():
    """All four bridges ship in the simplification + default sets."""
    for rule in (
        MUL_SQUARE,
        DIV_SQRT_TO_RSQRT,
        POW_TO_RSQRT,
        RECIP_SQRT_TO_RSQRT,
    ):
        assert rule in SIMPLIFICATION_RULES
        assert rule.tags == {tags.SIMPLIFICATION}
        assert rule.name in {r.name for r in DEFAULT}
        assert rule.name in {r.name for r in ALL_RULES}
    # mul_square is square_expand's lemma; the rsqrt folds are axioms.
    assert MUL_SQUARE.kind == "lemma"
    assert MUL_SQUARE.derivation == ("square_expand",)
    for rule in (DIV_SQRT_TO_RSQRT, POW_TO_RSQRT, RECIP_SQRT_TO_RSQRT):
        assert rule.kind == "axiom", rule.name

def test_mul_square_match_instantiate_roundtrip():
    u = Var("u", TensorType((4, 8)))
    assert pattern_metavars(MUL_SQUARE.lhs) == {"u"}
    term = instantiate_pattern(MUL_SQUARE.lhs, {"u": u})
    assert term == Op.make("mul", u, u)
    found = match_pattern(MUL_SQUARE.lhs, term)
    assert found is not None and found == {"u": u}
    assert instantiate_pattern(MUL_SQUARE.rhs, found) == Op.make(
        "square", u
    )

def test_rsqrt_law_match_instantiate_roundtrips():
    u = Var("u", TensorType((4, 8)))
    lhs_terms = {
        DIV_SQRT_TO_RSQRT: Op.make(
            "div", Const(1), Op.make("sqrt", u)
        ),
        POW_TO_RSQRT: Op.make("pow", u, Const(-0.5)),
        RECIP_SQRT_TO_RSQRT: Op.make("reciprocal", Op.make("sqrt", u)),
    }
    for rule, term in lhs_terms.items():
        found = match_pattern(rule.lhs, term)
        assert found is not None, rule.name
        assert found["u"] == u
        assert instantiate_pattern(rule.rhs, found) == Op.make(
            "rsqrt", u
        )

def test_rsqrt_conds_accept_and_reject_const_leaves():
    """``const-cmp`` reads ``Const.value`` numerically — ``Const(1)``,
    ``Const(1.0)`` and ``Const(-0.5)`` accept; a ``Var`` leaf or a
    wrong literal declines."""
    u = Var("u", TensorType((4, 8)))
    w = Var("w", TensorType((4, 8)))
    assert DIV_SQRT_TO_RSQRT.cond is not None
    assert POW_TO_RSQRT.cond is not None
    assert DIV_SQRT_TO_RSQRT.check({"ONE": Const(1), "u": u})
    assert DIV_SQRT_TO_RSQRT.check({"ONE": Const(1.0), "u": u})
    assert not DIV_SQRT_TO_RSQRT.check({"ONE": Const(2), "u": u})
    assert not DIV_SQRT_TO_RSQRT.check({"ONE": w, "u": u})
    assert POW_TO_RSQRT.check({"P": Const(-0.5), "u": u})
    assert POW_TO_RSQRT.check({"P": Const(-0.5), "u": u})
    assert not POW_TO_RSQRT.check({"P": Const(0.5), "u": u})
    assert not POW_TO_RSQRT.check({"P": w, "u": u})

@pytest.mark.slow
def test_every_miss_spelling_reaches_rms_norm():
    """Each recorded-miss spelling of the manual RMSNorm lands the
    ``rms_norm`` member in the root e-class under ``ALL_RULES`` — the
    canon laws mint the ``rsqrt``/``pow(·, 2)`` members the fold's
    pattern pins, then ``rms_norm_fold`` fires."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    want = Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)
    for name, rec in _rms_reduces(u).items():
        eg = EGraph()
        root = eg.add_term(Op.make("mul", Op.make("mul", u, rec), w))
        eg.run(ALL_RULES, root, max_iterations=6, max_nodes=60_000)
        fires = eg.rule_fires
        assert eg.find(root) == eg.find(eg.add_term(want)), name
        assert fires.get("rms_norm_fold", 0) > 0, name

@pytest.mark.slow
def test_canonical_rsqrt_spellings_fold_with_their_bridges():
    """The bridges are load-bearing, not incidental: with the canon
    laws removed, ``mul(x,x)``/``1/√``/``pow(·,-0.5)`` spellings do
    NOT fold — exactly the recorded misses."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    want = Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)
    bridge_names = {
        "mul_square",
        "div_sqrt_to_rsqrt",
        "pow_to_rsqrt",
        "recip_sqrt_to_rsqrt",
    }
    without = [r for r in ALL_RULES if r.name not in bridge_names]
    for name, rec in _rms_reduces(u).items():
        term = Op.make("mul", Op.make("mul", u, rec), w)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(without, root, max_iterations=6, max_nodes=60_000)
        fires = eg.rule_fires
        folded = eg.find(root) == eg.find(eg.add_term(want))
        if name == "canonical_rsqrt":
            # The canonical spelling folds WITHOUT the bridges —
            # the fold itself was never broken.
            assert folded and fires.get("rms_norm_fold", 0) > 0
        else:
            # Every miss spelling: no fold fires (the canonical
            # ``rsqrt``/``pow(·, 2)`` member is never minted).
            assert not folded, name
            assert fires.get("rms_norm_fold", 0) == 0, name
