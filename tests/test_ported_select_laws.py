"""Substrate-pure tests ported from ``graphopt:tests/test_select_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.ir import IR, Op, Param, TensorType, Var

from mlopt_core.laws import (
    DEFAULT,
    SELECT_MUL,
    SIMPLIFICATION_RULES,
    tags,
)

from mlopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)


def test_select_mul_registered_and_tagged():
    """``select_mul`` is a shipped simplification, in the default set."""
    assert SELECT_MUL in SIMPLIFICATION_RULES
    assert SELECT_MUL.tags == {tags.SIMPLIFICATION}
    assert "select_mul" in {r.name for r in DEFAULT}

def test_select_mul_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the same term, and
    the shared ``dim``/``index`` metavariables bind consistently."""
    u = Var("u", TensorType((4, 3, 8)))
    v = Var("v", TensorType((4, 3, 8)))
    subst = {"u": u, "v": v, "$attr:D": 1, "$attr:I": 2}
    assert pattern_metavars(SELECT_MUL.lhs) == {
        "u",
        "v",
        "$attr:D",
        "$attr:I",
    }
    term = instantiate_pattern(SELECT_MUL.lhs, subst)
    found = match_pattern(SELECT_MUL.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(SELECT_MUL.lhs, found) == term
    # the RHS instantiates to the fused select-of-product form.
    rhs = instantiate_pattern(SELECT_MUL.rhs, subst)
    assert rhs == Op.make(
        "select", Op.make("mul", u, v), dim=1, index=2
    )

def test_select_mul_declines_when_operands_do_not_broadcast():
    """The ``bcast`` clause declines operands whose product cannot
    denote — the derivable-gate audit's 8 rhs-err fix.

    ``u=(2,3)``, ``v=(2,3,1)``, ``D=0``: ``u[D] == v[D]`` (both 2), so
    the old ``dim-eq-attr``-only guard accepted it, yet ``mul(u, v)``
    cannot broadcast (``3`` vs ``3`` then ``2`` vs ``3``) — the minted
    RHS ``sel(u ⊙ v)`` is ill-typed.  ``("shaped", ("bcast", u, v))``
    is exactly "``mul(u, v)`` denotes".
    """
    from mlopt_core.laws.cond import eval_cond

    bad = {
        "u": Var("u", TensorType((2, 3))),
        "v": Var("v", TensorType((2, 3, 1))),
        "$attr:D": 0,
        "$attr:I": 0,
    }
    assert not eval_cond(SELECT_MUL.cond, bad)
    assert not SELECT_MUL.check(bad)
    # same-rank operands that DO broadcast stay accepted.
    good = {
        "u": Var("u", TensorType((2, 3))),
        "v": Var("v", TensorType((2, 3))),
        "$attr:D": 0,
        "$attr:I": 0,
    }
    assert SELECT_MUL.check(good)
