"""Substrate-pure tests ported from ``graphopt:tests/test_softmax_laws.py``.

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
    SIMPLIFICATION_RULES,
    SOFTMAX_FOLD,
    tags,
)

from mlopt_core.laws.tensor import (
    _check_sum_keepdim,
    _derive_softmax_dim,
)

from mlopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)


def test_softmax_fold_registered_and_tagged():
    """``softmax_fold`` is a shipped simplification, in the default set."""
    assert SOFTMAX_FOLD in SIMPLIFICATION_RULES
    assert SOFTMAX_FOLD.tags == {tags.SIMPLIFICATION}
    assert "softmax_fold" in {r.name for r in DEFAULT}

def test_softmax_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the same term, and
    ``derive`` produces the ``$attr:SD`` the RHS needs."""
    u = Var("u", TensorType((4, 8)))
    subst = {"u": u, "$attr:RD": (-1,), "$attr:RK": True}
    assert pattern_metavars(SOFTMAX_FOLD.lhs) == {
        "u",
        "$attr:RD",
        "$attr:RK",
    }
    term = instantiate_pattern(SOFTMAX_FOLD.lhs, subst)
    found = match_pattern(SOFTMAX_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(SOFTMAX_FOLD.lhs, found) == term
    # the RHS instantiates to the folded softmax — the sum's dim tuple
    # unwrapped to the scalar softmax dim.
    rhs_subst = {**found, **SOFTMAX_FOLD.derive(found)}
    rhs = instantiate_pattern(SOFTMAX_FOLD.rhs, rhs_subst)
    assert rhs == Op.make("softmax", u, dim=-1)

def test_check_sum_keepdim_accepts_legal_shapes():
    """keepdim=True with a single reduce axis — tuple or int spelling."""
    assert _check_sum_keepdim({"$attr:RK": True, "$attr:RD": (-1,)})
    assert _check_sum_keepdim({"$attr:RK": True, "$attr:RD": 1})

def test_check_sum_keepdim_declines_illegal_shapes():
    """No keepdim, a multi-axis reduce, or a dimless (full) reduce."""
    assert not _check_sum_keepdim({"$attr:RK": False, "$attr:RD": (-1,)})
    assert not _check_sum_keepdim(
        {"$attr:RK": True, "$attr:RD": (-2, -1)}
    )
    assert not _check_sum_keepdim({"$attr:RK": True, "$attr:RD": None})

def test_derive_softmax_dim_unwraps_the_tuple():
    """The sum's ``dim`` tuple becomes softmax's scalar ``dim``; an int
    passes through unchanged."""
    assert _derive_softmax_dim({"$attr:RD": (-1,)}) == {"$attr:SD": -1}
    assert _derive_softmax_dim({"$attr:RD": 1}) == {"$attr:SD": 1}
