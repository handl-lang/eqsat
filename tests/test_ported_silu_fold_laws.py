"""Substrate-pure tests ported from ``graphopt:tests/test_silu_fold_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.ir import IR, Op, Param, TensorType, Var

from mlopt_core.laws import (
    ALL_RULES,
    DEFAULT,
    SILU_EXPAND,
    SILU_FOLD,
    SIMPLIFICATION_RULES,
    SWIGLU_FUSE,
    tags,
)

from mlopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)


def test_silu_fold_registered_and_tagged():
    """``silu_fold`` is a shipped simplification, in the default set."""
    assert SILU_FOLD in SIMPLIFICATION_RULES
    assert SILU_FOLD.tags == {tags.SIMPLIFICATION}
    assert "silu_fold" in {r.name for r in DEFAULT}
    assert "silu_fold" in {r.name for r in ALL_RULES}

def test_silu_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the shared leaf
    and the ``silu`` RHS."""
    u = Var("u", TensorType((4, 8)))
    subst = {"x": u}
    assert pattern_metavars(SILU_FOLD.lhs) == {"x"}
    term = instantiate_pattern(SILU_FOLD.lhs, subst)
    assert term == Op.make("mul", u, Op.make("sigmoid", u))
    found = match_pattern(SILU_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(SILU_FOLD.lhs, found) == term
    assert instantiate_pattern(SILU_FOLD.rhs, found) == Op.make(
        "silu", u
    )
