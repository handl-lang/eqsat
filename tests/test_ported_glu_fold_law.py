"""Substrate-pure tests ported from ``graphopt:tests/test_glu_fold_law.py``.

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
    GLU_FOLD,
    SIMPLIFICATION_RULES,
    _check_glu_fold,
    _check_glu_shaped,
    tags,
)

from mlopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)


def _manual_glu_lhs(u=None, v=None, **kw):
    """``mul(chunk(u,i0), sigmoid(chunk(v,i1)))`` — *v* defaults to *u*
    (a legal LHS instance; pass a different var or ``chunks``/``dim``/
    ``index`` kwargs to build the declined shapes)."""
    u = u if u is not None else Var("u", TensorType((4, 8)))
    v = v if v is not None else u
    chunks = kw.get("chunks", 2)
    d0 = kw.get("d0", -1)
    d1 = kw.get("d1", -1)
    i0 = kw.get("i0", 0)
    i1 = kw.get("i1", 1)
    return Op.make(
        "mul",
        Op.make("chunk", u, chunks=chunks, dim=d0, index=i0),
        Op.make(
            "sigmoid",
            Op.make("chunk", v, chunks=chunks, dim=d1, index=i1),
        ),
    )

def test_glu_fold_registered_and_tagged():
    """``glu_fold`` is a shipped simplification, in the default set."""
    assert GLU_FOLD in SIMPLIFICATION_RULES
    assert GLU_FOLD.tags == {tags.SIMPLIFICATION}
    assert GLU_FOLD.kind == "axiom"
    assert "glu_fold" in {r.name for r in DEFAULT}
    assert "glu_fold" in {r.name for r in ALL_RULES}

def test_glu_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the shared leaf,
    the split-axis attr, and the ``glu`` RHS."""
    u = Var("u", TensorType((4, 8)))
    subst = {"u": u, "$attr:D": -1}
    assert pattern_metavars(GLU_FOLD.lhs) == {"$attr:D", "u"}
    term = instantiate_pattern(GLU_FOLD.lhs, subst)
    assert term == _manual_glu_lhs(u=u)
    found = match_pattern(GLU_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(GLU_FOLD.lhs, found) == term
    assert instantiate_pattern(GLU_FOLD.rhs, found) == Op.make(
        "glu", u, dim=-1
    )

def test_check_glu_fold_branches():
    """Direct ``bound`` probes pin every decline path of the check."""
    shaped = Var("u", TensorType((4, 8)))
    assert _check_glu_fold({"u": shaped, "$attr:D": -1})
    # unshaped u — _shape_of returns non-tuple
    assert not _check_glu_fold(
        {"u": Var("u", TensorType(None)), "$attr:D": -1}
    )
    # non-int dim attr
    assert not _check_glu_fold({"u": shaped, "$attr:D": (-1,)})
    # dim out of range
    assert not _check_glu_fold({"u": shaped, "$attr:D": 5})
    # scalar u — rank 0 has no axis to halve (also the cond's veto)
    assert not _check_glu_fold(
        {"u": Var("u", TensorType(())), "$attr:D": 0}
    )
    assert not _check_glu_shaped({"u": Var("u", TensorType(()))})
    assert _check_glu_shaped({"u": shaped})
