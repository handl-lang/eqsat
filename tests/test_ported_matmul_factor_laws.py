"""Substrate-pure tests ported from ``graphopt:tests/test_matmul_factor_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.egraph import EGraph, verify_certificate

from mlopt_core.ir import Const, Op, Param, TensorType, Var

from mlopt_core.laws import (
    ALL_RULES,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    PARALLEL_MUL_FUSE,
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
    SWIGLU_FUSE,
    WEIGHT_DISTRIBUTE,
    WEIGHT_DISTRIBUTE_LINEAR,
    WEIGHT_FACTOR,
    WEIGHT_FACTOR_LINEAR,
    tags,
)

from mlopt_core.laws.tensor import (
    _check_fuse_pair,
    _check_mm_rhs_addends,
    _check_mm_rhs_weights,
)


def _saturate(term, rules, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root

def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))

def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))

def _counterexample_term():
    """The documented witness: x(16,)@a(16,) + x(16,)@b(16,16)."""
    x, a, b = _v("x", 16), _p("a", 16), _p("b", 16, 16)
    return Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )

def _swiglu_lhs(sa, sb):
    x = _v("x", 5, sa[1])
    a, b = _p("A", *sa), _p("B", *sb)
    return Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, a)),
        Op.make("linear", x, b),
    )

def test_check_mm_rhs_addends_accepts_legal_shapes():
    assert _check_mm_rhs_addends(
        {"a": _p("a", 4, 3), "b": _p("b", 4, 3)}
    )
    # both >= 2: rank mismatch is legal (batch broadcast)
    assert _check_mm_rhs_addends(
        {"a": _p("a", 4, 3), "b": _p("b", 2, 4, 3)}
    )
    # both vectors
    assert _check_mm_rhs_addends({"a": _p("a", 4), "b": _p("b", 4)})

def test_check_mm_rhs_addends_declines_illegal_shapes():
    # the counterexample shape: vector x matrix
    assert not _check_mm_rhs_addends(
        {"a": _p("a", 16), "b": _p("b", 16, 16)}
    )
    assert not _check_mm_rhs_addends(
        {"a": _p("a", 4, 1), "b": _p("b", 4)}
    )
    # scalar addend can never feed matmul
    assert not _check_mm_rhs_addends({"a": _p("a"), "b": _p("b", 4, 3)})
    # unknown shape (a ()-shaped transpose operand reports None):
    # cannot prove alignment — declines
    assert not _check_mm_rhs_addends(
        {"a": Op.make("transpose", Const(1.0)), "b": _p("b", 4)}
    )
    # missing bindings decline rather than crash
    assert not _check_mm_rhs_addends({})

def test_check_mm_rhs_weights_uses_weight_keys():
    assert _check_mm_rhs_weights(
        {"W": _p("w", 3, 4), "W2": _p("w2", 3, 4)}
    )
    assert not _check_mm_rhs_weights(
        {"W": _p("w", 4), "W2": _p("w2", 4, 3)}
    )
    assert not _check_mm_rhs_weights(
        {"W": _p("w"), "W2": _p("w2", 4, 3)}
    )
    assert not _check_mm_rhs_weights({})

def test_weight_factor_declines_on_mixed_rank():
    """The counterexample binding must NOT fire."""
    eg, _root = _saturate(_counterexample_term(), [WEIGHT_FACTOR])
    assert eg.rule_fires.get("weight_factor_matmul", 0) == 0

def test_factor_and_distribute_decline_on_mixed_rank():
    """The same binding vetoes the distribute direction too."""
    w, a, b = _p("w", 5, 4), _p("a", 4), _p("b", 4, 1)
    term = Op.make("matmul", w, Op.make("add", a, b))
    eg, _root = _saturate(term, [DISTRIBUTE_MUL, FACTOR_MUL])
    assert eg.rule_fires.get("distribute_matmul_over_add", 0) == 0
    assert eg.rule_fires.get("factor_matmul", 0) == 0

def test_weight_distribute_declines_on_mixed_rank():
    x, a, b = _v("x", 5, 4), _p("a", 4), _p("b", 4, 3)
    term = Op.make("matmul", x, Op.make("add", a, b))
    eg, _root = _saturate(term, [WEIGHT_DISTRIBUTE])
    assert eg.rule_fires.get("weight_distribute_matmul", 0) == 0

def test_equivalence_class_still_holds_under_guard():
    """On a legal binding the 4-member class still reaches both forms.

    ``factor_matmul`` and ``weight_factor_matmul`` mint the IDENTICAL
    merged member, so when both rules run together the second match is
    a no-op merge — honestly not counted as a fire.  Each rule is
    therefore checked alone on its own LHS instance, and the class is
    checked at the member level: saturate one form under all four
    rules, the other form must land in the root e-class.
    """
    x, a, b = _v("x", 5, 4), _p("a", 4, 3), _p("b", 4, 3)
    factored = Op.make(
        "add", Op.make("matmul", x, a), Op.make("matmul", x, b)
    )
    merged = Op.make("matmul", x, Op.make("add", a, b))
    rules = [
        WEIGHT_FACTOR,
        WEIGHT_DISTRIBUTE,
        FACTOR_MUL,
        DISTRIBUTE_MUL,
    ]
    # each rule still fires alone on its own LHS instance
    for rule, inst in (
        (WEIGHT_FACTOR, factored),
        (FACTOR_MUL, factored),
        (WEIGHT_DISTRIBUTE, merged),
        (DISTRIBUTE_MUL, merged),
    ):
        eg_i, _ = _saturate(inst, [rule])
        assert eg_i.rule_fires.get(rule.name, 0) > 0, rule.name

    eg, root = _saturate(factored, rules)
    assert eg.find(root) == eg.find(eg.add_term(merged))

    eg2, root2 = _saturate(merged, rules)
    assert eg2.find(root2) == eg2.find(eg2.add_term(factored))

def test_fuse_pair_declines_on_unequal_weight_dims():
    """A (4,i) + B (1,i): the LHS broadcasts and evals, but the fused
    ``chunk(·,2)`` cannot recover the unequal halves — vetoed."""
    eg, _root = _saturate(
        _swiglu_lhs((4, 4), (1, 4)), [SWIGLU_FUSE, PARALLEL_MUL_FUSE]
    )
    assert eg.rule_fires.get("swiglu_fuse", 0) == 0
    eg2, _r2 = _saturate(
        Op.make(
            "mul",
            Op.make("linear", _v("x", 5, 4), _p("A", 4, 4)),
            Op.make("linear", _v("x", 5, 4), _p("B", 1, 4)),
        ),
        [PARALLEL_MUL_FUSE],
    )
    assert eg2.rule_fires.get("parallel_mul_fuse", 0) == 0

def test_fuse_pair_check_accepts_equal_and_wildcard():
    assert _check_fuse_pair({"A": _p("A", 4, 4), "B": _p("B", 4, 4)})
    # None dims are wildcards — only provable mismatches veto
    assert _check_fuse_pair(
        {"A": _p("A", 4, None), "B": _p("B", 4, 4)}
    )
    assert not _check_fuse_pair({"A": _p("A", 4, 4), "B": _p("B", 1, 4)})
    assert not _check_fuse_pair(
        {"A": _p("A", 4, 4), "B": _p("B", 2, 4, 4)}
    )
    assert not _check_fuse_pair({})

def test_swiglu_fuse_still_fires_on_equal_weights():
    x = _v("x", 5, 4)
    a, b = _p("A", 4, 4), _p("B", 4, 4)
    term = Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, a)),
        Op.make("linear", x, b),
    )
    eg, root = _saturate(term, [SWIGLU_FUSE])
    assert eg.rule_fires.get("swiglu_fuse", 0) > 0
    fused = Op.make("linear", x, Op.make("concat", a, b, dim=0))
    rhs = Op.make(
        "mul",
        Op.make(
            "silu", Op.make("chunk", fused, chunks=2, dim=-1, index=0)
        ),
        Op.make("chunk", fused, chunks=2, dim=-1, index=1),
    )
    assert eg.find(root) == eg.find(eg.add_term(rhs))

def test_weight_factor_fires_and_certifies_on_equal_shapes():
    x, w1, w2 = _v("x", 4, 4), _p("W", 4, 4), _p("W2", 4, 4)
    src = Op.make(
        "add", Op.make("matmul", x, w1), Op.make("matmul", x, w2)
    )
    eg, root = _saturate(src, [WEIGHT_FACTOR])
    assert eg.rule_fires.get("weight_factor_matmul", 0) > 0
    merged = Op.make("matmul", x, Op.make("add", w1, w2))
    assert eg.find(root) == eg.find(eg.add_term(merged))
    cert = eg.certificate(src, merged)
    verify_certificate(src, cert)
    assert cert.rules_used == ["weight_factor_matmul"]

def test_all_four_rules_carry_the_guard():
    """Registration check: the whole right-operand family is guarded."""
    for r in (
        DISTRIBUTE_MUL,
        FACTOR_MUL,
        WEIGHT_FACTOR,
        WEIGHT_DISTRIBUTE,
        WEIGHT_FACTOR_LINEAR,
        WEIGHT_DISTRIBUTE_LINEAR,
        SWIGLU_FUSE,
        PARALLEL_MUL_FUSE,
    ):
        assert r.check is not None, r.name
    # and the tags the pipeline budgets on are unchanged — the
    # weight-merge pair is annotated alpha-duplicate of the
    # distribute/factor axioms (REDUNDANT: kept, not deleted)
    assert WEIGHT_FACTOR.tags == {
        tags.CATEGORICAL,
        tags.EXPANSIVE,
        tags.REDUNDANT,
    }
    assert WEIGHT_DISTRIBUTE.tags == WEIGHT_FACTOR.tags
    assert SWIGLU_FUSE.tags == {tags.FUSION, tags.SUBSUMED}
    names = {r.name for r in ALL_RULES}
    assert {
        "distribute_matmul_over_add",
        "factor_matmul",
        "weight_factor_matmul",
        "weight_distribute_matmul",
    } <= names
