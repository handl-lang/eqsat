"""Substrate-pure tests ported from ``graphopt:tests/test_layout_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var

from mlopt_core.laws import (
    LAYOUT_RULES,
    LINEAR_FROM_MM_T,
    LINEAR_FROM_MM_T_BARE,
    LINEAR_TO_MM_T,
)

from mlopt_core.laws.layout import _matmul_rules

from mlopt_core.typing import _axis_pair


def _saturate(term, rules=None, iters=8, max_nodes=50_000):
    """Build an e-graph over *term* and saturate with LAYOUT_RULES."""
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        LAYOUT_RULES if rules is None else rules,
        eid,
        max_iterations=iters,
        max_nodes=max_nodes,
    )
    return eg, eid

def _fired(eg, name):
    return eg.rule_fires.get(name, 0) > 0

def test_axis_pair_normalisation():
    assert _axis_pair(0, 1, 3) == (0, 1)
    assert _axis_pair(-2, -1, 3) == (1, 2)
    assert _axis_pair(0, 1, 0) is None  # rank-0: no axes
    assert _axis_pair("x", 1, 2) is None  # non-int first dim
    assert _axis_pair(0, "x", 2) is None  # non-int second dim
    assert _axis_pair(0, 5, 2) is None  # second dim out of range
    assert _axis_pair(-3, 1, 2) is None  # first dim < -rank

def test_decline_binary_rank_mismatch():
    """(4,5) * (5,) — the rank-1 operand's trailing-dim alignment would
    realign under the swap; declined."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((5,)))
    lhs = Op.make("transpose", Op.make("mul", a, b), dim0=0, dim1=1)
    eg, eid = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_push_mul"
        ],
    )
    assert not _fired(eg, "transpose_push_mul")
    assert {n.op for n in eg.get_class(eid).nodes} == {"transpose"}

def test_decline_binary_unknown_shape():
    """An operand whose shape is unrecoverable (unknown op over a
    scalar) cannot be checked — declined."""
    a = Op.make("zzz_unknown_op", Const(1))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make("transpose", Op.make("mul", a, b), dim0=0, dim1=1)
    eg, _ = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_push_mul"
        ],
    )
    assert not _fired(eg, "transpose_push_mul")

def test_decline_unary_scalar_operand():
    """Rank-1 input: no axis pair exists to commute — declined."""
    v = Var("v", TensorType((5,)))
    lhs = Op.make("transpose", Op.make("sigmoid", v), dim0=0, dim1=0)
    eg, _ = _saturate(
        lhs,
        rules=[
            r
            for r in LAYOUT_RULES
            if r.name == "transpose_push_sigmoid"
        ],
    )
    assert not _fired(eg, "transpose_push_sigmoid")

def test_decline_unary_self_pair():
    """transpose(x, 0, 0) is a no-op view — not a layout worth holding."""
    x = Var("x", TensorType((4, 5)))
    lhs = Op.make("transpose", Op.make("tanh", x), dim0=0, dim1=0)
    eg, _ = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_push_tanh"
        ],
    )
    assert not _fired(eg, "transpose_push_tanh")

def test_decline_unary_unknown_shape():
    """transpose(f(<unshapeable>)) cannot be checked — declined."""
    x = Op.make("zzz_unknown_op", Const(1))
    lhs = Op.make("transpose", Op.make("neg", x), dim0=0, dim1=1)
    eg, _ = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_push_neg"
        ],
    )
    assert not _fired(eg, "transpose_push_neg")

def test_decline_out_of_range_dim():
    """dim1=9 on a rank-2 operand is out of range — declined."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make("transpose", Op.make("mul", a, b), dim0=0, dim1=9)
    eg, _ = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_push_mul"
        ],
    )
    assert not _fired(eg, "transpose_push_mul")

def test_decline_non_int_dim_attr():
    """A non-int attr bound for the axis metavariable — declined."""
    a = Var("a", TensorType((4, 5)))
    b = Var("b", TensorType((4, 5)))
    lhs = Op.make(
        "transpose", Op.make("mul", a, b), dim0="bogus", dim1=1
    )
    eg, _ = _saturate(
        lhs,
        rules=[
            r for r in LAYOUT_RULES if r.name == "transpose_push_mul"
        ],
    )
    assert not _fired(eg, "transpose_push_mul")

def test_decline_non_pointwise():
    """softmax is NOT elementwise — no rule touches it."""
    x = Var("x", TensorType((4, 5)))
    lhs = Op.make(
        "transpose",
        Op.make("softmax", x, dim=-1),
        dim0=0,
        dim1=1,
    )
    eg, eid = _saturate(lhs)
    assert not any(
        n.op == "softmax"
        and any(cn.op == "transpose" for cn in eg.get_class(c).nodes)
        for n in eg.get_class(eid).nodes
        for c in n.children
    )

def test_involution_declines_mismatched_pairs():
    """Different axis pairs do not cancel: T(0,1)∘T(1,2) ≠ id."""
    x = Var("x", TensorType((2, 4, 5)))
    lhs = Op.make(
        "transpose",
        Op.make("transpose", x, dim0=0, dim1=1),
        dim0=1,
        dim1=2,
    )
    eg, _ = _saturate(
        lhs,
        rules=[
            r
            for r in LAYOUT_RULES
            if r.name == "transpose_transpose_dd"
        ],
    )
    assert not _fired(eg, "transpose_transpose_dd")

def test_involution_declines_degenerate_inner():
    """A self-pair inner transpose (0,0) is not a swap — declined."""
    x = Var("x", TensorType((4, 5)))
    lhs = Op.make(
        "transpose",
        Op.make("transpose", x, dim0=0, dim1=0),
        dim0=0,
        dim1=0,
    )
    eg, _ = _saturate(
        lhs,
        rules=[
            r
            for r in LAYOUT_RULES
            if r.name == "transpose_transpose_dd"
        ],
    )
    assert not _fired(eg, "transpose_transpose_dd")

def test_involution_declines_unknown_inner_shape():
    x = Op.make("zzz_unknown_op", Const(1))
    lhs = Op.make(
        "transpose",
        Op.make("transpose", x, dim0=0, dim1=1),
        dim0=0,
        dim1=1,
    )
    eg, _ = _saturate(
        lhs,
        rules=[
            r
            for r in LAYOUT_RULES
            if r.name == "transpose_transpose_dd"
        ],
    )
    assert not _fired(eg, "transpose_transpose_dd")

def test_involution_declines_out_of_range_outer():
    """Outer dim out of range for the operand's rank — declined."""
    x = Var("x", TensorType((4, 5)))
    lhs = Op.make(
        "transpose",
        Op.make("transpose", x, dim0=0, dim1=1),
        dim0=0,
        dim1=7,
    )
    eg, _ = _saturate(
        lhs,
        rules=[
            r
            for r in LAYOUT_RULES
            if r.name == "transpose_transpose_dd"
        ],
    )
    assert not _fired(eg, "transpose_transpose_dd")

def test_decline_matmul_wrong_axis_pair():
    """transpose(mm, 0, 1) on a rank-3 output swaps batch+row —
    declined (that is not the .mT identity)."""
    a = Var("a", TensorType((2, 3, 4)))
    b = Var("b", TensorType((2, 4, 6)))
    lhs = Op.make(
        "transpose",
        Op.make("matmul", a, b),
        dim0=0,
        dim1=1,
    )
    eg, _ = _saturate(
        lhs,
        rules=[r for r in LAYOUT_RULES if r.name == "transpose_matmul"],
    )
    assert not _fired(eg, "transpose_matmul")

def test_decline_matmul_vector_operand():
    """Rank-1 B: .mT on a vector is the identity — the product law
    does not hold; declined."""
    a = Var("a", TensorType((3, 4)))
    b = Var("b", TensorType((4,)))
    lhs = Op.make(
        "transpose",
        Op.make("matmul", a, b),
        dim0=-2,
        dim1=-1,
    )
    eg, _ = _saturate(
        lhs,
        rules=[r for r in LAYOUT_RULES if r.name == "transpose_matmul"],
    )
    assert not _fired(eg, "transpose_matmul")

def test_decline_matmul_bad_batch_broadcast():
    """Provably ill-typed batch dims (2 vs 3) — declined."""
    a = Var("a", TensorType((2, 3, 4)))
    b = Var("b", TensorType((3, 4, 6)))
    lhs = Op.make(
        "transpose",
        Op.make("matmul", a, b),
        dim0=-2,
        dim1=-1,
    )
    eg, _ = _saturate(
        lhs,
        rules=[r for r in LAYOUT_RULES if r.name == "transpose_matmul"],
    )
    assert not _fired(eg, "transpose_matmul")

def test_decline_matmul_contraction_mismatch():
    """A[-1] != B[-2] with concrete ints — declined."""
    a = Var("a", TensorType((3, 4)))
    b = Var("b", TensorType((5, 6)))
    lhs = Op.make(
        "transpose",
        Op.make("matmul", a, b),
        dim0=-2,
        dim1=-1,
    )
    eg, _ = _saturate(
        lhs,
        rules=[r for r in LAYOUT_RULES if r.name == "transpose_matmul"],
    )
    assert not _fired(eg, "transpose_matmul")

def test_decline_mm_rev_non_swap_inner():
    """Inner transposes on non-last axes are not the .mT form —
    declined."""
    p = Var("p", TensorType((2, 3, 4)))
    q = Var("q", TensorType((2, 5, 3)))
    # P transposed on (0,1) — batch dims, not the matrix pair.
    lhs = Op.make(
        "matmul",
        Op.make("transpose", p, dim0=0, dim1=1),
        Op.make("transpose", q, dim0=0, dim1=1),
    )
    eg, _ = _saturate(lhs, rules=_matmul_rules())
    assert not _fired(eg, "matmul_transpose_rev_dd")

def test_decline_mm_rev_one_side_only():
    """P swaps last-two but Q swaps (0,1) on rank 3 — declined."""
    p = Var("p", TensorType((3, 4)))
    q = Var("q", TensorType((2, 5, 3)))
    lhs = Op.make(
        "matmul",
        Op.make("transpose", p, dim0=-2, dim1=-1),
        Op.make("transpose", q, dim0=0, dim1=1),
    )
    eg, _ = _saturate(lhs, rules=_matmul_rules())
    assert not _fired(eg, "matmul_transpose_rev_dd")

def test_decline_mm_rev_rank1_operand():
    """A rank-1 transpose operand has no matrix pair — declined."""
    p = Var("p", TensorType((4,)))
    q = Var("q", TensorType((3, 4)))
    lhs = Op.make(
        "matmul",
        Op.make("transpose", p, dim0=-2, dim1=-1),
        Op.make("transpose", q, dim0=-2, dim1=-1),
    )
    eg, _ = _saturate(lhs, rules=_matmul_rules())
    assert not _fired(eg, "matmul_transpose_rev_dd")

def test_decline_linear_weight_not_rank2():
    """W rank-3 — F.linear takes exactly a (out, in) weight."""
    x = Var("x", TensorType((2, 4)))
    w = Param("w", TensorType((2, 8, 4)))
    lhs = Op.make(
        "matmul",
        x,
        Op.make("transpose", w, dim0=-2, dim1=-1),
    )
    eg, _ = _saturate(
        lhs,
        rules=[LINEAR_FROM_MM_T, LINEAR_FROM_MM_T_BARE],
    )
    assert not _fired(eg, "linear_from_matmul_t")
    assert not _fired(eg, "linear_from_matmul_t_bare")

def test_decline_linear_dim_mismatch():
    """x's last dim must equal W's in-dim — declined otherwise."""
    x = Var("x", TensorType((2, 5)))
    w = Param("w", TensorType((8, 4)))
    lhs = Op.make("linear", x, w)
    eg, _ = _saturate(lhs, rules=[LINEAR_TO_MM_T])
    assert not _fired(eg, "linear_to_matmul_t")

def test_decline_linear_scalar_x():
    """A scalar x has no in-features axis — declined."""
    x = Op.make("zzz_unknown_op", Const(1))
    w = Param("w", TensorType((8, 4)))
    lhs = Op.make("linear", x, w)
    eg, _ = _saturate(lhs, rules=[LINEAR_TO_MM_T])
    assert not _fired(eg, "linear_to_matmul_t")

def test_decline_mm_t_non_swap_transpose():
    """matmul(x, W.transpose(0,0)) is x@W — NOT linear(x, W)."""
    x = Var("x", TensorType((2, 4)))
    w = Param("w", TensorType((8, 4)))
    lhs = Op.make(
        "matmul",
        x,
        Op.make("transpose", w, dim0=0, dim1=0),
    )
    eg, _ = _saturate(
        lhs,
        rules=[LINEAR_FROM_MM_T, LINEAR_FROM_MM_T_BARE],
    )
    assert not _fired(eg, "linear_from_matmul_t")
