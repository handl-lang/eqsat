"""Substrate-pure tests ported from ``graphopt:tests/test_morphisms.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import mlopt_orchestrator.morphisms as M

from mlopt_core.ir import IR, Op, Param, TensorType, Var

from mlopt_core.laws import RuleSet


def test_sig_post_norm_not_pre():
    """A scale AFTER the projection is a post norm, not pre."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    g = Param("p_gain", TensorType((16,)))
    ir = IR(
        root=Op.make("mul", Op.make("linear", x, w), g),
        inputs=[x],
        input_names={"x"},
        params={"p_w": w, "p_gain": g},
    )
    sig = M.block_signature(ir)
    assert sig.norm.kind == "diag" and sig.norm.affine
    assert sig.norm.pre is False

def test_sig_hand_built_matmul_forms():
    """matmul: weight on either side; activation x activation and
    weight x weight matmuls are not projections at all."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    w2 = Param("p_w2", TensorType((16, 16)))

    sig = M.block_signature(
        IR(root=Op.make("matmul", x, w), inputs=[x], input_names={"x"})
    )
    assert [r.name for r in sig.in_projs] == ["p_w"]

    sig = M.block_signature(
        IR(root=Op.make("matmul", w, x), inputs=[x], input_names={"x"})
    )
    assert [r.name for r in sig.in_projs] == ["p_w"]

    sig = M.block_signature(
        IR(
            root=Op.make("matmul", x, y),
            inputs=[x, y],
            input_names={"x", "y"},
        )
    )
    assert not sig.in_projs and not sig.out_proj

    sig = M.block_signature(
        IR(root=Op.make("matmul", w, w2), inputs=[], input_names=set())
    )
    assert not sig.in_projs and not sig.out_proj
    assert sig.shape[0] is None  # no inputs

def test_sig_folded_weight_expr():
    """A param-only weight expression (``W1 + W2``) is still a weight —
    the ref carries ``name=""`` and an inferred shape."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    w2 = Param("p_w2", TensorType((16, 16)))
    ir = IR(
        root=Op.make("linear", x, Op.make("add", w, w2)),
        inputs=[x],
        input_names={"x"},
    )
    sig = M.block_signature(ir)
    (ref,) = sig.out_proj
    assert ref.name == "" and ref.shape == (16, 16)

def test_sig_var_weight_and_short_arity():
    """``linear`` with a *var* weight is not a projection; a bare
    1-arg ``linear`` node is skipped by arity."""
    x = Var("x", TensorType((8, 16)))
    wv = Var("w", TensorType((16, 16)))
    ir = IR(
        root=Op.make("linear", x, wv),
        inputs=[x, wv],
        input_names={"x", "w"},
    )
    assert not M.block_signature(ir).in_projs
    # Raw arity-1 node (dodges schema validation) — skipped.
    odd = Op("linear", (x,), {})
    ir = IR(root=odd, inputs=[x], input_names={"x"})
    assert not M.block_signature(ir).in_projs
    # Bare var root — no ops at all.
    ir = IR(root=x, inputs=[x], input_names={"x"})
    sig = M.block_signature(ir)
    assert not sig.in_projs and sig.norm.kind == "none"

def test_sig_norm_edge_cases():
    """Norm detection edges: both-sides-rsqrt mul, var x var mul,
    param x param mul, matrix gain (not a channel diagonal), and a
    non-tuple-shaped gain."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    rms = Op.make("rsqrt", Op.make("mean", Op.make("pow", x, 2)))

    # both sides carry an rsqrt core -> not an rms node, no var side.
    t = Op.make("mul", rms, rms)
    assert M._norm_nodes(t) == []

    # var x var -> elementwise product, not a diagonal.
    t = Op.make("mul", x, Op.make("silu", x))
    assert M._norm_nodes(t) == []

    # param-only x param-only -> no var side.
    t = Op.make("mul", w, Param("p_w2", TensorType((16, 16))))
    assert M._norm_nodes(t) == []

    # matrix gain is not a scalar/channel diagonal.
    t = Op.make("mul", x, w)
    assert M._norm_nodes(t) == []

    # param-only gain with unknown shape -> still not a diagonal.
    weird = Op("mystery", (w,), {})
    t = Op.make("mul", x, weird)
    assert M._norm_nodes(t) == []

    # 1-arg mul node (raw) — skipped by arity.
    assert M._norm_nodes(Op("mul", (x,), {})) == []

def test_residual_spine_variants():
    """The add-spine monoid check: bare input addends, nested adds,
    shared add subtrees, a non-input Var, and no add spine at all."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    f = Op.make("linear", x, w)

    assert M._residual_spine(Op.make("add", f, x), [x]) is True
    assert M._residual_spine(Op.make("sub", f, x), [x]) is True
    assert M._residual_spine(Op.make("linear", x, w), [x]) is False
    # A Var that is not among the inputs does not count.
    assert M._residual_spine(Op.make("add", f, y), [x]) is False
    # Nested: add(add(f, g), x)
    g = Op.make("sigmoid", f)
    nested = Op.make("add", Op.make("add", f, g), x)
    assert M._residual_spine(nested, [x]) is True
    # Shared add-subtree addend (interned): exercises the seen-set.
    shared = Op.make("add", f, g)
    root = Op.make("add", shared, shared)
    assert M._residual_spine(root, [x]) is False

def test_weights_tied_predicate():
    """Signature-level tie: equal shape + equal stem or full name."""
    w1 = M.WeightRef(Param("p_w", TensorType((4, 4))), "p_w", (4, 4))
    w2 = M.WeightRef(Param("p_w", TensorType((4, 4))), "p_w", (4, 4))
    a = M.WeightRef(Param("a_0_w", TensorType((4, 4))), "a_0_w", (4, 4))
    b = M.WeightRef(Param("a_1_w", TensorType((4, 4))), "a_1_w", (4, 4))
    c = M.WeightRef(Param("c_q_w", TensorType((4, 4))), "c_q_w", (4, 4))
    d = M.WeightRef(Param("d_w", TensorType((8, 4))), "d_w", (8, 4))
    anon = M.WeightRef(Op.make("add", w1.term, w2.term), "", (4, 4))
    noshp = M.WeightRef(
        Param("e", TensorType((None, 4))), "e", (None, 4)
    )

    assert M.weights_tied(w1, w2) is True  # identical names
    assert M.weights_tied(a, b) is True  # equal stems (index tokens)
    assert M.weights_tied(a, c) is False  # different stems
    assert M.weights_tied(a, d) is False  # different shapes
    assert M.weights_tied(anon, w1) is False  # nameless weight expr
    assert M.weights_tied(a, noshp) is False  # unknown/mismatched shape

def test_dims_compatible_and_diag_predicates():
    """Unit coverage for the signature predicates' shape logic."""

    def mk(i: object, o: object) -> M.BlockSig:
        return M.BlockSig(
            in_projs=(),
            out_proj=(),
            norm=M.NormSig("none", False, False),
            act=(),
            residual=False,
            shape=(i, o),
        )

    a = mk(None, (8, 16))
    b_ok = mk((8, 16), None)
    b_bad = mk((8, 32), None)
    b_none = mk(None, None)
    b_none_dim = mk((8, None), None)
    b_str = mk("not-a-tuple", None)
    assert M._dims_compatible(a, b_ok)
    assert not M._dims_compatible(a, b_bad)
    assert M._dims_compatible(a, b_none)
    assert M._dims_compatible(a, b_none_dim)
    assert M._dims_compatible(a, b_str)

def test_distribute_over_edges():
    """The constructed bilinear step: matmul on either side, a
    non-mid data operand, non-Op terms, and empty-arg nodes."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    mid = Op.make("add", x, Op.make("linear", x, w))

    # matmul(mid, W) — data on the left
    t = Op.make("matmul", mid, w)
    got = M._distribute_over(t, mid)
    assert got.op == "add" and all(a.op == "matmul" for a in got.args)
    # matmul(W, mid) — data on the right
    t = Op.make("matmul", w, mid)
    got = M._distribute_over(t, mid)
    assert got.op == "add" and all(a.op == "matmul" for a in got.args)
    # linear(mid, W, b) — the bias rides one branch
    b = Param("p_b", TensorType((16,)))
    t = Op.make("linear", mid, w, b)
    got = M._distribute_over(t, mid)
    assert got.op == "add"
    right = got.args[1]
    assert right.op == "linear" and len(right.args) == 3
    # a projection whose data is NOT the mid node — untouched
    t = Op.make("linear", x, w)
    assert M._distribute_over(t, mid) == t
    # matmul(x, y): neither side is mid — falls through to rebuild
    t = Op.make("matmul", x, y)
    assert M._distribute_over(t, mid) == t
    # non-Op and empty-arg nodes pass through
    assert M._distribute_over(x, mid) is x
    assert M._distribute_over(Op("mul", (x,), {}), mid).op == "mul"
    assert M._distribute_over(Op("linear", (), {}), mid).op == "linear"
    assert (
        M._distribute_over(Op("matmul", (mid,), {}), mid).op == "matmul"
    )

def test_saturate_empty_rules_and_offers():
    """``_saturate`` edge cases: empty rule set skips the run; an offer
    identical to the joint term is not re-offered."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    t = Op.make("linear", x, w)
    empty = RuleSet("empty", ())
    eg, eid = M._saturate(
        t, empty, max_iterations=2, max_enodes=100, symmetry_budget=None
    )
    assert eg.extract_best(eid, M.flops_cost) == t
    eg, eid = M._saturate(
        t,
        empty,
        max_iterations=2,
        max_enodes=100,
        symmetry_budget=4,
        offers=[(t, "self-offer — skipped")],
    )
    assert eg.extract_best(eid, M.flops_cost) == t

def test_recipe_rules_cache():
    """Recipe sets build once and memoise."""
    a = M._recipe_rules("compose")
    b = M._recipe_rules("compose")
    assert a is b
    assert "assoc_linear" in a
    assert len(M._recipe_rules("tie")) == 0
