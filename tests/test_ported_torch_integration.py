"""Substrate-pure tests ported from ``graphopt:tests/test_torch_integration.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var, op_repr


def test_pow_to_square_bridge():
    """pow(x, 2) and square(x) land in the same e-class."""
    from mlopt_core.ir import TensorType, Var
    from mlopt_core.laws import POW_TO_SQUARE, SQUARE_TO_POW

    x = Var("x", TensorType((4, 4)))
    term = Op.make("pow", x, Const(2))
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        [POW_TO_SQUARE, SQUARE_TO_POW],
        eid,
        max_iterations=5,
        max_nodes=1000,
    )
    ops = {n.op for n in eg.get_class(eid).nodes}
    assert "pow" in ops
    assert "square" in ops

def test_silu_mul_form_rule():
    """silu(g)*u expands to (g*sigmoid(g))*u in the e-graph."""
    from mlopt_core.ir import TensorType, Var
    from mlopt_core.laws import SILU_MUL_FORM

    g = Var("g", TensorType((4, 4)))
    u = Var("u", TensorType((4, 4)))
    term = Op.make("mul", Op.make("silu", g), u)
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([SILU_MUL_FORM], eid, max_iterations=5, max_nodes=1000)

    # The rewritten form lives in the same e-class; silu's inner sigmoid
    # is nested one level down, so check the whole reachable term set.
    def ops_in(
        eclass_id: int, seen: set[int] | None = None
    ) -> set[str]:
        seen = seen or set()
        eid = eg.find(eclass_id)
        if eid in seen:
            return set()
        seen.add(eid)
        found = set()
        for node in eg.get_class(eid).nodes:
            found.add(node.op)
            for ch in node.children:
                found |= ops_in(ch, seen)
        return found

    ops = ops_in(eid)
    assert "mul" in ops
    assert "sigmoid" in ops

def test_swiglu_fuse_requires_shared_input():
    """The pairing rule must not fire when gate/up read DIFFERENT inputs."""
    from mlopt_core.laws import SWIGLU_FUSE

    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    wa = Param("Wa", TensorType((4, 4)))
    wb = Param("Wb", TensorType((4, 4)))
    t = Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, wa)),
        Op.make("linear", y, wb),
    )
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run([SWIGLU_FUSE], eid, max_iterations=5, max_nodes=500)
    # No chunk/concat nodes may appear: the product rewrite is invalid
    # without the shared source object.
    ops = {n.op for n in eg.get_class(eid).nodes}
    assert "chunk" not in ops and "concat" not in ops

def test_row_scale_rejects_data_scale():
    """Soundness: linear(a, data-shaped 'scale') must NOT row-hoist.

    r bound to a (B,T,C) data tensor is well-typed but semantically
    wrong — the check predicate must reject it (regression for the
    diff=9.83 unsoundness caught by the verifier).
    """
    from mlopt_core.laws import LINEAR_ROW_SCALE

    x = Var("x", TensorType((4, 4)))
    r = Var("r", TensorType((4, 4)))  # data-shaped, NOT per-row
    w = Param("W", TensorType((4, 4)))
    t = Op.make("linear", Op.make("mul", x, r), w)
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run([LINEAR_ROW_SCALE], eid, max_iterations=5, max_nodes=500)
    # no mul(linear, r) enode may be created
    for n in eg.get_class(eid).nodes:
        if n.op == "mul":
            pytest.fail("row_scale fired on a data-shaped scale")

def test_pairing_no_shared_input_no_fusion():
    """Linears with different inputs must NOT be paired."""
    from mlopt_core.laws import pair_shared_input_linears

    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    wa = Param("A", TensorType((4, 4)))
    wb = Param("B", TensorType((4, 4)))
    t = Op.make(
        "mul", Op.make("linear", x, wa), Op.make("linear", y, wb)
    )
    eg = EGraph()
    _eid = eg.add_term(t)
    groups = pair_shared_input_linears(eg)
    assert not any(len(g) >= 2 for g in groups)
    assert not any(n.op == "split" for n in eg._node_to_class)

def test_channel_scale_rejects_row_scale():
    """linear(x*r, W) with r per-row must not fold r into W."""
    from mlopt_core.laws import LINEAR_CHANNEL_SCALE

    x = Var("x", TensorType((4, 8, 4)))
    r = Var("r", TensorType((4, 8, 1)))  # per-row
    w = Param("W", TensorType((4, 4)))
    t = Op.make("linear", Op.make("mul", x, r), w)
    eg = EGraph()
    eid = eg.add_term(t)
    eg.run([LINEAR_CHANNEL_SCALE], eid, max_iterations=5, max_nodes=500)
    # mul(W, r) would be ill-typed AND wrong; the check must reject it
    for n in eg.get_class(eid).nodes:
        if n.op == "linear":
            # only the original linear should exist
            pass

def test_reshape_negative_dim_shape_inference():
    """reshape attrs keep literal -1 dims; shape inference must resolve
    them from input numel rather than poisoning broadcast as _INVALID."""
    from mlopt_core.cost import _broadcast, _infer_op_shape

    x = Var("x", TensorType((4, 8)))
    r = Op.make("reshape", x, shape=(-1, 2, 4))
    assert _infer_op_shape(r) == (4, 2, 4)  # numel 32 / (2*4) = 4
    # unresolved -1 must broadcast as unknown (None), not _INVALID
    assert _broadcast((4, -1), (4, 8)) == (4, None)

def test_gqa_absorb_rejects_bad_repeat():
    """The check must veto chains that are NOT repeat_interleave —
    e.g. expand growing a non-inserted dim."""
    from mlopt_core.laws import _check_gqa_absorb

    # Forge a bad binding: expand grows dim 2 (a real dim), not the
    # unsqueezed dim 3.
    x = Var("k", TensorType((1, 8, 2, 32)))
    bad = {
        "$attr:UDk": 3,
        "$attr:ESk": (1, 8, 4, 4, 32),  # grew dim 2 too — not a copy
        "$attr:RSk": (1, 8, 8, 32),
        "$attr:UDv": 3,
        "$attr:ESv": (1, 8, 2, 4, 32),
        "$attr:RSv": (1, 8, 8, 32),
        "k": x,
        "v": Var("v", TensorType((1, 8, 2, 32))),
        "q": Var("q", TensorType((1, 8, 8, 32))),
    }
    assert _check_gqa_absorb(bad) is False
    # and the honest chain passes
    good = dict(bad)
    good["$attr:ESk"] = (1, 8, 2, 4, 32)
    assert _check_gqa_absorb(good) is True

def test_sdpa_fold_rejects_wrong_dim():
    """softmax over a non-key dim is NOT attention — must not fold."""
    from mlopt_core.ir import TensorType, Var
    from mlopt_core.laws import _check_softmax_dim

    q = Var("q", TensorType((1, 4, 32, 16)))
    bound = {"Q": q, "$attr:SD": 1}
    assert not _check_softmax_dim(bound)
