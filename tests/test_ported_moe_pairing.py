"""Substrate-pure tests ported from ``graphopt:tests/test_moe_pairing.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.egraph import EGraph

from mlopt_core.ir import Op, Param, TensorType, Var

from mlopt_core.laws.pairing import (
    batch_tiled_expert_sums,
    pair_shared_input_linears,
)


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))

def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))

def _expert_sum_eg(gated=True, e=4, b=2, i=16, o=8, cover=None):
    """Build ``Σ_e [g_e ⊙] linear(h_e, select(W,0,e))`` in an e-graph.

    ``cover`` overrides the tile indices actually summed (default: all
    ``e`` tiles).  Returns ``(eg, root_eid, base_param)``.
    """
    eg = EGraph()
    base = _p("W", e, o, i)
    leaves = []
    for idx in range(e) if cover is None else cover:
        h = eg.add_term(_v(f"h{idx}", b, i))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        if gated:
            g = eg.add_term(_v(f"g{idx}", b, 1))
            lin = eg.add_enode("mul", (lin, g), {})
        leaves.append(lin)
    acc = leaves[0]
    for lin in leaves[1:]:
        acc = eg.add_enode("add", (acc, lin), {})
    return eg, acc, base

def _root_members(eg, root, op):
    return [n for n in eg._classes[eg.find(root)].nodes if n.op == op]

def test_pairing_select_tiles_emit_reshape():
    """Unit level: members weighted by dim-0 ``select`` tiles covering
    a shared stacked base fuse as ``reshape(base)``, not a runtime
    concat over non-folding views."""
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 8))
    base = _p("W", 4, 16, 8)
    for e in range(4):
        we = eg.add_term(Op.make("select", base, dim=0, index=e))
        eg.add_enode("linear", (x, we))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    assert len(groups[0]) == 4
    split = next(iter(groups[0].values()))
    fused_cid = eg.find(split.children[0])
    lin = next(
        n for n in eg._classes[fused_cid].nodes if n.op == "linear"
    )
    wc = eg._classes[eg.find(lin.children[1])]
    reshapes = [n for n in wc.nodes if n.op == "reshape"]
    assert len(reshapes) == 1
    assert dict(reshapes[0].attrs)["shape"] == (4 * 16, 8)
    assert not any(n.op == "concat" for n in wc.nodes)
    # split sizes ride in fused order — select-index order here.
    assert dict(split.attrs)["sizes"] == (16, 16, 16, 16)

def test_select_tile_guard_paths():
    """``_select_tile`` declines malformed tiles — each guard arm maps
    to a select enode that cannot be a stacked-parameter slice.

    The member classes keep a plain Param leaf (so the cluster forms
    and the pass inspects them); the degenerate select enode sits
    beside it and must be skipped:
      * a select with a non-integer ``index`` attr;
      * a select over a class with no resolvable term (cyclic);
      * a select on ``dim != 0``;
      * a select over a base with unknown dims;
      * a select enode with the wrong arity.
    """
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 8))

    def member(param_name):
        """A (16,8) weight class that also carries a weird select."""
        return eg.add_term(_p(param_name, 16, 8))

    # non-integer index
    wa = member("wa")
    sel = eg.add_enode(
        "select",
        (eg.add_term(_p("b1", 4, 16, 8)),),
        {"dim": 0, "index": "e"},
    )
    eg.union(sel, wa)
    # unresolvable base (cyclic-only e-class)
    wb = member("wb")
    leaf = eg.add_term(_p("cyc", 4, 16, 8))
    neg = eg.add_enode("neg", (leaf,))
    eg.union(neg, leaf)
    cyc = eg.find(leaf)
    for n in list(eg._classes[cyc].nodes):
        if n.op == "leaf":
            eg._classes[cyc].nodes.discard(n)
    sel = eg.add_enode("select", (cyc,), {"dim": 0, "index": 0})
    eg.union(sel, wb)
    # dim != 0
    wc = member("wc")
    sel = eg.add_enode(
        "select",
        (eg.add_term(_p("b3", 4, 16, 8)),),
        {"dim": 1, "index": 0},
    )
    eg.union(sel, wc)
    # base with an unknown extent
    wd = member("wd")
    sel = eg.add_enode(
        "select",
        (eg.add_term(_p("bn", None, 16, 8)),),
        {"dim": 0, "index": 0},
    )
    eg.union(sel, wd)
    # wrong arity (a select enode is unary)
    we = member("we")
    sel = eg.add_enode(
        "select",
        (
            eg.add_term(_p("b5", 4, 16, 8)),
            eg.add_term(_p("b6", 4, 16, 8)),
        ),
        {"dim": 0, "index": 0},
    )
    eg.union(sel, we)
    # one ordinary member so a real group still forms
    wn = eg.add_term(_p("wn", 16, 8))
    for w in (wa, wb, wc, wd, we, wn):
        eg.add_enode("linear", (x, w))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    split = next(iter(groups[0].values()))
    fused_cid = eg.find(split.children[0])
    lin = next(
        n for n in eg._classes[fused_cid].nodes if n.op == "linear"
    )
    wc_class = eg._classes[eg.find(lin.children[1])]
    # No tile survived a guard — the fused weight is the runtime concat.
    assert any(n.op == "concat" for n in wc_class.nodes)
    assert not any(n.op == "reshape" for n in wc_class.nodes)

def test_pairing_partial_tile_keeps_runtime_concat():
    """Selects covering only SOME indices of a base can't reshape —
    the fused weight stays the honest runtime concat."""
    eg = EGraph()
    x = eg.add_term(_v("x", 2, 8))
    base = _p("W", 4, 16, 8)
    for e in (0, 2):
        we = eg.add_term(Op.make("select", base, dim=0, index=e))
        eg.add_enode("linear", (x, we))
    groups = pair_shared_input_linears(eg)
    assert len(groups) == 1
    split = next(iter(groups[0].values()))
    fused_cid = eg.find(split.children[0])
    lin = next(
        n for n in eg._classes[fused_cid].nodes if n.op == "linear"
    )
    wc = eg._classes[eg.find(lin.children[1])]
    assert any(n.op == "concat" for n in wc.nodes)
    assert not any(n.op == "reshape" for n in wc.nodes)

def test_batch_ungated_expert_sum_offered():
    """``Σ_e linear(h_e, W[e])`` gains the grouped-GEMM member
    ``reshape(sum(matmul(reshape(stack h), W.mT), 0))`` — no gate."""
    eg, root, _ = _expert_sum_eg(gated=False)
    n = batch_tiled_expert_sums(eg)
    assert n >= 1
    resh = _root_members(eg, root, "reshape")
    assert len(resh) == 1
    # reshape -> sum -> matmul -> reshape -> stack
    summ = eg._classes[eg.find(resh[0].children[0])]
    assert any(n.op == "sum" for n in summ.nodes)
    mm = next(
        n
        for cid in [
            eg.find(s.children[0]) for s in summ.nodes if s.op == "sum"
        ]
        for n in eg._classes[cid].nodes
        if n.op == "matmul"
    )
    flat = eg._classes[eg.find(mm.children[0])]
    stk = next(n for n in flat.nodes if n.op == "reshape")
    assert dict(stk.attrs)["shape"] == (4, 2, 16)
    st = eg._classes[eg.find(stk.children[0])]
    assert any(
        n.op == "stack" and len(n.children) == 4 for n in st.nodes
    )

def test_batch_gated_expert_sum_offered():
    """``Σ_e g_e ⊙ linear(h_e, W[e])`` gains the member with one
    broadcast gate multiply over the batched bmm output."""
    eg, root, _ = _expert_sum_eg(gated=True)
    assert batch_tiled_expert_sums(eg) >= 1
    resh = _root_members(eg, root, "reshape")
    assert len(resh) == 1
    summ = eg._classes[eg.find(resh[0].children[0])]
    muls = [
        n
        for cid in [
            eg.find(s.children[0]) for s in summ.nodes if s.op == "sum"
        ]
        for n in eg._classes[cid].nodes
        if n.op == "mul"
    ]
    assert len(muls) == 1
    # one mul operand is the bmm output, the other the stacked gates
    for ch in muls[0].children:
        ops = {n.op for n in eg._classes[eg.find(ch)].nodes}
        assert ops & {"matmul", "reshape"}

def test_batch_residual_leaf_rejoins_add():
    """``x + Σ_e leaf`` offers ``add(x, batched)`` — the non-expert
    residual folds back in through a rebuilt add."""
    eg, inner, _ = _expert_sum_eg(gated=False)
    x = eg.add_term(_v("x", 2, 8))
    root = eg.add_enode("add", (x, inner), {})
    assert batch_tiled_expert_sums(eg) >= 1
    # root class keeps its original add AND gains add(batched, x)
    adds = _root_members(eg, root, "add")
    resh_children = 0
    for a in adds:
        for c in a.children:
            if any(
                n.op == "reshape" for n in eg._classes[eg.find(c)].nodes
            ):
                resh_children += 1
    assert resh_children >= 1

def test_batch_partial_cover_declines():
    """Tiles {0,2} of a 4-expert base can't bmm off the stacked param —
    no offer (a runtime stack of selects would cost, not save)."""
    eg, root, _ = _expert_sum_eg(gated=False, cover=[0, 2])
    assert batch_tiled_expert_sums(eg) == 0
    assert not _root_members(eg, root, "reshape")

def test_batch_duplicate_tile_declines():
    """Two leaves on the same tile index can't form a full cover."""
    eg = EGraph()
    base = _p("W", 4, 8, 16)
    h0 = eg.add_term(_v("h0", 2, 16))
    h1 = eg.add_term(_v("h1", 2, 16))
    h2 = eg.add_term(_v("h2", 2, 16))
    w0 = eg.add_term(Op.make("select", base, dim=0, index=0))
    w0b = eg.add_term(Op.make("select", base, dim=0, index=0))
    w1 = eg.add_term(Op.make("select", base, dim=0, index=1))
    a = eg.add_enode("linear", (h0, w0), {})
    b = eg.add_enode("linear", (h1, w0b), {})
    c = eg.add_enode("linear", (h2, w1), {})
    eg.add_enode("add", (a, eg.add_enode("add", (b, c), {})), {})
    assert batch_tiled_expert_sums(eg) == 0

def test_batch_shape_mismatch_declines():
    """Heterogeneous expert inputs can't stack — no offer."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    hs = [eg.add_term(_v("h0", 2, 16)), eg.add_term(_v("h1", 4, 16))]
    ls = []
    for idx in range(2):
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        ls.append(eg.add_enode("linear", (hs[idx], we), {}))
    eg.add_enode("add", tuple(ls), {})
    assert batch_tiled_expert_sums(eg) == 0

def test_batch_unknown_or_low_rank_declines():
    """Unknown dims (``None``) and rank-1 inputs both decline — the
    flattened bmm needs a fully-known, rank >= 2 operand."""
    for shape in ((None, 16), (16,)):
        eg = EGraph()
        base = _p("W", 2, 8, 16)
        hs = [
            eg.add_term(_v("h0", *shape)),
            eg.add_term(_v("h1", *shape)),
        ]
        ls = []
        for idx in range(2):
            we = eg.add_term(Op.make("select", base, dim=0, index=idx))
            ls.append(eg.add_enode("linear", (hs[idx], we), {}))
        eg.add_enode("add", tuple(ls), {})
        assert batch_tiled_expert_sums(eg) == 0

def test_batch_mixed_gating_declines():
    """A sum of gated AND ungated leaves has no uniform batched form."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        if idx == 0:
            g = eg.add_term(_v("g0", 2, 1))
            lin = eg.add_enode("mul", (lin, g), {})
        leaves.append(lin)
    eg.add_enode("add", tuple(leaves), {})
    assert batch_tiled_expert_sums(eg) == 0

def test_batch_wrong_gate_shape_declines():
    """Gates that aren't per-token scalars (…, 1) can't broadcast
    against the flattened (E, N, o) bmm output — decline."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        lin = eg.add_enode("linear", (h, we), {})
        g = eg.add_term(_v(f"g{idx}", 2, 8))  # full-width gate
        leaves.append(eg.add_enode("mul", (lin, g), {}))
    eg.add_enode("add", tuple(leaves), {})
    assert batch_tiled_expert_sums(eg) == 0

def test_batch_second_base_leftovers_rejoin():
    """A sum mixing one fully-covered base with another base's leaf
    batches the covered group and re-adds the stranger serially."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    other = _p("V", 4, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        leaves.append(eg.add_enode("linear", (h, we), {}))
    h3 = eg.add_term(_v("h3", 2, 16))
    w3 = eg.add_term(Op.make("select", other, dim=0, index=1))
    stranger = eg.add_enode("linear", (h3, w3), {})
    root = eg.add_enode(
        "add",
        (leaves[0], eg.add_enode("add", (leaves[1], stranger), {})),
        {},
    )
    assert batch_tiled_expert_sums(eg) >= 1
    # the offered member re-adds the un-covered leaf's class
    adds = _root_members(eg, root, "add")
    assert any(
        eg.find(ch) == eg.find(stranger)
        for a in adds
        for ch in a.children
    )

def test_batch_cyclic_sum_declines():
    """An add class referencing itself on the decomposition path is a
    residual cycle — folding it into the offer would build a
    self-referential member, so the offer is declined even though two
    real expert leaves are present."""
    eg = EGraph()
    base = _p("W", 2, 8, 16)
    leaves = []
    for idx in range(2):
        h = eg.add_term(_v(f"h{idx}", 2, 16))
        we = eg.add_term(Op.make("select", base, dim=0, index=idx))
        leaves.append(eg.add_enode("linear", (h, we), {}))
    v = eg.add_term(_v("v", 2, 8))
    inner = eg.add_enode("add", (v, v), {})
    outer = eg.add_enode("add", (inner, v), {})
    eg.union(inner, outer)
    cyc = eg.find(inner)
    # Drop the acyclic add(v,v) member so the class's only add enode
    # is the self-loop — a deterministic on-path cycle.
    for n in list(eg._classes[cyc].nodes):
        if n.op == "add" and all(eg.find(c) != cyc for c in n.children):
            eg._classes[cyc].nodes.discard(n)
    eg.add_enode(
        "add",
        (leaves[0], eg.add_enode("add", (leaves[1], cyc), {})),
        {},
    )
    assert batch_tiled_expert_sums(eg) == 0

def test_batch_leaf_classification_edges():
    """``_expert_leaf`` declines non-expert classes; ``_linear_tile``
    declines biased linears, non-tile weights, and non-rank-3 bases."""
    from mlopt_core.laws.pairing import _expert_leaf, _linear_tile

    eg = EGraph()
    x = eg.add_term(_v("x", 2, 16))
    y = eg.add_term(_v("y", 2, 16))
    # plain mul/leaf classes are neither gated nor ungated experts
    plain_mul = eg.add_enode("mul", (x, y), {})
    assert _expert_leaf(eg, plain_mul) is None
    assert _linear_tile(eg, x) is None
    # bias (3-child) linear is skipped by the arity guard
    bias = eg.add_term(_p("b", 8))
    w = eg.add_term(_p("Wp", 8, 16))
    biased = eg.add_enode("linear", (x, w, bias), {})
    assert _linear_tile(eg, biased) is None
    # select tile of a rank-2 base is not a rank-3 expert stack
    w2d = _p("W2d", 4, 16)
    sel = eg.add_term(Op.make("select", w2d, dim=0, index=0))
    l2d = eg.add_enode("linear", (x, sel), {})
    assert _linear_tile(eg, l2d) is None
    # gate-first argument order still classifies as a gated leaf
    base = _p("W", 2, 8, 16)
    h = eg.add_term(_v("h", 2, 16))
    we = eg.add_term(Op.make("select", base, dim=0, index=0))
    lin = eg.add_enode("linear", (h, we), {})
    g = eg.add_term(_v("g", 2, 1))
    mul = eg.add_enode("mul", (g, lin), {})  # gate first
    leaf = _expert_leaf(eg, mul)
    assert leaf is not None and leaf[4] is not None
    # a mul enode with the wrong arity is skipped
    weird = eg.add_enode("mul", (x, y, g), {})
    assert _expert_leaf(eg, weird) is None

def test_batch_single_expert_and_addless_graphs_skip():
    """Sums with <2 expert leaves and classes with no add enode are
    skipped — nothing to batch."""
    eg = EGraph()
    base = _p("W", 1, 8, 16)
    h = eg.add_term(_v("h", 2, 16))
    we = eg.add_term(Op.make("select", base, dim=0, index=0))
    lin = eg.add_enode("linear", (h, we), {})
    z = eg.add_term(_v("z", 2, 8))
    eg.add_enode("add", (lin, z), {})  # 1 leaf + residual -> skip
    assert batch_tiled_expert_sums(eg) == 0
