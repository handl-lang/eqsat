"""Substrate-pure tests ported from ``graphopt:tests/test_headshare.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.egraph import EGraph

from mlopt_core.egraph.types import ENode

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var

from mlopt_core.laws.headshare import (
    _Ctx,
    _joint_sig,
    _piece,
    _resolve,
    headshare_keys_hold,
    share_duplicate_attention_heads,
)

from mlopt_core.laws.headshare import (  # noqa: E402
    _bias_sources,
    _bounds,
    _cls_shape,
    _const_along,
    _h_concat,
    _h_cut,
    _h_dimop,
    _h_gather,
    _h_perm,
    _h_proj,
    _h_split,
    _h_squeeze,
    _h_stack,
    _h_unsq,
    _heads_merge,
    _heads_regroup,
    _interpret,
    _merge_one,
    _merge_sites,
    _piece_select,
    _prod,
    _site_regroup,
    _site_to_heads,
    _skel,
)


B, T, H, D, I = 2, 5, 4, 4, 16  # batch, seq, heads, head-dim, model-dim

def _t(*shape):
    return TensorType(tuple(shape))

def _chain(x, w, b=None):
    """transpose(reshape(linear(x, w[, b]), (B,T,H,D)), 1, 2)."""
    lin = (
        Op.make("linear", x, w, b) if b is not None
        else Op.make("linear", x, w)
    )
    return Op.make(
        "transpose",
        Op.make("reshape", lin, shape=(B, T, H, D)),
        dim0=1,
        dim1=2,
    )

def _sdpa(x, wq, wk, wv, **attrs):
    return Op.make(
        "sdpa", _chain(x, wq), _chain(x, wk), _chain(x, wv), **attrs
    )

def _graph(term):
    eg = EGraph()
    return eg, eg.add_term(term)

class _StubEG:
    """Duck-typed e-graph stand-in for impossible-to-build states."""

    def __init__(self, term=None):
        self.term = term

    def find(self, cid):
        return cid

    def _any_term_cached(self, cid):
        return self.term

def test_missing_tensors_decline():
    x = Var("x", _t(B, T, I))
    q, k, v = (Param(f"p_{n}", _t(H * D, I)) for n in "qkv")
    eg, _ = _graph(_sdpa(x, q, k, v))
    assert share_duplicate_attention_heads(eg, {}) == []

def test_pure_helpers():
    assert _prod((2, 3)) == 6
    assert _prod((2, None)) is None
    assert _cls_shape(_StubEG(), 0) is None

def test_site_to_heads_branches():
    src = (("p", "rows", 0),)
    site = ("s", (B, T, H * D), src, frozenset())
    st = _site_to_heads(site, (B, T, H, D))
    assert st[0] == "h" and len(st[3]) == H
    assert _site_to_heads(site, (B * T,)) is None  # too few dims
    assert _site_to_heads(("s", (), src, frozenset()), (B, T)) is None
    node_none = ("s", (B, T, None), src, frozenset())
    assert _site_to_heads(node_none, (B, T, H, D)) is None
    assert _site_to_heads(site, (B, T, H, 5)) is None  # hb*d != width
    assert _site_to_heads(site, (B, T + 1, H, D)) is None  # prefix

def test_site_regroup_branches():
    src = (("p", "rows", 0),)
    site = ("s", (B, T, H * D), src, frozenset())
    out = _site_regroup(site, (B * T, H * D))
    assert out is not None and out[1] == (B * T, H * D)
    assert _site_regroup(site, ()) is None
    assert _site_regroup(("s", (), src, frozenset()), (1,)) is None
    assert _site_regroup(site, (B, T, H * D - 1)) is None
    # a site whose last axis is a broadcast keeps the mark
    sb = ("s", (B, T, 1), src, frozenset({2}))
    assert _site_regroup(sb, (B * T, 1))[3] == frozenset({1})

def test_skel_fallback_branches():
    keys = []
    assert _skel(Const(2.5), keys) == ("c", 2.5)
    assert _skel("rawkey", keys) == ("l", "'rawkey'")
    # a piece skeleton embeds consts and vars
    x = Var("x", _t(B, T, I))
    w = Param("p", _t(I, D))
    keys = []
    sk = _skel(
        Op.make("mul", Op.make("matmul", x, w), Const(2.0)), keys
    )
    assert sk[0] == "op" and keys == [("p", "whole", 0, 0)]

def test_bounds_rejects_unknown_extent():
    """Slice/narrow bounds need a concrete integer extent."""
    node = ENode("slice", (), ())
    assert _bounds({"dim": -1}, node, None) is None
    assert _bounds({"dim": -1}, node, "x") is None

def test_merge_sites_incompatible_widths_decline():
    """Sites of different non-uniform widths cannot share blocks."""
    x = Var("x", _t(B, T, I))
    eg = EGraph()
    ctx = _Ctx(eg, {})
    xe = eg.add_term(x)
    pe = eg.add_term(Param("p", _t(B, T, I)))
    node = ENode("mul", (xe, pe), ())
    s4 = ("s", (B, T, 4), (("p_a", "vec", 0),), frozenset())
    s8 = ("s", (B, T, 8), (("p_b", "vec", 0),), frozenset())
    # w0 != w1, neither width-1 — the pair is dropped by every arm
    assert _merge_sites(ctx, node, (B, T, I), (s4,), (s8,)) == []
    assert _merge_sites(ctx, node, (B, T, I), (s8,), (s4,)) == []
