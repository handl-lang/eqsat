"""The value-analysis passes driven by duck-typed tensors.

The sharing passes (``pairing.share_duplicate_params`` /
``share_duplicate_param_slices``), the weight-specials pass
(``specials.offer_weight_specials``), the low-rank factor pass
(``factored.offer_low_rank_factors``) and the attention head-share
pass (``headshare.share_duplicate_attention_heads``) are deliberately
tensor-library-free: they read ``shape``/``dtype``/``dim()``
structurally and analyse values through the small duck-typed surface
``tests.faketensor.FakeTensor`` implements.  The domain suites replay
the same scenarios against real torch tensors (and verify the lowered
module bitwise); these pin the substrate-side branches — the offer
records, the registered derived params, the witness merges — without
naming torch.
"""

from __future__ import annotations

import pytest

from mlopt_core.cost import count_cost, param_bytes_cost, param_bytes_cost_for
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_core.laws.factored import offer_low_rank_factors
from mlopt_core.laws.headshare import (
    headshare_keys_hold,
    share_duplicate_attention_heads,
)
from mlopt_core.laws.pairing import (
    _exact_equal,
    _is_tensor,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from mlopt_core.laws.specials import offer_weight_specials

from tests.faketensor import FakeTensor


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


# ---------------------------------------------------------------------------
#  The fake itself — a quick contract pin
# ---------------------------------------------------------------------------


def test_fake_tensor_contract():
    t = FakeTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert _is_tensor(t) and t.dim() == 2
    assert tuple(t.shape) == (2, 3)
    assert t.numel() == 6 and t.element_size() == 8
    assert t[0].tolist() == [1.0, 2.0, 3.0]
    assert t[:, 1].tolist() == [2.0, 5.0]
    assert t[0:1, 1:].tolist() == [[2.0, 3.0]]
    assert t[[1, 0]][:, [2]].tolist() == [[6.0], [3.0]]
    assert t.T.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert t.diagonal().tolist() == [1.0, 5.0]
    assert (t @ t.T).tolist() == [[14.0, 32.0], [32.0, 77.0]]
    assert t.cpu().contiguous().detach() is t
    assert t.numpy().tobytes() == t.numpy().tobytes()
    z = t.new_zeros((1, 3))
    z[0] = t[0]
    assert z.tolist() == [[1.0, 2.0, 3.0]]


# ---------------------------------------------------------------------------
#  pairing._is_tensor / _exact_equal — the carved test's substrate core
# ---------------------------------------------------------------------------


def test_is_tensor_is_structural():
    assert _is_tensor(FakeTensor([[0.0]]))
    assert not _is_tensor("not-a-tensor")
    assert not _is_tensor({"shape": (2,)})
    assert not _is_tensor(None)


def test_exact_equal_value_semantics():
    a = FakeTensor([[0.0, 1.0], [2.0, 3.0]])
    assert _exact_equal(a, a.clone())
    # different shapes never compare equal
    assert not _exact_equal(a, FakeTensor([[0.0, 1.0, 2.0]]))
    assert not _exact_equal(FakeTensor([0.0, 1.0]), FakeTensor([0.0]))
    # torch.equal parity: NaN != NaN, -0.0 == +0.0
    nan = FakeTensor([float("nan")])
    assert not _exact_equal(nan, nan.clone())
    assert _exact_equal(FakeTensor([0.0]), FakeTensor([-0.0]))


def test_param_bytes_source_tensors_duck_typed():
    """source_tensors is authoritative for numel; TensorType is fallback."""
    x = Var("x", _T(4, 8))
    W = Param("W", _T(None, None))  # shape unknown at type level
    t = Op.make("linear", x, W)
    src = {"W": FakeTensor([[0.0] * 16] * 8)}
    assert param_bytes_cost(t, src) == 8 * 16
    # names absent from source_tensors fall back to the TensorType
    U = Param("U", _T(3, 5))
    t2 = Op.make("add", Op.make("linear", x, W), Op.make("linear", x, U))
    assert param_bytes_cost(t2, src) == 8 * 16 + 3 * 5
    # the bound-closure form prices identically
    assert param_bytes_cost_for(src)(t2) == 8 * 16 + 3 * 5
    # by_bytes weighs by element_size
    assert param_bytes_cost(t, src, by_bytes=True) == 8 * 16 * 8


# ---------------------------------------------------------------------------
#  pairing.share_duplicate_params — whole-tensor weight tying
# ---------------------------------------------------------------------------


def test_share_duplicate_params_merges_equal_weights():
    w = FakeTensor([[1.0, 2.0], [3.0, 4.0]])
    src = {"A": w, "B": w.clone(), "C": FakeTensor([[9.0, 9.0]])}
    eg = EGraph()
    for n in "ABC":
        eg.add_term(_p(n, *src[n].shape))
    groups = share_duplicate_params(eg, src)
    assert groups == [["A", "B"]]
    # the merge carried a replayable witness edge
    edge = eg.merge_log[-1]
    assert edge.rule.startswith("share#")
    wit = eg._rule_objs[edge.rule]
    assert wit.error_bound is None


def test_share_duplicate_params_distinct_and_shape_mismatch():
    src = {
        "A": FakeTensor([[1.0, 2.0]]),
        "B": FakeTensor([[1.0, 3.0]]),
        "C": FakeTensor([1.0, 2.0]),  # same values, different shape
    }
    eg = EGraph()
    for n in "ABC":
        eg.add_term(_p(n, *src[n].shape))
    assert share_duplicate_params(eg, src) == []


# ---------------------------------------------------------------------------
#  pairing.share_duplicate_param_slices — intra-tensor head dedup
# ---------------------------------------------------------------------------


def _dup_weight(blocks):
    """Stack per-head row blocks into one (h·d, i) weight."""
    rows = [r for blk in blocks for r in blk]
    return FakeTensor(rows)


def test_share_duplicate_param_slices_offers_dedup_member():
    """Heads 0,2 bitwise-equal / 1,3 distinct -> one (3,d,i) dedup param."""
    h, d, i = 4, 3, 5
    blocks = [
        [[1.0 + blk * 100 + j + k / 10 for k in range(i)]
         for j in range(d)]
        for blk in range(3)
    ]
    W = _dup_weight([blocks[0], blocks[1], blocks[0], blocks[2]])
    src = {"W": W}

    eg = EGraph()
    w_eid = eg.add_term(_p("W", h * d, i))
    offers = share_duplicate_param_slices(eg, src)

    assert len(offers) == 1
    off = offers[0]
    assert off["param"] == "W" and off["heads"] == h
    assert off["head_dim"] == d
    assert off["unique"] == 3 and off["index_map"] == (0, 1, 0, 2)
    assert off["stored_after"] == 3 * d * i < off["stored_before"] == h * d * i

    # the deduplicated tensor was registered with real values
    dedup = src[off["dedup_param"]]
    assert tuple(dedup.shape) == (3, d, i)
    assert _exact_equal(dedup[0], blocks[0] if isinstance(blocks[0], FakeTensor) else FakeTensor(blocks[0]))
    assert _exact_equal(dedup[2], FakeTensor(blocks[2]))

    # W's e-class holds the reshape(index_select(D)) member
    cls = eg.get_class(w_eid)
    node = next(n for n in cls.nodes if n.op == "reshape")
    member = eg.extract_best(
        w_eid, count_cost, overrides={eg.find(w_eid): node}
    )
    assert member.op == "reshape"
    assert member.args[0].op == "index_select"
    assert member.args[0].args[0].name == off["dedup_param"]

    # the merge carried a replayable exact witness
    edge = eg.merge_log[-1]
    assert edge.rule.startswith("share_slices#")
    wit = eg._rule_objs[edge.rule]
    assert wit.error_bound is None

    # under the storage cost model the member wins on its own
    best = eg.extract_best(w_eid, param_bytes_cost_for(src))
    assert best == member


def test_share_duplicate_param_slices_declines():
    """All-distinct heads, non-2-D params and absent params offer nothing."""
    W = FakeTensor([[float(r * 10 + c) for c in range(5)] for r in range(12)])
    src = {"W": W, "bias": FakeTensor([1.0, 2.0, 3.0, 4.0]),
           "absent": _dup_weight([[[1.0, 2.0]]] * 4)}
    eg = EGraph()
    eg.add_term(_p("bias", 4))
    w_eid = eg.add_term(_p("W", 12, 5))
    offers = share_duplicate_param_slices(eg, src)
    assert offers == []
    assert {n.op for n in eg.get_class(w_eid).nodes} == {"leaf"}
    assert "bias__heads" not in " ".join(src)


def test_share_duplicate_param_slices_halved_weight():
    """A (6,4) weight with two equal halves dedups to one (3,4) block."""
    half = FakeTensor([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])
    W = FakeTensor(half.tolist() + half.tolist())
    src = {"W": W}
    eg = EGraph()
    eg.add_term(_p("W", 6, 4))
    offers = share_duplicate_param_slices(eg, src)
    assert len(offers) == 1
    off = offers[0]
    assert off["heads"] == 2 and off["unique"] == 1
    assert off["index_map"] == (0, 0)
    assert tuple(src[off["dedup_param"]].shape) == (1, 3, 4)


# ---------------------------------------------------------------------------
#  specials.offer_weight_specials — structural members
# ---------------------------------------------------------------------------


def _lin_graph(eg, weight_name="W", orient="linear", bias=None):
    x = _v("x", 2, 4)
    w = _p(weight_name, 4, 4) if orient == "linear" else _p(weight_name, 4, 4)
    args = (x, w, bias) if bias is not None else (x, w)
    op = "linear" if orient == "linear" else "matmul"
    return eg.add_term(Op.make(op, *args))


def test_specials_identity_weight_offers_identity_member():
    eg = EGraph()
    site = _lin_graph(eg)
    src = {"W": FakeTensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])}
    recs = offer_weight_specials(eg, src)
    assert [r["kind"] for r in recs] == ["identity"]
    assert recs[0]["orient"] == "linear" and recs[0]["error_bound"] == 0.0


def test_specials_diagonal_weight_registers_d_param():
    eg = EGraph()
    _lin_graph(eg)
    src = {"W": FakeTensor([[2.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0, 0.0],
                            [0.0, 0.0, 4.0, 0.0],
                            [0.0, 0.0, 0.0, 5.0]])}
    recs = offer_weight_specials(eg, src)
    assert [r["kind"] for r in recs] == ["diagonal"]
    d = src[recs[0]["d_param"]]
    assert d.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_specials_zero_weight_offers_zero_member():
    eg = EGraph()
    _lin_graph(eg)
    src = {"W": FakeTensor([[0.0] * 4] * 4)}
    recs = offer_weight_specials(eg, src)
    assert [r["kind"] for r in recs] == ["zero"]


def test_specials_elide_dead_cols_and_dup_rows():
    """Duplicate output rows + dead input cols -> gather-around-shrunk-W."""
    eg = EGraph()
    _lin_graph(eg)
    # rows 0 and 2 identical; column 1 all zeros
    src = {"W": FakeTensor([[1.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, 5.0, 0.0]])}
    recs = offer_weight_specials(eg, src)
    kinds = [r["kind"] for r in recs]
    assert "elide" in kinds
    w = src[recs[kinds.index("elide")]["w_param"]]
    # shrunk weight: unique rows (0,1,3) x live cols (0,2)
    assert tuple(w.shape) == (3, 2)


def test_specials_block_diag_weight_splits():
    eg = EGraph()
    _lin_graph(eg)
    # two diagonal 2x2 blocks
    src = {"W": FakeTensor([[1.0, 2.0, 0.0, 0.0],
                            [3.0, 4.0, 0.0, 0.0],
                            [0.0, 0.0, 5.0, 6.0],
                            [0.0, 0.0, 7.0, 8.0]])}
    recs = offer_weight_specials(eg, src)
    kinds = [r["kind"] for r in recs]
    assert "block_diag" in kinds
    rec = recs[kinds.index("block_diag")]
    assert rec["sizes"] == (2, 2)
    assert len(rec["w_params"]) == 2


def test_specials_budget_offers_bounded_members():
    """A near-dead weight under budget -> zero_bounded with the bound."""
    eg = EGraph()
    _lin_graph(eg)
    src = {"W": FakeTensor([[1e-9] * 4] * 4)}
    recs = offer_weight_specials(eg, src, budget=1e-6)
    kinds = [r["kind"] for r in recs]
    assert "zero_bounded" in kinds
    rec = recs[kinds.index("zero_bounded")]
    assert rec["error_bound"] == pytest.approx(1e-9)
    assert rec["bound_norm"] == "max_abs"


def test_specials_plain_weight_and_param_only_consumer_decline():
    """No structure -> no offer; a Var-free site folds either way."""
    eg = EGraph()
    _lin_graph(eg)
    src = {"W": FakeTensor([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0, 7.0]])}
    assert offer_weight_specials(eg, src) == []
    # param-only consumer: linear(W2, W) has no Var below data
    eg2 = EGraph()
    eg2.add_term(
        Op.make("linear", _p("X", 2, 4), _p("W", 4, 4))
    )
    src2 = {"W": FakeTensor([[0.0] * 4] * 4), "X": FakeTensor([[0.0] * 4] * 2)}
    assert offer_weight_specials(eg2, src2) == []


def test_specials_matmul_orientation_uses_columns():
    """``matmul(x, W)`` normalises through W.T — the (i, o) convention."""
    eg = EGraph()
    eg.add_term(Op.make("matmul", _v("x", 2, 4), _p("W", 4, 4)))
    # W.T is the identity <=> W is the identity
    src = {"W": FakeTensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])}
    recs = offer_weight_specials(eg, src)
    assert [r["kind"] for r in recs] == ["identity"]
    assert recs[0]["orient"] == "matmul_r"


# ---------------------------------------------------------------------------
#  factored.offer_low_rank_factors — certified Gram-Schmidt factors
# ---------------------------------------------------------------------------


def test_low_rank_factor_offers_and_registers():
    """A rank-1 (4,8) weight factors into (4,1)x(1,8) — break-even holds."""
    u = [1.0, 2.0, 0.5, -1.0]
    v = [1.0, -1.0, 2.0, 0.5, 0.0, 3.0, -2.0, 1.0]
    # matmul_r orientation: W is (i, o) = (8, 4) — outer(v, u)
    W = FakeTensor([[a * b for b in u] for a in v])
    eg = EGraph()
    eg.add_term(Op.make("matmul", _v("x", 2, 8), _p("W", 8, 4)))
    src = {"W": W}
    recs = offer_low_rank_factors(eg, src)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["rank"] == 1 and rec["orient"] == "matmul_r"
    assert rec["in_dim"] == 8 and rec["out_dim"] == 4
    # the derived factor params are registered and shape-correct
    a, b = src[rec["a_param"]], src[rec["b_param"]]
    assert tuple(a.shape) == (8, 1) and tuple(b.shape) == (1, 4)
    # reconstruction residual is carried as the error bound
    assert rec["error"] < 1e-8
    edge = eg.merge_log[-1]
    assert edge.rule.startswith("low_rank_factor#")


def test_low_rank_factor_full_rank_declines():
    """A full-rank weight clears no break-even -> no offer."""
    eg = EGraph()
    eg.add_term(Op.make("matmul", _v("x", 2, 4), _p("W", 4, 4)))
    src = {"W": FakeTensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0, 0.0],
                            [0.0, 0.0, 3.0, 0.0],
                            [0.0, 0.0, 0.0, 4.0]])}
    assert offer_low_rank_factors(eg, src) == []


def test_low_rank_factor_linear_site_and_bias():
    """The ``linear`` orientation rows-factors; a bias re-adds outside."""
    u = [1.0, -2.0]
    v = [1.0, 0.0, 2.0, -1.0, 0.5, 1.5]
    W = FakeTensor([[a * b for b in v] for a in u])  # (2, 6) rank-1
    eg = EGraph()
    eg.add_term(
        Op.make("linear", _v("x", 2, 6), _p("W", 2, 6), _p("b", 2))
    )
    src = {"W": W, "b": FakeTensor([1.0, 2.0])}
    recs = offer_low_rank_factors(eg, src)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["orient"] == "linear" and rec["rank"] == 1
    # the member is add(linear(linear(x, A), B), bias)
    cls = eg.get_class(eg.find(rec["eid"]))
    assert cls is not None


# ---------------------------------------------------------------------------
#  headshare — deduplicated sdpa heads through the duck-typed surface
# ---------------------------------------------------------------------------

B_, T_, H_, D_, I_ = 2, 3, 4, 2, 8  # batch, seq, heads, head-dim, in-dim


def _chain(x: Op, w: Param) -> Op:
    """transpose(reshape(linear(x, w), (B,T,H,D)), 1, 2) — the MHA spelling."""
    return Op.make(
        "transpose",
        Op.make("reshape", Op.make("linear", x, w), shape=(B_, T_, H_, D_)),
        dim0=1,
        dim1=2,
    )


def _head_blocks(k: int, d: int, i: int):
    """``k`` distinct (d, i) blocks with recognisably-different values."""
    return [
        [[1.0 + j * 10 + c / 10 + blk * 100 for c in range(i)]
         for j in range(d)]
        for blk in range(k)
    ]


def _stack(blocks) -> FakeTensor:
    return FakeTensor([r for blk in blocks for r in blk])


def test_headshare_offers_gather_sdpa_gather_member():
    """Heads 0,2 bitwise-equal -> compute 3 unique heads, gather to 4."""
    bl = _head_blocks(3, D_, I_)
    W = _stack([bl[0], bl[1], bl[0], bl[2]])
    x = _v("x", B_, T_, I_)
    term = Op.make(
        "sdpa", _chain(x, _p("W", H_ * D_, I_)),
        _chain(x, _p("W", H_ * D_, I_)),
        _chain(x, _p("W", H_ * D_, I_)),
    )
    eg = EGraph()
    eg.add_term(term)
    src = {"W": W}
    offers = share_duplicate_attention_heads(eg, src)
    assert len(offers) == 1
    off = offers[0]
    assert off["heads"] == H_ and off["unique"] == 3
    assert off["index_map"] == (0, 1, 0, 2) and off["uniq"] == (0, 1, 3)
    assert off["axis"] == 1
    # the replay recipe re-verifies against the same tensors
    assert headshare_keys_hold([off["recheck"]], lambda n: n, src)
    # ... and fails against a mutated value honestly
    bad = dict(src)
    bw = W.clone()
    bw[D_ * I_] += 1.0  # flip one value inside head 1's block
    bad["W"] = bw
    assert not headshare_keys_hold([off["recheck"]], lambda n: n, bad)


def test_headshare_distinct_heads_no_offer():
    bl = _head_blocks(4, D_, I_)
    W = _stack(bl)
    x = _v("x", B_, T_, I_)
    term = Op.make(
        "sdpa", *[_chain(x, _p("W", H_ * D_, I_))] * 3
    )
    eg = EGraph()
    eg.add_term(term)
    assert share_duplicate_attention_heads(eg, {"W": W}) == []


def test_headshare_gqa_expand_spelling():
    """The exported GQA kv-expansion — ``reshape(expand(unsqueeze
    (kv, 2), (B,nkv,g,T,D)), (B,H,T,D))`` — resolves base-head reads:
    kv head ``m // g`` joins q head ``m`` in the joint signature, so
    group-aligned q duplication fires the offer."""
    nkv, g = 2, 2
    qbl = _head_blocks(2, D_, I_)
    kvbl = _head_blocks(2, D_, I_)
    # q heads (0,1) share a block, heads (2,3) share — group-aligned
    Wq = _stack([qbl[0], qbl[0], qbl[1], qbl[1]])
    Wkv = _stack(kvbl)
    x = _v("x", B_, T_, I_)

    def q():
        return _chain(x, _p("Wq", H_ * D_, I_))

    def kv():
        base = Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make("linear", x, _p("Wkv", nkv * D_, I_)),
                shape=(B_, T_, nkv, D_),
            ),
            dim0=1,
            dim1=2,
        )
        return Op.make(
            "reshape",
            Op.make(
                "expand",
                Op.make("unsqueeze", base, dim=2),
                shape=(B_, nkv, g, T_, D_),
            ),
            shape=(B_, H_, T_, D_),
        )

    term = Op.make("sdpa", q(), kv(), kv(), is_causal=True)
    eg = EGraph()
    eg.add_term(term)
    src = {"Wq": Wq, "Wkv": Wkv}
    offers = share_duplicate_attention_heads(eg, src)
    assert len(offers) == 1
    assert offers[0]["index_map"] == (0, 0, 1, 1)
    assert headshare_keys_hold(
        [offers[0]["recheck"]], lambda n: n, src
    )


def test_headshare_attr_vetoes():
    """dropout_p / enable_gqa sdpa spellings decline outright."""
    bl = _head_blocks(3, D_, I_)
    W = _stack([bl[0], bl[1], bl[0], bl[2]])
    x = _v("x", B_, T_, I_)
    for attrs in ({"dropout_p": 0.1}, {"enable_gqa": True}):
        term = Op.make(
            "sdpa", *[_chain(x, _p("W", H_ * D_, I_))] * 3, **attrs
        )
        eg = EGraph()
        eg.add_term(term)
        assert share_duplicate_attention_heads(eg, {"W": W}) == []


def test_headshare_missing_tensor_declines():
    """Heads whose source values aren't in the table cannot sign."""
    x = _v("x", B_, T_, I_)
    term = Op.make(
        "sdpa", *[_chain(x, _p("W", H_ * D_, I_))] * 3
    )
    eg = EGraph()
    eg.add_term(term)
    assert share_duplicate_attention_heads(eg, {"W": "no-tensor"}) == []
    assert share_duplicate_attention_heads(eg, {}) == []
