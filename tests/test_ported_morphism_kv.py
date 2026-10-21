"""Substrate-pure tests ported from ``graphopt:tests/test_morphism_kv.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import mlopt_orchestrator.morphisms as M

import mlopt_orchestrator.morphisms_kv as K

from mlopt_core.ir import IR, Op, Param, TensorType, Var


def test_kv_named():
    """Token substring matching."""
    assert K._kv_named("p_attn_k_proj_weight", ("k_proj",))
    assert not K._kv_named("p_q_proj_weight", ("k_proj", "v_proj"))

def test_kv_proj_sites_orientations():
    """linear / matmul-right / qkv sites; everything else skipped."""
    x = Var("x", TensorType((8, 16)))
    wk = Param("p_k_proj_weight", TensorType((8, 16)))
    wv = Param("p_v_proj_weight", TensorType((8, 16)))
    wq = Param("p_q_proj_weight", TensorType((8, 16)))
    wqkv = Param("p_qkv_proj_weight", TensorType((24, 16)))
    wqkv_bad = Param("p_qkv_proj_weight", TensorType((22, 16)))
    wqkv_noshp = Param("p_qkv_proj_weight", TensorType((None, 16)))
    wp = Param("p_w", TensorType((16, 8)))
    wb = Param("p_b", TensorType((8,)))
    expr = Op.make("add", wk, wv)

    # linear sites: kv tokens only, Param weights only.
    body = Op.make(
        "add",
        Op.make("linear", x, wk),
        Op.make("linear", x, wv, wb),
    )
    ss = K._kv_proj_sites(body, "b0", K._KV_TOKENS, ())
    assert len(ss) == 2 and all(s.orient == "linear" for s in ss)
    assert all(s.data is x for s in ss)

    # q_proj is not a token; compound weight terms are not sites.
    body = Op.make(
        "add",
        Op.make("linear", x, wq),
        Op.make("linear", x, expr),
    )
    assert K._kv_proj_sites(body, "b0", K._KV_TOKENS, ()) == []

    # param-only data operand is not a projection site.
    body = Op.make("linear", Param("p_y", TensorType((8, 16))), wk)
    assert K._kv_proj_sites(body, "b0", K._KV_TOKENS, ()) == []

    # qkv: divisible thirds gives two section sites; bad rows skipped;
    # unknown rows skipped.
    body = Op.make("linear", x, wqkv)
    ss = K._kv_proj_sites(body, "b0", (), K._QKV_TOKENS)
    assert [s.span for s in ss] == [(8, 8), (16, 8)]
    assert all(s.orient == "qkv" for s in ss)
    body = Op.make("linear", x, wqkv_bad)
    assert K._kv_proj_sites(body, "b0", (), K._QKV_TOKENS) == []
    body = Op.make("linear", x, wqkv_noshp)
    assert K._kv_proj_sites(body, "b0", (), K._QKV_TOKENS) == []

    # matmul: right-side Param weight is a site; left-side is not.
    body = Op.make("matmul", x, wp)
    ss = K._kv_proj_sites(body, "b0", ("w",), ())
    assert [s.orient for s in ss] == ["matmul_r"]
    body = Op.make("matmul", wp, x)
    assert K._kv_proj_sites(body, "b0", ("w",), ()) == []
    body = Op.make("matmul", x, Var("y", TensorType((16, 8))))
    assert K._kv_proj_sites(body, "b0", ("y",), ()) == []

    # odd arities pass through untouched.
    odd_lin = Op("linear", (x,), {})
    odd_mm = Op("matmul", (x, wp, wb), {})
    assert K._kv_proj_sites(odd_lin, "b0", K._KV_TOKENS, ()) == []
    assert K._kv_proj_sites(odd_mm, "b0", ("w",), ()) == []

def test_latent_groups():
    """Sites group by their shared data term; singletons drop."""
    x = Var("x", TensorType((8, 16)))
    y = Var("y", TensorType((8, 16)))
    wk = Param("p_k_proj_weight", TensorType((8, 16)))
    wv = Param("p_v_proj_weight", TensorType((8, 16)))
    w2 = Param("p_k2_proj_weight", TensorType((8, 16)))
    body = Op.make(
        "add",
        Op.make("linear", x, wk),
        Op.make(
            "add", Op.make("linear", x, wv), Op.make("linear", y, w2)
        ),
    )
    groups = K._latent_groups({"b": body}, K._KV_TOKENS, ())
    # The y-sited projection is a singleton — only the x group stays.
    ((data, ss),) = groups.items()
    assert data is x and len(ss) == 2

def test_replace_nodes_and_add_chain():
    """Node-keyed rebuild + the add fold."""
    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    a = Op.make("linear", x, w)
    b = Op.make("silu", a)
    new = Op.make("matmul", x, w)
    got = K._replace_nodes(b, {a: new})
    assert got == Op.make("silu", new)
    assert K._replace_nodes(x, {a: new}) is x
    assert K._add_chain([a]) is a
    two = K._add_chain([a, b])
    assert two == Op.make("add", a, b)
    three = K._add_chain([a, b, x])
    assert three == Op.make("add", a, Op.make("add", b, x))

def test_kv_rewrite_orientations():
    """The latent-factored substitutions per site spelling."""
    x = Var("x", TensorType((8, 16)))
    ut = Param("p_kv_latent_ut", TensorType((16, 4)))
    C = Op.make("matmul", x, ut)
    wk = Param("p_k_proj_weight", TensorType((8, 16)))
    wb = Param("p_b", TensorType((8,)))
    n_lin = Op.make("linear", x, wk)
    n_bias = Op.make("linear", x, wk, wb)
    wr = Param("p_wr_weight", TensorType((16, 8)))
    n_mm = Op.make("matmul", x, wr)
    wqkv = Param("p_qkv_proj_weight", TensorType((24, 16)))
    n_qkv = Op.make("linear", x, wqkv)
    n_qkvb = Op.make(
        "linear", x, wqkv, Param("p_qb", TensorType((24,)))
    )

    def drv(shp, nm):
        return Param(nm, TensorType(shp))

    s_lin = K._KVSite("b", n_lin, x, wk, "linear")
    s_bias = K._KVSite("b", n_bias, x, wk, "linear")
    s_mm = K._KVSite("b", n_mm, x, wr, "matmul_r")
    derived = {
        s_lin: drv((8, 4), "p_kv_latent_d0"),
        s_bias: drv((8, 4), "p_kv_latent_db"),
        s_mm: drv((4, 8), "p_kv_latent_d1"),
    }
    body = Op.make(
        "add", n_lin, Op.make("add", n_mm, Op.make("silu", x))
    )
    out = K._kv_rewrite(body, [s_lin, s_mm], C, derived, {}, {}, {})
    lin_new = Op.make("linear", C, derived[s_lin])
    mm_new = Op.make("matmul", C, derived[s_mm])
    ops = M._iter_ops(out)
    assert lin_new in ops and mm_new in ops
    # the silu arm is untouched
    assert Op.make("silu", x) in ops
    # bias rides the recovery untouched
    out2 = K._kv_rewrite(n_bias, [s_bias], C, derived, {}, {}, {})
    assert out2.args[2] is wb and out2.args[1] is derived[s_bias]

    # qkv: two section sites -> one concat node, derived leaves only.
    ss = [
        K._KVSite("b", n_qkv, x, wqkv, "qkv", (8, 8)),
        K._KVSite("b", n_qkv, x, wqkv, "qkv", (16, 8)),
    ]
    dq = {s: drv((8, 4), f"p_d{i}") for i, s in enumerate(ss)}
    out3 = K._kv_rewrite(
        n_qkv,
        ss,
        C,
        dq,
        {n_qkv: drv((8, 16), "p_wq")},
        {},
        {},
    )
    assert out3.op == "concat" and len(out3.args) == 3
    assert out3.args[0].op == "linear"
    assert all(
        a.op == "linear" and a.args[0] is C for a in out3.args[1:]
    )
    ssb = [
        K._KVSite("b", n_qkvb, x, wqkv, "qkv", sp)
        for sp in ((16, 8), (8, 8))
    ]
    dqb = {s: drv((8, 4), f"p_b{i}") for i, s in enumerate(ssb)}
    sb = {s: drv((8,), f"p_sb{i}") for i, s in enumerate(ssb)}
    out4 = K._kv_rewrite(
        n_qkvb,
        ssb,
        C,
        dqb,
        {n_qkvb: drv((8, 16), "p_wqb")},
        {n_qkvb: drv((8,), "p_qbb")},
        sb,
    )
    assert out4.op == "concat" and len(out4.args[0].args) == 3
    assert all(len(a.args) == 3 for a in out4.args)

def test_saturate_offer_kwargs():
    """The third offer element carries witness kwargs."""
    from mlopt_core.laws import RuleSet

    x = Var("x", TensorType((8, 16)))
    w = Param("p_w", TensorType((16, 16)))
    t = Op.make("linear", x, w)
    t2 = Op.make("linear", x, Op.make("add", w, w))
    eg, eid = M._saturate(
        t,
        RuleSet("e", ()),
        max_iterations=2,
        max_enodes=100,
        symmetry_budget=None,
        offers=[
            (t2, "offer with kwargs", {"error_bound": 0.5, "note": "n"})
        ],
    )
    assert eg is not None and eid is not None

def test_morphisms_reexport():
    """The sibling-module law re-exports through ``morphisms``."""
    assert M.KVLatentShare is K.KVLatentShare

def test_package_getattr():
    """The package-level lazy export resolves the KV law."""
    import mlopt_orchestrator as O

    assert O.KVLatentShare is K.KVLatentShare
