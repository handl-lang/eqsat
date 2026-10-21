"""Substrate-pure tests ported from ``graphopt:tests/test_attention_laws.py``.

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
    ATTENTION_RULES,
    NATURALITY_SCALAR_REV,
    preset,
)

from mlopt_core.laws.attention import (
    _check_left_scale,
    _check_linear_mm_absorb,
    _check_linear_mm_absorb_bias,
    _check_linear_out_scale,
    _check_rope_cat_compose,
    _check_rope_scale,
    _check_uniform,
    _half_bounds,
    _is_uniform,
    _rope_axis,
    _rope_cat,
    _rope_rh,
    _to_end,
)

from mlopt_core.laws.ruleset import ATTENTION, DEFAULT, PRESETS

from mlopt_core.laws.tensor import SDPA_EXPAND_RULES


_END = 1 << 62  # the int64 sentinel exports spell for x[..., h:]

def _v(name: str, *shape: int | None) -> Var:
    return Var(name, TensorType(tuple(shape)))

def _p(name: str, *shape: int | None) -> Param:
    return Param(name, TensorType(tuple(shape)))

def _saturate(term, rules=None, iters=8, max_nodes=200_000):
    """Build an e-graph over *term* and saturate with ATTENTION_RULES."""
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        ATTENTION_RULES if rules is None else rules,
        eid,
        max_iterations=iters,
        max_nodes=max_nodes,
    )
    return eg, eid

def _fired(eg, name):
    return eg.rule_fires.get(name, 0) > 0

def _check_declined(eg, name):
    """The rule must not have fired — no substitution passed check."""
    assert not _fired(eg, name)

def _sl(t, dim, a, e, step=None):
    """Concrete slice term; ``step`` set only when given (export form)."""
    kw = {"dim": dim, "start": a, "end": e}
    if step is not None:
        kw["step"] = step
    return Op.make("slice", t, **kw)

def _cat_rope(x, c, s, h, dim=3, cdim=-1, end=None):
    """Concrete half-split cat-rope: cat(x1·c - x2·s, x2·c + x1·s).

    ``dim`` defaults to the *positive* last-axis index — exports emit
    ``slice(dim=3)`` while ``concat`` keeps ``dim=-1`` (the slice
    binding does not normalise negative dims, so terms must spell the
    exported form).
    """
    e = _END if end is None else end
    x1 = _sl(x, dim, 0, h)
    x2 = _sl(x, dim, h, e)
    return Op.make(
        "concat",
        Op.make("sub", Op.make("mul", x1, c), Op.make("mul", x2, s)),
        Op.make("add", Op.make("mul", x2, c), Op.make("mul", x1, s)),
        dim=cdim,
    )

def _r_by_name(name):
    return next(r for r in ATTENTION_RULES if r.name == name)

def test_is_uniform():
    x2 = _v("t", 4, 5)
    assert _is_uniform(Const(0.5))
    assert _is_uniform(_v("u", 1, 1, 1))
    assert not _is_uniform(_v("u", 2))
    assert not _is_uniform(
        Op.make("einsum", x2, x2, equation="ij,jk->ik")
    )  # shape unknown

def test_to_end():
    assert _to_end(_END, 16)
    assert _to_end(16, 16)
    assert not _to_end(8, 16)
    assert not _to_end("e", 16)
    assert not _to_end(None, 16)

def test_rope_axis():
    b = {
        "$attr:I1D": -1,
        "$attr:I2D": 3,
        "$attr:O1D": -1,
        "$attr:O2D": 3,
        "$attr:CD1": -1,
        "$attr:CD2": 3,
    }
    assert _rope_axis(b, 4) == 3
    b["$attr:O2D"] = 1  # a different axis — not one shared rope axis
    assert _rope_axis(b, 4) is None
    b["$attr:O2D"] = "D"  # unbound metavar — not an int
    assert _rope_axis(b, 4) is None

def test_half_bounds():
    b = {
        "$attr:I1A": 0,
        "$attr:I1E": 8,
        "$attr:I2A": 8,
        "$attr:I2E": _END,
    }
    assert _half_bounds(b, "I1", "I2", 16) == 8
    b["$attr:I1A"] = 2  # does not start at 0
    assert _half_bounds(b, "I1", "I2", 16) == 0
    b["$attr:I1A"] = 0
    b["$attr:I2A"] = 9  # not contiguous
    assert _half_bounds(b, "I1", "I2", 16) == 0
    b["$attr:I2A"] = 8
    b["$attr:I2E"] = 10  # does not cover the axis
    assert _half_bounds(b, "I1", "I2", 16) == 0
    b["$attr:I2E"] = _END
    assert _half_bounds(b, "I1", "I2", 15) == 0  # odd extent
    b["$attr:I1E"] = "E"  # unbound metavar
    assert _half_bounds(b, "I1", "I2", 16) == 0

def test_check_rope_scale():
    x = _v("x", 2, 3, 8, 16)
    # scalar — always fine
    assert _check_rope_scale({"a": Const(0.5), "x": x})
    # per-row (…,1) broadcasting to x's shape
    assert _check_rope_scale({"a": _v("a", 2, 3, 8, 1), "x": x})
    # per-channel table — pair-varying, wrong
    assert not _check_rope_scale({"a": _v("a", 2, 3, 8, 16), "x": x})
    # (T,1) row table broadcasts onto x's last dims — accepted
    assert _check_rope_scale({"a": _v("a", 8, 1), "x": x})
    # last-dim-1 but the row dim mismatches — decline
    assert not _check_rope_scale({"a": _v("a", 5, 1), "x": x})
    # x shape unrecoverable — decline for non-scalar a
    y = Op.make("einsum", x, x, equation="ijkl,lm->ijkm")
    assert not _check_rope_scale({"a": _v("a", 1, 1, 1, 1), "x": y})
    # a shape unrecoverable — decline
    assert not _check_rope_scale({"a": y, "x": x})

def test_check_left_scale():
    a = _v("A", 2, 4, 5)
    b = _v("B", 2, 5, 6)
    assert _check_left_scale({"s": Const(0.5), "A": a, "B": b})
    assert _check_left_scale({"s": _v("s", 2, 4, 1), "A": a, "B": b})
    # per-KEY-column scale: (s·A)@B ≠ s·(A@B) — decline
    assert not _check_left_scale(
        {"s": _v("s", 2, 6, 1), "A": a, "B": b}
    )
    # a rank-(…,5) factor is not a row scale — decline
    assert not _check_left_scale({"s": _v("s", 5), "A": a, "B": b})
    # ill-typed product — decline
    bad = _v("B2", 2, 7, 6)
    assert not _check_left_scale({"s": Const(0.5), "A": a, "B": bad})
    # scalar A — the product is provably ill-typed too
    assert not _check_left_scale(
        {"s": Const(0.5), "A": Const(1), "B": b}
    )
    # unrecoverable factor shape — decline
    e = Op.make("einsum", a, b, equation="ijkl,lmn->ijkn")
    assert not _check_left_scale({"s": e, "A": a, "B": b})

def test_check_linear_mm_absorb():
    w = _p("W", 8, 6)
    r = _p("R", 8, 8)
    assert _check_linear_mm_absorb({"W": w, "R": r})
    # wildcards pass — only provable mismatches veto
    assert _check_linear_mm_absorb({"W": _p("W", None, 6), "R": r})
    assert _check_linear_mm_absorb({"W": w, "R": _p("R", None, 8)})
    # contraction mismatch
    assert not _check_linear_mm_absorb({"W": w, "R": _p("R", 9, 8)})
    # rank-3 per-position table — F.linear cannot take a batched weight
    assert not _check_linear_mm_absorb({"W": w, "R": _p("R", 4, 8, 8)})
    # rank ≠ 2 on either side
    assert not _check_linear_mm_absorb({"W": _p("W", 8, 6, 1), "R": r})
    e = Op.make("einsum", w, r, equation="ij,jk->ik")
    assert not _check_linear_mm_absorb({"W": e, "R": r})
    assert not _check_linear_mm_absorb({"W": w, "R": e})

def test_check_linear_mm_absorb_bias():
    w, r, b = _p("W", 8, 6), _p("R", 8, 8), _p("b", 8)
    assert _check_linear_mm_absorb_bias({"W": w, "R": r, "b": b})
    assert _check_linear_mm_absorb_bias(
        {"W": w, "R": r, "b": _p("b", None)}
    )
    # bias must be the (o,) vector of W's out dim
    assert not _check_linear_mm_absorb_bias(
        {"W": w, "R": r, "b": _p("b", 9)}
    )
    assert not _check_linear_mm_absorb_bias(
        {"W": w, "R": r, "b": _p("b", 8, 1)}
    )
    e = Op.make("einsum", b, b, equation="i,i->i")
    assert not _check_linear_mm_absorb_bias({"W": w, "R": r, "b": e})
    # base check fails propagate
    assert not _check_linear_mm_absorb_bias(
        {"W": w, "R": _p("R", 4, 8, 8), "b": b}
    )

def test_check_linear_out_scale():
    assert _check_linear_out_scale({"s": Const(2.0)})
    assert _check_linear_out_scale({"s": _v("s", 1, 1)})
    assert not _check_linear_out_scale({"s": _v("s", 8)})
    assert not _check_linear_out_scale({"s": _v("s", 2, 8, 1)})

def test_check_rope_cat_compose_unit():
    """Direct bound-level coverage of the compose check's arms."""
    x = _v("x", 2, 3, 8, 16)
    c = _v("c", 1, 1, 8, 8)
    s = _v("s", 1, 1, 8, 8)
    base = {
        "x": x,
        "C1": c,
        "C2": _v("c2", 1, 1, 8, 8),
        "S1": s,
        "S2": _v("s2", 1, 1, 8, 8),
        "$attr:I1D": -1,
        "$attr:I2D": 3,
        "$attr:O1D": -1,
        "$attr:O2D": 3,
        "$attr:CD1": -1,
        "$attr:CD2": 3,
        "$attr:I1A": 0,
        "$attr:I1E": 8,
        "$attr:I2A": 8,
        "$attr:I2E": _END,
        "$attr:O1A": 0,
        "$attr:O1E": 8,
        "$attr:O2A": 8,
        "$attr:O2E": _END,
    }
    assert _check_rope_cat_compose(dict(base))
    b = dict(base, x=Const(0))  # ()-shaped x — no axis
    assert not _check_rope_cat_compose(b)
    b = dict(base, x=Op.make("einsum", x, x, equation="i,i->i"))
    assert not _check_rope_cat_compose(b)  # unrecoverable x shape
    b = dict(base)
    b["$attr:O1D"] = 1  # axes disagree
    assert not _check_rope_cat_compose(b)
    b = dict(base, x=_v("x", 2, 3, 8, None))  # extent unknown
    assert not _check_rope_cat_compose(b)
    b = dict(base)
    b["$attr:I1E"] = 7  # inner split not the half
    assert not _check_rope_cat_compose(b)
    b = dict(base)
    b["$attr:O2A"] = 7  # outer cut off the boundary
    assert not _check_rope_cat_compose(b)
    b = dict(base, C1=Op.make("einsum", c, c, equation="i,i->i"))
    assert not _check_rope_cat_compose(b)  # c1 shape unknown
    b = dict(base, C2=_v("c2", 8, 8))  # different broadcast shape
    assert not _check_rope_cat_compose(b)
    b = dict(base, S1=Op.make("einsum", s, s, equation="i,i->i"))
    assert not _check_rope_cat_compose(b)
    b = dict(base, S2=_v("s2", 8, 8))
    assert not _check_rope_cat_compose(b)

def test_check_uniform():
    assert _check_uniform({"a": Const(1.0)})
    assert not _check_uniform({"a": _v("a", 3)})

def test_rope_cat_compose_declines():
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c1, s1 = _p("c1", 1, 1, T, dh), _p("s1", 1, 1, T, dh)
    c2, s2 = _p("c2", 1, 1, T, dh), _p("s2", 1, 1, T, dh)
    rule = _r_by_name("rope_cat_compose")

    # Outer slices cut off the concat boundary (h+1 instead of h).
    inner = _cat_rope(x, c1, s1, dh)
    o1 = _sl(inner, 3, 0, dh + 1)
    o2 = _sl(inner, 3, dh + 1, _END)
    bad = Op.make(
        "concat",
        Op.make("sub", Op.make("mul", o1, c2), Op.make("mul", o2, s2)),
        Op.make("add", Op.make("mul", o2, c2), Op.make("mul", o1, s2)),
        dim=-1,
    )
    eg, _ = _saturate(bad, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

    # Outer slices on a different axis than the inner concat's.
    o1 = _sl(inner, 2, 0, T)
    o2 = _sl(inner, 2, T, _END)
    bad = Op.make(
        "concat",
        Op.make("sub", Op.make("mul", o1, c2), Op.make("mul", o2, s2)),
        Op.make("add", Op.make("mul", o2, c2), Op.make("mul", o1, s2)),
        dim=-1,
    )
    eg, _ = _saturate(bad, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

    # Factor tables broadcast differently — composition unsound.
    c2_flat = _p("c2f", T, dh)  # (T,h) vs c1's (1,1,T,h)
    lhs = _cat_rope(inner, c2_flat, s2, dh)
    eg, _ = _saturate(lhs, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

    # Strided slices are a different rope spelling — the unstrided
    # pattern deliberately does not match (attr key-sets differ).
    st1 = _sl(x, 3, 0, _END, step=2)
    st2 = _sl(x, 3, 1, _END, step=2)
    st_rope = Op.make(
        "stack",
        Op.make(
            "sub", Op.make("mul", st1, c1), Op.make("mul", st2, s1)
        ),
        Op.make(
            "add", Op.make("mul", st2, c1), Op.make("mul", st1, s1)
        ),
        dim=-1,
    )
    eg, _ = _saturate(st_rope, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

def test_rope_cat_scale_declines_per_channel():
    """A per-channel factor (…,2h) breaks the rotary pairs — decline."""
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c, s = _p("c", 1, 1, T, dh), _p("s", 1, 1, T, dh)
    a = _p("a", 1, 1, 1, 2 * dh)  # pair-varying — not commutable
    lhs = _cat_rope(Op.make("mul", x, a), c, s, dh)
    eg, _ = _saturate(lhs, rules=[_r_by_name("rope_cat_scale_out")])
    _check_declined(eg, "rope_cat_scale_out")

def test_linear_mm_absorb_declines():
    o, i = 8, 6
    x, w = _v("x", 2, 8, i), _p("W", o, i)
    rule = _r_by_name("linear_mm_absorb")
    # Per-position rotation table — not a fixed weight; F.linear
    # cannot take a rank-3 weight either.
    r3 = _p("R", 8, o, o)
    eg, _ = _saturate(
        Op.make("matmul", Op.make("linear", x, w), r3), rules=[rule]
    )
    _check_declined(eg, "linear_mm_absorb")
    # Contraction mismatch — R does not consume the linear's out dim.
    rbad = _p("R", o + 1, o)
    eg, _ = _saturate(
        Op.make("matmul", Op.make("linear", x, w), rbad), rules=[rule]
    )
    _check_declined(eg, "linear_mm_absorb")

def test_linear_out_scale_declines():
    B, T, i, o = 2, 8, 6, 8
    x, w = _v("x", B, T, i), _p("W", o, i)
    # Per-out-channel scale: mul(W, s) would mis-broadcast — decline.
    s = _p("s", o)
    eg, _ = _saturate(
        Op.make("mul", Op.make("linear", x, w), s),
        rules=[_r_by_name("linear_out_scale")],
    )
    _check_declined(eg, "linear_out_scale")
    # Per-row scale: belongs to LINEAR_ROW_SCALE, not the weight fold.
    s = _v("s", B, T, 1)
    eg, _ = _saturate(
        Op.make("mul", Op.make("linear", x, w), s),
        rules=[_r_by_name("linear_out_scale")],
    )
    _check_declined(eg, "linear_out_scale")

def test_scale_view_declines_nonuniform():
    t = _v("t", 2, 8)
    a = _v("a", 8)  # per-column factor — not uniform, wrong side
    lhs = Op.make("mul", Op.make("transpose", t, dim0=0, dim1=1), a)
    eg, _ = _saturate(lhs, rules=[_r_by_name("scale_in_transpose")])
    _check_declined(eg, "scale_in_transpose")

def test_attention_preset_wiring():
    assert PRESETS["attention"] is ATTENTION
    assert preset("attention") is ATTENTION
    # the preset is the attention-path laws plus the sdpa_expand
    # decompose family (defined beside its fold siblings in
    # laws.tensor — the tag, not the module, marks it opt-in)
    assert set(ATTENTION.rules) == {
        *ATTENTION_RULES,
        *SDPA_EXPAND_RULES,
    }
    # opt-in only: nothing leaks into DEFAULT
    assert not (set(ATTENTION.rules) & set(DEFAULT.rules))
    # composes by name with the default set
    both = DEFAULT + ATTENTION
    assert "rope_cat_compose" in both
    assert "sdpa_fold_addmul" in both
    assert "sdpa_expand" in both
    # every rule carries the ATTENTION tag
    assert all("attention" in r.tags for r in ATTENTION_RULES)
    assert all("attention" in r.tags for r in SDPA_EXPAND_RULES)

def test_rope_cat_and_rh_pattern_builders():
    """The pattern builders mint the exported spellings (guard against
    drift between law patterns and the IR boundary)."""
    pat = _rope_cat("x", "C", "S", "I1", "I2", "CD")
    assert pat.op == "concat"
    sub, add = pat.args
    assert sub.op == "sub" and add.op == "add"
    assert sub.args[0].args[0].op == "slice"
    rh = _rope_rh("x", "C", "S", "I1", "I2", "CD")
    assert rh.op == "add" and rh.args[1].args[0].op == "concat"
    assert rh.args[1].args[0].args[0].op == "neg"
