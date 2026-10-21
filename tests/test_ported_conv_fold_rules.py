"""Substrate-pure tests ported from ``graphopt:tests/test_conv_fold_rules.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core import meta

from mlopt_core.ir import Op, Param, TensorType, Var, op_repr

from mlopt_core.laws import (
    ALL_RULES,
    CONV_FOLD_RULES,
    DEFAULT,
)

from mlopt_core.laws.cond import (
    cond_from_data,
    cond_to_data,
    derive_from_data,
    derive_to_data,
)


def _T(*shape):
    return TensorType(tuple(shape))

_BY_NAME = {r.name: r for r in CONV_FOLD_RULES}

def _leaves(x_shape, cout, cin, k=3):
    return (
        Var("x", _T(*x_shape)),
        Param("W", _T(cout, cin, k, k)),
        Param("b", _T(cout)),
        Param("sv", _T(1, cout, 1, 1)),
        Param("cv", _T(1, cout, 1, 1)),
    )

def _conv(x, w, b=None, **attrs):
    return Op.make("conv2d", x, w, *(() if b is None else (b,)), **attrs)

def test_conv_fold_family_census():
    """6 shapes × 5 attr prefixes × 2 bias arities, names distinct."""
    assert len(CONV_FOLD_RULES) == 60
    assert len({r.name for r in CONV_FOLD_RULES}) == 60
    # the anchor the probe needs: conv2d-headed RHS material exists
    assert any(getattr(r.rhs, "op", None) == "conv2d" for r in CONV_FOLD_RULES)
    assert any(
        "conv2d" in op_repr(r.lhs) for r in CONV_FOLD_RULES
    ), "a conv2d LHS is needed for the unfold direction"
    for stem in ("conv_channel_scale", "conv_channel_shift"):
        assert f"{stem}_sp" in _BY_NAME  # the seed's attr set

def test_conv_fold_rules_are_data_and_opt_in():
    """cond+dspec serialize; the family stays out of DEFAULT/ALL_RULES."""
    names = {r.name for r in DEFAULT.rules} | {r.name for r in ALL_RULES}
    assert not ({r.name for r in CONV_FOLD_RULES} & names)
    for r in CONV_FOLD_RULES:
        assert cond_from_data(cond_to_data(r.cond)) == r.cond
        assert derive_from_data(derive_to_data(r.dspec)) == r.dspec
        assert r.check is not None and r.derive is not None
        assert meta._synthesizable(r)

def test_conv_channel_scale_fires_both_orders():
    x, w, b, sv, _cv = _leaves((1, 4, 8, 8), 6, 4)
    conv = _conv(x, w, b, stride=(1, 1), padding=(1, 1))
    for name, lhs in (
        ("conv_channel_scale_sp", Op.make("mul", conv, sv)),
        ("conv_channel_scale_l_sp", Op.make("mul", sv, conv)),
    ):
        out = meta.apply_rewrite_at(_BY_NAME[name], lhs, ())
        assert out is not None, name
        assert op_repr(out) == (
            "(conv2d x, (mul W, (reshape sv, shape=(6, 1, 1, 1))), "
            "(mul b, (reshape sv, shape=(6,))), "
            "stride=(1, 1), padding=(1, 1))"
        )

def test_conv_channel_scale_attr_prefix_variants():
    """Every attr-prefix variant pins its own key set exactly."""
    x, w, b, sv, _cv = _leaves((1, 4, 8, 8), 6, 4)
    # bare conv (all-default attrs elided) — the no-suffix variant
    t = Op.make("mul", _conv(x, w, b), sv)
    out = meta.apply_rewrite_at(_BY_NAME["conv_channel_scale"], t, ())
    assert out is not None
    assert "stride" not in op_repr(out)
    # full tail incl. groups — the _spdg variant
    wg = Param("Wg", _T(6, 2, 3, 3))
    tg = Op.make(
        "mul",
        _conv(
            x,
            wg,
            b,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=2,
        ),
        sv,
    )
    out = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_scale_spdg"], tg, ()
    )
    assert out is not None
    assert "groups=2" in op_repr(out)
    # attr-set mismatch declines: the _sp variant does not fire on the
    # bare conv, nor the bare variant on the attr-carrying conv
    assert meta.apply_rewrite_at(_BY_NAME["conv_channel_scale_sp"], t, ()) is None
    t_sp = Op.make("mul", _conv(x, w, b, stride=(1, 1), padding=(1, 1)), sv)
    assert meta.apply_rewrite_at(_BY_NAME["conv_channel_scale"], t_sp, ()) is None

def test_conv_channel_scale_guard_vetoes():
    x, w, b, sv, _cv = _leaves((1, 4, 8, 8), 6, 4)
    conv = _conv(x, w, b, stride=(1, 1), padding=(1, 1))
    rule = _BY_NAME["conv_channel_scale_sp"]
    # wrong channel extent (5 != W's 6)
    sv5 = Param("sv5", _T(1, 5, 1, 1))
    assert meta.apply_rewrite_at(rule, Op.make("mul", conv, sv5), ()) is None
    # channel axis misplaced — (1,1,6,1) scales spatial, not channel
    sv_mis = Param("sv_mis", _T(1, 1, 6, 1))
    assert (
        meta.apply_rewrite_at(rule, Op.make("mul", conv, sv_mis), ())
        is None
    )
    # a scalar/uniform view is not a channel scale
    sv_uni = Param("sv_uni", _T(1, 1, 1, 1))
    assert (
        meta.apply_rewrite_at(rule, Op.make("mul", conv, sv_uni), ())
        is None
    )
    # wrong bias extent
    b5 = Param("b5", _T(5))
    assert (
        meta.apply_rewrite_at(
            rule, Op.make("mul", _conv(x, w, b5, stride=(1, 1),
                                       padding=(1, 1)), sv), ()
        )
        is None
    )
    # in-channel mismatch (x[1]=4 != W[1]=3): the conv does not denote
    w3 = Param("w3", _T(6, 3, 3, 3))
    assert (
        meta.apply_rewrite_at(
            rule, Op.make("mul", _conv(x, w3, b, stride=(1, 1),
                                       padding=(1, 1)), sv), ()
        )
        is None
    )
    # unshaped members cannot prove the view
    xu = Var("xu", TensorType((None, None, None, None)))
    assert (
        meta.apply_rewrite_at(
            rule,
            Op.make("mul", _conv(xu, w, b, stride=(1, 1),
                                 padding=(1, 1)), sv),
            (),
        )
        is None
    )
    # groups-aware in-channels: x[1]=4 != W[1]*G = 3*2
    wg_bad = Param("wg_bad", _T(6, 3, 3, 3))
    t_g = Op.make(
        "mul",
        _conv(x, wg_bad, b, stride=(1, 1), padding=(1, 1),
              dilation=(1, 1), groups=2),
        sv,
    )
    assert (
        meta.apply_rewrite_at(_BY_NAME["conv_channel_scale_spdg"], t_g, ())
        is None
    )
    # symbolic dims decline
    wu = Param("wu", TensorType((6, None, 3, 3)))
    assert (
        meta.apply_rewrite_at(
            rule,
            Op.make("mul", _conv(x, wu, b, stride=(1, 1),
                                 padding=(1, 1)), sv),
            (),
        )
        is None
    )

def test_conv_channel_shift_fires_add_and_sub():
    x, w, b, _sv, cv = _leaves((1, 4, 8, 8), 6, 4)
    conv = _conv(x, w, b, stride=(1, 1), padding=(1, 1))
    for name, lhs, want_op in (
        ("conv_channel_shift_sp", Op.make("add", conv, cv), "add"),
        ("conv_channel_shift_l_sp", Op.make("add", cv, conv), "add"),
        ("conv_channel_shift_sub_sp", Op.make("sub", conv, cv), "sub"),
    ):
        out = meta.apply_rewrite_at(_BY_NAME[name], lhs, ())
        assert out is not None, name
        assert op_repr(out) == (
            f"(conv2d x, W, ({want_op} b, "
            "(reshape cv, shape=(6,))), "
            "stride=(1, 1), padding=(1, 1))"
        )
    # the shift guard is the same channel-view check
    cv_mis = Param("cv_mis", _T(6, 1, 1, 1))
    assert (
        meta.apply_rewrite_at(
            _BY_NAME["conv_channel_shift_sp"],
            Op.make("add", conv, cv_mis),
            (),
        )
        is None
    )

def test_conv_fold_nobias_arity():
    """The bias-free arity mints the bias operand for the shift."""
    x, w, _b, sv, cv = _leaves((1, 4, 8, 8), 6, 4)
    conv = _conv(x, w, stride=(1, 1), padding=(1, 1))
    out = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_scale_sp_nobias"],
        Op.make("mul", conv, sv),
        (),
    )
    assert op_repr(out) == (
        "(conv2d x, (mul W, (reshape sv, shape=(6, 1, 1, 1))), "
        "stride=(1, 1), padding=(1, 1))"
    )
    out = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_shift_sp_nobias"],
        Op.make("add", conv, cv),
        (),
    )
    assert op_repr(out) == (
        "(conv2d x, W, (reshape cv, shape=(6,)), "
        "stride=(1, 1), padding=(1, 1))"
    )
    out = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_shift_sub_sp_nobias"],
        Op.make("sub", conv, cv),
        (),
    )
    assert "(neg (reshape cv, shape=(6,)))" in op_repr(out)

def test_conv_channel_scale_rev_round_trip():
    """The unfold: absorb then unfold recovers the scaled member."""
    x, w, b, sv, _cv = _leaves((1, 4, 8, 8), 6, 4)
    conv = _conv(x, w, b, stride=(1, 1), padding=(1, 1))
    t = Op.make("mul", conv, sv)
    folded = meta.apply_rewrite_at(_BY_NAME["conv_channel_scale_sp"], t, ())
    out = meta.apply_rewrite_at(
        _BY_NAME["conv_channel_scale_rev_sp"], folded, ()
    )
    assert op_repr(out) == op_repr(t)
    # the view-pinning guard declines a WRONG reshape target:
    # reshape(sv, (6,1,1,1)) is the only honest weight view; a (1,6,1,1)
    # target (channel on axis 1) is a different value
    bad = _conv(
        x,
        Op.make("mul", w, Op.make("reshape", sv, shape=(1, 6, 1, 1))),
        Op.make("mul", b, Op.make("reshape", sv, shape=(6,))),
        stride=(1, 1),
        padding=(1, 1),
    )
    assert (
        meta.apply_rewrite_at(_BY_NAME["conv_channel_scale_rev_sp"], bad, ())
        is None
    )

def test_conv_fold_anchors_are_conv2d_headed():
    """Gate A's named gap is closed: conv2d heads RHSes in the bank."""
    heads = {getattr(r.rhs, "op", None) for r in CONV_FOLD_RULES}
    assert "conv2d" in heads
    scale = _BY_NAME["conv_channel_scale_sp"]
    assert getattr(scale.rhs, "op", None) == "conv2d"
    shift = _BY_NAME["conv_channel_shift_sub_sp"]
    assert getattr(shift.rhs, "op", None) == "conv2d"
