"""Substrate-pure tests ported from ``graphopt:tests/test_morphism_multiinput.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import mlopt_orchestrator.morphisms as M

from mlopt_core.ir import IR, Op, Param, TensorType, Var


_T = TensorType((8, 16))

_TL = TensorType((16, 16))

def _v(name: str) -> Var:
    return Var(name, _T)

def _ir(root, *vs: Var) -> IR:
    return IR(
        root=root,
        inputs=list(vs),
        input_names={v.name for v in vs},
        params={},
    )

def _kinds(root, *names: str) -> tuple:
    """``_classify_inputs`` kinds for a root over named input vars."""
    vs = tuple(_v(n) for n in names)
    return tuple(i.kind for i in M._classify_inputs(_ir(root, *vs)))

def _sig(inputs: tuple) -> M.BlockSig:
    return M.BlockSig(
        in_projs=(),
        out_proj=(),
        norm=M.NormSig(kind="none", affine=False, pre=False),
        act=(),
        residual=False,
        shape=(None, None),
        inputs=inputs,
    )

def test_classify_dead_and_stream():
    """An unused context input is ``const_table``/``dead``."""
    x, c = _v("x"), _v("c")
    w = Param("w", _TL)
    kinds = M._classify_inputs(_ir(Op.make("linear", x, w), x, c))
    assert [(i.kind, i.role) for i in kinds] == [
        ("activation", "stream"),
        ("const_table", "dead"),
    ]

def test_classify_runtime_weight_opaque():
    """A var in a projection's *weight* position is opaque."""
    x, w = _v("x"), _v("w")
    kinds = _kinds(Op.make("linear", x, w), "x", "w")
    assert kinds == ("activation", "opaque")
    kinds = _kinds(Op.make("matmul", x, w), "x", "w")
    assert kinds == ("activation", "opaque")
    # …and the var-weight matmul reverses the roles.
    kinds = _kinds(Op.make("matmul", w, x), "x", "w")
    assert kinds == ("opaque", "activation")

def test_classify_addend_and_sdpa():
    """Bare addends and sdpa k/v positions are context reads."""
    x, c = _v("x"), _v("c")
    assert _kinds(Op.make("add", x, c), "x", "c") == (
        "const_table",
        "const_table",
    )
    k, v2 = _v("k"), _v("v")
    kinds = _kinds(Op.make("sdpa", x, k, v2), "x", "k", "v")
    assert kinds == ("activation", "const_table", "const_table")

def test_classify_norm_and_tables():
    """Norm pos>0 and table ops mark context; a norm subject is stream."""
    x, w = _v("x"), _v("w")
    n = Op.make("layer_norm", x, w, normalized_shape=(16,))
    assert _kinds(n, "x", "w") == ("activation", "opaque")
    p = Param("pw", _TL)
    t, idx = _v("t"), _v("i")
    root = Op.make(
        "add",
        Op.make("linear", x, p),
        Op.make("index_select", t, 0, idx),
    )
    assert _kinds(root, "x", "t", "i") == (
        "activation",
        "const_table",
        "const_table",
    )

def test_classify_pointwise():
    """Pointwise roles hinge on whether siblings carry vars."""
    x, c = _v("x"), _v("c")
    # var * Const → stream (the sibling carries no var)
    assert _kinds(Op.make("mul", x, M.Const(2.0)), "x", "c") == (
        "activation",
        "const_table",
    )
    # var * var → both context (a two-stream product is ambiguous)
    assert _kinds(Op.make("mul", x, c), "x", "c") == (
        "const_table",
        "const_table",
    )

def test_classify_mutation_and_unknown():
    """Write destinations are mutated state; unknown ops are opaque."""
    x, c = _v("x"), _v("c")
    sel = Op.make("select", x, dim=0, index=0)
    w = Param("w", _TL)
    root = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make("select_scatter", c, sel, dim=0, index=0),
    )
    kinds = M._classify_inputs(_ir(root, x, c))
    # x's select→select_scatter-src use is opaque (an opaque use
    # outranks its stream use); the write target is mutated — the IR
    # sees the mutation, no probe needed.
    assert (kinds[0].kind, kinds[0].role) == ("opaque", "unhandled")
    assert (kinds[1].kind, kinds[1].role) == ("state", "mutated")
    # An unclassifiable op marks its operand opaque.
    kinds = _kinds(Op.make("frobnicate", x, c), "x", "c")
    assert kinds == ("opaque", "opaque")

def test_classify_view_root_and_pos():
    """Views are transparent: a var ending at the root is stream."""
    x, c = _v("x"), _v("c")
    # root == the var itself → stream; the other input is dead.
    kinds = M._classify_inputs(_ir(x, x, c))
    assert (kinds[0].kind, kinds[0].role) == ("activation", "stream")
    assert (kinds[1].kind, kinds[1].role) == ("const_table", "dead")
    # A var at a non-arg0 position of a view op is a context read.
    kinds = _kinds(
        Op.make(
            "add", Op.make("mul", x, x), Op.make("expand_as", c, x)
        ),
        "x",
        "c",
    )
    assert kinds == ("const_table", "const_table")
    # A var under a view at arg0 keeps walking to the terminal op.
    kinds = _kinds(
        Op.make("add", Op.make("linear", x, Param("w", _TL)), c),
        "x",
        "c",
    )
    assert kinds == ("activation", "const_table")
    # A view op that IS the root: the var flows to the output — stream.
    kinds = _kinds(Op.make("transpose", x, 0, 1), "x", "c")
    assert kinds == ("activation", "const_table")

def test_classify_dedup_uses():
    """A var feeding the same op twice visits that op once."""
    x, c = _v("x"), _v("c")
    kinds = _kinds(Op.make("add", Op.make("mul", x, x), c), "x", "c")
    # x*x: both positions see a var sibling → context (not activation).
    assert kinds == ("const_table", "const_table")

def test_sig_liftable_verdicts():
    """The lift verdict reads the InputSig kinds."""
    assert M._sig_liftable(_sig(())) is None
    a = M.InputSig(0, "x", "activation", "stream", (8, 16))
    t = M.InputSig(1, "t", "const_table", "table", (8, 16))
    st = M.InputSig(1, "s", "state", "read_only", (8, 16))
    mu = M.InputSig(1, "m", "state", "mutated", (8, 16))
    op = M.InputSig(1, "w", "opaque", "unhandled", (16, 16))
    assert M._sig_liftable(_sig((a, t))) is None
    assert M._sig_liftable(_sig((a, st))) is None
    assert "mutates" in M._sig_liftable(_sig((a, mu)))
    assert "no activation" in M._sig_liftable(_sig((t,)))
    assert "multi-activation" in M._sig_liftable(_sig((a, a)))
    assert "opaque" in M._sig_liftable(_sig((a, op)))
    # _act_index returns the sig position, 0 when no activation exists.
    assert M._act_index((t,)) == 0
    assert M._act_index((t, a)) == 0
    assert (
        M._act_index(
            (t, M.InputSig(1, "x2", "activation", "stream", (8, 16)))
        )
        == 1
    )

def test_block_signature_infers_shapes():
    """``block_signature`` pulls shapes from the activation var."""
    x, c = _v("x"), _v("c")
    p = Param("w", _TL)
    ir = _ir(Op.make("add", Op.make("linear", x, p), c), x, c)
    sig = M.block_signature(ir)
    assert sig.shape == ((8, 16), (8, 16))
    assert [w.name for w in sig.in_projs] == ["w"]
