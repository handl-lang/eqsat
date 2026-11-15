"""Contract tests for mlopt_core.typing — the shape/type-inference layer.

Phase 1a extracted ``_shape_of`` / ``_infer_op_shape`` / ``_INVALID`` /
``_broadcast`` / ``_numel`` out of mlopt_core.cost and moved the carrier
op cases into ``register_shape_rule`` handlers.  These tests pin the
contract the layer publishes:

* shapes are best-effort tuples (dims may be ``None``);
* ``_INVALID`` = provably ill-typed (poisonous to extraction);
* ``None`` = unknown — including the carrier-internal ``()`` policy
  (a ``()`` operand under an axis op, or a carrier's ``()`` slot read
  as a tensor, reports unknown, never a fabricated scalar);
* ``()`` = genuine scalar only;
* carrier ops (aff/apply/om/omd/trace/bdiag/parl families) answer via
  their registered handlers in ``_SHAPE_RULES``.
"""

import pytest
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_core.typing import (
    _INVALID,
    _SHAPE_RULES,
    _broadcast,
    _numel,
    _shape_of,
    register_shape_rule,
    register_value_shape,
    value_shape_of,
)

B, T, K, D = 4, 8, 3, 16

x234 = Var("x234", TensorType((2, 3, 4)))
x_BTk = Var("x_BTk", TensorType((B, T, K)))
x_BTd = Var("x_BTd", TensorType((B, T, D)))
vec8 = Var("vec8", TensorType((8,)))
mat = Var("mat", TensorType((10, 10)))
scalar = Const(1.0)


#: (op term, expected shape) — one row per contract point.
SHAPE_CASES = [
    # --- slice incl. strided step -----------------------------------------
    pytest.param(
        Op.make("slice", x234, dim=2, start=0, end=None, step=1),
        (2, 3, 4),
        id="slice/full-range",
    ),
    pytest.param(
        Op.make("slice", x234, dim=2, start=0, end=None, step=2),
        (2, 3, 2),
        id="slice/strided-step2",
    ),
    pytest.param(
        Op.make("slice", x234, dim=2, start=1, end=4, step=2),
        (2, 3, 2),
        id="slice/bounded-strided",
    ),
    pytest.param(
        Op.make("slice", x234, dim=1, start=0, end=2, step=1),
        (2, 2, 4),
        id="slice/dim-attr-spelling",
    ),
    # --- split: sizes/index + section index -------------------------------
    pytest.param(
        Op.make("split", vec8, sizes=(3, 5), index=1, dim=0),
        (5,),
        id="split/sizes-index-spelling",
    ),
    pytest.param(
        Op.make("split", vec8, sizes=(3, 5), index=0, dim=0),
        (3,),
        id="split/sizes-index-zero",
    ),
    pytest.param(
        Op.make("split", vec8, sizes=4, index=1, dim=0),
        (4,),
        id="split/equal-sections",
    ),
    pytest.param(
        Op.make("split", vec8, sizes=(3, 5), index=5, dim=0),
        (None,),
        id="split/index-past-end-unknown",
    ),
    # --- chunk ------------------------------------------------------------
    pytest.param(
        Op.make(
            "chunk", Var("c", TensorType((2, 8))), chunks=4, dim=-1
        ),
        (2, 2),
        id="chunk/last-dim",
    ),
    # --- concat arity: variadic cat sums every operand on the axis --------
    pytest.param(
        Op.make(
            "concat",
            Var("a", TensorType((2, 3))),
            Var("b", TensorType((2, 5))),
            Var("c", TensorType((2, 1))),
            dim=1,
        ),
        (2, 9),
        id="concat/three-operands",
    ),
    # Rank mismatch is best-effort, not poisonous: returns the base shape.
    pytest.param(
        Op.make(
            "concat",
            Var("a", TensorType((2, 3))),
            Var("b", TensorType((5,))),
            dim=1,
        ),
        (2, 3),
        id="concat/rank-mismatch-falls-back",
    ),
    # --- concat numel-0 operands (decode step-1 empty cache) --------------
    # torch.cat skips provably-empty operands: the (0,) lazy-init
    # placeholder must not collapse the result to rank 1.
    pytest.param(
        Op.make(
            "concat",
            Var("empty", TensorType((0,))),
            Var("k", TensorType((2, 3, 4, 5))),
            dim=-2,
        ),
        (2, 3, 4, 5),
        id="concat/empty-first-decode-cache",
    ),
    pytest.param(
        Op.make(
            "concat",
            Var("k", TensorType((2, 3, 4, 5))),
            Var("empty", TensorType((0,))),
            Var("v", TensorType((2, 3, 6, 5))),
            dim=-2,
        ),
        (2, 3, 10, 5),
        id="concat/empty-among-live-sums-cat-axis",
    ),
    pytest.param(
        Op.make(
            "concat",
            Var("e1", TensorType((0, 5))),
            Var("e2", TensorType((0, 7))),
            dim=1,
        ),
        (0, 12),
        id="concat/all-empty-still-sums",
    ),
    # --- reshape numel guard ----------------------------------------------
    pytest.param(
        Op.make("reshape", x234, shape=(4, 6)),
        (4, 6),
        id="reshape/preserving",
    ),
    pytest.param(
        Op.make("reshape", x234, shape=(4, -1)),
        (4, 6),
        id="reshape/minus-one-resolved",
    ),
    pytest.param(
        Op.make("reshape", x234, shape=(4, 7)),
        _INVALID,
        id="reshape/numel-mismatch-invalid",
    ),
    # --- broadcast ----------------------------------------------------------
    pytest.param(
        Op.make(
            "mul",
            Var("a", TensorType((4, 8, 1))),
            Var("b", TensorType((4, 8, 32))),
        ),
        (4, 8, 32),
        id="broadcast/trailing-one",
    ),
    pytest.param(
        Op.make(
            "add",
            Var("a", TensorType((4, 8))),
            Var("b", TensorType((4, 7))),
        ),
        _INVALID,
        id="broadcast/hard-mismatch-invalid",
    ),
    # --- scalar semantics: () is genuine only at real scalar ops ----------
    pytest.param(Op.make("sum", x234), (), id="sum/full-reduce-scalar"),
    pytest.param(
        Op.make("sum", x234, dim=1, keepdim=True),
        (2, 1, 4),
        id="sum/keepdim",
    ),
    pytest.param(
        Op.make("matmul", vec8, vec8), (), id="matmul/dot-scalar"
    ),
    # --- the () -> None carrier policy --------------------------------------
    pytest.param(
        Op.make("transpose", scalar),
        None,
        id="policy/transpose-of-scalar",
    ),
    pytest.param(
        Op.make("slice", scalar, dim=0),
        None,
        id="policy/slice-of-scalar",
    ),
    pytest.param(
        Op.make("sum", scalar, dim=0),
        None,
        id="policy/explicit-dim-reduce-of-scalar",
    ),
    pytest.param(
        Op.make("concat", scalar, scalar, dim=0),
        None,
        id="policy/concat-of-scalars",
    ),
    # --- carrier ops report via registered handlers -----------------------
    pytest.param(
        Op.make("om", x_BTk, x_BTk, x_BTd),
        (B, T, D),
        id="carrier/om-accumulator-shape",
    ),
    pytest.param(
        Op.make("om", x_BTk, x_BTk, scalar),
        None,
        id="carrier/om-scalar-slot-unknown",
    ),
    pytest.param(
        Op.make("om_elem", x_BTk, x_BTd),
        (B, T, D),
        id="carrier/om-elem-applied-shape",
    ),
    pytest.param(
        Op.make("apply", Var("f", TensorType((D, D))), x_BTd),
        (B, T, D),
        id="carrier/apply-evaluates-h",
    ),
    pytest.param(
        Op.make("aff", Var("A", TensorType((D, D))), vec8),
        (D, D),
        id="carrier/aff-linear-part-shape",
    ),
    pytest.param(
        Op.make("aff", scalar, vec8),
        None,
        id="carrier/aff-scalar-slot-unknown",
    ),
    pytest.param(
        Op.make("applyd", Var("f", TensorType((D,))), vec8),
        (8,),
        id="carrier/applyd-evaluates-h",
    ),
    pytest.param(
        Op.make("trace", mat, usize=4),
        (6, 6),
        id="carrier/trace-drops-usize",
    ),
    pytest.param(
        Op.make("trace", mat, usize=(2, 2)),
        (6, 6),
        id="carrier/trace-usize-list",
    ),
    pytest.param(
        Op.make(
            "bdiag",
            Var("p", TensorType((2, 3))),
            Var("q", TensorType((4, 5))),
        ),
        (6, 8),
        id="carrier/bdiag-juxtaposes",
    ),
    pytest.param(
        Op.make(
            "parl",
            Var("p", TensorType((2, 3))),
            Var("q", TensorType((4, 5))),
        ),
        (6, 8),
        id="carrier/parl-juxtaposes",
    ),
    # --- zero-arg constant morphisms (registered, attrs-only) -------------
    pytest.param(
        Op.make("eye", dim=5), (5, 5), id="const-morphism/eye"
    ),
    pytest.param(
        Op.make("cswap", d1=2, d2=3),
        (5, 5),
        id="const-morphism/cswap",
    ),
    # --- shape-preserving ops w/ attrs (default arm) ----------------------
    pytest.param(
        Op.make(
            "rms_norm",
            Var("h", TensorType((4, 64))),
            Param("g", TensorType((64,))),
            dim=(64,),
            eps=1e-6,
        ),
        (4, 64),
        id="rms_norm/dim-attr-passthrough",
    ),
]


@pytest.mark.parametrize("term,expected", SHAPE_CASES)
def test_infer_op_shape(term, expected):
    got = _shape_of(term)
    if expected is _INVALID:
        assert got is _INVALID
    else:
        assert got == expected


def test_broadcast_helper():
    assert _broadcast((4, 8, 1), (8, 32)) == (4, 8, 32)
    assert _broadcast(None, (2, 3)) == (2, 3)
    assert _broadcast((2, 3), None) == (2, 3)
    assert _broadcast((4, 8), (4, 7)) is _INVALID
    # Unresolved -1 dims degrade to unknown, never a mismatch.
    assert _broadcast((-1, 8), (4, 8)) == (None, 8)


def test_numel_helper():
    assert _numel(None) == 1
    assert _numel(()) == 1
    assert _numel((2, 3, 4)) == 24
    assert _numel((2, None, 4)) == 8


def test_register_shape_rule_hook():
    """The registry is real dispatch: a custom rule installs and
    _infer_op_shape consults it ahead of the match's default arm."""
    try:
        register_shape_rule(
            "myop_test", lambda op, shapes: (7, *shapes[0][1:])
        )
        got = _shape_of(Op.make("myop_test", x234))
        assert got == (7, 3, 4)
    finally:
        _SHAPE_RULES.pop("myop_test", None)
    # After unregistering the same term falls back to shapes[0].
    assert _shape_of(Op.make("myop_test", x234)) == (2, 3, 4)


def test_register_value_shape_hook():
    """The value-shape registry: a registered resolver is consulted by
    ``value_shape_of``; with none it falls back to ``_shape_of``."""
    import mlopt_core.typing as ty

    prev = ty._VALUE_SHAPE
    try:
        ty._VALUE_SHAPE = None
        # fallback: the cost convention, exactly _shape_of
        assert value_shape_of(x234) == (2, 3, 4)
        assert value_shape_of(x234) == _shape_of(x234)
        register_value_shape(lambda _t: (9, 9))
        assert value_shape_of(x234) == (9, 9)
        # the registered resolver owns the read — _shape_of is untouched
        assert _shape_of(x234) == (2, 3, 4)
    finally:
        ty._VALUE_SHAPE = prev


def test_leaf_typing():
    assert _shape_of(Const(3.0)) == ()
    v = Var("v", TensorType((None, 4)))
    assert _shape_of(v) == (None, 4)


def test_memo_is_content_keyed_dag():
    """Shared subterms resolve once through the memo (DAG, not tree).

    Terms are interned and content-hashed — the memo keys on the term
    object itself (no id(), no GC-reuse hazard).
    """
    shared = Op.make("mul", x234, x234)
    term = Op.make("add", shared, shared)
    memo: dict = {}
    assert _shape_of(term, memo) == (2, 3, 4)
    assert shared in memo


def test_concat_empty_operand_decode_cache():
    """Decode step-1 KV-cache append: ``concat(dim=-2)`` of the
    ``(0,)`` lazy-init placeholder and the live ``(B,H,T,D)`` cache.

    Before the numel-0 fix the empty FIRST operand collapsed the
    inferred shape to ``(0,)`` — rank 1 — which vetoed every
    shape-gated law (``axes-last2`` on the bound K transpose, the
    ``sdpa_fold`` family, ``SDPA_CAT_LAWS``) on fresh-cache decode IR.
    """
    empty = Var("c_k_lifted", TensorType((0,)))
    k = Op.make(
        "transpose",
        Op.make(
            "reshape",
            Var("k_new", TensorType((2, 3, 4, 5))),
            shape=(2, 3, 4, 5),
        ),
        dim=(-2, -1),
    )
    cat = Op.make(
        "concat",
        Op.make("detach_", Op.make("contiguous", empty)),
        k,
        dim=-2,
    )
    # The concat itself reports the live rank-4 shape (cat axis
    # summed: the empty operand contributes 0 along dim -2).
    # k is the transposed cache: (2, 3, 5, 4).
    assert _shape_of(cat) == (2, 3, 5, 4)
    # Downstream consumers see rank 4, not the poisoned (0,): an
    # axes-last-two transpose resolves instead of declining.
    kt = Op.make("transpose", cat, dim=(-2, -1))
    assert _shape_of(kt) == (2, 3, 4, 5)


def test_partial_unknown_shape_not_operand0():
    """Regression: the partial-unknown early return used to answer
    ``shapes[0]`` for ANY op — ``matmul(x, ?)`` reported x's own shape,
    a fabricated result that poisons every broadcast consumer."""
    known = Var("xk", TensorType((4, 8)))
    unknown = Var("uu", TensorType(None))
    # Shape-transforming ops: honest unknown, never operand 0's shape.
    assert _shape_of(Op.make("matmul", known, unknown)) is None
    assert _shape_of(Op.make("linear", known, unknown)) is None
    assert _shape_of(Op.make("concat", known, unknown, dim=0)) is None
    assert _shape_of(Op.make("select", known, unknown)) is None
    # Broadcast binaries ALSO decline: their output is the broadcast
    # of EVERY operand and can exceed the known one, so a ``None``
    # sibling means genuinely unknown — borrowing the known shape
    # fabricates a too-small concrete shape (the kokoro
    # ``transpose_noop`` miscompile: an LSTM-unknown F0 path
    # inferred ``(1, 1, 1)`` and a required transpose was proven a
    # no-op and deleted).
    assert _shape_of(Op.make("add", known, unknown)) is None
    assert _shape_of(Op.make("sub", known, unknown)) is None
    assert _shape_of(Op.make("add", unknown, known)) is None
    assert _shape_of(Op.make("mul", known, unknown)) is None
    # Shape-preserving unary ops keep the operand-0 answer.
    assert _shape_of(Op.make("neg", known)) == (4, 8)
    assert _shape_of(
        Op.make("relu", Op.make("transpose", known, dim0=0, dim1=1))
    ) == (8, 4)


def test_broadcast_binop_unknown_operand_declines():
    """Kokoro ``transpose_noop`` miscompile regression.

    The F0Ntrain→decoder term carried ``mul(style_param(1,256,1),
    lstm_output)`` where the LSTM output's shape is unshapeable
    (``None``).  The partial-unknown fallback reported the param's
    ``(1, 256, 1)``; the ``conv1d`` then inferred ``(1, 1, 1)``, the
    ``upsample``/``squeeze``/``unsqueeze`` chain propagated it, and
    ``transpose(1, 2)`` on the fabricated ``(1, 1, 1)`` satisfied
    ``axes-noop`` ("both swapped extents are 1") — so the law deleted
    a REQUIRED transpose and the delivered module crashed on
    ``multiply (1,1,38400) x (1,1,9)``.  With the fallback removed
    the whole chain stays ``None`` and the guard declines.
    """
    unknown = Op.make(
        "einsum",
        a=Param("w", TensorType((2, 2))),
        b=Param("v", TensorType((2, 2))),
        equation="ij,jk->ik",
    )
    assert _shape_of(unknown) is None
    fused = Op.make("mul", Param("g", TensorType((1, 256, 1))), unknown)
    assert _shape_of(fused) is None
    conv = Op.make(
        "conv1d",
        fused,
        Param("cw", TensorType((1, 256, 1))),
        Param("cb", TensorType((1,))),
        stride=(1,),
        padding=(0,),
    )
    assert _shape_of(conv) is None
    tr = Op.make("transpose", conv, dim0=1, dim1=2)
    assert _shape_of(tr) is None
    # The guard that fired wrongly now declines on the unknown shape.
    from mlopt_core.laws.cond import eval_cond

    assert not eval_cond(
        ("axes-noop", "u", "D0", "D1"),
        {"u": conv, "$attr:D0": 1, "$attr:D1": 2},
    )
