"""Substrate-pure tests ported from ``graphopt:tests/test_morphism_fusednorm_tie.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import mlopt_orchestrator.morphisms as M

from mlopt_core.ir import IR, Op, Param, TensorType, Var


def test_norm_nodes_fused_edges():
    """``_norm_nodes`` guard arms on the fused op."""
    x = Var("x", TensorType((8, 16)))
    p = Param("p_w", TensorType((16,)))
    p2 = Param("p_w2", TensorType((16,)))

    # Bare 0-arg node (raw construction) — arity guard.
    assert M._norm_nodes(Op("rms_norm", (), {})) == []
    # Param-only subject — compile-time data, not a stream norm.
    assert M._norm_nodes(Op.make("rms_norm", p, p2)) == []
    # Weightless form — rms structure, not affine.
    t = Op.make("rms_norm", x)
    assert M._norm_nodes(t) == [(t, "rms", False)]
    # Weighted form — affine.
    t = Op.make("rms_norm", x, p)
    assert M._norm_nodes(t) == [(t, "rms", True)]

def test_norm_unfolds_guards():
    """``_norm_unfolds`` only unfuses weighted fused norms."""
    x = Var("x", TensorType((8, 16)))
    p = Param("p_w", TensorType((16,)))

    # Non-norm ops and the weightless spelling offer nothing.
    assert M._norm_unfolds(Op.make("linear", x, p)) == []
    assert M._norm_unfolds(Op.make("rms_norm", x)) == []
    # A var-carrying "weight" operand is not a foldable gain.
    t = Op.make(
        "rms_norm", x, Op.make("mul", x, p), dim=(16,), eps=1e-5
    )
    assert M._norm_unfolds(t) == []
    # The weighted form unfolds to the pointwise-gain member.
    t = Op.make("rms_norm", x, p, dim=(16,), eps=1e-5)
    (host, expanded, law) = M._norm_unfolds(t)[0]
    assert host is t and "rms_norm" in law
    assert expanded == Op.make(
        "mul", Op.make("rms_norm", x, dim=(16,), eps=1e-5), p
    )

def test_sig_table_edges():
    """Table collection guards: empty arity and var-carried tables."""
    ir = IR(root=Op("embedding", (), {}), inputs=[], input_names=set())
    assert M.block_signature(ir).tables == ()
    v, tab = Var("idx", TensorType((8,))), Var("tab", TensorType((32, 16)))
    ir = IR(
        root=Op.make("embedding", tab, v),
        inputs=[tab, v],
        input_names={"tab", "idx"},
    )
    assert M.block_signature(ir).tables == ()
