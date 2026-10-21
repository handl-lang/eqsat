"""Substrate-pure tests ported from ``graphopt:tests/test_om_mha.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.cost import _infer_op_shape, _shape_of, flops_cost

from mlopt_core.ir import IR, Const, Op, Param, TensorType


def test_scalar_operand_axis_ops_report_none():
    """A ``()``-shaped operand under an op that cannot be scalar means
    a carrier-internal member is being read as a tensor — the shape is
    unknown (None), not a fabricated scalar and never a crash."""
    scalar = Const(1.0)
    v4 = Param("v4", TensorType((4,)))
    m44 = Param("m44", TensorType((4, 4)))
    # the ZeroDivisionError site: transpose on a ()-shaped operand
    assert (
        _infer_op_shape(Op.make("transpose", scalar, dim0=0, dim1=1))
        is None
    )
    # every other axis-indexing op takes the same policy
    assert _infer_op_shape(Op.make("squeeze", scalar, dim=0)) is None
    assert (
        _infer_op_shape(Op.make("select", scalar, dim=0, index=0))
        is None
    )
    assert _infer_op_shape(Op.make("slice", scalar, dim=0)) is None
    assert _infer_op_shape(Op.make("unbind", scalar, dim=0)) is None
    assert (
        _infer_op_shape(Op.make("chunk", scalar, dim=0, chunks=2))
        is None
    )
    assert (
        _infer_op_shape(
            Op.make("split", scalar, dim=0, sizes=(1, 1), index=0)
        )
        is None
    )
    assert (
        _infer_op_shape(Op.make("concat", scalar, scalar, dim=0))
        is None
    )
    assert _infer_op_shape(Op.make("flatten", scalar)) is None
    # explicit-dim reduce on a scalar (dim=None full-reduce stays ())
    assert _infer_op_shape(Op.make("sum", scalar, dim=0)) is None
    assert _infer_op_shape(Op.make("sum", scalar)) == ()
    # scalar operand under matmul/linear is ill-typed, not ()
    assert _infer_op_shape(Op.make("matmul", scalar, m44)) is None
    assert _infer_op_shape(Op.make("linear", scalar, m44)) is None
    # carrier convention slots resolve () -> None
    assert _infer_op_shape(Op.make("om_elem", scalar, v4)) is None
    assert (
        _infer_op_shape(
            Op.make("applyd", Op.make("aff_diag", v4, v4), scalar)
        )
        is None
    )
    # genuine scalar results are unaffected
    assert _infer_op_shape(Op.make("matmul", v4, v4)) == ()  # dot
    assert _shape_of(scalar) == ()
