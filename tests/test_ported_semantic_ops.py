"""Substrate-pure tests ported from ``graphopt:tests/test_semantic_ops.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.ir import Const, IR, Op, TensorType, Var

from mlopt_core.typing import _INVALID, _shape_of


def test_arange_shape_const_operands():
    t = Op.make("arange", Const(0.0), Const(6.0), Const(2.0))
    assert _shape_of(t) == (3,)

def test_scaled_extent_bool_guard():
    """``isinstance(True, int)`` is True — a bool dim/scale is not a
    concrete extent, and ``_scaled_extent`` declines it."""
    from mlopt_core.typing import _scaled_extent

    assert _scaled_extent(8, True) is None
    assert _scaled_extent(True, 2.0) is None
