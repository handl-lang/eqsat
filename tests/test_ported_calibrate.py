"""Substrate-pure tests ported from ``graphopt:tests/test_calibrate.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core.cost import (
    dag_cost,
    depth_cost_for,
    roofline_cost,
    roofline_cost_for,
)

from mlopt_core.ir import Op, Param, TensorType, Var


def _mm_term():
    """Compute-bound: 256x256 @ 256x256 matmul (~34 MFLOP, ~0.8 MB)."""
    x = Var("x", TensorType((256, 256)))
    w = Param("W", TensorType((256, 256)))
    return Op.make("matmul", x, w)

def test_roofline_cost_for_no_profile_is_default():
    fn = roofline_cost_for()
    assert fn(_mm_term()) == pytest.approx(roofline_cost(_mm_term()))
