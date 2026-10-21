"""Substrate-pure tests ported from ``graphopt:tests/test_batched_verify.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.cost import _INVALID, _shape_of, flops_cost

from mlopt_core.ir import Op, TensorType, Var


def _V(name: str, shape) -> Var:
    return Var(name, TensorType(tuple(shape)))

def test_reshape_inference_flags_numel_mismatch():
    """``reshape(x, S)`` where ``numel(S) != numel(x)`` is ill-typed:
    cost must say ``_INVALID`` (never extractable), not report S."""
    B, T, nh, hd = 16, 256, 12, 64
    x = _V("x", (B, T, nh * hd))
    bad = Op.make("reshape", x, shape=(B, T // 2, nh, hd))
    assert _shape_of(bad) is _INVALID
    # the exact reported flavour: T halved onto the head axes
    bad2 = Op.make("reshape", x, shape=(16, 128, 12, 64))
    assert _shape_of(bad2) is _INVALID
    # a consistent reshape still reports its attr
    good = Op.make("reshape", x, shape=(B, T, nh, hd))
    assert _shape_of(good) == (B, T, nh, hd)
