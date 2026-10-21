"""Substrate-pure tests ported from ``graphopt:tests/test_om_decayed_chunk.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.cost import flops_cost, launch_aware_cost

from mlopt_core.ir import IR, Const, Op, TensorType, Var

from mlopt_core.typing import _INVALID, _shape_of


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))

def test_shape_rule_reports_the_triangle():
    """Scalar γ → (T,T); a per-head (H,) rate → (H,T,T)."""
    assert _shape_of(Op.make("decay", _v("g"), t=8)) == (8, 8)
    assert _shape_of(Op.make("decay", _v("g", 3), t=8)) == (3, 8, 8)

def test_shape_rule_handles_bad_attrs():
    """A missing/non-int ``t`` is unknown; a negative one is _INVALID."""
    g = _v("g")
    assert _shape_of(Op.make("decay", g)) is None
    assert _shape_of(Op.make("decay", g, t="x")) is None
    assert _shape_of(Op.make("decay", g, t=-2)) is _INVALID

def test_flops_price_the_spelled_subgraph():
    """decay costs the spelled sub+clamp+pow+tril: 5·T²."""
    assert flops_cost(Op.make("decay", _v("g"), t=8)) == 5 * 64
