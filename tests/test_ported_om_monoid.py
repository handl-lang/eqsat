"""Substrate-pure tests ported from ``graphopt:tests/test_om_monoid.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.cost import dag_cost, flops_cost

from mlopt_core.ir import IR, Op, TensorType, Var, op_repr


def test_om_cost_model_shapes_and_flops():
    """_infer_op_shape prices the carrier by its eventual tensor."""
    from mlopt_core.cost import _shape_of

    s = Var("s", TensorType((4, 6)))
    v = Var("v", TensorType((6, 3)))
    e = Op.make("om_elem", s, v)
    assert _shape_of(e) == (4, 3)
    f = Op.make("om_compose", e, e)
    assert _shape_of(f) == (4, 3)
    out = Op.make("om_apply", f)
    assert _shape_of(out) == (4, 3)
    assert flops_cost(out) > 0
    # om packaging is a view op: zero local cost.
    m = Var("m", TensorType((4, 1)))
    lv = Var("l", TensorType((4, 1)))
    a = Var("a", TensorType((4, 3)))
    triple = Op.make("om", m, lv, a)
    assert _shape_of(triple) == (4, 3)
    assert dag_cost(triple, flops_cost) == 0.0
