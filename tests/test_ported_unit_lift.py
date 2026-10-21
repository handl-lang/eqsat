"""Substrate-pure tests ported from ``graphopt:tests/test_unit_lift.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core import laws as R

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var, op_repr


_UNIT_RULES = (
    "affd_lift_unit",
    "affd_lift_unit_post",
    "affd_lift_unit_step",
    "affd_lift_unit_step_post",
)

def test_unit_lift_ignores_non_state_adds():
    """``add(matmul(W,x), mul(a,y))`` — neither operand is state-like —
    fires none of the four unit rules.  (The generic mul-form lifts may
    still see ``mul(a, leaf)``; that permissiveness predates this law
    and is asserted separately.)"""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    W = Param("W", TensorType((4, 4)))
    a = Param("a", TensorType((4,)))
    term = Op.make("add", Op.make("matmul", W, x), Op.make("mul", a, y))

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)

    for name in _UNIT_RULES:
        assert eg.rule_fires.get(name, 0) == 0

def test_unit_lift_ignores_add_of_increments():
    """``add(select(x,0,t), select(x,0,s))`` — two per-step increments,
    no state — fires no unit rule.  A Const operand is likewise not a
    state: ``add(Const, x)`` stays put."""
    x = Var("x", TensorType((4, 4)))
    term = Op.make(
        "add",
        Op.make("select", x, dim=0, index=0),
        Op.make("select", x, dim=0, index=1),
    )
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)
    for name in _UNIT_RULES:
        assert eg.rule_fires.get(name, 0) == 0

    term2 = Op.make(
        "add", Const(2.0), Op.make("select", x, dim=0, index=0)
    )
    eg2 = EGraph()
    r2 = eg2.add_term(term2)
    eg2.run(R.SCAN_DIAG_LAWS, r2, max_iterations=8)
    for name in _UNIT_RULES:
        assert eg2.rule_fires.get(name, 0) == 0

def test_unit_lift_admits_leaf_state():
    """The guard admits a leaf h operand — that is how ``add(h0, x1)``
    enters the carrier.  Fires exactly the base rule (no step variant:
    no applyd precedes it)."""
    x = Var("x", TensorType((4, 4)))
    h0 = Param("h0", TensorType((4,)))
    term = Op.make("add", h0, Op.make("select", x, dim=0, index=0))

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)

    assert eg.rule_fires.get("affd_lift_unit", 0) == 1
    assert eg.rule_fires.get("affd_lift_unit_step", 0) == 0
