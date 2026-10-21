"""Substrate-pure tests ported from ``graphopt:tests/test_optimize_routing.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.cost import flops_cost

from mlopt_core.ir import IR, Op, Param, TensorType, Var


def _T(shape):
    return TensorType(tuple(shape))

def _P(name, shape):
    return Param(name=name, typ=_T(shape))

def test_delivered_cost_declines_non_plannable_carrier():
    """A carrier-rooted term whose batched plan can't be built prices
    as generic — the module would degrade to serial eval anyway."""
    from mlopt_core.cost import executor_cost_for
    from mlopt_orchestrator.optimize import _delivered_cost



    h = Var("h", _T((2,)))
    t = Op.make(
        "applyd",
        Op.make(
            "affd_compose",
            Op.make("aff_diag", _P("a2", (2,)), _P("b2", (2,))),
            Op.make("aff_diag", _P("a4", (4,)), _P("b4", (4,))),
        ),
        h,
    )
    assert _delivered_cost(t) == executor_cost_for(
        lowering="generic"
    )(t)

def test_carrier_upgrade_returns_incumbent_when_no_carriers():
    """No carrier enodes at root -> incumbent returned unchanged."""
    from mlopt_core.egraph import EGraph
    from mlopt_orchestrator.optimize import _carrier_upgrade



    x = Var("x", _T((3,)))
    t = Op.make("add", x, _P("p_w", (3,)))
    eg = EGraph()
    root = eg.add_term(t)
    assert _carrier_upgrade(eg, root, t, flops_cost) is t
