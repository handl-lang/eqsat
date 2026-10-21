"""Substrate-pure tests ported from ``graphopt:tests/test_diagram.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import mlopt_orchestrator.diagram as D

from mlopt_core.ir import IR, Op, Param, TensorType, Var

from mlopt_orchestrator import (
    ContractionSearch,
    DiagramMove,
    MorphismMatch,
    Optimizer,
    ReifySpec,
    optimize_diagram,
)


def test_factor_move_protocol():
    """Moves satisfy the DiagramMove protocol."""
    for mv in D.DEFAULT_MOVES:
        assert isinstance(mv, DiagramMove)
        assert mv.name

def test_split_sites_unit():
    """_split_sites guards: view ops only, arg0 must be the Param."""
    w = Param("p_w", TensorType((2, 4, 4)))
    x = Var("x", TensorType((8, 4)))
    site = Op.make("select", w, dim=0, index=1)
    on_var = Op.make("select", x, dim=0, index=1)
    other = Op.make("reshape", w, shape=(8, 4))
    body = Op.make("add", site, on_var, other)
    names = frozenset({"p_w"})
    sites = D._split_sites(body, names)
    assert sites == [site]
    # Param arg under a different name — not the shared leaf
    body2 = Op.make(
        "select",
        Param("p_other", TensorType((2, 4, 4))),
        dim=0,
        index=0,
    )
    assert D._split_sites(body2, names) == []
    # a select with no args at all
    assert D._split_sites(Op.make("select"), names) == []

def test_move_set_default_and_override():
    """DEFAULT_MOVES is the documented claim order; moves override."""
    names = [m.name for m in D.DEFAULT_MOVES]
    assert names == [
        "factor_shared",
        "merge_projs",
        "reorder_compose",
        "split_leaf",
    ]
    strat = ContractionSearch(moves=(D.SplitLeaf(),))
    assert [m.name for m in strat.moves] == ["split_leaf"]

def test_public_exports():
    """The package __init__ exposes the diagram surface."""
    import mlopt_orchestrator as co

    for name in (
        "DEFAULT_MOVES",
        "ContractionSearch",
        "DEdge",
        "DEnd",
        "DNode",
        "Diagram",
        "DiagramMove",
        "FactorShared",
        "MergeProjs",
        "ReorderCompose",
        "SplitLeaf",
        "diagram_of_graph",
        "lift_diagram",
        "optimize_diagram",
    ):
        assert getattr(co, name) is getattr(D, name)
