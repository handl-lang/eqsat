"""Substrate-pure tests ported from ``graphopt:tests/test_synthesis_guards.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core import meta

from mlopt_core.egraph import EGraph, Rewrite

from mlopt_core.ir import Op, TensorType, Var, op_repr


def test_unsatisfiable_parent_check_rejected():
    """r2's check is always False: the composite can never fire, so
    validation must reject the pair instead of emitting a dead — or
    worse, unjustified — rule."""
    src = Rewrite(
        "t_guard_src",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
    )
    dead = Rewrite(
        "t_guard_dead",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "a", "b"),
        check=lambda bound: False,
    )
    derived = meta.synthesize_rules([src, dead], fuel=500)
    assert all(
        "t_guard_dead" not in meta.provenance(d) for d in derived
    )

def test_unproducible_derive_rejected():
    """r2's RHS needs an attr metavar that only derive could supply,
    but derive vetoes — the composition is un-justifiable and must not
    be emitted."""
    src = Rewrite(
        "t_guard_src2",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
    )
    veto = Rewrite(
        "t_guard_veto",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "a", "b", scale="S"),
        derive=lambda bound: None,
    )
    derived = meta.synthesize_rules([src, veto], fuel=500)
    assert all(
        "t_guard_veto" not in meta.provenance(d) for d in derived
    )

def test_check_referencing_missing_metavar_rejected():
    """A parent check that consults a metavar it never binds is
    un-reexpressible-by-construction (the composite binding can never
    supply it) — the pair is rejected."""
    src = Rewrite(
        "t_guard_src3",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
    )
    ghost = Rewrite(
        "t_guard_ghost",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "a", "b"),
        check=lambda bound: bound.get("ghost") is not None,
    )
    derived = meta.synthesize_rules([src, ghost], fuel=500)
    assert all(
        "t_guard_ghost" not in meta.provenance(d) for d in derived
    )
