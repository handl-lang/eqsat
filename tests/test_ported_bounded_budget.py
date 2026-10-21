"""Substrate-pure tests ported from ``graphopt:tests/test_bounded_budget.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_orchestrator import optimize as opt_mod

from mlopt_orchestrator.optimize import (
    Compositional,
    OptimizationResourceError,
    Optimizer,
    _ban_bound_members,
    _bounded_gate,
    _bounded_term,
    _check_budget_engine,
)


def test_lower_verify_over_bound_declines():
    """A delivery whose measured error exceeds the *propagated*
    bound declines — the gate fails closed on bound violation."""
    stats = {
        "error_bounds": [
            {
                "rule": "weight_special#0",
                "bound": 1e-3,
                "norm": "max_abs",
                "output_bound": 1e-3,
            }
        ]
    }

    class R:
        max_abs = 2e-3  # measured output error exceeds the bound
        max_rel = 2e-3
        passed = True

    out = opt_mod._bound_verify(stats, R(), 1e-3, 1e-3)
    assert out.passed is False
    assert stats["error_bounds_honored"] is False
    assert stats["error_bounds"][0]["measured_max_rel"] == 2e-3

def test_lower_verify_unpropagatable_bound_declines():
    """A bound whose norm has no propagation rule — or whose site
    input cannot be measured — fails closed: ``output_bound`` stays
    ``None`` and the gate declines."""
    stats = {
        "error_bounds": [
            {"rule": "r#1", "bound": 1e-3, "norm": "chebyshev"},
            {
                "rule": "r#2",
                "bound": 1e-3,
                "norm": "max_abs",
            },
        ]
    }
    # Unknown norm -> no exponent -> unpropagatable; and a max_abs
    # entry on a non-tensor input has nothing to amplify by.
    out = opt_mod._propagate_bounds(stats, None, None, {}, None, (3.0,))
    assert out is None
    assert stats["error_bound_output"] is None
    e0, e1 = stats["error_bounds"]
    assert e0["output_bound"] is None and e0["site_input"] == "none"
    assert e1["output_bound"] is None and e1["site_input"] == "none"

    class R:
        max_abs = 1e-6
        max_rel = 1e-6
        passed = True

    out2 = opt_mod._bound_verify(stats, R(), 1e-3, out)
    assert out2.passed is False
    assert stats["error_bounds_honored"] is False

def test_bounded_gate_matrix():
    """``_bounded_gate`` = sink pass AND measured ``max|Δy|`` within
    the propagated OUTPUT bound (absolute units — the bound is in
    output space, not the certificate's weight space)."""

    class R:
        def __init__(self, passed, max_abs):
            self.passed = passed
            self.max_abs = max_abs

    assert _bounded_gate(R(True, 0.5), 0.0)
    assert not _bounded_gate(R(False, 0.0), 0.0)
    assert _bounded_gate(R(True, 0.5), 1.0)
    assert not _bounded_gate(R(True, 1.5), 1.0)
    assert not _bounded_gate(R(False, 0.5), 1.0)
    # An unpropagatable bound (None) fails closed.
    assert not _bounded_gate(R(True, 0.5), None)

def test_arm_law_budgets():
    """``error_budget`` arms budget-aware laws left at ``None`` —
    on a copy, never mutating the caller's law — and leaves
    explicitly-budgeted and budget-free laws alone."""
    from mlopt_orchestrator.morphisms import (
        WeightTie,
        _arm_law_budgets,
    )
    from mlopt_orchestrator.morphisms_kv import KVLatentShare

    kv = KVLatentShare()
    (armed,) = _arm_law_budgets([kv], 1e-3)
    assert armed is not kv and armed.budget == 1e-3
    assert kv.budget is None  # the caller's object is untouched

    kv2 = KVLatentShare(budget=5e-4)
    (out,) = _arm_law_budgets([kv2], 1e-3)
    assert out is kv2 and out.budget == 5e-4  # explicit budget wins

    tie = WeightTie()  # no budget attribute at all
    (out2,) = _arm_law_budgets([tie], 1e-3)
    assert out2 is tie and not hasattr(tie, "budget")

    # ``None`` budget: laws pass through untouched.
    assert _arm_law_budgets([kv], None) == (kv,)
