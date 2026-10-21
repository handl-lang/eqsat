"""Substrate-pure tests ported from ``graphopt:tests/test_autotune.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlopt_orchestrator.autotune as at_mod

import pytest

from mlopt_core.ir import Op, Param, TensorType, Var


def _term():
    x = Var("x", TensorType((64,)))
    w = Param("W", TensorType((64, 64)))
    return Op.make("matmul", x, w)

def test_fused_charges_graph_overhead_detection(monkeypatch):
    """When the installed fused model already consumes the field the
    shim must not double-charge — detected behaviourally."""

    def fake_fused(profile):
        ov = (
            profile.get("graph_overhead_us", 0.0)
            if isinstance(profile, dict)
            else 0.0
        )

        def cost(t, memo=None):
            return 100.0 + ov * 1e3

        return cost

    monkeypatch.setattr(at_mod, "fused_cost_for", fake_fused)
    prof = {
        "tflops": 2.5,
        "gbps": 89.0,
        "launch_us": 8.7,
        "dispatch_us": 1.0,
        "graph_overhead_us": 50.0,
    }
    assert at_mod._fused_charges_graph_overhead(prof) is True
    # native consumption detected → the price carries it, no surplus
    assert at_mod._compiled_model_ns(_term(), prof) == pytest.approx(
        100.0 + 50e3
    )

def test_graph_overhead_detection_exotic_profiles():
    """Unpriceable / uncarryable profile types report not-native."""
    # plain object() — clone accepts no attribute → None → False
    assert at_mod._fused_charges_graph_overhead(object()) is False

    @dataclass(frozen=True)
    class Frozen:  # dataclass WITHOUT the field → replace() raises
        x: int = 0

    assert at_mod._fused_charges_graph_overhead(Frozen()) is False

    class Bare:  # clones fine, but has no tflops — pricing raises
        pass

    assert at_mod._fused_charges_graph_overhead(Bare()) is False

    class Obj:  # priceable attribute profile — and the installed
        # fused model now natively consumes graph_overhead_us, so
        # the behavioural probe reports True on it.
        tflops = 2.5
        gbps = 89.0
        launch_us = 8.7
        dispatch_us = 1.0

    assert at_mod._fused_charges_graph_overhead(Obj()) is True
    assert at_mod._compiled_model_ns(_term(), Obj()) > 0.0
