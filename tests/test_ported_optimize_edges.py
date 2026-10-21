"""Substrate-pure tests ported from ``graphopt:tests/test_optimize_edges.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import types

import pytest

from mlopt_orchestrator.optimize import OptimizationResourceError, _check_resources, _current_memory_mb, _looks_like_oom, _oom_to_resource_error, discover_alternatives


def test_oom_to_resource_error_adapter():
    @_oom_to_resource_error
    def raises(e):
        raise e

    with pytest.raises(OptimizationResourceError):
        raises(MemoryError("boom"))
    with pytest.raises(OptimizationResourceError):
        raises(RuntimeError("CUDA out of memory"))
    # a pre-existing resource error passes through unchanged
    marker = OptimizationResourceError("already")
    with pytest.raises(OptimizationResourceError) as ei:
        raises(marker)
    assert ei.value is marker
    # ordinary errors are not wrapped
    with pytest.raises(ValueError):
        raises(ValueError("plain"))
    # functools.wraps keeps the signature metadata
    assert raises.__name__ == "raises"

def test_current_memory_mb_and_check_resources():
    mb = _current_memory_mb()
    assert isinstance(mb, float) and mb > 0

    eg = types.SimpleNamespace(n_enodes=50)
    _check_resources(eg, max_enodes=100, max_memory_mb=None)
    with pytest.raises(OptimizationResourceError, match="e-nodes"):
        _check_resources(eg, max_enodes=50, max_memory_mb=None)
    with pytest.raises(OptimizationResourceError, match="e-nodes"):
        _check_resources(eg, max_enodes=10, max_memory_mb=None)
    # a memory bound below the real footprint always trips
    with pytest.raises(OptimizationResourceError, match="memory"):
        _check_resources(eg, max_enodes=None, max_memory_mb=0.0)
    # eg=None skips the enode check entirely
    _check_resources(None, max_enodes=1, max_memory_mb=None)
    _check_resources(eg, max_enodes=None, max_memory_mb=None)
