"""Substrate-pure tests ported from ``graphopt:tests/test_meter.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import pytest

from mlopt_core.failures import FailureClass, is_nonfinite

from mlopt_core.ports import Meter, TimingResult


def test_timing_result_defaults():
    """A bare three-field measurement carries no provenance."""
    r = TimingResult(median_s=0.5, iqr_s=0.1, n_calls=10)
    assert r.device is None
    assert r.warmup == 0
    assert r.failure is FailureClass.OK

def test_timing_result_is_frozen():
    r = TimingResult(0.5, 0.1, 10)
    with pytest.raises(Exception):
        r.device = "cpu"
