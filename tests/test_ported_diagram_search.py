"""Substrate-pure tests ported from ``graphopt:tests/test_diagram_search.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_orchestrator import (
    ContractionSearch,
    MorphismMatch,
    Optimizer,
    ReifySpec,
)


def test_search_bad_mode():
    """An unknown mode is a constructor-time ``ValueError``."""
    import pytest

    with pytest.raises(ValueError, match="greedy"):
        ContractionSearch(mode="bogus")
