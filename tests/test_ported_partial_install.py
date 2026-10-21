"""Substrate-pure tests ported from ``graphopt:tests/test_partial_install.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import importlib

from types import SimpleNamespace

import mlopt_core.ops as ops_mod

import pytest

from mlopt_core.ops import OpTable


_real_import_module = importlib.import_module

def _import_blocker(blocked: str, missing: str | None = None):
    """An ``import_module`` stand-in that reports ``blocked`` as
    absent — under ``missing``'s name (defaults to the blocked module
    itself; a DIFFERENT name simulates a failed import inside the
    carrier, not a missing carrier)."""

    def fake(name, *a, **kw):
        if name == blocked:
            raise ModuleNotFoundError(
                f"No module named {missing or name!r}",
                name=missing or name,
            )
        return _real_import_module(name, *a, **kw)

    return SimpleNamespace(import_module=fake)

def test_full_reraises_errors_inside_a_carrier(monkeypatch):
    """The skip is narrowed by ``ModuleNotFoundError.name``: a missing
    import inside an installed carrier (reported under the missing
    dependency's name, not the carrier's) still propagates."""
    monkeypatch.setattr(
        ops_mod,
        "importlib",
        _import_blocker("mlopt_carriers.om", missing="torch"),
    )
    with pytest.raises(ModuleNotFoundError, match="torch"):
        OpTable.full()
