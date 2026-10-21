"""Substrate-pure tests ported from ``graphopt:tests/test_div_naturality_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.laws import (
    ALL_RULES,
    DIV_ADD_FACTOR,
    MATMUL_DIV_DISTRIBUTE,
    MUL_DIV_SWAP,
    SIMPLIFICATION_RULES,
)


class TestRegistration:
    def test_in_simplification_and_all_rules(self):
        assert MUL_DIV_SWAP in SIMPLIFICATION_RULES
        assert MATMUL_DIV_DISTRIBUTE in SIMPLIFICATION_RULES
        assert MUL_DIV_SWAP in ALL_RULES
        assert MATMUL_DIV_DISTRIBUTE in ALL_RULES

    def test_kernel_kinds(self):
        assert MUL_DIV_SWAP.kind == "axiom"
        assert MATMUL_DIV_DISTRIBUTE.kind == "axiom"
        assert not MUL_DIV_SWAP.derivation
        assert not MATMUL_DIV_DISTRIBUTE.derivation
