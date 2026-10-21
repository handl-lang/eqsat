"""Substrate-pure tests ported from ``graphopt:tests/test_lawdata.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import mlopt_core.lawdata as core_data

from mlopt_core import meta

from mlopt_core.opmeta import COMMUTATIVE_OPS, REGISTRY


class TestCoreLawdata:
    def test_coherent_names_is_the_table(self):
        assert meta.COHERENT_RULE_NAMES is core_data.COHERENT_RULE_NAMES

    def test_canonicalize_vocab_is_bound(self):
        assert meta._AC_IDENTITY is core_data.AC_IDENTITY
        assert meta._ASSOC_ONLY is core_data.ASSOC_ONLY

    def test_ac_identity_ops_are_commutative(self):
        assert set(core_data.AC_IDENTITY) <= set(COMMUTATIVE_OPS)

    def test_instantiation_bank_is_bound(self):
        assert meta._LEAF_SHAPES is core_data.INSTANTIATE_LEAF_SHAPES
        assert meta._ATTR_POOL is core_data.INSTANTIATE_ATTR_POOL
        assert all(
            isinstance(s, tuple)
            for s in core_data.INSTANTIATE_LEAF_SHAPES
        )
