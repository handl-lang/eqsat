"""Substrate-pure tests ported from ``graphopt:tests/test_crossblock_cse.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import mlopt_orchestrator.crossblock_cse as C

import mlopt_orchestrator.morphisms as M

import mlopt_orchestrator.morphisms_kv as K


def test_share_components():
    """Union-find over share links, ordered by execution position."""
    order = ["a", "b", "c", "d"]
    comps = C._share_components(
        [("a", "b"), ("b", "c"), ("c", "a")], order
    )
    assert comps == [("a", "b", "c")]
    comps = C._share_components([("b", "c"), ("a", "d")], order)
    assert comps == [("a", "d"), ("b", "c")]
    assert C._share_components([], order) == []

def test_exports():
    """The law re-exports through morphisms and the package surface."""
    import mlopt_orchestrator as O

    assert M.CrossBlockCSE is C.CrossBlockCSE
    assert O.CrossBlockCSE is C.CrossBlockCSE

def test_kv_machinery_reused():
    """The pass reuses the morphisms_kv evidence/family helpers."""
    assert C._input_families is K._input_families
    assert C._family_prep is K._family_prep
    assert C._family_evidence is K._family_evidence
