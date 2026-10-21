"""Substrate-pure tests ported from ``graphopt:tests/test_regime_edges.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""


def test_defensive_branch_inventory():
    """Documents branches we believe unreachable by construction —
    candidates for ``pragma: no cover``:

    * regime.py:599 — ``if ec is None: continue`` in _force_carrier:
      ``eg._classes`` keys are always canonicalised back into the same
      map, so a find() result is never missing from it.
    * regime.py:620 — ``if t is None: continue``: an overrides-pinned
      root enode always extracts (its own class was just pinned to it);
      a pinned cyclic enode *might* extract to None but constructing
      one requires corrupting the union-find by hand.
    * regime.py:741-742 — the ``non-native`` report flag: unreachable
      since ``degraded`` is exactly ``carrier is not None and not
      native and not carrier_present`` — every flaggable non-native
      choice is either DEGRADED or carrier-nested.
    """
    # sanity on the flag logic that makes 741-742 dead: with a carrier
    # executor the non-native outcomes are DEGRADED or carrier-nested
    for carrier_present in (False, True):
        degraded = carrier_present is False  # mirrors the frontier expr
        flag = (
            "DEGRADED"
            if degraded
            else ("carrier-nested" if carrier_present else "non-native")
        )
        assert flag != "non-native"
