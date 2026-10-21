"""Substrate-pure tests ported from ``graphopt:tests/test_registries.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""


def test_register_carrier_extends_composition(monkeypatch):
    """The extension point (Phase 3): a carrier module registered at
    runtime joins ``full()``/``carrier_torch_bindings()`` without
    editing core's built-in ``_CARRIER_MODULES`` tuple.  Appended in
    registration order, idempotent, and the built-in order is kept."""
    import sys
    from types import SimpleNamespace

    import mlopt_core.ops as ops_mod

    # isolate the registry — monkeypatch restores it afterwards
    monkeypatch.setattr(ops_mod, "_registered_carriers", [])
    fake = SimpleNamespace(
        TORCH_BINDINGS={"mycarrier_op": lambda *a, **k: None}
    )
    monkeypatch.setitem(sys.modules, "my_carrier_pkg", fake)

    ops_mod.register_carrier("my_carrier_pkg")
    ops_mod.register_carrier("my_carrier_pkg")  # idempotent
    names = ops_mod._carrier_names()
    assert names[: len(ops_mod._CARRIER_MODULES)] == ops_mod._CARRIER_MODULES
    assert names.count("my_carrier_pkg") == 1
    # the registered module's bindings join the merged carrier table
    assert "mycarrier_op" in ops_mod.carrier_torch_bindings()
