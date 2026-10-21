"""Substrate-pure tests ported from ``graphopt:tests/test_e2e_pipeline.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.ops import OpTable


def test_optable_register_fragment_kwargs():
    """dict source + shape_rules/attr_schema fragments compose on one
    table — the fragments land in their registries."""
    t = OpTable()
    t.register(
        {"myop": lambda *a, **k: None},
        shape_rules={"myop": lambda op, shapes: (1,)},
        attr_schema={"myop": {1: "dim"}},
    )
    assert "myop" in t.torch_bindings
    assert "myop" in t.shape_rules
    assert "myop" in t.attr_schemas
