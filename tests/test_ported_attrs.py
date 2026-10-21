"""Substrate-pure tests ported from ``graphopt:tests/test_attrs.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.attrs import (
    ATTR_REQUIRED,
    ATTR_SCHEMA,
    attr_of,
)

from mlopt_core.ir import Op, TensorType, Var


def _var(shape: tuple, name: str) -> Var:
    return Var(name=name, typ=TensorType(tuple(shape)))

class TestAttrOf:
    def test_dict_source_first_name_wins(self):
        attrs = {"dim": -2, "arg1": 0}
        assert attr_of(attrs, "dim", "arg1") == -2
        assert attr_of(attrs, "arg1", "dim") == 0

    def test_dict_source_fallback_and_default(self):
        assert attr_of({"arg1": 3}, "dim", "arg1") == 3
        assert attr_of({}, "dim", "arg1", default=-1) == -1
        assert attr_of({}, "dim") is None

    def test_term_source_reads_attrs(self):
        t = Op.make("concat", self.x, self.x, dim=-2)
        assert attr_of(t, "dim", "arg1") == -2
        t = Op.make("concat", self.x, self.x, arg1=0)
        assert attr_of(t, "dim", "arg1") == 0

    def test_present_none_is_a_hit_not_a_fallback(self):
        # ``d.get(k, ...)`` returns the stored None — first-PRESENT
        # wins, not first-truthy.
        assert attr_of({"dim": None}, "dim", "arg1", default=9) is None

    def test_non_term_non_dict_source_is_default(self):
        assert attr_of(object(), "dim", "arg1", default=7) == 7
        assert attr_of(None, "dim", default=5) == 5

    x = _var((2, 8), "x")

class TestSchemaHygiene:
    def test_required_names_are_schema_names(self):
        for op, req in ATTR_REQUIRED.items():
            names = set(ATTR_SCHEMA[op].values())
            assert req <= names, (
                f"{op}: required {sorted(req)} not all canonical names"
            )

    def test_no_duplicate_canonical_names_per_op(self):
        for op, schema in ATTR_SCHEMA.items():
            names = list(schema.values())
            assert len(names) == len(set(names)), (
                f"{op}: duplicate canonical names in schema"
            )
