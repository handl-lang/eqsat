"""``mlopt_core.term_json`` — THE kernel Term-JSON codec (ADR-0173/0178).

Ports ``kit_discovery.tests.test_hnl_codec`` onto the unified codec
and pins the three wire divergences the consolidation had to absorb:

* **leaf spelling** — canonical ``{"Arr": {"value": {"__mlopt_leaf__":
  kind, …}}}`` records vs the legacy ``__mlopt_leaf__:-prefixed
  ``Const`` string (``leaf="const-str"`` profile; decode accepts both
  permanently);
* **dag ``Op.dim``** — node-table *index* (engine-eqsat ``dag.rs``
  schema) emitted, nested ``{"Const": …}`` record accepted on decode;
* **``Op.source``** — ``"Native"`` emitted always (law matching pins
  ``source:"native"`` under exact-keyset attrs); ``{"Foreign": …}``
  decode-accepted and ignored.

Plus the shared machinery both predecessor codecs had: the
``__nonfinite__`` tag, the ``attrs``/``lit.__mlopt_attrs__`` channel
rule, the ``dag.program`` node-table validation, and the
``wire_node_budget`` accounting (per-call int or shared
``[emitted, limit]`` cell across a term list).
"""

from __future__ import annotations

import json
import math

import pytest
from mlopt_core.term import Const, Op, Param, TensorType, Var
from mlopt_core.term_json import (
    TermJsonError,
    term_from_json,
    term_to_json,
    terms_to_json,
)


def _x() -> Var:
    return Var("x", TensorType((2, 3)))


def _w() -> Param:
    return Param("w", TensorType((3, 4)))


def _assert_nan(v: float) -> None:
    assert isinstance(v, float) and math.isnan(v)


# ---------------------------------------------------------------------------
#  Round-trip — the tree envelope
# ---------------------------------------------------------------------------


class TestTreeRoundTrip:
    def test_unshared_term_is_plain_tree(self) -> None:
        t = Op.make("add", _x(), _w())
        j = term_to_json(t)
        assert "Op" in j and "Opaque" not in j
        assert term_from_json(j) == t

    def test_leaf_kinds_round_trip(self) -> None:
        for leaf in (_x(), _w(), Const(3), Const(2.5)):
            assert term_from_json(term_to_json(leaf)) == leaf

    def test_attrs_round_trip(self) -> None:
        t = Op.make(
            "conv2d",
            _x(),
            _w(),
            stride=[1, 1],
            padding=[0, 0],
            groups=1,
        )
        assert term_from_json(term_to_json(t)) == t

    def test_json_safe_output(self) -> None:
        t = Op.make("clamp", _x(), min=-1.0, max=1.0, validate=False)
        # allow_nan=False proves no bare NaN/Infinity literal escaped.
        blob = json.dumps(term_to_json(t), allow_nan=False)
        assert term_from_json(json.loads(blob)) == t

    def test_unsupported_leaf_declines(self) -> None:
        with pytest.raises(TypeError, match="wireable"):
            term_to_json("m")  # a metavar string is not a program

    def test_source_is_native_on_the_wire(self) -> None:
        """Divergence 3: ``"Native"`` is emitted, never Foreign."""
        j = term_to_json(Op.make("neg", _x()))
        assert j["Op"]["source"] == "Native"
        assert "lit" not in j["Op"] and "attrs" not in j["Op"]

    def test_attrs_field_is_canonical(self) -> None:
        """Divergence-adjacent: ``attrs`` field, never ``lit``."""
        t = Op.make("clamp", _x(), min=0.5, validate=False)
        j = term_to_json(t)
        assert j["Op"]["attrs"] == {"min": 0.5}
        assert "lit" not in j["Op"]


# ---------------------------------------------------------------------------
#  Leaf spelling — Arr record canonical, Const-string legacy
# ---------------------------------------------------------------------------


class TestLeafSpellings:
    def test_arr_leaf_record_shape(self) -> None:
        j = term_to_json(_x())
        arr = j["Arr"]
        assert arr["value"] == {
            "__mlopt_leaf__": "var",
            "name": "x",
            "shape": [2, 3],
        }

    def test_const_str_profile_encodes_legacy_spelling(self) -> None:
        j = term_to_json(_x(), leaf="const-str")
        v = j["Const"]["value"]
        assert v.startswith("__mlopt_leaf__:")
        rec = json.loads(v[len("__mlopt_leaf__:") :])
        assert rec == {"k": "var", "n": "x", "s": [2, 3]}

    def test_const_str_profile_round_trips(self) -> None:
        t = Op.make("add", _x(), _w())
        for leaf in ("arr", "const-str"):
            j = term_to_json(t, dag=True, leaf=leaf)
            assert term_from_json(j, validate=False) == t

    def test_decode_accepts_both_spellings(self) -> None:
        """Decode is permanently auto — old records keep reading."""
        x = _x()
        arr = term_to_json(x)
        cstr = term_to_json(x, leaf="const-str")
        assert term_from_json(arr) == x
        assert term_from_json(cstr) == x
        # …inside dag node tables too
        assert term_from_json(
            term_to_json(Op.make("neg", x), dag=True, leaf="const-str")
        ) == Op.make("neg", x)

    def test_decode_leaf_param_is_encode_only(self) -> None:
        with pytest.raises(TermJsonError, match="encode-only"):
            term_from_json(term_to_json(_x()), leaf="arr")

    def test_unknown_leaf_profile_declines(self) -> None:
        with pytest.raises(TermJsonError, match="leaf profile"):
            term_to_json(_x(), leaf="bogus")

    def test_bad_const_str_payload_declines(self) -> None:
        with pytest.raises(TermJsonError, match="leaf record"):
            term_from_json(
                {"Const": {"value": "__mlopt_leaf__:{bad json"}}
            )
        with pytest.raises(TermJsonError, match="leaf record"):
            term_from_json({"Const": {"value": "__mlopt_leaf__:[1,2]"}})
        with pytest.raises(TermJsonError, match="leaf kind"):
            term_from_json(
                {
                    "Const": {
                        "value": "__mlopt_leaf__:"
                        + json.dumps({"k": "?", "n": "x", "s": None})
                    }
                }
            )

    def test_const_string_resembling_prefix_stays_const(self) -> None:
        back = term_from_json({"Const": {"value": "__mlopt_leaf__x"}})
        assert (
            isinstance(back, Const) and back.value == "__mlopt_leaf__x"
        )

    def test_arr_leaf_bad_kind_declines(self) -> None:
        j = {
            "Arr": {
                "dim": {"Const": {"value": 1}},
                "value": {
                    "__mlopt_leaf__": "widget",
                    "name": "x",
                    "shape": [2],
                },
            }
        }
        with pytest.raises(TermJsonError, match="leaf kind"):
            term_from_json(j)


# ---------------------------------------------------------------------------
#  ``__nonfinite__`` — the codec tag
# ---------------------------------------------------------------------------


class TestNonfinite:
    @pytest.mark.parametrize(
        "value,check",
        [
            (float("nan"), _assert_nan),
            (float("inf"), lambda v: v == float("inf")),
            (float("-inf"), lambda v: v == float("-inf")),
        ],
    )
    def test_const_nonfinite_round_trip(self, value, check) -> None:
        t = Const(value)
        j = term_to_json(t)
        wire = j["Const"]["value"]
        assert isinstance(wire, dict) and set(wire) == {"__nonfinite__"}
        json.dumps(j, allow_nan=False)
        back = term_from_json(j)
        assert isinstance(back, Const)
        check(back.value)

    def test_nonfinite_in_attrs(self) -> None:
        t = Op.make(
            "clamp",
            _x(),
            min=float("nan"),
            max=float("inf"),
            validate=False,
        )
        j = term_to_json(t)
        attrs = j["Op"]["attrs"]
        assert attrs["min"] == {"__nonfinite__": "nan"}
        assert attrs["max"] == {"__nonfinite__": "inf"}
        json.dumps(j, allow_nan=False)
        back = term_from_json(j, validate=False)
        assert back == t
        _assert_nan(back.attrs["min"])
        assert back.attrs["max"] == float("inf")

    def test_nonfinite_nested_in_attr_containers(self) -> None:
        t = Op.make(
            "pad",
            _x(),
            pad=(float("-inf"), 1, 2),
            validate=False,
        )
        j = term_to_json(t)
        wire = j["Op"]["attrs"]["pad"]
        assert wire["__tuple__"][0] == {"__nonfinite__": "-inf"}
        back = term_from_json(j, validate=False)
        assert back == t
        assert back.attrs["pad"][0] == float("-inf")

    def test_decode_handwritten_tag(self) -> None:
        j = {"Const": {"value": {"__nonfinite__": "-inf"}}}
        back = term_from_json(j)
        assert isinstance(back, Const)
        assert back.value == float("-inf")

    def test_bad_tag_payload_declines(self) -> None:
        j = {"Const": {"value": {"__nonfinite__": "bogus"}}}
        with pytest.raises(TermJsonError, match="__nonfinite__"):
            term_from_json(j)

    def test_finite_floats_untouched(self) -> None:
        t = Op.make("clamp", _x(), min=0.5, max=-2.25, validate=False)
        j = term_to_json(t)
        assert j["Op"]["attrs"] == {"min": 0.5, "max": -2.25}
        assert term_from_json(j, validate=False) == t

    def test_nonfinite_inside_dag_record(self) -> None:
        nan = Const(float("nan"))
        shared = Op.make("add", _x(), nan, validate=False)
        root = Op.make("mul", shared, shared)
        j = term_to_json(root)
        rec = j["Opaque"]
        consts = [n for n in rec["nodes"] if "Const" in n]
        assert consts[0]["Const"]["value"] == {"__nonfinite__": "nan"}
        back = term_from_json(j, validate=False)
        assert back == root
        _assert_nan(back.args[0].args[1].value)


# ---------------------------------------------------------------------------
#  ``dag.program`` — the sharing envelope
# ---------------------------------------------------------------------------


def _shared_term() -> Op:
    """A small DAG: ``s1`` feeds three parents, ``s2`` two."""
    s1 = Op.make("add", _x(), _w())
    s2 = Op.make("mul", s1, s1)
    s3 = Op.make("sub", s2, s1)
    return Op.make("mul", s3, s2)


class TestDagRecord:
    def test_shared_term_encodes_as_record(self) -> None:
        t = _shared_term()
        j = term_to_json(t)
        assert set(j) == {"Opaque"}
        rec = j["Opaque"]
        assert rec["__tag__"] == "dag.program"
        assert isinstance(rec["nodes"], list)
        assert isinstance(rec["root"], int)
        # compactness: distinct nodes vs the tree's spellings
        assert len(rec["nodes"]) <= 8
        for i, n in enumerate(rec["nodes"]):
            if "Op" in n:
                for a in n["Op"]["args"]:
                    assert isinstance(a, int) and 0 <= a < i
                # Divergence 2: dim is a node-table INDEX, not a
                # nested {"Const": …} record (the engine-eqsat schema).
                assert isinstance(n["Op"]["dim"], int)
                assert 0 <= n["Op"]["dim"] < i

    def test_dim_index_points_at_const_one(self) -> None:
        j = term_to_json(_shared_term())
        nodes = j["Opaque"]["nodes"]
        for n in nodes:
            if "Op" in n:
                dim_node = nodes[n["Op"]["dim"]]
                assert dim_node == {"Const": {"value": 1}}

    def test_dag_round_trip_preserves_sharing(self) -> None:
        t = _shared_term()
        j = term_to_json(t)
        back = term_from_json(j)
        assert back == t
        seen: dict[int, int] = {}

        def count(n: object) -> None:
            if isinstance(n, Op):
                for a in n.args:
                    seen[id(a)] = seen.get(id(a), 0) + 1
                    count(a)

        count(back)
        shared_ids = {i for i, c in seen.items() if c > 1}
        assert len(shared_ids) >= 2
        # re-encoding the decode is a fixed point — same record
        assert term_to_json(back) == j

    def test_unshared_term_stays_tree_under_auto(self) -> None:
        t = Op.make("add", Op.make("mul", _x(), _w()), Const(1))
        j = term_to_json(t)
        assert "Op" in j

    def test_dag_true_forces_record(self) -> None:
        t = Op.make("add", _x(), _w())
        j = term_to_json(t, dag=True)
        assert j["Opaque"]["__tag__"] == "dag.program"
        assert term_from_json(j) == t

    def test_dag_false_declines_shared_term(self) -> None:
        with pytest.raises(TermJsonError, match="shared subtrees"):
            term_to_json(_shared_term(), dag=False)

    def test_dag_false_allows_unshared_term(self) -> None:
        t = Op.make("add", _x(), _w())
        assert term_from_json(term_to_json(t, dag=False)) == t

    def test_allow_dag_false_declines_record(self) -> None:
        j = term_to_json(_shared_term())
        with pytest.raises(TermJsonError, match="cannot represent"):
            term_from_json(j, allow_dag=False)

    def test_unknown_opaque_tag_declines(self) -> None:
        j = {"Opaque": {"__tag__": "something.else", "data": 1}}
        with pytest.raises(TermJsonError, match="Opaque"):
            term_from_json(j)

    def test_untyped_opaque_declines(self) -> None:
        with pytest.raises(TermJsonError, match="Opaque"):
            term_from_json({"Opaque": {"raw": True}})

    def test_non_dict_encoding_declines(self) -> None:
        with pytest.raises(TermJsonError, match="bad term encoding"):
            term_from_json(["not", "a", "dict"])

    def test_unknown_node_kind_declines(self) -> None:
        with pytest.raises(TermJsonError, match="undecodable"):
            term_from_json({"mvar": "x"})

    def test_unencodable_leaf_in_dag_declines(self) -> None:
        t = Op.make("add", "m", "m", validate=False)
        with pytest.raises(TypeError, match="wireable"):
            term_to_json(t)
        with pytest.raises(TypeError, match="wireable"):
            term_to_json(t, dag=True)

    def test_unserializable_const_declines(self) -> None:
        with pytest.raises(TypeError, match="unserializable"):
            term_to_json(Const({1, 2}))  # type: ignore[arg-type]

    def test_dag_node_args_not_list_declines(self) -> None:
        j = {
            "Opaque": {
                "__tag__": "dag.program",
                "nodes": [{"Op": {"name": "neg"}}],
                "root": 0,
            }
        }
        with pytest.raises(TermJsonError, match="index list"):
            term_from_json(j)

    def test_leaf_only_dag_round_trip(self) -> None:
        j = term_to_json(_w(), dag=True)
        assert j["Opaque"]["__tag__"] == "dag.program"
        assert term_from_json(j) == _w()

    def test_dag_node_with_attrs_round_trip(self) -> None:
        s = Op.make("clamp", _x(), min=0.25, validate=False)
        t = Op.make("mul", s, s)
        j = term_to_json(t)
        ops = [n["Op"] for n in j["Opaque"]["nodes"] if "Op" in n]
        assert any("attrs" in o for o in ops)
        assert term_from_json(j, validate=False) == t

    def test_nested_non_dict_arg_declines(self) -> None:
        j = {"Op": {"name": "neg", "args": ["x"]}}
        with pytest.raises(TermJsonError, match="bad term encoding"):
            term_from_json(j)

    def test_nested_opaque_declines(self) -> None:
        inner = {
            "Opaque": {"__tag__": "dag.program", "nodes": [], "root": 0}
        }
        j = {
            "Op": {
                "name": "add",
                "args": [inner, inner],
                "source": {"Foreign": "tensor"},
            }
        }
        with pytest.raises(TermJsonError, match="nested Opaque"):
            term_from_json(j)


class TestDagRecordValidation:
    """Malformed node tables decline loudly, never guess."""

    def _rec(self, **kw):
        rec = {"__tag__": "dag.program", "nodes": [], "root": 0}
        rec.update(kw)
        return {"Opaque": rec}

    def test_root_out_of_range(self) -> None:
        j = self._rec(nodes=[{"Const": {"value": 1}}], root=5)
        with pytest.raises(TermJsonError, match="root"):
            term_from_json(j)

    def test_root_not_int(self) -> None:
        j = self._rec(nodes=[{"Const": {"value": 1}}], root="0")
        with pytest.raises(TermJsonError, match="root"):
            term_from_json(j)

    def test_nodes_not_list(self) -> None:
        j = self._rec(nodes={"0": {}}, root=0)
        with pytest.raises(TermJsonError, match="nodes"):
            term_from_json(j)

    def test_arg_index_not_earlier(self) -> None:
        j = self._rec(
            nodes=[
                {
                    "Op": {
                        "name": "neg",
                        "args": [0],
                        "source": {"Foreign": "tensor"},
                    }
                }
            ],
            root=0,
        )
        with pytest.raises(TermJsonError, match="earlier node index"):
            term_from_json(j)

    def test_arg_index_not_int(self) -> None:
        j = self._rec(
            nodes=[
                {"Const": {"value": 1}},
                {
                    "Op": {
                        "name": "neg",
                        "args": ["0"],
                        "source": {"Foreign": "tensor"},
                    }
                },
            ],
            root=1,
        )
        with pytest.raises(TermJsonError, match="earlier node index"):
            term_from_json(j)

    def test_unknown_node_tag(self) -> None:
        j = self._rec(nodes=[{"Widget": {"x": 1}}], root=0)
        with pytest.raises(TermJsonError, match="unknown tag"):
            term_from_json(j)

    def test_node_not_single_tagged_record(self) -> None:
        j = self._rec(
            nodes=[{"Const": {"value": 1}, "extra": 2}], root=0
        )
        with pytest.raises(TermJsonError, match="tagged record"):
            term_from_json(j)

    def test_nested_dim_record_decodes(self) -> None:
        """Divergence 2 (decode side): the legacy hnl_codec dag
        spelling — ``dim`` as a nested ``{"Const": …}`` record — is
        accepted and ignored (mlopt terms carry no dim)."""
        j = {
            "Opaque": {
                "__tag__": "dag.program",
                "nodes": [
                    {"Const": {"value": 1}},
                    {
                        "Op": {
                            "dim": {"Const": {"value": 1}},
                            "name": "neg",
                            "args": [0],
                            "source": {"Foreign": "tensor"},
                        }
                    },
                ],
                "root": 1,
            }
        }
        assert term_from_json(j) == Op.make("neg", Const(1))

    def test_bad_dim_index_declines(self) -> None:
        j = {
            "Opaque": {
                "__tag__": "dag.program",
                "nodes": [
                    {"Const": {"value": 1}},
                    {
                        "Op": {
                            "dim": 5,
                            "name": "neg",
                            "args": [0],
                            "source": "Native",
                        }
                    },
                ],
                "root": 1,
            }
        }
        with pytest.raises(TermJsonError, match="dim"):
            term_from_json(j)


# ---------------------------------------------------------------------------
#  The ``attrs`` field — canonical, ``lit`` legacy
# ---------------------------------------------------------------------------


class TestAttrsField:
    def test_attrs_field_decodes(self) -> None:
        j = {
            "Op": {
                "name": "clamp",
                "args": [],
                "attrs": {"min": {"__nonfinite__": "nan"}},
            }
        }
        back = term_from_json(j, validate=False)
        assert isinstance(back, Op)
        _assert_nan(back.attrs["min"])

    def test_lit_legacy_attrs_decodes(self) -> None:
        j = {
            "Op": {
                "name": "clamp",
                "args": [],
                "lit": {"__mlopt_attrs__": {"min": 0.0}},
            }
        }
        back = term_from_json(j, validate=False)
        assert back.attrs["min"] == 0.0

    def test_attrs_and_lit_ambiguous_declines(self) -> None:
        j = {
            "Op": {
                "name": "clamp",
                "args": [],
                "attrs": {"min": 0.0},
                "lit": {"__mlopt_attrs__": {"min": 1.0}},
            }
        }
        with pytest.raises(TermJsonError, match="ambiguous"):
            term_from_json(j)

    def test_attrs_null_is_absent(self) -> None:
        j = {
            "Op": {
                "name": "neg",
                "args": [{"Const": {"value": 1}}],
                "attrs": None,
            }
        }
        assert term_from_json(j) == Op.make("neg", Const(1))

    def test_attrs_field_not_object_declines(self) -> None:
        j = {"Op": {"name": "neg", "args": [], "attrs": [1, 2]}}
        with pytest.raises(TermJsonError, match="must be an object"):
            term_from_json(j)

    def test_lit_without_attrs_record_declines(self) -> None:
        j = {"Op": {"name": "neg", "args": [], "lit": {"other": 1}}}
        with pytest.raises(TermJsonError, match="not an attrs record"):
            term_from_json(j)

    def test_attrs_record_not_object_declines(self) -> None:
        j = {
            "Op": {
                "name": "neg",
                "args": [],
                "lit": {"__mlopt_attrs__": 5},
            }
        }
        with pytest.raises(TermJsonError, match="must be an object"):
            term_from_json(j)

    def test_bad_attrs_value_declines(self) -> None:
        j = {
            "Op": {
                "name": "neg",
                "args": [],
                "attrs": {"x": {"bogus": 1}},
            }
        }
        with pytest.raises(TermJsonError, match="bad attrs payload"):
            term_from_json(j)


# ---------------------------------------------------------------------------
#  ``source`` — decode accepts and ignores every ArrSource spelling
# ---------------------------------------------------------------------------


class TestSourceIgnored:
    @pytest.mark.parametrize(
        "source",
        [
            "Native",
            {"Foreign": "tensor"},
            {"Foreign": "onnx"},
            "Judgment",
            {"DynamicForeign": "x"},
            None,
        ],
    )
    def test_source_spellings_decode_identically(self, source) -> None:
        j = {
            "Op": {
                "name": "neg",
                "args": [{"Const": {"value": 1}}],
                "source": source,
            }
        }
        assert term_from_json(j) == Op.make("neg", Const(1))


# ---------------------------------------------------------------------------
#  ``wire_node_budget`` — per-call and shared-cell accounting
# ---------------------------------------------------------------------------


class TestBudget:
    def _doubling(self) -> Op:
        x = _x()
        t = x
        for _ in range(20):
            t = Op.make("add", t, t)
        return t

    def test_shared_budget_cell_bounds_term_list(self) -> None:
        t = self._doubling()
        state = [0, 10]
        with pytest.raises(TermJsonError, match="wire_node_budget"):
            terms_to_json([t], budget=10)
        # a shared [emitted, limit] cell across calls — the _kdag
        # ``state`` convention — bounds the LIST, not each term
        with pytest.raises(TermJsonError, match="wire_node_budget"):
            term_to_json(t, dag=True, budget=state)
        assert state[0] > 10

    def test_budget_counts_distinct_dag_nodes(self) -> None:
        # ~22 distinct nodes encode fine under a generous bound
        out = terms_to_json([self._doubling()], budget=100)
        assert out[0]["Opaque"]["__tag__"] == "dag.program"

    def test_tree_path_counts_expansion(self) -> None:
        # an *unshared* chain so dag=False reaches the tree walker —
        # distinct Const args keep every subtree unique
        t = _x()
        for i in range(5):
            t = Op.make("add", t, Const(i))
        state = [0, 4]
        with pytest.raises(TermJsonError, match="wire_node_budget"):
            term_to_json(t, dag=False, budget=state)

    def test_budget_state_must_be_pair(self) -> None:
        with pytest.raises(TermJsonError, match="budget state"):
            term_to_json(_x(), budget=[0])  # type: ignore[arg-type]
