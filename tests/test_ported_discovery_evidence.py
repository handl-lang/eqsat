"""Substrate-pure tests ported from ``graphopt:tests/test_discovery_evidence.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import json

import pytest

from mlopt_core.laws import ALL_RULES

from mlopt_core.laws.serialize import (
    alpha_key,
    object_from_data,
    object_kind,
    object_to_data,
)


_BY_NAME = {r.name: r for r in ALL_RULES}

def test_object_codec_marks_and_validates_kind():
    """The record codec: kind stamped, defaulted, and validated."""
    rule = _BY_NAME["id_add"]
    data = object_to_data(rule, kind="law")
    assert data["kind"] == "law" and object_kind(data) == "law"
    # a non-law kind stamps provenance; every kind is Rewrite-shaped
    abs_data = object_to_data(rule, kind="abstraction")
    assert abs_data["kind"] == "abstraction"
    assert object_from_data(json.loads(json.dumps(abs_data))).name == (
        "id_add"
    )
    # a legacy record — no "kind" key — reads as a law
    del abs_data["kind"]
    assert object_kind(abs_data) == "law"
    assert object_from_data(abs_data).name == "id_add"
    # unknown kinds fail loudly at both ends — never a silent admit
    with pytest.raises(ValueError, match="unknown object kind"):
        object_to_data(rule, kind="widget")
    with pytest.raises(ValueError, match="unknown object kind"):
        object_from_data({"kind": "widget", "version": 2})
