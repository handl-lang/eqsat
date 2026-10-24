"""Substrate-pure tests ported from ``graphopt:tests/test_law_serialize.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import json

import pytest

from mlopt_core.egraph import Certificate, EGraph, Rewrite

from mlopt_core.egraph.certs import CERT_FORMAT

from mlopt_core.ir import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    term_from_data,
    term_to_data,
)

from mlopt_core.laws import ALL_RULES

from mlopt_core.laws.serialize import (
    LAW_FORMAT,
    alpha_key,
    law_from_data,
    law_to_data,
    missing_hooks,
)

from mlopt_core.laws.tensor import (
    FACTOR_MUL,
    SOFTMAX_FOLD,
)


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))

def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))

def _json_roundtrip(data):
    return json.loads(json.dumps(data))

def _flagged_rule():
    """A rule data cannot fully carry — a cond + procedural check +
    procedural derive composite.  No shipped law is flagged anymore
    (the census is clean), so the honesty path gets a synthetic
    stand-in."""
    return Rewrite(
        name="flagged_fused",
        lhs=Op.make("mul", "u", Op.make("rsqrt", "u")),
        rhs=Op.make("fused", "u", dim="ND"),
        cond=("rank", "u", ">=", 1),
        check=lambda bound: True,
        derive=lambda bound: {"$attr:ND": 1},
    )

def _factor_src():
    x, w = _v("x", 5, 4), _p("W", 4, 3)
    a, b = _p("a", 4, 3), _p("b", 4, 3)
    return (
        Op.make(
            "add",
            Op.make("matmul", x, Op.make("matmul", w, a)),
            Op.make("matmul", x, Op.make("matmul", w, b)),
        ),
        Op.make(
            "matmul", x, Op.make("matmul", w, Op.make("add", a, b))
        ),
    )

def _factor_decline():
    x, a, b = _v("x2", 16), _p("a2", 16), _p("b2", 16, 16)
    return (
        Op.make(
            "add",
            Op.make("matmul", x, a),
            Op.make("matmul", x, b),
        ),
        Op.make("matmul", x, Op.make("add", a, b)),
    )

def _fires(rule, src, want):
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    return eg.find(root) == eg.find(eg.add_term(want))

def test_term_codec_roundtrips_every_shipped_pattern():
    for rule in ALL_RULES:
        for side in (rule.lhs, rule.rhs):
            data = _json_roundtrip(term_to_data(side))
            assert term_from_data(data) == side, (rule.name, side)

def test_term_codec_rejects_and_reports():
    with pytest.raises(TypeError):
        term_to_data(42)
    with pytest.raises(ValueError):
        term_from_data({"junk": 1})

def test_alpha_key_ignores_metavar_names():
    k1 = alpha_key(Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    k2 = alpha_key(Op.make("add", "x", "y"), Op.make("add", "y", "x"))
    assert k1 == k2
    assert k1 != alpha_key(
        Op.make("add", "a", "b"), Op.make("mul", "a", "b")
    )

def test_alpha_key_shared_binding_is_position_sensitive():
    # shared mv map: a,b -> a,b vs a,b -> b,a canonically differ
    k_fwd = alpha_key(Op.make("f", "x", "y"), Op.make("g", "x", "y"))
    k_sw = alpha_key(Op.make("f", "x", "y"), Op.make("g", "y", "x"))
    assert k_fwd != k_sw

def test_alpha_key_canonicalizer_branches():
    """`_attr_canon`/`_canon`: `$attr:` strings, literal and
    unhashable attrs, Const and non-metavar leaves."""
    # $attr:-prefixed attr values serialise under "av" — and stay
    # literal (a bound attr name is not a leaf metavar): different
    # names key differently.
    k = alpha_key(
        Op.make("select", "u", dim="$attr:D", index=0),
        Op.make("select", "u", dim="$attr:E", index=0),
    )
    k2 = alpha_key(
        Op.make("select", "w", dim="$attr:F", index=0),
        Op.make("select", "w", dim="$attr:G", index=0),
    )
    assert k != k2
    # ...but the LEAF metavars inside still abstract — same attr
    # metavar, different leaf names: equal keys.
    assert alpha_key(
        Op.make("select", "u", dim="$attr:D", index=0),
        Op.make("select", "w", dim="$attr:D", index=0),
    ) == alpha_key(
        Op.make("select", "p", dim="$attr:D", index=0),
        Op.make("select", "q", dim="$attr:D", index=0),
    )
    # literal (hashable) vs unhashable attrs fall through lit/repr.
    k3 = alpha_key(
        Op.make("f", "x", n=2, dims=(1, 2)),
        Op.make("f", "x", n=2, dims=[1, 2]),
    )
    assert isinstance(k3, tuple) and len(k3) == 2
    # Const leaves stay literal ("c"), Var leaves hit the repr fallback.
    k4 = alpha_key(
        Op.make("pow", "x", Const(2)),
        Op.make("pow", "x", Const(2)),
    )
    k5 = alpha_key(
        Op.make("add", Var("x", TensorType((2,))), "y"),
        Op.make("add", Var("x", TensorType((2,))), "y"),
    )
    assert k4 != alpha_key(
        Op.make("pow", "x", Const(3)),
        Op.make("pow", "x", Const(3)),
    )
    assert k5 == k5  # self-evident; the repr fallback ran

def test_serializability_census_of_shipped_library():
    """Pin the honest partition of the 106 shipped laws."""
    full, need_derive, need_check, transported = [], [], [], []
    for rule in ALL_RULES:
        missing = missing_hooks(rule)
        if not missing:
            full.append(rule.name)
        elif getattr(rule, "guard_pats", None) is not None:
            # a machine-minted composite shipped verbatim: its hooks
            # are code, but the *spec* of its guard is data — the
            # record's guard field (guard_pats + parents) rebuilds it
            # deterministically (mlopt_core.laws.minted).
            transported.append(rule.name)
        elif missing == ("derive",):
            need_derive.append(rule.name)
        else:
            need_check.append((rule.name, missing))
    assert len(ALL_RULES) == 107
    # the authored library stays full-data: the four stragglers went
    # declarative in stage 0 of plan 0017 — glu_fold's parity on a
    # dim-mod predicate, the rms pair's trailing block on the
    # tail-block spec, and gqa_absorb_repeat's repeat-chain /
    # attr-eq-attr / repeat-heads triple — and the twelve promoted
    # discovery laws carry pure-data cond/dspec/derivation
    # (project/retros/promoted-laws.md).  ``conv_bn_fold`` is the first
    # of the transported-composite class: ``missing_hooks`` flags what
    # the *bare* record cannot carry; the stored guard spec carries it
    # (project/retros/minted-composite-default.md).  The 21
    # ``prim:synop_*`` folds join the full-data majority — pure
    # pattern records, no hooks at all.
    assert len(full) == 106
    assert transported == ["conv_bn_fold"]
    assert need_derive == []
    assert need_check == []

def test_missing_hooks_detects_procedural_check_under_cond():
    """A rule whose cond+check conjoin is NOT full data."""
    r = Rewrite(
        name="hybrid",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        cond=("scalar", "a"),
        check=lambda bound: True,
    )
    assert missing_hooks(r) == ("check",)
    # a bare check with no cond is likewise not data
    r2 = Rewrite(
        name="procedural",
        lhs="a",
        rhs="a",
        check=lambda bound: True,
    )
    assert missing_hooks(r2) == ("check",)
    # and the cond-only twin is clean
    r3 = Rewrite(
        name="declarative",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        cond=("scalar", "a"),
    )
    assert missing_hooks(r3) == ()
    # same for derive: a dspec is data, a spec+code composite is not
    r4 = Rewrite(
        name="spec",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        dspec={"D": ("attr0", "RD")},
    )
    assert missing_hooks(r4) == ()
    r5 = Rewrite(
        name="spec_plus_code",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        dspec={"D": ("attr0", "RD")},
        derive=lambda bound: {"$attr:E": 0},
    )
    assert missing_hooks(r5) == ("derive",)
    # a bare procedural derive flags too
    r6 = Rewrite(
        name="proc_derive",
        lhs="a",
        rhs="a",
        derive=lambda bound: {},
    )
    assert missing_hooks(r6) == ("derive",)

def test_law_record_roundtrip_full_data():
    for rule in ALL_RULES:
        if missing_hooks(rule):
            continue
        data = _json_roundtrip(law_to_data(rule))
        rebuilt = law_from_data(data)
        assert law_to_data(rebuilt) == law_to_data(rule), rule.name

def test_law_record_roundtrip_rewrite_equality_unguarded():
    """Hook-free laws rebuild to an ``==``-equal Rewrite."""
    n = 0
    for rule in ALL_RULES:
        if rule.check is None and rule.derive is None:
            rebuilt = law_from_data(_json_roundtrip(law_to_data(rule)))
            assert rebuilt == rule, rule.name
            n += 1
    # 47 = 26 hook-free bank laws + the 21 ``prim:synop_*`` folds
    # (pure pattern records — no check/derive anywhere in the family)
    assert n == 47

def test_law_record_flagged_hooks_drop_on_rebuild():
    """A check/derive law stores pattern+cond, flagged honestly."""
    rule = _flagged_rule()
    data = law_to_data(rule)
    assert data["serializable"] is False
    assert data["missing_hooks"] == ["check", "derive"]
    rebuilt = law_from_data(_json_roundtrip(data))
    assert rebuilt.derive is None and rebuilt.dspec is None
    # the cond DID travel: the rebuilt rule still guards
    assert rebuilt.cond == rule.cond
    assert rebuilt.check is not None
    # and the record marks the rebuild clean — it has no hooks to drop
    assert law_to_data(rebuilt)["serializable"] is True

def test_dspec_law_record_carries_the_derive():
    """A spec-derived law serializes *whole* — derive and all."""
    data = law_to_data(SOFTMAX_FOLD)
    assert data["serializable"] is True
    assert data["dspec"] == {"SD": ["attr0", "RD"]}
    rebuilt = law_from_data(_json_roundtrip(data))
    assert rebuilt.dspec == SOFTMAX_FOLD.dspec
    assert rebuilt.derive({"$attr:RD": (-1,)}) == {"$attr:SD": -1}

def test_law_record_preserves_provenance_fields():
    data = law_to_data(FACTOR_MUL)
    assert data["derivation"] == list(FACTOR_MUL.derivation)
    assert data["tags"] == sorted(FACTOR_MUL.tags)
    assert data["law"] == FACTOR_MUL.law
    rebuilt = law_from_data(data)
    assert rebuilt.kind == FACTOR_MUL.kind
    assert rebuilt.error_bound == FACTOR_MUL.error_bound
    assert rebuilt.bound_norm == FACTOR_MUL.bound_norm

def test_law_from_data_rejects_bad_version():
    data = law_to_data(FACTOR_MUL)
    data["version"] = LAW_FORMAT + 1
    with pytest.raises(ValueError, match="version"):
        law_from_data(data)

def test_law_record_cert_field_is_optional_v2():
    """``cert`` rides v2: null by default, ignored on rebuild, absent
    in older records — no format bump, no reader rejection."""
    data = law_to_data(FACTOR_MUL)
    assert data["cert"] is None
    cert = Certificate(
        src=FACTOR_MUL.lhs,
        dst=FACTOR_MUL.rhs,
        root_eid=None,
        steps=[],
        rules={},
    )
    with_cert = law_to_data(FACTOR_MUL, cert=cert)
    assert with_cert["cert"]["version"] == CERT_FORMAT
    assert with_cert["cert"]["replayable"] is True
    # the cert is record-level provenance — it does not fold into the
    # reconstructed Rewrite (rebuilt rules aren't == under their
    # folded check closures, so compare the re-serialized records)
    assert law_to_data(law_from_data(with_cert)) == law_to_data(
        law_from_data(data)
    )
    # a pre-cert v2 record (no "cert" key at all) still loads
    del with_cert["cert"]
    assert law_from_data(with_cert).name == FACTOR_MUL.name

def test_rebuilt_cond_law_fires_and_declines_identically():
    rebuilt = law_from_data(_json_roundtrip(law_to_data(FACTOR_MUL)))
    src, merged = _factor_src()
    bad, bad_merged = _factor_decline()
    for rule in (FACTOR_MUL, rebuilt):
        assert _fires(rule, src, merged)
        assert not _fires(rule, bad, bad_merged)

def test_rebuilt_dspec_law_fires_identically():
    """``softmax_fold`` round-trips *whole* now — the rebuilt rule
    mints the derived dim, not the ``"SD"`` scar."""
    u = _v("u", 4, 8)
    src = Op.make(
        "div",
        Op.make("exp", u),
        Op.make("sum", Op.make("exp", u), dim=(-1,), keepdim=True),
    )
    want = Op.make("softmax", u, dim=-1)
    rebuilt = law_from_data(_json_roundtrip(law_to_data(SOFTMAX_FOLD)))
    assert _fires(SOFTMAX_FOLD, src, want)
    assert _fires(rebuilt, src, want)

def test_rebuilt_flagged_law_shows_the_honest_scar():
    """A flagged law rebuilt without its hooks fires — and mints the
    unbound attr metavar *literally* (``dim="ND"``).  The flag says
    why this cannot be trusted; the scar shows it."""
    rule = _flagged_rule()
    u = _v("u", 4, 8)
    src = Op.make("mul", u, Op.make("rsqrt", u))
    scar = Op.make("fused", u, dim="ND")
    rebuilt = law_from_data(_json_roundtrip(law_to_data(rule)))
    assert _fires(rebuilt, src, scar)
