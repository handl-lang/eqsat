"""Substrate-pure tests ported from ``graphopt:tests/test_sdpa_expand.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import pytest

from mlopt_core.cost import count_cost, flops_cost

from mlopt_core.egraph import EGraph, verify_certificate

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var

from mlopt_core.laws.ruleset import ATTENTION, DEFAULT

from mlopt_core.laws.tensor import (
    SDPA_EXPAND_RULES,
    SDPA_FOLD_MUL_NOMASK,
)


_EXPAND_BY_NAME = {r.name: r for r in SDPA_EXPAND_RULES}

def _v(name: str, *shape: int | None) -> Var:
    return Var(name, TensorType(tuple(shape)))

def _saturate(term, rules, iters=8, max_nodes=100_000):
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(rules, eid, max_iterations=iters, max_nodes=max_nodes)
    return eg, eid

def _fired(eg, name):
    return eg.rule_fires.get(name, 0) > 0

@pytest.mark.parametrize(
    "attrs",
    [
        {"is_causal": True},  # causal — needs a minted mask; declined
        {"dropout_p": 0.1},  # train dropout — different math
        {"dropout_p": 0.0, "is_causal": True},
        {"enable_gqa": True},  # kv-repeat semantics — not the same op
        {
            "dropout_p": 0.0,
            "is_causal": False,
            "enable_gqa": True,
        },
    ],
)
def test_sdpa_expand_declines_nontrivial_attrs(attrs):
    """No benign pattern matches a causal/dropout/GQA spelling."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v, **attrs)
    eg, _ = _saturate(lhs, SDPA_EXPAND_RULES)
    assert not any(_fired(eg, r.name) for r in SDPA_EXPAND_RULES)

def test_sdpa_expand_declines_masked_arity():
    """The 4-operand masked form does not expand: bool keep-mask vs
    additive float mask is a dtype fact the IR cannot read."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    m = _v("m", B, H, T, T)
    lhs = Op.make("sdpa", q, k, v, m)
    eg, _ = _saturate(lhs, SDPA_EXPAND_RULES)
    assert not any(_fired(eg, r.name) for r in SDPA_EXPAND_RULES)

def test_sdpa_expand_declines_unshaped_q():
    """The implicit-scale mint vetoes honestly when q's head dim is
    unknown — a law never mints an attr it cannot verify."""
    B, H, T, d = 2, 4, 8, 16
    q = _v("q", B, H, T, None)
    k, v = _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v)
    eg, _ = _saturate(lhs, [_EXPAND_BY_NAME["sdpa_expand"]])
    assert not _fired(eg, "sdpa_expand")

def test_sdpa_fold_mul_nomask_declines_nonconst_scale():
    """A non-Const scale operand declines — ``scale`` needs a float."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    s = _v("s", B, H, T, T)  # a full score-shaped scale — not a Const
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)
    )
    lhs = Op.make(
        "matmul",
        Op.make("softmax", Op.make("mul", scores, s), dim=-1),
        v,
    )
    eg, _ = _saturate(lhs, [SDPA_FOLD_MUL_NOMASK])
    assert not _fired(eg, "sdpa_fold_mul_nomask")

def test_expand_fold_round_trip_scaled():
    """``sdpa(scale=s)`` expands and the fold's redex exists — the
    refold union lands on the already-present fused member (a no-op
    merge, so it does not count as a fire; the matcher is the proof)."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v, scale=0.5)
    rules = list(DEFAULT.rules) + list(SDPA_EXPAND_RULES)
    eg, eid = _saturate(lhs, rules)
    assert _fired(eg, "sdpa_expand_s")
    # the expanded member IS the fold's LHS redex — the pair rejoins
    assert list(eg.matches(SDPA_FOLD_MUL_NOMASK.lhs, eid))
    best = eg.extract_best(eid, flops_cost)
    assert isinstance(best, Op) and best.op == "sdpa"

def test_certificate_replays_through_expansion():
    """The derivation stands alone: the expand + refold steps replay
    under ``verify_certificate`` without the e-graph."""
    B, H, T, d = 2, 4, 8, 16
    q, k, v = _v("q", B, H, T, d), _v("k", B, H, T, d), _v("v", B, H, T, d)
    lhs = Op.make("sdpa", q, k, v)
    rules = list(DEFAULT.rules) + list(SDPA_EXPAND_RULES)
    eg, eid = _saturate(lhs, rules)
    assert _fired(eg, "sdpa_expand")
    # certify the expanded member — its derivation records the
    # minted Const leaf (the ``$leaf:`` channel replays deterministically)
    expanded = next(
        n
        for n in eg.get_class(eid).nodes
        if n.op == "matmul"
    )
    term = eg.extract_best(
        eid, count_cost, overrides={eg.find(eid): expanded}
    )
    cert = eg.certificate(lhs, term)
    out = verify_certificate(lhs, cert)
    assert out is not None
    assert cert.stats.get("n_steps", 0) >= 1

def test_sdpa_expand_never_in_default():
    """The decompose direction is opt-in — nothing leaks into DEFAULT."""
    assert not (set(SDPA_EXPAND_RULES) & set(DEFAULT.rules))
    assert "sdpa_fold_mul_nomask" in {r.name for r in DEFAULT.rules}
    # the fold direction IS default — every pipeline refolds the mint
    names = {r.name for r in DEFAULT.rules}
    assert not any(n.startswith("sdpa_expand") for n in names)

def test_expand_family_rule_names():
    """The generated family covers the benign key-sets, one rule each."""
    assert {r.name for r in SDPA_EXPAND_RULES} == {
        "sdpa_expand",
        "sdpa_expand_d",
        "sdpa_expand_dc",
        "sdpa_expand_dcg",
        "sdpa_expand_s",
        "sdpa_expand_ds",
        "sdpa_expand_dcs",
        "sdpa_expand_dcgs",
    }
