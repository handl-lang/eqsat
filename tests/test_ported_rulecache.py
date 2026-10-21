"""Substrate-pure tests ported from ``graphopt:tests/test_rulecache.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core import meta

from mlopt_core.egraph import EGraph, Rewrite

from mlopt_core.ir import Const, Op, TensorType, Var, op_repr

from mlopt_core.rulecache import (
    RuleCache,
    cache_key,
    synthesize_rules_cached,
)


def _chain_bank() -> list[Rewrite]:
    """``f→g``, ``g→h``, ``h→k`` — composable pairs in one direction."""
    return [
        Rewrite(
            name=f"{a}_to_{b}",
            lhs=Op.make(a, "x"),
            rhs=Op.make(b, "x"),
        )
        for a, b in (("f", "g"), ("g", "h"), ("h", "k"))
    ]

def _guarded_chain_bank() -> list[Rewrite]:
    """The same chain, every rule carrying a trivially-true guard.

    Guarded composites cannot subsume their re-mints, so the strata
    differ from the unguarded bank's — and the composites carry the
    ``guard_pats`` spec the store/load path exercises.
    """
    return [
        Rewrite(
            name=f"{a}_to_{b}",
            lhs=Op.make(a, "x"),
            rhs=Op.make(b, "x"),
            check=lambda bound: True,
        )
        for a, b in (("f", "g"), ("g", "h"), ("h", "k"))
    ]

def test_storing_unguardable_hooks_raises(tmp_path):
    """A hand-built rule with hooks but no guard_pats spec cannot be
    serialized soundly — store must refuse rather than drop the guard."""
    rule = Rewrite(
        "hand_guard",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
        check=lambda bound: True,
    )
    cache = RuleCache(tmp_path)
    with pytest.raises(TypeError):
        cache.store("k", [rule])

def test_cached_fixpoint_roundtrips_the_strata(tmp_path, monkeypatch):
    """A ``rounds=3`` call mints the depth-3 composite and the stored
    entry reloads the same strata — ``rounds`` keys the cache."""
    bank = _chain_bank()
    fresh = meta.synthesize_rules(bank, fuel=200, rounds=3)
    assert len(fresh) == 3  # f→h, g→k, then syn-parented f→k

    d1 = synthesize_rules_cached(
        bank, fuel=200, rounds=3, cache_dir=tmp_path
    )
    assert [r.name for r in d1] == [r.name for r in fresh]

    # warm reload — same strata, and the deep composite still rewrites
    def _boom(*a, **kw):  # any re-derivation is a failure here
        raise AssertionError("synthesize_rules ran on a warm call")

    monkeypatch.setattr(meta, "synthesize_rules", _boom)
    d2 = synthesize_rules_cached(
        bank, fuel=200, rounds=3, cache_dir=tmp_path
    )
    assert [r.name for r in d2] == [r.name for r in fresh]
    deep = next(
        r
        for r in d2
        if any(p.startswith("syn_") for p in (r.parents or ()))
    )
    leaf = Var("v", TensorType((2, 2)))
    assert meta.apply_rewrite_at(
        deep, Op.make("f", leaf), ()
    ) == Op.make("k", leaf)

def test_rounds_and_cap_key_distinct_entries(tmp_path):
    """``rounds=1`` and ``rounds=3`` — and an ``max_emitted`` cap — are
    distinct problem instances, never a shared entry."""
    bank = _chain_bank()
    d1 = synthesize_rules_cached(
        bank, fuel=200, rounds=1, cache_dir=tmp_path
    )
    d3 = synthesize_rules_cached(
        bank, fuel=200, rounds=3, cache_dir=tmp_path
    )
    capped = synthesize_rules_cached(
        bank, fuel=200, rounds=3, max_emitted=2, cache_dir=tmp_path
    )
    assert len(d1) == 2 and len(d3) == 3 and len(capped) == 2
    assert cache_key(bank, fuel=200, rounds=1) != cache_key(
        bank, fuel=200, rounds=3
    )
    assert cache_key(
        bank, fuel=200, rounds=3, max_emitted=2
    ) != cache_key(bank, fuel=200, rounds=3)
    # one JSON entry per distinct problem instance
    assert len(list(tmp_path.glob("rules-*.json"))) == 3
    # and the rounds=1 mint is the honest single pass, not a truncated
    # read of the rounds=3 entry
    assert {r.name for r in d1}.isdisjoint(
        r.name
        for r in d3
        if any(p.startswith("syn_") for p in (r.parents or ()))
    )

def test_on_round_observes_cold_mints_only(tmp_path):
    """The observer keys through its hook signature: a vetoing callable
    is a different problem instance, and a warm hit calls nothing —
    the observer reports live rounds, not the replay."""
    bank = _chain_bank()
    seen: list[int] = []

    def watch(_i: int, em: list[Rewrite]) -> None:
        seen.append(len(em))

    d1 = synthesize_rules_cached(
        bank, fuel=200, rounds=3, on_round=watch, cache_dir=tmp_path
    )
    assert seen == [2, 1, 2]
    seen.clear()
    d2 = synthesize_rules_cached(
        bank, fuel=200, rounds=3, on_round=watch, cache_dir=tmp_path
    )
    assert [r.name for r in d2] == [r.name for r in d1]
    assert seen == []  # hit — no live rounds to observe

    # a vetoing observer truncates the fixpoint and gets its own entry
    vetoed = synthesize_rules_cached(
        bank,
        fuel=200,
        rounds=3,
        on_round=lambda i, em: False,
        cache_dir=tmp_path,
    )
    assert len(vetoed) == 2  # round 1 only
    assert len(list(tmp_path.glob("rules-*.json"))) == 2

def test_guarded_syn_parent_resolves_payload_internally(
    tmp_path, monkeypatch
):
    """A guarded round-2 composite's ``syn_*`` parent is not in the
    caller's bank — it is an earlier record of the same payload, and
    the load rebuilds the composite guard from it (the fixpoint output
    round-trips self-contained, not all-or-nothing against the base
    ruleset)."""
    bank = _guarded_chain_bank()
    d1 = synthesize_rules_cached(
        bank,
        fuel=200,
        rounds=3,
        numeric_check=False,
        cache_dir=tmp_path,
    )
    deep = [
        r
        for r in d1
        if any(p.startswith("syn_") for p in (r.parents or ()))
    ]
    assert len(deep) == 1 and deep[0].check is not None

    def _boom(*a, **kw):
        raise AssertionError("synthesize_rules ran on a warm call")

    monkeypatch.setattr(meta, "synthesize_rules", _boom)
    d2 = synthesize_rules_cached(
        bank,
        fuel=200,
        rounds=3,
        numeric_check=False,
        cache_dir=tmp_path,
    )
    assert [r.name for r in d2] == [r.name for r in d1]
    deep2 = next(
        r
        for r in d2
        if any(p.startswith("syn_") for p in (r.parents or ()))
    )
    # the transported guard rebuilt and the composite still rewrites
    assert deep2.check is not None and deep2.guard_pats is not None
    leaf = Var("v", TensorType((2, 2)))
    assert meta.apply_rewrite_at(
        deep2, Op.make("f", leaf), ()
    ) == Op.make("k", leaf)
