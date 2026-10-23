"""Structure pins for the prior-art law bank.

The bank is the opt-in ``PRIOR_ART`` preset: known compiler
transforms (TASO / Inductor-XLA / corpus-family equivalences)
encoded as certified laws — tagged ``prior_art``, human-authored
(never in ``MACHINE_*``), never in ``DEFAULT``.  The fp64 sweep +
``verify_law`` cert run lives in the graphopt driver
(``bench/drivers/audit/prior_art_verify.py`` — the oracle/verifier
machinery is discovery-side, not substrate); the results document
is ``graphopt:docs/results_prior_art_bank.md``.
"""

import pytest

from mlopt_core import laws
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op
from mlopt_core.laws import ruleset, tags
from mlopt_core.laws.priorart import PRIOR_ART_RULES
from mlopt_core.laws.provenance import (
    MACHINE_SHIPPED,
    MACHINE_STORED,
    law_provenance,
)


def test_prior_art_count():
    # Scoped at ~50-150 by design; 189 encoded (where_same /
    # logical_not_inv dropped as honest dtype misses — see the
    # "Dropped" note in priorart.py; +2 for the instance_norm /
    # group_norm definitional expands promoted from graphopt's
    # fusion-schema premise closures; +2 for the review-queue
    # demotions — softsign_expand (the composite-free kernel
    # denotation) and mul_unsq_rev (XLA's broadcast-sink
    # direction of the shipped mul_unsq_nat)).
    assert len(PRIOR_ART_RULES) == 189


def test_prior_art_names_unique():
    names = [r.name for r in PRIOR_ART_RULES]
    assert len(names) == len(set(names))


def test_prior_art_all_tagged():
    for r in PRIOR_ART_RULES:
        assert tags.PRIOR_ART in r.tags, r.name


def test_prior_art_not_machine_admitted():
    """The bank is prior art — no name joins either machine set."""
    names = {r.name for r in PRIOR_ART_RULES}
    assert not (names & MACHINE_SHIPPED)
    assert not (names & MACHINE_STORED)
    for r in PRIOR_ART_RULES:
        assert law_provenance(r) == "human"


def test_prior_art_no_collision_with_shipped_bank():
    """RuleSet composition raises on same-name/different-rule —
    the bank must be disjoint from every shipped collection."""
    shipped = (
        {r.name for r in laws.ALL_RULES}
        | {r.name for r in laws.ALL_RULES_WITH_LAYOUT}
        | {r.name for r in laws.ATTENTION_RULES}
        | {r.name for r in laws.BATCH_NORM_RULES}
        | {r.name for r in laws.CONV_FOLD_RULES}
        | {r.name for r in laws.POINTWISE_FACTOR_RULES}
        | {r.name for r in laws.SCAN_LAWS}
        | {r.name for r in laws.SCAN_DIAG_LAWS}
        | {r.name for r in laws.SDPA_EXPAND_RULES}
    )
    overlap = shipped & {r.name for r in PRIOR_ART_RULES}
    assert not overlap, f"name collisions: {sorted(overlap)}"


def test_prior_art_out_of_default():
    """PRIOR_ART composes nowhere into DEFAULT / all_rules()."""
    default_names = {r.name for r in laws.all_rules()}
    pa_names = {r.name for r in PRIOR_ART_RULES}
    assert not (default_names & pa_names)
    for r in PRIOR_ART_RULES:
        assert not any(r is d for d in laws.DEFAULT)


def test_prior_art_preset_registered():
    pa = laws.PRIOR_ART
    assert pa.name == "prior_art"
    assert list(pa) == PRIOR_ART_RULES
    assert ruleset.preset("prior_art") is pa
    assert "prior_art" in ruleset.PRESETS


def test_prior_art_preset_composes():
    """The opt-in preset merges over the default set (the
    ``with_layout`` / ``carriers`` composition path)."""
    merged = ruleset.RuleSet("x", tuple(laws.DEFAULT) + tuple(laws.PRIOR_ART))
    assert len(list(merged)) == len(laws.DEFAULT) + len(laws.PRIOR_ART)


def test_prior_art_declarative_roundtrip():
    """Every declarative ``cond``/``dspec`` must have a registered
    spell (the serialize registry is the audit surface — the
    certs minted off these rules travel as data)."""
    from mlopt_core.laws.cond import cond_to_data, derive_to_data

    for r in PRIOR_ART_RULES:
        assert isinstance(r, Rewrite)
        if r.cond is not None:
            assert cond_to_data(r.cond) is not None, r.name
        if r.dspec is not None:
            assert derive_to_data(r.dspec) is not None, r.name


def test_prior_art_patterns_wellformed():
    """LHS/RHS are grounded patterns: every RHS metavar is bound by
    the LHS, and every op is registered."""
    from mlopt_core.meta import pattern_metavars
    from mlopt_core.opmeta import REGISTRY

    for r in PRIOR_ART_RULES:
        lh, rh = pattern_metavars(r.lhs), pattern_metavars(r.rhs)
        minted = set()
        items = (
            r.dspec.items() if hasattr(r.dspec or (), "items") else r.dspec or ()
        )
        for k, _v in items:
            if k.startswith("$leaf:"):
                minted.add(k[6:])
            elif k.startswith("$attr:"):
                minted.add(k)
            else:
                minted.add(f"$attr:{k}")
        unbound = rh - lh - minted
        if r.derive is not None:
            continue  # procedural derives mint attrs the static check can't see
        assert not unbound, f"{r.name}: RHS metavars {sorted(unbound)}"

        def ops(t):
            if isinstance(t, Op):
                yield t.op
                for a in t.args:
                    yield from ops(a)

        for t in (r.lhs, r.rhs):
            for op in ops(t):
                assert op in REGISTRY, (r.name, op)
