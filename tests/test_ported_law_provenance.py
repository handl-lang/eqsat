"""Substrate-pure tests ported from ``graphopt:tests/test_law_provenance.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.laws import (
    ALL_RULES,
    ALL_RULES_WITH_LAYOUT,
    ATTENTION_RULES,
    MACHINE_LAWS,
    MACHINE_SHIPPED,
    MACHINE_STORED,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    RuleSet,
    law_provenance,
)

from mlopt_orchestrator.optimize import Optimizer, default_rules


_PROMOTION_TABLE = frozenset(
    {
        "softsign_fold",
        "sdpa_fold_nomask",
        "sdpa_fold_div_nomask",
        "mul_unsq_nat",
        "mul_unsq_pad_l",
        "mul_unsq_pad_r",
        "sub_unsq_pad_l",
        "add_unsq_pad_r",
        "mul_reshape_inert_l",
        "transpose_noop",
        "chunk_single",
        "linear_channel_to_row_scale",
        "conv_bn_fold",
    }
)

def test_every_law_classifies_by_provenance():
    """The marker is total: every shipped rule resolves to a class."""
    surface = (
        list(ALL_RULES_WITH_LAYOUT)
        + list(ATTENTION_RULES)
        + list(SCAN_LAWS)
        + list(SCAN_DIAG_LAWS)
        + list(default_rules())
    )
    assert surface, "expected a nonempty law surface"
    for rule in surface:
        assert law_provenance(rule) in ("human", "machine")
        assert law_provenance(rule.name) == law_provenance(rule)

def test_machine_shipped_is_exactly_the_promotion_table():
    """``MACHINE_SHIPPED`` pins the thirteen promoted laws, all shipped."""
    assert MACHINE_SHIPPED == _PROMOTION_TABLE
    shipped_names = {r.name for r in ALL_RULES}
    assert shipped_names >= MACHINE_SHIPPED
    # 73 human-authored + 13 promoted = the 86-rule library
    # (``add_unsq_pad_r`` — the auto-cond candidate admitted off the
    # real-corpus funnel, docs/results_realcorpus.md).
    assert len(ALL_RULES) == 86
    assert len(shipped_names - MACHINE_SHIPPED) == 73

def test_machine_stored_objects_are_unshipped_names():
    """The store ledger names objects that are NOT in the library —
    a promoted object vacates the store set (its name is the shipped
    law's), so the sets stay disjoint by construction."""
    shipped_names = {r.name for r in ALL_RULES}
    assert MACHINE_STORED.isdisjoint(shipped_names)
    assert MACHINE_STORED <= MACHINE_LAWS
    assert MACHINE_LAWS == MACHINE_SHIPPED | MACHINE_STORED

def test_provenance_defaults_to_human_authorship():
    """Unknown names are the library's default: human-authored."""
    assert law_provenance("no_such_law") == "human"
