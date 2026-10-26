"""Pay status is part of the ledger, not a footnote.

`docs/reports/pay_measured.md` (graphopt) timed the shipped prim folds on
a delivered module: nine of them came back slower than the same pipeline
without them, and 48 were never measured at all.  Those two facts belong
next to the authorship flag, because "machine-shipped" says nothing about
speed.

These tests pin the classification so a later summary cannot quietly call
an unmeasured fold "paying".
"""

from __future__ import annotations

from mlopt_core.laws import minted, provenance


def test_status_is_total() -> None:
    """Every name answers — including names the ledger has never seen."""
    assert provenance.pay_status("id_add") == "n/a"
    assert provenance.pay_status("no_such_law") == "n/a"
    assert provenance.pay_status(
        minted.prim_fold("synop_bb8c54c201")[0]
    ) == ("measured-slow")


def test_shipped_folds_are_classified() -> None:
    """9 measured-slow, 48 unmeasured — none silently omitted."""
    statuses = [
        provenance.pay_status(op) for op in minted.SYNOP_SHIPPED
    ]
    assert statuses.count("measured-slow") == len(
        provenance.PAY_MEASURED_SLOW
    )
    assert statuses.count("unmeasured") == len(
        minted.SYNOP_SHIPPED
    ) - len(provenance.PAY_MEASURED_SLOW)
    assert "n/a" not in statuses


def test_expand_premises_are_unmeasured() -> None:
    """A definitional premise is not a pay claim either."""
    for op in minted.SYNOP_SHIPPED:
        assert provenance.pay_status(f"{op}.expand") == "unmeasured"


def test_both_spellings_agree() -> None:
    """`synop_*` and `prim:synop_*` classify identically."""
    op = next(iter(minted.SYNOP_SHIPPED))
    assert provenance.pay_status(op) == provenance.pay_status(
        f"prim:{op}"
    )
