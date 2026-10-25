"""The minted-record store — the bank as data, pinned.

``mlopt_core.laws.minted`` used to embed four JSON literals (≈40 000
lines) and replay them by hand-pasted edits.  They now live in
``laws/data/minted.db`` — one row per record, one tier per mint depth —
and the module reads them at import.

These tests pin the two things that refactor could quietly break:

* **the payload** — every record's canonical SHA matches the snapshot
  taken before the literals moved (`tests/golden/minted_records.json`);
* **the objects** — every shipped fold and its definitional premise
  still materializes to the same ``law_to_data`` row.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from mlopt_core.laws import minted, store
from mlopt_core.laws.serialize import law_to_data

GOLDEN = Path(__file__).parent / "golden" / "minted_records.json"


def _sha(record: object) -> str:
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_store_ships(golden: dict) -> None:
    """The store exists, has all four tiers, and no tier lost a row."""
    tiers = store.load()
    assert set(tiers) == set(store.TIERS)
    for tier, table in tiers.items():
        assert len(table) == len(golden["payload_sha"][str(tier)])


def test_payload_matches_golden(golden: dict) -> None:
    """Every record round-trips byte-identically through the store."""
    for tier, table in store.load().items():
        got = {name: _sha(rec) for name, rec in table.items()}
        assert got == golden["payload_sha"][str(tier)], (
            f"tier {tier} drifted"
        )


def test_materialized_laws_match_golden(golden: dict) -> None:
    """Each shipped fold and premise materializes to its recorded row."""
    assert minted.SYNOP_SHIPPED, "no shipped prims"
    for synop in sorted(minted.SYNOP_SHIPPED):
        fold, expand = minted.prim_fold(synop)
        assert law_to_data(fold) == golden["laws"][synop]["fold"], synop
        assert law_to_data(expand) == golden["laws"][synop]["expand"], (
            synop
        )


def test_fold_and_premise_sets_agree() -> None:
    """Every fold has its premise, and the naming stays in sync."""
    folds = {r.name for r in minted.prim_folds()}
    premises = {r.name for r in minted.prim_premises()}
    assert len(folds) == len(minted.SYNOP_SHIPPED)
    assert premises == {
        minted.expand_name(s) for s in minted.SYNOP_SHIPPED
    }


def test_read_only_by_default() -> None:
    """The shipped store is an artifact — a read connection cannot write."""
    conn = store.connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM records")
    finally:
        conn.close()


def test_emit_is_idempotent(tmp_path: Path) -> None:
    """Re-emitting identical records writes nothing; a change writes one."""
    db = tmp_path / "minted.db"
    rows = store.load()
    assert store.emit(rows, db, replace=True) == sum(
        len(t) for t in rows.values()
    )
    assert store.emit(rows, db) == 0

    changed = {tier: dict(table) for tier, table in rows.items()}
    name = next(iter(changed[1]))
    changed[1][name] = {**changed[1][name], "note": "edited"}
    assert store.emit(changed, db) == 1
    assert store.load(db)[1][name]["note"] == "edited"


def test_query_surface() -> None:
    """The point of the store: the bank answers SQL."""
    counts = store.tiers()
    assert counts == {0: 4, 1: 21, 2: 2, 3: 34}
    assert sum(store.summary().values()) == sum(counts.values())
    row = store.query(
        "SELECT name, sha FROM records WHERE tier = 1 ORDER BY seq LIMIT 1"
    )[0]
    assert row["name"].startswith("synop_")
    # content-addressing: the sha column really is the payload hash
    payload = store.query(
        "SELECT payload FROM records WHERE tier = 1 AND name = ?",
        (row["name"],),
    )[0]["payload"]
    assert _sha(json.loads(payload)) == row["sha"]


def test_measurements_upsert(tmp_path: Path) -> None:
    """A measurement is a fact beside the record, replaced on re-measure."""
    db = tmp_path / "minted.db"
    store.emit(store.load(), db, replace=True)

    rows = [
        {
            "name": "synop_x",
            "metric": "pay_wall",
            "arm": "marginal",
            "value": 1.42,
            "unit": "ratio",
        },
        {
            "name": "synop_x",
            "metric": "pay_wall",
            "arm": "e2e",
            "value": 1.31,
            "unit": "ratio",
        },
    ]
    assert store.record_measurements(rows, db, run="test") == 2
    got = store.measurements("synop_x", db)
    assert {r["arm"]: r["value"] for r in got} == {
        "marginal": 1.42,
        "e2e": 1.31,
    }

    # re-measuring updates in place — no second row, same count
    again = [
        {
            "name": "synop_x",
            "metric": "pay_wall",
            "arm": "marginal",
            "value": 1.05,
            "unit": "ratio",
        }
    ]
    store.record_measurements(again, db, run="test")
    got = store.measurements("synop_x", db)
    assert len(got) == 2
    assert {r["arm"]: r["value"] for r in got}["marginal"] == 1.05
