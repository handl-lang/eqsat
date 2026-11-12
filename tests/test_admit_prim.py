"""Auto-admission — ``admit_prim_record`` and the ``AUTO_TIER``.

The gauntlet stays the discovery side's gate; what these tests pin
is the *shipped object's* contract: a record that reaches
``admit_prim_record`` is asserted to carry its admission already,
and the function enforces the bank's own invariants loudly — decode,
naming, the bare-call RHS, the premise anchor, no silent shadowing.

Admissions in these tests target a *scratch* store (a tmp copy of
the shipped db, or a fresh file) — the shipped artifact is never
mutated by the suite.  The one exception is
``test_shipped_path_registers_live``, which monkeypatches the store
path at a tmp copy and restores every mutated global afterwards.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op, Var, op_repr, term_to_data
from mlopt_core.laws import minted, provenance, store
from mlopt_core.laws.serialize import law_to_data, missing_hooks


def _T() -> Any:
    from mlopt_core.ir import TensorType

    return TensorType((2, 4))


def _prim_record(name: str, *, expand: bool = False) -> dict[str, Any]:
    """A minimal well-formed prim record — the ``prims.yaml`` shape.

    ``pattern -> <name>(args)`` fold plus the mint's measurements;
    *expand* embeds the definitional premise verbatim (the depth-N
    record shape).
    """
    lhs = Op.make("add", Op.make("mul", "x", "y"), "z")
    rhs = Op.make(name, "x", "y", "z")
    fold = Rewrite(f"prim:{name}", lhs, rhs)
    rec: dict[str, Any] = {
        "alpha": "(add (mul 'x' 'y') 'z')",
        "mods": ["m_a", "m_b"],
        "sites": 4,
        "equal_sites": 4,
        "handler": {"verdict": "refused"},
        "fold": law_to_data(fold),
    }
    if expand:
        rec["expand"] = law_to_data(Rewrite(f"{name}.expand", rhs, lhs))
    return rec


@pytest.fixture()
def scratch_db(tmp_path: Path) -> Path:
    """A writable copy of the shipped store — never the artifact."""
    db = tmp_path / "minted.db"
    shutil.copyfile(store.DB_PATH, db)
    return db


# ---------------------------------------------------------------------------
#  the admit path — scratch store
# ---------------------------------------------------------------------------


def test_admit_materializes_fold_and_expand(scratch_db: Path) -> None:
    name = "ksynop_admit_a"
    rec = _prim_record(name)
    fold, expand = minted.admit_prim_record(name, rec, db=scratch_db)
    assert fold.name == f"prim:{name}"
    assert expand.name == f"{name}.expand"
    assert fold.premises == (expand,)
    assert expand.lhs == fold.rhs
    # depth-1 convention: the premise's RHS IS the fold LHS
    assert expand.rhs == fold.lhs
    assert missing_hooks(fold) == ()
    # the shipped integration fields — same as the hand-promoted tiers
    assert "machine-minted primitive fold" in fold.law
    assert fold.tags == frozenset({minted.tags.FUSION})
    # the op declaration registered with the fold
    from mlopt_core.ir import generator

    g = generator(name)
    assert g is not None and g.n_in == 3 and g.n_out == 1
    # the record persisted at tier AUTO_TIER, verbatim
    rows = store.load(scratch_db)[store.AUTO_TIER]
    assert rows[name] == rec
    # the shipped tables do NOT claim a scratch admission
    assert name not in minted.auto_records()
    with pytest.raises(KeyError):
        minted.prim_fold(name)
    # …and provenance honestly still says human — it is unshipped
    assert provenance.law_provenance(f"prim:{name}") == "human"


def test_admit_embedded_expand_premise(scratch_db: Path) -> None:
    """A depth-N-shaped record: the embedded premise anchors."""
    name = "ksynop2_admit_b"
    rec = _prim_record(name, expand=True)
    fold, expand = minted.admit_prim_record(name, rec, db=scratch_db)
    # the embedded premise is used verbatim — its RHS is the
    # *expansion* (here equal to the LHS by construction of the
    # fixture, but the record — not a synthesized one — is what
    # materialized).
    assert expand.name == f"{name}.expand"
    assert expand.lhs == fold.rhs
    assert expand.rhs == fold.lhs


def test_admit_is_idempotent(scratch_db: Path) -> None:
    name = "ksynop_admit_c"
    rec = _prim_record(name)
    f1, _ = minted.admit_prim_record(name, rec, db=scratch_db)
    f2, _ = minted.admit_prim_record(name, rec, db=scratch_db)
    assert f1 is f2 or (f1.lhs == f2.lhs and f1.rhs == f2.rhs)
    # still exactly one row for this name — the scratch copy may
    # already carry real AUTO_TIER admissions, so count the delta
    rows = store.load(scratch_db)[store.AUTO_TIER]
    assert rows[name] == rec
    assert len(rows) == len(minted.auto_records()) + 1


def test_admit_refuses_name_shadowing(scratch_db: Path) -> None:
    name = "ksynop_admit_d"
    minted.admit_prim_record(name, _prim_record(name), db=scratch_db)
    # a DIFFERENT record under the same name must raise — the
    # idempotent path only accepts the identical payload
    other = _prim_record(name)
    other["sites"] = 99
    with pytest.raises(ValueError, match="different record"):
        minted.admit_prim_record(name, other, db=scratch_db)


def test_admit_refuses_shipped_name(scratch_db: Path) -> None:
    """A name the hand-promoted tiers already ship cannot be re-admitted."""
    name = next(iter(minted.SYNOP_SHIPPED))
    with pytest.raises(ValueError, match="already shipped"):
        minted.admit_prim_record(
            name, _prim_record(name), db=scratch_db
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["fold"].update(name="prim:other"), "prim:other"),
        (
            lambda r: r["fold"].update(
                rhs=term_to_data(Op.make("add", "x", "y"))
            ),
            "RHS",
        ),
        (
            lambda r: r.update(
                expand=law_to_data(
                    Rewrite(
                        "x.expand",
                        Op.make("sub", "x", "y"),
                        Op.make("add", "x", "y"),
                    )
                )
            ),
            "expand",
        ),
        (lambda r: r.pop("fold"), "fold"),
    ],
)
def test_admit_validates_record_shape(
    scratch_db: Path, mutate: Any, match: str
) -> None:
    name = "ksynop_admit_e"
    rec = _prim_record(name, expand=True)
    mutate(rec)
    with pytest.raises(ValueError, match=match):
        minted.admit_prim_record(name, rec, db=scratch_db)


def test_admit_rejects_non_dict_record(scratch_db: Path) -> None:
    with pytest.raises(ValueError, match="fold"):
        minted.admit_prim_record(
            "ksynop_x", {"no_fold": 1}, db=scratch_db
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        minted.admit_prim_record("", _prim_record("k"), db=scratch_db)


# ---------------------------------------------------------------------------
#  the shipped path — monkeypatched store, every global restored
# ---------------------------------------------------------------------------


def test_shipped_path_registers_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``db`` resolving to the store path mutates the live tables.

    The shipped artifact is never touched: the store path is
    monkeypatched at a tmp copy, ``_AUTO_RECORDS`` /
    ``_PRIM_MATERIALIZED`` are restored in ``finally``.
    """
    db = tmp_path / "minted.db"
    shutil.copyfile(store.DB_PATH, db)
    monkeypatch.setattr(store, "DB_PATH", db)
    name = "synop_admit_live"
    rec = _prim_record(name)
    try:
        fold, _expand = minted.admit_prim_record(name, rec, db=db)
        assert fold.name == f"prim:{name}"
        # the live tables see it now — prim_fold resolves, the
        # accessors include it, provenance reports machine
        assert name in minted.auto_records()
        assert name in minted.synop_shipped()
        assert minted.prim_fold(name)[0].name == f"prim:{name}"
        assert any(
            f.name == f"prim:{name}" for f in minted.prim_folds()
        )
        assert any(
            p.name == f"{name}.expand" for p in minted.prim_premises()
        )
        assert provenance.law_provenance(f"prim:{name}") == "machine"
        assert provenance.law_provenance(f"{name}.expand") == "machine"
        assert provenance.pay_status(f"prim:{name}") == "unmeasured"
        # idempotent against the now-live record
        fold2, _ = minted.admit_prim_record(name, rec, db=db)
        assert fold2.name == fold.name
    finally:
        minted._AUTO_RECORDS.pop(name, None)
        minted._PRIM_MATERIALIZED.pop(name, None)


def test_shipped_tier_consistency() -> None:
    """Whatever the shipped AUTO_TIER carries satisfies the contract.

    Runs green both before the first admission (empty tier) and
    after — the census follows the data, not a hardcoded count.
    """
    auto = minted.auto_records()
    for name, rec in auto.items():
        fold_data = rec["fold"]
        assert fold_data["name"] == f"prim:{name}"
        assert name in minted.SYNOP_SHIPPED
        assert f"prim:{name}" in provenance.MACHINE_SHIPPED
        assert f"{name}.expand" in provenance.MACHINE_SHIPPED
        assert f"prim:{name}" in provenance.AUTO_ADMITTED
        assert provenance.law_provenance(f"prim:{name}") == "machine"
        # and it materializes through the same path as tiers 1-3
        fold, _expand = minted.prim_fold(name)
        assert fold.rhs.op == name
        assert missing_hooks(fold) == ()


def test_extract_pays_on_spelled_site() -> None:
    """A scratch-admitted fold fires and wins extraction, torch-free.

    Mirrors ``test_prim_fold_fires_and_extracts_on_spelled_site``
    for the auto-admission class: instantiate the LHS, saturate with
    the fold alone, and the ``<name>(args)`` call must be the
    extracted best.
    """
    from mlopt_core.cost import count_cost, dag_cost
    from mlopt_core.egraph import EGraph
    from mlopt_core.egraph.terms import _term_match
    from mlopt_core.meta import instantiate_pattern, pattern_metavars

    name = "ksynop_admit_extract"
    rec = _prim_record(name)
    fold, _ = minted.admit_prim_record(name, rec, db=":memory:")
    subst = {mv: Var(mv, _T()) for mv in pattern_metavars(fold.lhs)}
    site = instantiate_pattern(fold.lhs, subst)
    assert _term_match(fold.lhs, site) is not None
    eg = EGraph()
    root = eg.add_term(site)
    eg.run([fold], root, max_iterations=4, max_nodes=20_000)
    ops = {n.op for ec in eg._classes.values() for n in ec.nodes}
    assert name in ops
    best = eg.extract_best(root, count_cost)
    assert best is not None
    assert op_repr(best).count(name) >= 1
    assert dag_cost(best, count_cost) <= dag_cost(site, count_cost)
