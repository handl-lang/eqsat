"""Drift guard: ``data/eqsat_laws.json`` is generated, not hand-kept.

``handl-lang/kit-eqsat`` is the canonical law-corpus home
(ADR-0177 §1); the shipped snapshot must equal the kit's
``LAW_PACKS`` corpus byte-for-byte under the emitter's canonical
rendering.  Regenerate with::

    .venv/bin/python scripts/emit_eqsat_laws.py

The kit-eqsat checkout resolves exactly as the emitter does
(``$MLOPT_KIT_EQSAT`` → the ``handl-lang`` sibling →
``~/GitHub/handl-lang/kit-eqsat``); when it is absent — a standalone
clone or tarball — the test skips rather than failing, mirroring
kit-eqsat's own ``tests/corpus_drift.rs`` convention.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EMITTER_PATH = _REPO_ROOT / "scripts" / "emit_eqsat_laws.py"
_LAWS_PATH = (
    _REPO_ROOT
    / "packages"
    / "mlopt-orchestrator"
    / "src"
    / "mlopt_orchestrator"
    / "data"
    / "eqsat_laws.json"
)


def _emitter():
    """Load ``scripts/emit_eqsat_laws.py`` as a module (not a package)."""
    spec = importlib.util.spec_from_file_location(
        "emit_eqsat_laws", _EMITTER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_snapshot_matches_kit_corpus() -> None:
    emit = _emitter()
    kit = emit.find_kit_eqsat()
    if kit is None:
        pytest.skip(
            "kit-eqsat sibling checkout not found — set "
            "MLOPT_KIT_EQSAT or clone handl-lang/kit-eqsat beside "
            "this repo"
        )
    records = emit.corpus_records(kit)
    shipped = json.loads(_LAWS_PATH.read_text())
    assert shipped == records, (
        "eqsat_laws.json drifted from the kit-eqsat corpus — "
        "rerun scripts/emit_eqsat_laws.py"
    )
    # Byte-exact under the canonical rendering: the committed file is
    # the emitter's output, not a hand-formatted lookalike.
    assert _LAWS_PATH.read_text() == emit.render(records)
    # The corpus is the full non-demo set — the unsound matcher demo
    # stays engine-eqsat-only by design.
    assert len(records) == 332
    assert "reshape_flatten_demo" not in {r["name"] for r in records}


def test_minted_store_projects_into_shipped_snapshot() -> None:
    """Every minted prim fold the Python bank can fire is on the wire.

    ``minted.db`` is the ledger and kit-eqsat's ``prim_folds`` pack
    the projected corpus; a live ``admit_prim_record`` widens the
    Python bank at import while the snapshot only moves on
    re-projection — the two-speed drift the demotion plan calls out.
    This pins ``AUTO_TIER ⊆ projected records`` *content-exact*: each
    store record's kit projection must equal the shipped snapshot
    record, so an unprojected admission (or a store edit) fails here
    instead of silently diverging the engines.  Needs no kit-eqsat
    checkout — the committed snapshot is the reference.
    """
    from mlopt_core.laws import minted

    shipped = {r["name"]: r for r in json.loads(_LAWS_PATH.read_text())}
    records = minted.prim_kit_records()
    assert records, "minted store projected no prim records"
    missing = sorted(set(records) - set(shipped))
    assert not missing, (
        f"minted prim records absent from eqsat_laws.json — "
        f"re-project into kit-eqsat's prim_folds pack and rerun "
        f"scripts/emit_eqsat_laws.py: {missing}"
    )
    drifted = sorted(
        name for name, rec in records.items() if shipped[name] != rec
    )
    assert not drifted, (
        f"minted prim records drifted from the snapshot — "
        f"re-project into kit-eqsat's prim_folds pack and rerun "
        f"scripts/emit_eqsat_laws.py: {drifted}"
    )
