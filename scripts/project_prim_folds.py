#!/usr/bin/env python3
"""Project ``minted.db`` prim records into kit-eqsat's ``prim_folds`` pack.

``mlopt_core.laws.minted`` is the ledger (ADR-0177 §1); the
``prim:synop*`` folds the Python bank fires project to
``LAW_FORMAT = 2`` records through
:func:`mlopt_core.laws.minted.prim_kit_records` and land in
kit-eqsat as ``data/laws/prim_folds.json`` — the pack
``mlopt_orchestrator``'s shipped ``eqsat_laws.json`` snapshot then
regenerates via ``scripts/emit_eqsat_laws.py``.

This script is the eqsat-side half of that re-projection: it emits
the pack's record array (``prim_records()`` order — tiers 1-3 then
the live auto-admission tier, which is the order the pack ships) so
a kit-eqsat checkout can drop it in verbatim, and ``--check`` drift-
guards the sibling checkout's ``prim_folds.json`` record-for-record.
Record equality is the contract — the pack's on-disk byte spelling
is kit-eqsat's own rendering concern.

Usage::

    .venv/bin/python scripts/project_prim_folds.py            # emit JSON
    .venv/bin/python scripts/project_prim_folds.py --check    # drift check

Kit-eqsat resolution is shared with ``emit_eqsat_laws.py``
(``$MLOPT_KIT_EQSAT`` → the ``handl-lang`` sibling →
``~/GitHub/handl-lang/kit-eqsat``); ``--check`` exits 2 when no
checkout resolves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import emit_eqsat_laws  # sibling script, not a package

PRIM_FOLDS_PATH = (
    Path("crates")
    / "handl-kit-eqsat"
    / "data"
    / "laws"
    / "prim_folds.json"
)


def projected_records() -> list[dict]:
    """Return the ``prim_folds`` pack content, in shipped order."""
    from mlopt_core.laws import minted

    return list(minted.prim_kit_records().values())


def main(argv: list[str] | None = None) -> int:
    """Run the projector; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the projection against the kit-eqsat sibling's "
            "prim_folds.json instead of emitting it"
        ),
    )
    args = parser.parse_args(argv)

    records = projected_records()
    if not args.check:
        print(json.dumps(records, indent=1, sort_keys=True))
        return 0

    kit = emit_eqsat_laws.find_kit_eqsat()
    if kit is None:
        print(
            "kit-eqsat checkout not found — set MLOPT_KIT_EQSAT or "
            "clone handl-lang/kit-eqsat beside this repo",
            file=sys.stderr,
        )
        return 2
    path = kit / PRIM_FOLDS_PATH
    try:
        shipped = json.loads(path.read_text())
    except OSError as e:
        print(
            f"kit-eqsat prim_folds pack unreadable: {path} ({e})",
            file=sys.stderr,
        )
        return 2
    if shipped != records:
        print(
            f"drift: {path} != minted.db projection — re-run "
            "scripts/project_prim_folds.py and update the pack "
            "(then scripts/emit_eqsat_laws.py for the snapshot)",
            file=sys.stderr,
        )
        return 1
    print(f"prim_folds.json in sync with minted.db ({kit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
