#!/usr/bin/env python3
"""Regenerate ``mlopt_orchestrator``'s shipped eqsat law snapshot.

``packages/mlopt-orchestrator/src/mlopt_orchestrator/data/eqsat_laws.json``
is a *generated artifact*: ``handl-lang/kit-eqsat`` is the canonical
law-corpus home (ADR-0177 §1 — laws are kit data, ``LAW_FORMAT = 2``
JSON records), and this script concatenates its
``crates/handl-kit-eqsat/data/laws/<pack>.json`` packs in
``LAW_PACKS`` order into the flat record array the Python bridge
ships.  ``tests/test_eqsat_laws_corpus.py`` drift-guards the result —
hand-editing the JSON is a test failure, not a workflow.

Usage::

    .venv/bin/python scripts/emit_eqsat_laws.py           # rewrite
    .venv/bin/python scripts/emit_eqsat_laws.py --check   # drift check

Kit-eqsat resolution: ``$MLOPT_KIT_EQSAT``, then ``../kit-eqsat``
relative to this repo (the ``handl-lang`` sibling layout), then
``~/GitHub/handl-lang/kit-eqsat``.  The pack order is read out of
``crates/handl-kit-eqsat/src/lib.rs``'s ``LAW_PACKS`` table, so a pack
added upstream lands here without touching this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAWS_PATH = (
    REPO_ROOT
    / "packages"
    / "mlopt-orchestrator"
    / "src"
    / "mlopt_orchestrator"
    / "data"
    / "eqsat_laws.json"
)
KIT_LIB_RS = Path("crates") / "handl-kit-eqsat" / "src" / "lib.rs"


def find_kit_eqsat() -> Path | None:
    """Locate the kit-eqsat checkout, or ``None`` when absent."""
    env = os.environ.get("MLOPT_KIT_EQSAT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT.parent / "kit-eqsat")
    candidates.append(
        Path.home() / "GitHub" / "handl-lang" / "kit-eqsat"
    )
    for cand in candidates:
        if (cand / KIT_LIB_RS).is_file():
            return cand
    return None


def law_pack_names(kit: Path) -> list[str]:
    """Pack order from ``LAW_PACKS`` in kit-eqsat's ``lib.rs``."""
    src = (kit / KIT_LIB_RS).read_text()
    m = re.search(
        r"pub const LAW_PACKS[^=]*=\s*&\[(.*?)\];", src, re.DOTALL
    )
    if m is None:
        raise SystemExit(
            f"{kit / KIT_LIB_RS}: no LAW_PACKS table found — "
            "kit-eqsat layout changed; update this script"
        )
    names = re.findall(r'name:\s*"([^"]+)"', m.group(1))
    if not names:
        raise SystemExit(
            f"{kit / KIT_LIB_RS}: LAW_PACKS holds no packs"
        )
    return names


def corpus_records(kit: Path) -> list[dict]:
    """Return the flat corpus: every ``LAW_PACKS`` pack's records in order."""
    records: list[dict] = []
    for name in law_pack_names(kit):
        path = (
            kit
            / "crates"
            / "handl-kit-eqsat"
            / "data"
            / "laws"
            / f"{name}.json"
        )
        try:
            pack = json.loads(path.read_text())
        except OSError as e:
            raise SystemExit(
                f"kit-eqsat pack unreadable: {path} ({e})"
            ) from e
        if not isinstance(pack, list):
            raise SystemExit(
                f"kit-eqsat pack is not a record array: {path}"
            )
        records.extend(pack)
    return records


def render(records: list[dict]) -> str:
    """Return the canonical on-disk spelling of the snapshot."""
    return json.dumps(records, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the emitter; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 on drift instead of rewriting the snapshot",
    )
    args = parser.parse_args(argv)

    kit = find_kit_eqsat()
    if kit is None:
        print(
            "kit-eqsat checkout not found — set MLOPT_KIT_EQSAT or "
            "clone handl-lang/kit-eqsat beside this repo",
            file=sys.stderr,
        )
        return 2

    text = render(corpus_records(kit))
    if args.check:
        current = LAWS_PATH.read_text() if LAWS_PATH.is_file() else ""
        if current != text:
            print(
                f"drift: {LAWS_PATH} != kit-eqsat corpus "
                f"({kit}) — rerun scripts/emit_eqsat_laws.py",
                file=sys.stderr,
            )
            return 1
        print(f"eqsat_laws.json in sync with {kit}")
        return 0

    LAWS_PATH.write_text(text)
    n = text.count('"name"')
    print(f"wrote {LAWS_PATH} ({n} records) from {kit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
