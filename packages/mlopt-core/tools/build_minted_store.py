#!/usr/bin/env python3
"""Build (or refresh) the minted-record store.

Two subcommands, because the payload has two lives:

``dump``   read the four embedded record tables out of
           :mod:`mlopt_core.laws.minted` and write one JSON file
           (``{tier: {name: record}}``).  This is the bootstrap path —
           it only works while the literals are still in the module —
           and it is how a *new* mint run's records leave the evidence
           store for review.

``build``  turn that JSON into the shipped SQLite store
           (``src/mlopt_core/laws/data/minted.db``), with ``--check``
           refusing to write when the store already matches.

Run from the repo root::

    python packages/mlopt-core/tools/build_minted_store.py dump  --out /tmp/minted.json
    python packages/mlopt-core/tools/build_minted_store.py build --json /tmp/minted.json
    python packages/mlopt-core/tools/build_minted_store.py show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # packages/mlopt-core
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlopt_core.laws import store  # noqa: E402

#: Module attribute holding each tier's table, in tier order.
_ATTRS = {
    0: "_RECORDS",
    1: "_PRIM_RECORDS",
    2: "_PRIM2_RECORDS",
    3: "_PRIM3_RECORDS",
}


def cmd_dump(args: argparse.Namespace) -> int:
    """Dump the embedded tier tables to one JSON file."""
    from mlopt_core.laws import minted

    tiers = {}
    for tier, attr in _ATTRS.items():
        table = getattr(minted, attr, None)
        if table is None:
            print(
                f"tier {tier}: {attr} absent (already migrated)",
                file=sys.stderr,
            )
            continue
        tiers[tier] = table
        print(
            f"tier {tier}: {len(table)} records from {attr}",
            file=sys.stderr,
        )
    out = Path(args.out)
    out.write_text(
        json.dumps(
            {str(t): r for t, r in tiers.items()},
            sort_keys=True,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build (or verify) the SQLite store from dumped JSON."""
    raw = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rows = {int(tier): table for tier, table in raw.items()}
    if args.check:
        same = store.load(args.db) if Path(args.db).exists() else {}
        current = {
            t: {n: r for n, r in table.items()}
            for t, table in same.items()
        }
        if current == rows:
            print("store matches the payload — nothing to write")
            return 0
        print("store differs from the payload", file=sys.stderr)
        return 1
    written = store.emit(rows, args.db, replace=True)
    counts = store.tiers(args.db)
    print(f"wrote {written} rows → {args.db}")
    print(
        "per tier: "
        + ", ".join(f"{t}:{n}" for t, n in sorted(counts.items()))
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print the store tier and kind counts."""
    print(f"store: {args.db}")
    print(
        "per tier: "
        + ", ".join(
            f"{t}:{n}" for t, n in sorted(store.tiers(args.db).items())
        )
    )
    print(
        "per kind: "
        + json.dumps(store.summary(args.db), sort_keys=True)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested subcommand."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db", default=str(store.DB_PATH), help="store path"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dump", help="dump the embedded tables to JSON")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("build", help="build the store from dumped JSON")
    p.add_argument("--json", required=True)
    p.add_argument(
        "--check", action="store_true", help="fail if it would change"
    )
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("show", help="print tier/kind counts")
    p.set_defaults(func=cmd_show)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
