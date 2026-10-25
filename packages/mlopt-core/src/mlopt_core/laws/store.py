"""Minted records as a queryable store, not embedded source.

A minted object — the promoted composite's premise closure plus the
shipped ``synop*`` primitive families — is *data*: a ``law_json`` row
with its pattern pair, guard, certificate and handler.  Until now those
rows lived as four pretty-printed JSON literals inside
:mod:`mlopt_core.laws.minted` (≈40 000 lines of generated source,
replayed by hand-pasted edits).  They live here instead: one SQLite
file shipped as package data, read at import, queried with SQL.

Layout — one row per object, one table::

    records(tier INTEGER, seq INTEGER, name TEXT, kind TEXT,
            sha TEXT, payload TEXT, PRIMARY KEY (tier, name))

``tier`` mirrors the four sets the mint produces:

===== =========================================================
tier  what
===== =========================================================
0     promoted composite scaffolding (``_RECORDS``)
1     depth-1 prims (``_PRIM_RECORDS``, 21)
2     depth-2 prims (``_PRIM2_RECORDS``, 2)
3     depth-3 prims (``_PRIM3_RECORDS``, 34)
===== =========================================================

``payload`` is the canonical JSON text of the record (``sort_keys``, no
spaces) — loading it reproduces exactly the dict the old literal
produced.  ``sha`` is the first 16 hex of its SHA-256, so the mint can
address a row by content.  ``seq`` preserves the original insertion
order, which the fold/premise numbering depends on.

Reading is **read-only** by default: the shipped file is an artifact,
never a scratchpad.  :func:`emit` is the write path for the mint.

Query it directly::

    sqlite3 minted.db "select tier, count(*) from records group by 1"
    sqlite3 minted.db "select name, kind from records where kind='prim'"
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "DB_PATH",
    "TIERS",
    "connect",
    "emit",
    "load",
    "query",
    "summary",
    "tiers",
]

#: The shipped store, beside this module (package data).
DB_PATH: Path = Path(__file__).with_name("data") / "minted.db"

#: The tier numbers :func:`load` returns, in the mint's own order.
TIERS: tuple[int, ...] = (0, 1, 2, 3)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    tier    INTEGER NOT NULL,
    seq     INTEGER NOT NULL,
    name    TEXT    NOT NULL,
    kind    TEXT,
    sha     TEXT    NOT NULL,
    payload TEXT    NOT NULL,
    PRIMARY KEY (tier, name)
);
CREATE INDEX IF NOT EXISTS records_tier_seq ON records (tier, seq);
CREATE INDEX IF NOT EXISTS records_kind     ON records (kind);
"""


def _canonical(record: Any) -> str:
    """Return the record's canonical JSON text (stored and hashed)."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def connect(
    path: Path | str | None = None, *, read_only: bool = True
) -> sqlite3.Connection:
    """Open the store.  Read-only unless a writer asks for otherwise."""
    p = Path(path) if path is not None else DB_PATH
    if read_only:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(p)
        conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def load(path: Path | str | None = None) -> dict[int, dict[str, Any]]:
    """Load every record, tier → {name: record}, in original order.

    Returns a dict for all four tiers (an empty dict for a tier with no
    rows), so ``load()[1]`` is the depth-1 prim table whatever ships.
    """
    out: dict[int, dict[str, Any]] = {t: {} for t in TIERS}
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT tier, name, payload FROM records ORDER BY tier, seq"
        ).fetchall()
    except (
        sqlite3.OperationalError
    ) as exc:  # no such table — not a store
        raise FileNotFoundError(
            f"minted store not found: {path or DB_PATH}"
        ) from exc
    finally:
        conn.close()
    for row in rows:
        out.setdefault(int(row["tier"]), {})[row["name"]] = json.loads(
            row["payload"]
        )
    return out


def emit(
    rows: Mapping[int, Mapping[str, Any]],
    path: Path | str | None = None,
    *,
    replace: bool = False,
) -> int:
    """Write records — the mint's emit path.  Returns the rows written.

    *rows* maps tier → {name: record}.  With ``replace`` the store is
    rewritten from *rows*; otherwise rows upsert by ``(tier, name)`` and
    keep their order, new names appending.  Idempotent: an identical
    record is not rewritten.
    """
    conn = connect(path, read_only=False)
    try:
        with conn:
            if replace:
                conn.execute("DELETE FROM records")
            written = 0
            for tier in sorted(rows):
                next_seq = conn.execute(
                    "SELECT COALESCE(MAX(seq) + 1, 0) FROM records WHERE tier = ?",
                    (int(tier),),
                ).fetchone()[0]
                for name, record in rows[tier].items():
                    payload = _canonical(record)
                    sha = _sha(payload)
                    kind = (
                        record.get("kind")
                        if isinstance(record, dict)
                        else None
                    )
                    cur = conn.execute(
                        "SELECT sha FROM records WHERE tier = ? AND name = ?",
                        (int(tier), name),
                    ).fetchone()
                    if cur is None:
                        conn.execute(
                            "INSERT INTO records (tier, seq, name, kind, sha, payload)"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                int(tier),
                                next_seq,
                                name,
                                kind,
                                sha,
                                payload,
                            ),
                        )
                        next_seq += 1
                        written += 1
                    elif cur["sha"] != sha:
                        conn.execute(
                            "UPDATE records SET kind = ?, sha = ?, payload = ?"
                            " WHERE tier = ? AND name = ?",
                            (kind, sha, payload, int(tier), name),
                        )
                        written += 1
            return written
    finally:
        conn.close()


def query(
    sql: str,
    params: tuple[Any, ...] = (),
    *,
    path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """Run a read-only SQL query against the store."""
    conn = connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def summary(path: Path | str | None = None) -> dict[str, int]:
    """Row counts per kind — the cheapest "what is in the bank" query."""
    rows = query(
        "SELECT COALESCE(kind, 'unnamed') AS kind, COUNT(*) AS n"
        " FROM records GROUP BY kind ORDER BY n DESC",
        path=path,
    )
    return {row["kind"]: int(row["n"]) for row in rows}


def tiers(path: Path | str | None = None) -> dict[int, int]:
    """Record counts per tier."""
    rows = query(
        "SELECT tier, COUNT(*) AS n FROM records GROUP BY tier",
        path=path,
    )
    return {int(row["tier"]): int(row["n"]) for row in rows}


def iter_records(
    path: Path | str | None = None,
) -> Iterator[sqlite3.Row]:
    """Stream rows (tier, name, kind, sha, payload) in store order."""
    yield from query(
        "SELECT tier, name, kind, sha, payload FROM records ORDER BY tier, seq",
        path=path,
    )
