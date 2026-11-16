"""Compat shim + the lemma-store codec (ADR-0178 D5, option B).

The term *types* moved to :mod:`mlopt_core.term` — every historical
``from mlopt_core.ir import Op, Var, ...`` keeps working through the
re-export below (private helpers other modules reach for —
``_attr_key``, ``_dag_postorder``, ``_OP_REGISTRY`` — are re-exported
explicitly since ``import *`` skips underscore names).

What stays here is the **lemma-store codec** — ``term_to_data`` /
``term_from_data`` and its ``{"__tag__": "dag.program", "nodes":
[{"op": ...}]}`` dialect.  That is a *pattern/lemma* wire (it carries
metavariable ``{"mvar": ...}`` leaves and the ``{"op", "args",
"attrs"}`` node spelling), NOT the kernel Term-JSON envelope — the
term wire lives in :mod:`mlopt_core.term_json`.  Three different
things are named ``dag.program``; do not conflate them (plan §8 R3).
"""

from __future__ import annotations

from typing import Any

from mlopt_core.term import _OP_REGISTRY as _OP_REGISTRY
from mlopt_core.term import _OP_REPR_BUDGET as _OP_REPR_BUDGET
from mlopt_core.term import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
    attr_from_data,
    attr_to_data,
    generator,
    op_def,
    op_repr,
    op_repr_dag,
)

# Private helpers + ``GenDef`` other modules reach through this shim —
# ``X as X`` marks the deliberate re-export (``__all__`` stays the
# public surface; these were module attributes before the move).
from mlopt_core.term import GenDef as GenDef
from mlopt_core.term import _attr_key as _attr_key
from mlopt_core.term import _dag_postorder as _dag_postorder
from mlopt_core.term import _dag_shared as _dag_shared
from mlopt_core.term import _hashable_attr as _hashable_attr
from mlopt_core.term import _sexp as _sexp
from mlopt_core.term import _tree_sizes as _tree_sizes

__all__ = [
    "IR",
    "Const",
    "Op",
    "Param",
    "TensorType",
    "Var",
    "attr_from_data",
    "attr_to_data",
    "generator",
    "op_def",
    "op_repr",
    "op_repr_dag",
    "term_from_data",
    "term_to_data",
]

# ---------------------------------------------------------------------------
#  Term serialization — the JSON-safe data form (the lemma-store seam)
# ---------------------------------------------------------------------------


#: The DAG envelope's tag — the same ``dag.program`` record name
#: ``kit_discovery.hnl_codec`` uses inside the handl Term-JSON
#: ``Opaque`` slot, carried here in the lemma-store encoding's own
#: spelling: ``{"__tag__": "dag.program", "nodes": [...], "root": n}``.
_DAG_TAG = "dag.program"

#: Tree-expansion budget above which :func:`term_to_data` switches to
#: the ``dag.program`` node-table envelope.  Terms are hash-consed
#: DAGs, and a sharing-heavy one tree-expands exponentially — the
#: whisper encoder's cross-attention fan-out made ``mlopt stage lift``
#: OOM building the nested dicts.  Past the budget the record stores
#: every distinct node once with ``Op.args`` as *indices* into the
#: table (children before parents), so the encoding is linear in
#: distinct nodes.  Below it the classic tree spelling is kept:
#: consumers that walk the encoding structurally (law-pattern
#: metavariable scanners) see the format they already know, and the
#: format only changes where the tree spelling was infeasible.
_TERM_DATA_DAG_BUDGET = 100_000


def _leaf_to_data(t: Any) -> Any:
    """Encode a non-:class:`Op` leaf — the tree form's leaf spellings."""
    if isinstance(t, str):
        return {"mvar": t}
    if isinstance(t, Const):
        return {"const": t.value}
    if isinstance(t, Var):
        return {"var": t.name, "shape": list(t.typ.shape)}
    if isinstance(t, Param):
        return {"param": t.name, "shape": list(t.typ.shape)}
    raise TypeError(f"unserializable term: {t!r}")


def _leaf_from_data(d: dict) -> Any:
    """Decode a non-:class:`Op` leaf record (no child positions)."""
    if "mvar" in d:
        return d["mvar"]
    if "const" in d:
        return Const(d["const"])
    if "var" in d:
        return Var(d["var"], TensorType(tuple(d["shape"])))
    if "param" in d:
        return Param(d["param"], TensorType(tuple(d["shape"])))
    raise ValueError(f"bad term encoding: {d!r}")


def _term_to_data_dag(root: Op) -> dict[str, Any]:
    """Encode *root* as the ``dag.program`` node-table record.

    Every distinct subterm — ops AND leaves — becomes one ``nodes``
    entry in post-order (children before parents), and an ``Op``'s
    ``args`` are *indices* into the table, so a shared subtree is
    emitted once and referenced: the record is linear in distinct
    nodes where the tree spelling is exponential.  Non-``Op`` nodes
    keep their ordinary leaf encodings (``{"mvar": …}`` /
    ``{"const": …}`` / ``{"var": …}`` / ``{"param": …}``).
    """
    from mlopt_core.traversal import postorder

    index: dict[Any, int] = {}
    nodes: list[Any] = []
    for t in postorder(root):
        if isinstance(t, Op):
            rec: Any = {
                "op": t.op,
                "args": [index[a] for a in t.args],
                "attrs": {
                    k: attr_to_data(v) for k, v in t.attrs.items()
                },
            }
        else:
            rec = _leaf_to_data(t)
        index[t] = len(nodes)
        nodes.append(rec)
    return {"__tag__": _DAG_TAG, "nodes": nodes, "root": index[root]}


def term_to_data(t: Any) -> Any:
    """Encode a (sub)term as JSON-safe data: Op tree, metavar, leaf.

    The scheme — canonical since ``rulecache`` adopted it — is
    ``{"mvar": name}`` for a pattern metavariable (a bare ``str``
    leaf), ``{"const": v}`` for :class:`Const`, ``{"var": name,
    "shape": [...]}`` / ``{"param": name, "shape": [...]}`` for typed
    leaves, and ``{"op": name, "args": [...], "attrs": {k: attr}}``
    for :class:`Op` nodes.  Note ``"var"`` means a typed :class:`Var`
    leaf — a *metavariable* is ``"mvar"``.

    An :class:`Op` whose *tree* expansion exceeds
    ``_TERM_DATA_DAG_BUDGET`` nodes (only possible through DAG
    sharing — hash-consing makes equal subtrees the same object)
    encodes as the ``{"__tag__": "dag.program", "nodes": [...],
    "root": n}`` node-table envelope instead: the tree spelling
    would be exponential, the table is linear in distinct nodes.
    :func:`term_from_data` reads both spellings.
    """
    if isinstance(t, Op):
        if _tree_sizes(t)[t] > _TERM_DATA_DAG_BUDGET:
            return _term_to_data_dag(t)
        return {
            "op": t.op,
            "args": [term_to_data(a) for a in t.args],
            "attrs": {k: attr_to_data(v) for k, v in t.attrs.items()},
        }
    return _leaf_to_data(t)


def _term_from_data_dag(d: dict) -> Any:
    """Decode the ``dag.program`` record, rebuilding shared nodes once.

    ``nodes`` is a post-order table — every ``Op.args`` entry is an
    *index* of an earlier node, so children are always built before
    their parents and each distinct node is constructed exactly once:
    the decoded term's sharing is the record's sharing (and
    ``Op.make`` interning makes structurally equal nodes the same
    object regardless).
    """
    nodes = d.get("nodes")
    root = d.get("root")
    if not isinstance(nodes, list):
        raise ValueError("dag.program: 'nodes' must be a list")
    if (
        not isinstance(root, int)
        or isinstance(root, bool)
        or not 0 <= root < len(nodes)
    ):
        raise ValueError(
            f"dag.program: bad root index {root!r} ({len(nodes)} nodes)"
        )
    built: list[Any] = []
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            raise ValueError(
                f"dag.program: node {i} is not an object: {n!r}"
            )
        if "op" in n:
            args = n.get("args")
            if not isinstance(args, list):
                raise ValueError(
                    f"dag.program: node {i} args must be an index list"
                )
            kids = []
            for a in args:
                if (
                    isinstance(a, bool)
                    or not isinstance(a, int)
                    or not 0 <= a < i
                ):
                    raise ValueError(
                        f"dag.program: node {i} arg {a!r} is not an "
                        f"earlier node index (0..{i - 1})"
                    )
                kids.append(built[a])
            built.append(
                Op.make(
                    n["op"],
                    *kids,
                    **{
                        k: attr_from_data(v)
                        for k, v in (n.get("attrs") or {}).items()
                    },
                )
            )
        else:
            built.append(_leaf_from_data(n))
    return built[root]


def term_from_data(d: Any) -> Any:
    """Decode :func:`term_to_data` output back to a term.

    Reads both envelopes: the plain tree and the top-level
    ``dag.program`` node-table record (decoded with sharing intact —
    each table node is built exactly once).
    """
    if not isinstance(d, dict) or len(d) < 1:
        raise ValueError(f"bad term encoding: {d!r}")
    if d.get("__tag__") == _DAG_TAG:
        return _term_from_data_dag(d)
    if "mvar" in d:
        return d["mvar"]
    if "const" in d:
        return Const(d["const"])
    if "var" in d:
        return Var(d["var"], TensorType(tuple(d["shape"])))
    if "param" in d:
        return Param(d["param"], TensorType(tuple(d["shape"])))
    if "op" in d:
        return Op.make(
            d["op"],
            *[term_from_data(a) for a in d["args"]],
            **{k: attr_from_data(v) for k, v in d["attrs"].items()},
        )
    raise ValueError(f"bad term encoding: {d!r}")
