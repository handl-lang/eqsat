"""mlopt term ↔ kernel Term-JSON — THE term wire codec (ADR-0173/0178).

One codec for the whole stack: this is the generalisation of
``kit_discovery.hnl_codec`` (the ``{"Op"/"Arr"/"Const"/"Opaque"}``
serde-shaped handl ``Term`` envelope) that also absorbs the eqsat
bridge's ``_kdag`` node-table spelling, so the orchestrator no longer
needs a private parallel codec (the dep graph forbids it importing
``kit_discovery`` — ``mlopt-orchestrator`` deps on ``mlopt-core``
alone).

**Envelope conventions (pinned here, single home):**

* ``Op`` nodes carry ``dim = {"Const": {"value": 1}}`` in tree
  position and a *node-table index* in ``dag.program`` records (the
  engine-eqsat ``dag.rs`` schema — a nested dim record there is a
  named refusal), ``source = "Native"`` (load-bearing: law patterns
  pin ``source:"native"`` under exact-keyset attr matching —
  ``{"Foreign": "tensor"}`` would silently stop laws firing), and
  attrs under the canonical ``attrs`` field (``lit.__mlopt_attrs__``
  is decode-legacy only, never both).
* ``Var``/``Param`` leaves are ``{"Arr": {"dim": …, "value":
  {"__mlopt_leaf__": kind, "name": …, "shape": […]|null}}}``
  records — the structured leaf spelling.  The legacy eqsat spelling
  ``{"Const": {"value": "__mlopt_leaf__:" + <json {"k","n","s"}>}}``
  decodes permanently (old records) via ``leaf="const-str"`` /
  ``leaf="auto"`` but is never emitted by the canonical profile.
* ``Const`` values and attr payloads ride the recursive
  ``{"__nonfinite__": "nan"|"inf"|"-inf"}`` tag — ``serde_json``
  cannot hold non-finite floats.
* Sharing rides ``{"Opaque": {"__tag__": "dag.program", "nodes":
  [...], "root": n}}`` — a post-order node table, children before
  parents, ``Op.args`` (and ``Op.dim``) as indices.

This is NOT the lemma-store codec: ``mlopt_core.ir.term_to_data`` /
``term_from_data`` is a different wire (``{"op", "args", "attrs"}``
nodes, ``{"mvar": …}`` metavariable leaves, its own top-level
``{"__tag__": "dag.program"}`` dialect).  Three different things are
named ``dag.program`` — do not conflate them.

Honest declines, never silent wrong answers: ``dag=False`` refuses a
shared term rather than tree-expanding it; ``allow_dag=False``
refuses the record; unknown ``Opaque`` tags, malformed node tables,
and undecodable records raise :class:`TermJsonError`.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from mlopt_core.term import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    attr_from_data,
    attr_to_data,
)

__all__ = [
    "TermJsonError",
    "term_from_json",
    "term_to_json",
    "terms_to_json",
]


class TermJsonError(ValueError):
    """The JSON is not a term this codec can faithfully represent."""


# ---------------------------------------------------------------------------
#  Wire conventions — the kernel Term serde envelope
# ---------------------------------------------------------------------------

#: Non-finite scalar tag — the host-tensor wire spelling.
_NONFINITE_TAG = "__nonfinite__"
_NONFINITE_VALUES = ("nan", "inf", "-inf")

#: The dag record's ``__tag__`` value, carried inside ``Term::Opaque``.
_DAG_TAG = "dag.program"

#: ``Term::Op.dim`` on this wire is always ``Const(1)`` — nested-record
#: spelling in tree position, shared node index in dag records.
_DIM1 = {"Const": {"value": 1}}

#: Law patterns pin ``source:"native"`` under exact-keyset attr
#: matching — the search wire MUST emit ``"Native"``, never
#: ``{"Foreign": "tensor"}`` (decode accepts/ignores any ArrSource).
_SOURCE = "Native"

#: Legacy eqsat leaf spelling: a ``Const`` string carrying
#: ``{"k": kind, "n": name, "s": shape}`` — decode-only.
_LEAF_PREFIX = "__mlopt_leaf__:"
_LEAF_KINDS = ("var", "param")

#: Canonical leaf record payload (inside ``Arr.value``).
_LEAF_RECORD_TAG = "__mlopt_leaf__"

#: Encode leaf profiles.
_LEAF_PROFILES = ("arr", "const-str")


# ---------------------------------------------------------------------------
#  ``__nonfinite__`` — scalar ↔ wire value
# ---------------------------------------------------------------------------


def _wire_value(v: Any) -> Any:
    """Encode *v* (post-``attr_to_data`` shape) as JSON-safe data."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        if math.isnan(v):
            return {_NONFINITE_TAG: "nan"}
        return {_NONFINITE_TAG: "inf" if v > 0 else "-inf"}
    if isinstance(v, (list, tuple)):
        return [_wire_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _wire_value(x) for k, x in v.items()}
    raise TypeError(f"unserializable wire value: {v!r}")


def _unwire_value(v: Any) -> Any:
    """Decode :func:`_wire_value` output — tags back to floats."""
    if isinstance(v, dict):
        if set(v) == {_NONFINITE_TAG}:
            s = v[_NONFINITE_TAG]
            if s not in _NONFINITE_VALUES:
                raise TermJsonError(
                    f"bad {_NONFINITE_TAG} payload {s!r} — expected "
                    "'nan', 'inf' or '-inf'"
                )
            if s == "nan":
                return float("nan")
            return float("inf") if s == "inf" else float("-inf")
        return {k: _unwire_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unwire_value(x) for x in v]
    return v


# ---------------------------------------------------------------------------
#  Shared helpers — leaf records, leaf payloads, the attrs channel
# ---------------------------------------------------------------------------


def _shape_data(typ: TensorType) -> Any:
    return None if typ.shape is None else list(typ.shape)


def _shape_of(v: Any) -> Any:
    """Return *v* in ``TensorType.shape``'s runtime-legal form.

    ``None`` is the wholly-unknown wildcard the annotation does not
    admit — the same cast ``hnl_codec`` used for the contract.
    """
    if v is None:
        return cast("tuple[int | None, ...]", None)
    return cast("tuple[int | None, ...]", tuple(v))


def _leaf_record(kind: str, name: str, typ: TensorType) -> dict:
    """Return the ``Arr.value`` payload — the canonical structured leaf."""
    return {
        _LEAF_RECORD_TAG: kind,
        "name": name,
        "shape": _shape_data(typ),
    }


def _leaf_payload(t: Var | Param) -> str:
    """Return the legacy ``Const``-string leaf payload (encode-only)."""
    shape = None if t.typ.shape is None else list(t.typ.shape)
    kind = "var" if isinstance(t, Var) else "param"
    return _LEAF_PREFIX + json.dumps(
        {"k": kind, "n": t.name, "s": shape}, separators=(",", ":")
    )


def _attrs_payload(attrs: dict[str, Any]) -> dict:
    """Return the canonical ``attrs`` field payload, wire-encoded."""
    return {k: _wire_value(attr_to_data(v)) for k, v in attrs.items()}


def _attrs_from_node(node: dict) -> dict[str, Any]:
    """Decode an ``Op`` payload's attrs — one channel, never ambiguous.

    Accepts the canonical ``attrs`` field and the legacy
    ``lit.__mlopt_attrs__`` record.  Both present is a decline: two
    spellings of one fact must not be silently merged.
    """
    lit = node.get("lit")
    field = node.get("attrs")
    if lit is not None and field is not None:
        raise TermJsonError(
            "Op carries both lit.__mlopt_attrs__ and attrs — "
            "ambiguous attrs channel"
        )
    if field is not None:
        if not isinstance(field, dict):
            raise TermJsonError(
                f"Op.attrs must be an object, got {field!r}"
            )
        raw = field
    elif lit is not None:
        if not isinstance(lit, dict) or "__mlopt_attrs__" not in lit:
            raise TermJsonError(
                f"Op.lit is not an attrs record: {lit!r}"
            )
        raw = lit["__mlopt_attrs__"]
        if not isinstance(raw, dict):
            raise TermJsonError(
                f"__mlopt_attrs__ must be an object, got {raw!r}"
            )
    else:
        return {}
    try:
        return {
            k: attr_from_data(_unwire_value(v)) for k, v in raw.items()
        }
    except (TypeError, ValueError) as e:
        raise TermJsonError(f"bad attrs payload: {e}") from e


def _budget_state(budget: int | list[int] | None) -> list[int] | None:
    """Normalise the ``budget`` argument to a ``[emitted, limit]`` cell.

    An int bounds this call; a two-element list is a *shared* cell —
    the ``_kdag`` ``state`` convention — so a caller encoding a term
    LIST gets one bound across all of them (each term still gets its
    own node table).  ``None`` is unbounded (the historical
    ``hnl_codec`` behaviour) — no cell, no counting.
    """
    if budget is None:
        return None
    if isinstance(budget, list):
        if len(budget) != 2 or not all(
            isinstance(x, int) for x in budget
        ):
            raise TermJsonError(
                f"budget state must be [emitted, limit] ints, "
                f"got {budget!r}"
            )
        return budget
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TermJsonError(f"bad node budget: {budget!r}")
    return [0, budget]


def _count(state: list[int] | None, what: str) -> None:
    """Charge one emitted node against the shared budget cell."""
    if state is None:
        return
    state[0] += 1
    if state[0] > state[1]:
        raise TermJsonError(
            f"term wire {what} exceeds wire_node_budget "
            f"({state[1]} nodes) — shared-subtree blowup; raise the "
            "budget or encode through the dag.program record"
        )


# ---------------------------------------------------------------------------
#  Encode — mlopt term → Term JSON
# ---------------------------------------------------------------------------


def _leaf_json(t: Var | Param, leaf: str) -> dict:
    if leaf == "arr":
        kind = "var" if isinstance(t, Var) else "param"
        return {
            "Arr": {
                "dim": _DIM1,
                "value": _leaf_record(kind, t.name, t.typ),
            }
        }
    if leaf == "const-str":
        return {"Const": {"value": _leaf_payload(t)}}
    raise TermJsonError(f"unknown leaf profile {leaf!r}")


def _enc(t: Any, leaf: str, state: list[int] | None) -> dict:
    """Tree-encode *t* — valid only where no subtree is shared."""
    if state is not None:
        _count(state, "tree expansion")
    if isinstance(t, Op):
        node: dict[str, Any] = {
            "dim": _DIM1,
            "name": t.op,
            "args": [_enc(a, leaf, state) for a in t.args],
            "source": _SOURCE,
        }
        if t.attrs:
            node["attrs"] = _attrs_payload(t.attrs)
        return {"Op": node}
    if isinstance(t, (Var, Param)):
        return _leaf_json(t, leaf)
    if isinstance(t, Const):
        return {"Const": {"value": _wire_value(t.value)}}
    raise TypeError(f"not a wireable term: {t!r}")


def _has_sharing(root: Any) -> bool:
    """Whether any node is referenced by more than one parent.

    Counts *child occurrences* over the DAG — mlopt's ``Op.make``
    interning means structurally equal subtrees are the same object,
    so counting is exactly semantic sharing.  The walk is linear in
    distinct nodes even on exponential tree-expansions.
    """
    counts: dict[Any, int] = {}
    seen: set[Any] = set()

    def walk(t: Any) -> None:
        if not isinstance(t, Op) or t in seen:
            return
        seen.add(t)
        for a in t.args:
            counts[a] = counts.get(a, 0) + 1
            walk(a)

    walk(root)
    return any(c > 1 for c in counts.values())


def _enc_dag(t: Any, leaf: str, state: list[int] | None) -> dict:
    """Encode *t* as the ``dag.program`` node-table record.

    Every distinct node becomes one ``nodes`` entry in post-order
    (children before parents); an ``Op``'s ``dim`` and ``args`` are
    *indices* into the table — the engine-eqsat ``dag.rs`` schema.
    Dedup keys on the term object itself (content-hashable:
    ``Op.make`` interning plus the frozen-leaf dataclasses), so
    structurally equal subtrees merge even without object identity.
    ``state`` is the shared ``[emitted, budget]`` cell.
    """
    index: dict[Any, int] = {}
    nodes: list[dict] = []
    dim_slot: list[int] = []

    def dim_index() -> int:
        # ``Term::Op.dim`` is always ``Const(1)`` on this wire — emit
        # the leaf node once per record and share its index.
        if not dim_slot:
            dim_slot.append(emit(Const(1)))
        return dim_slot[0]

    def emit(x: Any) -> int:
        try:
            hit = index.get(x)
        except TypeError:
            hit = None  # unhashable — falls through to the type checks
        if hit is not None:
            return hit
        _count(state, "node table")
        if isinstance(x, Op):
            arg_idx = [emit(a) for a in x.args]
            op_node: dict[str, Any] = {
                "dim": dim_index(),
                "name": x.op,
                "args": arg_idx,
                "source": _SOURCE,
            }
            if x.attrs:
                op_node["attrs"] = _attrs_payload(x.attrs)
            record: dict[str, Any] = {"Op": op_node}
        elif isinstance(x, (Var, Param)):
            record = _leaf_json(x, leaf)
        elif isinstance(x, Const):
            record = {"Const": {"value": _wire_value(x.value)}}
        else:
            raise TypeError(f"not a wireable term: {x!r}")
        index[x] = len(nodes)
        nodes.append(record)
        return index[x]

    root = emit(t)
    return {
        "Opaque": {
            "__tag__": _DAG_TAG,
            "nodes": nodes,
            "root": root,
        }
    }


def term_to_json(
    term: Any,
    *,
    dag: bool | str | None = None,
    leaf: str = "arr",
    budget: int | list[int] | None = None,
) -> dict:
    """Encode *term* as kernel Term-JSON.

    ``dag=None`` (default) picks the envelope honestly: a term with
    shared subtrees serializes as the ``dag.program`` record, an
    unshared one as the plain tree.  ``dag=True`` forces the record;
    ``dag=False`` refuses a shared term with :class:`TermJsonError`
    rather than tree-expanding it; ``dag="tree"`` force-expands
    sharing into the tree spelling anyway — the legacy eqsat
    ``_kterm`` mode, always budget-counted so the exponential
    expansion stays a named overage, never a surprise.

    ``leaf`` selects the ``Var``/``Param`` spelling: ``"arr"`` (the
    canonical structured ``Arr`` record) or ``"const-str"`` (the
    legacy ``__mlopt_leaf__:``-prefixed ``Const`` string — the
    engine-eqsat profile until the ``arr``-enode round-trip lands
    everywhere it must).

    ``budget`` bounds emitted wire nodes: an int bounds this call, a
    ``[emitted, limit]`` list is a shared cell across calls (the
    ``_kdag`` ``state`` convention — see :func:`terms_to_json`).
    """
    if leaf not in _LEAF_PROFILES:
        raise TermJsonError(
            f"unknown leaf profile {leaf!r} — expected one of "
            f"{_LEAF_PROFILES}"
        )
    if dag not in (None, True, False, "tree"):
        raise TermJsonError(f"bad dag mode {dag!r}")
    state = _budget_state(budget)
    if dag == "tree":
        return _enc(term, leaf, state)
    shared = _has_sharing(term)
    if dag is False and shared:
        raise TermJsonError(
            "term has shared subtrees — dag=False cannot represent "
            "sharing (use the dag.program record or dag=None)"
        )
    if dag is True or (dag is None and shared):
        return _enc_dag(term, leaf, state)
    return _enc(term, leaf, state)


def terms_to_json(
    terms: list,
    *,
    leaf: str = "arr",
    budget: int | None = None,
) -> list:
    """Encode a term LIST as ``dag.program`` records, one budget shared.

    Every term crosses as the node-table envelope — uniform (unshared
    terms pay nothing: the table is the same nodes the tree spelling
    would emit, plus index indirection) and shallow (term depth never
    reaches ``serde_json``'s recursion limit).  ``budget`` bounds
    TOTAL distinct emitted nodes across the whole list — the
    ``wire_node_budget`` contract the eqsat bridge pins.
    """
    state = _budget_state(budget)
    return [
        term_to_json(t, dag=True, leaf=leaf, budget=state)
        for t in terms
    ]


# ---------------------------------------------------------------------------
#  Decode — Term JSON → mlopt term
# ---------------------------------------------------------------------------


def _leaf_from_payload(value: str) -> Any:
    """Decode a ``_LEAF_PREFIX`` const back to ``Var``/``Param``."""
    try:
        rec = json.loads(value[len(_LEAF_PREFIX) :])
    except ValueError as e:
        raise TermJsonError(f"bad leaf record {value!r}: {e}") from e
    if not isinstance(rec, dict):
        raise TermJsonError(f"bad leaf record {value!r}")
    try:
        name = rec["n"]
        shape = _shape_of(rec.get("s"))
    except (KeyError, TypeError) as e:
        raise TermJsonError(f"bad leaf record {value!r}: {e}") from e
    kind = rec.get("k")
    if kind == "var":
        return Var(name, TensorType(shape))
    if kind == "param":
        return Param(name, TensorType(shape))
    raise TermJsonError(f"bad leaf kind in {value!r}")


def _dec_arr_leaf(node: dict) -> Any:
    """Decode an ``Arr`` record's ``__mlopt_leaf__`` payload."""
    arr = node["Arr"]
    if not isinstance(arr, dict):
        raise TermJsonError(f"bad Arr record: {arr!r}")
    rec = arr.get("value")
    if not isinstance(rec, dict) or _LEAF_RECORD_TAG not in rec:
        raise TermJsonError(
            f"Arr leaf has no {_LEAF_RECORD_TAG} payload: {rec!r}"
        )
    kind = rec[_LEAF_RECORD_TAG]
    if kind not in _LEAF_KINDS:
        raise TermJsonError(f"bad leaf kind {kind!r} in {rec!r}")
    try:
        name = rec["name"]
        shape = _shape_of(rec.get("shape"))
    except (KeyError, TypeError) as e:
        raise TermJsonError(f"bad leaf record {rec!r}: {e}") from e
    if kind == "var":
        return Var(name, TensorType(shape))
    return Param(name, TensorType(shape))


def _dec_leaf(j: dict) -> Any:
    """Decode an ``Arr``/``Const`` node (no children to resolve)."""
    if "Arr" in j:
        return _dec_arr_leaf(j)
    if "Const" in j:
        v = j["Const"]["value"]
        if isinstance(v, str) and v.startswith(_LEAF_PREFIX):
            return _leaf_from_payload(v)
        return Const(_unwire_value(v))
    raise TermJsonError(f"undecodable term record: {j!r}")


def _dec(j: Any, validate: bool) -> Any:
    """Decode a tree-position Term JSON node."""
    if not isinstance(j, dict) or len(j) < 1:
        raise TermJsonError(f"bad term encoding: {j!r}")
    if "Opaque" in j:
        raise TermJsonError(
            "nested Opaque node — only a top-level dag.program record "
            "is supported"
        )
    if "Op" in j:
        node = j["Op"]
        return Op.make(
            node["name"],
            *[_dec(a, validate) for a in node["args"]],
            validate=validate,
            **_attrs_from_node(node),
        )
    if "Arr" in j or "Const" in j:
        return _dec_leaf(j)
    tag = next(iter(j), None)
    raise TermJsonError(
        f"undecodable term record: tag {tag!r} has no mlopt term "
        "spelling"
    )


def _dec_dag(payload: Any, validate: bool) -> Any:
    """Decode the ``dag.program`` record, rebuilding shared nodes once.

    ``nodes`` is a post-order table — every ``Op.args`` entry (and an
    integer ``Op.dim``) is an *index* of an earlier node, so children
    are always built before their parents and each distinct node is
    constructed exactly once: the decoded term's sharing is the
    record's sharing.  A *nested-record* ``dim`` (the legacy
    ``hnl_codec`` dag spelling) is accepted and ignored — mlopt terms
    carry no dim.
    """
    if not isinstance(payload, dict):
        raise TermJsonError(
            f"dag.program: payload must be an object: {payload!r}"
        )
    nodes = payload.get("nodes")
    root = payload.get("root")
    if not isinstance(nodes, list):
        raise TermJsonError("dag.program: 'nodes' must be a list")
    if (
        not isinstance(root, int)
        or isinstance(root, bool)
        or not 0 <= root < len(nodes)
    ):
        raise TermJsonError(
            f"dag.program: bad root index {root!r} ({len(nodes)} nodes)"
        )
    built: list[Any] = []
    for i, n in enumerate(nodes):
        if not isinstance(n, dict) or len(n) != 1:
            raise TermJsonError(
                f"dag.program: node {i} is not a tagged record: {n!r}"
            )
        if "Op" in n:
            node = n["Op"]
            args = node.get("args")
            if not isinstance(args, list):
                raise TermJsonError(
                    f"dag.program: node {i} args must be an index list"
                )
            kids = []
            for a in args:
                if (
                    not isinstance(a, int)
                    or isinstance(a, bool)
                    or not 0 <= a < i
                ):
                    raise TermJsonError(
                        f"dag.program: node {i} arg {a!r} is not an "
                        f"earlier node index (0..{i - 1})"
                    )
                kids.append(built[a])
            dim = node.get("dim")
            if isinstance(dim, int) and not isinstance(dim, bool):
                if not 0 <= dim < i:
                    raise TermJsonError(
                        f"dag.program: node {i} dim {dim!r} is not an "
                        f"earlier node index (0..{i - 1})"
                    )
            elif dim is not None and not isinstance(dim, dict):
                raise TermJsonError(
                    f"dag.program: node {i} dim must be a node index "
                    f"or a Const record, got {dim!r}"
                )
            built.append(
                Op.make(
                    node["name"],
                    *kids,
                    validate=validate,
                    **_attrs_from_node(node),
                )
            )
        elif "Arr" in n or "Const" in n:
            built.append(_dec_leaf(n))
        else:
            raise TermJsonError(
                f"dag.program: node {i} has unknown tag {list(n)[:3]}"
            )
    return built[root]


def term_from_json(
    data: Any,
    *,
    allow_dag: bool = True,
    validate: bool = True,
    leaf: str = "auto",
) -> Any:
    """Decode Term JSON back to an mlopt term.

    Understands both envelopes: the plain tree and the
    ``dag.program`` record (decoded with sharing intact), and BOTH
    leaf spellings permanently — the canonical ``Arr`` record and the
    legacy ``__mlopt_leaf__:``-prefixed ``Const`` string (``leaf`` is
    accepted for signature stability; decode is always ``"auto"`` —
    old records must keep reading).

    ``allow_dag=False`` refuses the record for consumers that cannot
    represent sharing.  ``validate`` forwards to ``Op.make`` — the
    eqsat bridge decodes law-extracted terms with ``validate=False``
    (the Python ``Op.make`` contract default is ``True``).

    Any other ``Opaque`` tag declines — mlopt terms have no opaque
    slot, so carrying one would be a silent wrong answer.
    """
    if leaf != "auto":
        raise TermJsonError(
            f"decode accepts both leaf spellings — leaf={leaf!r} "
            "is encode-only"
        )
    if not isinstance(data, dict) or len(data) < 1:
        raise TermJsonError(f"bad term encoding: {data!r}")
    if "Opaque" in data:
        payload = data["Opaque"]
        if (
            isinstance(payload, dict)
            and payload.get("__tag__") == _DAG_TAG
        ):
            if not allow_dag:
                raise TermJsonError(
                    "dag.program record — this consumer cannot "
                    "represent sharing (allow_dag=False)"
                )
            return _dec_dag(payload, validate)
        tag = (
            payload.get("__tag__")
            if isinstance(payload, dict)
            else None
        )
        raise TermJsonError(
            f"Opaque term (tag {tag!r}) has no mlopt term spelling"
        )
    return _dec(data, validate)
