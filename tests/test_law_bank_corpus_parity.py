"""Drift guard: the Python law bank vs the kit-eqsat ``LAW_FORMAT = 2`` corpus.

``handl-lang/kit-eqsat`` is the canonical *records* home (ADR-0177 §1);
``mlopt_core.laws`` is the canonical *implementation* — the oracle the
``engine="eqsat"`` search is differentially measured against (the
demotion plan keeps ``laws/`` Python-side permanently: discovery,
procedural hooks, certificates).  The two spell the same 2-cells in
different concrete syntax, so this guard pins the *comparable*
surface and names every known divergence rather than pretending the
records are projections:

* **Coverage** — every corpus record must name a shipped bank rule
  (``prim:*`` records resolve through the minted store's materialized
  folds).  A kit record Python does not know would silently widen the
  eqsat wire bank past the oracle; the 301 Python-only rules are the
  documented port backlog (prior_art 164, attention 37, scan 22,
  layout/headshare/pairing/specials/factored tails, ``conv_bn_fold``'s
  procedural hooks), not drift.
* **Op-tree shape** — for every non-prim record, the ``lhs``/``rhs``
  op-name trees (args structure, ``mvar``/``const`` leaf kinds,
  attrs ignored) must equal ``law_to_data``'s.  The kit pattern
  spelling (``{"op","args","attrs","dim","lit","source"}`` nodes,
  ``{"bind"|"lit"}`` attr tags, ``{"fields": …}`` envelopes) is a
  different *encoding* of the same patterns — verified equal at this
  granularity, which is as deep as a spelling-free comparison goes.
* **Attrs** — a best-effort canonicalisation (``bind`` → metavar,
  ``lit``/raw → literal, ``fields`` unwrapped, whole-attrs
  ``{"lit": {record}}`` unwrapped to per-name literals, seq
  spellings unified) must match except for the named
  ``_ATTR_DIVERGENT`` set: ``{"bind": X}`` whole-attrs binds the
  entire attrs payload to one metavariable — the attr *name* is
  unrecoverable and the match is deliberately more permissive than
  the Python exact-keyset per-key spelling (engine-eqsat's
  ``AttrsSpec::Bind``, documented at ``builtin.rs``), so those
  records keep a sentinel the Python side never produces.
* **Record fields** — ``law`` / ``cond`` / ``dspec`` / ``tags`` /
  ``derivation`` / ``error_bound`` / ``bound_norm`` / ``serializable``
  / ``missing_hooks`` compare against ``law_to_data`` — verbatim
  except for the named ``_FIELD_DIVERGENT`` set and the two
  canonicalised comparisons below.  ``cond`` compares after
  :func:`_canon_cond` flattens nested same-op ``and``/``or`` nodes:
  both evaluators (``eval_cond``, engine-eqsat's ``Cond::from_data``)
  read them as n-ary combinators, so ``("and", a, ("and", b, c))`` and
  ``("and", a, b, c)`` are the same guard spelled two ways — the Rust
  port re-authored 19 records' conds with different (but
  leaf-identical) nesting, which is a spelling delta like the attrs
  envelopes, not a semantic one.  ``derivation`` compares as
  *provenance consistency*, not completeness: engine-eqsat carries
  it as record-level provenance that is never folded into the
  2-cell (``law.rs``), and on the Python side its only consumer is
  the ``Rewrite.kind`` axiom/lemma taxonomy (coherence reporting,
  not search behaviour).  The guard therefore flags only a kit
  record that asserts a derivation name the Python rule does not
  carry (kit-minus-py non-empty) — the port under-records provenance
  on eight fold/reverse laws (``[]`` where Python names the
  expansion it folds), which claims less, never something false.
* **prim records** — ``prim:*`` names resolve through the minted
  store and their records must equal ``minted.prim_kit_records()``
  verbatim (the projection ``scripts/project_prim_folds.py --check``
  guards against the kit checkout; this pins it inside the suite).

The kit-eqsat checkout resolves exactly as ``scripts/emit_eqsat_laws.py``
does (``$MLOPT_KIT_EQSAT`` → the ``handl-lang`` sibling →
``~/GitHub/handl-lang/kit-eqsat``); when absent the tests skip,
mirroring ``tests/test_eqsat_laws_corpus.py``.
"""

from __future__ import annotations

import importlib.util
import pkgutil
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EMITTER_PATH = _REPO_ROOT / "scripts" / "emit_eqsat_laws.py"


def _emitter():
    """Load ``scripts/emit_eqsat_laws.py`` as a module (not a package)."""
    spec = importlib.util.spec_from_file_location(
        "emit_eqsat_laws", _EMITTER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kit_records() -> list[dict]:
    """Return the flat kit-eqsat corpus, or skip when unresolvable."""
    emit = _emitter()
    kit = emit.find_kit_eqsat()
    if kit is None:
        pytest.skip(
            "kit-eqsat sibling checkout not found — set "
            "MLOPT_KIT_EQSAT or clone handl-lang/kit-eqsat beside "
            "this repo"
        )
    return emit.corpus_records(kit)


# ---------------------------------------------------------------------------
#  The Python bank census — every Rewrite the laws package ships
# ---------------------------------------------------------------------------


def _bank() -> dict[str, Any]:
    """Return ``{rule name: Rewrite}`` across ``mlopt_core.laws``.

    Walks every submodule and collects the public rule collections
    (``*_RULES`` / ``*_LAWS`` / ``*_FOLDS`` tuples of ``Rewrite``) —
    the same surface ``laws.ruleset`` presets compose.  Names are
    unique across the bank (``RuleSet.union`` raises on collision);
    a duplicate name here is itself drift and fails the census test.
    """
    import mlopt_core.laws as laws_pkg

    bank: dict[str, Any] = {}
    collisions: list[str] = []
    for mod_info in pkgutil.iter_modules(laws_pkg.__path__):
        module = __import__(
            f"mlopt_core.laws.{mod_info.name}", fromlist=["*"]
        )
        for attr in dir(module):
            if not (
                attr.endswith("RULES")
                or attr.endswith("_LAWS")
                or attr.endswith("_FOLDS")
            ):
                continue
            value = getattr(module, attr)
            if not isinstance(value, (list, tuple)):
                continue
            for rule in value:
                if not (
                    hasattr(rule, "lhs")
                    and hasattr(rule, "rhs")
                    and hasattr(rule, "name")
                ):
                    continue
                prev = bank.get(rule.name)
                if prev is not None and prev is not rule:
                    collisions.append(rule.name)
                bank[rule.name] = rule
    assert not collisions, (
        f"duplicate rule names across mlopt_core.laws: {collisions}"
    )
    return bank


# ---------------------------------------------------------------------------
#  Pattern comparison — op-tree shape, then canonical attrs
# ---------------------------------------------------------------------------


def _op_tree(node: Any) -> Any:
    """Return the op-name skeleton of a term in either spelling.

    Kit nodes (``{"op","args","attrs","dim","lit","source"}``) and
    ``term_to_data`` nodes (``{"op","args","attrs"}``) both reduce to
    ``(op, child op-trees)``; leaves reduce to their kind
    (``mvar`` / ``const`` / typed-leaf / ``dag.program`` envelope).
    Attrs are deliberately ignored — see :func:`_canon_attrs`.
    """
    if isinstance(node, dict):
        if "op" in node:
            return (
                node["op"],
                tuple(_op_tree(a) for a in node.get("args") or ()),
            )
        if "mvar" in node:
            return ("mvar",)
        if "const" in node:
            return ("const",)
        if "var" in node or "param" in node:
            return ("leaf",)
        if node.get("__tag__") == "dag.program":
            return ("dag",)
        return ("?",)
    return ("?",)


def _canon_lit(v: Any) -> Any:
    """Canonical literal form shared by both attr spellings."""
    if isinstance(v, dict):
        if "__tuple__" in v:
            return ("seq", tuple(_canon_lit(x) for x in v["__tuple__"]))
        if "__list__" in v:
            return ("seq", tuple(_canon_lit(x) for x in v["__list__"]))
        return (
            "dict",
            tuple(sorted((k, _canon_lit(x)) for k, x in v.items())),
        )
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_canon_lit(x) for x in v))
    return ("lit", v)


def _kit_attrs(attrs: Any) -> dict[str, Any]:
    """Canonicalise one kit attr map to ``{name: ("bind"|"lit", value)}``.

    Handles the observed envelopes: ``null`` → ``{}``; a sole
    ``"fields"`` key unwraps to the per-name map; per-name values may
    be ``{"bind": name}`` / ``{"lit": value}`` tags or raw literals
    (the conv packs spell literals bare).  Two *whole-attrs*
    envelopes sit beside them (engine-eqsat's ``AttrsSpec``):

    * ``{"lit": {record}}`` pins the entire attrs payload to the
      record — per-name literal attrs exactly, the same surface the
      Python flat-kwargs spelling produces under exact-keyset
      matching, so it unwraps to ``{name: _canon_lit(value)}``.
    * ``{"bind": name}`` / ``{"lit": scalar}`` bind or pin the whole
      payload with no per-name structure — the attr *name* is
      unrecoverable and the match is deliberately more permissive
      than the Python per-key spelling, so they canonicalise to
      sentinels the Python side never produces, landing the record
      in the named divergence set rather than silently matching.
    """
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        return {}
    if set(attrs) == {"lit"} and isinstance(attrs["lit"], dict):
        return {k: _canon_lit(v) for k, v in attrs["lit"].items()}
    if set(attrs) in ({"bind"}, {"lit"}):
        tag = next(iter(attrs))
        return {f"__attrs_{tag}__": _canon_lit(attrs[tag])}
    if set(attrs) == {"fields"} and isinstance(attrs["fields"], dict):
        attrs = attrs["fields"]
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, dict) and set(value) == {"bind"}:
            out[key] = ("bind", value["bind"])
        elif isinstance(value, dict) and set(value) == {"lit"}:
            out[key] = _canon_lit(value["lit"])
        else:
            out[key] = _canon_lit(value)
    return out


def _py_attrs(attrs: Any) -> dict[str, Any]:
    """Canonicalise ``term_to_data`` attrs the same way.

    A bare string attr value is an attribute metavariable (bound
    under ``"$attr:"`` at match time); anything else is a literal.
    """
    out: dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        if isinstance(value, str):
            out[key] = ("bind", value)
        else:
            out[key] = _canon_lit(value)
    return out


def _op_nodes(node: Any) -> list[dict]:
    """Yield every op node of a term in either spelling."""
    if isinstance(node, dict):
        if "op" in node:
            yield node
            for arg in node.get("args") or ():
                yield from _op_nodes(arg)
    elif isinstance(node, list):
        for item in node:
            yield from _op_nodes(item)


def _attrs_match(kit_node: dict, py_node: dict) -> bool:
    """Return True when two op nodes' canonical attrs agree."""
    return _kit_attrs(kit_node.get("attrs")) == _py_attrs(
        py_node.get("attrs")
    )


# ---------------------------------------------------------------------------
#  The named divergences — measured, pinned, never silently widened
# ---------------------------------------------------------------------------

#: Non-prim records whose canonical *attrs* diverge from the Python
#: spelling.  ``{"lit": {record}}`` whole-attrs literals unwrap cleanly
#: (per-name literal attrs — the same surface Python's flat kwargs
#: produce under exact-keyset matching); what remains is
#: ``{"bind": name}`` whole-attrs binds — engine-eqsat's
#: ``AttrsSpec::Bind`` binds the *entire* attrs payload to one
#: metavariable, deliberately more permissive than the Python
#: exact-keyset per-key spelling (``builtin.rs`` documents the
#: caveat; the attr name is unrecoverable, so the sentinel never
#: equals a Python spelling).  Empty today: ``qkv_fuse``'s
#: reshape/sdpa nodes were re-spelled to ``attrs_fields`` per-key
#: binds (``shape``/``scale``), closing the last divergence.
_ATTR_DIVERGENT = frozenset()

#: ``(rule name, record field)`` pairs whose content legitimately
#: differs between the Rust port and the Python oracle — pinned so a
#: *new* divergence fails and a *fixed* one forces the allowlist to
#: shrink.  Empty today: the eight ``derivation`` trails resolved as
#: provenance under-recording (see the module docstring — kit ``[]``
#: vs Python's expansion names is consistent, not contradictory).
_FIELD_DIVERGENT = frozenset()

#: Record fields compared against ``law_to_data`` — every field
#: except ``lhs``/``rhs`` (the pattern pair, covered by the op-tree +
#: attrs aspects) and ``cert``/``version``/``kind`` (record-level
#: metadata the codec does not fold into the Rewrite).  ``cond`` and
#: ``derivation`` get canonicalised comparisons in ``_field_diffs`` —
#: everything else is verbatim.
_COMPARED_FIELDS = (
    "law",
    "cond",
    "dspec",
    "tags",
    "derivation",
    "error_bound",
    "bound_norm",
    "serializable",
    "missing_hooks",
)


def _non_prim(records: list[dict]) -> list[dict]:
    """Return the corpus records that are not minted-store projections."""
    return [
        r for r in records if not r.get("name", "").startswith("prim:")
    ]


def _canon_cond(node: Any) -> Any:
    """Canonicalise a cond tree for comparison: flatten same-op nesting.

    ``and``/``or`` are n-ary combinators in both cond evaluators —
    ``("and", a, ("and", b, c))`` and ``("and", a, b, c)`` evaluate
    identically (``eval_cond``'s ``all``/``any`` over children;
    engine-eqsat's ``Cond::And(Vec<Cond>)`` parses recursively).  The
    Rust port re-authored some records' conds with different
    parenthesisation than the Python source, so the field comparison
    flattens nested same-op nodes (order preserved — leaf sequence
    equality is still asserted) before the verbatim check.  Every
    other node passes through untouched, so a real guard delta
    (different op, operand, or order) still fails.
    """
    if not isinstance(node, (list, tuple)) or not node:
        return node
    head = node[0]
    children = [_canon_cond(c) for c in node[1:]]
    if head in ("and", "or"):
        flat: list[Any] = [head]
        for child in children:
            if isinstance(child, list) and child and child[0] == head:
                flat.extend(child[1:])
            else:
                flat.append(child)
        return flat
    return [head, *children]


def _field_diffs(record: dict, py_data: dict) -> set[tuple[str, str]]:
    """Return the ``(name, field)`` divergences for one record."""
    name = record["name"]
    out: set[tuple[str, str]] = set()
    for field in _COMPARED_FIELDS:
        kit_value = record.get(field)
        py_value = py_data.get(field)
        if field == "cond":
            kit_value = _canon_cond(kit_value)
            py_value = _canon_cond(py_value)
        elif field == "derivation":
            # Provenance consistency, not completeness: the kit record
            # may under-record the oracle's derivation trail (the
            # port annotates ``with_derivation`` only where the
            # constructor carried it), but it must never assert a
            # name the Python rule does not.
            if set(kit_value or ()) - set(py_value or ()):
                out.add((name, field))
            continue
        if py_value != kit_value:
            out.add((name, field))
    return out


def _pattern_diverges(record: dict, py_data: dict) -> str | None:
    """Return ``"ops"``/``"attrs"``/``None`` for one record's pattern.

    Op-tree shape is compared first (both sides, both patterns); a
    shape mismatch reports ``"ops"``.  Attrs are then compared
    node-by-node in lockstep — the trees are already known equal, so
    zipping is safe — and any canonical-attr mismatch reports
    ``"attrs"``.
    """
    for side in ("lhs", "rhs"):
        kit_tree = _op_tree(record.get(side))
        py_tree = _op_tree(py_data.get(side))
        if kit_tree != py_tree:
            return "ops"
        kit_nodes = list(_op_nodes(record.get(side)))
        py_nodes = list(_op_nodes(py_data.get(side)))
        if len(kit_nodes) != len(py_nodes):
            return "ops"
        if any(
            not _attrs_match(k, p)
            for k, p in zip(kit_nodes, py_nodes, strict=True)
        ):
            return "attrs"
    return None


# ---------------------------------------------------------------------------
#  The guards
# ---------------------------------------------------------------------------


def test_bank_census_resolves_all_corpus_names() -> None:
    """Every kit record names a shipped bank rule — no silent widening."""
    records = _kit_records()
    bank = _bank()
    unknown = sorted(
        r["name"] for r in records if r.get("name") not in bank
    )
    assert not unknown, (
        "kit-eqsat corpus records with no Python bank rule — the eqsat "
        "wire bank would fire laws the oracle cannot replay: "
        f"{unknown}"
    )


def test_non_prim_op_trees_match() -> None:
    """Every non-prim record's ``lhs``/``rhs`` op skeleton equals the oracle's."""
    from mlopt_core.laws.serialize import law_to_data

    bank = _bank()
    bad: dict[str, str] = {}
    for record in _non_prim(_kit_records()):
        aspect = _pattern_diverges(
            record, law_to_data(bank[record["name"]])
        )
        if aspect == "ops":
            bad[record["name"]] = aspect
    assert not bad, (
        "kit-eqsat records whose op-tree shape diverges from the "
        "Python bank (attrs ignored): "
        f"{sorted(bad)}"
    )


def test_non_prim_attrs_match_except_named() -> None:
    """Canonical attrs agree except the pinned ``_ATTR_DIVERGENT`` set."""
    from mlopt_core.laws.serialize import law_to_data

    bank = _bank()
    divergent: set[str] = set()
    for record in _non_prim(_kit_records()):
        if _pattern_diverges(
            record, law_to_data(bank[record["name"]])
        ) == ("attrs"):
            divergent.add(record["name"])
    assert divergent == _ATTR_DIVERGENT, (
        "attr-parity drift — new: "
        f"{sorted(divergent - _ATTR_DIVERGENT)}, fixed: "
        f"{sorted(_ATTR_DIVERGENT - divergent)}; update "
        "_ATTR_DIVERGENT only with a measured reason"
    )


def test_non_prim_record_fields_match_except_named() -> None:
    """Guard/spec/provenance fields agree except ``_FIELD_DIVERGENT``."""
    from mlopt_core.laws.serialize import law_to_data

    bank = _bank()
    divergent: set[tuple[str, str]] = set()
    for record in _non_prim(_kit_records()):
        divergent |= _field_diffs(
            record, law_to_data(bank[record["name"]])
        )
    assert divergent == _FIELD_DIVERGENT, (
        "law-record field drift — new: "
        f"{sorted(divergent - _FIELD_DIVERGENT)}, fixed: "
        f"{sorted(_FIELD_DIVERGENT - divergent)}; update "
        "_FIELD_DIVERGENT only with a measured reason"
    )


def test_prim_records_match_minted_projection() -> None:
    """Every ``prim:*`` corpus record equals the minted-store projection."""
    from mlopt_core.laws import minted

    projected = minted.prim_kit_records()
    prim_records = {
        r["name"]: r
        for r in _kit_records()
        if r.get("name", "").startswith("prim:")
    }
    assert prim_records, "corpus carries no prim:* records"
    missing = sorted(set(projected) - set(prim_records))
    assert not missing, (
        "minted prim records absent from the kit corpus — re-project "
        f"via scripts/project_prim_folds.py: {missing}"
    )
    drifted = sorted(
        name
        for name, rec in prim_records.items()
        if name in projected and projected[name] != rec
    )
    assert not drifted, (
        "prim corpus records drifted from the minted-store projection "
        f"— re-project via scripts/project_prim_folds.py: {drifted}"
    )
