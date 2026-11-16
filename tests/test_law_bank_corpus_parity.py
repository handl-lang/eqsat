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
  ``lit``/raw → literal, ``fields`` unwrapped, seq spellings unified)
  must match except for the named ``_ATTR_DIVERGENT`` set: the Rust
  port re-authored those patterns' attribute structure (the
  ``{"bind": X}``-as-whole-attrs and ``{"lit": {record}}`` envelopes
  lose/gain attr names the flat-kwargs Python spelling keeps).
* **Record fields** — ``law`` / ``cond`` / ``dspec`` / ``tags`` /
  ``derivation`` / ``error_bound`` / ``bound_norm`` / ``serializable``
  / ``missing_hooks`` compare verbatim against ``law_to_data`` except
  for the named ``_FIELD_DIVERGENT`` set: the measured semantic deltas
  between the Rust port and the oracle (8 ``derivation`` trails —
  each is a port-or-decline item, not a bug this guard can
  adjudicate).  ``cond`` compares after :func:`_canon_cond` flattens
  nested same-op ``and``/``or`` nodes: both evaluators
  (``eval_cond``, engine-eqsat's ``Cond::from_data``) read them as
  n-ary combinators, so ``("and", a, ("and", b, c))`` and
  ``("and", a, b, c)`` are the same guard spelled two ways — the Rust
  port re-authored 19 records' conds with different (but
  leaf-identical) nesting, which is a spelling delta like the attrs
  envelopes, not a semantic one.
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
    (the conv packs spell literals bare).  A whole-attrs
    ``{"bind": …}`` / ``{"lit": …}`` tag (single-attr ops like
    ``reshape``) cannot recover the attr *name* — it canonicalises to
    a sentinel the Python side never produces, landing the record in
    the named divergence set rather than silently matching.
    """
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        return {}
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
#: spelling.  All are Rust-port re-authorings or attrs-envelope
#: spellings the comparator cannot normalise (whole-attrs
#: ``{"bind": …}`` / ``{"lit": {record}}``); the op-tree shape still
#: matches for every one of them.
_ATTR_DIVERGENT = frozenset(
    {
        "gqa_absorb_repeat",
        "parallel_mul_fuse",
        "qkv_fuse",
        "qkv_fuse_asym",
        "sdpa_expand",
        "sdpa_expand_d",
        "sdpa_expand_dc",
        "sdpa_expand_dcg",
        "sdpa_expand_dcgs",
        "sdpa_expand_dcs",
        "sdpa_expand_ds",
        "sdpa_expand_s",
        "swiglu_fuse",
    }
)

#: ``(rule name, record field)`` pairs whose content legitimately
#: differs between the Rust port and the Python oracle — the
#: port-or-decline backlog, pinned so a *new* divergence fails and a
#: *fixed* one forces the allowlist to shrink.
_FIELD_DIVERGENT = frozenset(
    {
        # Derivation trails recorded differently engine-side.
        ("assoc_linear_rev", "derivation"),
        ("assoc_matmul_rev", "derivation"),
        ("factor_mul_add", "derivation"),
        ("mul_square", "derivation"),
        ("right_factor_matmul", "derivation"),
        ("silu_fold", "derivation"),
        ("silu_mul_form", "derivation"),
        ("square_to_pow", "derivation"),
    }
)

#: Record fields compared verbatim against ``law_to_data`` — every
#: field except ``lhs``/``rhs`` (the pattern pair, covered by the
#: op-tree + attrs aspects) and ``cert``/``version``/``kind``
#: (record-level metadata the codec does not fold into the Rewrite).
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
