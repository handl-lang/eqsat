"""Shipped minted objects — stored-composite materialization in core.

The discovery pipeline's stored objects (``kit_discovery.evidence``)
are `Rewrite`-shaped: a record carries the pattern pair, ``cond`` /
``dspec`` / ``derivation`` data, the ``serializable`` / ``missing_hooks``
honesty flags, and — for a synthesized composite — a ``"guard"`` field:
the ``guard_pats`` re-expression maps plus the two parent names from
which ``mlopt_core.meta._compose_guards`` rebuilds the composed
``check`` / ``derive`` hooks at admit time.  That record IS the object —
the hooks are code, but the *spec* of the guard is data.

A minted object earns the library only through the admission gauntlet
(``evidence.run_gauntlet`` — every stage green, ``usable``).  Shipping it
then means materializing the *same measured object* at import: the
records are embedded verbatim below (``_RECORDS``), and
:func:`materialize` replays the store's ``_rebuild_stored`` semantics
over the embedded table — each ``syn_*`` node rebuilds its pattern from
its record and its guard from the named parents, resolved first against
the embedded records (recursively, so a composite-of-composites
rebuilds) and then against the shipped library by name.

Two deliberate differences from the store path, both louder:

* a parent that resolves short of its declared strength — absent from
  both tables, or a record whose ``missing_hooks`` exceed what the
  transport restored — **raises** rather than degrading to the bare
  pattern: a shipped object may never silently lose its guard;
* the leaf parents are the *library objects themselves*
  (``BATCH_NORM_RULES`` / ``CONV_FOLD_RULES`` members), not records —
  a live ``Rewrite`` is its own full-strength declaration, and the
  composed hooks close over exactly the hooks it was minted against.

The shipped composite is therefore the first library member whose
``missing_hooks`` is non-empty — ``("check", "derive")`` — by class, not
by accident: its guard is the transported premise composition, carried
inspectably as ``guard_pats`` + ``parents`` on the object.  The
serialization census (``tests/test_law_serialize.py``) records this
*transported* class separately from the full-data majority.

The records live in the SQLite store ``laws/data/minted.db``, not in
this file: :mod:`mlopt_core.laws.store` loads them at import, and
``tools/build_minted_store.py`` is the write path.  Tier 0 is the
verbatim ``law_json`` rows ``store_object`` wrote for the conv+BN fold's
premise closure of minted intermediates
(``bench/drivers/probes/conv_fold_premise_pay_probe.py`` mint;
``project/retros/minted-composite-default.md`` records the admission
account).  They are frozen artifacts: the composite's guard names them
by minted name, and they are scaffolding — they do not ship as rules.

The second embedded family — :data:`_PRIM_RECORDS` — is a different
minting channel at a different shipping boundary: the
``zoo-prim`` mint's 21 admitted ``prim:synop_*`` folds (graphopt
``bench/drivers/overnight.py --corpus zoo-prim``;
``docs/reports/results_overnight_prim_v2.json``), promoted verbatim
out of the domain-side registry into ``DEFAULT``.  The records are
pattern data only — a ``synop_*`` call is a *name* to core, and the
fold needs nothing more to fire, extract and serialize honestly:
``missing_hooks`` is empty on all 21 (``serializable: true`` —
the plain-record majority, not the transported class).  What does
NOT travel is the denotation: a ``synop_*`` op's meaning is the torch
binding that evaluates the spelled subgraph, and that binding lives
domain-side (``kit_discovery.shipped_primitives.register()`` seats
it on the ambient executor table) — core knows the patterns; sinks
supply the semantics, like every op already works.  The op
*declaration* (:func:`~mlopt_core.ir.op_def`) is torch-free and DOES
travel: materializing the folds registers the ``synop_*`` generators,
making them first-class IR citizens wherever the laws are.

The third embedded family — :data:`_PRIM2_RECORDS` — is the same
channel one loop deeper: the ``--prim-bootstrap`` mint's 2 admitted
``prim:synop2_*`` folds (graphopt ``bench/drivers/overnight.py
--prim-bootstrap``; ``docs/reports/results_overnight_prim_bootstrap.json``).
Each fold's LHS is a *synop-carrying skeleton* — a manual shape
spelled through a shipped ``synop_*`` op — and the record therefore
embeds one more verbatim ``law_to_data`` row the depth-1 family does
not need: the definitional ``synop2_*.expand`` premise itself.  For a
depth-1 fold the premise is just ``fold.rhs -> fold.lhs``; for a
depth-2 fold the mint's premise is ``synop2_*(args) -> xpat`` where
``xpat`` is the skeleton *fully re-expanded to manual ops* — the
anchored cert derivation then completes ``rhs -> lhs`` in two steps:
the premise unfolds the call to its manual denotation and the shipped
depth-1 ``prim:synop_*`` fold re-folds the embedded shape to the
skeleton.  The minted ``alpha``/``alpha_fold`` pair records both
faces; the two minted folds share one expanded ``alpha`` (they are
two distinct placements of ``synop_bb8c54c201`` in the same add/mul
tree — different fold paths, hence different laws).

The fourth embedded family — :data:`_PRIM3_RECORDS` — is the
bootstrap mint at depth 3 under the repriced frontier: the
``--prim-deep --prim-bootstrap --prim-vocab`` run's 34 admitted
``prim:synop3_*`` folds (graphopt ``bench/drivers/overnight.py
--corpus zoo-prim``; ``docs/reports/results_overnight_prim_vocab.json``).
Same record shape as depth-2 — the fold LHS is a skeleton spelled
through shipped ``synop_*``/``synop2_*`` ops (``embeds_shipped``),
``alpha_fold``/``alpha`` record the skeleton and the fully-expanded
manual denotation, and the definitional ``synop3_*.expand`` premise
embeds verbatim (its RHS is the expansion, not the skeleton).  The
anchored ``rhs -> lhs`` cert derivation is now up to three steps —
the premise unfolds the call to manual ops, then the shipped
depth-1/depth-2 bank folds re-fold each embedded shape — and the
records add ``embeds``: the same-run minted synops the denotation
nests (the loop-closure provenance the depth-2 run measured).  The
pool's stratified cap kept the 34 deeper-tier mints beside 48
depth-2 rows that stayed unshipped (already-named denotations and
tier duplicates of the shipped surface).

The fifth family — :data:`_AUTO_RECORDS` — is the **auto-admission
tier**: records :func:`admit_prim_record` wrote into the store's
tier :data:`~mlopt_core.laws.store.AUTO_TIER` after the
discovery-side gauntlet cleared them (``kit_discovery``'s
``bench.referees`` arm — oracle → pay → sweep → cert, all stages
green).  Same record shape as tiers 1-3 — the fold's
``law_to_data`` row plus the mint's own measurements —
distinguished only by *who* promoted them: these rows never passed
through a human's hand-port, so the tier IS the provenance class.
``admit_prim_record`` validates the record's structural invariants
(the fold decodes, names ``prim:<name>``, its RHS is the bare
``<name>(args)`` call, an embedded ``expand`` premise starts at
that call) and refuses a name collision — it does NOT re-run the
gauntlet: verification stays the discovery side's gate, and a
record that reaches this function is asserted to carry its
admission already.  A persisted admission ships on the next import
(``prim_folds`` → ``PRIM_FOLDS`` → ``ALL_RULES``/``DEFAULT``, the
same path the hand-promoted tiers take); within the admitting
process the returned ``(fold, expand)`` pair materializes
immediately for explicit composition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlopt_core import meta
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op, Param, TensorType, op_def
from mlopt_core.laws import store as _store
from mlopt_core.laws import tags
from mlopt_core.laws.serialize import object_from_data
from mlopt_core.rulecache import _dec_binding
from mlopt_core.typing import (
    _SHAPE_RULES,
    register_shape_rule,
    shape_of,
)

#: The shipped records, loaded from the SQLite store
#: (``laws/data/minted.db``).  Tier 0 is the promoted composite
#: scaffolding; tiers 1-3 are the depth-1/2/3 prim tables.  Editing the
#: bank is a store write (``tools/build_minted_store.py``), never a
#: source edit — the payload used to live here as ~40 000 lines of
#: JSON literals.
_STORE = _store.load()

__all__ = [
    "SYNOP2_SHIPPED",
    "SYNOP3_SHIPPED",
    "SYNOP_SHIPPED",
    "admit_prim_record",
    "auto_records",
    "expand_name",
    "kit_law_record",
    "materialize",
    "prim2_folds",
    "prim2_premises",
    "prim2_records",
    "prim3_folds",
    "prim3_premises",
    "prim3_records",
    "prim_fold",
    "prim_folds",
    "prim_kit_records",
    "prim_premises",
    "prim_records",
    "synop_name",
    "synop_shipped",
]

#: The verbatim ``law_json`` records of the minted intermediates the
#: shipped composites descend from, keyed by minted (``syn_*``) name.
#: Leaf parents are NOT here — they resolve against the shipped library
#: by name at materialize time.
_RECORDS: dict[str, Any] = _STORE[0]


# ---------------------------------------------------------------------------
#  The prim family — ``prim:synop_*`` folds, shipped laws-only
# ---------------------------------------------------------------------------

#: The verbatim records of the 21 admitted ``prim:synop_*`` folds —
#: the ``zoo-prim`` mint's shipped set (graphopt
#: ``bench/drivers/overnight.py --corpus zoo-prim``, protocol
#: ``seed=0``/``xbind=2``: 24 minted, 21 admitted by the full
#: gauntlet, all ``machine-original`` —
#: ``docs/reports/results_overnight_prim_v2.json``), promoted
#: verbatim out of the domain-side registry
#: (``kit_discovery.shipped_primitives``) into the core bank.
#: Keyed by the minted ``synop_*`` op name — ``sha256(alpha-normal
#: pattern)[:10]``, a content hash, not a coined label.  Each record
#: carries the fold's ``law_to_data`` row (``fold``) plus the mint's
#: own measurements: ``alpha`` (the content the op name hashes),
#: ``mods`` (the ≥2 bound-module gate), ``sites`` / ``equal_sites``
#: (the fp64 denotation check the arm ran — equal everywhere) and
#: ``handler`` (the grammar minter's verdict — ``"minted"`` where a
#: handler body verified, ``"refused"`` where the primitive ships on
#: its definitional denotation alone).  Frozen artifacts: a
#: deterministic re-mint of the pool reproduces them
#: byte-identically.
_PRIM_RECORDS: dict[str, Any] = _STORE[1]

#: The ``--prim-bootstrap`` mint's depth-2 records — the 2 admitted
#: ``prim:synop2_*`` folds, verbatim (graphopt
#: ``docs/reports/results_overnight_prim_bootstrap.json``'s
#: ``prim_ops`` rows plus their ``law_to_data`` rows).  Same shape as
#: :data:`_PRIM_RECORDS` plus three depth-2 fields: ``alpha_fold``
#: (the synop-carrying skeleton's alpha-normal key — the fold LHS's
#: spelling, distinct from ``alpha`` which is the fully-expanded
#: manual denotation), ``embeds_shipped`` (the shipped depth-1 ops the
#: skeleton composes over) / ``embeds`` (the pool's loop-closure
#: provenance), and ``expand`` — the definitional premise's own
#: record, embedded because ``expand.rhs`` is the *expanded* manual
#: denotation, not the fold's LHS skeleton.
_PRIM2_RECORDS: dict[str, Any] = _STORE[2]

#: The ``--prim-vocab`` run's depth-3 records — the 34 admitted
#: ``prim:synop3_*`` folds, verbatim (graphopt
#: ``docs/reports/results_overnight_prim_vocab.json``'s ``prim_ops``
#: rows plus their ``law_to_data`` rows).  Same shape as
#: :data:`_PRIM2_RECORDS` — ``alpha_fold`` is the synop-carrying
#: skeleton (spelled through shipped ``synop_*``/``synop2_*`` ops,
#: named by ``embeds_shipped``), ``alpha`` the fully-expanded manual
#: denotation, and ``expand`` the embedded definitional premise whose
#: RHS is that expansion.  ``embeds`` is the depth-3 field the
#: depth-2 records did not need as a list (both carried it
#: incidentally): the same-run minted synops the expanded denotation
#: nests — the loop-closure provenance of the vocab-priced frontier.
_PRIM3_RECORDS: dict[str, Any] = _STORE[3]

#: The auto-admission tier — records
#: :func:`admit_prim_record` promoted into the store after the
#: discovery-side gauntlet cleared them (store tier
#: ``AUTO_TIER``).  Same record shape as the hand-promoted prim
#: tiers; the live dict also picks up admissions made earlier in
#: this process (``admit_prim_record`` on the shipped store mutates
#: it, so ``prim_fold``/``prim_folds`` see them immediately — the
#: *rule lists* built at import do not).
_AUTO_RECORDS: dict[str, Any] = _STORE.get(4, {})

#: The minted op names the prim family ships — all three depths:
#: ``synop_<sha256[:10]>`` (the zoo-prim mint), the depth-2
#: ``synop2_<sha256[:10]>`` (the vocabulary-bootstrap mint) and the
#: depth-3 ``synop3_<sha256[:10]>`` (the vocab-priced bootstrap).
#: A ``synop_*`` op name means "a shipped machine-minted primitive"
#: wherever this set is consulted — expansion, denotation lookup,
#: provenance.
SYNOP_SHIPPED: frozenset[str] = (
    frozenset(_PRIM_RECORDS)
    | frozenset(_PRIM2_RECORDS)
    | frozenset(_PRIM3_RECORDS)
    | frozenset(_AUTO_RECORDS)
)


def _auto_depth(prefix: str) -> frozenset[str]:
    """Auto-admitted record names carrying a ``synopN_`` depth prefix."""
    return frozenset(n for n in _AUTO_RECORDS if n.startswith(prefix))


#: The live shipped set — :data:`SYNOP_SHIPPED` plus any records
#: :func:`admit_prim_record` added to the shipped store *this
#: process* (the frozenset above is the import-time snapshot; this
#: is the same set re-read live).
def synop_shipped() -> frozenset[str]:
    """Return every shipped ``*synop*`` op name, live."""
    return SYNOP_SHIPPED | frozenset(_AUTO_RECORDS)


#: The depth-2 subset — the bootstrap mint's ``synop2_*`` ops plus
#: any auto-admitted record named ``synop2_*``.
SYNOP2_SHIPPED: frozenset[str] = frozenset(
    _PRIM2_RECORDS
) | _auto_depth("synop2_")

#: The depth-3 subset — the vocab-priced bootstrap's ``synop3_*``
#: ops (skeletons spelled through both shipped tiers) plus any
#: auto-admitted record named ``synop3_*``.
SYNOP3_SHIPPED: frozenset[str] = frozenset(
    _PRIM3_RECORDS
) | _auto_depth("synop3_")

#: Law text for the fold rules — a structural fold collapsing a
#: consumer pattern into one kernel, same class as ``conv_bn_fold``.
_PRIM_FOLD_LAW = (
    "machine-minted primitive fold — the manual subgraph collapses "
    "to the {synop} primitive (zoo-prim mint; admitted by the full "
    "gauntlet, machine-original; denotation definitional — the op's "
    "torch binding IS the spelled subgraph)"
)

#: Law text for the definitional premise — the cert leg's anchor.
_PRIM_EXPAND_LAW = (
    "{synop} unfolds to its mined denotation — the definitional "
    "premise the cert leg's anchored derivation unions in when the "
    "bare bank cannot derive the fold"
)


def synop_name(fold_name: str) -> str:
    """Return the op a ``prim:synop_*`` fold name mints (identity otherwise)."""
    return fold_name[5:] if fold_name.startswith("prim:") else fold_name


def expand_name(synop: str) -> str:
    """Return the definitional-premise law name for a ``synop_*`` op."""
    return f"{synop_name(synop)}.expand"


def prim_records() -> dict[str, Any]:
    """Return the embedded prim records — the frozen mint artifacts.

    All four families, merged: the 21 depth-1 ``synop_*`` records,
    the 2 depth-2 ``synop2_*`` records, the 34 depth-3 ``synop3_*``
    records (the deeper tiers additionally carry ``alpha_fold`` /
    ``embeds_shipped`` / ``embeds`` / ``expand``) and the
    auto-admission tier's records.
    """
    return {
        **_PRIM_RECORDS,
        **_PRIM2_RECORDS,
        **_PRIM3_RECORDS,
        **_AUTO_RECORDS,
    }


def auto_records() -> dict[str, Any]:
    """Return the auto-admission tier's records — a live snapshot.

    Keys are the minted op names (``synop_*``/``synopN_*``); values
    are the same record shape :func:`prim_records` carries.  The
    dict reflects admissions made earlier in this process, so a
    caller polling it sees :func:`admit_prim_record`'s writes.
    """
    return dict(_AUTO_RECORDS)


# ---------------------------------------------------------------------------
#  kit-eqsat projection — minted record → LAW_FORMAT = 2 kit data
# ---------------------------------------------------------------------------
#
#  ``minted.db`` stays the ledger; kit-eqsat's ``prim_folds`` pack is
#  the *projected* corpus (ADR-0177 §1 — laws are kit data).  The
#  store's ``fold`` sub-records are already ``LAW_FORMAT = 2`` shaped
#  except for one spelling difference in ``attrs``: the Python-side
#  encoding is a flat ``{name: value}`` map (``{}`` when empty, a
#  string value a metavar bind, anything else a literal), while the
#  kit wire spells it ``{"fields": {name: {"bind"|"lit": value}}}``
#  (``null`` for no attrs).  :func:`kit_law_record` is that
#  projection; :func:`prim_kit_records` applies it to every shipped
#  prim fold.  The drift test in
#  ``tests/test_eqsat_laws_corpus.py`` pins the projection against
#  the committed ``eqsat_laws.json`` snapshot record-for-record, so
#  a live :func:`admit_prim_record` that reaches the store without a
#  kit re-projection fails the suite instead of silently widening
#  the Python bank past the eqsat wire corpus.


def _kit_attr(v: Any) -> dict[str, Any]:
    """Tag one attr value for the kit wire: str → ``bind``, else ``lit``.

    Latent edge (audit-8): every shipped bind spelling is a metavar
    name (``aNNN`` or the uppercase shape names), so ``str`` →
    ``bind`` is correct today — but a future string *literal* attr
    (e.g. ``mode="relu"``) would misproject as a bind. If literal
    string attrs ever ship, this needs a metavar-shape discriminator.
    """
    if isinstance(v, str):
        return {"bind": v}
    return {"lit": v}


def _kit_attrs(attrs: dict[str, Any]) -> dict[str, Any] | None:
    """Encode a node's attrs in the kit ``fields`` envelope."""
    if not attrs:
        return None
    return {"fields": {k: _kit_attr(v) for k, v in attrs.items()}}


def _kit_node(node: Any) -> Any:
    """Recursively project ``"attrs"`` maps inside *node*."""
    if isinstance(node, dict):
        return {
            k: (
                _kit_attrs(v)
                if k == "attrs" and isinstance(v, dict)
                else _kit_node(v)
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_kit_node(v) for v in node]
    return node


def kit_law_record(fold: dict[str, Any]) -> dict[str, Any]:
    """Project one minted ``fold`` record to its kit-eqsat spelling.

    The record is already ``LAW_FORMAT = 2`` shaped — ``version`` /
    ``kind`` / ``name`` / the pattern pair / the guard fields pass
    through untouched; only the ``attrs`` spellings inside the term
    trees are rewritten (see the section comment above).
    """
    return _kit_node(fold)


def prim_kit_records() -> dict[str, dict[str, Any]]:
    """Return ``{"prim:<name>": kit record}`` for every shipped prim fold.

    Covers all four record families :func:`prim_records` merges —
    tiers 1-3 plus the live auto-admission tier — in
    :func:`prim_records` order, which is the order kit-eqsat's
    ``prim_folds`` pack ships.  The minted store is the ledger; this
    dict is the interchange projection (ADR-0177 §1).
    """
    return {
        f"prim:{name}": kit_law_record(rec["fold"])
        for name, rec in prim_records().items()
    }


def prim2_records() -> dict[str, Any]:
    """Return the embedded depth-2 records — the bootstrap mint's."""
    return _PRIM2_RECORDS


def prim3_records() -> dict[str, Any]:
    """Return the embedded depth-3 records — the vocab-priced mint's."""
    return _PRIM3_RECORDS


_PRIM_MATERIALIZED: dict[str, tuple[Rewrite, Rewrite]] = {}


def prim_fold(synop: str) -> tuple[Rewrite, Rewrite]:
    """Build ``(fold, expand)`` for one prim record — memoized.

    The fold rebuilds through the shipped record codec
    (:func:`~mlopt_core.laws.serialize.object_from_data`) — the same
    object the mint admitted — then gets the shipped integration
    fields (``law`` text, ``FUSION`` tag) and its premise attached,
    exactly as ``pipeline.Proposal.as_rule`` minted it; the op's
    :func:`~mlopt_core.ir.op_def` declaration registers with the fold
    (the generator entry is torch-free — the *denotation* is the
    domain-side piece).  Interned: the same objects come back on
    every call.

    *synop* resolves across all four tables — a ``synopN_*`` name
    materializes its depth-N record, an auto-admitted name its
    tier-4 record.  The deeper-tier records embed
    the definitional premise verbatim (``rec["expand"]``): its RHS is
    the skeleton *re-expanded to manual ops*, NOT the fold's LHS —
    the cert leg's anchored derivation then re-folds through the
    shipped lower-tier ``prim:synop*`` folds (an N-step anchored
    derivation; the lower-tier folds are bank members, not further
    premises).  A record whose embedded premise does not start at the
    fold's own call raises rather than materializing a mismatched
    axiom.
    """
    if synop in _PRIM_MATERIALIZED:
        return _PRIM_MATERIALIZED[synop]
    if synop in _PRIM_RECORDS:
        rec = _PRIM_RECORDS[synop]
    elif synop in _PRIM2_RECORDS:
        rec = _PRIM2_RECORDS[synop]
    elif synop in _PRIM3_RECORDS:
        rec = _PRIM3_RECORDS[synop]
    elif synop in _AUTO_RECORDS:
        rec = _AUTO_RECORDS[synop]
    else:
        raise KeyError(synop)
    fold = object_from_data(rec["fold"])
    if "expand" in rec:
        # depth-2+: the premise's RHS is the full manual denotation —
        # verbatim from the mint, not rebuildable as ``rhs -> lhs``.
        ex = object_from_data(rec["expand"])
        if ex.lhs != fold.rhs:
            raise ValueError(
                f"prim record {synop!r} embeds an expand premise "
                f"whose LHS is not the fold's own call — a "
                f"mismatched axiom would anchor nothing"
            )
        denotation = ex.rhs
    else:
        denotation = fold.lhs
    expand = Rewrite(
        expand_name(synop),
        fold.rhs,
        denotation,
        law=_PRIM_EXPAND_LAW.format(synop=synop),
        tags=frozenset({tags.SIMPLIFICATION}),
    )
    out = Rewrite(
        name=fold.name,
        lhs=fold.lhs,
        rhs=fold.rhs,
        law=_PRIM_FOLD_LAW.format(synop=synop),
        tags=frozenset({tags.FUSION}),
        error_bound=fold.error_bound,
        bound_norm=fold.bound_norm,
        derivation=fold.derivation,
        cond=fold.cond,
        dspec=fold.dspec,
    )
    object.__setattr__(out, "premises", (expand,))
    _PRIM_MATERIALIZED[synop] = (out, expand)
    _seat_synop_shape_rule(synop, out, expand)
    op_def(
        synop,
        len(fold.rhs.args),
        1,
        law=(
            f"synthesized primitive — denotation is the mined manual "
            f"subgraph {rec['alpha']}"
        ),
    )
    return _PRIM_MATERIALIZED[synop]


def prim_folds() -> tuple[Rewrite, ...]:
    """Return the shipped fold laws — ``pattern -> synop*_*(args)``.

    All four families: the 21 ``prim:synop_*`` folds, the 2 depth-2
    ``prim:synop2_*`` folds, the 34 depth-3 ``prim:synop3_*`` folds
    (skeleton -> ``synopN_*(args)``) and the auto-admission tier's
    folds (live — an :func:`admit_prim_record` earlier in this
    process is included).
    """
    return tuple(
        prim_fold(n)[0]
        for n in (
            *_PRIM_RECORDS,
            *_PRIM2_RECORDS,
            *_PRIM3_RECORDS,
            *_AUTO_RECORDS,
        )
    )


def _depth_names(prefix: str) -> tuple[str, ...]:
    """Record names at *prefix* depth across every shipped table."""
    return tuple(
        n
        for table in (
            _PRIM_RECORDS,
            _PRIM2_RECORDS,
            _PRIM3_RECORDS,
            _AUTO_RECORDS,
        )
        for n in table
        if n.startswith(prefix)
    )


def prim2_folds() -> tuple[Rewrite, ...]:
    """Return the depth-2 fold laws — the ``synop2_*`` slice."""
    return tuple(prim_fold(n)[0] for n in _depth_names("synop2_"))


def prim3_folds() -> tuple[Rewrite, ...]:
    """Return the depth-3 fold laws — the ``synop3_*`` slice."""
    return tuple(prim_fold(n)[0] for n in _depth_names("synop3_"))


def prim_premises() -> tuple[Rewrite, ...]:
    """Return the definitional ``synop*_*.expand`` premises — axioms.

    NOT a saturation set: expanding ``synop -> pattern`` inside a
    search would re-add the unfolded spelling the fold collapses.
    These are the anchored-derivation axioms a cert/verify leg unions
    into its axiom set; each fold also carries its own premise as
    ``rule.premises``.  All four families — a deeper-tier premise's
    RHS is the skeleton's *expanded* manual denotation (the embedded
    ``expand`` record), so its anchored derivation still closes
    through the shipped lower-tier folds.
    """
    return tuple(
        prim_fold(n)[1]
        for n in (
            *_PRIM_RECORDS,
            *_PRIM2_RECORDS,
            *_PRIM3_RECORDS,
            *_AUTO_RECORDS,
        )
    )


def prim2_premises() -> tuple[Rewrite, ...]:
    """Return the depth-2 ``synop2_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _depth_names("synop2_"))


def prim3_premises() -> tuple[Rewrite, ...]:
    """Return the depth-3 ``synop3_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _depth_names("synop3_"))


# ---------------------------------------------------------------------------
#  shape rules — the denotation IS the op's shape spec
# ---------------------------------------------------------------------------
#
# A ``prim:synop_*`` fold is a proved equality, so the op's output shape
# is exactly its ``.expand`` denotation's — but ``typing._SHAPE_RULES``
# has no entry for minted names and the substrate default reports
# ``None``.  Since the honest-unknown change an unknown operand makes
# every broadcast consumer ``None`` too, so ONE shape-blind synop
# poisons every shape-guarded law downstream: the r4 AUTO_TIER admits
# (``prim:synop_0b304215b3`` / ``prim:synop_f0576b5150``) started
# folding ``detach_(contiguous …)`` / ``transpose(reshape …)`` sites
# and the ``zoo_time_cond_conv`` ``mul_unsq_pad_*`` strips declined on
# the fabricated-``None`` chain (graphopt a249ef2 seated this same
# rule consumer-side in ``kit_torch.torch_bindings`` — this is the
# substrate home; seating at materialize time also covers records
# :func:`admit_prim_record` promotes mid-process, which an
# import-time consumer pass cannot see).
#
# The rule substitutes the call's leaf/attr metavars into the
# denotation and reads ``shape_of`` — pure term data, no backend.
# Anything unresolvable declines to ``None`` like any other unknown
# op; a domain that registers its own rule for a synop name keeps it
# (first seat wins — the substrate never clobbers a domain override).

#: Sentinel for a metavar the instantiated call does not bind — an
#: honest decline, never a guess.
_SYNOP_SHAPE_UNBOUND = object()


def _synop_shape_rule(denotation: Any, call: Op) -> Any:
    """Build the ``(op, shapes) -> shape`` rule for one synop record.

    *denotation* is the ``.expand`` premise's RHS — the spelled
    subgraph the op denotes, with metavar leaves; *call* is the fold's
    RHS — the ``<name>(args)`` pattern whose leaf positions and attr
    bindings map metavars onto the instantiated op's operands.
    """
    # leaf metavar -> position in the call's arg list; attr
    # metavar -> the op-attr name carrying it (call.attrs maps
    # op-attr name -> metavar name).
    leaf_pos = {
        m: i for i, m in enumerate(call.args) if isinstance(m, str)
    }
    attr_key = {
        m: k for k, m in call.attrs.items() if isinstance(m, str)
    }

    def _rule(op: Op, shapes: list, /) -> Any:
        leaf_map: dict[str, Any] = {}
        for m, i in leaf_pos.items():
            if i >= len(op.args):
                return None
            s = shapes[i] if i < len(shapes) else None
            # A known operand shape rides a cheap Param placeholder
            # — ``shape_of`` answers it without re-walking the arg
            # term; anything else falls back to the real term.
            leaf_map[m] = (
                Param(m, TensorType(s))
                if isinstance(s, tuple)
                else op.args[i]
            )

        def _attr_val(v: Any) -> Any:
            if isinstance(v, str):
                key = attr_key.get(v)
                if key is None:
                    return v  # literal string attr, not a metavar
                return op.attrs.get(key, _SYNOP_SHAPE_UNBOUND)
            if isinstance(v, (tuple, list)):
                return type(v)(_attr_val(x) for x in v)
            return v

        def _subst(node: Any) -> Any:
            if isinstance(node, str):
                return leaf_map.get(node, _SYNOP_SHAPE_UNBOUND)
            if isinstance(node, Op):
                args = tuple(_subst(a) for a in node.args)
                attrs = {k: _attr_val(v) for k, v in node.attrs.items()}
                if any(a is _SYNOP_SHAPE_UNBOUND for a in args) or any(
                    v is _SYNOP_SHAPE_UNBOUND for v in attrs.values()
                ):
                    return _SYNOP_SHAPE_UNBOUND
                return Op.make(node.op, *args, **attrs)
            if isinstance(node, (tuple, list)):
                return type(node)(_subst(x) for x in node)
            return node

        try:
            term = _subst(denotation)
            if term is _SYNOP_SHAPE_UNBOUND or term is None:
                return None
            return shape_of(term)
        except Exception:
            return None  # honest decline, never a guess

    return _rule


def _seat_synop_shape_rule(
    name: str, fold: Rewrite, expand: Rewrite
) -> None:
    """Register the denotation-evaluating shape rule for *name*.

    Called at materialize time — :func:`prim_fold` for the shipped
    tiers, :func:`_materialize_rec` / :func:`prim_fold` for a live
    :func:`admit_prim_record` — so every record that becomes a real
    ``synop*`` op gets its shape semantics with it.  First seat wins:
    a domain rule already registered under the name is kept.
    """
    if name in _SHAPE_RULES:
        return
    register_shape_rule(name, _synop_shape_rule(expand.rhs, fold.rhs))


# ---------------------------------------------------------------------------
#  auto-admission — the gauntlet-cleared record becomes a shipped law
# ---------------------------------------------------------------------------


def _canonical(record: Any) -> str:
    """Return the store's canonical JSON text — the idempotence comparator."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _prim_tables() -> tuple[dict[str, Any], ...]:
    """Every record table a prim name could already occupy."""
    return (
        _PRIM_RECORDS,
        _PRIM2_RECORDS,
        _PRIM3_RECORDS,
        _AUTO_RECORDS,
    )


def admit_prim_record(
    name: str,
    record: dict[str, Any],
    *,
    db: Path | str | None = None,
) -> tuple[Rewrite, Rewrite]:
    """Admit a gauntlet-cleared prim record into the shipped store.

    *name* is the minted op name (``synop_*``/``synopN_*`` — the
    record's key, NOT the ``prim:``-prefixed fold name); *record* is
    the ``prims.yaml``-shaped row the mint emitted — ``fold`` (the
    fold's ``law_to_data``), the mint's measurements (``alpha`` /
    ``mods`` / ``sites`` / ``equal_sites`` / ``handler``) and, for a
    depth-N mint, ``alpha_fold`` / ``embeds_shipped`` / ``embeds`` /
    ``expand`` verbatim.

    The contract is honest about where verification lives: the
    discovery-side gauntlet (``bench.referees.judge`` — oracle → pay
    → sweep → cert, every stage green) is the gate, and this
    function does NOT re-run it — a record reaching here is asserted
    to carry its admission.  What it DOES enforce is the shipped
    object's own invariants, loudly:

    * the record's ``fold`` row decodes through
      :func:`~mlopt_core.laws.serialize.object_from_data`;
    * the decoded fold is named ``prim:<name>`` and its RHS is the
      bare ``<name>(args)`` call — the shape every shipped prim fold
      has;
    * an embedded ``expand`` premise decodes and its LHS is the
      fold's own call (a mismatched axiom would anchor nothing —
      the same check :func:`prim_fold` makes at materialize);
    * *name* is not already shipped: an identical record is an
      idempotent no-op returning the materialized pair; a
      *different* record under a shipped name raises — silent
      shadowing is a smuggle, not an admit.

    The write is :func:`mlopt_core.laws.store.emit` into tier
    ``AUTO_TIER`` of *db* (default: the shipped store this module
    loaded).  On the shipped store the record also joins
    :data:`_AUTO_RECORDS`, so :func:`prim_fold` /
    :func:`prim_folds` / :func:`prim_premises` see it immediately;
    the ``PRIM_FOLDS``/``ALL_RULES``/``DEFAULT`` lists built at
    import pick it up on the NEXT import — a persisted admission,
    not a retroactive one.  For any other *db* the pair is
    materialized and returned without touching the shipped tables.

    Returns the ``(fold, expand)`` pair — the fold carries the
    ``FUSION`` tag, the fold law text and ``premises=(expand,)``,
    exactly as the hand-promoted tiers materialize, and the op's
    :func:`~mlopt_core.ir.op_def` declaration registers with it.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"prim record name must be a nonempty str, got {name!r}"
        )
    if not isinstance(record, dict) or not isinstance(
        record.get("fold"), dict
    ):
        raise ValueError(
            f"prim record {name!r} must carry its fold's "
            "law_to_data row under 'fold'"
        )
    try:
        fold = object_from_data(record["fold"])
    except Exception as exc:
        raise ValueError(
            f"prim record {name!r} fold does not decode: {exc}"
        ) from exc
    if fold.name != f"prim:{name}":
        raise ValueError(
            f"prim record {name!r} fold is named {fold.name!r} — "
            f"the shipped convention is 'prim:{name}'"
        )
    if not (
        isinstance(fold.rhs, Op)
        and fold.rhs.op == name
        and all(isinstance(a, str) for a in fold.rhs.args)
    ):
        raise ValueError(
            f"prim record {name!r} fold RHS is not the bare "
            f"{name}(args) call over metavar leaves — the op a "
            "search collapses to"
        )
    if "expand" in record:
        try:
            ex = object_from_data(record["expand"])
        except Exception as exc:
            raise ValueError(
                f"prim record {name!r} expand premise does not "
                f"decode: {exc}"
            ) from exc
        if ex.lhs != fold.rhs:
            raise ValueError(
                f"prim record {name!r} embeds an expand premise "
                f"whose LHS is not the fold's own call — a "
                f"mismatched axiom would anchor nothing"
            )
    want = _canonical(record)
    for table in _prim_tables():
        if name in table:
            if _canonical(table[name]) == want:
                # idempotent re-admission — the same measured object
                return prim_fold(name)
            raise ValueError(
                f"prim name {name!r} is already shipped with a "
                "different record — refusing to shadow it"
            )
    # The target store itself may hold rows the in-memory tables do
    # not: a scratch db, or a tier-4 row another process persisted
    # since this module's import.  ``emit`` upserts by ``(tier,
    # name)`` — without this check a differing payload would be
    # silently overwritten, which is a smuggle, not an admit.
    try:
        rows = _store.query(
            "SELECT payload FROM records WHERE name = ?",
            (name,),
            path=db,
        )
    except Exception:
        rows = []  # no store yet — emit creates it
    identical = False
    for row in rows:
        if row["payload"] == want:
            identical = True
        else:
            raise ValueError(
                f"prim name {name!r} is already recorded in the "
                "target store with a different record — refusing "
                "to shadow it"
            )
    shipped = (
        db is None or Path(db).resolve() == _store.DB_PATH.resolve()
    )
    if identical:
        # the same measured object already persisted — idempotent.
        # On the shipped store, self-heal the live table (another
        # process may have written the row since import).
        if shipped:
            _AUTO_RECORDS[name] = record
            return prim_fold(name)
        return _materialize_rec(name, record)
    _store.emit({_store.AUTO_TIER: {name: record}}, db)
    if shipped:
        # the shipped store this module read at import — the live
        # tables see the admission now (the import-time rule lists
        # do not; they rebuild next import).
        _AUTO_RECORDS[name] = record
        return prim_fold(name)
    # a scratch/alternate store — the pair materializes for the
    # caller without the shipped tables claiming the name.
    return _materialize_rec(name, record)


def _materialize_rec(name: str, rec: dict[str, Any]):
    """Materialize ``(fold, expand)`` from a record dict not in a table.

    The record-local twin of :func:`prim_fold`'s body — used when an
    admission targeted a non-shipped store, so the returned pair is
    real without the shipped tables claiming the name.
    """
    fold = object_from_data(rec["fold"])
    if "expand" in rec:
        denotation = object_from_data(rec["expand"]).rhs
    else:
        denotation = fold.lhs
    expand = Rewrite(
        expand_name(name),
        fold.rhs,
        denotation,
        law=_PRIM_EXPAND_LAW.format(synop=name),
        tags=frozenset({tags.SIMPLIFICATION}),
    )
    out = Rewrite(
        name=fold.name,
        lhs=fold.lhs,
        rhs=fold.rhs,
        law=_PRIM_FOLD_LAW.format(synop=name),
        tags=frozenset({tags.FUSION}),
        error_bound=fold.error_bound,
        bound_norm=fold.bound_norm,
        derivation=fold.derivation,
        cond=fold.cond,
        dspec=fold.dspec,
    )
    object.__setattr__(out, "premises", (expand,))
    op_def(
        name,
        len(fold.rhs.args),
        1,
        law=(
            "synthesized primitive — denotation is the mined manual "
            f"subgraph {rec.get('alpha', '')}"
        ),
    )
    _seat_synop_shape_rule(name, out, expand)
    return out, expand


def _resolve(name: str, lib: dict[str, Rewrite], seen: frozenset[str]):
    """Rebuild *name*'s Rewrite; return ``(rule, transported)``.

    *transported* names the hooks the ``"guard"`` spec restored —
    ``()`` for a library-resolved leaf (a live object is its own
    full-strength declaration) or an unguarded record.  Raises
    ``ValueError`` on any shortfall the store path would degrade
    through: shipping a weakened composite is a smuggle, not a
    partial admit.
    """
    rec = _RECORDS.get(name)
    if rec is None:
        try:
            return lib[name], ()
        except KeyError:
            raise ValueError(
                f"minted object {name!r} names a parent that is "
                f"neither an embedded record nor a library rule"
            ) from None
    if name in seen:
        raise ValueError(f"cyclic guard spec at {name!r}")
    rule = object_from_data(rec)
    guard = rec.get("guard")
    if not isinstance(guard, dict):
        return rule, ()
    parents = tuple(guard.get("parents") or ())
    if len(parents) != 2:
        raise ValueError(
            f"minted object {name!r} carries a guard spec with "
            f"{len(parents)} parents — composition is binary"
        )
    try:
        pats = (
            _dec_binding(guard["pat1"]),
            _dec_binding(guard["pat2"]),
        )
    except Exception as exc:
        raise ValueError(
            f"minted object {name!r} carries an undecodable guard spec"
        ) from exc
    resolved = [
        _resolve_parent(pname, lib, seen | {name}) for pname in parents
    ]
    return _compose_record(name, rule, resolved, parents, pats)


def _resolve_parent(
    pname: str, lib: dict[str, Rewrite], seen: frozenset[str]
) -> Rewrite:
    """Resolve one guard parent — record first (recursive), then lib.

    A record-resolved parent must rebuild at *full declared strength*
    (``missing_hooks`` minus transported empty) — the same bar
    ``evidence._rebuild_stored`` holds, raised here into an error
    rather than a silent degrade.
    """
    prule, ptrans = _resolve(pname, lib, seen)
    prec = _RECORDS.get(pname)
    if prec is not None and set(prec.get("missing_hooks") or ()) - set(
        ptrans
    ):
        raise ValueError(
            f"minted parent {pname!r} rebuilt weaker than it "
            f"declared (missing "
            f"{sorted(set(prec['missing_hooks']) - set(ptrans))}) "
            f"— composing over it would smuggle a weakened guard"
        )
    return prule


def _compose_record(
    name: str,
    rule: Rewrite,
    resolved: list[Rewrite],
    parents: tuple[str, ...],
    pats: tuple[Any, Any],
):
    """Re-run ``_compose_guards`` over resolved parents; rewrap *rule*.

    Returns ``(out, transported)`` — *transported* names the hooks the
    guard spec restored.  Raises ``ValueError`` when the spec yields no
    hooks (a guard-less rebuild is not the declared object).
    """
    check, drv = meta._compose_guards(
        resolved[0], resolved[1], pats[0], pats[1]
    )
    if check is None and drv is None:
        raise ValueError(
            f"minted object {name!r} declares a guard spec but "
            f"transports no hooks"
        )
    # Same policy as synthesize_rules.offer / rulecache._dec_rule: the
    # composite derive attaches only where the RHS carries the "@i:"
    # placeholders it refills.
    if not (
        drv is not None and meta._rhs_derive_placeholders(rule.rhs)
    ):
        drv = None
    out = Rewrite(
        name=rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        law=rule.law,
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        derivation=rule.derivation,
        cond=rule.cond,
        dspec=rule.dspec,
        check=check,
        derive=drv,
    )
    object.__setattr__(out, "parents", parents)
    object.__setattr__(out, "guard_pats", pats)
    meta.SYNTH_PARENTS[out.name] = (parents[0], parents[1])
    transported = tuple(
        h
        for h, f in (("check", check), ("derive", drv))
        if f is not None
    )
    return out, transported


def materialize(
    name: str,
    lib: dict[str, Rewrite],
    *,
    ship_as: str | None = None,
    tags: Any = None,
    law: str | None = None,
) -> Rewrite:
    """Materialize an embedded minted record as a shipped ``Rewrite``.

    *name* is the minted record's own (``syn_*``) name; *lib* is the
    name→rule table the leaf parents resolve against (the shipped
    library families — e.g. ``BATCH_NORM_RULES`` + ``CONV_FOLD_RULES``).
    *ship_as* renames the root object for the library (promotion
    practice: minted ``syn_*`` names ship under their law name, as the
    promoted-laws table did); *tags* / *law* override the record's
    fields for the shipped object — integration decisions, recorded
    here rather than smuggled into the verbatim record.  The
    intermediate nodes keep their minted names: they are scaffolding,
    not shipped rules.

    The returned object carries ``parents`` / ``guard_pats`` — the
    same provenance ``meta.synthesize_rules`` mints carry — so the
    store's ``_guard_transport_spec`` re-encodes it faithfully on a
    re-store, and ``missing_hooks`` on it honestly reports the
    procedural hooks its class carries.
    """
    rule, _ = _resolve(name, lib, frozenset())
    if ship_as is not None:
        parents = getattr(rule, "parents", None)
        object.__setattr__(rule, "name", ship_as)
        if parents is not None:
            meta.SYNTH_PARENTS[ship_as] = parents
    if tags is not None:
        object.__setattr__(rule, "tags", frozenset(tags))
    if law is not None:
        object.__setattr__(rule, "law", law)
    return rule
