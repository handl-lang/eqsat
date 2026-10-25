"""Shipped minted objects — stored-composite materialization in core.

The discovery pipeline's stored objects (``mlopt_discovery.evidence``)
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
domain-side (``mlopt_discovery.shipped_primitives.register()`` seats
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
"""

from __future__ import annotations

from typing import Any

from mlopt_core import meta
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import op_def
from mlopt_core.laws import store as _store
from mlopt_core.laws import tags
from mlopt_core.laws.serialize import object_from_data
from mlopt_core.rulecache import _dec_binding

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
    "expand_name",
    "materialize",
    "prim2_folds",
    "prim2_premises",
    "prim2_records",
    "prim3_folds",
    "prim3_premises",
    "prim3_records",
    "prim_fold",
    "prim_folds",
    "prim_premises",
    "prim_records",
    "synop_name",
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
#: (``mlopt_discovery.shipped_primitives``) into the core bank.
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
)

#: The depth-2 subset — the bootstrap mint's ``synop2_*`` ops.
SYNOP2_SHIPPED: frozenset[str] = frozenset(_PRIM2_RECORDS)

#: The depth-3 subset — the vocab-priced bootstrap's ``synop3_*``
#: ops (skeletons spelled through both shipped tiers).
SYNOP3_SHIPPED: frozenset[str] = frozenset(_PRIM3_RECORDS)

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

    All three depths, merged: the 21 depth-1 ``synop_*`` records plus
    the 2 depth-2 ``synop2_*`` records plus the 34 depth-3
    ``synop3_*`` records (the deeper tiers additionally carry
    ``alpha_fold`` / ``embeds_shipped`` / ``embeds`` / ``expand``).
    """
    return {**_PRIM_RECORDS, **_PRIM2_RECORDS, **_PRIM3_RECORDS}


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

    *synop* resolves across all three tables — a ``synopN_*`` name
    materializes its depth-N record.  The deeper-tier records embed
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

    All three depths: the 21 ``prim:synop_*`` folds, then the 2
    depth-2 ``prim:synop2_*`` folds, then the 34 depth-3
    ``prim:synop3_*`` folds (skeleton -> ``synopN_*(args)``).
    """
    return tuple(
        prim_fold(n)[0]
        for n in (*_PRIM_RECORDS, *_PRIM2_RECORDS, *_PRIM3_RECORDS)
    )


def prim2_folds() -> tuple[Rewrite, ...]:
    """Return the 2 depth-2 fold laws — the bootstrap mint's slice."""
    return tuple(prim_fold(n)[0] for n in _PRIM2_RECORDS)


def prim3_folds() -> tuple[Rewrite, ...]:
    """Return the 34 depth-3 fold laws — the vocab-priced mint's."""
    return tuple(prim_fold(n)[0] for n in _PRIM3_RECORDS)


def prim_premises() -> tuple[Rewrite, ...]:
    """Return the definitional ``synop*_*.expand`` premises — axioms.

    NOT a saturation set: expanding ``synop -> pattern`` inside a
    search would re-add the unfolded spelling the fold collapses.
    These are the anchored-derivation axioms a cert/verify leg unions
    into its axiom set; each fold also carries its own premise as
    ``rule.premises``.  All three depths — a deeper-tier premise's
    RHS is the skeleton's *expanded* manual denotation (the embedded
    ``expand`` record), so its anchored derivation still closes
    through the shipped lower-tier folds.
    """
    return tuple(
        prim_fold(n)[1]
        for n in (*_PRIM_RECORDS, *_PRIM2_RECORDS, *_PRIM3_RECORDS)
    )


def prim2_premises() -> tuple[Rewrite, ...]:
    """Return the depth-2 ``synop2_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _PRIM2_RECORDS)


def prim3_premises() -> tuple[Rewrite, ...]:
    """Return the depth-3 ``synop3_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _PRIM3_RECORDS)


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
