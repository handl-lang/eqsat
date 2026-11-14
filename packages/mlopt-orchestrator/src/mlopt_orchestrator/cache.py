"""Structural-signature cache — repeated-block search reuse.

Transformer-style stacks repeat ONE block structure N times with
different parameters: :func:`structural_key` canonicalises each
block's exported IR modulo leaf *names* (op-DAG, constants, input
positions, shapes and dtypes pinned) and the compositional pass
stores the extracted term under it.  A hit re-leafs the template
under the current block's own names (:func:`_cache_replay`),
re-derives the value-dependent pieces (exact-tie clusters,
``__heads`` dedup stacks, the causal-mask fold) and verifies
numerically before grafting — an unservable replay declines to the
ordinary per-block search.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.ir import IR, Const, Op, Param, Var
from mlopt_core.laws import headshare_keys_hold
from mlopt_core.laws.pairing import _raw_bytes
from mlopt_core.pipeline import PipelineStats, SearchResult
from mlopt_core.ports import Sink, Source

# ---------------------------------------------------------------------------
#  Structural-signature cache — repeated-block search reuse
# ---------------------------------------------------------------------------
#
# Stacked transformer-style models repeat one block structure N times
# with different parameters: ``Compositional`` would run N identical
# equality-saturation searches.  The cache canonicalises each block's
# exported IR modulo leaf *names* (``structural_key``) and stores the
# extracted term under it; a hit re-leafs that term with the current
# block's own parameter names, re-derives the value-dependent pieces
# (tied-weight assumptions, ``__heads`` dedup stacks, the causal-mask
# fold), lowers it and verifies numerically before grafting — a replay
# that cannot honour the template's assumptions declines to the
# ordinary per-block search.


def _canon_attr(v: Any, ctx: dict) -> Any:
    """Canonical key material for one op attr value.

    Term-likes in attr position canonicalise through the same walk
    (attrs may embed subterms); containers recurse; other unhashable
    values fall back to ``repr``.
    """
    if isinstance(v, (Op, Param, Var, Const, list, tuple, dict)):
        return _canon_term(v, ctx)
    try:
        hash(v)
    except TypeError:
        return repr(v)
    return v


def _canon_leaf(t: Any, ctx: dict) -> Any:
    """Canonical key for a leaf — Param/Var/Const or a non-term node.

    No memo needed: Param positions are idempotent (``setdefault`` on
    the first-occurrence index) and a leaf key is O(1) to recompute.
    A Var absent from ``inputs`` falls back to its name — a
    conservative miss, never a collision.
    """
    if isinstance(t, Param):
        pos = ctx["params"].setdefault(t.name, len(ctx["params"]))
        return (
            "param",
            pos,
            tuple(t.typ.shape),
            ctx["meta"].get(t.name),
        )
    if isinstance(t, Var):
        pos = ctx["vars"].get(t.name)
        return (
            "var",
            pos if pos is not None else t.name,
            tuple(t.typ.shape),
            ctx["meta"].get(t.name),
        )
    if isinstance(t, Const):
        return ("const", type(t.value).__name__, t.value)
    return ("leaf", type(t).__name__, repr(t))


def _canon_term(t: Any, ctx: dict) -> Any:
    """Canonical form of one term under the leaf-renaming ``ctx``.

    ``ctx`` carries ``vars`` (Var name -> input position), ``params``
    (Param name -> first-occurrence position, built by the walk),
    ``meta`` (leaf name -> extra key material, e.g. dtype) and the
    DAG ``memo`` — interned ``Op`` objects only (always hashable), so
    shared subterms canonicalise once and hash-consed duplicates are
    free.
    """
    if isinstance(t, (list, tuple)):
        return ("seq", tuple(_canon_term(a, ctx) for a in t))
    if isinstance(t, dict):
        return (
            "map",
            tuple(
                sorted((k, _canon_term(v, ctx)) for k, v in t.items())
            ),
        )
    if not isinstance(t, Op):
        return _canon_leaf(t, ctx)
    hit = ctx["memo"].get(t)
    if hit is not None:
        return hit
    key = (
        "op",
        t.op,
        tuple(_canon_term(a, ctx) for a in t.args),
        tuple(
            sorted((n, _canon_attr(v, ctx)) for n, v in t.attrs.items())
        ),
    )
    ctx["memo"][t] = key
    return key


def _canonical(
    root: Any,
    inputs: Iterable,
    leaf_meta: Mapping | None,
) -> tuple[tuple, tuple[str, ...]]:
    """Canonical key plus the Param first-occurrence name order.

    The key is ``(body, inputs_sig)``: the body canonicalises the term
    DAG modulo leaf names; ``inputs_sig`` pins the input arity and
    every input's declared shape/meta — including inputs the root
    never references.  The second return lists Param names in the same
    first-occurrence order the key's positional indices encode.
    """
    meta = leaf_meta if leaf_meta is not None else {}
    inputs = tuple(inputs)  # may be any iterable — consumed twice
    ctx = {
        "meta": meta,
        "vars": {v.name: i for i, v in enumerate(inputs)},
        "params": {},
        "memo": {},
    }
    inputs_sig = tuple(
        (tuple(v.typ.shape), meta.get(v.name)) for v in inputs
    )
    return (_canon_term(root, ctx), inputs_sig), tuple(ctx["params"])


def structural_key(
    root: Any,
    inputs: Iterable = (),
    leaf_meta: Mapping | None = None,
) -> tuple:
    """Canonical hashable signature of a term, modulo leaf names.

    Two exported blocks are structurally identical iff their roots
    have equal keys: the same op-DAG (op names, argument order, attr
    values), the same ``Const`` leaves (type-tagged, so ``Const(2)``
    and ``Const(2.0)`` differ), and ``Var``/``Param`` leaves that vary
    at most by name.  A ``Var`` canonicalises to its position in
    ``inputs`` (so ``sub(x, y)`` and ``sub(y, x)`` differ), a ``Param``
    to its first-occurrence position in the deterministic traversal.

    ``leaf_meta`` maps a leaf name to extra key material — the
    compositional pass supplies each leaf's dtype (``TensorType``
    carries shape only), so same-shape different-dtype blocks never
    share an entry.
    """
    return _canonical(root, inputs, leaf_meta)[0]


def _param_order(root: Any) -> tuple[str, ...]:
    """Param leaf names in ``structural_key``'s first-occurrence order."""
    return _canonical(root, (), None)[1]


def _param_leaves(term: Any) -> Any:
    """Yield every ``Param`` leaf reachable in *term* (DAG-memoized)."""
    seen: set = set()

    def walk(t: Any) -> Any:
        try:
            if t in seen:
                return
            seen.add(t)
        except TypeError:
            pass  # unhashable container arg — walk through it anyway
        if isinstance(t, Param):
            yield t
        elif isinstance(t, Op):
            for a in t.args:
                yield from walk(a)
        elif isinstance(t, (list, tuple)):
            for a in t:
                yield from walk(a)

    yield from walk(term)


def _term_param_names(term: Any) -> set[str]:
    """Set of ``Param`` leaf names reachable in *term*."""
    return {p.name for p in _param_leaves(term)}


def _param_shapes_ok(term: Any, params: Mapping) -> bool:
    """Check remapped Param leaves' declared shapes against values.

    A leaf whose declared shape is fully concrete must match the
    supplied tensor's shape — a cheap structural guard that catches a
    term cached under the wrong key before any lowering runs.  Leaves
    with unknown dims (``None``) or names absent from ``params`` are
    skipped; coverage is enforced separately.
    """
    for p in _param_leaves(term):
        declared = tuple(p.typ.shape)
        if None in declared:
            continue
        t = params.get(p.name)
        if t is not None and tuple(getattr(t, "shape", ())) != declared:
            return False
    return True


def _leaf_equal(a: Any, b: Any) -> bool:
    """Exact value equality of two tensor-like leaves (torch-free).

    Mirrors ``mlopt_core.laws.pairing._exact_equal`` — same shape and
    elementwise equal via the objects' own ``==``/``.all()``.
    """
    try:
        if a is None or b is None or tuple(a.shape) != tuple(b.shape):
            return False
        r = a == b
        return bool(r.all() if hasattr(r, "all") else r)
    except Exception:
        return False


def _dedup_blocks(
    t: Any, h: int
) -> tuple[list, tuple[int, ...], int, int] | None:
    """First-occurrence unique head-blocks of ``t`` plus the index map.

    Mirrors ``share_duplicate_param_slices``: split the (o, i) tensor
    into ``h`` row-blocks, keep first occurrences by raw bytes, return
    ``(unique_blocks, index_map, d, i)`` — ``None`` when the split is
    degenerate.  Tensor ops are duck-typed (``new_zeros`` / slicing /
    the shared ``_raw_bytes`` signature) — the orchestrator stays
    backend-neutral.
    """
    o, i = int(t.shape[0]), int(t.shape[1])
    if h < 2 or o % h or o // h <= 0:
        return None
    d = o // h
    blocks = [t[j * d : (j + 1) * d] for j in range(h)]
    sigs = [_raw_bytes(b) for b in blocks]
    uniq_sigs: list[bytes] = []
    uniq_blocks: list = []
    imap: list[int] = []
    for j, s in enumerate(sigs):
        try:
            imap.append(uniq_sigs.index(s))
        except ValueError:
            uniq_sigs.append(s)
            uniq_blocks.append(blocks[j])
            imap.append(len(uniq_sigs) - 1)
    return uniq_blocks, tuple(imap), d, i


def _derive_dedup(
    rec: Mapping, pmap: Mapping, source_tensors: Mapping
) -> tuple[Any, str] | None:
    """Recompute a ``{base}__heads{h}`` dedup stack for new values.

    The replay is servable only when the new block's redundancy
    pattern reproduces the recorded index map exactly — otherwise the
    template term's ``index_select`` would gather the wrong rows.
    Returns ``(value, current_derived_name)`` or ``None``.
    """
    base = pmap.get(rec["base"])
    t = source_tensors.get(base) if base is not None else None
    h, imap = rec["heads"], tuple(rec["imap"])
    if (
        t is None
        or getattr(t, "dim", lambda: -1)() != 2
        or len(imap) != h
    ):
        return None
    try:
        out = _dedup_blocks(t, h)
    except Exception:
        return None
    if out is None or out[1] != imap:
        return None
    uniq_blocks, _cur_imap, d, i = out
    dedup = uniq_blocks[0].new_zeros((len(uniq_blocks), d, i))
    for j, u in enumerate(uniq_blocks):
        dedup[j] = u
    return dedup.detach(), f"{base}__heads{h}"


def _remap_term(
    term: Any, pmap: Mapping, vmap: Mapping, dmap: Mapping
) -> Any | None:
    """Re-leaf *term* under the name maps; None if a leaf is unmappable.

    ``pmap``: template Param name -> current name; ``vmap``: template
    Var name -> the current ``Var`` object; ``dmap``: derived Param
    name -> current derived name.  An unmapped leaf aborts — an
    unrenamed template name would mis-bind at eval time.
    """
    bad: list[str] = []
    memo: dict = {}

    def go(t: Any) -> Any:
        hit = memo.get(t)
        if hit is not None:
            return hit
        if isinstance(t, Op):
            out = Op.make(
                t.op, *(go(a) for a in t.args), **dict(t.attrs)
            )
        elif isinstance(t, Param):
            new = pmap.get(t.name, dmap.get(t.name))
            if new is None:
                bad.append(t.name)
                out = t
            else:
                out = Param(new, t.typ)
        elif isinstance(t, Var):
            out = vmap.get(t.name)
            if out is None:
                bad.append(t.name)
                out = t
        else:
            out = t
        memo[t] = out
        return out

    result = go(term)
    return None if bad else result


def _cache_entry(res: SearchResult) -> dict[str, Any]:
    """Build the replayable template from a completed search.

    The entry stores the extracted term — the PRE-causal-fold form
    when the fold fired, so a replay re-specializes against the new
    block's own mask values — plus the leaf correspondence tables and
    the value-dependent assumptions recorded in
    ``stats["param_sharing"]``: exact-tie clusters to re-check and
    ``__heads`` derivations to recompute.  An unrecorded derived name
    gets a ``None`` recipe, making hits on this entry decline.
    """
    term = res.stats.get("pre_causal_term", res.term)
    term_params = _term_param_names(term)
    sharing = res.stats.get("param_sharing") or {}
    order = _param_order(res.ir.root)
    return {
        "term": term,
        "params": tuple(order),
        "inputs": tuple(v.name for v in res.ir.inputs),
        "term_params": term_params,
        "ties": sharing.get("ties", ()),
        "headshare": sharing.get("headshare", ()),
        "derived": {
            n: sharing.get("derived", {}).get(n)
            for n in term_params - set(order)
        },
        "stats": res.stats,
    }


def _block_key(ir: IR, source_tensors: Mapping, args: tuple) -> tuple:
    """Structural signature of one exported block.

    ``leaf_meta`` carries each leaf's dtype — the IR's ``TensorType``
    is shape-only, so a float64 block and a float32 twin must not
    share a template.  Param dtypes come from ``source_tensors``;
    input dtypes from the captured positional ``args``.
    """
    meta = {
        n: str(getattr(t, "dtype", type(t).__name__))
        for n, t in source_tensors.items()
    }
    for i, v in enumerate(ir.inputs):
        a = args[i] if i < len(args) else None
        meta[v.name] = str(getattr(a, "dtype", type(a).__name__))
    return structural_key(ir.root, inputs=ir.inputs, leaf_meta=meta)


def _replay_derived(
    entry: dict, pmap: Mapping, source_tensors: Mapping, params: dict
) -> dict[str, str] | None:
    """Materialise derived-param values for the hit block.

    Each ``__heads``-style recipe recomputes the deduplicated stack
    from the CURRENT block's base tensor and lands it under the
    current derived name — the name remap alone cannot supply it since
    the value was computed, at search time, from the template's own
    weights.  Returns the derived-name remap, or ``None`` when any
    recipe fails (unservable entry → caller falls back to search).
    """
    dmap: dict[str, str] = {}
    for dname, rec in entry["derived"].items():
        if rec is None:
            return None
        pair = _derive_dedup(rec, pmap, source_tensors)
        if pair is None:
            return None
        value, cur_name = pair
        dmap[dname] = cur_name
        params[cur_name] = value
    return dmap


def _ties_hold(
    ties: Iterable, term_params: set, pmap: Mapping, params: Mapping
) -> bool:
    """Re-check the template's exact-tie assumptions on new values.

    A param dropped in favour of a tied representative must hold an
    equal tensor in this block too — otherwise the canonical leaf
    substitutes a different weight.  Clusters the term kept wholesale
    (or dropped wholesale) commit to nothing and are skipped; names
    unmappable in this block are likewise inert.  All names are
    template-side, mapped to current names through ``pmap``.
    """
    for cluster in ties:
        kept = [
            pmap[n] for n in cluster if n in term_params and n in pmap
        ]
        dropped = [
            pmap[n]
            for n in cluster
            if n not in term_params and n in pmap
        ]
        if not kept or not dropped:
            continue
        anchor = params.get(kept[0])
        if any(not _leaf_equal(params.get(n), anchor) for n in dropped):
            return False
    return True


def _cache_replay(
    entry: dict | None,
    ir: IR,
    source_tensors: Mapping,
    *,
    sink: Sink,
    source: Source,
    model: Any,
) -> SearchResult | None:
    """Instantiate a cached template for a structurally-equal block.

    Returns a lowering-ready :class:`SearchResult` — the template term
    re-leafed under this block's leaf names, billed against this
    block's own ``param_values`` — or ``None`` when the entry cannot
    serve this block (leaf-arity mismatch, an unrecorded derived
    parameter, a violated tie/dedup assumption, an unmappable leaf, or
    a shape/param coverage failure).  ``None`` sends the caller down
    the ordinary per-block search path.
    """
    if entry is None:
        return None
    order = _param_order(ir.root)
    if len(order) != len(entry["params"]) or len(
        entry["inputs"]
    ) != len(ir.inputs):
        return None
    pmap = dict(zip(entry["params"], order, strict=True))
    vmap = dict(zip(entry["inputs"], ir.inputs, strict=True))

    params = dict(source_tensors)
    dmap = _replay_derived(entry, pmap, source_tensors, params)
    if dmap is None or not _ties_hold(
        entry["ties"], entry["term_params"], pmap, params
    ):
        return None
    # HeadShare offers lean on bitwise head equality — re-verify the
    # recorded key slices against this block's own tensors (names
    # remapped through pmap + the derived-name map).
    if not headshare_keys_hold(
        entry.get("headshare", ()),
        lambda n: dmap.get(n, pmap.get(n)),
        params,
    ):
        return None

    term = _remap_term(entry["term"], pmap, vmap, dmap)
    if term is None:
        return None
    # Re-run the causal-mask const fold on this block's own values:
    # a mask still evaluating to the causal triangle re-folds; one
    # that does not stays a materialised argument — either way a
    # correct term.
    fold = getattr(sink, "specialize_causal", None)
    if fold is not None:
        term = fold(term, params, {})
    if not _term_param_names(term) <= set(
        params
    ) or not _param_shapes_ok(term, params):
        return None

    stats = PipelineStats(entry["stats"])
    stats["cache_replay"] = True
    return SearchResult(
        ir=ir,
        # The replayed result carries no saturated e-graph — the search
        # was skipped.  ``lower`` never touches the engine fields.
        eg=EGraph(),
        root_eid=-1,
        term=term,
        param_values=params,
        stats=stats,
        source=source,
        model=model,
    )
