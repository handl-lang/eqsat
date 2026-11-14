"""Structural term utilities + certificate verification."""

from __future__ import annotations

import contextlib
from typing import Any, cast

from mlopt_core.egraph.certs import (
    Certificate,
    CertificateVerificationError,
)
from mlopt_core.ir import Op, op_repr_dag
from mlopt_core.traversal import expand, postorder

_MEMO_MISS = object()


def _term_eq(a: Any, b: Any, _memo: dict | None = None) -> bool:
    """Structural term equality — the ``op_repr`` check, DAG-aware.

    ``op_repr(a) == op_repr(b)`` renders each term as a *tree*: a
    sharing-heavy term DAG (``_oldest_term`` resolutions, unrolled
    scans — eclass cones are shared by construction) tree-expands
    exponentially, so a single equality check is itself the
    certificate path's superlinear blowup (measured: a depth-22
    shared DAG cost ~15 s per compare; depth ~40 never finishes).
    This compares the same structure directly over an explicit
    worklist — iterative, so deep terms cannot hit ``RecursionError``
    — and memoises decided object pairs, so shared sub-DAGs are
    visited once.

    Semantics mirror ``op_repr`` equality: Op nodes compare op name,
    the repr-keyed ``_ak`` attr signature (the interning identity —
    one notch stricter than ``op_repr``'s ``str`` rendering, which
    conflated ``1`` with ``"1"``), and args pairwise; non-Op leaves
    compare by ``repr`` exactly as ``op_repr`` renders them.  An Op
    never equals a non-Op.  The ``_memo`` dict may be shared across
    calls to amortise repeated compares over the same cones.

    The pair graph is walked children-before-parents (the
    :func:`mlopt_core.traversal.postorder` convention applied to
    ``(a, b)`` pairs), then evaluated in that order — every child
    pair's verdict is in ``_memo`` before its parent is decided.
    """
    if a is b:
        return True
    if _memo is None:
        _memo = {}
    try:
        hit = _memo.get((a, b), _MEMO_MISS)
    except TypeError:
        hit = _MEMO_MISS  # unhashable leaf — skip memoisation
    if hit is not _MEMO_MISS:
        return cast("bool", hit)

    def _pair_memo_get(pair: tuple) -> Any:
        try:
            return _memo.get(pair, _MEMO_MISS)
        except TypeError:
            return _MEMO_MISS  # unhashable leaf — skip memoisation

    def _pair_val(x: Any, y: Any) -> bool:
        """Decided verdict for a child pair (memo, else recompute)."""
        try:
            v = _memo.get((x, y), _MEMO_MISS)
        except TypeError:
            v = _MEMO_MISS
        if v is _MEMO_MISS:
            # Unhashable pair — the memo could not store it; the
            # recursive entry point is itself iterative, so this
            # nested call cannot recurse deeply.
            return _term_eq(x, y, _memo)
        return cast("bool", v)

    # Collect the unresolved pair frontier: postorder over pairs,
    # pruning pairs already decided in the shared memo.  The root
    # pair's done-frame lands last, so ``order[-1]`` is ``(a, b)``.
    order: list[tuple[Any, Any]] = []
    expanded: set = set()
    work: list[tuple[tuple[Any, Any], bool]] = [((a, b), False)]
    while work:
        pair, done = work.pop()
        if done:
            order.append(pair)
            continue
        x, y = pair
        if _pair_memo_get(pair) is not _MEMO_MISS:
            continue
        try:
            fresh = pair not in expanded
        except TypeError:
            fresh = True  # unhashable pair — cannot dedup
        if not fresh:
            continue
        with contextlib.suppress(TypeError):
            expanded.add(pair)  # unhashable pair — cannot dedup
        work.append((pair, True))
        if x is y:
            continue  # identical objects — True at eval, no descent
        if (
            isinstance(x, Op)
            and isinstance(y, Op)
            and len(x.args) == len(y.args)
        ):
            work.extend(
                ((xi, yi), False)
                for xi, yi in zip(x.args, y.args, strict=True)
            )
        # Mismatched arity / Op-vs-leaf pairs have no children to
        # decide — they evaluate False at their own frame.

    # Evaluate children before parents; every child pair's verdict is
    # memoised by the time its parent is reached.
    result = False
    for x, y in order:
        if x is y:
            res = True
        elif isinstance(x, Op) and isinstance(y, Op):
            res = (
                x.op == y.op
                and x._ak == y._ak
                and len(x.args) == len(y.args)
                and all(
                    _pair_val(xi, yi)
                    for xi, yi in zip(x.args, y.args, strict=True)
                )
            )
        elif isinstance(x, Op) or isinstance(y, Op):
            res = False
        else:
            res = repr(x) == repr(y)
        with contextlib.suppress(
            TypeError
        ):  # pragma: no cover — unhashable leaf
            _memo[(x, y)] = res
        if x is a and y is b:
            result = res
    return result


def _brief(term: Any, limit: int = 160) -> str:
    """Bounded render for diagnostics.

    ``op_repr`` can be exponentially large on a shared term DAG;
    ``op_repr_dag`` is linear in the distinct nodes, then truncated
    to ``limit`` chars.
    """
    s = op_repr_dag(term)
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def _term_paths(term: Any, prefix: tuple = ()) -> Any:
    """Yield the child-index path of every subterm, root first.

    Delegates to :func:`mlopt_core.traversal.expand` — the iterative
    tree-spelling walk, yielding the identical pre-order sequence;
    ``prefix`` prepends to every path exactly as the recursive walk
    did (``prefix`` alone for the root).
    """
    for path, _node in expand(term):
        yield prefix + path


def _iter_ops(term: Any):
    """Yield every Op node in a term (for the structural-size tie-break).

    Delegates to :func:`mlopt_core.traversal.expand` — the identical
    pre-order, per-occurrence sequence as the historical recursive
    walk, but iterative (no ``RecursionError`` on deep terms).  Note
    this is the *tree* spelling: a shared subterm yields once per use
    site; :func:`mlopt_core.traversal.ops` is the deduped DAG view.
    """
    for _path, node in expand(term):
        if isinstance(node, Op):
            yield node


# ---------------------------------------------------------------------------
#  Certificate verification — replay on real terms, no e-graph
# ---------------------------------------------------------------------------


def _term_match(
    pattern: Any, term: Any, _subst: dict | None = None
) -> dict | None:
    """Structural match of a pattern against a plain *term* (no e-graph).

    Mirrors :meth:`EGraph._match` semantics at term granularity: string
    leaves in the pattern are metavariables bound to subterms (repeated
    metavariables must bind structurally equal terms); string-valued
    attributes are attribute metavariables bound under ``"$attr:"``
    keys; concrete leaves (Var/Const/Param) match by ``repr``.
    Concrete attr values compare numerically (``!=``) — match-time
    leniency, deliberately kept while term/enode *identity* is
    spelling-strict (see :class:`mlopt_core.ir.Op`).
    Returns the bindings dict, or ``None`` on mismatch.

    Iterative pre-order DFS over an explicit stack — the identical
    binding/check order as the historical recursion (node-local
    checks, then args left-to-right), so deep patterns/terms cannot
    hit ``RecursionError``.
    """
    subst = {} if _subst is None else _subst
    stack = [(pattern, term)]
    while stack:
        p, t = stack.pop()
        if isinstance(p, str):
            prev = subst.get(p)
            if prev is None:
                subst[p] = t
                continue
            if not _term_eq(prev, t):
                return None
            continue
        if isinstance(p, Op):
            if not isinstance(t, Op) or t.op != p.op:
                return None
            if len(t.args) != len(p.args):
                return None
            if set(t.attrs) != set(p.attrs):
                return None
            for k, pv in p.attrs.items():
                nv = t.attrs[k]
                if isinstance(pv, str):
                    key = "$attr:" + pv
                    if key in subst:
                        if subst[key] != nv:
                            return None
                    else:
                        subst[key] = nv
                elif nv != pv:
                    return None
            # Push arg pairs reversed so the leftmost pops first —
            # the recursive walk's left-to-right DFS order.
            stack.extend(
                reversed(list(zip(p.args, t.args, strict=True)))
            )
            continue
        # concrete leaf (Const/Param/Var embedded in the pattern)
        if repr(p) != repr(t):
            return None
    return subst


def _term_instantiate(pattern: Any, subst: dict) -> Any:
    """Instantiate a pattern with term-valued bindings (pure terms).

    The term-level analogue of :meth:`EGraph._instantiate`: metavariable
    strings map to terms, ``"$attr:"`` keys resolve attribute
    metavariables, concrete leaves pass through unchanged.

    Iterative: children are instantiated before parents over the
    :func:`mlopt_core.traversal.postorder` walk (memoised per node —
    ``Op.make`` interning made the old per-occurrence rebuild return
    the same objects anyway), so deep patterns cannot hit
    ``RecursionError``.
    """
    memo: dict = {}
    out: Any = None
    have_out = False
    for n in postorder(pattern):
        if isinstance(n, str):
            val = subst[n]
        elif isinstance(n, Op):
            args = [memo[a] for a in n.args]
            attrs = {}
            for k, v in n.attrs.items():
                if isinstance(v, str):
                    attrs[k] = subst.get("$attr:" + v, v)
                else:
                    attrs[k] = v
            val = Op.make(n.op, *args, **attrs)
        else:
            val = n
        with contextlib.suppress(
            TypeError
        ):  # pragma: no cover — unhashable leaf
            memo[n] = val
        if n is pattern:
            out = val
            have_out = True
    if have_out:
        return out
    # Unreachable for hash-consed terms (the root is always yielded),
    # kept as the honest fallback for exotic unhashable inputs.
    return memo.get(pattern, pattern)


def _subterm(term: Any, path: tuple) -> Any:
    """Return the subterm of *term* at child-index path, or None."""
    for i in path:
        if not isinstance(term, Op) or i >= len(term.args):
            return None
        term = term.args[i]
    return term


def _replace_subterm(term: Any, path: tuple, new: Any) -> Any:
    """*term* with the subterm at *path* replaced by *new*.

    Iterative: descend the spine once, then rebuild bottom-up — the
    identical result and the identical ``CertificateVerificationError``
    (naming the node where descent failed, via ``_brief``) as the
    historical recursion, without the ``RecursionError`` on deep
    paths.  Negative indices keep their Python indexing semantics.
    """
    if not path:
        return new
    spine: list[tuple[Op, int]] = []
    node = term
    for i in path:
        if not isinstance(node, Op) or i >= len(node.args):
            raise CertificateVerificationError(
                f"cannot descend path {list(path)} in {_brief(node)}"
            )
        spine.append((node, i))
        node = node.args[i]
    for parent, i in reversed(spine):
        args = list(parent.args)
        args[i] = new
        new = Op.make(parent.op, *args, **dict(parent.attrs))
    return new


def verify_certificate(
    src_term: Any, cert: Certificate, *, strict: bool = False
) -> Any:
    """Replay a certificate's rule applications on real terms.

    For each step: descend to ``path`` in the evolving term, re-match
    the rule's LHS against the subterm found there, check the recorded
    bindings and the recorded LHS/RHS instances, re-run ``check``/
    ``derive`` side conditions, instantiate the RHS, and substitute.
    ``egraph_dependent`` steps cannot be replayed standalone — the
    recorded ``lhs`` must still be present at ``path``, then ``rhs`` is
    substituted as a trusted assertion; ``strict=True`` rejects any
    certificate containing them.

    Returns the reconstructed term — ``cert.dst`` on success.
    Raises :class:`CertificateVerificationError` on any mismatch.
    """
    if strict and cert.n_egraph_dependent:
        raise CertificateVerificationError(
            f"{cert.n_egraph_dependent} e-graph-dependent step(s) "
            "cannot be replayed standalone"
        )
    if cert.src is not None and not _term_eq(src_term, cert.src):
        raise CertificateVerificationError(
            f"source mismatch: certificate proves {_brief(cert.src)}, "
            f"got {_brief(src_term)}"
        )
    current = src_term
    eq_memo: dict = {}
    for i, step in enumerate(cert.steps):
        sub = _subterm(current, step.path)
        if sub is None:
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): path {list(step.path)} "
                f"absent in {_brief(current)}"
            )
        if step.egraph_dependent:
            if not _term_eq(sub, step.lhs, eq_memo):
                raise CertificateVerificationError(
                    f"step {i} (e-graph-dependent): expected "
                    f"{_brief(step.lhs)} at {list(step.path)}, "
                    f"found {_brief(sub)}"
                )
            current = _replace_subterm(current, step.path, step.rhs)
            continue
        rule = cert.rules.get(step.rule)
        if rule is None:
            raise CertificateVerificationError(
                f"step {i}: unknown rule {step.rule!r}"
            )
        if not _term_eq(step.lhs, sub, eq_memo):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): recorded LHS instance "
                f"{_brief(step.lhs)} != subterm {_brief(sub)}"
            )
        m = _term_match(rule.lhs, sub)
        if m is None:
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): LHS does not match "
                f"{_brief(sub)}"
            )
        for k, v in step.bindings.items():
            if k not in m:
                continue  # derived binding — checked via derive/rhs below
            same = (
                m[k] == v
                if k.startswith("$attr:")
                else _term_eq(m[k], v, eq_memo)
            )
            if not same:
                raise CertificateVerificationError(
                    f"step {i} ({step.rule}): binding {k} tampered"
                )
        if rule.check is not None and not rule.check(m):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): side condition fails on replay"
            )
        inst = dict(m)
        for k, v in step.bindings.items():
            inst.setdefault(k, v)  # derived $attr bindings
        if rule.derive is not None:
            extra = rule.derive(m)
            if extra is None:
                raise CertificateVerificationError(
                    f"step {i} ({step.rule}): derive vetoed on replay"
                )
            inst.update(extra)
        rhs = _term_instantiate(rule.rhs, inst)
        if not _term_eq(rhs, step.rhs, eq_memo):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): recorded RHS "
                f"{_brief(step.rhs)} != instantiated {_brief(rhs)}"
            )
        current = _replace_subterm(current, step.path, rhs)
    if not _term_eq(current, cert.dst, eq_memo):
        raise CertificateVerificationError(
            f"replay produced {_brief(current)}, "
            f"certificate claims {_brief(cert.dst)}"
        )
    return current
