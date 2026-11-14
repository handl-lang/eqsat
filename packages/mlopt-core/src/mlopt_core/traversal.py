"""Shared DAG/subterm walkers for IR terms — iterative, deduplicated.

Terms are hash-consed DAGs (:class:`mlopt_core.ir.Op` interns by
structure), so every "walk the term" need has two distinct semantics
that used to be re-implemented ad hoc across ``ir``/``egraph.terms``/
``typing``/``meta``/``cost`` (the audit's "~10 parallel traversals"):

* **DAG walks** — visit each *distinct* node once.  ``subterms`` /
  ``ops`` / ``leaves`` / ``postorder``.  Dedup is by the node's own
  ``__eq__``/``__hash__`` (content identity — interned ``Op``s make
  that object identity in practice; ``Const``/``Var``/``Param`` leaves
  dedup structurally, which is the "counted once" contract).
* **Tree spells** — visit every *occurrence*, exactly like the
  historical recursive walkers (``_term_paths``/``_iter_ops``).
  :func:`expand` yields ``(path, node)`` positions in pre-order with
  child-index paths; ``depth`` bounds the descent and ``budget`` caps
  the total positions emitted, so a sharing-heavy DAG cannot expand
  exponentially *silently* — the caller picks the bound.

Every walker is an explicit stack — no recursion — so a deep *unshared*
chain (100k+ ops) cannot hit ``RecursionError``.  Unhashable nodes
(degenerate leaves that never occur as ``Op`` args, which are
hash-checked at mint) degrade to per-occurrence visits rather than
raising.

``postorder`` visits children **right-to-left** — the historical
``ir._dag_postorder`` stack order — so migrating that walker onto it
keeps ``op_repr_dag``'s ``#n`` binding order byte-identical.  For
memoized bottom-up passes (``typing._shape_of``/``has_var_leaf``) any
postorder works; pass ``prune`` to stop at already-computed nodes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from mlopt_core.ir import Op

__all__ = [
    "expand",
    "leaves",
    "ops",
    "postorder",
    "subterms",
]


def _children(node: Any) -> tuple:
    """Return the child positions of *node* (``()`` for leaves)."""
    return node.args if isinstance(node, Op) else ()


def _fresh(seen: set, node: Any) -> bool:
    """Return True iff *node* is not yet in *seen*; record it if so.

    Unhashable nodes cannot dedup — they are always "fresh" (visited
    per occurrence), matching the pre-dedup recursive walkers.
    """
    try:
        if node in seen:
            return False
        seen.add(node)
        return True
    except TypeError:
        return True


def subterms(term: Any) -> Iterator[Any]:
    """Yield every distinct subterm of *term*, root first (pre-order).

    DAG-deduplicated: a shared subterm is yielded once, at its first
    pre-order position.  Leaves (``Var``/``Param``/``Const``/metavar
    strings/anything non-:class:`Op`) are subterms too.
    """
    seen: set = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if not _fresh(seen, node):
            continue
        yield node
        stack.extend(reversed(_children(node)))


def ops(term: Any) -> Iterator[Op]:
    """Yield every distinct :class:`Op` node under *term*, pre-order.

    Non-Op leaves are skipped (they have no Op descendants).  A
    multiply-referenced op is yielded once — use :func:`expand` for the
    per-occurrence tree spelling.
    """
    seen: set = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if not isinstance(node, Op) or not _fresh(seen, node):
            continue
        yield node
        stack.extend(reversed(node.args))


def leaves(term: Any) -> Iterator[Any]:
    """Yield every distinct non-:class:`Op` leaf under *term*.

    Includes *term* itself when it is a leaf.  Dedup is structural —
    two equal ``Const``/``Var``/``Param`` leaves yield once.
    """
    seen: set = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if isinstance(node, Op):
            stack.extend(reversed(node.args))
            continue
        if _fresh(seen, node):
            yield node


def postorder(
    term: Any,
    *,
    prune: Callable[[Any], bool] | None = None,
) -> Iterator[Any]:
    """Yield every distinct node under *term*, children before parents.

    Children are visited **right-to-left** (the ``_dag_postorder``
    stack convention — see module docstring).  Each node is yielded
    once, after all of its descendants.

    ``prune(node)`` marks an already-computed frontier: pruned nodes
    are neither yielded nor descended into — callers read their value
    from their own memo instead.  This is the memoized-walk contract
    ``typing._shape_of``/``has_var_leaf`` use to keep shared-subterm
    DAGs linear.
    """
    seen: set = set()
    # (node, done) frames; children pushed in arg order so the
    # rightmost pops first — the _dag_postorder convention.
    work: list[tuple[Any, bool]] = [(term, False)]
    while work:
        node, done = work.pop()
        if done:
            yield node
            continue
        if not _fresh(seen, node):
            continue
        if prune is not None and prune(node):
            continue
        work.append((node, True))
        work.extend((c, False) for c in _children(node))


def expand(
    term: Any,
    depth: int | None = None,
    budget: int | None = None,
) -> Iterator[tuple[tuple, Any]]:
    """Yield ``(path, node)`` for the *tree spelling* of *term*.

    Pre-order, left-to-right, one position per occurrence — a shared
    subterm appears once per use site, exactly like the historical
    recursive ``_term_paths``/``_iter_ops`` walks (filter ``Op`` nodes
    for the latter).  ``path`` is the tuple of child indices from the
    root (``()`` for *term* itself).

    Bounds (either may be ``None`` for unbounded — which on a
    sharing-heavy DAG is the exponential tree size, so *prefer*
    ``budget``):

    * ``depth`` — maximum path length in edges; positions deeper than
      ``depth`` are not visited (``depth=0`` yields only the root).
    * ``budget`` — maximum number of positions yielded; iteration
      stops once it is exhausted.

    The walk is an explicit stack: deep unshared chains cannot raise
    ``RecursionError``.  No dedup — this is the occurrence view;
    :func:`subterms`/``ops``/``leaves`/``postorder` are the DAG view.
    """
    yielded = 0
    stack: list[tuple[tuple, Any]] = [((), term)]
    while stack:
        if budget is not None and yielded >= budget:
            return
        path, node = stack.pop()
        yield path, node
        yielded += 1
        if depth is not None and len(path) >= depth:
            continue
        if isinstance(node, Op):
            stack.extend(
                reversed(
                    [((*path, i), a) for i, a in enumerate(node.args)]
                )
            )
