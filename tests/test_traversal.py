"""Tests for ``mlopt_core.traversal`` — the shared DAG walkers."""

from __future__ import annotations

from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.traversal import (
    expand,
    leaves,
    ops,
    postorder,
    subterms,
)


def _v(name: str = "x") -> Var:
    return Var(name, TensorType((2, 3)))


def _chain(depth: int, leaf: Var | None = None) -> Op:
    """Build ``neg(neg(...(leaf)))`` iteratively — no recursion."""
    t = leaf if leaf is not None else _v()
    for _ in range(depth):
        t = Op.make("neg", t)
    return t


# ---------------------------------------------------------------------------
#  DAG dedup — a shared subterm is counted once
# ---------------------------------------------------------------------------


def test_subterms_dedups_shared_subterm():
    x = _v()
    f = Op.make("neg", x)
    t = Op.make("add", f, f)
    nodes = list(subterms(t))
    assert nodes.count(f) == 1
    assert nodes.count(x) == 1
    assert len(nodes) == 3  # add, neg, x — f counted once


def test_ops_dedups_shared_subterm():
    x = _v()
    f = Op.make("neg", x)
    g = Op.make("exp", f)
    t = Op.make("add", g, g)
    assert list(ops(t)) == [t, g, f] or set(ops(t)) == {t, g, f}
    assert len(list(ops(t))) == 3


def test_leaves_dedups_and_covers_non_op_root():
    x, y = _v("x"), _v("y")
    t = Op.make("add", x, Op.make("mul", y, x))
    assert set(leaves(t)) == {x, y}
    # A bare leaf root is its own leaf set.
    assert list(leaves(x)) == [x]
    assert list(ops(x)) == []
    assert list(subterms(x)) == [x]
    assert list(postorder(x)) == [x]


def test_postorder_children_before_parents():
    x = _v()
    f = Op.make("neg", x)
    t = Op.make("add", f, Op.make("exp", f))
    order = list(postorder(t))
    # Every node appears after all of its Op descendants.
    pos = {n: i for i, n in enumerate(order)}
    for n in order:
        if isinstance(n, Op):
            for a in n.args:
                if isinstance(a, Op):
                    assert pos[a] < pos[n]
    # Deduped: f once.
    assert order.count(f) == 1
    assert len(order) == 4  # add, neg, exp, x


# ---------------------------------------------------------------------------
#  Deep terms — no RecursionError
# ---------------------------------------------------------------------------


def test_deep_term_no_recursion_error():
    depth = 100_000
    t = _chain(depth)
    # Every walker completes without RecursionError.
    assert len(list(postorder(t))) == depth + 1  # negs + Var leaf
    assert len(list(subterms(t))) == depth + 1
    assert len(list(ops(t))) == depth
    # spot-check the leaf is present
    assert any(isinstance(n, Var) for n in subterms(t))


def test_deep_term_expand_budget():
    t = _chain(50_000)
    # Unbounded tree spelling of a chain is linear; budget caps it.
    paths = list(expand(t, budget=10))
    assert len(paths) == 10
    assert paths[0] == ((), t)
    assert paths[1][0] == (0,)


# ---------------------------------------------------------------------------
#  expand — tree spelling, paths, depth, budget
# ---------------------------------------------------------------------------


def test_expand_tree_spelling_counts_occurrences():
    x = _v()
    f = Op.make("neg", x)
    t = Op.make("add", f, f)
    positions = list(expand(t))
    # Tree spelling: f appears twice (once per use site).
    assert [n for _p, n in positions].count(f) == 2
    paths = [p for p, _n in positions]
    assert paths == [(), (0,), (0, 0), (1,), (1, 0)]


def test_expand_depth_bound():
    x = _v()
    f = Op.make("neg", x)
    t = Op.make("add", f, f)
    positions = list(expand(t, depth=1))
    # root + its two direct children; grandchildren pruned.
    assert [p for p, _n in positions] == [(), (0,), (1,)]
    positions = list(expand(t, depth=0))
    assert [p for p, _n in positions] == [()]


def test_expand_budget_on_exponential_dag():
    """``t_{i+1} = add(t_i, t_i)`` — tree size 2**n; budget caps it."""
    x = _v()
    t: Op | Var = x
    for _ in range(20):
        t = Op.make("add", t, t)
    positions = list(expand(t, budget=1000))
    assert len(positions) == 1000
    # DAG walkers stay linear on the same term.
    assert len(list(ops(t))) == 20
    assert len(list(postorder(t))) == 21  # 20 ops + Var


def test_postorder_prune_frontier():
    """``prune`` stops the walk at already-computed nodes."""
    x = _v()
    f = Op.make("neg", x)
    g = Op.make("exp", f)
    t = Op.make("add", g, f)
    # With f pruned, x is unreachable through f — but x is also not
    # under g... it IS under g (g->f->x).  Prune f: x never yielded.
    pruned = list(postorder(t, prune=lambda n: n is f))
    assert f not in pruned
    assert x not in pruned  # only reachable via f
    assert g in pruned and t in pruned


def test_postorder_matches_dag_postorder_convention():
    """Right-to-left child order — the ``_dag_postorder`` contract."""
    from mlopt_core.ir import _dag_postorder

    x = _v()
    a = Op.make("neg", x)
    b = Op.make("exp", x)
    c = Op.make("sqrt", x)
    t = Op.make("add", Op.make("mul", a, b), Op.make("div", c, a))
    ops_only = [n for n in postorder(t) if isinstance(n, Op)]
    assert ops_only == _dag_postorder(t)
