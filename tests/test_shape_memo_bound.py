"""Bounded shape memos — audit item #26 (``laws.base._SHAPE_MEMO``).

The module-global shape memos in ``mlopt_core.laws`` (and the two
matching ones in ``mlopt_orchestrator`` — ``regime`` /
``morphisms_kv``) are shared across
every check-hook/cond evaluation for the process lifetime, so they
must be bounded :class:`~mlopt_core.egraph.extract._CostMemo` maps —
eviction is safe because every entry is a pure function of its
content-keyed term.
"""

from collections import OrderedDict

from mlopt_core.egraph.extract import _CostMemo
from mlopt_core.ir import Const, Op
from mlopt_core.laws import base, cond


def test_shape_memo_is_bounded_lru():
    for memo in (base._SHAPE_MEMO, cond._MEMO):
        assert isinstance(memo, _CostMemo)
        assert isinstance(memo, OrderedDict)
        assert memo._limit > 0


def test_shape_memo_eviction_recomputes_identically():
    """A tiny stand-in memo proves eviction is semantics-preserving."""
    saved = base._SHAPE_MEMO
    tiny = _CostMemo(2)
    base._SHAPE_MEMO = tiny
    try:
        t1 = Op.make("add", Const(1.0), Const(2.0))
        t2 = Op.make("mul", Const(3.0), Const(4.0))
        first = base._shape_of(t1)
        # Each _shape_of call memoises every node under its term, so a
        # limit-2 memo evicts constantly — the bound is what matters.
        assert len(tiny) <= 2
        base._shape_of(t2)  # exercised, no raise
        # t1 was evicted by t2's walk; recomputation gives the same value.
        assert base._shape_of(t1) == first
        assert len(tiny) <= 2
    finally:
        base._SHAPE_MEMO = saved


def test_orchestrator_shape_memos_are_bounded():
    """The orchestrator's two ``_SHAPE_MEMO`` maps share the same contract."""
    from mlopt_orchestrator import morphisms_kv, regime

    for memo in (regime._SHAPE_MEMO, morphisms_kv._SHAPE_MEMO):
        assert isinstance(memo, _CostMemo)
        assert isinstance(memo, OrderedDict)
        assert memo._limit > 0


def test_regime_shape_memo_eviction_recomputes_identically():
    """A tiny stand-in memo proves eviction is semantics-preserving."""
    from mlopt_orchestrator import regime

    saved = regime._SHAPE_MEMO
    tiny = _CostMemo(2)
    regime._SHAPE_MEMO = tiny
    try:
        t1 = Op.make("add", Const(1.0), Const(2.0))
        t2 = Op.make("mul", Const(3.0), Const(4.0))
        first = regime.footprint_cost(t1)
        regime.footprint_cost(t2)  # exercised, no raise
        # t1 was evicted by t2's walk; recomputation gives the same value.
        assert regime.footprint_cost(t1) == first
        assert len(tiny) <= 2
    finally:
        regime._SHAPE_MEMO = saved
