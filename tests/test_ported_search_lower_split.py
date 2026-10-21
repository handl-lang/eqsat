"""Substrate-pure tests ported from ``graphopt:tests/test_search_lower_split.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import pytest

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Op, Param, TensorType, Var

from mlopt_core.pipeline import LowerResult, SearchResult


class _NoPolicyEngine:
    """An engine shaped like the native one: ``run()`` takes no policy."""

    def run(self, rules, root_eid, max_iterations=100):
        """Return empty stats — never actually run."""
        return {}

class _OpaqueEngine:
    """An engine whose ``run`` has no introspectable signature."""

    run = dict.get

def test_policy_kwargs_go_only_to_engines_that_take_them():
    """A policy-less engine gets a clear error, not a bare TypeError."""
    from mlopt_core.policies import GreedyPolicy
    from mlopt_orchestrator.optimize import _policy_kwargs

    policy = GreedyPolicy()
    # no policy asked for: no kwargs, whatever the engine
    assert _policy_kwargs(None, _NoPolicyEngine()) == {}
    # a policy-less engine is named in the error
    with pytest.raises(TypeError, match="does not accept a policy"):
        _policy_kwargs(policy, _NoPolicyEngine())
    # an un-introspectable run() cannot be confirmed either
    with pytest.raises(TypeError, match="does not accept a policy"):
        _policy_kwargs(policy, _OpaqueEngine())
    # the reference engine accepts it
    assert _policy_kwargs(policy, EGraph()) == {"policy": policy}

def test_search_result_flops_fallback():
    """A hand-built SearchResult with no cost_fn prices alternatives
    under flops_cost."""
    from mlopt_core.egraph import EGraph as _EG

    x = Var("x", TensorType((3,)))
    w = Param(name="p_w", typ=TensorType((3,)))
    term = Op.make("add", x, w)
    eg = _EG()
    eid = eg.add_term(term)
    res = SearchResult(
        ir=IR(root=term, inputs=[x], input_names={"x"}, params={}),
        eg=eg,
        root_eid=eid,
        term=term,
        param_values={},
        stats={},
    )
    alts = res.alternatives()
    assert alts == [] or isinstance(alts[0][0], float)
    assert res.model is None and res.source is None
