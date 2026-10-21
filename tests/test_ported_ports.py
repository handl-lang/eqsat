"""Substrate-pure tests ported from ``graphopt:tests/test_ports.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.cost import (
    CostModel,
    count_cost,
    dag_cost,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost,
    param_bytes_cost_for,
)

from mlopt_core.ops import OpTable

from mlopt_core.ports import (
    BatchedExecutor,
    Binding,
    CostFn,
    Executor,
    OpRegistry,
    PlannedExecutor,
    RuleLike,
    RuleSetLike,
    RuleSetProvider,
    ShapeRule,
    Sink,
    Source,
    TorchBinding,
    Verifier,
    signature_conforms,
)

from mlopt_core.laws import CATEGORICAL, CATEGORICAL_RULES, all_rules

from mlopt_core.typing import _SHAPE_RULES


def test_builtin_cost_models_are_cost_fns():
    for fn in (
        flops_cost,
        launch_aware_cost,
        count_cost,
        param_bytes_cost,
        param_bytes_cost_for(),
        CostModel(),
    ):
        assert isinstance(fn, CostFn), fn
        assert signature_conforms(fn, CostFn), fn

def test_cost_fn_wrong_signature_rejected():
    """A callable requiring args the extract sites never supply is not
    a CostFn — ``isinstance`` can't see this, ``signature_conforms``
    can."""
    assert not signature_conforms(lambda: 0.0, CostFn)
    assert not signature_conforms(lambda a, b, c: 0.0, CostFn)
    # dag_cost is a meta-wrapper: needs a cost_fn argument of its own.
    assert not signature_conforms(dag_cost, CostFn)
    assert not isinstance(object(), CostFn)

def test_verifier_wrong_signature_rejected():
    # Call sites always pass rtol/atol by name — strict probe.
    assert not signature_conforms(lambda a, b: None, Verifier, strict=True)
    assert not signature_conforms(object(), Verifier)

def test_optable_conforms_to_registry():
    assert isinstance(OpTable.core(), OpRegistry)
    assert isinstance(OpTable.full(), OpRegistry)
    assert not isinstance({}, OpRegistry)

def test_shape_rules_conform():
    for name in ("om_elem", "aff", "apply", "eye"):
        assert isinstance(_SHAPE_RULES[name], ShapeRule)
        assert signature_conforms(_SHAPE_RULES[name], ShapeRule)
    assert not signature_conforms(lambda a: None, ShapeRule)

def test_rule_collections_conform():
    assert isinstance(CATEGORICAL_RULES, RuleSetLike)
    assert isinstance(all_rules(), RuleSetLike)
    assert isinstance(CATEGORICAL, RuleSetLike)
    assert all(isinstance(r, RuleLike) for r in CATEGORICAL_RULES)
    assert isinstance(all_rules, RuleSetProvider)
    # presence-level: a dict IS iterable (RuleSetLike) — documented caveat.
    assert not isinstance(42, RuleSetLike)
    assert not isinstance(42, RuleSetProvider)
    assert not isinstance(object(), RuleLike)
