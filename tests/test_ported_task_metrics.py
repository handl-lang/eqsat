"""Substrate-pure tests ported from ``graphopt:tests/test_task_metrics.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core.metrics import (
    ArgmaxStability,
    CosineRanking,
    LogitKL,
    MaxRel,
    TopKAgreement,
    _RowMetric,
)

from mlopt_core.ports import TaskMetric

from mlopt_orchestrator import optimize as opt_mod

from mlopt_orchestrator.optimize import (
    Compositional,
    Optimizer,
    _task_contract,
)


def test_metrics_conform_to_taskmetric_port():
    for m in (
        MaxRel(),
        TopKAgreement(3),
        ArgmaxStability(),
        LogitKL(),
        CosineRanking(),
    ):
        assert isinstance(m, TaskMetric)
        assert isinstance(m.name, str) and m.name
        assert isinstance(m.tolerance, float)

def test_metric_names_and_defaults():
    assert MaxRel().name == "max_rel"
    assert MaxRel().tolerance == 1e-4
    assert TopKAgreement(5).name == "top5_agreement"
    assert ArgmaxStability().name == "argmax_stability"
    assert LogitKL().name == "logit_kl"
    assert CosineRanking().name == "cosine_ranking"
    with pytest.raises(ValueError, match="k >= 1"):
        TopKAgreement(0)

def test_rowmetric_base_dist_is_abstract():
    with pytest.raises(NotImplementedError):
        _RowMetric()._dist([1.0], [1.0])

def test_maxrel_matches_verify_equiv_semantics():
    """``max|a-b| / (max|a| + 1e-8)`` — the historical gate, restated."""
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[1.0, 2.0], [3.0, 4.5]]
    assert MaxRel().distance(a, b) == pytest.approx(0.5 / 4.0)
    assert MaxRel().distance(a, a) == 0.0
    # Empty output set: nothing to compare — distance 0.
    assert MaxRel().distance([], []) == 0.0
    # Empty tuple output: no leaf pairs at all.
    assert MaxRel().distance((), ()) == 0.0

def test_argmax_stability():
    a = [[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]]
    same = [[1.1, 3.0, 2.0], [4.1, 1.0, 0.0]]
    flipped = [[3.0, 1.0, 2.0], [0.0, 1.0, 4.0]]
    assert ArgmaxStability().distance(a, same) == 0.0
    assert ArgmaxStability().distance(a, flipped) == 1.0
    half = [[3.0, 1.0, 2.0], [4.0, 1.0, 0.0]]
    assert ArgmaxStability().distance(a, half) == 0.5
    # Ties break on the first index (torch semantics).
    assert ArgmaxStability().distance([[1.0, 1.0]], [[1.0, 1.0]]) == 0.0
    assert ArgmaxStability().distance([], []) == 0.0

def test_topk_agreement():
    a = [[10.0, 9.0, 8.0, 7.0]]
    full = [[10.1, 9.0, 8.0, 7.0]]
    swap = [[7.0, 8.0, 9.0, 10.0]]
    # k=1 is argmax agreement; k=4 is the full index set.
    assert TopKAgreement(1).distance(a, full) == 0.0
    assert TopKAgreement(1).distance(a, swap) == 1.0
    assert TopKAgreement(4).distance(a, swap) == 0.0
    # k=2 partial overlap: one of {0,1} survives the swap.
    assert TopKAgreement(2).distance(a, swap) == 1.0
    mid = [[10.0, 7.0, 9.0, 8.0]]  # top-2 {0,2} vs ref {0,1} → 1/2
    assert TopKAgreement(2).distance(a, mid) == 0.5
    # k clamped to the row width.
    assert TopKAgreement(9).distance(a, swap) == 0.0
    assert TopKAgreement(9).distance(a, a) == 0.0
    assert TopKAgreement(2).distance([], []) == 0.0
    # Deterministic tie-break on equal values: index order wins.
    tied = [[5.0, 5.0, 5.0]]
    assert TopKAgreement(2).distance(tied, tied) == 0.0

def test_logit_kl():
    a = [[3.0, 1.0, 0.0]]
    assert LogitKL().distance(a, a) == pytest.approx(0.0)
    close = [[3.01, 1.0, 0.0]]
    d = LogitKL().distance(a, close)
    assert 0.0 < d < 1e-3  # stays inside the shipped default tolerance
    hard = [[-3.0, 1.0, 3.0]]
    assert LogitKL().distance(a, hard) > 0.1
    # q masses ~0 where p masses 1 → the row reports inf.
    starved = [[-1e3, 0.0, 1e3]]
    assert LogitKL().distance([[1e3, 0.0, 0.0]], starved) == float(
        "inf"
    )
    # A p~0 coordinate is skipped (pi <= 0 continue arm).
    assert LogitKL().distance([[-1e3, 1.0]], [[1.0, -1e3]]) == float(
        "inf"
    )
    same_shift = [[-1e3, 1.0]]
    assert LogitKL().distance(same_shift, same_shift) == pytest.approx(
        0.0
    )
    assert LogitKL().distance([], []) == 0.0
    # distance is the MAX over rows — the worst row decides.
    two = [[3.0, 1.0, 0.0], [1.0, 2.0, 3.0]]
    one_bad = [[3.01, 1.0, 0.0], [3.0, 2.0, 1.0]]
    assert LogitKL().distance(two, one_bad) == pytest.approx(
        LogitKL().distance([[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]])
    )

def test_cosine_ranking():
    # Preserved ordering → 0; a row swap that inverts the cosine
    # ranking → positive distance.
    emb = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.0]]
    assert CosineRanking().distance(emb, emb) == 0.0
    swapped = [emb[3], emb[1], emb[2], emb[0]]
    assert CosineRanking().distance(emb, swapped) > 0.0
    # Fewer than 3 rows admit no pair ordering.
    assert CosineRanking().distance([[1.0]], [[2.0]]) == 0.0
    assert CosineRanking().distance([], []) == 0.0
    # Fully tied cosine rows: every pair is uninformative → 0.
    tied = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert CosineRanking().distance(tied, tied) == 0.0
    # A zero vector has no cosine direction — handled, not NaN.
    withzero = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert CosineRanking().distance(withzero, withzero) == 0.0

def test_output_pytree_pairing():
    m = ArgmaxStability()
    # tuple outputs pair positionally; the worst leaf decides.
    d = m.distance(([[0.0, 1.0]], [[1.0, 0.0]]), ([[0.0, 1.0]], [[0.0, 1.0]]))
    assert d == 1.0
    # dict outputs pair by key.
    d = m.distance(
        {"logits": [[0.0, 1.0]], "aux": [[1.0, 0.0]]},
        {"logits": [[0.0, 1.0]], "aux": [[0.0, 1.0]]},
    )
    assert d == 1.0
    # Structural mismatches are loud.
    with pytest.raises(TypeError, match="keys differ"):
        m.distance({"a": [[1.0]]}, {"b": [[1.0]]})
    with pytest.raises(TypeError, match="arities differ"):
        m.distance(([[1.0]], [[1.0]]), ([[1.0]],))
    # A dict against a non-dict leaf is a metric failure, not a match.
    with pytest.raises(TypeError, match="tensor-like"):
        m.distance({"a": 1}, [[1.0]])

def test_metric_rows_rejections():
    m = MaxRel()
    with pytest.raises(TypeError, match="tensor-like"):
        m.distance(3.0, 3.0)
    with pytest.raises(TypeError, match="tensor-like"):
        m.distance([[1.0, "x"]], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="row counts"):
        m.distance([[1.0], [2.0]], [[1.0]])
    with pytest.raises(ValueError, match="row widths"):
        m.distance([[1.0, 2.0]], [[1.0]])

def test_task_contract_validation():
    assert _task_contract(None, None) is None
    with pytest.raises(TypeError, match="requires a task="):
        _task_contract(None, 0.5)

    class NoDistance:
        tolerance = 0.0

    with pytest.raises(TypeError, match="no callable distance"):
        _task_contract(NoDistance(), None)

    class NoTol:
        name = "duck"
        tolerance = None

        def distance(self, a, b):
            return 0.0

    with pytest.raises(TypeError, match="task_tol"):
        _task_contract(NoTol(), None)
    # An explicit task_tol rescues a tolerance-less metric.
    assert _task_contract(NoTol(), 0.25) == ("duck", 0.25)
    # The metric's own tolerance is the default; task_tol overrides.
    assert _task_contract(TopKAgreement(1), None) == (
        "top1_agreement",
        0.0,
    )
    assert _task_contract(ArgmaxStability(), 0.5) == (
        "argmax_stability",
        0.5,
    )

def test_accepted_by_helpers():
    """The small verdict helpers — bound/pointwise/declined arms."""

    class R:
        def __init__(self, passed):
            self.passed = passed

    assert opt_mod._accepted_by({}, R(False)) == "declined"
    assert opt_mod._accepted_by({"task": {"passed": True}}, R(True)) == "task"
    assert (
        opt_mod._accepted_by({"error_bound_total": 1e-3}, R(True))
        == "bound"
    )
    assert opt_mod._accepted_by({}, R(True)) == "pointwise"
    assert opt_mod._verify_metric(None, 0.0) == "max_rel"
    assert opt_mod._verify_metric(None, 1e-3) == "bound"
    assert opt_mod._verify_metric(("top1_agreement", 0.0), 0.0) == (
        "top1_agreement"
    )
