"""Substrate-pure tests ported from ``graphopt:tests/test_factored_params.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.cost import (
    dag_cost,
    executor_cost_for,
    flops_cost,
    launch_aware_cost,
)

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Op, Param, TensorType, Var, op_repr

from mlopt_core.laws import (
    ALL_RULES,
    ASSOC_LINEAR,
    ASSOC_LINEAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
)

from mlopt_core.laws.factored import (
    _cls_has_var,
    _factor_cols,
    _factor_rows,
    _fresh_name,
    _leaf_params,
    offer_low_rank_factors,
)


def _mm(x, a, b):
    """``(x @ A) @ B`` — the factored spelling."""
    return Op.make("matmul", Op.make("matmul", x, a), b)

def _mm_mat(x, a, b):
    """``x @ (A @ B)`` — the materialised spelling."""
    return Op.make("matmul", x, Op.make("matmul", a, b))

def _assoc_run(term, iters=8):
    """Saturate with just the assoc laws and return (eg, eid)."""
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        [
            ASSOC_MATMUL,
            ASSOC_MATMUL_REV,
            ASSOC_LINEAR,
            ASSOC_LINEAR_REV,
        ],
        eid,
        max_iterations=iters,
    )
    return eg, eid

def test_matmul_assoc_picks_factored_low_rank():
    """``x@(A@B)`` and ``(x@A)@B`` unify; flops pick the chain at r≪d."""
    m, i, r, o = 8, 64, 4, 128
    x = Var("x", TensorType((m, i)))
    a = Param("A", TensorType((i, r)))
    b = Param("B", TensorType((r, o)))
    for seed_term in (_mm(x, a, b), _mm_mat(x, a, b)):
        eg, eid = _assoc_run(seed_term)
        best = eg.extract_best(eid, flops_cost)
        assert best == _mm(x, a, b)
        # The flop break-even is r·(i+o) < i·o — here 4·192 < 8192.
        assert dag_cost(best, flops_cost) < dag_cost(
            _mm_mat(x, a, b), flops_cost
        )

def test_matmul_assoc_picks_materialized_high_rank():
    """At r ≥ i·o/(i+o) the dense product is the honest winner."""
    m, i, r, o = 8, 64, 48, 128  # r·(i+o)=9216 ≥ 8192
    x = Var("x", TensorType((m, i)))
    a = Param("A", TensorType((i, r)))
    b = Param("B", TensorType((r, o)))
    eg, eid = _assoc_run(_mm(x, a, b))
    best = eg.extract_best(eid, flops_cost)
    assert best == _mm_mat(x, a, b)

def test_assoc_linear_rev_splits_fused_weight():
    """``linear(x, B@A)`` reaches the factored member — the gap the
    new ``assoc_linear_rev`` law closes (``assoc_linear`` alone is
    one-directional)."""
    m, i, r, o = 8, 64, 4, 128
    x = Var("x", TensorType((m, i)))
    a = Param("lora_A", TensorType((r, i)))  # HF convention names
    b = Param("lora_B", TensorType((o, r)))
    fused = Op.make("linear", x, Op.make("matmul", b, a))
    eg = EGraph()
    eid = eg.add_term(fused)
    # Forward alone cannot reach the chain.
    eg.run([ASSOC_LINEAR], eid, max_iterations=4)
    assert eg.extract_best(eid, flops_cost) == fused
    # With the reverse law it can — and flops select it.
    eg, eid = _assoc_run(fused)
    assert eg.extract_best(eid, flops_cost) == Op.make(
        "linear", Op.make("linear", x, a), b
    )

def test_linear_chain_fuses_and_splits_both_directions():
    """``linear(linear(x,A),B)`` keeps both members; flops pick split."""
    m, i, r, o = 8, 64, 4, 128
    x = Var("x", TensorType((m, i)))
    a = Param("lora_A", TensorType((r, i)))
    b = Param("lora_B", TensorType((o, r)))
    chain = Op.make("linear", Op.make("linear", x, a), b)
    eg, eid = _assoc_run(chain)
    best = eg.extract_best(eid, flops_cost)
    assert best == chain
    # The fused member was generated — it just loses on flops.
    assert eg.rule_fires.get("assoc_linear")
    fused = Op.make("linear", x, Op.make("matmul", b, a))
    assert dag_cost(fused, flops_cost) > dag_cost(chain, flops_cost)

def test_executor_cost_launch_overhead_crossover():
    """The default pipeline model is honest, not flops-blind: at tiny
    dims the extra kernel launch beats the flop saving and the fused
    single GEMM wins; once the work is real, factored wins.  This is
    the documented regime boundary — selection is cost-driven, so the
    'factored path' is *where the cost model lands*."""
    cf = executor_cost_for(lowering="generic")
    x = Var("x", TensorType((8, 64)))
    a = Param("A", TensorType((64, 4)))
    b = Param("B", TensorType((4, 128)))
    assert dag_cost(_mm_mat(x, a, b), cf) < dag_cost(_mm(x, a, b), cf)
    # Compute-bound regime: the same rank at d=1024/2048 flips the
    # verdict — two launches no longer dominate ~10x fewer flops.
    x2 = Var("x", TensorType((64, 1024)))
    a2 = Param("A2", TensorType((1024, 64)))
    b2 = Param("B2", TensorType((64, 2048)))
    assert dag_cost(_mm(x2, a2, b2), cf) < dag_cost(
        _mm_mat(x2, a2, b2), cf
    )
    # ...and extraction agrees.
    eg, eid = _assoc_run(_mm_mat(x2, a2, b2))
    assert eg.extract_best(eid, cf) == _mm(x2, a2, b2)

def test_lora_flop_arithmetic():
    """Document the break-even on the headline fixture.

    ``Linear(d, 2d)`` adapter, rank r, batch M: materialised adapter
    costs ``2·M·d·2d`` flops; factored ``x@A@B`` costs
    ``2·M·r·d + 2·M·r·2d = 2·M·r·3d``.  Factored wins iff
    ``r < 2d/3`` — the ``r·(i+o) < i·o`` break-even, far more generous
    than the 'r < ~d/4' rule of thumb.  (Launch overhead tightens the
    real crossover — see ``test_executor_cost_launch_overhead_crossover``.)
    """
    d, r, m = 256, 4, 32
    x = Var("x", TensorType((m, d)))
    a = Param("lora_A", TensorType((d, r)))
    b = Param("lora_B", TensorType((r, 2 * d)))
    fac, mat = _mm(x, a, b), _mm_mat(x, a, b)
    f_fac, f_mat = dag_cost(fac, flops_cost), dag_cost(mat, flops_cost)
    assert f_fac == 2 * m * r * (d + 2 * d)
    assert f_mat == 2 * m * d * 2 * d  # inner A@B folds: compile-time
    assert f_fac / f_mat == r * 3 * d / (d * 2 * d)
    assert f_fac < f_mat  # r=4 ≪ 2d/3 ≈ 170

def test_lora_unscaled_adapter_merges_into_base():
    """``x@W + x@A@B`` (no scaling): the flop-optimal extraction IS
    the merge — ``linear(x, W + B@A)`` materialises one weight at
    compile time and drops the adapter's flops entirely.  Folding
    ``A·B`` into ``W_base`` is what the cost model prefers here, so
    it folds — verified numerically identical on delivery."""
    d, r, o, m = 32, 4, 64, 8
    x = Var("x", TensorType((m, d)))
    w = Param("p_w", TensorType((o, d)))
    a = Param("lora_A", TensorType((r, d)))
    b = Param("lora_B", TensorType((o, r)))
    term = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make("linear", Op.make("linear", x, a), b),
    )
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(ALL_RULES, eid, max_iterations=8)
    best = eg.extract_best(eid, launch_aware_cost)
    assert best == Op.make(
        "linear", x, Op.make("add", w, Op.make("matmul", b, a))
    )
    assert dag_cost(best, flops_cost) < dag_cost(term, flops_cost)

def test_lora_scaled_adapter_stays_factored():
    """HF spelling ``x@W + s·(x@A@B)``: the multiplicative scale sits
    between the merge pattern's operands, so the adapter branch keeps
    a factored chain — the scale gets pushed inside instead."""
    d, r, o, m = 32, 4, 64, 8
    x = Var("x", TensorType((m, d)))
    w = Param("p_w", TensorType((o, d)))
    a = Param("lora_A", TensorType((r, d)))
    b = Param("lora_B", TensorType((o, r)))
    s = Param("lora_s", TensorType((1,)))
    term = Op.make(
        "add",
        Op.make("linear", x, w),
        Op.make(
            "mul",
            Op.make("linear", Op.make("linear", x, a), b),
            s,
        ),
    )
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(ALL_RULES, eid, max_iterations=8)
    best = eg.extract_best(eid, launch_aware_cost)
    # Still a two-stage chain under the scale — never collapsed to
    # one dense (o, i) GEMM.
    assert "matmul" not in op_repr(best)
    assert dag_cost(best, flops_cost) <= dag_cost(term, flops_cost)

def test_leaf_params_and_cls_has_var():
    """Class-level leaf/var reachability helpers."""
    x = Var("x", TensorType((8, 32)))
    p = Param("p", TensorType((64, 32)))
    eg = EGraph()
    xid = eg.add_term(x)
    pid = eg.add_term(p)
    tid = eg.add_term(Op.make("linear", x, p))
    leaves = _leaf_params(eg, pid)
    assert leaves == [p]
    # A Var-leaf class yields no Param members; a class with only an
    # op member is walked past it.
    assert _leaf_params(eg, xid) == []
    assert _leaf_params(eg, eg.find(tid)) == []
    memo: dict[int, bool] = {}
    has_var = _cls_has_var(eg, memo)
    assert has_var(tid, frozenset())
    assert not has_var(pid, frozenset())
    # Memoised hit and the cycle guard (on a class never queried —
    # the memo lookup precedes the stack check).
    p2 = Param("p2", TensorType((4,)))
    p2id = eg.add_term(p2)
    assert has_var(tid, frozenset())  # memo hit
    assert not has_var(p2id, frozenset({p2id}))

def test_op_repr_smoke():
    """op_repr keeps the factored/member distinction readable."""
    x = Var("x", TensorType((8, 32)))
    a = Param("A", TensorType((32, 4)))
    b = Param("B", TensorType((4, 64)))
    assert "matmul" in op_repr(_mm(x, a, b))
