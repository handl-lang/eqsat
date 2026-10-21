"""Substrate-pure tests ported from ``graphopt:tests/test_meta.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core import meta

from mlopt_core import laws as R

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var, op_repr


def _T(n=4):
    return TensorType((n, n))

def _opdepth(t, memo):
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max(
            (_opdepth(a, memo) for a in t.args), default=0
        )
    return memo[k]

def _recurrence_seed():
    """Two raw steps with DISTINCT leaves (maximally general seed)."""
    A1, A2 = Param("A1", _T()), Param("A2", _T())
    h0, u, x = Var("h0", _T()), Var("u", _T()), Var("x", _T())
    seed = Op.make(
        "add",
        Op.make(
            "matmul", A2, Op.make("add", Op.make("matmul", A1, h0), u)
        ),
        x,
    )
    return seed, A1, A2, h0, u, x

def test_classification_partitions_all_rules():
    rules = meta.module_rules(R)
    coherent, contentful = meta.classify_rules(rules)
    assert len(coherent) + len(contentful) == len(rules)
    assert {r.name for r in coherent} <= meta.COHERENT_RULE_NAMES
    # the spec'd coherent members are actually present in rules.py
    for name in (
        "comm_add",
        "comm_mul",
        "assoc_add",
        "assoc_mul",
        "id_add",
        "id_mul",
        "double_neg",
        "assoc_matmul",
        "assoc_matmul_rev",
        "aff_assoc",
        "aff_assoc_rev",
    ):
        assert name in {r.name for r in coherent}
    # contentful spot checks
    cnames = {r.name for r in contentful}
    for name in (
        "distribute_matmul_over_add",
        "factor_matmul",
        "weight_factor_matmul",
        "aff_lift",
        "aff_lift_step",
        "aff_unlift",
        "aff_compose_unfold",
        "sub_to_add",
        "silu_expand",
    ):
        assert name in cnames

def test_canonicalize_collapses_coherent_variants():
    a, b, c = Var("a", _T()), Var("b", _T()), Var("c", _T())
    t1 = Op.make("add", Op.make("add", a, b), c)
    t2 = Op.make("add", c, Op.make("add", b, a))
    t3 = Op.make(
        "add", Op.make("add", a, Const(0.0)), Op.make("add", b, c)
    )
    canon = {op_repr(meta.canonicalize(t)) for t in (t1, t2, t3)}
    assert len(canon) == 1  # comm+assoc+id all collapse to one form

def test_canonicalize_identity_and_involution():
    x = Var("x", _T())
    assert meta.canonicalize(Op.make("add", x, Const(0.0))) == x
    assert meta.canonicalize(Op.make("add", Const(0.0), x)) == x
    assert meta.canonicalize(Op.make("mul", x, Const(1.0))) == x
    nn = Op.make("neg", Op.make("neg", x))
    assert meta.canonicalize(nn) == x
    # add of only-zeros collapses to the identity
    assert meta.canonicalize(
        Op.make("add", Const(0.0), Const(0.0))
    ) == Const(0.0)

def test_canonicalize_assoc_only_preserves_order():
    """matmul is associative but NOT commutative: order must survive."""
    a, b, c = Var("a", _T()), Var("b", _T()), Var("c", _T())
    left = Op.make("matmul", Op.make("matmul", a, b), c)
    right = Op.make("matmul", a, Op.make("matmul", b, c))
    swapped = Op.make("matmul", Op.make("matmul", c, b), a)
    assert op_repr(meta.canonicalize(left)) == op_repr(
        meta.canonicalize(right)
    )
    assert op_repr(meta.canonicalize(swapped)) != op_repr(
        meta.canonicalize(left)
    )

def test_canonicalize_rebalances_compose_chain():
    """A right-leaning aff_compose chain canonicalizes to the balanced
    (Blelloch) bracketing — coherence as a computed canonical form."""
    A = Param("A", _T())
    leaves = [Op.make("aff", A, Var(f"x{i}", _T())) for i in range(4)]
    right = leaves[0]
    for lf in leaves[1:]:
        right = Op.make("aff_compose", lf, right)
    canon = meta.canonicalize(right)
    # balanced tree over 4 leaves has depth 3 (aff_compose levels = 2)
    assert _opdepth(canon, {}) == _opdepth(leaves[0], {}) + 2
    # and it is genuinely balanced: root children are both composes
    assert canon.op == "aff_compose"
    assert all(a.op == "aff_compose" for a in canon.args)

def test_synthesis_rediscovers_lift_step_composite():
    """SCAN_LAWS minus aff_lift_step + a 2-step seed re-derives the
    unfolded equivalent: two raw steps -> nested affine applies."""
    seed, A1, A2, h0, u, x = _recurrence_seed()
    rules = [r for r in R.SCAN_LAWS if r.name != "aff_lift_step"]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)

    # some derived rule fires on the raw 2-step term at the root...
    hits = [
        d
        for d in derived
        if meta.match_pattern(d.lhs, seed, {}) is not None
    ]
    assert hits, "no derived rule matches the two-step seed"

    # ...and one of them produces the nested-apply fused form —
    # exactly what aff_lift_step + aff_compose_unfold would yield.
    expected = Op.make(
        "apply",
        Op.make("aff", A2, x),
        Op.make("apply", Op.make("aff", A1, u), h0),
    )
    found = False
    for d in hits:
        subst = meta.match_pattern(d.lhs, seed, {})
        out = meta.instantiate_pattern(d.rhs, subst)
        if out == expected:
            found = True
    assert found, (
        f"no derived rule produced the fused form: "
        f"{[op_repr(d.rhs) for d in hits]}"
    )

def test_synthesis_with_lift_step_emits_composed_form():
    """With aff_lift_step available, the same pair composition emits the
    COMPOSED fusion: apply(aff_compose(aff(A2,x), aff(A1,u)), h)."""
    seed, A1, A2, h0, u, x = _recurrence_seed()
    derived = meta.synthesize_rules(R.SCAN_LAWS, [seed], fuel=3000)

    expected = Op.make(
        "apply",
        Op.make(
            "aff_compose", Op.make("aff", A2, x), Op.make("aff", A1, u)
        ),
        h0,
    )
    found = False
    for d in derived:
        subst = meta.match_pattern(d.lhs, seed, {})
        if (
            subst is not None
            and meta.instantiate_pattern(d.rhs, subst) == expected
        ):
            found = True
    assert found

def test_derived_rules_fire_in_egraph():
    """A synthesized rule plugs back into the e-graph and rewrites."""
    seed, _A1, _A2, _h0, _u, _x = _recurrence_seed()
    rules = [r for r in R.SCAN_LAWS if r.name != "aff_lift_step"]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    assert derived

    eg = EGraph()
    root = eg.add_term(seed)
    eg.run(derived, root, max_iterations=5, max_nodes=10_000)
    assert any(eg.rule_fires.get(d.name, 0) > 0 for d in derived)
