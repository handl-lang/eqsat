"""Substrate-pure tests ported from ``graphopt:tests/test_meta_edges.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import types

import pytest

from mlopt_core import meta

from mlopt_core.egraph import EGraph, Rewrite

from mlopt_core.ir import Const, Op, Param, TensorType, Var


def _T(*shape):
    return TensorType(tuple(shape) if shape else (4, 4))

def _opcount(t, memo=None):
    """Additively-counts Op nodes — a valid ``(term, memo)`` cost fn."""
    if memo is None:
        memo = {}
    if t in memo:
        return memo[t]
    out = (
        1.0 + sum(_opcount(a, memo) for a in t.args)
        if isinstance(t, Op)
        else 0.0
    )
    memo[t] = out
    return out

def _guard_pair(
    r1_check=None,
    r1_derive=None,
    r2_check=None,
    r2_derive=None,
    pat1=None,
    pat2=None,
):
    r1 = Rewrite(
        "g1",
        Op.make("n1", "x", "y"),
        Op.make("n1o", "x", "y"),
        check=r1_check,
        derive=r1_derive,
    )
    r2 = Rewrite(
        "g2",
        Op.make("n2", "u"),
        Op.make("n2o", "u"),
        check=r2_check,
        derive=r2_derive,
    )
    p1 = pat1 if pat1 is not None else {"x": "x", "y": "y"}
    p2 = pat2 if pat2 is not None else {"u": "u"}
    return r1, r2, p1, p2

def test_canonicalize_metavar_leaves_and_idempotence():
    """str metavar leaves are ordinary sortable leaves: identities drop,
    children sort by ``op_repr``, and the result is a fixed point."""
    t = Op.make(
        "add",
        Op.make("mul", "b", Const(1.0)),
        Op.make("add", "a", Const(0.0)),
    )
    canon = meta.canonicalize(t)
    # mul(b, 1.0) collapses to "b" (multiplicative identity) and the
    # nested add flattens and drops 0.0: leaves "a","b" sort in order.
    assert canon == Op.make("add", "a", "b")
    assert meta.canonicalize(canon) == canon

def test_canonicalize_non_op_leaves_and_memo():
    x = Var("x", _T())
    assert meta.canonicalize(x) == x
    assert meta.canonicalize(Const(3.0)) == Const(3.0)
    # a bare metavar is just another non-Op leaf
    assert meta.canonicalize("metavar") == "metavar"
    # the memo actually memoises: second call hits the cache entry
    memo = {}
    t = Op.make("add", x, Const(0.0))
    first = meta.canonicalize(t, memo)
    assert first == x and t in memo
    assert meta.canonicalize(t, memo) == first

def test_canonicalize_attrs_survive_rebuild():
    """canonicalize rebuilds every Op with its own attrs; nested ops
    with different attrs are never confused with chain members."""
    x = Var("x", _T())
    t = Op.make(
        "transpose",
        Op.make("transpose", x, dim0=0, dim1=1),
        dim0=-1,
        dim1=-2,
    )
    canon = meta.canonicalize(t)
    # rebuilt through Op.make (hash-consed: identical structure → the
    # same interned object, attrs intact on both levels)
    assert dict(canon.attrs) == {"dim0": -1, "dim1": -2}
    assert dict(canon.args[0].attrs) == {"dim0": 0, "dim1": 1}
    assert meta.canonicalize(canon) == canon

def test_canonicalize_balanced_ordering_invariants():
    """matmul chains flatten order-preservingly and rebuild balanced:
    left- and right-leaning 4-chains canonicalize identically, a
    permutation does not."""
    a, b, c, d = (Var(n, _T()) for n in "abcd")
    left = Op.make(
        "matmul", Op.make("matmul", Op.make("matmul", a, b), c), d
    )
    right = Op.make(
        "matmul", a, Op.make("matmul", b, Op.make("matmul", c, d))
    )
    cl, cr = meta.canonicalize(left), meta.canonicalize(right)
    assert cl == cr
    # genuinely balanced: both root children are matmuls
    assert cl.op == "matmul"
    assert cl.args[0].op == "matmul" and cl.args[1].op == "matmul"
    perm = Op.make("matmul", Op.make("matmul", c, b), a)
    assert meta.canonicalize(perm) != cl

def test_stratified_run_without_objective_extracts_nothing():
    """``extract=True`` with no ``cost_fn``/``extract_fn`` is a pure
    saturate: no best/canonical_best keys at all."""
    x, y = Var("x", _T()), Var("y", _T())
    out = meta.stratified_run(EGraph(), [], Op.make("add", x, y))
    assert "best" not in out and "canonical_best" not in out
    assert out["canonical_input"] == Op.make("add", x, y)
    assert out["coherent_dropped"] == []
    assert out["contentful_used"] == []

def test_stratified_run_extract_flag_off():
    """``extract=False`` wins even when a cost_fn is supplied."""
    x, y = Var("x", _T()), Var("y", _T())
    out = meta.stratified_run(
        EGraph(),
        [],
        Op.make("add", x, y),
        extract=False,
        cost_fn=_opcount,
    )
    assert "best" not in out

def test_stratified_run_cost_fn_extraction():
    """With a cost fn the best member is extracted, and (by default)
    re-canonicalized; ``canonicalize_output=False`` serves it raw."""
    x, y = Var("x", _T()), Var("y", _T())
    # custom comm is *contentful* by name (not in COHERENT_RULE_NAMES),
    # so it still saturates — the e-graph holds both add orders.
    comm = Rewrite(
        "t_comm",
        Op.make("add", "a", "b"),
        Op.make("add", "b", "a"),
        law="test",
    )
    term = Op.make("add", x, y)
    out = meta.stratified_run(
        EGraph(), [comm], term, cost_fn=_opcount
    )
    assert out["contentful_used"] == ["t_comm"]
    assert out["best"] is not None
    assert out["canonical_best"] == meta.canonicalize(out["best"])
    raw = meta.stratified_run(
        EGraph(),
        [comm],
        term,
        cost_fn=_opcount,
        canonicalize_output=False,
    )
    assert raw["canonical_best"] == raw["best"]
    # stratification is honest: names, not semantics, classify rules
    named_comm = Rewrite(
        "comm_add",
        Op.make("add", "a", "b"),
        Op.make("add", "b", "a"),
    )
    out2 = meta.stratified_run(EGraph(), [named_comm], term)
    assert out2["coherent_dropped"] == ["comm_add"]

def test_iter_module_rules_lists_tuples_and_dedup():
    r1 = Rewrite("one", "a", "b")
    r2 = Rewrite("two", "a", Op.make("mul", "a", "a"))
    fake = types.SimpleNamespace(
        single=r1,
        listed=[r2, "noise", r1],  # dedup r1, skip non-Rewrite
        tupled=(r2,),
        other=42,
    )
    got = meta._iter_module_rules(fake)
    assert got == [r1, r2]  # deduped, first-occurrence order

def test_common_prefix_by_design_unequal_zip():
    """zip(strict=False): unequal-length inputs are *intended* —
    the shared prefix of a position and its ancestor must not raise."""
    assert meta._common_prefix((0, 1, 2), (0, 1, 3)) == (0, 1)
    assert meta._common_prefix((0, 1, 2), (0, 1)) == (0, 1)
    assert meta._common_prefix((), (0, 1)) == ()
    assert meta._common_prefix((0, 1), (0, 1)) == (0, 1)

def test_overlapping_positions():
    assert meta._overlapping((0,), (0, 1))  # ancestor/descendant
    assert meta._overlapping((), (0, 1))  # root contains everything
    assert meta._overlapping((0,), (0,))
    assert not meta._overlapping((0,), (1,))  # siblings

def test_has_attr_metavars():
    assert meta._has_attr_metavars(
        Op.make("transpose", "t", dim0="D", dim1=-1)
    )
    assert meta._has_attr_metavars(
        Op.make(
            "add",
            "a",
            Op.make("transpose", "t", dim0="D", dim1=-1),
        )
    )
    assert not meta._has_attr_metavars(Op.make("add", "a", "b"))
    assert not meta._has_attr_metavars(Var("v", _T()))
    assert not meta._has_attr_metavars("leaf")

def test_synthesizable_rejects_unbound_metavars():
    # RHS term metavar the LHS never binds → unusable
    unbound = Rewrite(
        "t_u", Op.make("add", "a", "b"), Op.make("mul", "a", "c")
    )
    assert not meta._synthesizable(unbound)
    # RHS attr metavar neither LHS-bound nor derivable → unusable
    unbound_attr = Rewrite(
        "t_ua",
        Op.make("transpose", "a", dim0="D1", dim1=-1),
        Op.make("transpose", "a", dim0="D1", dim1="D2"),
    )
    assert not meta._synthesizable(unbound_attr)
    # …but a derive that can produce it makes the rule usable
    derivable = Rewrite(
        "t_d",
        Op.make("transpose", "a", dim0="D1", dim1=-1),
        Op.make("transpose", "a", dim0="D1", dim1="D2"),
        derive=lambda b: {"$attr:D2": 0},
    )
    assert meta._synthesizable(derivable)
    # attr metavars bound by the LHS are always fine
    bound = Rewrite(
        "t_b",
        Op.make("transpose", "a", dim0="D1", dim1="D2"),
        Op.make("transpose", "a", dim0="D2", dim1="D1"),
    )
    assert meta._synthesizable(bound)

def test_match_pattern_attr_rebinding_and_keysets():
    pat = Op.make("transpose", "t", dim0="D", dim1="D")
    x = Var("x", _T())
    ok = Op.make("transpose", x, dim0=0, dim1=0)
    bad = Op.make("transpose", x, dim0=0, dim1=1)
    m = meta.match_pattern(pat, ok, {})
    assert m is not None and m["$attr:D"] == 0
    # second occurrence of the same attr metavar must agree
    assert meta.match_pattern(pat, bad, {}) is None
    # attr key-sets must match exactly
    extra = Op.make("transpose", x, dim0=0, dim1=0, extra=1)
    assert meta.match_pattern(pat, extra, {}) is None

def test_match_pattern_leaf_cases():
    x, y = Var("x", _T()), Var("y", _T())
    # repeated term metavar: both sites must bind equal subterms
    pat = Op.make("add", "a", "a")
    assert meta.match_pattern(pat, Op.make("add", x, x)) is not None
    assert meta.match_pattern(pat, Op.make("add", x, y)) is None
    # concrete pattern leaves compare structurally
    pat = Op.make("add", "a", Const(0.0))
    good = Op.make("add", x, Const(0.0))
    bad = Op.make("add", x, Const(1.0))
    m = meta.match_pattern(pat, good)
    assert m == {"a": x}
    assert meta.match_pattern(pat, bad) is None

def test_apply_rewrite_at_guard_vetoes():
    x, y = Var("x", _T()), Var("y", _T())
    t = Op.make("add", x, y)
    base = dict(
        lhs=Op.make("add", "a", "b"), rhs=Op.make("mul", "a", "b")
    )
    # every hook failure mode is a veto, never an exception
    for kw in (
        {"check": lambda b: False},
        {"check": lambda b: 1 / 0},
        {"derive": lambda b: 1 / 0},
        {"derive": lambda b: None},
    ):
        r = Rewrite("t_v", **base | kw)
        assert meta.apply_rewrite_at(r, t, ()) is None, kw
    # RHS metavar the binding cannot produce → KeyError → veto
    orphan = Rewrite(
        "t_o", Op.make("add", "a", "b"), Op.make("mul", "a", "c")
    )
    assert meta.apply_rewrite_at(orphan, t, ()) is None
    # no LHS match anywhere → None
    assert meta.apply_rewrite_at(
        Rewrite("t_s", Op.make("sub", "a"), Op.make("neg", "a")),
        t,
        (),
    ) is None
    # …and a rewrite at a non-root path rebuilds the spine
    swap = Rewrite(
        "t_sw", Op.make("add", "a", "b"), Op.make("add", "b", "a")
    )
    nested = Op.make("neg", t)
    assert meta.apply_rewrite_at(swap, nested, (0,)) == Op.make(
        "neg", Op.make("add", y, x)
    )

def test_fire_guarded_derive_placeholders():
    v = Var("v", _T())
    src = Op.make("mark", v)
    rule = Rewrite(
        "t_m",
        Op.make("mark", "a"),
        Op.make("marked", "a", tag="Q"),
        derive=lambda b: {"$attr:Q": 7, "extra": Const(2.0)},
    )
    out = meta._fire_guarded(rule, src, (), "@1:")
    assert out is not None
    subst, rewritten, conc = out
    # the derived attr stays SYMBOLIC in the instantiated rhs…
    assert rewritten == Op.make("marked", v, tag="@1:Q")
    # …while conc records the concrete value this instance derived —
    # attr outputs under "@i:" placeholders AND term-level ("$leaf:")
    # outputs alike
    assert conc == {"@1:Q": 7, "@1:extra": Const(2.0)}
    assert subst == {"a": v}

def test_fire_guarded_vetoes():
    v = Var("v", _T())
    src = Op.make("mark", v)
    base = dict(lhs=Op.make("mark", "a"), rhs=Op.make("marked", "a"))
    for kw in (
        {"check": lambda b: False},
        {"check": lambda b: 1 / 0},
        {"derive": lambda b: 1 / 0},
        {"derive": lambda b: None},
    ):
        r = Rewrite("t_v", **base | kw)
        assert meta._fire_guarded(r, src, (), "@1:") is None, kw
    # RHS attr metavar the derive doesn't produce → KeyError → None
    orphan = Rewrite(
        "t_o",
        Op.make("mark", "a"),
        Op.make("marked", "a", tag="Q"),
        derive=lambda b: {},
    )
    assert meta._fire_guarded(orphan, src, (), "@1:") is None
    # no match → None
    nomatch = Rewrite(
        "t_n", Op.make("other", "a"), Op.make("mark", "a")
    )
    assert meta._fire_guarded(nomatch, src, (), "@1:") is None

def test_reexpress_term_leaf_cases():
    v, w = Var("v", _T()), Var("w", _T())
    names = {v: "v0"}
    assert meta._reexpress_term("mv", names, set()) == "mv"
    assert meta._reexpress_term(w, names, {w}) is w  # keep pins
    assert meta._reexpress_term(v, names, set()) == "v0"
    assert meta._reexpress_term(Const(1.5), names, set()) == Const(1.5)
    got = meta._reexpress_term(Op.make("mul", v, w), names, {w})
    assert got == Op.make("mul", "v0", w)
    # a leaf the derived rule neither binds nor pins → un-reexpressible
    with pytest.raises(meta._Unreexpressible):
        meta._reexpress_term(Param("P", _T()), names, set())

def test_reexpress_binding_resolution():
    v = Var("v", _T())
    # "$attr:" entries whose value is a string resolve through subst
    got = meta._reexpress_binding({"$attr:d": "D"}, {"$attr:D": -1})
    assert got == {"$attr:d": -1}
    # a dangling reference is an un-reexpressible binding → None
    assert meta._reexpress_binding({"$attr:d": "D"}, {}) is None
    # non-string $attr values are baked in
    assert meta._reexpress_binding({"$attr:d": -2}, {}) == {
        "$attr:d": -2
    }
    # term patterns instantiate; a missing key → None
    got = meta._reexpress_binding({"x": "v0"}, {"v0": v})
    assert got == {"x": v}
    assert meta._reexpress_binding({"x": "v0"}, {}) is None

def test_compose_guards_unguarded_returns_nones():
    r1, r2, p1, p2 = _guard_pair()
    assert meta._compose_guards(r1, r2, p1, p2) == (None, None)

def test_compose_guards_eval_veto_branches():
    bound = {
        k: Var(k, _T()) for k in ("x", "y", "u", "v1", "v2")
    }

    # pat1 un-reexpressible on the fired binding → veto
    # (a guarded parent is needed for a composite check to exist)
    r1, r2, _p1, p2 = _guard_pair(r1_check=lambda b: True)
    chk, _drv = meta._compose_guards(
        r1, r2, {"x": "zz", "y": "y"}, p2
    )
    assert not chk(bound)

    # r1.check False / raising
    for c in (lambda b: False, lambda b: 1 / 0):
        r1, r2, p1, p2 = _guard_pair(r1_check=c)
        chk, _ = meta._compose_guards(r1, r2, p1, p2)
        assert not chk(bound)

    # r1.derive raising / returning None
    for d in (lambda b: 1 / 0, lambda b: None):
        r1, r2, p1, p2 = _guard_pair(r1_derive=d)
        chk, _ = meta._compose_guards(r1, r2, p1, p2)
        assert not chk(bound)

    # r1.derive non-attr outputs merge into the shared binding —
    # here r2's binding pattern reads one of them back
    r1, r2, p1, p2 = _guard_pair(
        r1_derive=lambda b: {"$attr:S": 3},
        pat2={"$attr:dim": "@1:S", "u": "u"},
        r2_check=lambda b: b.get("$attr:dim") == 3,
    )
    chk, _ = meta._compose_guards(r1, r2, p1, p2)
    assert chk(bound)

    # pat2 un-reexpressible → veto
    r1, r2, p1, _p2 = _guard_pair(r2_check=lambda b: True)
    chk, _ = meta._compose_guards(r1, r2, p1, {"u": "zz"})
    assert not chk(bound)

    # r2.check False / raising; r2.derive raising / None
    for c in (lambda b: False, lambda b: 1 / 0):
        r1, r2, p1, p2 = _guard_pair(r2_check=c)
        chk, _ = meta._compose_guards(r1, r2, p1, p2)
        assert not chk(bound)
    for d in (lambda b: 1 / 0, lambda b: None):
        r1, r2, p1, p2 = _guard_pair(r2_derive=d)
        chk, _ = meta._compose_guards(r1, r2, p1, p2)
        assert not chk(bound)

def test_compose_guards_derive_output_namespaces():
    bound = {k: Var(k, _T()) for k in ("x", "y", "u")}
    r1, r2, p1, p2 = _guard_pair(
        r1_derive=lambda b: {"$attr:S": 3, "plain": Const(1.0)},
        r2_derive=lambda b: {"$attr:T": 4},
    )
    chk, drv = meta._compose_guards(r1, r2, p1, p2)
    out = drv(bound)
    # every output lands under the parent's namespace — attr outputs
    # as "$attr:@i:" entries, term ("$leaf:") outputs as "@i:" leaf
    # placeholders
    assert out["$attr:@1:S"] == 3
    assert out["$attr:@2:T"] == 4
    assert out["@1:plain"] == Const(1.0)
    assert chk(bound)
    # a binding the composite cannot evaluate vetoes the derive
    bad = {k: v for k, v in bound.items() if k != "u"}
    assert drv(bad) is None and not chk(bad)

def test_rhs_derive_placeholders_scan():
    rhs = Op.make(
        "o1",
        Op.make("o2", "a", tag="@1:SD"),
        mode="@2:M",
        extra="@9:Z",
    )
    assert meta._rhs_derive_placeholders(rhs) == {"@1:SD", "@2:M"}
    assert meta._rhs_derive_placeholders(
        Op.make("o1", "a", tag="SD")
    ) == set()
    assert meta._rhs_derive_placeholders("leaf") == set()

def test_concrete_matched_leaves():
    zero = Const(0.0)
    # a concrete pattern leaf pins the value it matched
    assert meta._concrete_matched_leaves(zero, zero) == {zero}
    # a metavar pins nothing — its binding is free to generalise
    assert meta._concrete_matched_leaves("a", Var("v", _T())) == set()
    x = Var("x", _T())
    pat = Op.make("add", "a", zero)
    # Op pattern vs non-Op or differently-rooted term → no recursion
    assert meta._concrete_matched_leaves(pat, x) == set()
    assert (
        meta._concrete_matched_leaves(pat, Op.make("mul", x, zero))
        == set()
    )
    assert meta._concrete_matched_leaves(
        pat, Op.make("add", x, zero)
    ) == {zero}

def test_leaf_generalize_keep_names_fresh():
    v, w = Var("v", _T()), Var("w", _T())
    pin = Param("P", _T())
    names, cnt = {}, [0]
    region = Op.make("add", v, Op.make("mul", w, pin))
    got = meta._leaf_generalize(
        region, names, {pin}, cnt, assign_fresh=True
    )
    # kept leaves stay literal; the rest become shared metavars
    assert got == Op.make("add", "v0", Op.make("mul", "v1", pin))
    assert names == {v: "v0", w: "v1"}
    # RHS pass reuses names but introduces NO fresh metas — an unseen
    # leaf (a constant a parent pattern minted) stays concrete
    got2 = meta._leaf_generalize(
        Op.make("add", v, Const(4.0)),
        names,
        set(),
        cnt,
        assign_fresh=False,
    )
    assert got2 == Op.make("add", "v0", Const(4.0))

def test_is_tautology_and_alpha_key():
    # same metavars on both sides → tautology; commuted is a real rule
    assert meta._is_tautology(
        Op.make("add", "x", "y"), Op.make("add", "x", "y")
    )
    assert meta._is_tautology("v", "v")
    assert not meta._is_tautology(
        Op.make("add", "x", "y"), Op.make("add", "y", "x")
    )
    assert not meta._is_tautology("v", "w")
    # structure, not surface names: a repeated vs distinct binding
    assert not meta._is_tautology(
        Op.make("add", "x", "x"), Op.make("add", "x", "y")
    )

def test_replays_success_failure_fuel():
    x, y = Var("x", _T()), Var("y", _T())
    t0 = Op.make("add", x, y)
    r1 = Rewrite("p1", Op.make("add", "a", "b"), Op.make("mul", "a", "b"))
    r2 = Rewrite("p2", Op.make("mul", "a", "b"), Op.make("mul", "b", "a"))
    assert meta._replays(r1, r1, r2, t0, Op.make("mul", y, x))
    # an unreachable target is a genuine non-replay
    assert not meta._replays(r1, r1, r2, t0, Op.make("neg", x))
    # fuel exhausted mid-search is also a clean False
    assert not meta._replays(r1, r1, r2, t0, Op.make("mul", y, x), fuel=0)

def test_instantiation_stream_pool_cap():
    subs = list(meta._instantiation_stream(["x"], ["d1", "d2", "d3"]))
    # 12^3 attr combos would be 1728 — the pool caps at 400 per
    # leaf-shape profile, so exactly 2 * 400 substitutions come out
    assert len(subs) == 2 * meta._MAX_INSTANTIATIONS
    assert all(isinstance(s["x"], Var) for s in subs[:5])
    assert {s["$attr:d1"] for s in subs} <= set(meta._ATTR_POOL)

def test_validate_candidate_wellformedness():
    r1 = Rewrite("p1", Op.make("add", "a", "b"), Op.make("mul", "a", "b"))
    r2 = Rewrite("p2", Op.make("mul", "a", "b"), Op.make("mul", "b", "a"))
    # RHS term metavar not bound by the LHS → ill-formed
    bad = Rewrite("c1", Op.make("add", "x", "y"), Op.make("mul", "x", "z"))
    assert not meta._validate_candidate(bad, r1, r2, numeric=False)
    # RHS attr metavar neither bound nor derivable → ill-formed
    bad_attr = Rewrite(
        "c2",
        Op.make("transpose", "x", dim0="D", dim1=-1),
        Op.make("transpose", "x", dim0="D", dim1="E"),
    )
    assert not meta._validate_candidate(bad_attr, r1, r2, numeric=False)

def test_validate_candidate_witness_edges():
    x = Var("x", _T())
    cand = Rewrite("c", Op.make("add", "x", "y"), Op.make("mul", "x", "y"))
    r1 = Rewrite("p1", Op.make("add", "a", "b"), Op.make("mul", "a", "b"))
    r2 = Rewrite("p2", Op.make("mul", "a", "b"), Op.make("mul", "b", "a"))
    # a witness missing a metavar can't instantiate the LHS → rejected
    assert not meta._validate_candidate(
        cand, r1, r2, numeric=False, witness={"x": x}
    )
    # a witness binding a metavar to a leftover str makes the fired
    # output non-ground → rejected
    assert not meta._validate_candidate(
        cand, r1, r2, numeric=False, witness={"x": x, "y": "left"}
    )

def test_validate_candidate_replay_and_numeric():
    x, y = Var("x", _T()), Var("y", _T())
    comm = Rewrite(
        "cm", Op.make("add", "a", "b"), Op.make("add", "b", "a")
    )
    cand = Rewrite("c", Op.make("add", "x", "y"), Op.make("add", "x", "y"))
    w = {"x": x, "y": y}
    # comm∘comm honestly replays to identity — numeric AND structural
    assert meta._validate_candidate(cand, comm, comm, True, witness=w)
    assert meta._validate_candidate(cand, comm, comm, False, witness=w)

def test_subsumed_instance_and_guarded_exclusion():
    cand = Rewrite(
        "c", Op.make("add", "x", "y"), Op.make("add", "y", "x")
    )
    # an existing rule whose RHS has an unbound metavar is skipped
    # (its instantiation KeyErrors out, it cannot subsume anything)
    weird = Rewrite(
        "w", Op.make("add", "a", "b"), Op.make("mul", "a", "zzz")
    )
    assert not meta._subsumed(cand, [weird])
    # an alpha-instance of an existing unguarded rule IS subsumed
    comm = Rewrite(
        "cm", Op.make("add", "p", "q"), Op.make("add", "q", "p")
    )
    assert meta._subsumed(cand, [comm])
    # …but a guarded rule never subsumes, even with identical patterns
    gcomm = Rewrite(
        "g",
        Op.make("add", "p", "q"),
        Op.make("add", "q", "p"),
        check=lambda b: True,
    )
    assert not meta._subsumed(cand, [gcomm])

def test_synthesize_fuel_bounded_early_exits():
    x, y = Var("x", _T()), Var("y", _T())
    seed = Op.make("add", x, y)
    swap = Rewrite(
        "t_sw", Op.make("add", "a", "b"), Op.make("add", "b", "a")
    )
    comm = Rewrite(
        "t_cm", Op.make("mul", "a", "b"), Op.make("mul", "b", "a")
    )
    # fuel=1: the very first r1∘r2 attempt exhausts the budget —
    # every inner loop's spent-check must break cleanly, over BOTH seeds
    out = meta.synthesize_rules([swap, comm], [seed, seed], fuel=1)
    assert isinstance(out, list)
    assert all(not meta._is_tautology(d.lhs, d.rhs) for d in out)
    # fuel=0 with no seeds still runs the symbolic loop's guard
    assert meta.synthesize_rules([swap, comm], [], fuel=0) == []

def test_synthesize_drops_tautological_pairs():
    """comm at the same site twice is a critical pair whose composite
    is the identity — offered, detected, dropped."""
    x, y = Var("x", _T()), Var("y", _T())
    comm = Rewrite("cm", Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    out = meta.synthesize_rules([comm], [Op.make("add", x, y)], fuel=200)
    assert out == []

def test_synthesize_seed_attr_placeholder_resolution():
    """Inside the seed path, r2's attr metavar bindings that hit r1's
    "@1:" placeholders resolve via the concrete derive map — while a
    literal placeholder-shaped string it CANNOT resolve vetoes the
    pair (``bad``), never producing a garbage binding."""
    v = Var("v", _T())
    seed = Op.make("mark", Op.make("other", v, tag="@9:zz"))
    r1 = Rewrite(
        "r1",
        Op.make("mark", "a"),
        Op.make("marked", "a", tag="Q"),
        derive=lambda b: {"$attr:Q": 7},
    )
    r2a = Rewrite(
        "r2a",
        Op.make("marked", "x", tag="T"),
        Op.make("sink", "x"),
    )
    r2b = Rewrite(
        "r2b",
        Op.make("other", "x", tag="U"),
        Op.make("sink", "x"),
    )
    out = meta.synthesize_rules([r1, r2a, r2b], [seed], fuel=400)
    assert isinstance(out, list)
    # whatever survives is well-formed: every RHS term metavar is
    # bound by the LHS or is a "@i:" leaf placeholder the derived
    # rule's own composite derive refills (the "$leaf:" channel)
    for d in out:
        rhs_t = {
            m
            for m in meta.pattern_metavars(d.rhs)
            if not m.startswith("$attr:")
        }
        lhs_t = {
            m
            for m in meta.pattern_metavars(d.lhs)
            if not m.startswith("$attr:")
        }
        assert rhs_t - lhs_t <= {
            m for m in rhs_t if m.startswith(("@1:", "@2:"))
        }
        assert rhs_t <= lhs_t | set(meta._rhs_derive_placeholders(d.rhs))

def test_synthesize_seed_guard_vetoes_are_continues():
    """r2's check/derive failure modes inside the seed loop each hit
    ``continue`` — they never crash and never emit."""
    x, y = Var("x", _T()), Var("y", _T())
    seed = Op.make("add", x, y)
    r1 = Rewrite("r1", Op.make("add", "a", "b"), Op.make("mul", "a", "b"))
    boom_check = Rewrite(
        "rc",
        Op.make("mul", "a", "b"),
        Op.make("mul", "b", "a"),
        check=lambda b: 1 / 0,
    )
    boom_derive = Rewrite(
        "rd",
        Op.make("mul", "a", "b"),
        Op.make("mul", "b", "a"),
        derive=lambda b: 1 / 0,
    )
    veto_derive = Rewrite(
        "rv",
        Op.make("mul", "a", "b"),
        Op.make("mul", "b", "a"),
        derive=lambda b: None,
    )
    # a derive returning NON-attr keys merges them into inst2 — the
    # rule stays synthesizable because its rhs needs nothing extra
    nonattr = Rewrite(
        "rn",
        Op.make("mul", "a", "b"),
        Op.make("mul", "b", "a"),
        derive=lambda b: {"k": Const(2.0)},
    )
    rules = [r1, boom_check, boom_derive, veto_derive, nonattr]
    out = meta.synthesize_rules(rules, [seed], fuel=400)
    parents = {meta.provenance(d) for d in out}
    # the vetoing second parents produced nothing
    assert ("r1", "rc") not in parents
    assert ("r1", "rd") not in parents
    assert ("r1", "rv") not in parents
    # every emitted rule is still well-formed + carries provenance
    for d in out:
        assert not meta._is_tautology(d.lhs, d.rhs)
        assert d.law.startswith("synthesized:")

def test_synthesize_unreexpressible_binding_rejected():
    """r2 binding a leaf that escapes the abstracted region makes the
    re-expression raise _Unreexpressible: the pair is skipped on the
    seed path (the symbolic path may still emit its own honest form)."""
    x = Var("x", _T())
    pin = Param("PIN", _T())
    seed = Op.make("tag", x)
    r1 = Rewrite("r1", Op.make("tag", "a"), Op.make("add", "a", pin))
    r2 = Rewrite("r2", Op.make("add", "u", "w"), Op.make("mul", "u", "w"))
    out = meta.synthesize_rules([r1, r2], [seed], fuel=400)
    # no SEED-channeled rule descends from (r1, r2): that specific
    # composition's fired binding held PIN, outside the leaf names
    assert all(
        not (meta.provenance(d) == ("r1", "r2") and "(seed)" in d.law)
        for d in out
    )
    # and every emitted rule is still well-formed + verified
    for d in out:
        assert not meta._is_tautology(d.lhs, d.rhs)

def test_provenance_registry_and_parents_attr():
    plain = Rewrite("plain", "a", "b")
    assert meta.provenance(plain) == ()
    meta.SYNTH_PARENTS["ghost"] = ("p1", "p2")
    try:
        assert meta.provenance(Rewrite("ghost", "a", "b")) == (
            "p1",
            "p2",
        )
    finally:
        del meta.SYNTH_PARENTS["ghost"]
    # an instance-level .parents wins over the registry
    r = Rewrite("x", "a", "b")
    object.__setattr__(r, "parents", ("u", "v"))
    meta.SYNTH_PARENTS["x"] = ("p", "q")
    try:
        assert meta.provenance(r) == ("u", "v")
    finally:
        del meta.SYNTH_PARENTS["x"]

class _StubEval:
    """A ``ConcreteEval`` stub over scalars — torch-free tri-state.

    ``make_env`` hands each typed leaf a small float; ``eval_term``
    evaluates ``add``/``mul``/``neg`` arithmetically, raises
    ``KeyError`` for an op in *unbound* (the "no backend binding"
    signal) and ``RuntimeError`` for an op in *broken* (a term that
    cannot denote — the oracle's rhs-err analogue).  ``extra`` maps
    further op names to evaluators so a test can mint a working
    custom op.
    """

    def __init__(self, unbound=(), broken=(), extra=None):
        self.unbound = set(unbound)
        self.broken = set(broken)
        self.extra = dict(extra or {})

    def make_env(self, leaf_shapes):
        return {
            leaf: float(i + 1)
            for i, leaf in enumerate(sorted(leaf_shapes, key=repr))
        }

    def eval_term(self, term, env):
        if isinstance(term, Op):
            if term.op in self.unbound:
                raise KeyError(term.op)
            if term.op in self.broken:
                raise RuntimeError(f"{term.op} cannot denote")
            args = [self.eval_term(a, env) for a in term.args]
            if term.op in self.extra:
                return self.extra[term.op](*args)
            fn = {
                "add": lambda a, b: a + b,
                "mul": lambda a, b: a * b,
                "neg": lambda a: -a,
            }[term.op]
            return fn(*args)
        if isinstance(term, Const):
            return term.value
        return env[term]

    def allclose(self, a, b, tol):
        return abs(a - b) <= tol


def _with_eval(monkeypatch, backend):
    """Install *backend* as the concrete-eval backend (auto-restored)."""
    monkeypatch.setattr(meta, "_concrete_eval", backend)


def test_validate_candidate_numeric_eval_failures(monkeypatch):
    """The numeric leg's tri-state: eval failures are measured, not skipped.

    An eval error is not "ops without bindings": a bound LHS instance
    that cannot denote vetoes only that instantiation, while the
    claimed OUTPUT failing on a live instance is a measured
    counterexample — the ill-typed-RHS signal the emit gate must
    honour.  ``KeyError`` stays the structural-only escape.
    """
    x, y = Var("x", _T()), Var("y", _T())
    w = {"x": x, "y": y}
    comm = Rewrite("cm", Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    swap = Rewrite("sw", Op.make("add", "a", "b"), Op.make("mul", "a", "b"))
    idm = Rewrite("id", Op.make("mul", "a", "b"), Op.make("mul", "a", "b"))
    cand = Rewrite("c", Op.make("add", "x", "y"), Op.make("mul", "x", "y"))

    # the claimed output cannot denote on a live instance → reject
    _with_eval(monkeypatch, _StubEval(broken={"mul"}))
    assert not meta._validate_candidate(
        cand, swap, idm, numeric=True, witness=w
    )
    # ...but an UNBOUND op degrades to structural-only, as before
    _with_eval(monkeypatch, _StubEval(unbound={"mul"}))
    assert meta._validate_candidate(
        cand, swap, idm, numeric=True, witness=w
    )
    # the bound LHS instance itself cannot denote → vetoed, not emitted
    _with_eval(monkeypatch, _StubEval(broken={"add"}))
    assert not meta._validate_candidate(
        cand, swap, idm, numeric=True, witness=w
    )
    # an unbound LHS op is the same structural-only escape
    _with_eval(monkeypatch, _StubEval(unbound={"add"}))
    assert meta._validate_candidate(
        cand, swap, idm, numeric=True, witness=w
    )
    # both sides denote but differ → allclose still rejects
    _with_eval(monkeypatch, _StubEval())
    assert not meta._validate_candidate(
        cand, swap, idm, numeric=True, witness=w
    )
    # and a true composite (comm twice returns to start) passes the
    # whole leg
    ident = Rewrite(
        "ok", Op.make("add", "x", "y"), Op.make("add", "x", "y")
    )
    assert meta._validate_candidate(
        ident, comm, comm, numeric=True, witness=w
    )


def test_validate_candidate_numeric_no_backend():
    """No registered backend: the numeric leg stays structural-only."""
    x, y = Var("x", _T()), Var("y", _T())
    comm = Rewrite("cm", Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    ident = Rewrite(
        "ok", Op.make("add", "x", "y"), Op.make("add", "x", "y")
    )
    assert meta._concrete_eval is None  # eqsat is torch-free
    assert meta._validate_candidate(
        ident, comm, comm, numeric=True, witness={"x": x, "y": y}
    )


def test_synthesize_drops_ill_typed_composite(monkeypatch):
    """End-to-end: a composite whose RHS errors on the witness never emits.

    ``negid`` then ``weird`` on ``add(neg(neg(x)), y)`` mints
    ``add(neg(neg(v0)), v1) -> weird(v0, v1)`` — the derivation
    honestly replays, but with a ``weird`` that cannot denote on the
    seed binding the emitted composite would be ill-typed (the
    oracle's rhs-err signal), so the emit gate refuses it.  The same
    pair with a working ``weird`` mints normally.
    """
    x, y = Var("x", _T()), Var("y", _T())
    seed = Op.make("add", Op.make("neg", Op.make("neg", x)), y)
    negid = Rewrite("nn", Op.make("neg", Op.make("neg", "a")), "a")
    weird = Rewrite(
        "wd", Op.make("add", "a", "b"), Op.make("weird", "a", "b")
    )
    rules = [negid, weird]

    _with_eval(monkeypatch, _StubEval(broken={"weird"}))
    out = meta.synthesize_rules(rules, [seed], fuel=400)
    assert all(
        "weird" not in {t.op for t in _walk_ops(d.rhs)} for d in out
    )

    _with_eval(
        monkeypatch,
        _StubEval(extra={"weird": lambda a, b: a + b}),
    )
    out = meta.synthesize_rules(rules, [seed], fuel=400)
    assert any(
        "weird" in {t.op for t in _walk_ops(d.rhs)} for d in out
    )


def _walk_ops(t):
    """Yield every Op node inside *t* (test-local term walk)."""
    if isinstance(t, Op):
        yield t
        for a in t.args:
            yield from _walk_ops(a)


def test_defensive_branch_inventory():
    """Documents (and lightly probes) branches we believe unreachable
    by construction — candidates for ``pragma: no cover``:

    * meta.py:256   — ``len(flat) == 1`` in the assoc-only branch: a
      same-op chain of BINARY ops (matmul/aff_compose/affd_compose)
      always yields ≥2 leaves.
    * meta.py:1240->1242 — ``offer()`` is only ever called with
      ``pats`` set (both the seed and symbolic paths pass it); the
      ``pats is None`` arm is a defensive default.
    * meta.py:1324-1325, 1388-1389, 1417-1418 — the rhs
      ``instantiate_pattern`` KeyError escapes: ``_synthesizable``
      already guarantees every rhs metavar is lhs-bound or
      derive-produced, so the substitution is always total.
    * meta.py:1410-1411, 1414 — ``ok = False`` when an rhs attr
      metavar is unbound and the parent has no derive: _synthesizable
      filtered exactly those rules out of ``usable``.
    """
    # sanity for the 256 claim: every binary-assoc chain yields ≥2
    a, b, c = (Var(n, _T()) for n in "abc")
    for op in ("matmul", "aff_compose", "affd_compose"):
        t = Op.make(op, Op.make(op, a, b), c)
        flat = meta._flatten_chain(t)
        assert len(flat) >= 2, op
