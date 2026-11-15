"""Substrate-pure tests ported from ``graphopt:tests/test_rulesets.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core import laws

from mlopt_core.egraph import Rewrite

from mlopt_core.ir import Const, Op, TensorType, Var

from mlopt_core.laws import (
    ALL_RULES,
    ALL_RULES_WITH_LAYOUT,
    CATEGORICAL_RULES,
    COMM_ADD,
    DISTRIBUTE_MUL,
    GQA_ABSORB,
    ID_ADD,
    LAYOUT_RULES,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    SDPA_FOLD_RULES,
    SIMPLIFICATION_RULES,
    SWIGLU_FUSE,
    R,
    RuleSet,
    all_rules,
    preset,
    tags,
)

from mlopt_core.ports import RuleLike, RuleSetLike


SUBSUMED_NAMES = {
    "swiglu_fuse",
    "parallel_mul_fuse",
    "qkv_fuse",
    "qkv_fuse_asym",
}

SYMMETRY_NAMES = {
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "linear_row_scale",
    "linear_row_scale_rev",
    "linear_channel_scale",
    "linear_channel_scale_rev",
}

def _a() -> RuleSet:
    return RuleSet("a", (COMM_ADD, ID_ADD))

def _b() -> RuleSet:
    return RuleSet("b", (DISTRIBUTE_MUL,))

def test_rewrite_gains_tags_and_r_kwarg():
    plain = R("t", Op.make("add", "a", "b"), "a")
    assert plain.tags == frozenset()
    tagged = R(
        "t2",
        Op.make("add", "a", "b"),
        "a",
        tags=[tags.SIMPLIFICATION],
    )
    assert tagged.tags == frozenset({tags.SIMPLIFICATION})
    # Construction order (positional Rewrite) stays intact.
    direct = Rewrite("x", "a", "b")
    assert direct.tags == frozenset()

def test_ruleset_construction_dedup_and_containment():
    rs = RuleSet("t", [COMM_ADD, ID_ADD])
    assert isinstance(rs.rules, tuple)  # normalised
    assert len(rs) == 2
    assert list(rs) == [COMM_ADD, ID_ADD]
    assert "comm_add" in rs
    assert COMM_ADD in rs
    assert "nope" not in rs
    assert DISTRIBUTE_MUL not in rs
    assert 42 not in rs
    with pytest.raises(ValueError, match="duplicate rule name"):
        RuleSet("bad", (COMM_ADD, COMM_ADD))

def test_ruleset_value_semantics():
    a, b = _a(), _b()
    assert a == _a()
    assert a != b
    assert (a == "not a ruleset") is False
    assert hash(a) == hash(_a())
    # name/description are labels, not identity.
    assert a == RuleSet("other", (COMM_ADD, ID_ADD), description="d")

def test_ruleset_union_algebra():
    a, b = _a(), _b()
    assert a + a == a  # idempotent
    assert a + b == b + a  # order-insensitive
    assert len(a + b) == 3
    assert (a + b) - b == a
    assert (a + b) & b == b
    assert (a + b) - b - a == RuleSet("empty", ())
    # union accepts plain iterables too.
    assert a.union([DISTRIBUTE_MUL]) == a + b
    assert (a - [COMM_ADD]) == RuleSet("x", (ID_ADD,))
    assert (a & [COMM_ADD]) == RuleSet("x", (COMM_ADD,))

def test_ruleset_union_name_conflict():
    other = RuleSet(
        "o",
        (R("comm_add", Op.make("add", "x", "y"), "x"),),
    )
    a = _a()
    with pytest.raises(ValueError, match="override=True"):
        a + other
    replaced = a.union(other, override=True)
    assert replaced.rules[0] is other.rules[0]
    assert len(replaced) == 2

def test_ruleset_named_and_tagged_subsets():
    rs = _a() + _b()
    sub = rs.named("comm_add")
    assert isinstance(sub, RuleSet)
    assert list(sub) == [COMM_ADD]
    assert rs.tagged(tags.EXPANSIVE) == RuleSet(
        "exp", (COMM_ADD, DISTRIBUTE_MUL)
    )
    assert rs.tagged("nonexistent") == RuleSet("empty", ())
    # tags / names list through as labels for inspection.
    assert sub.name == "a+b[comm_add]"
    assert rs.tagged(tags.EXPANSIVE).name == "a+b.tagged"

def test_ruleset_priorities():
    rs = _a().with_priorities(comm_add=laws.EARLY)
    assert rs.priority_of(COMM_ADD) == laws.EARLY == 0
    assert rs.priority_of("comm_add") == 0
    assert rs.priority_of(ID_ADD) == laws.NORMAL == 10
    assert rs.priority_of("missing") == laws.NORMAL
    assert laws.LATE == 20
    merged = rs.with_priorities(id_add=laws.LATE)
    assert merged.priority_of("id_add") == 20
    assert merged.priority_of("comm_add") == 0  # merge, not replace
    with pytest.raises(KeyError, match="with_priorities"):
        rs.with_priorities(nope=1)

def test_ruleset_priorities_ride_composition():
    a = _a().with_priorities(comm_add=1)
    b = _b().with_priorities(distribute_matmul_over_add=2)
    u = a + b  # the operand's own priorities merge in
    assert u.priority_of("comm_add") == 1
    assert u.priority_of("distribute_matmul_over_add") == 2
    # subsets keep only the members' priorities.
    assert (u - b).priorities == {"comm_add": 1}
    assert (u & b).priorities == {"distribute_matmul_over_add": 2}
    assert u.named("comm_add").priorities == {"comm_add": 1}
    assert u.tagged(tags.SYMMETRY).priorities == {"comm_add": 1}

def test_ruleset_is_a_rulesetlike():
    assert isinstance(laws.DEFAULT, RuleSetLike)
    assert all(isinstance(r, RuleLike) for r in laws.DEFAULT)

def test_tag_assignments():
    assert COMM_ADD.tags == {tags.SYMMETRY, tags.EXPANSIVE}
    assert SWIGLU_FUSE.tags == {tags.FUSION, tags.SUBSUMED}
    assert ID_ADD.tags == {tags.SIMPLIFICATION}
    assert DISTRIBUTE_MUL.tags == {tags.CATEGORICAL, tags.EXPANSIVE}
    assert all(tags.FUSION in r.tags for r in SDPA_FOLD_RULES)
    assert GQA_ABSORB.tags == {tags.FUSION}
    assert all(tags.LAYOUT in r.tags for r in LAYOUT_RULES)
    assert all(r.tags == {tags.SCAN} for r in SCAN_LAWS)
    assert all(r.tags == {tags.SCAN} for r in SCAN_DIAG_LAWS)

def test_every_rule_is_tagged():
    """Tag hygiene: no shipped rule is left untagged."""
    assert all(r.tags for r in all_rules())
    assert all(r.tags for r in LAYOUT_RULES)
    assert all(r.tags for r in SCAN_LAWS)

def test_preset_contents():
    assert len(laws.SIMPLIFICATION) == len(SIMPLIFICATION_RULES)
    # the historical "categorical" selection — minus the subsumed folds
    assert len(laws.CATEGORICAL) == len(CATEGORICAL_RULES) - 4
    assert (
        len(laws.FUSION)
        == sum(1 for r in all_rules() if tags.FUSION in r.tags)
        == 81  # 21 bank fusion laws + the 60 prim:synop*/synopN_* folds
    )
    assert {r.name for r in laws.SYMMETRY} == SYMMETRY_NAMES
    # CARRIERS names the carrier-package families — empty core-side;
    # the scan monoids are the core share of CARRIER_SEARCH.
    assert not laws.CARRIERS
    assert len(laws.CARRIER_SEARCH) == (
        len(SCAN_LAWS) + len(SCAN_DIAG_LAWS)
    )
    assert len(laws.WITH_LAYOUT) == len(ALL_RULES_WITH_LAYOUT)
    assert laws.PRESETS["default"] is laws.DEFAULT
    assert {p.name for p in laws.PRESETS.values()} == set(laws.PRESETS)

def test_default_excludes_symmetry_and_subsumed():
    names = {r.name for r in laws.DEFAULT}
    assert names.isdisjoint(SYMMETRY_NAMES)
    assert names.isdisjoint(SUBSUMED_NAMES)
    assert not laws.DEFAULT.tagged(tags.SUBSUMED)
    assert not laws.DEFAULT.tagged(tags.SYMMETRY)
    # what stays: categorical + simplification(non-sym) + fusion(non-sub)
    assert "distribute_matmul_over_add" in names
    assert "id_add" in names
    assert "gqa_absorb_repeat" in names
    # the scan monoids stay out of the default search — they are the
    # regime's CARRIER_SEARCH set, not pipeline algebra.
    assert names.isdisjoint({r.name for r in SCAN_LAWS})

def test_full_is_default_plus_symmetry():
    assert laws.DEFAULT + laws.SYMMETRY == laws.FULL
    # FULL is exactly the retired ``ruleset="all"`` selection —
    # the whole core equational surface minus the pairing-subsumed
    # folds (the pairing pass owns those).
    assert {r.name for r in laws.FULL} == {
        r.name for r in ALL_RULES if r.name not in SUBSUMED_NAMES
    }
    assert not laws.FULL.tagged(tags.SUBSUMED)

def test_simplification_plus_categorical_is_legacy_all():
    """SIMPLIFICATION + CATEGORICAL is exactly the old ruleset="all"."""
    legacy = {
        r.name for r in all_rules() if r.name not in SUBSUMED_NAMES
    }
    assert (
        RuleSet(
            "legacy", tuple(r for r in all_rules() if r.name in legacy)
        )
        == laws.SIMPLIFICATION + laws.CATEGORICAL
    )

def test_preset_lookup():
    assert preset("default") is laws.DEFAULT
    assert preset("full") is laws.FULL
    assert preset("simplification") is laws.SIMPLIFICATION
    with pytest.raises(ValueError, match="Unknown ruleset"):
        preset("bogus")

def test_resolve_rules_variants():
    from mlopt_orchestrator.optimize import (
        _resolve_rules,
        default_rules,
    )

    assert _resolve_rules(None) is default_rules()
    assert _resolve_rules(laws.CATEGORICAL) is laws.CATEGORICAL
    assert _resolve_rules("categorical") is laws.CATEGORICAL
    assert _resolve_rules("default") is default_rules()
    assert _resolve_rules("default_rules") is default_rules()
    # orchestrator-level names resolve to the regime rule sets.
    cs = _resolve_rules("carrier_search")
    assert {r.name for r in laws.CARRIER_SEARCH} <= {r.name for r in cs}
    assert _resolve_rules("xc") == _resolve_rules("xc")
    custom = _resolve_rules([COMM_ADD, ID_ADD])
    assert isinstance(custom, RuleSet) and len(custom) == 2
    with pytest.raises(ValueError, match="Unknown ruleset"):
        _resolve_rules("bogus")

def test_regime_rule_sets():
    import mlopt_orchestrator.regime as regime

    search = regime.default_rules()
    assert isinstance(search, RuleSet)
    # CARRIER_SEARCH = the scan monoids + decode / om / trace.
    assert {r.name for r in laws.CARRIER_SEARCH} <= {
        r.name for r in search
    }
    assert search == regime.CARRIER_LAWS
    assert isinstance(regime.XC_LAWS, RuleSet)

def test_optimize_getattr_unknown():
    import mlopt_orchestrator.optimize as opt

    assert opt.DEFAULT_RULES is opt.default_rules()
    with pytest.raises(AttributeError):
        _ = opt.bogus_name


# ---------------------------------------------------------------------------
#  RuleSet.scoped — the composition/scope combinator
# ---------------------------------------------------------------------------


def test_scoped_noop_returns_self():
    a = _a()
    assert a.scoped() is a
    # a relabel alone still derives a set.
    relabelled = a.scoped(name="a2")
    assert relabelled.name == "a2" and relabelled == a


def test_scoped_membership_and_rename():
    rs = (_a() + _b()).with_priorities(comm_add=1)
    sub = rs.scoped(include=["comm_add"], rename={"comm_add": "ca"})
    assert [r.name for r in sub] == ["ca"]
    # priorities follow the rename.
    assert sub.priorities == {"ca": 1}
    assert sub.priority_of("ca") == 1
    # exclude composes with include on original names.
    both = rs.scoped(exclude=["comm_add"])
    assert [r.name for r in both] == ["id_add", "distribute_matmul_over_add"]
    # unknown rename sources raise like with_priorities.
    with pytest.raises(KeyError, match="rename sources"):
        rs.scoped(rename={"nope": "x"})
    # rename onto an existing member name is a duplicate-name error.
    with pytest.raises(ValueError, match="duplicate rule name"):
        rs.scoped(rename={"comm_add": "id_add"})


def test_scoped_predicate_gates_check():
    calls = []

    def orig_check(bound):
        calls.append(bound)
        return True

    rule = R(
        "gated",
        Op.make("add", "x", "y"),
        "x",
        check=orig_check,
    )
    rs = RuleSet("g", (rule,))
    off = rs.scoped(predicate=lambda bound: False)
    scoped_rule = off.rules[0]
    # scope vetoes BEFORE the rule's own check runs.
    assert scoped_rule.check({"x": None}) is False
    assert calls == []
    on = rs.scoped(predicate=lambda bound: True)
    assert on.rules[0].check({"x": None}) is True
    assert calls == [{"x": None}]
    # the unscoped twin is untouched.
    assert rule.check({"x": None}) is True


def test_scoped_op_metavar_scope():
    linear = Op.make("linear", "x", "w")
    matmul = Op.make("matmul", "x", "w")
    rule = R("anyroot", "t", "t")
    rs = RuleSet("s", (rule,))
    scoped = rs.scoped(op="linear", metavar="t")
    check = scoped.rules[0].check
    assert check({"t": linear}) is True
    assert check({"t": matmul}) is False
    # a tuple of ops and a missing binding both behave.
    scoped2 = rs.scoped(op=("linear", "matmul"), metavar="t")
    assert scoped2.rules[0].check({"t": matmul}) is True
    assert scoped2.rules[0].check({}) is False
    # op= without metavar= is a usage error.
    with pytest.raises(ValueError, match="metavar="):
        rs.scoped(op="linear")
    # predicate and op= conjoin.
    both = rs.scoped(
        predicate=lambda b: b.get("t") is linear,
        op="linear",
        metavar="t",
    )
    assert both.rules[0].check({"t": linear}) is True
    assert both.rules[0].check({"t": matmul}) is False


def test_scoped_preserves_rule_data_and_cond():
    rule = R(
        "cond_rule",
        Op.make("add", "a", "b"),
        "a",
        cond=("scalar", "a"),
    )
    rs = RuleSet("c", (rule,))
    scoped = rs.scoped(predicate=lambda b: True)
    derived = scoped.rules[0]
    # cond/dspec data carried verbatim; check still honours the fold.
    assert derived.cond == rule.cond
    scalar = Const(3.0)
    wide = Var("w", TensorType((2, 3)))
    assert derived.check({"a": scalar}) is True
    assert derived.check({"a": wide}) is False
    # the original rule object is not mutated.
    assert rule.check is not None and rule.cond == ("scalar", "a")


def test_scoped_members_are_distinct_rules():
    a = _a()
    scoped = a.scoped(predicate=lambda b: True)
    # same names but different check hooks -> union collision.
    with pytest.raises(ValueError, match="override=True"):
        a + scoped
    # rename namespaces the scoped copy so both can coexist — scope
    # just the one member so the rest keep their unscoped identity.
    namespaced = a.scoped(
        include=["comm_add"],
        predicate=lambda b: True,
        rename={"comm_add": "comm_add@scope"},
    )
    merged = a + namespaced
    assert len(merged) == 3
    assert "comm_add@scope" in merged
