"""The ``bf16_safe`` preset — the sdpa-surface exclusion.

The fused ``sdpa`` kernel keeps softmax/accumulation in its internal
precision while the spelled IR path rounds probabilities to the
working dtype; under ``verify=True`` (reference = spelled lowering of
the exported IR) the gap compounds with depth at bf16 — measured on
whisper-turbo (32L, rel 0.418 vs the 0.0625 floor;
``graphopt:docs/reports/turbo_bf16_delivery.md``).  The preset is
``DEFAULT`` minus every rule whose *pattern* mentions ``sdpa`` —
pattern-level, so carrier-package and minted rules classify without
a name registry.
"""

from mlopt_core import laws
from mlopt_core.ir import Op
from mlopt_core.laws import R, RuleSet, preset, sdpa_surface


def _names(rs) -> set:
    return {r.name for r in rs}


def test_sdpa_surface_predicate():
    # Producers: spelled attention -> sdpa (rhs mentions sdpa).
    assert sdpa_surface(laws.SDPA_FOLD_NOMASK)
    assert sdpa_surface(laws.GQA_ABSORB)
    # Decomposers / in-kernel rewrites: sdpa on either side counts.
    assert all(sdpa_surface(r) for r in laws.SDPA_EXPAND_RULES)
    # A synthetic decompose-direction rule (sdpa lhs, spelled rhs).
    synth = R(
        "synth_sdpa_decomp",
        Op.make("sdpa", "q", "k", "v"),
        Op.make("matmul", Op.make("softmax", "s"), "v"),
    )
    assert sdpa_surface(synth)
    # sdpa nested deep inside a pattern arg still counts.
    deep = R(
        "synth_sdpa_nested",
        Op.make("add", Op.make("mul", Op.make("sdpa", "q", "k", "v"))),
        "x",
    )
    assert sdpa_surface(deep)
    # Negative cases: sdpa-free rules, non-Op patterns, sdpa only in
    # attrs (impossible — attrs are scalars — but a Const leaf is not
    # an Op).
    assert not sdpa_surface(laws.COMM_ADD)
    assert not sdpa_surface(R("bare", "a", "b"))


def test_sdpa_surface_covers_default_surface():
    """Every sdpa-mentioning DEFAULT member is classified."""
    surfaced = {r.name for r in laws.DEFAULT if sdpa_surface(r)}
    assert surfaced == {
        "gqa_absorb_repeat",
        "sdpa_fold_add",
        "sdpa_fold_add_drop",
        "sdpa_fold_adddiv",
        "sdpa_fold_adddiv_drop",
        "sdpa_fold_addmul",
        "sdpa_fold_addmul_drop",
        "sdpa_fold_div_nomask",
        "sdpa_fold_masked_fill",
        "sdpa_fold_masked_fill_drop",
        "sdpa_fold_masked_filldiv",
        "sdpa_fold_masked_filldiv_drop",
        "sdpa_fold_masked_fillmul",
        "sdpa_fold_masked_fillmul_drop",
        "sdpa_fold_mul_nomask",
        "sdpa_fold_nomask",
    }


def test_bf16_safe_is_default_minus_sdpa_surface():
    rs = laws.BF16_SAFE
    assert isinstance(rs, RuleSet)
    assert rs.name == "bf16_safe"
    assert _names(rs) == _names(laws.DEFAULT) - {
        r.name for r in laws.DEFAULT if sdpa_surface(r)
    }
    # Members are the ORIGINAL Rewrite objects — the preset stays
    # laws.serialize-portable (no scoped-closure derivatives).
    by_name = {r.name: r for r in laws.DEFAULT}
    assert all(by_name[n] is r for n, r in ((r.name, r) for r in rs))
    # No surviving member touches the sdpa surface.
    assert not any(sdpa_surface(r) for r in rs)
    # Spot pins: the verified-safe core keeps its algebra.
    assert "distribute_matmul_over_add" in _names(rs)
    assert "id_add" in _names(rs)
    assert "softmax_fold" in _names(rs)
    # Still a legal composition operand (union with SYMMETRY keeps
    # every member and stays sdpa-free).
    grown = rs + laws.SYMMETRY
    assert not any(sdpa_surface(r) for r in grown)
    assert _names(grown) == _names(rs) | _names(laws.SYMMETRY)


def test_bf16_safe_preset_lookup():
    assert preset("bf16_safe") is laws.BF16_SAFE
    assert laws.PRESETS["bf16_safe"] is laws.BF16_SAFE
    assert "bf16_safe" in {p.name for p in laws.PRESETS.values()}


def test_bf16_safe_is_not_default():
    assert laws.BF16_SAFE != laws.DEFAULT
    assert laws.BF16_SAFE != laws.FULL


def test_resolve_rules_bf16_safe():
    from mlopt_orchestrator.optimize import (
        _resolve_rules,
        bf16_safe_rules,
        default_rules,
    )

    rs = _resolve_rules("bf16_safe")
    assert rs is bf16_safe_rules()
    assert rs.name == "bf16_safe"
    # eqsat has no carriers registered — the composed default IS the
    # core DEFAULT, so the filtered set equals the core preset.
    assert _names(rs) == _names(laws.BF16_SAFE)
    assert not any(sdpa_surface(r) for r in rs)
    # The unfiltered default still carries the sdpa folds.
    assert any(sdpa_surface(r) for r in default_rules())
    # Aliases and caching.
    assert _resolve_rules("bf16-safe") is rs
    assert bf16_safe_rules() is rs


def test_bf16_safe_survives_custom_composition():
    """A user composing presets can apply the same predicate."""
    composed = laws.DEFAULT + preset("attention")
    kept = RuleSet(
        "custom_bf16",
        tuple(r for r in composed if not sdpa_surface(r)),
    )
    # The sdpa_expand decompose family leaves with the folds.
    assert not _names(kept) & {r.name for r in laws.SDPA_EXPAND_RULES}
    assert not any(sdpa_surface(r) for r in kept)
