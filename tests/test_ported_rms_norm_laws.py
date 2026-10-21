"""Substrate-pure tests ported from ``graphopt:tests/test_rms_norm_laws.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var

from mlopt_core.laws import (
    DEFAULT,
    RMS_NORM_FOLD,
    RMS_NORM_FOLD_NOGAIN,
    SIMPLIFICATION_RULES,
    _check_rms_consts,
    _check_rms_fold,
    _check_rms_fold_nogain,
    _derive_rms_norm,
    tags,
)

from mlopt_core.laws.tensor import _rms_normalized_shape

from mlopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)


def _bound(u_shape, w_shape=None, dims=(-1,), mk=True, eps=1e-6):
    """A ``bound`` dict as the matcher would build it."""
    b = {
        "u": Var("u", TensorType(u_shape)),
        "P": Const(2),
        "EPS": Const(eps),
        "$attr:MD": dims,
        "$attr:MK": mk,
    }
    if w_shape is not None:
        b["w"] = Param("w", TensorType(w_shape))
    return b

def test_rms_norm_folds_registered_and_tagged():
    """Both folds are shipped simplifications, in the default set."""
    for rule in (RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN):
        assert rule in SIMPLIFICATION_RULES
        assert rule.tags == {tags.SIMPLIFICATION}
    names = {r.name for r in DEFAULT}
    assert "rms_norm_fold" in names
    assert "rms_norm_fold_nogain" in names

def test_rms_norm_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the same term, and
    ``derive`` produces the ``$attr:ND``/``$attr:EP`` the RHS needs."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    subst = {
        "u": u,
        "w": w,
        "P": Const(2),
        "EPS": Const(1e-6),
        "$attr:MD": (-1,),
        "$attr:MK": True,
    }
    assert pattern_metavars(RMS_NORM_FOLD.lhs) == {
        "u",
        "w",
        "P",
        "EPS",
        "$attr:MD",
        "$attr:MK",
    }
    term = instantiate_pattern(RMS_NORM_FOLD.lhs, subst)
    found = match_pattern(RMS_NORM_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(RMS_NORM_FOLD.lhs, found) == term
    # the RHS instantiates to the fused rms_norm — the reduce dims
    # becoming the trailing normalized_shape tuple, the bound Const
    # unwrapped to the float eps attr.
    rhs_subst = {**found, **RMS_NORM_FOLD.derive(found)}
    rhs = instantiate_pattern(RMS_NORM_FOLD.rhs, rhs_subst)
    assert rhs == Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)

def test_rms_norm_fold_nogain_match_instantiate_roundtrip():
    """The gain-free LHS — same metvars minus ``w``."""
    u = Var("u", TensorType((4, 8)))
    subst = {
        "u": u,
        "P": Const(2),
        "EPS": Const(1e-6),
        "$attr:MD": (-1,),
        "$attr:MK": True,
    }
    term = instantiate_pattern(RMS_NORM_FOLD_NOGAIN.lhs, subst)
    found = match_pattern(RMS_NORM_FOLD_NOGAIN.lhs, term)
    assert found == subst
    rhs_subst = {**found, **RMS_NORM_FOLD_NOGAIN.derive(found)}
    rhs = instantiate_pattern(RMS_NORM_FOLD_NOGAIN.rhs, rhs_subst)
    assert rhs == Op.make("rms_norm", u, dim=(8,), eps=1e-6)

def test_check_rms_consts_cond_verdicts():
    """The cond front: keepdim, numeric-Const eps, exponent 2."""
    ok = _bound((4, 8))
    assert _check_rms_consts(ok)
    assert not _check_rms_consts({**ok, "$attr:MK": False})
    assert not _check_rms_consts({**ok, "EPS": Var("e", TensorType(()))})
    assert not _check_rms_consts({**ok, "P": Const(3)})
    assert not _check_rms_consts({**ok, "P": Var("p", TensorType(()))})

def test_rms_normalized_shape_accepts_trailing_blocks():
    """Tuple/int reduce dims over the last k axes mint ns = u.shape[-k:]."""
    assert _rms_normalized_shape(_bound((4, 8), dims=(-1,))) == (8,)
    assert _rms_normalized_shape(_bound((4, 8), dims=-1)) == (8,)
    assert _rms_normalized_shape(_bound((4, 8), dims=(1,))) == (8,)
    assert _rms_normalized_shape(_bound((2, 4, 8), dims=(-2, -1))) == (
        4,
        8,
    )
    assert _rms_normalized_shape(_bound((2, 4, 8), dims=(0, 1, 2))) == (
        2,
        4,
        8,
    )
    assert _rms_normalized_shape(_bound((8,), dims=(-1,))) == (8,)

def test_rms_normalized_shape_declines():
    """Non-trailing, duplicate, out-of-range, non-int, or unshaped."""
    assert _rms_normalized_shape(_bound((4, 8), dims=(0,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(-1, -1))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(-3,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(2,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=())) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=None)) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=True)) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(True,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims="x")) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(1.5,))) is None
    # u's shape must be concrete — it IS the minted attr
    assert _rms_normalized_shape(_bound((None, 8))) is None
    assert _rms_normalized_shape(_bound((4, None))) is None

def test_check_rms_fold_weight_shape_gate():
    """The gained fold: ``w``'s shape must BE the normalized shape."""
    assert _check_rms_fold(_bound((4, 8), w_shape=(8,)))
    assert not _check_rms_fold(_bound((4, 8), w_shape=(4,)))
    assert not _check_rms_fold(_bound((4, 8), w_shape=(4, 8)))
    # the (None,)-dim weight cannot prove ns — strict posture
    assert not _check_rms_fold(_bound((4, 8), w_shape=(None,)))
    # multi-dim: w must cover the whole trailing block
    b = _bound((2, 4, 8), w_shape=(4, 8), dims=(-2, -1))
    assert _check_rms_fold(b)
    assert not _check_rms_fold(
        _bound((2, 4, 8), w_shape=(8,), dims=(-2, -1))
    )
    # ...but the gain-free fold fires on the same dims regardless of w
    assert _check_rms_fold_nogain(b)

def test_derive_rms_norm_mints_shape_and_eps():
    """``derive`` returns ``dim`` = u.shape[-k:] and the unwrapped eps."""
    b = _bound((2, 4, 8), dims=(-2, -1), eps=1e-5)
    assert _derive_rms_norm(b) == {
        "$attr:ND": (4, 8),
        "$attr:EP": 1e-5,
    }
    # the int exponent form floats the eps too
    b = _bound((4, 8), eps=0)
    assert _derive_rms_norm(b) == {"$attr:ND": (8,), "$attr:EP": 0.0}
    # defensive vetoes — the hook may be called on declined bindings
    assert _derive_rms_norm(_bound((4, 8), dims=(0,))) is None
    assert (
        _derive_rms_norm({**_bound((4, 8)), "EPS": Var("e", TensorType(()))})
        is None
    )
