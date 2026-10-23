"""Prior-art law bank — known compiler transforms as certified laws.

A *scoped* encoding of the transforms the literature and the major
graph compilers already own, spelled in the same ``R(...)`` pattern
language as the rest of the bank.  Three sources, in the order the
sections appear:

1. **TASO's substitution set** (SOSP'19) — the canonical ~161
   graph substitutions: concat associativity and split/merge
   round-trips, pointwise pulls through concat, the matmul concat
   distributions (row/column concat and the split-K decomposition),
   fused-projection merges, and the ``linear`` bias epilogue fold
   (Inductor's ``addmm``).
2. **High-frequency Inductor/XLA patterns** — no-op elimination
   (view/relayout/cast strips), constant folding and strength
   reductions, activation decompositions (the epilogue-expand
   spellings), reduction identities and the ``std``/``var``/
   ``linalg_vector_norm`` algebra.
3. **Zoo-corpus transforms** — the named equivalences the
   ``mlopt_discovery.zoo`` workload families exercise: the HF
   rotate-half ↔ Llama cat-pair rotary equivalence, the quant
   STE value idiom, per-channel (de)quant scale migration into a
   weight, the AdaLN-Zero ``1+s`` idiom, eval-mode dropout strips,
   the one-hot → embedding gather fold, the layer_norm
   decomposition, and the ``instance_norm``/``group_norm``
   definitional expands (the fused norm ops' manual spellings —
   oneDNN's fused-norm kernels, Inductor's group_norm fusion).

What is deliberately NOT here — the honest miss list (see
``docs/results_prior_art_bank.md`` for the per-law account):

* ``mul(x, 0) = 0``, ``div(x, x) = 1``, ``sub(x, x) = 0`` — IEEE-unsound
  at NaN/Inf (XLA only applies these under fast-math).
* ``exp(log x) = x`` — domain-unsound for ``x <= 0``.
* ``pow(pow(x,a),b) = pow(x,a*b)`` — unsound for negative bases.
* ``pow`` of a negative-Const base; ``xlogy(x,y) = x*log(y)`` —
  unsound at ``x = 0`` (``xlogy`` pins 0, the product NaNs).
* ``view``-headed laws (no eval binding), ``max``/``min`` dim-forms
  (tuple-valued — a different typing class), ``narrow`` (binding is
  broken upstream), ``conv`` kernel-shape merges that need a
  rank-changing pad the pattern language cannot mint (the
  ``group_norm`` decomp *is* carried — its reshape target is
  attr-derived off the bound leaves, not pattern-minted).

Every rule is tagged :data:`mlopt_core.laws.tags.PRIOR_ART` and held
out of ``ALL_RULES``/``DEFAULT`` — the bank composes through the
``prior_art`` preset only (the ``BATCH_NORM_RULES`` /
``CONV_FOLD_RULES`` posture: opt-in reach, no default-set change).
These are all HUMAN_AUTHORED / PRIOR_ART class: none of them is a
machine admission, so no name joins
:mod:`mlopt_core.laws.provenance`'s ``MACHINE_*`` sets — they
classify ``"human"`` by registry default.
"""

# ruff: noqa: RUF001 RUF002 RUF003 -- law strings use math notation
# (σ, ⊙, ↔, ŝ) deliberately; ASCII would misstate it.

import math
from typing import Any

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op
from mlopt_core.laws import tags
from mlopt_core.laws.base import R, _shape_of

__all__ = ["PRIOR_ART_RULES"]

#: Tag bundles — every rule here is ``PRIOR_ART``; the second tag is
#: the structural class.  ``EXPANSIVE`` marks the bidirectional
#: commutation/flatten pairs so the bounded-saturation budget applies
#: when the preset is composed into a search.
_PA = (tags.PRIOR_ART,)
_PA_SIM = (tags.PRIOR_ART, tags.SIMPLIFICATION)
_PA_FUS = (tags.PRIOR_ART, tags.FUSION)
_PA_CAT = (tags.PRIOR_ART, tags.CATEGORICAL)
_PA_EXP = (tags.PRIOR_ART, tags.EXPANSIVE)


# ---------------------------------------------------------------------------
#  Shared guards and helpers
# ---------------------------------------------------------------------------


def _aval(bound: dict, name: str) -> Any:
    return bound.get(f"$attr:{name}")


def _check_unsq_sq(bound: dict) -> bool:
    """squeeze(unsqueeze(u, UD), SD): the squeezed axis IS the inserted one.

    Both dims normalize ``mod rank(u)+1`` — the rank of the unsqueezed
    tensor.  Same-axis pairs (any spelling of the same normalized
    axis) collapse back to ``u``.
    """
    s = _shape_of(bound.get("u"))
    ud, sd = _aval(bound, "UD"), _aval(bound, "SD")
    if not (
        isinstance(s, tuple)
        and isinstance(ud, int)
        and isinstance(sd, int)
    ):
        return False
    n = len(s) + 1
    return ud % n == sd % n


def _check_sq_unsq(bound: dict) -> bool:
    """unsqueeze(squeeze(u, D), D): sound only when u[D] was 1."""
    s = _shape_of(bound.get("u"))
    d = _aval(bound, "D")
    if not (isinstance(s, tuple) and s and isinstance(d, int)):
        return False
    ext = s[d % len(s)]
    return isinstance(ext, int) and ext == 1


def _check_squeeze_all_noop(bound: dict) -> bool:
    """squeeze(u) removes EVERY size-1 axis — a no-op iff none exist."""
    s = _shape_of(bound.get("u"))
    return isinstance(s, tuple) and all(
        isinstance(d, int) and d != 1 for d in s
    )


def _check_expand_noop(bound: dict) -> bool:
    """expand(u, S) is a no-op when S is u's shape (``-1`` = keep-dim)."""
    s = _shape_of(bound.get("u"))
    tgt = _aval(bound, "S")
    if not (
        isinstance(s, tuple)
        and isinstance(tgt, (tuple, list))
        and len(tgt) == len(s)
    ):
        return False
    return all(t == -1 or t == d for t, d in zip(tgt, s, strict=True))


def _check_permute_id(bound: dict) -> bool:
    """permute(u, P) is a no-op when P is the identity permutation."""
    s = _shape_of(bound.get("u"))
    p = _aval(bound, "P")
    return (
        isinstance(s, tuple)
        and isinstance(p, (tuple, list))
        and tuple(p) == tuple(range(len(s)))
    )


def _check_flatten_noop(bound: dict) -> bool:
    """flatten(u, s, e) is a no-op when the merged range is one axis."""
    s = _shape_of(bound.get("u"))
    sd = _aval(bound, "SD")
    ed = _aval(bound, "ED")
    if not (isinstance(s, tuple) and s and isinstance(sd, int)):
        return False
    ed = -1 if ed is None else ed
    if not isinstance(ed, int):
        return False
    n = len(s)
    return sd % n == ed % n


def _check_roll_noop(bound: dict) -> bool:
    """roll(u, shifts) is a no-op when every shift is 0."""
    sh = _aval(bound, "SH")
    if isinstance(sh, int):
        return sh == 0
    return isinstance(sh, (tuple, list)) and all(v == 0 for v in sh)


def _check_pad_noop(bound: dict) -> bool:
    """pad(u, P) is a no-op when every pad entry is 0."""
    p = _aval(bound, "P")
    return isinstance(p, (tuple, list)) and all(v == 0 for v in p)


def _check_repeat_noop(bound: dict) -> bool:
    """repeat(u, S) is a no-op under the full-cover guard.

    Every factor must be 1 AND S must cover exactly u's rank — a
    longer repeat prepends axes (repeat left-pads the rank).
    """
    s = _shape_of(bound.get("u"))
    p = _aval(bound, "S")
    return (
        isinstance(s, tuple)
        and isinstance(p, (tuple, list))
        and len(p) == len(s)
        and all(v == 1 for v in p)
    )


def _check_flip_inv(bound: dict) -> bool:
    """flip(flip(u, D), D) — the same dim TUPLE reverses twice.

    The eval binding iterates the dims attr — an int dims can't
    spell flip, so the guard needs the tuple spelling.
    """
    d0, d1 = _aval(bound, "D1"), _aval(bound, "D2")
    return (
        isinstance(d0, (tuple, list))
        and all(isinstance(v, int) for v in d0)
        and d0 == d1
    )


def _check_rope_equiv(bound: dict) -> bool:
    """Check the rotate-half ↔ cat-pair rotary equivalence's shape half.

    The two spellings are equal exactly when the rotate-half's
    ``concat(neg(x2), x1)`` splits the same axis the cat form writes —
    contiguous ``[0,h)`` + ``[h,2h)`` — and the factor tables are the
    half-table spelled ``concat(t, t, axis)`` (the HF ``repeat(2)``
    table).  The table terms being shared metavars is structural; the
    residual guard is the half-split on ``x``'s concat axis.
    """
    xs = _shape_of(bound.get("x"))
    if not isinstance(xs, tuple) or not xs:
        return False
    cd, s1d, s2d = (
        _aval(bound, "CD"),
        _aval(bound, "I1D"),
        _aval(bound, "I2D"),
    )
    if not all(isinstance(d, int) for d in (cd, s1d, s2d)):
        return False
    nd = len(xs)
    # The slice/concat eval bindings use the dim RAW (no ``% rank``
    # normalization — a negative dim shifts the tuple index), so the
    # guard requires the nonneg literal axis spelling.
    axis = cd
    if not 0 <= axis < nd or s1d != axis or s2d != axis:
        return False
    width = xs[axis]
    if not isinstance(width, int) or width % 2 != 0:
        return False
    h = width // 2
    a1, e1 = _aval(bound, "I1A"), _aval(bound, "I1E")
    a2, e2 = _aval(bound, "I2A"), _aval(bound, "I2E")
    if not all(isinstance(v, int) for v in (a1, e1, a2)):
        return False
    return (
        a1 == 0
        and e1 == h
        and a2 == h
        and isinstance(e2, int)
        and e2 >= width
    )


def _check_linear_out_channel(bound: dict) -> bool:
    """``c`` scales linear's OUTPUT channel: scalar or (…, W[0])."""
    w, c = _shape_of(bound.get("W")), _shape_of(bound.get("c"))
    if not (isinstance(w, tuple) and len(w) == 2):
        return False
    if not isinstance(c, tuple) or not c:
        return False
    return c[-1] is None or w[0] is None or c[-1] == w[0]


def _derive_mean_count(bound: dict) -> dict | None:
    """Derive the ``mean`` divisor: the product of the reduced extents."""
    s = _shape_of(bound.get("u"))
    d = _aval(bound, "D")
    if not (isinstance(s, tuple) and s):
        return None
    dims = d if isinstance(d, (tuple, list)) else (d,)
    n = 1
    for dd in dims:
        if not isinstance(dd, int):
            return None
        ext = s[dd % len(s)]
        if not isinstance(ext, int):
            return None
        n *= ext
    return {"N": Const(n)}


def _check_mean_count(bound: dict) -> bool:
    """Check the fold direction: the Const divisor IS the reduced count."""
    c = bound.get("C")
    got = _derive_mean_count(bound)
    return (
        isinstance(c, Const)
        and got is not None
        and c.value == got["N"].value
    )


def _derive_ln(bound: dict) -> dict | None:
    """Dims + eps leaf for the layer_norm expansion.

    ``RD`` mints the trailing-block reduce-dims tuple (an ``$attr:``
    binding — ``normalized_shape`` is the *shape*, the reduce wants
    the axes), and ``EPS`` mints the Const leaf the expanded ``add``
    needs (the kernel's ``eps`` is an attr, not a leaf).
    """
    s = _shape_of(bound.get("u"))
    nd = _aval(bound, "ND")
    eps = _aval(bound, "E")
    if not (
        isinstance(s, tuple)
        and isinstance(nd, (tuple, list))
        and 0 < len(nd) <= len(s)
        and all(isinstance(d, int) for d in nd)
        and tuple(s[len(s) - len(nd) :]) == tuple(nd)
        and isinstance(eps, (int, float))
    ):
        return None
    return {
        "$attr:RD": tuple(range(len(s) - len(nd), len(s))),
        "EPS": Const(float(eps)),
    }


def _check_ln(bound: dict) -> bool:
    return _derive_ln(bound) is not None


#: The broadcast-grid guard shared by the select-distribution laws —
#: identical to ``select_mul``'s: the paired axis must agree (the
#: selected index reads the same row) and the pair must broadcast.
_COND_SEL_PAIR = (
    "and",
    ("dim-eq-attr", "u", "D", "v", "D"),
    ("shaped", ("bcast", "u", "v")),
)

#: F.linear(x, W) denotes — x @ W.T — and W is a matrix (F.linear
#: contract); the bias epilogue's partner guards read the output grid.
_SPEC_WT = ("transpose-out", "W", None, None)
_COND_LINEAR_XW = (
    "and",
    ("mm-shape-ok", "x", _SPEC_WT),
    ("rank", "W", "==", 2),
)
_SPEC_LINEAR_OUT = ("mm-out", "x", _SPEC_WT)

_COND_LINEAR_BIAS = (
    "and",
    _COND_LINEAR_XW,
    ("bcast-into", "b", _SPEC_LINEAR_OUT),
    ("rank", "b", "<=", 1),
)

#: Dim-1 reduce keepdim=True is the identity; keepdim=False squeezes.
#: ``D`` is the spelled dim attr — int or a singleton tuple.
_COND_DIM1 = (
    "or",
    ("attr-cmp-dim", 1, "==", "u", "D"),
    (
        "and",
        ("attr-len", "D", "==", 1),
        ("attr-cmp-dim", 1, "==", "u", ("attr0", "D")),
    ),
)

#: The squeeze axis for the drop half: D as int, or D[0] for tuples.
_DSPEC_SQ_DIM = {
    "SD": (
        "if",
        ("attr-type", "D", "int"),
        ("attr", "D"),
        ("attr0", "D"),
    )
}


# ---------------------------------------------------------------------------
#  §1 — TASO substitutions: concat structure
# ---------------------------------------------------------------------------

CONCAT_SINGLE = R(
    "concat_single",
    Op.make("concat", "u", dim="D"),
    "u",
    law="concat of one tensor is the tensor (TASO cat-noop).",
    tags=_PA_SIM,
)

CONCAT_FLAT_L = R(
    "concat_flat_l",
    Op.make(
        "concat", Op.make("concat", "a", "b", dim="D"), "c", dim="D"
    ),
    Op.make("concat", "a", "b", "c", dim="D"),
    law="concat is associative: cat(cat(a,b),c) = cat(a,b,c) — the "
    "n-ary flatten (TASO concat associativity; the KV-cache append "
    "chain).",
    tags=_PA_EXP,
)

CONCAT_FLAT_R = R(
    "concat_flat_r",
    Op.make(
        "concat", "a", Op.make("concat", "b", "c", dim="D"), dim="D"
    ),
    Op.make("concat", "a", "b", "c", dim="D"),
    law="concat(a, cat(b,c)) = cat(a,b,c) — right-nested flatten.",
    tags=_PA_EXP,
)

CONCAT_CHUNK2 = R(
    "concat_chunk2",
    Op.make(
        "concat",
        Op.make("chunk", "u", chunks=2, dim="D", index=0),
        Op.make("chunk", "u", chunks=2, dim="D", index=1),
        dim="D",
    ),
    "u",
    law="cat(chunk0(u), chunk1(u)) = u — the split/merge round-trip "
    "(TASO); holds for odd extents too since chunk splits "
    "first-big.",
    tags=_PA_SIM,
)

CONCAT_CHUNK3 = R(
    "concat_chunk3",
    Op.make(
        "concat",
        Op.make("chunk", "u", chunks=3, dim="D", index=0),
        Op.make("chunk", "u", chunks=3, dim="D", index=1),
        Op.make("chunk", "u", chunks=3, dim="D", index=2),
        dim="D",
    ),
    "u",
    law="The 3-way split/merge round-trip.",
    tags=_PA_SIM,
)

CONCAT_SPLIT2 = R(
    "concat_split2",
    Op.make(
        "concat",
        Op.make("split", "u", sizes="SZ", dim="D", index=0),
        Op.make("split", "u", sizes="SZ", dim="D", index=1),
        dim="D",
    ),
    "u",
    law="cat(split0(u), split1(u)) = u — the uneven split's "
    "round-trip (two sizes cover the axis).",
    cond=("attr-len", "SZ", "==", 2),
    tags=_PA_SIM,
)

STACK_FOLD = R(
    "stack_fold",
    Op.make(
        "concat",
        Op.make("unsqueeze", "a", dim="D"),
        Op.make("unsqueeze", "b", dim="D"),
        dim="D",
    ),
    Op.make("stack", "a", "b", dim="D"),
    law="cat(unsqueeze(a,d), unsqueeze(b,d), d) = stack(a,b,d) — "
    "torch.stack IS the cat of the unsqueezed pair.",
    tags=_PA_SIM,
)

STACK_EXPAND = R(
    "stack_expand",
    Op.make("stack", "a", "b", dim="D"),
    Op.make(
        "concat",
        Op.make("unsqueeze", "a", dim="D"),
        Op.make("unsqueeze", "b", dim="D"),
        dim="D",
    ),
    law="stack(a,b,d) decomposes to the unsqueeze+concat pair.",
    tags=_PA_EXP,
    derivation=("stack_fold",),
)


def _concat_pull(op: str) -> Rewrite:
    return R(
        f"concat_pull_{op}",
        Op.make(
            "concat",
            Op.make(op, "a"),
            Op.make(op, "b"),
            dim="D",
        ),
        Op.make(op, Op.make("concat", "a", "b", dim="D")),
        law=f"cat({op}(a), {op}(b)) = {op}(cat(a,b)) — the pointwise "
        "op commutes with the partition (TASO concat-pointwise).",
        tags=_PA_SIM,
    )


CONCAT_PULL_UNARY: list[Rewrite] = [
    _concat_pull(op)
    for op in (
        "neg",
        "abs",
        "relu",
        "sigmoid",
        "silu",
        "tanh",
        "gelu",
        "exp",
        "square",
    )
]


def _concat_pull2(op: str) -> Rewrite:
    return R(
        f"concat_pull_{op}",
        Op.make(
            "concat",
            Op.make(op, "a1", "b1"),
            Op.make(op, "a2", "b2"),
            dim="D",
        ),
        Op.make(
            op,
            Op.make("concat", "a1", "a2", dim="D"),
            Op.make("concat", "b1", "b2", dim="D"),
        ),
        law=f"cat({op}(a1,b1), {op}(a2,b2)) = {op}(cat(a1,a2), "
        "cat(b1,b2)) — the pointwise pair commutes with the "
        "partition.",
        tags=_PA_SIM,
    )


CONCAT_PULL_BINARY: list[Rewrite] = [
    _concat_pull2(op) for op in ("add", "mul", "sub")
]


# ---------------------------------------------------------------------------
#  §1 — TASO: matmul/linear concat merges and the split-K decomposition
# ---------------------------------------------------------------------------

#: The concat axis must not be the contraction axis of the left
#: operand (its last axis); every other axis distributes the product.
_COND_MM_CL = (
    "and",
    ("dim-eq", "a", -1, "b", -1),
    ("not", ("axis", "a", "CD", -1)),
)

MM_CONCAT_LEFT = R(
    "mm_concat_left",
    Op.make(
        "concat",
        Op.make("matmul", "a", "W"),
        Op.make("matmul", "b", "W"),
        dim="CD",
    ),
    Op.make("matmul", Op.make("concat", "a", "b", dim="CD"), "W"),
    law="cat(a@W, b@W) = cat(a,b)@W on a non-contraction axis — "
    "row/batch blocks of a shared contraction merge into one matmul "
    "(TASO matmul-concat).",
    cond=_COND_MM_CL,
    tags=_PA_FUS,
)

MM_CONCAT_LEFT_REV = R(
    "mm_concat_left_rev",
    Op.make("matmul", Op.make("concat", "a", "b", dim="CD"), "W"),
    Op.make(
        "concat",
        Op.make("matmul", "a", "W"),
        Op.make("matmul", "b", "W"),
        dim="CD",
    ),
    law="Reverse: split a matmul's non-contraction axis into two "
    "products (eqsat weighs split vs merged).",
    cond=_COND_MM_CL,
    tags=_PA_EXP,
    derivation=("mm_concat_left",),
)

#: The right-operand concat must sit on the OUTPUT axis (its last
#: axis); the contraction axis (-2) must agree AND the shared
#: left operand's contraction extent must match it.
#: The output concat dim is pinned to ``-1``: for a rank-1 output
#: (vector ``x``) a spelled ``dim=1`` normalizes to the same axis
#: under the guard's ``% rank`` but is out-of-range for
#: ``torch.cat`` — the literal ``-1`` is valid on every output rank.
_COND_MM_CR = (
    "and",
    ("rank", "x", "<=", 2),
    ("rank", "b", "==", 2),
    ("rank", "c", "==", 2),
    ("attr-eq", "CD", -1),
    ("axis", "b", "CD", -1),
    ("dim-eq", "b", -2, "c", -2),
    ("dim-eq", "x", -1, "b", -2),
)

MM_CONCAT_RIGHT = R(
    "mm_concat_right",
    Op.make(
        "concat",
        Op.make("matmul", "x", "b"),
        Op.make("matmul", "x", "c"),
        dim="CD",
    ),
    Op.make("matmul", "x", Op.make("concat", "b", "c", dim="CD")),
    law="cat(x@b, x@c) = x@cat(b,c) along the output axis — column "
    "blocks of a shared input merge into one matmul.",
    cond=_COND_MM_CR,
    tags=_PA_FUS,
)

MM_CONCAT_RIGHT_REV = R(
    "mm_concat_right_rev",
    Op.make("matmul", "x", Op.make("concat", "b", "c", dim="CD")),
    Op.make(
        "concat",
        Op.make("matmul", "x", "b"),
        Op.make("matmul", "x", "c"),
        dim="CD",
    ),
    law="Reverse column-block split (eqsat weighs both forms).",
    cond=_COND_MM_CR,
    tags=_PA_EXP,
    derivation=("mm_concat_right",),
)

#: Split-K: matmul(concat(a,b,-1), W) = a@W_top + b@W_bot where the
#: split sizes are the two operands' contraction extents.  Scoped
#: to the rank-2 GEMM — the batched-matmul batch-dim agreement
#: isn't expressible in the pattern language.
_COND_MM_SPLITK = (
    "and",
    ("rank", "a", "==", 2),
    ("rank", "b", "==", 2),
    ("dim-eq", "a", -2, "b", -2),
    ("rank", "W", "==", 2),
    ("axis", "a", "CD", -1),
    (
        "attr-cmp",
        ("dim", "W", -2),
        "==",
        ("add", ("dim", "a", -1), ("dim", "b", -1)),
    ),
)

_DSPEC_SPLITK = {
    "SZ": ("tuple", ("dim", "a", -1), ("dim", "b", -1)),
}


def _mm_splitk_rhs() -> Op:
    w_top = Op.make("split", "W", sizes="SZ", dim=-2, index=0)
    w_bot = Op.make("split", "W", sizes="SZ", dim=-2, index=1)
    return Op.make(
        "add",
        Op.make("matmul", "a", w_top),
        Op.make("matmul", "b", w_bot),
    )


MM_SPLITK = R(
    "mm_splitk",
    Op.make("matmul", Op.make("concat", "a", "b", dim="CD"), "W"),
    _mm_splitk_rhs(),
    law="(cat(a,b,-1)) @ W = a@W_top + b@W_bot — the split-K "
    "decomposition: the contraction axis partitions, the weight "
    "splits on its -2 axis (TASO).",
    cond=_COND_MM_SPLITK,
    dspec=_DSPEC_SPLITK,
    tags=_PA_CAT,
)

MM_SPLITK_REV = R(
    "mm_splitk_rev",
    Op.make(
        "add",
        Op.make(
            "matmul",
            "a",
            Op.make("split", "W", sizes="SZ", dim=-2, index=0),
        ),
        Op.make(
            "matmul",
            "b",
            Op.make("split", "W", sizes="SZ", dim=-2, index=1),
        ),
    ),
    Op.make("matmul", Op.make("concat", "a", "b", dim=-1), "W"),
    law="The split-K refold — the two partial products merge into "
    "one contraction.",
    cond=(
        "and",
        ("rank", "a", "==", 2),
        ("rank", "b", "==", 2),
        ("dim-eq", "a", -2, "b", -2),
        ("rank", "W", "==", 2),
        (
            "attr-cmp",
            "SZ",
            "==",
            ("tuple", ("dim", "a", -1), ("dim", "b", -1)),
        ),
        (
            "attr-cmp",
            ("dim", "W", -2),
            "==",
            ("add", ("dim", "a", -1), ("dim", "b", -1)),
        ),
    ),
    tags=_PA_EXP,
    derivation=("mm_splitk",),
)

#: linear(x, cat(A,B,0)) = cat(linear(x,A), linear(x,B), -1) — the
#: fused-projection distribution (TASO; the qkv_fuse class).  The
#: output concat rides the operand rank — guarded to the last axis.
_COND_LIN_CAT = (
    "and",
    ("rank", "A", "==", 2),
    ("rank", "B", "==", 2),
    ("dim-eq", "A", 1, "B", 1),
    ("axis", "x", "CD", -1),
)

LINEAR_CONCAT_W = R(
    "linear_concat_w",
    Op.make(
        "concat",
        Op.make("linear", "x", "A"),
        Op.make("linear", "x", "B"),
        dim="CD",
    ),
    Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
    law="cat(xAᵀ, xBᵀ) = x[A;B]ᵀ on the output axis — the "
    "MergedColumnParallelLinear merge (TASO).",
    cond=_COND_LIN_CAT,
    tags=_PA_FUS,
)

LINEAR_CONCAT_W_REV = R(
    "linear_concat_w_rev",
    Op.make("linear", "x", Op.make("concat", "A", "B", dim="AD")),
    Op.make(
        "concat",
        Op.make("linear", "x", "A"),
        Op.make("linear", "x", "B"),
        dim=-1,
    ),
    law="Reverse: split a fused projection into per-block linears.",
    cond=(
        "and",
        ("rank", "A", "==", 2),
        ("rank", "B", "==", 2),
        ("dim-eq", "A", 1, "B", 1),
        ("axis", "A", "AD", 0),
    ),
    tags=_PA_EXP,
    derivation=("linear_concat_w",),
)

#: The addmm / bias-epilogue fold — add(linear(x,W), b) folds the
#: broadcast add into the linear's bias slot.
LINEAR_BIAS_FOLD = R(
    "linear_bias_fold",
    Op.make("add", Op.make("linear", "x", "W"), "b"),
    Op.make("linear", "x", "W", "b"),
    law="xWᵀ + b = linear(x, W, b) — the bias epilogue folds into "
    "the GEMM (Inductor's addmm; TASO's matmul+bias).",
    cond=_COND_LINEAR_BIAS,
    tags=_PA_FUS,
)

LINEAR_BIAS_FOLD_L = R(
    "linear_bias_fold_l",
    Op.make("add", "b", Op.make("linear", "x", "W")),
    Op.make("linear", "x", "W", "b"),
    law="The operand-swapped bias-epilogue fold.",
    cond=_COND_LINEAR_BIAS,
    tags=_PA_FUS,
)

LINEAR_BIAS_EXPAND = R(
    "linear_bias_expand",
    Op.make("linear", "x", "W", "b"),
    Op.make("add", Op.make("linear", "x", "W"), "b"),
    law="Reverse: expose the fused bias as a pointwise epilogue "
    "(eqsat weighs fused vs spelled).",
    cond=_COND_LINEAR_BIAS,
    tags=_PA_EXP,
    derivation=("linear_bias_fold",),
)

_COND_LIN_BIAS_MERGE = (
    "and",
    ("rank", "A", "==", 2),
    ("rank", "B", "==", 2),
    ("shape-compat", "A", "B"),
    ("shape-compat", "b1", "b2"),
)

LINEAR_BIAS_MERGE = R(
    "linear_bias_merge",
    Op.make(
        "add",
        Op.make("linear", "x", "A", "b1"),
        Op.make("linear", "x", "B", "b2"),
    ),
    Op.make(
        "linear",
        "x",
        Op.make("add", "A", "B"),
        Op.make("add", "b1", "b2"),
    ),
    law="linear(x,A,b1) + linear(x,B,b2) = linear(x, A+B, b1+b2) — "
    "the affine-map weight merge (model-soup / LoRA-merge class).",
    cond=_COND_LIN_BIAS_MERGE,
    tags=_PA_FUS,
)


# ---------------------------------------------------------------------------
#  §1 — TASO: conv output merge (the Inception-style parallel fold)
# ---------------------------------------------------------------------------

#: conv2d's positional attr tail — mirror of tensor.py's table; the
#: shared metavar names bind both convs' attrs to the SAME spelling.
_CONV_ATTR_MV = {
    "stride": "ST",
    "padding": "PD",
    "dilation": "DL",
    "groups": "GR",
}
_CONV_ATTR_SUFFIX = ("", "_sp", "_spdg")


def _conv_pat(args: tuple, attrs: tuple) -> Op:
    kw: dict[str, Any] = {k: _CONV_ATTR_MV[k] for k in attrs}
    return Op.make("conv2d", *args, **kw)


def _conv_merge_cond(attrs: tuple, bias: bool) -> Any:
    if "groups" in attrs:
        # groups>1 needs a channel interleave, not a plain cat —
        # scoped to groups==1.
        inc = (
            "and",
            ("attr-eq", "GR", 1),
            ("dim-eq", "x", 1, "W1", 1),
        )
    else:
        inc = ("dim-eq", "x", 1, "W1", 1)
    return (
        "and",
        ("concrete", "x"),
        ("rank", "x", "==", 4),
        ("concrete", "W1"),
        ("rank", "W1", "==", 4),
        inc,
        ("dim-eq", "W1", 1, "W2", 1),
        # the output concat rides the channel axis (output rank = x's)
        ("axis", "x", "CD", 1),
        *(
            (
                ("rank", "b1", "==", 1),
                ("dim-eq", "b1", 0, "W1", 0),
                ("rank", "b2", "==", 1),
                ("dim-eq", "b2", 0, "W2", 0),
            )
            if bias
            else ()
        ),
    )


def _make_conv_merge_rules() -> list[Rewrite]:
    """cat(conv(x,W1), conv(x,W2), ch) = conv(x, cat(W1,W2,0)).

    The output-channel concat merge — TASO's conv-merge (the
    Inception parallel fold): two same-geometry convs over a shared
    input are one conv over the output-concat'd filter bank.  The
    concat axis is the output's channel axis (axis 1 of a rank-4
    conv output).
    """
    out: list[Rewrite] = []
    tails = ((), ("stride", "padding"), _CONV_ATTR_MV.keys())
    for na, suf in enumerate(_CONV_ATTR_SUFFIX):
        attrs = tuple(tails[na])
        for bias in (True, False):
            bs = "" if bias else "_nobias"
            args1 = ("x", "W1", *(("b1",) if bias else ()))
            args2 = ("x", "W2", *(("b2",) if bias else ()))
            merged_args = (
                "x",
                Op.make("concat", "W1", "W2", dim=0),
                *(
                    (Op.make("concat", "b1", "b2", dim=0),)
                    if bias
                    else ()
                ),
            )
            out.append(
                R(
                    f"conv_out_merge{suf}{bs}",
                    Op.make(
                        "concat",
                        _conv_pat(args1, attrs),
                        _conv_pat(args2, attrs),
                        dim="CD",
                    ),
                    _conv_pat(merged_args, attrs),
                    law="cat(conv(x,W1), conv(x,W2)) = "
                    "conv(x, cat(W1,W2,0)) on the channel axis — "
                    "same-geometry convs merge into one (TASO's "
                    "conv-merge; the Inception parallel fold).",
                    cond=_conv_merge_cond(attrs, bias),
                    tags=_PA_FUS,
                )
            )
    return out


CONV_MERGE_RULES: list[Rewrite] = _make_conv_merge_rules()


# ---------------------------------------------------------------------------
#  §1 — TASO: index-view distributions and bridges
# ---------------------------------------------------------------------------


def _select_pull(op: str) -> Rewrite:
    return R(
        f"select_{op}",
        Op.make(
            op,
            Op.make("select", "u", dim="D", index="I"),
            Op.make("select", "v", dim="D", index="I"),
        ),
        Op.make("select", Op.make(op, "u", "v"), dim="D", index="I"),
        law=f"{op} commutes with select — the select_mul guard: same "
        "index on the same axis and a denoting broadcast grid.",
        cond=_COND_SEL_PAIR,
        tags=_PA_SIM,
    )


SELECT_PULL_RULES: list[Rewrite] = [
    _select_pull(op) for op in ("add", "sub", "div")
]

UNBIND_SELECT = R(
    "unbind_select",
    Op.make("unbind", "u", dim="D", index="I"),
    Op.make("select", "u", dim="D", index="I"),
    law="unbind(u, d)[i] IS select(u, d, i) — the exported unbind "
    "carries its picked index (the getitem-folded spelling).",
    tags=_PA_SIM,
)

#  Dropped (see also the block by MAXIMUM_IDEM below):
#    getitem_select — u[i] = select(u,0,i) holds for TENSOR u, but
#      ``u`` can also bind a tuple-returning term (``topk``/``max``
#      results) whose shape is still provable; the pattern language
#      has no tensor-vs-tuple kind guard, so the law is unguarded-
#      wrong there.

#: ``D`` is a valid NONNEGATIVE axis of ``u`` — the eval binding
#: indexes ``t[(slice(None),) * d + ...]``, which does NOT normalize
#: negative dims the way the guard-side ``_dim_at`` does
#: (``dim=-3`` on a rank-2 tensor slices axis 0 at eval while the
#: guard's ``d % len`` reads axis 1 — a real divergence the
#: nonneg-range guard closes).
_COND_AXIS = (
    "and",
    ("attr-type", "D", "int"),
    ("attr-cmp", "D", ">=", 0),
    ("attr-cmp", ("len", ("shape", "u")), ">", ("attr", "D")),
)

#: select(u, d, i) ⇄ slice — select DROPS the axis, slice keeps a
#: unit-width one, so the bridge carries a squeeze/unsqueeze pair.
_COND_SEL_RANGE = (
    "and",
    _COND_AXIS,
    ("attr-type", "I", "int"),
    ("attr-cmp", "I", ">=", 0),
    (
        "attr-cmp",
        ("nth", ("shape", "u"), ("attr", "D")),
        ">",
        ("attr", "I"),
    ),
)

SELECT_TO_SLICE = R(
    "select_to_slice",
    Op.make("select", "u", dim="D", index="I"),
    Op.make(
        "squeeze",
        Op.make("slice", "u", dim="D", start="I", end="E"),
        dim="D",
    ),
    law="select(u, d, i) = squeeze(slice(u, d, i, i+1), d) — the "
    "unit-width slice spelling of an index pick.",
    cond=_COND_SEL_RANGE,
    dspec={"E": ("add", ("attr", "I"), 1)},
    tags=_PA_SIM,
)

SLICE_TO_SELECT = R(
    "slice_to_select",
    Op.make("slice", "u", dim="D", start="S", end="E"),
    Op.make(
        "unsqueeze",
        Op.make("select", "u", dim="D", index="S"),
        dim="D",
    ),
    law="slice(u, d, i, i+1) = unsqueeze(select(u, d, i), d) — the "
    "one-element slice is the re-widened index pick.",
    cond=(
        "and",
        _COND_AXIS,
        ("attr-type", "S", "int"),
        ("attr-cmp", "S", ">=", 0),
        (
            "attr-cmp",
            ("nth", ("shape", "u"), ("attr", "D")),
            ">",
            ("attr", "S"),
        ),
        ("attr-cmp", "E", "==", ("add", ("attr", "S"), 1)),
    ),
    tags=_PA_SIM,
    derivation=("select_to_slice",),
)


# ---------------------------------------------------------------------------
#  §2 — Inductor/XLA: no-op elimination
# ---------------------------------------------------------------------------

CONTIGUOUS_ID = R(
    "contiguous_id",
    Op.make("contiguous", "u"),
    "u",
    law="contiguous(x) is a value-identity relayout — the IR's "
    "semantics is value-level, so the strip is exact.",
    tags=_PA_SIM,
)

CLONE_ID = R(
    "clone_id",
    Op.make("clone", "u"),
    "u",
    law="clone(x) preserves values exactly — a materialization, "
    "not a value change.",
    tags=_PA_SIM,
)

ALIAS_ID = R(
    "alias_id",
    Op.make("alias", "u"),
    "u",
    law="alias(x) is the identity leaf-relabel.",
    tags=_PA_SIM,
)

DETACH_ID = R(
    "detach_id",
    Op.make("detach", "u"),
    "u",
    law="detach(x) is the value identity — the autograd boundary "
    "the IR does not model (Inductor strips it in inference).",
    tags=_PA_SIM,
)

DETACH_INPLACE_ID = R(
    "detach_inplace_id",
    Op.make("detach_", "u"),
    "u",
    law="detach_(x) — the in-place spelling of the same value "
    "identity.",
    tags=_PA_SIM,
)


def _cast_idem(op: str) -> Rewrite:
    return R(
        f"{op}_idem",
        Op.make(op, Op.make(op, "u")),
        Op.make(op, "u"),
        law=f"{op}({op}(x)) = {op}(x) — dtype casts are idempotent.",
        tags=_PA_SIM,
    )


CAST_IDEM_RULES: list[Rewrite] = [
    _cast_idem(op) for op in ("float", "half", "double", "bfloat16")
]

UNSQUEEZE_SQUEEZE = R(
    "unsqueeze_squeeze",
    Op.make(
        "squeeze",
        Op.make("unsqueeze", "u", dim="UD"),
        dim="SD",
    ),
    "u",
    law="squeeze(unsqueeze(u, d), d) = u — the insert/remove pair "
    "cancels on the same normalized axis.",
    check=_check_unsq_sq,
    tags=_PA_SIM,
)

SQUEEZE_UNSQUEEZE = R(
    "squeeze_unsqueeze",
    Op.make(
        "unsqueeze",
        Op.make("squeeze", "u", dim="D"),
        dim="D",
    ),
    "u",
    law="unsqueeze(squeeze(u, d), d) = u when u[d] == 1 — the "
    "remove/insert pair cancels on a size-1 axis only.",
    check=_check_sq_unsq,
    tags=_PA_SIM,
)

SQUEEZE_NOOP = R(
    "squeeze_noop",
    Op.make("squeeze", "u", dim="D"),
    "u",
    law="squeeze(u, d) on a non-1 axis removes nothing — a no-op "
    "the exporter's belt-and-braces squeezes leave behind.",
    cond=("attr-cmp-dim", 1, "!=", "u", "D"),
    tags=_PA_SIM,
)

SQUEEZE_ALL_NOOP = R(
    "squeeze_all_noop",
    Op.make("squeeze", "u"),
    "u",
    law="squeeze(u) with no dim removes EVERY size-1 axis — a "
    "no-op when none exist.",
    check=_check_squeeze_all_noop,
    tags=_PA_SIM,
)

RESHAPE_NOOP = R(
    "reshape_noop",
    Op.make("reshape", "u", shape="S"),
    "u",
    law="reshape(u, shape(u)) = u — the no-op reshape strip "
    "(Inductor view-chain simplification).",
    cond=("attr-cmp", "S", "==", ("shape", "u")),
    tags=_PA_SIM,
)

RESHAPE_RESHAPE = R(
    "reshape_reshape",
    Op.make("reshape", Op.make("reshape", "u", shape="S0"), shape="S1"),
    Op.make("reshape", "u", shape="S1"),
    law="reshape(reshape(u, s0), s1) = reshape(u, s1) — view "
    "composition keeps only the final target.",
    tags=_PA_SIM,
)

EXPAND_NOOP = R(
    "expand_noop",
    Op.make("expand", "u", shape="S"),
    "u",
    law="expand(u, shape(u)) = u — expanding to the existing shape "
    "is a no-op (-1 keep-dims counted as identity).",
    check=_check_expand_noop,
    tags=_PA_SIM,
)

BROADCAST_TO_NOOP = R(
    "broadcast_to_noop",
    Op.make("broadcast_to", "u", shape="S"),
    "u",
    law="broadcast_to(u, shape(u)) = u — the same no-op for the "
    "materializing spelling.",
    cond=("attr-cmp", "S", "==", ("shape", "u")),
    tags=_PA_SIM,
)

BROADCAST_TO_EXPAND = R(
    "broadcast_to_expand",
    Op.make("broadcast_to", "u", shape="S"),
    Op.make("expand", "u", shape="S"),
    law="broadcast_to(u, s) = expand(u, s) — the strict spelling is "
    "a special case of the view.",
    tags=_PA_EXP,
)

SLICE_NOOP = R(
    "slice_noop",
    Op.make("slice", "u", dim="D", start=0, end="E"),
    "u",
    law="slice(u, d, 0, e) = u when e >= u[d] — the full-cover "
    "slice strip (the export's ``x[..., :e]`` no-op).",
    cond=("and", _COND_AXIS, ("attr-cmp-dim", "E", ">=", "u", "D")),
    tags=_PA_SIM,
)

SLICE_NOOP_NOSTART = R(
    "slice_noop_nostart",
    Op.make("slice", "u", dim="D", end="E"),
    "u",
    law="The start-elided full-cover slice — ``x[..., :e]`` with "
    "e >= u[d].",
    cond=("and", _COND_AXIS, ("attr-cmp-dim", "E", ">=", "u", "D")),
    tags=_PA_SIM,
)

INDEX_SELECT_ID = R(
    "index_select_id",
    Op.make("index_select", "u", dim="D", index="I"),
    "u",
    law="index_select(u, d, range(u[d])) = u — the identity gather.",
    cond=("attr-range", "I", "u", "D"),
    tags=_PA_SIM,
)

ROLL_NOOP = R(
    "roll_noop",
    Op.make("roll", "u", shifts="SH", dims="DM"),
    "u",
    law="roll(u, shifts=0) = u — a zero shift on every axis.",
    check=_check_roll_noop,
    tags=_PA_SIM,
)

PAD_NOOP = R(
    "pad_noop",
    Op.make("pad", "u", pad="P"),
    "u",
    law="pad(u, zeros) = u — an all-zero pad pad is a no-op.",
    check=_check_pad_noop,
    tags=_PA_SIM,
)

REPEAT_NOOP = R(
    "repeat_noop",
    Op.make("repeat", "u", shape="S"),
    "u",
    law="repeat(u, ones) = u — tiling by 1 on every axis.",
    check=_check_repeat_noop,
    tags=_PA_SIM,
)

PERMUTE_NOOP = R(
    "permute_noop",
    Op.make("permute", "u", dims="P"),
    "u",
    law="permute(u, identity) = u.",
    check=_check_permute_id,
    tags=_PA_SIM,
)

PERMUTE_COMPOSE = R(
    "permute_compose",
    Op.make("permute", Op.make("permute", "u", dims="P"), dims="Q"),
    Op.make("permute", "u", dims="C"),
    law="permute(permute(u,P),Q) = permute(u, P∘Q) — permutation "
    "composition, minted as gather(P, Q).",
    cond=(
        "and",
        ("attr-ints", "P"),
        ("attr-ints", "Q"),
        (
            "attr-cmp",
            ("len", ("attr", "P")),
            "==",
            ("len", ("shape", "u")),
        ),
        (
            "attr-cmp",
            ("len", ("attr", "Q")),
            "==",
            ("len", ("attr", "P")),
        ),
    ),
    dspec={"C": ("gather", ("attr", "P"), ("attr", "Q"))},
    tags=_PA_SIM,
)

MOVEDIM_INVOLUTION = R(
    "movedim_involution",
    Op.make(
        "movedim",
        Op.make("movedim", "u", source="S", destination="D"),
        source="D",
        destination="S",
    ),
    "u",
    law="movedim(movedim(u,s,d),d,s) = u — moving an axis forth "
    "and back is the identity.",
    cond=(
        "and",
        ("attr-type", "S", "int"),
        ("attr-type", "D", "int"),
    ),
    tags=_PA_SIM,
)

FLATTEN_NOOP = R(
    "flatten_noop",
    Op.make("flatten", "u", start_dim="SD", end_dim="ED"),
    "u",
    law="flatten(u, i, i) = u — a one-axis merge is a no-op.",
    check=_check_flatten_noop,
    tags=_PA_SIM,
)

UNFLATTEN_NOOP = R(
    "unflatten_noop",
    Op.make("unflatten", "u", dim="D", sizes="SZ"),
    "u",
    law="unflatten(u, d, (u[d],)) = u — a singleton split restores "
    "the same axis.",
    cond=(
        "and",
        ("attr-len", "SZ", "==", 1),
        (
            "attr-cmp",
            ("nth", ("attr", "SZ"), 0),
            "==",
            ("nth", ("shape", "u"), ("attr", "D")),
        ),
    ),
    tags=_PA_SIM,
)

SPLIT_SINGLE = R(
    "split_single",
    Op.make("split", "u", sizes="SZ", dim="D", index="I"),
    "u",
    law="split(u, (u[d],), d, 0) = u — a one-section split is the "
    "whole tensor (the chunk_single sibling).",
    cond=(
        "and",
        ("attr-eq", "I", 0),
        ("attr-len", "SZ", "==", 1),
        (
            "attr-cmp",
            ("nth", ("attr", "SZ"), 0),
            "==",
            ("nth", ("shape", "u"), ("attr", "D")),
        ),
    ),
    tags=_PA_SIM,
)

FLIP_INVOLUTION = R(
    "flip_involution",
    Op.make("flip", Op.make("flip", "u", dims="D1"), dims="D2"),
    "u",
    law="flip(flip(u, d), d) = u — reversing the same axes twice.",
    check=_check_flip_inv,
    tags=_PA_SIM,
)

TRIL_IDEM = R(
    "tril_idem",
    Op.make("tril", Op.make("tril", "u", diagonal="G"), diagonal="G"),
    Op.make("tril", "u", diagonal="G"),
    law="tril(tril(u, g), g) = tril(u, g) — masking is idempotent.",
    tags=_PA_SIM,
)

TRIU_IDEM = R(
    "triu_idem",
    Op.make("triu", Op.make("triu", "u", diagonal="G"), diagonal="G"),
    Op.make("triu", "u", diagonal="G"),
    law="triu(triu(u, g), g) = triu(u, g).",
    tags=_PA_SIM,
)

CLAMP_IDEM = R(
    "clamp_idem",
    Op.make(
        "clamp",
        Op.make("clamp", "u", min="LO", max="HI"),
        min="LO",
        max="HI",
    ),
    Op.make("clamp", "u", min="LO", max="HI"),
    law="clamp(clamp(u,l,h),l,h) = clamp(u,l,h) — same-bound "
    "nesting is a no-op.",
    tags=_PA_SIM,
)


def _red1_keep(op: str) -> Rewrite:
    return R(
        f"{op}_dim1_keep",
        Op.make(op, "u", dim="D", keepdim=True),
        "u",
        law=f"{op} over an extent-1 axis with keepdim is the "
        "identity (the reduce reads the only element).",
        cond=_COND_DIM1,
        tags=_PA_SIM,
    )


def _red1_drop(op: str) -> Rewrite:
    return R(
        f"{op}_dim1_drop",
        Op.make(op, "u", dim="D", keepdim=False),
        Op.make("squeeze", "u", dim="SD"),
        law=f"{op} over an extent-1 axis without keepdim IS "
        "squeeze(u, d) — the reduce reads the only element and "
        "drops the axis.",
        cond=_COND_DIM1,
        dspec=_DSPEC_SQ_DIM,
        tags=_PA_SIM,
    )


REDUCE_DIM1_RULES: list[Rewrite] = [
    r
    for op in ("sum", "mean", "prod", "amax", "amin")
    for r in (_red1_keep(op), _red1_drop(op))
]


# ---------------------------------------------------------------------------
#  §2 — reduction algebra: std/var/norm/log-softmax
# ---------------------------------------------------------------------------

STD_SQRT_VAR = R(
    "std_sqrt_var",
    Op.make(
        "std",
        "u",
        dim="D",
        correction="C",
        keepdim="K",
    ),
    Op.make(
        "sqrt",
        Op.make("var", "u", dim="D", correction="C", keepdim="K"),
    ),
    law="std = sqrt(var) — the same dims/correction/keepdim on both "
    "sides (XLA's std decomposition).",
    tags=_PA_SIM,
)

VAR_SQUARE_STD = R(
    "var_square_std",
    Op.make(
        "var",
        "u",
        dim="D",
        correction="C",
        keepdim="K",
    ),
    Op.make(
        "square",
        Op.make("std", "u", dim="D", correction="C", keepdim="K"),
    ),
    law="var = std² — the reverse spelling.",
    tags=_PA_EXP,
    derivation=("std_sqrt_var",),
)

STD_SQRT_VAR_BARE = R(
    "std_sqrt_var_bare",
    Op.make("std", "u"),
    Op.make("sqrt", Op.make("var", "u")),
    law="The whole-tensor spelling — std() = sqrt(var()).",
    tags=_PA_SIM,
)

VAR_SQUARE_STD_BARE = R(
    "var_square_std_bare",
    Op.make("var", "u"),
    Op.make("square", Op.make("std", "u")),
    law="The whole-tensor var = std².",
    tags=_PA_EXP,
    derivation=("std_sqrt_var_bare",),
)

NORM2_EXPAND = R(
    "norm2_expand",
    Op.make("linalg_vector_norm", "u", ord=2, dim="D", keepdim="K"),
    Op.make(
        "sqrt",
        Op.make(
            "sum",
            Op.make("square", "u"),
            dim="D",
            keepdim="K",
        ),
    ),
    law="‖u‖₂ = sqrt(Σ u²) — the L2 norm's spelled work (XLA "
    "linalg decomposition).",
    tags=_PA_EXP,
)

NORM2_FOLD = R(
    "norm2_fold",
    Op.make(
        "sqrt",
        Op.make(
            "sum",
            Op.make("square", "u"),
            dim="D",
            keepdim="K",
        ),
    ),
    Op.make("linalg_vector_norm", "u", ord=2, dim="D", keepdim="K"),
    law="sqrt(Σ u²) folds to the L2 kernel.",
    tags=_PA_SIM,
    derivation=("norm2_expand",),
)

MEAN_EXPAND = R(
    "mean_expand",
    Op.make("mean", "u", dim="D", keepdim="K"),
    Op.make(
        "div",
        Op.make("sum", "u", dim="D", keepdim="K"),
        "N",
    ),
    law="mean(u, d) = sum(u, d)/n with n the reduced extent — "
    "XLA's mean decomposition; n mints a Const leaf.",
    derive=_derive_mean_count,
    tags=_PA_EXP,
)

MEAN_FOLD = R(
    "mean_fold",
    Op.make(
        "div",
        Op.make("sum", "u", dim="D", keepdim="K"),
        "C",
    ),
    Op.make("mean", "u", dim="D", keepdim="K"),
    law="sum(u, d)/n = mean(u, d) when the Const divisor IS the "
    "reduced extent.",
    check=_check_mean_count,
    tags=_PA_SIM,
    derivation=("mean_expand",),
)

LOG_SOFTMAX_FOLD = R(
    "log_softmax_fold",
    Op.make("log", Op.make("softmax", "u", dim="D")),
    Op.make("log_softmax", "u", dim="D"),
    law="log(softmax(u, d)) = log_softmax(u, d) — the fused "
    "log-softmax fold (softcap/logprob heads).",
    tags=_PA_SIM,
)

LOG_SOFTMAX_EXPAND = R(
    "log_softmax_expand",
    Op.make("exp", Op.make("log_softmax", "u", dim="D")),
    Op.make("softmax", "u", dim="D"),
    law="exp(log_softmax(u, d)) = softmax(u, d) — the same kernel "
    "pair the other way.",
    tags=_PA_EXP,
    derivation=("log_softmax_fold",),
)


# ---------------------------------------------------------------------------
#  §2 — activation decompositions and folds (the epilogue spellings)
# ---------------------------------------------------------------------------

SIGMOID_EXPAND = R(
    "sigmoid_expand",
    Op.make("sigmoid", "x"),
    Op.make(
        "reciprocal",
        Op.make("add", Const(1), Op.make("exp", Op.make("neg", "x"))),
    ),
    law="σ(x) = 1/(1+e⁻ˣ) — the definition, expanded so pointwise "
    "laws see the interior (Inductor decomp).",
    tags=_PA_EXP,
)

SIGMOID_FOLD = R(
    "sigmoid_fold",
    Op.make(
        "reciprocal",
        Op.make("add", "ONE", Op.make("exp", Op.make("neg", "x"))),
    ),
    Op.make("sigmoid", "x"),
    law="1/(1+e⁻ˣ) folds to σ(x) — the manual-spelling fold "
    "(softmax_fold's sibling one kernel over).",
    cond=("const-cmp", "ONE", "==", 1),
    tags=_PA_SIM,
    derivation=("sigmoid_expand",),
)

SIGMOID_FOLD_DIV = R(
    "sigmoid_fold_div",
    Op.make(
        "div",
        "ONE",
        Op.make("add", "ONE2", Op.make("exp", Op.make("neg", "x"))),
    ),
    Op.make("sigmoid", "x"),
    law="The div-spelled 1/(1+e⁻ˣ) folds to σ(x).",
    cond=(
        "and",
        ("const-cmp", "ONE", "==", 1),
        ("const-cmp", "ONE2", "==", 1),
    ),
    tags=_PA_SIM,
    derivation=("sigmoid_expand",),
)

TANH_EXPAND = R(
    "tanh_expand",
    Op.make("tanh", "x"),
    Op.make(
        "sub",
        Op.make(
            "mul",
            Const(2),
            Op.make("sigmoid", Op.make("mul", Const(2), "x")),
        ),
        Const(1),
    ),
    law="tanh(x) = 2σ(2x) − 1 — the sigmoid-based decomposition "
    "(the TASO/XLA tanh spelling).",
    tags=_PA_EXP,
)

TANH_FOLD = R(
    "tanh_fold",
    Op.make(
        "sub",
        Op.make(
            "mul",
            "TWO",
            Op.make("sigmoid", Op.make("mul", "TWO2", "x")),
        ),
        "ONE",
    ),
    Op.make("tanh", "x"),
    law="2σ(2x) − 1 folds to tanh(x).",
    cond=(
        "and",
        ("const-cmp", "TWO", "==", 2),
        ("const-cmp", "TWO2", "==", 2),
        ("const-cmp", "ONE", "==", 1),
    ),
    tags=_PA_SIM,
    derivation=("tanh_expand",),
)

_GELU_ERF = Op.make(
    "mul",
    Op.make("mul", "x", Const(0.5)),
    Op.make(
        "add",
        Const(1),
        Op.make(
            "erf",
            Op.make("div", "x", Const(1.4142135623730951)),
        ),
    ),
)

GELU_EXPAND = R(
    "gelu_expand",
    Op.make("gelu", "x"),
    _GELU_ERF,
    law="gelu(x) = 0.5x(1 + erf(x/√2)) — the exact erf definition "
    "(aten's default gelu; the tanh approximation is a DIFFERENT "
    "op under this bank and is not claimed).",
    tags=_PA_EXP,
)

MISH_EXPAND = R(
    "mish_expand",
    Op.make("mish", "x"),
    Op.make("mul", "x", Op.make("tanh", Op.make("softplus", "x"))),
    law="mish(x) = x·tanh(softplus(x)) — the Mish definition.",
    tags=_PA_EXP,
)

HARDSIGMOID_EXPAND = R(
    "hardsigmoid_expand",
    Op.make("hardsigmoid", "x"),
    Op.make(
        "div",
        Op.make("relu6", Op.make("add", "x", Const(3))),
        Const(6),
    ),
    law="hardsigmoid(x) = relu6(x+3)/6 — the piecewise-linear "
    "approximation's definition.",
    tags=_PA_EXP,
)

HARDSIGMOID_FOLD = R(
    "hardsigmoid_fold",
    Op.make(
        "div",
        Op.make("relu6", Op.make("add", "x", "THREE")),
        "SIX",
    ),
    Op.make("hardsigmoid", "x"),
    law="relu6(x+3)/6 folds to hardsigmoid(x).",
    cond=(
        "and",
        ("const-cmp", "THREE", "==", 3),
        ("const-cmp", "SIX", "==", 6),
    ),
    tags=_PA_SIM,
    derivation=("hardsigmoid_expand",),
)

HARDSWISH_EXPAND = R(
    "hardswish_expand",
    Op.make("hardswish", "x"),
    Op.make(
        "mul",
        "x",
        Op.make(
            "div",
            Op.make("relu6", Op.make("add", "x", Const(3))),
            Const(6),
        ),
    ),
    law="hardswish(x) = x·relu6(x+3)/6.",
    tags=_PA_EXP,
)

SOFTPLUS_EXPAND = R(
    "softplus_expand",
    Op.make("softplus", "x"),
    Op.make("log1p", Op.make("exp", "x")),
    law="softplus(x) = log1p(eˣ) at the default beta=1 — the "
    "exported bare spelling.",
    tags=_PA_EXP,
)

SOFTPLUS_FOLD = R(
    "softplus_fold",
    Op.make("log1p", Op.make("exp", "x")),
    Op.make("softplus", "x"),
    law="log1p(eˣ) folds to softplus(x).",
    tags=_PA_SIM,
    derivation=("softplus_expand",),
)

ELU_EXPAND = R(
    "elu_expand",
    Op.make("elu", "x"),
    Op.make(
        "where",
        Op.make("gt", "x", Const(0)),
        "x",
        Op.make("expm1", "x"),
    ),
    law="elu(x) = x·[x>0] + (eˣ−1)·[x≤0] at the default α=1 — the "
    "where-spelled piecewise (XLA elu).",
    tags=_PA_EXP,
)

LEAKY_RELU_EXPAND = R(
    "leaky_relu_expand",
    Op.make("leaky_relu", "x"),
    Op.make("maximum", "x", Op.make("mul", "x", Const(0.01))),
    law="leaky_relu(x) = max(x, 0.01x) at the default slope 0.01.",
    tags=_PA_EXP,
)

RELU_FOLD_MAX_R = R(
    "relu_fold_max_r",
    Op.make("maximum", "u", Const(0)),
    Op.make("relu", "u"),
    law="max(u, 0) = relu(u).",
    tags=_PA_SIM,
)

RELU_FOLD_MAX_L = R(
    "relu_fold_max_l",
    Op.make("maximum", Const(0), "u"),
    Op.make("relu", "u"),
    law="max(0, u) = relu(u).",
    tags=_PA_SIM,
)

RELU_FOLD_CLAMP = R(
    "relu_fold_clamp",
    Op.make("clamp_min", "u", min=0),
    Op.make("relu", "u"),
    law="clamp_min(u, 0) = relu(u) — the exported spelling folds.",
    tags=_PA_SIM,
)

RELU_EXPAND = R(
    "relu_expand",
    Op.make("relu", "u"),
    Op.make("clamp_min", "u", min=0),
    law="relu(u) = clamp_min(u, 0).",
    tags=_PA_EXP,
    derivation=("relu_fold_clamp",),
)

RELU_IDEM = R(
    "relu_idem",
    Op.make("relu", Op.make("relu", "u")),
    Op.make("relu", "u"),
    law="relu(relu(u)) = relu(u) — the projection is idempotent.",
    tags=_PA_SIM,
)

RELU6_FOLD = R(
    "relu6_fold",
    Op.make("clamp", "u", min=0, max=6),
    Op.make("relu6", "u"),
    law="clamp(u, 0, 6) = relu6(u).",
    tags=_PA_SIM,
)

RELU6_EXPAND = R(
    "relu6_expand",
    Op.make("relu6", "u"),
    Op.make("clamp", "u", min=0, max=6),
    law="relu6(u) = clamp(u, 0, 6).",
    tags=_PA_EXP,
    derivation=("relu6_fold",),
)


# ---------------------------------------------------------------------------
#  §2 — pointwise idempotence / involution / parity
# ---------------------------------------------------------------------------

ABS_IDEM = R(
    "abs_idem",
    Op.make("abs", Op.make("abs", "u")),
    Op.make("abs", "u"),
    law="abs(abs(u)) = abs(u).",
    tags=_PA_SIM,
)

ABS_SQUARE = R(
    "abs_square",
    Op.make("abs", Op.make("square", "u")),
    Op.make("square", "u"),
    law="|u²| = u² — the square is already nonneg.",
    tags=_PA_SIM,
)

SQUARE_ABS = R(
    "square_abs",
    Op.make("square", Op.make("abs", "u")),
    Op.make("square", "u"),
    law="(|u|)² = u² — abs is invisible to the square.",
    tags=_PA_SIM,
)

ABS_NEG = R(
    "abs_neg",
    Op.make("abs", Op.make("neg", "u")),
    Op.make("abs", "u"),
    law="|−u| = |u|.",
    tags=_PA_SIM,
)

SQRT_SQUARE = R(
    "sqrt_square",
    Op.make("sqrt", Op.make("square", "u")),
    Op.make("abs", "u"),
    law="√(u²) = |u| — the canonical spelling (NOT u — the "
    "negative branch keeps the abs).",
    tags=_PA_SIM,
)

RSQRT_SQUARE = R(
    "rsqrt_square",
    Op.make("rsqrt", Op.make("square", "u")),
    Op.make("reciprocal", Op.make("abs", "u")),
    law="1/√(u²) = 1/|u| — the rsqrt twin.",
    tags=_PA_SIM,
)

#  Dropped (honest misses — see results_prior_art_bank.md):
#    where_same — where(c,u,u)=u needs a BOOL-typed condition; the
#      pattern language guards shape, not dtype, and the fp64
#      oracle's envs can't mint bool operands.
#    logical_not_inv — ¬¬u = u only for bool u; on fp64 operands
#      the double-negation materializes a bool mask ≠ u.

MAXIMUM_IDEM = R(
    "maximum_idem",
    Op.make("maximum", "u", "u"),
    "u",
    law="max(u, u) = u.",
    tags=_PA_SIM,
)

MINIMUM_IDEM = R(
    "minimum_idem",
    Op.make("minimum", "u", "u"),
    "u",
    law="min(u, u) = u.",
    tags=_PA_SIM,
)

#: Odd functions: f(−x) = −f(x); even: f(−x) = f(x).  Structural —
#: the negative slides out or dies.
TANH_NEG = R(
    "tanh_neg",
    Op.make("tanh", Op.make("neg", "u")),
    Op.make("neg", Op.make("tanh", "u")),
    law="tanh is odd: tanh(−u) = −tanh(u).",
    tags=_PA_SIM,
)

SIGMOID_NEG = R(
    "sigmoid_neg",
    Op.make("sigmoid", Op.make("neg", "u")),
    Op.make("sub", Const(1), Op.make("sigmoid", "u")),
    law="σ(−u) = 1 − σ(u) — the symmetric complement.",
    tags=_PA_SIM,
)

SOFTSIGN_NEG = R(
    "softsign_neg",
    Op.make("softsign", Op.make("neg", "u")),
    Op.make("neg", Op.make("softsign", "u")),
    law="softsign is odd.",
    tags=_PA_SIM,
)

ERF_NEG = R(
    "erf_neg",
    Op.make("erf", Op.make("neg", "u")),
    Op.make("neg", Op.make("erf", "u")),
    law="erf is odd.",
    tags=_PA_SIM,
)

SIN_NEG = R(
    "sin_neg",
    Op.make("sin", Op.make("neg", "u")),
    Op.make("neg", Op.make("sin", "u")),
    law="sin is odd.",
    tags=_PA_SIM,
)

COS_NEG = R(
    "cos_neg",
    Op.make("cos", Op.make("neg", "u")),
    Op.make("cos", "u"),
    law="cos is even: cos(−u) = cos(u).",
    tags=_PA_SIM,
)

EXP_NEG_RECIP = R(
    "exp_neg_recip",
    Op.make("exp", Op.make("neg", "u")),
    Op.make("reciprocal", Op.make("exp", "u")),
    law="e⁻ᵘ = 1/eᵘ.",
    tags=_PA_SIM,
)

LOG_EXP = R(
    "log_exp",
    Op.make("log", Op.make("exp", "u")),
    "u",
    law="log(eᵘ) = u — eᵘ > 0 always, so the inverse is total.",
    tags=_PA_SIM,
)

LOG_RECIP = R(
    "log_recip",
    Op.make("log", Op.make("reciprocal", "u")),
    Op.make("neg", Op.make("log", "u")),
    law="log(1/u) = −log(u).",
    tags=_PA_SIM,
)


# ---------------------------------------------------------------------------
#  §2 — scalar/Const algebra: canonical folds and strength reductions
# ---------------------------------------------------------------------------

POW_ONE = R(
    "pow_one",
    Op.make("pow", "u", "P"),
    "u",
    law="pow(u, 1) = u.",
    cond=("const-cmp", "P", "==", 1),
    tags=_PA_SIM,
)

POW_ZERO = R(
    "pow_zero",
    Op.make("pow", "u", Const(0)),
    Op.make("ones_like", "u"),
    law="pow(u, 0) = ones(u) — IEEE-total including NaN bases "
    "(pow(nan,0)=1); spelled ones_like so the mint carries u's "
    "shape (the eval compare is shape-strict, not broadcast).",
    tags=_PA_SIM,
)

POW_NEG_ONE = R(
    "pow_neg_one",
    Op.make("pow", "u", Const(-1)),
    Op.make("reciprocal", "u"),
    law="pow(u, −1) = 1/u — the reciprocal spelling.  The "
    "Const-operand corner is excluded: an int-typed base does int "
    "exponentiation (pow(2,−1) → 0), which the pattern language "
    "can't dtype-guard otherwise.",
    cond=("not", ("const-num", "u")),
    tags=_PA_SIM,
)

POW_HALF = R(
    "pow_half",
    Op.make("pow", "u", "P"),
    Op.make("sqrt", "u"),
    law="pow(u, 0.5) = sqrt(u) — the canonical square root.",
    cond=("const-cmp", "P", "==", 0.5),
    tags=_PA_SIM,
)

SQRT_TO_POW = R(
    "sqrt_to_pow",
    Op.make("sqrt", "u"),
    Op.make("pow", "u", Const(0.5)),
    law="sqrt(u) = u^0.5 — the pow spelling (bridge to the pow "
    "family).",
    tags=_PA_EXP,
    derivation=("pow_half",),
)

RECIP_RECIP = R(
    "recip_recip",
    Op.make("reciprocal", Op.make("reciprocal", "u")),
    "u",
    law="1/(1/u) = u — reciprocal involution.",
    tags=_PA_SIM,
)

DIV_ONE_RECIP = R(
    "div_one_recip",
    Op.make("div", "ONE", "u"),
    Op.make("reciprocal", "u"),
    law="1/u = reciprocal(u) — the div-spelled reciprocal folds.",
    cond=("const-cmp", "ONE", "==", 1),
    tags=_PA_SIM,
)

RECIP_EXPAND = R(
    "recip_expand",
    Op.make("reciprocal", "u"),
    Op.make("div", Const(1), "u"),
    law="reciprocal(u) = 1/u — the div spelling.",
    tags=_PA_EXP,
    derivation=("div_one_recip",),
)

ADD_NEG_TO_SUB = R(
    "add_neg_to_sub",
    Op.make("add", "a", Op.make("neg", "b")),
    Op.make("sub", "a", "b"),
    law="a + (−b) = a − b — the refold (sub_to_add's inverse).",
    tags=_PA_SIM,
)

SUB_NEG_TO_ADD = R(
    "sub_neg_to_add",
    Op.make("sub", "a", Op.make("neg", "b")),
    Op.make("add", "a", "b"),
    law="a − (−b) = a + b.",
    tags=_PA_SIM,
)

NEG_SUB = R(
    "neg_sub",
    Op.make("neg", Op.make("sub", "a", "b")),
    Op.make("sub", "b", "a"),
    law="−(a−b) = b−a.",
    tags=_PA_SIM,
)

SUB_ZERO_TO_NEG = R(
    "sub_zero_to_neg",
    Op.make("sub", Const(0), "u"),
    Op.make("neg", "u"),
    law="0 − u = −u.",
    tags=_PA_SIM,
)

ID_SUB = R(
    "id_sub",
    Op.make("sub", "u", Const(0)),
    "u",
    law="u − 0 = u.",
    tags=_PA_SIM,
)

ID_DIV = R(
    "id_div",
    Op.make("div", "u", Const(1)),
    "u",
    law="u/1 = u.",
    tags=_PA_SIM,
)

ADD_SELF = R(
    "add_self",
    Op.make("add", "u", "u"),
    Op.make("mul", "u", Const(2)),
    law="u+u = 2u — the strength reduction.",
    tags=_PA_SIM,
)

MUL2_TO_ADD = R(
    "mul2_to_add",
    Op.make("mul", "u", Const(2)),
    Op.make("add", "u", "u"),
    law="2u = u+u — the sharing direction.",
    tags=_PA_SIM,
    derivation=("add_self",),
)

MUL_NEG1 = R(
    "mul_neg1",
    Op.make("mul", "u", Const(-1)),
    Op.make("neg", "u"),
    law="u·(−1) = −u.",
    tags=_PA_SIM,
)

NEG_CONST = R(
    "neg_const",
    Op.make("neg", "C"),
    "NC",
    law="−c folds to the negated Const leaf.",
    cond=("const-num", "C"),
    dspec={"$leaf:NC": ("neg", ("const", "C"))},
    tags=_PA_SIM,
)


def _mul_const_assoc() -> list[Rewrite]:
    """mul(mul(·,c1),c2) → mul(·,c1·c2) — all four operand orders."""
    dspec = {"$leaf:C3": ("mul", ("const", "C1"), ("const", "C2"))}
    cond = ("and", ("const-num", "C1"), ("const-num", "C2"))
    out = []
    inner_l = Op.make("mul", "u", "C1")
    inner_r = Op.make("mul", "C1", "u")
    for i, (lhs, rhs) in enumerate(
        (
            (
                Op.make("mul", inner_l, "C2"),
                Op.make("mul", "u", "C3"),
            ),
            (
                Op.make("mul", inner_r, "C2"),
                Op.make("mul", "u", "C3"),
            ),
            (
                Op.make("mul", "C2", inner_l),
                Op.make("mul", "C3", "u"),
            ),
            (
                Op.make("mul", "C2", inner_r),
                Op.make("mul", "C3", "u"),
            ),
        )
    ):
        out.append(
            R(
                f"mul_const_assoc_{i}",
                lhs,
                rhs,
                law="(u·c1)·c2 = u·(c1c2) — the scalar-factor fold "
                "(Inductor const-fold).",
                cond=cond,
                dspec=dspec,
                tags=_PA_SIM,
            )
        )
    return out


def _add_const_assoc() -> list[Rewrite]:
    dspec = {"$leaf:C3": ("add", ("const", "C1"), ("const", "C2"))}
    cond = ("and", ("const-num", "C1"), ("const-num", "C2"))
    out = []
    inner_l = Op.make("add", "u", "C1")
    inner_r = Op.make("add", "C1", "u")
    for i, (lhs, rhs) in enumerate(
        (
            (Op.make("add", inner_l, "C2"), Op.make("add", "u", "C3")),
            (Op.make("add", inner_r, "C2"), Op.make("add", "u", "C3")),
            (Op.make("add", "C2", inner_l), Op.make("add", "C3", "u")),
            (Op.make("add", "C2", inner_r), Op.make("add", "C3", "u")),
        )
    ):
        out.append(
            R(
                f"add_const_assoc_{i}",
                lhs,
                rhs,
                law="(u+c1)+c2 = u+(c1+c2) — the offset fold.",
                cond=cond,
                dspec=dspec,
                tags=_PA_SIM,
            )
        )
    return out


MUL_CONST_RULES: list[Rewrite] = _mul_const_assoc()
ADD_CONST_RULES: list[Rewrite] = _add_const_assoc()

SUB_CONST_SUB = R(
    "sub_const_sub",
    Op.make("sub", Op.make("sub", "u", "C1"), "C2"),
    Op.make("sub", "u", "C3"),
    law="(u−c1)−c2 = u−(c1+c2).",
    cond=("and", ("const-num", "C1"), ("const-num", "C2")),
    dspec={"$leaf:C3": ("add", ("const", "C1"), ("const", "C2"))},
    tags=_PA_SIM,
)

SUB_CONST_SUB_L = R(
    "sub_const_sub_l",
    Op.make("sub", Op.make("sub", "C1", "u"), "C2"),
    Op.make("sub", "C3", "u"),
    law="(c1−u)−c2 = (c1−c2)−u.",
    cond=("and", ("const-num", "C1"), ("const-num", "C2")),
    dspec={"$leaf:C3": ("sub", ("const", "C1"), ("const", "C2"))},
    tags=_PA_SIM,
)

DIV_CONST_DIV = R(
    "div_const_div",
    Op.make("div", Op.make("div", "u", "C1"), "C2"),
    Op.make("div", "u", "C3"),
    law="(u/c1)/c2 = u/(c1·c2).",
    cond=("and", ("const-num", "C1"), ("const-num", "C2")),
    dspec={"$leaf:C3": ("mul", ("const", "C1"), ("const", "C2"))},
    tags=_PA_SIM,
)

DIV_CONST_TO_MUL = R(
    "div_const_to_mul",
    Op.make("div", "u", "C"),
    Op.make("mul", "u", "CI"),
    law="u/c = u·(1/c) — the div→mul strength reduction (the "
    "softcap/quant scale's canonical form).",
    cond=(
        "and",
        ("const-num", "C"),
        ("not", ("const-cmp", "C", "==", 0)),
    ),
    dspec={"$leaf:CI": ("recip", ("const", "C"))},
    tags=_PA_SIM,
)

MUL_CONST_DIV = R(
    "mul_const_div",
    Op.make("mul", Op.make("div", "u", "C1"), "C2"),
    Op.make("mul", "u", "C3"),
    law="(u/c1)·c2 = u·(c2/c1).",
    cond=("and", ("const-num", "C1"), ("const-num", "C2")),
    dspec={"$leaf:C3": ("fdiv", ("const", "C2"), ("const", "C1"))},
    tags=_PA_SIM,
)


# ---------------------------------------------------------------------------
#  §2 — zeros/ones_like units and the where/maximum monoid folds
# ---------------------------------------------------------------------------


def _zeros_like(op: str, unit: str, keep: Any) -> Rewrite:
    """Fold ``op(u, unit_like(v))`` to ``u``.

    ``unit`` is the spelled additive/multiplicative unit.  The grid
    guard: the pair must broadcast AND the result shape must BE
    ``u``'s — a smaller ``u`` broadcasting up produces the *grid*
    shape, not ``u`` (the same corner as ``ste_fold``).
    """
    return R(
        f"{op}_{unit}like_r",
        Op.make(op, "u", Op.make(unit + "_like", "v")),
        "u",
        law=f"{op}(u, {unit}_like(v)) = u — the spelled additive/"
        "multiplicative unit (grid-equality guard keeps it "
        "honest).",
        cond=(
            "and",
            ("shaped", ("bcast", "u", "v")),
            ("shape-eq", "u", ("bcast", "u", "v")),
        ),
        tags=_PA_SIM,
    )


ADD_ZEROS_R = _zeros_like("add", "zeros", "u")
SUB_ZEROS_R = _zeros_like("sub", "zeros", "u")
MUL_ONES_R = _zeros_like("mul", "ones", "u")
DIV_ONES_R = _zeros_like("div", "ones", "u")


# ---------------------------------------------------------------------------
#  §3 — zoo-corpus named transforms
# ---------------------------------------------------------------------------


#: The HF rotate-half rotary `x·cos + rotate_half(x)·sin` spelled
#: with half-width tables concat'd to full width — equal to the
#: Llama cat-pair `cat(x1·c − x2·s, x2·c + x1·s)`.
def _slice2(x: Any, p: str) -> Op:
    return Op.make("slice", x, dim=f"{p}D", start=f"{p}A", end=f"{p}E")


def _rope_rh_pat(x: Any, c: Any, s: Any) -> Op:
    """``x·cat(ch,ch) + cat(−x2, x1)·cat(sh,sh)`` — the HF spelling."""
    ch2 = Op.make("concat", c, c, dim="CD")
    sh2 = Op.make("concat", s, s, dim="CD")
    rot = Op.make(
        "concat",
        Op.make("neg", _slice2(x, "I2")),
        _slice2(x, "I1"),
        dim="CD",
    )
    return Op.make(
        "add",
        Op.make("mul", x, ch2),
        Op.make("mul", rot, sh2),
    )


def _rope_cat_pat(x: Any, c: Any, s: Any) -> Op:
    """``cat(x1·c − x2·s, x2·c + x1·s)`` — the Llama pair spelling."""
    x1, x2 = _slice2(x, "I1"), _slice2(x, "I2")
    return Op.make(
        "concat",
        Op.make(
            "sub",
            Op.make("mul", x1, c),
            Op.make("mul", x2, s),
        ),
        Op.make(
            "add",
            Op.make("mul", x2, c),
            Op.make("mul", x1, s),
        ),
        dim="CD",
    )


ROPE_RH_TO_CAT = R(
    "rope_rh_to_cat",
    _rope_rh_pat("x", "ch", "sh"),
    _rope_cat_pat("x", "ch", "sh"),
    law="x·cat(c,c) + cat(−x2,x1)·cat(s,s) IS cat(x1c−x2s, "
    "x2c+x1s) — the HF rotate-half and the Llama cat-pair rotary "
    "are the same transform when the cos/sin tables are the "
    "half-table duplicated (the standard emb.repeat(2) spelling).",
    check=_check_rope_equiv,
    tags=_PA_FUS,
)

#: The STE idiom folds only when the broadcast recovers ``y``
#: exactly — ``x + (y−x)`` broadcasts to ``bcast(x,y)``, which must
#: BE ``y``'s shape (a smaller ``x`` broadcasting up changes the
#: sum's extent, not just its values).
_COND_STE = ("shape-eq", "y", ("bcast", "x", "y"))

STE_FOLD = R(
    "ste_fold",
    Op.make(
        "add",
        "x",
        Op.make("detach", Op.make("sub", "y", "x")),
    ),
    "y",
    law="x + (y − x).detach() = y — the straight-through "
    "estimator's value idiom (the quant STE: w + (ŵ−w).detach() "
    "evaluates to ŵ) — guarded to the no-broadcast corner.",
    cond=_COND_STE,
    tags=_PA_SIM,
)

STE_FOLD_L = R(
    "ste_fold_l",
    Op.make(
        "add",
        Op.make("detach", Op.make("sub", "y", "x")),
        "x",
    ),
    "y",
    law="The operand-swapped STE spelling.",
    cond=_COND_STE,
    tags=_PA_SIM,
)

_DSPEC_OUT_CH = {"CV": ("tuple", ("dim", "W", 0), 1)}

LINEAR_OUT_CHANNEL = R(
    "linear_out_channel",
    Op.make("mul", Op.make("linear", "x", "W"), "c"),
    Op.make(
        "linear",
        "x",
        Op.make("mul", "W", Op.make("reshape", "c", shape="CV")),
    ),
    law="c·(xWᵀ) = x(W∘ĉ)ᵀ with ĉ = c viewed (o,1) — the "
    "per-output-channel scale folds into the weight (the AWQ/"
    "dequant scale migration; the output-side twin of "
    "linear_channel_scale).",
    cond=(
        "and",
        _COND_LINEAR_XW,
        ("shaped", "c"),
        ("or", ("rank", "c", "==", 1), ("dim-eq-const", "c", 0, 1)),
        ("dim-eq", "c", -1, "W", 0),
    ),
    dspec=_DSPEC_OUT_CH,
    tags=_PA_FUS,
)

LINEAR_OUT_CHANNEL_REV = R(
    "linear_out_channel_rev",
    Op.make(
        "linear",
        "x",
        Op.make("mul", "W", Op.make("reshape", "c", shape="CV")),
    ),
    Op.make("mul", Op.make("linear", "x", "W"), "c"),
    law="Reverse: unfold the folded channel scale back to a "
    "runtime mul.",
    cond=(
        "and",
        _COND_LINEAR_XW,
        ("shaped", "c"),
        ("or", ("rank", "c", "==", 1), ("dim-eq-const", "c", 0, 1)),
        ("dim-eq", "c", -1, "W", 0),
        (
            "attr-cmp",
            "CV",
            "==",
            ("tuple", ("dim", "W", 0), 1),
        ),
    ),
    tags=_PA_EXP,
    derivation=("linear_out_channel",),
)

ADALN_ONE_PLUS = R(
    "adaln_one_plus",
    Op.make("mul", "u", Op.make("add", Const(1), "s")),
    Op.make("add", "u", Op.make("mul", "u", "s")),
    law="u·(1+s) = u + u·s — the AdaLN-Zero ``1 + scale`` idiom "
    "spelled as distributivity (derivable through the pointwise "
    "factor laws — kept as the direct edge).",
    tags=_PA_EXP,
)

DROPOUT_EVAL = R(
    "dropout_eval",
    Op.make("dropout", "u", p="P", train=False),
    "u",
    law="dropout(u, p, train=False) = u — the eval-mode strip "
    "(Inductor's aten.dropout elimination).",
    tags=_PA_SIM,
)

DROPOUT_ZERO = R(
    "dropout_zero",
    Op.make("dropout", "u", p=0.0, train="T"),
    "u",
    law="dropout(u, p=0) = u — a zero drop rate is the identity "
    "in either mode.",
    tags=_PA_SIM,
)

ONEHOT_MATMUL_EMBED = R(
    "onehot_matmul_embed",
    Op.make(
        "matmul",
        Op.make(
            "double",
            Op.make("one_hot", "i", num_classes="N"),
        ),
        "W",
    ),
    Op.make("embedding", "W", "i"),
    law="to(dtype)(one_hot(i, N)) @ W = embedding(W, i) — the "
    "one-hot matmul IS a row gather (TASO/vocab-projection fold; "
    "the cast is the exported spelling — one_hot emits int64 which "
    "cannot contract a float weight; spelled ``double`` to ride "
    "the fp64 eval path).",
    cond=(
        "and",
        ("rank", "W", "==", 2),
        ("attr-cmp", ("dim", "W", -2), "==", ("attr", "N")),
    ),
    tags=_PA_FUS,
)

ADDCMUL_EXPAND = R(
    "addcmul_expand",
    Op.make("addcmul", "t", "a", "b"),
    Op.make("add", "t", Op.make("mul", "a", "b")),
    law="addcmul(t, a, b) = t + a·b — the fused op's spelled work "
    "(value=1 default).",
    tags=_PA_EXP,
)

ADDCDIV_EXPAND = R(
    "addcdiv_expand",
    Op.make("addcdiv", "t", "a", "b"),
    Op.make("add", "t", Op.make("div", "a", "b")),
    law="addcdiv(t, a, b) = t + a/b.",
    tags=_PA_EXP,
)

LERP_EXPAND = R(
    "lerp_expand",
    Op.make("lerp", "x", "y", "w"),
    Op.make("add", "x", Op.make("mul", "w", Op.make("sub", "y", "x"))),
    law="lerp(x, y, w) = x + w·(y−x) — the interpolation's definition.",
    tags=_PA_EXP,
)


def _ln_xhat(u: str = "u") -> Op:
    """``(u − mean(u,RD)) · rsqrt(var(u,RD,0) + EPS)`` — the LN core."""
    return Op.make(
        "mul",
        Op.make(
            "sub",
            u,
            Op.make("mean", u, dim="RD", keepdim=True),
        ),
        Op.make(
            "rsqrt",
            Op.make(
                "add",
                Op.make(
                    "var",
                    u,
                    dim="RD",
                    correction=0,
                    keepdim=True,
                ),
                "EPS",
            ),
        ),
    )


LAYER_NORM_EXPAND_PLAIN = R(
    "layer_norm_expand_plain",
    Op.make("layer_norm", "u", dim="ND", eps="E"),
    _ln_xhat(),
    law="layer_norm(u, ns, eps) IS (u−mean)·rsqrt(var+eps) with "
    "correction=0 and the trailing ns axes reduced — the fused "
    "kernel's decomposed form (XLA).",
    derive=_derive_ln,
    check=_check_ln,
    tags=_PA_EXP,
)

LAYER_NORM_EXPAND_AFFINE = R(
    "layer_norm_expand_affine",
    Op.make("layer_norm", "u", "w", "b", dim="ND", eps="E"),
    Op.make(
        "add",
        Op.make("mul", _ln_xhat(), "w"),
        "b",
    ),
    law="The affine variant: LN(u, ns, γ, β) = x̂·γ + β.",
    derive=_derive_ln,
    check=_check_ln,
    tags=_PA_EXP,
)


# ---------------------------------------------------------------------------
#  Norm-kernel denotations — the fused norm ops' manual spellings
#
#  The expand direction is the honest prior-art statement: the fold
#  (manual chain -> fused op) is each ``fold:`` candidate's own
#  alpha-sibling in the novelty sieve's premise cut, while the
#  denotation is the transform the kernel libraries already own —
#  oneDNN's ``_MklFusedInstanceNorm`` and Inductor's group_norm
#  fusion.  ``FUSION``-tagged like the kernel bank's fused-kernel
#  denotations (``mlopt_kernel.priorart``): the tag is what the
#  sieve's ``known_bank`` floor admits, so a minted "fold to the
#  fused norm" re-derives through the bridge and classifies
#  ``known`` rather than machine-original.  Promoted from the
#  graphopt fusion-schema premise closures (``pipeline._INORM_EXPAND``
#  / ``_GNORM_EXPAND``), whose declared premises they remain.
# ---------------------------------------------------------------------------


def _check_norm_affine(bound: dict) -> bool:
    """Require the affine pair rank-1 per-channel — the fused ops' only form.

    Guards the expand bridges so the definitional spelling mints
    only where ``W``/``B`` are ``(C,)`` vectors over ``U``'s channel
    axis (and ``U`` has a channel axis at all).
    """
    us = _shape_of(bound.get("U"))
    ws = _shape_of(bound.get("W"))
    bs = _shape_of(bound.get("B"))
    return (
        isinstance(us, tuple)
        and len(us) >= 3
        and isinstance(us[1], int)
        and ws == bs == (us[1],)
    )


def _derive_inorm_expand(bound: dict) -> dict | None:
    """Instance-norm's manual spelling, computed off the bound leaves.

    ``MD`` reduces every axis past ``(n, c)`` — the per-sample-channel
    statistics; ``K1``/``K2`` are the two ``unsqueeze`` re-broadcasts;
    ``WS``/``BS`` spell the ``(1, C, 1, 1, …)`` affine reshapes;
    ``EPS`` mints the ``Const`` leaf the fused op carries as a float
    attr.
    """
    us = _shape_of(bound.get("U"))
    eps = _aval(bound, "EP")
    if not (
        isinstance(us, tuple)
        and len(us) >= 3
        and isinstance(us[1], int)
        and isinstance(eps, (int, float))
    ):
        return None
    pad = (1,) * (len(us) - 2)
    return {
        "EPS": Const(float(eps)),
        "$attr:MD": tuple(range(2, len(us))),
        "$attr:K1": -1,
        "$attr:K2": -1,
        "$attr:WS": (1, us[1], *pad),
        "$attr:BS": (1, us[1], *pad),
    }


def _derive_gnorm_expand(bound: dict) -> dict | None:
    """Group-norm's manual spelling — the reshape-to-``(n, G, ·)`` fold.

    ``US`` is the group-view target; the reduce is its trailing axis
    (``MD=(-1,)``, ``MK=True``); ``RS`` restores the input shape and
    ``WS``/``BS`` spell the affine broadcast.  A channel block not
    divisible by the group count has no group fold — vetoed, never
    approximated.
    """
    us = _shape_of(bound.get("U"))
    g, eps = _aval(bound, "G"), _aval(bound, "EP")
    if not (
        isinstance(us, tuple)
        and len(us) >= 3
        and all(isinstance(d, int) for d in us)
        and isinstance(g, int)
        and isinstance(eps, (int, float))
    ):
        return None
    n, c, rest = us[0], us[1], us[2:]
    inner = c * math.prod(rest)
    if g <= 0 or inner % g:
        return None
    pad = (1,) * len(rest)
    return {
        "EPS": Const(float(eps)),
        "$attr:US": (n, g, inner // g),
        "$attr:MD": (-1,),
        "$attr:MK": True,
        "$attr:RS": us,
        "$attr:WS": (1, c, *pad),
        "$attr:BS": (1, c, *pad),
    }


def _inorm_manual() -> Op:
    """``(u−μ)·rsqrt(σ²+ε)·w+b`` — the manual instance-norm chain.

    Per-(n, c) statistics broadcast by ``unsqueeze`` rather than
    ``keepdim`` (``dim=MD`` shared between the mean/var branches, the
    broadcast dims ``K1``/``K2`` likewise), ``σ² = E[(u−μ)²]``, the
    channel affine spelled ``reshape``-broadcast — the
    ``InstanceNormBroadcast`` export verbatim.
    """
    d = Op.make(
        "sub",
        "U",
        Op.make(
            "unsqueeze",
            Op.make(
                "unsqueeze", Op.make("mean", "U", dim="MD"), dim="K1"
            ),
            dim="K2",
        ),
    )
    v = Op.make(
        "unsqueeze",
        Op.make(
            "unsqueeze",
            Op.make("mean", Op.make("mul", d, d), dim="MD"),
            dim="K1",
        ),
        dim="K2",
    )
    return Op.make(
        "add",
        Op.make(
            "mul",
            Op.make(
                "mul",
                d,
                Op.make("rsqrt", Op.make("add", v, "EPS")),
            ),
            Op.make("reshape", "W", shape="WS"),
        ),
        Op.make("reshape", "B", shape="BS"),
    )


def _gnorm_manual() -> Op:
    """``reshape((u'−μ)·rsqrt(E[(u'−μ)²]+ε))·w+b`` — manual group-norm.

    The group fold: ``u' = reshape(u, (n, G, ·))`` so the per-(n, g)
    reduce is a plain trailing ``mean`` — the ``ManualGroupNorm``
    export's spelling.
    """
    ru = Op.make("reshape", "U", shape="US")
    d = Op.make("sub", ru, Op.make("mean", ru, dim="MD", keepdim="MK"))
    v = Op.make("mean", Op.make("mul", d, d), dim="MD", keepdim="MK")
    return Op.make(
        "add",
        Op.make(
            "mul",
            Op.make(
                "reshape",
                Op.make(
                    "mul", d, Op.make("rsqrt", Op.make("add", v, "EPS"))
                ),
                shape="RS",
            ),
            Op.make("reshape", "W", shape="WS"),
        ),
        Op.make("reshape", "B", shape="BS"),
    )


INSTANCE_NORM_EXPAND = R(
    "instance_norm_expand",
    Op.make("instance_norm", "U", "W", "B", eps="EP"),
    _inorm_manual(),
    law="instance_norm(u, w, b, eps) IS (u−μ)·rsqrt(σ²+ε)·w+b — "
    "per-(n,c) statistics broadcast by unsqueeze, σ²=E[(u−μ)²], "
    "the affine spelled reshape-broadcast (oneDNN's "
    "``_MklFusedInstanceNorm`` denotation — the manual "
    "InstanceNormBroadcast chain the fused kernel computes).",
    check=_check_norm_affine,
    derive=_derive_inorm_expand,
    tags=_PA_FUS,
)

GROUP_NORM_EXPAND = R(
    "group_norm_expand",
    Op.make("group_norm", "U", "W", "B", num_groups="G", eps="EP"),
    _gnorm_manual(),
    law="group_norm(u, w, b, G, eps) IS the reshape-to-(n,G,·) "
    "manual chain — u'=reshape(u,(n,G,·)), normalize the trailing "
    "axis, reshape back, affine — Inductor's group_norm "
    "decomposition (the ManualGroupNorm spelling the fused kernel "
    "denotes; a channel block indivisible by G vetoes the derive).",
    check=_check_norm_affine,
    derive=_derive_gnorm_expand,
    tags=_PA_FUS,
)


# ---------------------------------------------------------------------------
#  The bank
# ---------------------------------------------------------------------------

PRIOR_ART_RULES: list[Rewrite] = [
    # -- TASO: concat structure
    CONCAT_SINGLE,
    CONCAT_FLAT_L,
    CONCAT_FLAT_R,
    CONCAT_CHUNK2,
    CONCAT_CHUNK3,
    CONCAT_SPLIT2,
    STACK_FOLD,
    STACK_EXPAND,
    *CONCAT_PULL_UNARY,
    *CONCAT_PULL_BINARY,
    # -- TASO: matmul/linear/conv merges
    MM_CONCAT_LEFT,
    MM_CONCAT_LEFT_REV,
    MM_CONCAT_RIGHT,
    MM_CONCAT_RIGHT_REV,
    MM_SPLITK,
    MM_SPLITK_REV,
    LINEAR_CONCAT_W,
    LINEAR_CONCAT_W_REV,
    LINEAR_BIAS_FOLD,
    LINEAR_BIAS_FOLD_L,
    LINEAR_BIAS_EXPAND,
    LINEAR_BIAS_MERGE,
    *CONV_MERGE_RULES,
    # -- TASO: index-view distributions/bridges
    *SELECT_PULL_RULES,
    UNBIND_SELECT,
    SELECT_TO_SLICE,
    SLICE_TO_SELECT,
    # -- no-op elimination
    CONTIGUOUS_ID,
    CLONE_ID,
    ALIAS_ID,
    DETACH_ID,
    DETACH_INPLACE_ID,
    *CAST_IDEM_RULES,
    UNSQUEEZE_SQUEEZE,
    SQUEEZE_UNSQUEEZE,
    SQUEEZE_NOOP,
    SQUEEZE_ALL_NOOP,
    RESHAPE_NOOP,
    RESHAPE_RESHAPE,
    EXPAND_NOOP,
    BROADCAST_TO_NOOP,
    BROADCAST_TO_EXPAND,
    SLICE_NOOP,
    SLICE_NOOP_NOSTART,
    INDEX_SELECT_ID,
    ROLL_NOOP,
    PAD_NOOP,
    REPEAT_NOOP,
    PERMUTE_NOOP,
    PERMUTE_COMPOSE,
    MOVEDIM_INVOLUTION,
    FLATTEN_NOOP,
    UNFLATTEN_NOOP,
    SPLIT_SINGLE,
    FLIP_INVOLUTION,
    TRIL_IDEM,
    TRIU_IDEM,
    CLAMP_IDEM,
    *REDUCE_DIM1_RULES,
    # -- reduction algebra
    STD_SQRT_VAR,
    VAR_SQUARE_STD,
    STD_SQRT_VAR_BARE,
    VAR_SQUARE_STD_BARE,
    NORM2_EXPAND,
    NORM2_FOLD,
    MEAN_EXPAND,
    MEAN_FOLD,
    LOG_SOFTMAX_FOLD,
    LOG_SOFTMAX_EXPAND,
    # -- activation decompositions/folds
    SIGMOID_EXPAND,
    SIGMOID_FOLD,
    SIGMOID_FOLD_DIV,
    TANH_EXPAND,
    TANH_FOLD,
    GELU_EXPAND,
    MISH_EXPAND,
    HARDSIGMOID_EXPAND,
    HARDSIGMOID_FOLD,
    HARDSWISH_EXPAND,
    SOFTPLUS_EXPAND,
    SOFTPLUS_FOLD,
    ELU_EXPAND,
    LEAKY_RELU_EXPAND,
    RELU_FOLD_MAX_R,
    RELU_FOLD_MAX_L,
    RELU_FOLD_CLAMP,
    RELU_EXPAND,
    RELU_IDEM,
    RELU6_FOLD,
    RELU6_EXPAND,
    # -- pointwise idempotence / parity
    ABS_IDEM,
    ABS_SQUARE,
    SQUARE_ABS,
    ABS_NEG,
    SQRT_SQUARE,
    RSQRT_SQUARE,
    MAXIMUM_IDEM,
    MINIMUM_IDEM,
    TANH_NEG,
    SIGMOID_NEG,
    SOFTSIGN_NEG,
    ERF_NEG,
    SIN_NEG,
    COS_NEG,
    EXP_NEG_RECIP,
    LOG_EXP,
    LOG_RECIP,
    # -- scalar/const algebra
    POW_ONE,
    POW_ZERO,
    POW_NEG_ONE,
    POW_HALF,
    SQRT_TO_POW,
    RECIP_RECIP,
    DIV_ONE_RECIP,
    RECIP_EXPAND,
    ADD_NEG_TO_SUB,
    SUB_NEG_TO_ADD,
    NEG_SUB,
    SUB_ZERO_TO_NEG,
    ID_SUB,
    ID_DIV,
    ADD_SELF,
    MUL2_TO_ADD,
    MUL_NEG1,
    NEG_CONST,
    *MUL_CONST_RULES,
    *ADD_CONST_RULES,
    SUB_CONST_SUB,
    SUB_CONST_SUB_L,
    DIV_CONST_DIV,
    DIV_CONST_TO_MUL,
    MUL_CONST_DIV,
    # -- zeros/ones units
    ADD_ZEROS_R,
    SUB_ZEROS_R,
    MUL_ONES_R,
    DIV_ONES_R,
    # -- zoo-corpus transforms
    ROPE_RH_TO_CAT,
    STE_FOLD,
    STE_FOLD_L,
    LINEAR_OUT_CHANNEL,
    LINEAR_OUT_CHANNEL_REV,
    ADALN_ONE_PLUS,
    DROPOUT_EVAL,
    DROPOUT_ZERO,
    ONEHOT_MATMUL_EMBED,
    ADDCMUL_EXPAND,
    ADDCDIV_EXPAND,
    LERP_EXPAND,
    LAYER_NORM_EXPAND_PLAIN,
    LAYER_NORM_EXPAND_AFFINE,
    INSTANCE_NORM_EXPAND,
    GROUP_NORM_EXPAND,
]
