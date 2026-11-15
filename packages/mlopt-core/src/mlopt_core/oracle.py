"""A reference :class:`~mlopt_core.ports.VerifyOracle` over Python floats.

Plan 0037 phase 0 / plan 0039.  ``FloatOracle`` is the shipped
*reference* implementation of the term-level verify gate: tensor
values are nested Python lists of ``float`` (f64), ``run`` is a
small term interpreter over a documented op subset, and ``compare``
restates the historical ``verify_equiv`` gate —
``max_rel = max|ref - opt| / (max|ref| + 1e-8)``,
``passed = max_rel < rtol`` plus ``max_abs <= atol`` when ``atol`` is
given — bit-for-bit in f64 (no dtype floor: f64 resolves ``rtol``
directly).

Honest limits — this is a *test-grade* oracle, not a production
executor:

* pure Python over nested lists — verify-input scale only (the same
  calibration-input caveat :mod:`mlopt_core.metrics` documents);
* a documented op subset — an unlisted op raises ``KeyError`` rather
  than silently approximating (see :data:`_UNARY` / :data:`_BINARY` /
  :data:`_REDUCTIONS` / the ``_apply`` dispatch);
* no stochastic ops — ``run`` is deterministic for a fixed
  ``(term, env)`` by construction, satisfying the oracle's seeded-RNG
  contract vacuously.

Production oracles (host-tensor f64 codec, native/kernel-ir
executors) implement the same two-method port in their own value
domain; nothing here is torch- or numpy-shaped.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from mlopt_core.ir import Const, Op, Param, Var
from mlopt_core.metrics import _leaves
from mlopt_core.ports import VerifyResult

__all__ = ["FloatOracle", "OracleReport"]

#: The rel-diff denominator floor — the same constant
#: ``verify_equiv`` / :class:`~mlopt_core.metrics.MaxRel` use, so the
#: oracle's gate agrees with the historical metric bit-for-bit.
_REL_FLOOR = 1e-8


@dataclass(frozen=True)
class OracleReport:
    """A :class:`~mlopt_core.ports.VerifyResult` with verify provenance.

    ``max_abs`` / ``max_rel`` / ``passed`` satisfy the port; ``rtol``
    records the tolerance the gate applied (``inf`` under the
    error-bound path, which replaces the relative tolerance) and
    ``per_output`` the per-leaf ``max_abs`` values, mirroring
    ``report.VerifyReport``'s shape so report consumers port cleanly.
    """

    max_abs: float
    max_rel: float
    passed: bool
    rtol: float | None = None
    per_output: tuple[float, ...] = ()


# ---------------------------------------------------------------------------
#  Nested-list tensor plumbing — shape, broadcast, ravel, nest
# ---------------------------------------------------------------------------


def _shape(v: Any) -> tuple[int, ...]:
    """Shape of a nested-list tensor; ``()`` for a scalar."""
    if not isinstance(v, list):
        return ()
    if not v:
        return (0,)
    sub = _shape(v[0])
    for e in v[1:]:
        if _shape(e) != sub:
            raise ValueError(
                f"ragged tensor: sibling shapes {_shape(e)} vs {sub}"
            )
    return (len(v), *sub)


def _ravel(v: Any) -> tuple[list[float], tuple[int, ...]]:
    """``(flat values, shape)`` of a nested-list tensor, row-major."""
    shape = _shape(v)
    flat: list[float] = []

    def rec(x: Any) -> None:
        if isinstance(x, list):
            for e in x:
                rec(e)
        else:
            flat.append(float(x))

    rec(v)
    return flat, shape


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major flat strides of ``shape``."""
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return tuple(st)


def _fill(shape: tuple[int, ...], val: float) -> Any:
    """Build a ``shape``-nested list with every leaf ``val``."""
    if not shape:
        return val
    return [_fill(shape[1:], val) for _ in range(shape[0])]


def _bshape(
    sa: tuple[int, ...], sb: tuple[int, ...]
) -> tuple[int, ...]:
    """Numpy-style broadcast of two shapes -> the output shape.

    A dim pair that is neither equal nor ``1``-broadcastable is a
    loud ``ValueError`` — the oracle never guesses a shape.
    """
    nd = max(len(sa), len(sb))
    pa = (1,) * (nd - len(sa)) + sa
    pb = (1,) * (nd - len(sb)) + sb
    out = []
    for x, y in zip(pa, pb, strict=True):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError(f"broadcast shape mismatch: {sa} vs {sb}")
    return tuple(out)


def _expand(v: Any, fs: tuple[int, ...], ts: tuple[int, ...]) -> Any:
    """Materialise ``v`` as a nested list of exactly ``ts``.

    ``fs`` is ``v``'s shape left-padded to ``ts``'s rank — the
    padding dims are all ``1`` and repeat their single slice.
    """
    if not ts:
        return v
    if not isinstance(v, list):
        return _fill(ts, float(v))
    f0, t0 = fs[0], ts[0]
    if f0 == t0:
        return [_expand(e, fs[1:], ts[1:]) for e in v]
    # f0 == 1 < t0 — repeat the single leading slice.
    return [_expand(v[0], fs[1:], ts[1:]) for _ in range(t0)]


def _bcast_to(v: Any, ts: tuple[int, ...]) -> Any:
    """Broadcast nested-list ``v`` to ``ts`` (or return it verbatim).

    Broadcasting left-pads the *shape* with ``1``s — which on a
    nested list means wrapping the value in singleton list levels
    first, so ``[10, 20]`` (shape ``(2,)``) broadcast against
    ``(2, 2)`` repeats ROW-wise ``[[10, 20], [10, 20]]``, matching
    numpy/torch trailing-axis alignment.
    """
    sv = _shape(v)
    if sv == ts:
        return v
    pad = len(ts) - len(sv)
    if pad < 0:
        raise ValueError(f"cannot broadcast shape {sv} to {ts}")
    w = v
    for _ in range(pad):
        w = [w]
    fs = (1,) * pad + sv
    for f, t in zip(fs, ts, strict=True):
        if f != t and f != 1:
            raise ValueError(f"cannot broadcast shape {sv} to {ts}")
    return _expand(w, fs, ts)


def _map1(f: Any, v: Any) -> Any:
    """Elementwise unary ``f`` over a nested-list tensor."""
    if isinstance(v, list):
        return [_map1(f, e) for e in v]
    return f(v)


def _map2(f: Any, a: Any, b: Any) -> Any:
    """Elementwise binary ``f`` over two equal-shape nested lists."""
    if isinstance(a, list):
        return [_map2(f, x, y) for x, y in zip(a, b, strict=True)]
    return f(a, b)


def _ew2(f: Any, a: Any, b: Any) -> Any:
    """Broadcasting elementwise binary op."""
    os_ = _bshape(_shape(a), _shape(b))
    ta = _bcast_to(a, os_)
    tb = _bcast_to(b, os_)
    return _map2(f, ta, tb)


def _nest(shape: tuple[int, ...], vals: Iterator[float]) -> Any:
    """Rebuild a ``shape``-nested list from a flat iterator."""
    if not shape:
        return next(vals)
    return [_nest(shape[1:], vals) for _ in range(shape[0])]


def _dims_of(attrs: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Normalise a reduction's ``dim`` / ``dims`` / ``axis`` attr.

    Accepts an int or an iterable of ints (negative dims wrap); the
    first spelling present wins.  ``None`` means "reduce everything".
    """
    raw = None
    for key in ("dim", "dims", "axis"):
        if key in attrs:
            raw = attrs[key]
            break
    if raw is None:
        return None
    if isinstance(raw, int):
        return (raw,)
    return tuple(int(d) for d in raw)


def _reduce(
    v: Any, dims: tuple[int, ...] | None, f: Any, keepdim: bool
) -> Any:
    """Reduce nested-list ``v`` over ``dims`` (``None`` = all).

    ``f`` sees each group's flat float list.  ``keepdim`` keeps
    size-1 dims at the reduced positions; a full reduction returns a
    scalar.
    """
    flat, shape = _ravel(v)
    nd = len(shape)
    if dims is not None and not dims:
        return v  # an empty dim list reduces nothing
    if dims is None:
        # Reduce everything; keepdim yields the all-ones shape.
        total = f(flat) if flat else f([])
        return _fill((1,) * nd, total) if keepdim else total
    ds = tuple(sorted({d % nd for d in dims}))
    keep = tuple(i for i in range(nd) if i not in ds)
    strides = _strides(shape)
    groups: dict[tuple[int, ...], list[float]] = {}
    for idx, val in enumerate(flat):
        rem = idx
        key = []
        for ax, s in enumerate(strides):
            i, rem = divmod(rem, s)
            if ax not in ds:
                key.append(i)
        key_t = tuple(key)
        groups.setdefault(key_t, []).append(val)
    if keepdim:
        out_shape = tuple(1 if i in ds else shape[i] for i in range(nd))
    else:
        out_shape = tuple(shape[i] for i in keep)
    # Walk the output index space in row-major order; each output
    # element's kept-index key selects its reduction group.
    out_flat: list[float] = []
    for multi in product(*(range(d) for d in out_shape)):
        if keepdim:
            key = tuple(multi[i] for i in keep)
        else:
            key = multi
        g = groups.get(key)
        out_flat.append(f(g) if g else f([]))
    return _nest(out_shape, iter(out_flat))


# ---------------------------------------------------------------------------
#  Op implementations — the documented subset
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _silu(x: float) -> float:
    return x * _sigmoid(x)


def _gelu(x: float) -> float:
    """Exact (erf) GELU — ``0.5·x·(1 + erf(x/√2))``."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


_UNARY: dict[str, Any] = {
    "neg": lambda x: -x,
    "square": lambda x: x * x,
    "sqrt": math.sqrt,
    "rsqrt": lambda x: 1.0 / math.sqrt(x),
    "exp": math.exp,
    "sigmoid": _sigmoid,
    "silu": _silu,
    "tanh": math.tanh,
    "gelu": _gelu,
    "abs": abs,
}

_BINARY: dict[str, Any] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
}

_REDUCTIONS: dict[str, Any] = {
    "sum": (lambda g: math.fsum(g)),
    "mean": (lambda g: math.fsum(g) / len(g)),
    "max": (lambda g: max(g)),
}


def _matmul(a: Any, b: Any) -> Any:
    """Compute ``a @ b`` for rank ≤ 2 operands.

    Covers matmul / matvec / vec-mat / dot; batched (rank > 2)
    matmul is outside the reference subset and raises — an honest
    refusal, not a guess.
    """
    sa, sb = _shape(a), _shape(b)
    if len(sa) > 2 or len(sb) > 2:
        raise KeyError(
            f"FloatOracle matmul supports rank <= 2, got {sa} @ {sb}"
        )
    if sa == () and sb == ():
        return float(a) * float(b)
    if sa == ():
        # scalar @ b — elementwise scale (numpy matmul semantics).
        return _ew2(lambda x, y: x * y, a, b)
    if sb == ():
        return _ew2(lambda x, y: x * y, a, b)
    if len(sa) == 1 and len(sb) == 1:
        if sa[0] != sb[0]:
            raise ValueError(f"dot shape mismatch: {sa} vs {sb}")
        return math.fsum(x * y for x, y in zip(a, b, strict=True))
    if len(sa) == 1:
        # vec @ mat -> (n,)
        n = sb[1]
        return [
            math.fsum(a[k] * b[k][j] for k in range(sa[0]))
            for j in range(n)
        ]
    if len(sb) == 1:
        # mat @ vec -> (m,)
        return [
            math.fsum(a[i][k] * b[k] for k in range(sa[1]))
            for i in range(sa[0])
        ]
    if sa[1] != sb[0]:
        raise ValueError(f"matmul shape mismatch: {sa} vs {sb}")
    return [
        [
            math.fsum(a[i][k] * b[k][j] for k in range(sa[1]))
            for j in range(sb[1])
        ]
        for i in range(sa[0])
    ]


def _transpose(v: Any, attrs: Mapping[str, Any]) -> Any:
    """``transpose`` — swap ``dim0``/``dim1`` (default last two)."""
    shape = _shape(v)
    nd = len(shape)
    if nd == 0:
        return v
    d0 = int(attrs.get("dim0", nd - 2))
    d1 = int(attrs.get("dim1", nd - 1))
    d0 %= nd
    d1 %= nd
    perm = list(range(nd))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    flat, _ = _ravel(v)
    out_shape = tuple(shape[i] for i in perm)
    strides = _strides(shape)
    out: list[float] = []
    for multi in product(*(range(d) for d in out_shape)):
        src = [0] * nd
        for op_i, dim in enumerate(perm):
            src[dim] = multi[op_i]
        out.append(
            flat[sum(i * s for i, s in zip(src, strides, strict=True))]
        )
    return _nest(out_shape, iter(out))


def _reshape(v: Any, attrs: Mapping[str, Any]) -> Any:
    """``reshape`` — ``shape`` attr tuple; one ``-1`` infers."""
    shape = _shape(v)
    target = attrs.get("shape", attrs.get("dims"))
    if target is None:
        raise KeyError("FloatOracle reshape needs a shape attr")
    t = tuple(int(d) for d in target)
    known = math.prod(d for d in t if d != -1)
    n_unknown = sum(1 for d in t if d == -1)
    total = math.prod(shape)
    if n_unknown > 1 or (n_unknown == 1 and total % known):
        raise ValueError(f"bad reshape target {t} for shape {shape}")
    if n_unknown == 1:
        t = tuple(total // known if d == -1 else d for d in t)
    if math.prod(t) != total:
        raise ValueError(f"reshape {shape} -> {t} size mismatch")
    flat, _ = _ravel(v)
    return _nest(t, iter(flat))


def _concat(args: list[Any], attrs: Mapping[str, Any]) -> Any:
    """``concat`` — join tensors along ``dim`` (default 0)."""
    if not args:
        raise KeyError("FloatOracle concat needs >= 1 operand")
    dim = int(attrs.get("dim", 0))

    def rec(items: list[Any], d: int) -> Any:
        if d == 0:
            out: list[Any] = []
            for it in items:
                out.extend(it)
            return out
        return [
            rec([it[i] for it in items], d - 1)
            for i in range(len(items[0]))
        ]

    shapes = [_shape(a) for a in args]
    nd = len(shapes[0])
    d = dim % nd
    # Every non-concat dim must agree across operands.
    for s in shapes[1:]:
        for i, (x, y) in enumerate(zip(shapes[0], s, strict=True)):
            if i != d and x != y:
                raise ValueError(
                    f"concat dim mismatch at axis {i}: {shapes[0]} vs {s}"
                )
    return rec(args, d)


class FloatOracle:
    """The reference :class:`~mlopt_core.ports.VerifyOracle`.

    ``run`` interprets a term over nested-list f64 tensors;
    ``compare`` restates the historical ``verify_equiv`` gate.  See
    the module docstring for the op subset and honest limits.
    """

    name = "float64"

    def run(self, term: Any, env: Mapping[str, Any]) -> Any:
        """Evaluate ``term`` under leaf bindings ``env``."""
        return self._eval(term, env, {})

    def compare(
        self,
        ref: Any,
        opt: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> VerifyResult:
        """Gate two outputs: ``max_rel < rtol`` and ``max_abs <= atol``.

        Pairs output pytrees leaf-wise (``tuple`` positionally,
        ``dict`` by key — a mismatched structure raises).  A shape
        mismatch inside a leaf pair is a *failed report*
        (``max_abs = max_rel = inf``), never a broadcast.
        """
        max_abs = 0.0
        max_rel = 0.0
        passed = True
        per: list[float] = []
        for a, b in _leaves(ref, opt):
            sa, sb = _shape(a), _shape(b)
            if sa != sb:
                return OracleReport(
                    max_abs=float("inf"),
                    max_rel=float("inf"),
                    passed=False,
                    rtol=rtol,
                    per_output=(float("inf"),),
                )
            fa, _ = _ravel(a)
            fb, _ = _ravel(b)
            d = max(
                (abs(x - y) for x, y in zip(fa, fb, strict=True)),
                default=0.0,
            )
            r = max((abs(x) for x in fa), default=0.0)
            mr = d / (r + _REL_FLOOR)
            per.append(d)
            max_abs = max(max_abs, d)
            max_rel = max(max_rel, mr)
            if not (mr < rtol and (atol is None or d <= atol)):
                passed = False
        return OracleReport(
            max_abs=max_abs,
            max_rel=max_rel,
            passed=passed,
            rtol=rtol,
            per_output=tuple(per),
        )

    # -- the interpreter -------------------------------------------------

    def _eval(
        self, term: Any, env: Mapping[str, Any], memo: dict
    ) -> Any:
        """Evaluate ``term``; ``memo`` dedups the hash-consed DAG."""
        if isinstance(term, Op):
            hit = memo.get(term)
            if hit is not None:
                return hit
            out = self._eval_op(term, env, memo)
            memo[term] = out
            return out
        if isinstance(term, (Var, Param)):
            try:
                return env[term.name]
            except KeyError:
                raise KeyError(
                    f"FloatOracle env has no leaf {term.name!r}"
                ) from None
        if isinstance(term, Const):
            return term.value
        raise TypeError(f"FloatOracle cannot evaluate {term!r}")

    def _eval_op(
        self, term: Op, env: Mapping[str, Any], memo: dict
    ) -> Any:
        """Dispatch one ``Op`` node over its evaluated args."""
        op, attrs = term.op, term.attrs
        if op == "pack":
            return tuple(self._eval(a, env, memo) for a in term.args)
        if op == "contiguous":
            # A memory-layout coercion — the identity on values.
            if len(term.args) != 1:
                raise KeyError("FloatOracle contiguous takes 1 arg")
            return self._eval(term.args[0], env, memo)
        if op == "float":
            # Dtype coercion: identity only when the target IS the
            # oracle's f64 domain — a request for another precision
            # cannot be honoured and must not silently pass.
            dtype = attrs.get("dtype")
            if dtype is not None and str(dtype).split(".")[-1] not in (
                "float64",
                "f64",
                "double",
            ):
                raise KeyError(
                    f"FloatOracle cannot honour float(dtype={dtype!r})"
                )
            if len(term.args) != 1:
                raise KeyError("FloatOracle float takes 1 arg")
            return self._eval(term.args[0], env, memo)
        args = [self._eval(a, env, memo) for a in term.args]
        if op in _UNARY:
            if len(args) != 1:
                raise KeyError(f"FloatOracle {op} takes 1 arg")
            return _map1(_UNARY[op], args[0])
        if op in _BINARY:
            if len(args) != 2:
                raise KeyError(f"FloatOracle {op} takes 2 args")
            return _ew2(_BINARY[op], args[0], args[1])
        if op in _REDUCTIONS:
            if len(args) != 1:
                raise KeyError(f"FloatOracle {op} takes 1 arg")
            dims = _dims_of(attrs)
            keep = bool(attrs.get("keepdim", False))
            return _reduce(args[0], dims, _REDUCTIONS[op], keep)
        if op == "matmul":
            if len(args) != 2:
                raise KeyError("FloatOracle matmul takes 2 args")
            return _matmul(args[0], args[1])
        if op == "transpose":
            if len(args) != 1:
                raise KeyError("FloatOracle transpose takes 1 arg")
            return _transpose(args[0], attrs)
        if op == "reshape":
            if len(args) != 1:
                raise KeyError("FloatOracle reshape takes 1 arg")
            return _reshape(args[0], attrs)
        if op == "concat":
            return _concat(args, attrs)
        raise KeyError(
            f"FloatOracle does not implement op {op!r} "
            f"(supported: {sorted(_SUPPORTED_OPS)})"
        )


_SUPPORTED_OPS = frozenset(
    set(_UNARY)
    | set(_BINARY)
    | set(_REDUCTIONS)
    | {
        "pack",
        "contiguous",
        "float",
        "matmul",
        "transpose",
        "reshape",
        "concat",
    }
)
