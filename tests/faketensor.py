"""A duck-typed tensor for exercising core's tensor-library-free paths.

The sharing/specials/factored/headshare passes never name a tensor
library — they read ``shape``/``dtype``/``dim()`` structurally and run
value analysis through a small duck-typed surface (slicing,
``detach/cpu/contiguous/numpy/tobytes``, elementwise comparisons and
arithmetic, ``new_zeros``).  ``_FakeTensor`` implements exactly that
surface over a flat list of Python floats, so the substrate suite can
drive the same branches the domain suites drive with torch tensors.

Signatures are *bitwise*: ``numpy().tobytes()`` returns the repr of the
flat data — injective modulo float repr (``-0.0`` vs ``0.0`` stay
distinct, like raw IEEE bytes).
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Any

__all__ = ["FakeTensor"]


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    st, s = [], 1
    for d in reversed(shape):
        st.insert(0, s)
        s *= d
    return tuple(st)


def _infer(data: Any) -> tuple[tuple[int, ...], list[float]]:
    """Nested lists -> ``(shape, flat)``; scalars -> ``((), [v])``."""
    if isinstance(data, (int, float)):
        return (), [float(data)]
    shape: list[int] = []
    d = data
    while isinstance(d, (list, tuple)):
        shape.append(len(d))
        if not d:
            break
        d = d[0]
    flat: list[float] = []

    def rec(x: Any, depth: int) -> None:
        if depth == len(shape):
            flat.append(float(x))
            return
        for v in x:
            rec(v, depth + 1)

    rec(data, 0)
    return tuple(shape), flat


def _bcast_shape(
    a: tuple[int, ...], b: tuple[int, ...]
) -> tuple[int, ...]:
    ra, rb = list(a), list(b)
    while len(ra) < len(rb):
        ra.insert(0, 1)
    while len(rb) < len(ra):
        rb.insert(0, 1)
    out = []
    for x, y in zip(ra, rb):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"cannot broadcast {a} with {b}")
        out.append(max(x, y))
    return tuple(out)


class _FakeMask:
    """Flat elementwise boolean mask — the ``(a == b)`` result."""

    def __init__(self, bits: Any) -> None:
        self._bits = [bool(b) for b in bits]

    def any(self) -> bool:
        return any(self._bits)

    def all(self) -> bool:
        return all(self._bits)

    def sum(self) -> int:
        return sum(self._bits)

    def cumsum(self, _dim: int = 0) -> "FakeTensor":
        out, s = [], 0
        for b in self._bits:
            s += int(b)
            out.append(s)
        return FakeTensor(out, (len(out),))


class _FakeNumpy:
    """``tensor.numpy()`` — only ``tobytes()`` is consumed."""

    def __init__(self, flat: list[float]) -> None:
        self._flat = list(flat)

    def tobytes(self) -> bytes:
        return repr(self._flat).encode()


class FakeTensor:
    """Flat float data + shape; implements the duck-typed tensor API."""

    def __init__(self, data: Any, shape: tuple[int, ...] | None = None):
        if shape is None:
            shape, flat = _infer(data)
        else:
            flat = [float(v) for v in data]
        self._flat = flat
        self._shape = tuple(int(d) for d in shape)

    # -- structural protocol -----------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return "fake64"

    def dim(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        return prod(self._shape)

    def element_size(self) -> int:
        return 8

    def tolist(self) -> Any:
        def rec(flat: list[float], shape: tuple[int, ...]) -> Any:
            if not shape:
                return flat[0]
            step = prod(shape[1:])
            return [
                rec(flat[j * step : (j + 1) * step], shape[1:])
                for j in range(shape[0])
            ]

        return rec(self._flat, self._shape)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FakeTensor(shape={self._shape}, data={self.tolist()!r})"

    # -- identity / device no-ops -------------------------------------

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def contiguous(self) -> "FakeTensor":
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(list(self._flat), self._shape)

    def numpy(self) -> _FakeNumpy:
        return _FakeNumpy(self._flat)

    def new_zeros(self, shape: tuple[int, ...]) -> "FakeTensor":
        return FakeTensor([0.0] * prod(tuple(shape)), tuple(shape))

    # -- indexing ------------------------------------------------------

    def _offsets(self, key: Any) -> tuple[list[int], tuple[int, ...]]:
        """Flat offsets of ``self[key]`` in read order + result shape."""
        if not isinstance(key, tuple):
            key = (key,)
        consumed = sum(
            1 for k in key if k is not None and k is not Ellipsis
        )
        expanded: list[Any] = []
        for k in key:
            if k is Ellipsis:
                expanded += [slice(None)] * (len(self._shape) - consumed)
            else:
                expanded.append(k)
        while consumed < len(self._shape):
            expanded.append(slice(None))
            consumed += 1
        axis_lists: list[list[int]] = []
        out_shape: list[int] = []
        axis = 0
        for k in expanded:
            if k is None:
                out_shape.append(1)
                continue
            n = self._shape[axis]
            if isinstance(k, int):
                axis_lists.append([k % n])
            elif isinstance(k, slice):
                r = list(range(n))[k]
                axis_lists.append(r)
                out_shape.append(len(r))
            else:  # list / tuple of ints — fancy gather
                r = [int(i) % n for i in k]
                axis_lists.append(r)
                out_shape.append(len(r))
            axis += 1
        strides = _strides(self._shape)
        offs = [
            sum(c * s for c, s in zip(combo, strides))
            for combo in product(*axis_lists)
        ]
        return offs, tuple(out_shape)

    def __getitem__(self, key: Any) -> "FakeTensor":
        offs, out_shape = self._offsets(key)
        return FakeTensor([self._flat[o] for o in offs], out_shape)

    def __setitem__(self, key: Any, value: Any) -> None:
        offs, _ = self._offsets(key)
        if isinstance(value, FakeTensor):
            vals = value._flat
        else:
            vals = [float(value)] * len(offs)
        if len(vals) != len(offs):
            raise ValueError(
                f"cannot assign {len(vals)} values into {len(offs)} slots"
            )
        for o, v in zip(offs, vals):
            self._flat[o] = v

    @property
    def T(self) -> "FakeTensor":
        """2-D transpose — the ``w.T`` orientation flip."""
        if len(self._shape) != 2:
            raise ValueError("T is defined on 2-D fakes only")
        o, i = self._shape
        return FakeTensor(
            [self._flat[r * i + c] for c in range(i) for r in range(o)],
            (i, o),
        )

    def diagonal(self) -> "FakeTensor":
        o, i = self._shape
        return FakeTensor(
            [self._flat[r * i + r] for r in range(min(o, i))],
            (min(o, i),),
        )

    # -- reductions -----------------------------------------------------

    def abs(self) -> "FakeTensor":
        return FakeTensor([abs(v) for v in self._flat], self._shape)

    def max(self) -> float:
        return max(self._flat)

    def sum(self) -> float:
        return float(sum(self._flat))

    def cumsum(self, _dim: int = 0) -> "FakeTensor":
        """1-D prefix sums — ``_row_spans``'s only use."""
        out, s = [], 0.0
        for v in self._flat:
            s += v
            out.append(s)
        return FakeTensor(out, self._shape)

    # -- comparisons -----------------------------------------------------

    def _cmp(self, other: Any, eq: bool) -> _FakeMask:
        if isinstance(other, FakeTensor):
            shape = _bcast_shape(self._shape, other._shape)
            a = self._expand(shape)
            b = other._expand(shape)
        else:
            a, b = self._flat, [float(other)] * len(self._flat)
        return _FakeMask(
            (x == y) if eq else (x != y) for x, y in zip(a, b)
        )

    def __eq__(self, other: Any) -> _FakeMask:  # type: ignore[override]
        return self._cmp(other, True)

    def __ne__(self, other: Any) -> _FakeMask:  # type: ignore[override]
        return self._cmp(other, False)

    __hash__ = None  # mutable container — unhashable like a tensor

    # -- broadcast arithmetic --------------------------------------------

    def _expand(self, out_shape: tuple[int, ...]) -> list[float]:
        if self._shape == out_shape:
            return list(self._flat)
        pad = len(out_shape) - len(self._shape)
        shape = (1,) * pad + self._shape
        strides = _strides(shape)
        out = []
        for idx in product(*(range(d) for d in out_shape)):
            src = [0 if s == 1 else i for i, s in zip(idx, shape)]
            out.append(
                self._flat[sum(i * st for i, st in zip(src, strides))]
            )
        return out

    def _binop(self, other: Any, fn) -> "FakeTensor":
        if isinstance(other, FakeTensor):
            shape = _bcast_shape(self._shape, other._shape)
            a, b = self._expand(shape), other._expand(shape)
        else:
            shape = self._shape
            a, b = self._flat, [float(other)] * len(self._flat)
        return FakeTensor([fn(x, y) for x, y in zip(a, b)], shape)

    def __mul__(self, other: Any) -> "FakeTensor":
        return self._binop(other, lambda x, y: x * y)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> "FakeTensor":
        return self._binop(other, lambda x, y: x + y)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "FakeTensor":
        return self._binop(other, lambda x, y: x - y)

    def __rsub__(self, other: Any) -> "FakeTensor":
        return self._binop(other, lambda x, y: y - x)

    def __truediv__(self, other: Any) -> "FakeTensor":
        return self._binop(other, lambda x, y: x / y)

    def __matmul__(self, other: "FakeTensor") -> "FakeTensor":
        """2-D @ 2-D — the factorisation pass's only use."""
        if len(self._shape) != 2 or len(other._shape) != 2:
            raise ValueError("matmul on 2-D fakes only")
        m, k = self._shape
        k2, n = other._shape
        if k != k2:
            raise ValueError(f"matmul shape mismatch {self._shape}@{other._shape}")
        out = [
            sum(
                self._flat[r * k + t] * other._flat[t * n + c]
                for t in range(k)
            )
            for r in range(m)
            for c in range(n)
        ]
        return FakeTensor(out, (m, n))
