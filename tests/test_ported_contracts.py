"""Substrate-pure tests ported from ``graphopt:tests/test_contracts.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from mlopt_core.cost import _INVALID, _shape_of

from mlopt_core.ir import Const, Op, Param, TensorType, Var


def _var(shape: tuple, name: str) -> Var:
    return Var(name=name, typ=TensorType(tuple(shape)))

_INT64_MAX = 2**63 - 1

class TestShapeContracts:
    def test_slice_strided_last_dim(self):
        # x[..., ::2] on (2, 3, 8): torch.export emits
        # slice(t, dim, start, end, step).
        x = _var((2, 3, 8), "x")
        t = Op.make(
            "slice", x, dim=2, start=0, end=_INT64_MAX, step=2
        )
        assert _shape_of(t) == (2, 3, 4)

    def test_slice_strided_odd_extent(self):
        x = _var((2, 3, 7), "x")
        t = Op.make("slice", x, dim=2, start=0, end=None, step=2)
        assert _shape_of(t) == (2, 3, 4)  # ceil(7/2)

    def test_slice_default_step_is_identity(self):
        x = _var((2, 8), "x")
        t = Op.make("slice", x, dim=1, start=0, end=_INT64_MAX)
        assert _shape_of(t) == (2, 8)

    def test_slice_dim_spelling(self):
        # 'dim' is the canonical axis name.
        x = _var((2, 8), "x")
        t = Op.make("slice", x, dim=1, start=2, end=6)
        assert _shape_of(t) == (2, 4)

    def test_split_sizes_index_spelling(self):
        x = _var((2, 12), "x")
        t = Op.make("split", x, sizes=(4, 8), dim=1, index=1)
        assert _shape_of(t) == (2, 8)

    def test_split_sizes_list_spelling(self):
        # exported list-split keeps the list under ``sizes`` and the
        # getitem-fold index under ``index``.
        x = _var((2, 12), "x")
        t = Op.make("split", x, sizes=(4, 8), index=1, dim=1)
        assert _shape_of(t) == (2, 8)

    def test_split_sizes_int_spelling(self):
        # torch.split(x, 4) — equal-size sections, int under ``sizes``.
        x = _var((2, 12), "x")
        t = Op.make("split", x, sizes=4, index=0, dim=1)
        assert _shape_of(t) == (2, 4)

    def test_concat_sums_along_dim(self):
        a, b, c = _var((2, 4), "a"), _var((2, 8), "b"), _var((2, 1), "c")
        t = Op.make("concat", a, b, c, dim=1)
        assert _shape_of(t) == (2, 13)

    def test_concat_dim_spelling(self):
        a, b = _var((2, 4), "a"), _var((2, 8), "b")
        t = Op.make("concat", a, b, dim=-1)
        assert _shape_of(t) == (2, 12)

    def test_reshape_valid(self):
        x = _var((2, 6), "x")
        assert _shape_of(Op.make("reshape", x, shape=(3, 4))) == (3, 4)

    def test_reshape_minus_one_resolved(self):
        x = _var((2, 6), "x")
        assert _shape_of(Op.make("reshape", x, shape=(3, -1))) == (3, 4)

    def test_reshape_numel_mismatch_is_invalid(self):
        x = _var((2, 6), "x")
        assert _shape_of(Op.make("reshape", x, shape=(3, 5))) is _INVALID

    def test_transpose_rank3(self):
        x = _var((2, 3, 4), "x")
        assert _shape_of(
            Op.make("transpose", x, dim0=1, dim1=2)
        ) == (2, 4, 3)
        assert _shape_of(
            Op.make("transpose", x, dim0=-2, dim1=-1)
        ) == (2, 4, 3)

    def test_matmul_matvec(self):
        w, v = _var((4, 3), "w"), _var((3,), "v")
        assert _shape_of(Op.make("matmul", w, v)) == (4,)

    def test_matmul_dot(self):
        a, b = _var((3,), "a"), _var((3,), "b")
        assert _shape_of(Op.make("matmul", a, b)) == ()

    def test_linear_matvec(self):
        x, w = _var((3,), "x"), _var((4, 3), "w")
        assert _shape_of(Op.make("linear", x, w)) == (4,)

    def test_broadcast_mismatch_is_invalid(self):
        a, b = _var((2, 3), "a"), _var((2, 4), "b")
        assert _shape_of(Op.make("add", a, b)) is _INVALID
        assert _shape_of(Op.make("mul", a, b)) is _INVALID

    def test_broadcast_valid(self):
        a, b = _var((2, 1), "a"), _var((1, 3), "b")
        assert _shape_of(Op.make("add", a, b)) == (2, 3)

    def test_rms_norm_passthrough_shape(self):
        x, w = _var((2, 8), "x"), _var((8,), "w")
        t = Op.make("rms_norm", x, w, dim=(8,), eps=1e-5)
        assert _shape_of(t) == (2, 8)
