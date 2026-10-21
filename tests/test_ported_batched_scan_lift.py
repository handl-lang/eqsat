"""Substrate-pure tests ported from ``graphopt:tests/test_batched_scan_lift.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.typing import _matmul_shape


class TestMatmulShape:
    def test_vector_cases(self):
        assert _matmul_shape((4, 4), (4,)) == (4,)
        assert _matmul_shape((4,), (4, 4)) == (4,)
        assert _matmul_shape((4,), (4,)) == ()
        assert _matmul_shape((4,), (5,)) is None
        assert _matmul_shape((3, 5), (4,)) is None
        assert _matmul_shape((5,), (3, 4)) is None

    def test_matrix_batch_cases(self):
        assert _matmul_shape((2, 3, 3), (2, 3, 1)) == (2, 3, 1)
        assert _matmul_shape((3, 3), (2, 3, 1)) == (2, 3, 1)
        assert _matmul_shape((4,), (2, 4, 5)) == (2, 5)
        # inner-dim / batch mismatch
        assert _matmul_shape((2, 3, 3), (2, 4, 1)) is None
        assert _matmul_shape((2, 3, 3), (5, 3, 3)) is None

    def test_invalid_operands(self):
        assert _matmul_shape(None, (4,)) is None
        assert _matmul_shape((4,), "x") is None
        assert _matmul_shape((), (4,)) is None
        assert _matmul_shape((4,), ()) is None
        # vector-matrix promotion: (k,) @ (…, k, n) -> (…, n)
        assert _matmul_shape((2,), (2, 4)) == (4,)
        assert _matmul_shape((4,), (2, 4)) is None

    def test_none_dims_are_wildcards(self):
        assert _matmul_shape((None, 3, 3), (None, 3, 1)) == (
            None,
            3,
            1,
        )
        assert _matmul_shape((3,), (None,)) == ()
