"""Substrate-pure tests ported from ``graphopt:tests/test_discovery_experiments.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.ir import Const, Op, Param, TensorType, Var

from mlopt_core.typing import INVALID, _shape_of


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)

def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))

def test_chunk_split_nonpositive_counts_invalid():
    """``chunks=0`` / ``sections=0`` divide by zero — the same class
    as the conv zero-stride defect: decline ``INVALID``, don't crash.
    A non-int count (``2.0``) declines to ``None`` rather than mint a
    float dimension."""
    x = _v("x", 8)
    assert _shape_of(_p("chunk", x, chunks=0)) is INVALID
    assert _shape_of(_p("tensor_split", x, sections=0, index=0)) is INVALID
    assert (
        _shape_of(_p("tensor_split", x, sections=-2, index=0))
        is INVALID
    )
    assert _shape_of(_p("chunk", x, chunks=2.0)) is None
    # valid spellings unchanged.
    assert _shape_of(_p("chunk", x, chunks=2)) == (4,)
    assert _shape_of(_p("tensor_split", x, sections=2, index=0)) == (4,)
    # an index-list sections payload keeps the bounds path.
    assert _shape_of(
        _p("tensor_split", x, sections=[3, 5], index=1)
    ) == (2,)
