"""Substrate-pure tests ported from ``graphopt:tests/test_om_decayed.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.cost import flops_cost, launch_aware_cost

from mlopt_core.ir import IR, Const, Op, TensorType, Var

from mlopt_core.typing import _INVALID, _shape_of


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))

def _spelled(s, d, v):
    return Op.make("matmul", Op.make("mul", s, d), v)

RAW = "raw_leaf"

def test_shape_rule_reports_the_applied_shape():
    """(s ⊙ d) @ v — out is (…,T,dv), incl. broadcast batch dims."""
    s, d, v = _v("s", 2, 8, 8), _v("d", 8, 8), _v("v", 2, 8, 16)
    assert _shape_of(Op.make("om_decayed", s, d, v)) == (2, 8, 16)

def test_shape_rule_scalar_operands_report_unknown():
    """A () operand is a carrier member read as a tensor — None."""
    d, v = _v("d", 4, 4), _v("v", 4, 2)
    s = _v("s", 4, 4)
    assert _shape_of(Op.make("om_decayed", Const(1.0), d, v)) is None
    assert _shape_of(Op.make("om_decayed", s, d, Const(1.0))) is None

def test_shape_rule_marks_misuse_invalid():
    """Unbroadcastable mul or a non-contracting pair is _INVALID."""
    assert (
        _shape_of(
            Op.make(
                "om_decayed",
                _v("s", 2, 3),
                _v("d", 4, 5),
                _v("v", 3, 4),
            )
        )
        is _INVALID
    )
    assert (
        _shape_of(
            Op.make(
                "om_decayed",
                _v("s", 4, 8),
                _v("d", 4, 8),
                _v("v", 3, 5),
            )
        )
        is _INVALID
    )

def test_flops_price_the_element_like_the_spelling():
    """om_decayed's FLOPs == mul's broadcast pass + the contraction."""
    s, d, v = _v("s", 2, 8, 8), _v("d", 8, 8), _v("v", 2, 8, 16)
    spelled = _spelled(s, d, v)
    carrier = Op.make("om_decayed", s, d, v)
    assert flops_cost(carrier) == flops_cost(spelled)

def test_flops_fall_back_without_extents():
    """Partial/unknown operand shapes degrade to bounded estimates."""
    s, d, v = _v("s", 4, None), _v("d", 4, None), _v("v", 4, 2)
    # broadcast is (4, None): numel counts the unknown dim as 1, and
    # the contraction axis is unknown so K degrades to 1.
    assert flops_cost(Op.make("om_decayed", s, d, v)) == 4 + 2 * 8
    # An unshaped operand falls straight back to 2·n_out (n_out = 1
    # when the term's own shape is unknown, the operand's otherwise).
    assert flops_cost(Op.make("om_decayed", RAW, d, v)) == 2.0
    assert (
        flops_cost(Op.make("om_decayed", Const(1.0), Const(1.0), v))
        == 2.0
    )
    assert flops_cost(Op.make("om_decayed", s, RAW, v)) == 2 * 4
