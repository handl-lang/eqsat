"""Substrate-pure tests ported from ``graphopt:tests/test_cert_nonlocal.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core.egraph import (
    CertificateVerificationError,
    EGraph,
    Rewrite,
    _term_instantiate,
    _term_match,
    verify_certificate,
)

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var, op_repr


def test_bare_union_without_witness_untouched():
    """The mechanism is opt-in: a plain union still records rule=None
    and certificates flag it egraph_dependent."""
    x = Var("x", TensorType((4,)))
    a = Op.make("neg", x)
    b = Op.make("square", x)
    eg = EGraph()
    ea, eb = eg.add_term(a), eg.add_term(b)
    assert eg.union(ea, eb)
    assert eg.merge_log[-1].rule is None
    cert = eg.certificate(a, b)
    assert cert.n_egraph_dependent == 1
    with pytest.raises(CertificateVerificationError):
        verify_certificate(a, cert, strict=True)

def test_union_witness_api_on_manual_merge():
    """``eg.union(a, b, witness=Rewrite(lhs, rhs))``: the pass asserts
    ``lhs == rhs`` for these concrete terms; the certificate replays
    the assertion as a named rewrite step."""
    x = Var("x", TensorType((4,)))
    src = Op.make("neg", x)
    dst = Op.make("mul", x, Const(-1.0))  # a true pointwise equality
    eg = EGraph()
    ea, eb = eg.add_term(src), eg.add_term(dst)
    w = Rewrite(
        "demo_witness", lhs=src, rhs=dst, law="demo: neg(x) = x * -1"
    )
    assert eg.union(ea, eb, witness=w)
    assert eg.merge_log[-1].rule == "demo_witness"

    cert = eg.certificate(src, dst)
    assert cert.n_egraph_dependent == 0
    assert cert.rules_used == ["demo_witness"]
    out = verify_certificate(src, cert, strict=True)
    assert op_repr(out) == op_repr(dst)

    # and the witness survives tampering checks like any rule step
    cert.steps[0].rhs = Op.make("mul", x, Const(2.0))
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert, strict=True)
