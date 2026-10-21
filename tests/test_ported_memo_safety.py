"""Substrate-pure tests ported from ``graphopt:tests/test_memo_safety.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.ir import Const, Op, Param, TensorType, Var, op_repr

from mlopt_core.meta import canonicalize, match_pattern

from mlopt_core.typing import has_var_leaf


def _T(*shape):
    return TensorType(tuple(shape))

def _fresh(term):
    """Re-mint *term* as an equal but object-distinct tree.

    Direct dataclass construction — never ``Op.make`` — so nothing is
    interned and no node object is shared with the input.
    """
    if isinstance(term, Op):
        return Op(
            term.op,
            tuple(_fresh(a) for a in term.args),
            dict(term.attrs),
        )
    if isinstance(term, Var):
        return Var(term.name, term.typ)
    if isinstance(term, Param):
        return Param(term.name, term.typ)
    if isinstance(term, Const):
        return Const(term.value)
    return term

def _mm():
    """matmul(x:(2,3), w:(3,4)) — separately minted on every call."""
    return Op(
        "matmul",
        (Var("x", _T(2, 3)), Param("w", _T(3, 4))),
        {},
    )

def test_fresh_mint_is_equal_but_distinct():
    t = _mm()
    t2 = _fresh(t)
    assert t2 is not t
    assert t2 == t and hash(t2) == hash(t)

def test_has_var_shared_memo_content_keyed():
    t1 = Op("add", (_mm(), Const(0.0)), {})
    t2 = _fresh(t1)
    memo: dict = {}
    assert has_var_leaf(t1, memo) is True
    n = len(memo)
    # Second call over the twin: content-keyed hits, id-keyed would
    # append a whole second set of entries for the twin objects.
    assert has_var_leaf(t2, memo) is True
    assert len(memo) == n

def test_canonicalize_shared_memo_consistent():
    x = Var("x", _T(2))
    y = Var("y", _T(2))
    inner = Op("add", (x, Const(0.0)), {})  # identity element drops
    t1 = Op("add", (inner, y), {})
    t2 = _fresh(t1)
    memo: dict = {}
    c1 = canonicalize(t1, memo)
    n = len(memo)
    c2 = canonicalize(t2, memo)
    assert c1 == c2
    assert op_repr(c1) == op_repr(c2)
    assert len(memo) == n

def test_match_pattern_metavar_binds_equal_distinct_terms():
    # A repeated metavar must bind STRUCTURALLY equal subterms — not
    # the same object.  an ``is`` check would refuse the twin.
    pat = Op("add", ("$a", "$a"), {})
    a1 = _mm()
    term = Op("add", (a1, _fresh(a1)), {})
    subst = match_pattern(pat, term)
    assert subst is not None
    assert subst["$a"] == a1
