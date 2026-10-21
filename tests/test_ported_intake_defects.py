"""Substrate-pure tests ported from ``graphopt:tests/test_intake_defects.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.ir import IR, Const, Op, TensorType, Var


def test_const_int_float_spellings_are_distinct():
    """``Const(8)`` and ``Const(8.0)`` are different leaves.

    Python's numeric tower makes ``8 == 8.0``; a ``Const`` is a
    structural object (the int-preservation contract in its docstring
    exists because ``x % 2`` needs an int64 operand).  The dataclass
    default ``__eq__``/``__hash__`` leaked that numeric equality into
    structural identity: ``Op.make`` interning returned the
    first-minted spelling, and ``EGraph.add_term``'s content-keyed
    memo mapped the second ``Const`` onto the first's leaf e-class —
    silently rewriting ``* 8.0`` into ``* 8`` in the graph.  The
    e-graph's leaf-key convention (``repr``) is the invariant the
    equality now matches.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import Const

    assert Const(8) != Const(8.0)
    assert Const(8.0) == Const(8.0)
    a = Op.make("mul", Const(8.0), Const(3))
    b = Op.make("mul", Const(8), Const(3))
    assert a is not b and a != b

    eg = EGraph()
    eg.add_term(Op.make("sub", Const(8), Const(8.0)))
    keys = {
        n.attrs[0][1]
        for ec in eg._classes.values()
        for n in ec.nodes
        if n.op == "leaf"
    }
    assert {"8", "8.0"} <= keys

def test_attr_int_float_spellings_are_distinct_terms():
    """``clamp(min=0)`` and ``clamp(min=0.0)`` are different terms.

    The ``Const`` defect one level down: ``_attr_key`` carried raw
    values and ``Op.__eq__`` compared the attrs dicts, so the numeric
    tower (``0 == 0.0``, ``hash(0) == hash(0.0)``) coalesced both
    spellings into one interned object — the second ``Op.make``
    returned the FIRST spelling's term, silently rewriting the attr
    before any law fired.  Identity is now repr-keyed like ``Const``;
    ``0``/``0.0``/``False`` are distinct spellings, and container
    spellings (``[3,5]`` vs ``(3,5)``, ``(3,5)`` vs ``(3.0,5.0)``)
    stay distinct too.
    """
    x = Op.make("leafless_op")  # operand content is irrelevant here
    a = Op.make("clamp", x, min=0)
    b = Op.make("clamp", x, min=0.0)
    assert a is not b and a != b
    assert Op.make("clamp", x, min=0) is a  # interning still dedups
    assert a.attrs == {"min": 0}  # the first spelling keeps its value
    assert Op.make("softmax", x, dim=True) != Op.make(
        "softmax", x, dim=1
    )
    c = Op.make("unflatten", x, dim=0, sizes=(3, 5))
    d = Op.make("unflatten", x, dim=0, sizes=(3.0, 5.0))
    e = Op.make("unflatten", x, dim=0, sizes=[3, 5])
    assert c != d and c != e

def test_attr_int_float_spellings_are_distinct_enodes():
    """``min=0`` and ``min=0.0`` land in different e-classes.

    ``ENode`` field-compare ran the same numeric-tower equality over
    the attr tuple, so ``add_enode``/``add_term`` merged the two
    spellings into one class — the term-level defect reaching the
    graph.  Identity is now the repr-keyed ``_sig``; both enodes keep
    their own class.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import TensorType, Var

    eg = EGraph()
    x = eg.add_leaf("x")
    a = eg.add_enode("clamp", (x,), {"min": 0})
    b = eg.add_enode("clamp", (x,), {"min": 0.0})
    assert eg.find(a) != eg.find(b)
    assert eg.add_enode("clamp", (x,), {"min": 0}) == eg.find(a)

    # add_term goes through the same strict enode identity — a Var
    # named "x" reprs to the same leaf key, so only ``min`` differs.
    c = eg.add_term(
        Op.make("clamp", Var("x", TensorType(())), min=0.0)
    )
    assert eg.find(c) == eg.find(b)

def test_attr_match_stays_numerically_lenient():
    """Pattern ``min=0`` still matches a term spelled ``min=0.0``.

    Strictness is *identity* — interning, ``__eq__``, enode keys.
    Matching keeps the numeric leniency the matchers always had
    (``nv != pv``), the same split ``Const`` got: strict object,
    lenient matcher.  Attr metavariables bind the node's raw value.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_core.egraph.terms import _term_match
    from mlopt_core.ir import Const
    from mlopt_core.meta import match_pattern

    pat = Op.make("clamp", "v", min=0)
    term = Op.make("clamp", Const(1), min=0.0)
    assert _term_match(pat, term) is not None
    assert match_pattern(pat, term) == {"v": Const(1)}

    # The e-graph matcher agrees — and an attr metavar binds the
    # node's raw (un-normalised) spelling, ``0.0`` here.
    eg = EGraph()
    root = eg.add_term(term)
    subs = list(eg.matches(Op.make("clamp", "v", min="m"), root))
    assert subs and subs[0]["$attr:m"] == 0.0

def test_infer_op_shape_declines_on_str_attr_metavar():
    """``_infer_op_shape`` returns ``None`` (unknown), not a TypeError,
    when an axis attribute is still a string metavariable name."""
    from mlopt_core.typing import _infer_op_shape

    unsq = Op.make("unsqueeze", Var("k", TensorType((2, 3))), dim="UDk")
    assert _infer_op_shape(unsq) is None
