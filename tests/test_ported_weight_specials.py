"""Substrate-pure tests ported from ``graphopt:tests/test_weight_specials.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Op, Param, TensorType, Var, op_repr

from mlopt_core.laws.specials import (
    _analyse,
    _blocks,
    _dedup,
    _offer_one_site,
    offer_weight_specials,
)


def test_dedup_index_map():
    """First-occurrence grouping on raw-byte signatures."""
    sigs = [b"aa", b"bb", b"aa", b"cc", b"bb"]
    imap, firsts = _dedup(sigs)
    assert imap == (0, 1, 0, 2, 1)
    assert firsts == (0, 1, 3)

def test_blocks_decline_cases():
    """``_blocks`` declines: <2 components, non-contiguous rows, dead
    rows splitting the tiling."""
    # one component — every row spans the full input
    assert _blocks([(0, 4), (0, 4)], 2, 4) is None
    # two components but row sets interleave ({0,2} vs {1})
    assert _blocks([(0, 1), (2, 3), (0, 1)], 3, 4) is None
    # clean 2-block tiling plus a trailing dead row -> not a pure
    # block-diagonal (the elide member covers dead outputs)
    assert _blocks([(0, 2), (2, 4), None], 3, 4) is None
    # clean diagonal-ordered 2x2 tiling
    assert _blocks([(0, 2), (0, 2), (2, 4), (2, 4)], 4, 4) == [
        (0, 2, 0, 2),
        (2, 4, 2, 4),
    ]

def test_offer_one_site_direct_decline():
    """Direct unit cover for the guard clause."""
    x = Var("x", TensorType((8, 32)))
    w = Param("p_w", TensorType((64, 32)))
    eg = EGraph()
    eid = eg.add_term(Op.make("linear", x, w))
    cid = eg.find(eid)
    node = next(
        n for n in eg._classes[cid].nodes if n.op == "linear"
    )
    assert (
        _offer_one_site(
            eg, cid, node, "linear", node.children[0], w, {}, {},
            True,
        )
        == []
    )
