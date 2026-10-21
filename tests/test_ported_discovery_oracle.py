"""Substrate-pure tests ported from ``graphopt:tests/test_discovery_oracle.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.ir import Const, Op, Param, TensorType, Var


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))

def test_rms_norm_fold_guard_accepts_a_tail_block_binding():
    """The ``tail-block`` spec is correct: the gained fold's guard
    accepts ``u=(2,3,4)``, ``w=(4,)`` with ``MD`` naming ``u``'s last
    axis (and the ``(0,1,2)`` spelling with ``w=(2,3,4)``).

    The ``0/6000`` synth measurement is an enumeration-ordering
    artifact — the four-free-operand diagonal buries the
    non-scalar-``u`` corner (all 94 front-passing sites at 6000 have
    ``u=()``) — not a mis-specified guard.
    """
    from mlopt_core.laws import ALL_RULES

    rules = {r.name: r for r in ALL_RULES}
    gained = rules["rms_norm_fold"]
    nogain = rules["rms_norm_fold_nogain"]
    base = {
        "u": _v("u", 2, 3, 4),
        "w": _v("w", 4),
        "EPS": Const(0.5),
        "P": Const(2),
        "$attr:MK": True,
        "$attr:MD": -1,
    }
    assert gained.check(base)
    assert nogain.check(base)
    multi = dict(base, w=_v("w", 2, 3, 4), **{"$attr:MD": (0, 1, 2)})
    assert gained.check(multi)
    # a scalar u has no trailing block — the guard declines.
    assert not gained.check(dict(base, u=_v("u")))
