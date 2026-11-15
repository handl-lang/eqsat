"""The ``VerifyOracle`` port — protocol, reference oracle, plumbing.

Plan 0037 phase 0 / plan 0039: ``lower(verify_oracle=)`` swaps the
sink's runnable-level ``verify`` for a term-level oracle gate —
``run(term, env)`` evaluates IR terms in the oracle's own value
domain and ``compare`` produces the ``VerifyResult``.  The default
path (``sink.verify``) is unchanged; these tests pin both.
"""

from __future__ import annotations

import pytest
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_core.ops import OpTable
from mlopt_core.oracle import FloatOracle, OracleReport
from mlopt_core.ports import (
    ExecutorSpec,
    VerifyOracle,
    VerifyResult,
)
from mlopt_orchestrator.optimize import Optimizer, _oracle_env

# ---------------------------------------------------------------------------
#  FloatOracle.run — the term interpreter over nested-list f64 tensors
# ---------------------------------------------------------------------------


def _env(**kw):
    return kw


def test_run_leaves_and_const():
    o = FloatOracle()
    x = Var("x", TensorType((2,)))
    p = Param("w", TensorType((2,)))
    env = _env(x=[1.0, 2.0], w=[3.0, 4.0])
    assert o.run(x, env) == [1.0, 2.0]
    assert o.run(p, env) == [3.0, 4.0]
    assert o.run(Const(2), env) == 2
    assert o.run(Const(2.5), env) == 2.5


def test_run_elementwise_and_broadcast():
    o = FloatOracle()
    x = Var("x", TensorType((2, 2)))
    env = _env(x=[[1.0, 2.0], [3.0, 4.0]])
    t = Op.make("add", x, Const(1))
    assert o.run(t, env) == [[2.0, 3.0], [4.0, 5.0]]
    # Row-vector broadcast against a (2, 2) tensor.
    b = Var("b", TensorType((2,)))
    env["b"] = [10.0, 20.0]
    t2 = Op.make("mul", x, b)
    assert o.run(t2, env) == [[10.0, 40.0], [30.0, 80.0]]


def test_run_matmul_variants():
    o = FloatOracle()
    a = Var("a", TensorType((2, 3)))
    b = Var("b", TensorType((3, 2)))
    v = Var("v", TensorType((3,)))
    env = _env(
        a=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        b=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        v=[1.0, 1.0, 1.0],
    )
    assert o.run(Op.make("matmul", a, b), env) == [
        [4.0, 5.0],
        [10.0, 11.0],
    ]
    # mat-vec and vec-mat.
    assert o.run(Op.make("matmul", a, v), env) == [6.0, 15.0]
    assert o.run(Op.make("matmul", v, b), env) == [2.0, 2.0]
    # dot.
    assert o.run(Op.make("matmul", v, v), env) == 3.0


def test_run_reductions():
    o = FloatOracle()
    x = Var("x", TensorType((2, 3)))
    env = _env(x=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert o.run(Op.make("sum", x), env) == 21.0
    assert o.run(Op.make("mean", x), env) == pytest.approx(3.5)
    assert o.run(Op.make("max", x), env) == 6.0
    # dim reduction + keepdim.
    t = Op.make("sum", x, dim=1, keepdim=True)
    assert o.run(t, env) == [[6.0], [15.0]]
    t2 = Op.make("sum", x, dims=(0,))
    assert o.run(t2, env) == [5.0, 7.0, 9.0]
    t3 = Op.make("mean", x, dim=-1)
    assert o.run(t3, env) == pytest.approx([2.0, 5.0])


def test_run_shape_ops():
    o = FloatOracle()
    x = Var("x", TensorType((2, 3)))
    env = _env(x=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert o.run(Op.make("transpose", x, dim0=0, dim1=1), env) == [
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ]
    assert o.run(Op.make("reshape", x, shape=(3, 2)), env) == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    assert o.run(Op.make("reshape", x, shape=(-1,)), env) == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    a = Var("a", TensorType((2,)))
    b = Var("b", TensorType((2,)))
    env.update(a=[7.0, 8.0], b=[9.0, 10.0])
    assert o.run(Op.make("concat", a, b, dim=0), env) == [
        7.0,
        8.0,
        9.0,
        10.0,
    ]


def test_run_pack_and_shared_dag():
    o = FloatOracle()
    x = Var("x", TensorType((2,)))
    env = _env(x=[1.0, 2.0])
    shared = Op.make("mul", x, x)  # hash-consed: one object, two uses
    root = Op.make("pack", shared, Op.make("add", shared, Const(1)))
    assert o.run(root, env) == ([1.0, 4.0], [2.0, 5.0])


def test_run_unsupported_and_missing_leaf_are_loud():
    o = FloatOracle()
    x = Var("x", TensorType((2,)))
    env = _env(x=[1.0, 2.0])
    with pytest.raises(KeyError, match="does not implement"):
        o.run(Op.make("sdpa", x, x, x), env)
    with pytest.raises(KeyError, match="no leaf"):
        o.run(Var("missing", TensorType((1,))), env)
    # A non-f64 dtype request is refused, not silently f64'd.
    with pytest.raises(KeyError, match="cannot honour"):
        o.run(Op.make("float", x, dtype="float32"), env)
    # f64 spelling is the identity.
    assert o.run(Op.make("float", x, dtype="float64"), env) == [
        1.0,
        2.0,
    ]


# ---------------------------------------------------------------------------
#  FloatOracle.compare — the historical verify_equiv gate, restated
# ---------------------------------------------------------------------------


def test_compare_pass_and_fail():
    o = FloatOracle()
    rep = o.compare([1.0, 2.0], [1.0, 2.0])
    assert isinstance(rep, VerifyResult)
    assert rep.passed and rep.max_abs == 0.0 and rep.max_rel == 0.0
    rep2 = o.compare([1.0, 2.0], [1.0, 2.5])
    assert not rep2.passed
    assert rep2.max_abs == pytest.approx(0.5)
    # max_rel denominator is max|ref| + 1e-8: 0.5 / 2.0.
    assert rep2.max_rel == pytest.approx(0.25)


def test_compare_atol_gate():
    o = FloatOracle()
    # max_rel 0.25 fails rtol=1e-4, but max_abs 0.5 <= atol=1.0 alone
    # cannot rescue it — BOTH gates must hold (verify_equiv semantics).
    rep = o.compare([1.0, 2.0], [1.0, 2.5], atol=1.0)
    assert not rep.passed
    rep2 = o.compare([1.0, 2.0], [1.0, 2.00001], rtol=1e-3, atol=1.0)
    assert rep2.passed


def test_compare_shape_mismatch_fails_closed():
    o = FloatOracle()
    rep = o.compare([1.0, 2.0], [1.0, 2.0, 3.0])
    assert not rep.passed
    assert rep.max_abs == float("inf") and rep.max_rel == float("inf")


def test_compare_pytree_outputs():
    o = FloatOracle()
    rep = o.compare(([1.0], {"k": [2.0]}), ([1.0], {"k": [2.0]}))
    assert rep.passed
    rep2 = o.compare(([1.0], {"k": [2.0]}), ([1.0], {"k": [2.5]}))
    assert not rep2.passed
    with pytest.raises(TypeError):
        o.compare(([1.0],), {"k": [1.0]})


# ---------------------------------------------------------------------------
#  The port itself
# ---------------------------------------------------------------------------


def test_float_oracle_satisfies_protocol():
    assert isinstance(FloatOracle(), VerifyOracle)
    assert isinstance(OracleReport(0.0, 0.0, True), VerifyResult)
    assert not isinstance(object(), VerifyOracle)


def test_duck_typed_oracle_conforms():
    """Any object with run+compare is a VerifyOracle — no subclass."""

    class Mini:
        def run(self, term, env):
            return 0.0

        def compare(self, ref, opt, *, rtol=1e-4, atol=None):
            return OracleReport(0.0, 0.0, True)

    assert isinstance(Mini(), VerifyOracle)


# ---------------------------------------------------------------------------
#  lower(verify_oracle=...) — the plumbing, end to end on a fake backend
# ---------------------------------------------------------------------------


class _Model:
    """The 'model' the fake source exports — identity on lists."""

    def __call__(self, x):
        return x


class _OracleModule:
    """A lowered module whose forward IS the oracle's ``run``."""

    def __init__(self, ir, params, oracle):
        self._ir = ir
        self._params = dict(params or {})
        self._oracle = oracle

    def forward(self, *xs):
        env = dict(self._params)
        for v, x in zip(self._ir.inputs, xs, strict=False):
            env[v.name] = x
        return self._oracle.run(self._ir.root, env)

    __call__ = forward


class _OracleSink:
    """A Sink whose verify delegates to the oracle's ``compare``."""

    supported_ops = frozenset(
        {"add", "sub", "mul", "div", "neg", "matmul", "sum", "mean"}
    )

    def __init__(self, oracle=None):
        self._oracle = oracle or FloatOracle()
        self.verify_calls = 0

    @property
    def ops(self):
        return OpTable.core()

    @property
    def executors(self):
        return {
            "generic": ExecutorSpec(
                name="generic",
                accepts=lambda ir, term: True,
                lower=self.lower,
                engaged=lambda mod: True,
            )
        }

    def lower(self, ir, params=None):
        return _OracleModule(ir, params, self._oracle)

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        self.verify_calls += 1
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        return self._oracle.compare(
            ref(*args), opt(*args), rtol=rtol, atol=atol
        )


class _OracleSource:
    """Source — a fixed two-input IR over add/mul."""

    def to_ir(self, model, example_inputs):
        x = Var("x", TensorType((2,)))
        y = Var("y", TensorType((2,)))
        w = Param("w", TensorType((2,)))
        root = Op.make(
            "add", Op.make("mul", x, w), Op.make("mul", y, w)
        )
        ir = IR(
            root=root,
            inputs=[x, y],
            input_names={"x", "y"},
            params={"w": w},
        )
        return ir, {"w": [2.0, 3.0]}


def _opt(oracle=None):
    sink = _OracleSink(oracle)
    return (
        Optimizer(
            source=_OracleSource(),
            sink=sink,
            engine="python-egraph",
        ),
        sink,
    )


def test_lower_with_verify_oracle_passes():
    oracle = FloatOracle()
    opt, sink = _opt(oracle)
    res = opt.search(_Model(), ([1.0, 1.0], [0.5, 0.5]))
    lr = opt.lower(res, ([1.0, 1.0], [0.5, 0.5]), verify_oracle=oracle)
    assert lr.verified is not None and lr.verified.passed
    assert lr.verified.max_abs == 0.0
    # The oracle gate replaced sink.verify — it was never called.
    assert sink.verify_calls == 0


def test_lower_verify_oracle_records_the_terms_it_saw():
    """The oracle observes result.ir.root vs result.term — terms,
    not the delivered module."""
    seen = []

    class Recording(FloatOracle):
        def run(self, term, env):
            seen.append(term)
            return super().run(term, env)

    oracle = Recording()
    opt, _sink = _opt(oracle)
    x = ([1.0, 1.0], [0.5, 0.5])
    res = opt.search(_Model(), x)
    lr = opt.lower(res, x, verify_oracle=oracle)
    assert lr.verified.passed
    # Two oracle runs in the verify gate: source root then extracted.
    assert seen[0] == res.ir.root
    assert seen[1] == res.term


def test_lower_verify_oracle_failure_declines_delivery():
    """A dishonest oracle that reports a diff refuses the delivery.

    The verify gate is substrate-enforced: ``lower(verify=True)``
    raises :class:`VerifyDeclinedError` — the declined
    ``LowerResult`` rides the exception as ``.result`` so the failing
    report stays inspectable.
    """
    from mlopt_orchestrator import VerifyDeclinedError

    class Liar(FloatOracle):
        def compare(self, ref, opt, *, rtol=1e-4, atol=None):
            return OracleReport(
                max_abs=1.0, max_rel=1.0, passed=False, rtol=rtol
            )

    opt, _sink = _opt()
    x = ([1.0, 1.0], [0.5, 0.5])
    res = opt.search(_Model(), x)
    with pytest.raises(VerifyDeclinedError) as ei:
        opt.lower(res, x, verify_oracle=Liar())
    declined = ei.value.result
    assert declined is not None
    assert (
        declined.verified is not None and not declined.verified.passed
    )
    assert declined.module is not None
    assert declined.stats["accepted_by"] == "declined"
    # The explicit opt-out still ships the module unverified.
    lr = opt.lower(res, x, verify=False, verify_oracle=Liar())
    assert lr.verified is None


def test_lower_default_path_unchanged_without_oracle():
    opt, sink = _opt()
    x = ([1.0, 1.0], [0.5, 0.5])
    res = opt.search(_Model(), x)
    lr = opt.lower(res, x)
    assert lr.verified is not None and lr.verified.passed
    assert sink.verify_calls == 1
    # verify=False still skips everything.
    lr2 = opt.lower(res, x, verify=False, verify_oracle=FloatOracle())
    assert lr2.verified is None


def test_lower_verify_slot_accepts_an_oracle():
    """``verify=<oracle>`` is shorthand for verify_oracle=oracle."""
    oracle = FloatOracle()
    opt, sink = _opt()
    x = ([1.0, 1.0], [0.5, 0.5])
    res = opt.search(_Model(), x)
    lr = opt.lower(res, x, verify=oracle)
    assert lr.verified is not None and lr.verified.passed
    assert sink.verify_calls == 0


def test_oracle_env_binding_rules():
    """_oracle_env: positional tuple, Mapping, and loud mismatches."""
    x = Var("x", TensorType((2,)))
    y = Var("y", TensorType((2,)))
    w = Param("w", TensorType((2,)))
    ir = IR(
        root=Op.make("add", x, y),
        inputs=[x, y],
        input_names={"x", "y"},
        params={"w": w},
    )
    env = _oracle_env(ir, {"w": [1.0, 1.0]}, ([1.0], [2.0]))
    assert env == {"w": [1.0, 1.0], "x": [1.0], "y": [2.0]}
    env2 = _oracle_env(ir, {}, {"x": [3.0], "y": [4.0]})
    assert env2["x"] == [3.0] and env2["y"] == [4.0]
    # Single bare value binds a one-input program.
    ir1 = IR(root=x, inputs=[x], input_names={"x"}, params={})
    assert _oracle_env(ir1, {}, [9.0]) == {"x": [9.0]}
    # Mismatched arity is loud.
    with pytest.raises(TypeError, match="inputs but x supplies"):
        _oracle_env(ir, {}, [1.0])
