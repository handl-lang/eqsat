"""``engine="eqsat"`` — the engine-eqsat (Rust) engine (ADR-0178 WP3+).

Since the WP4 parity sweep v4 GO ``"eqsat"`` is the *default*
(``engine=None`` resolves to it); ``engine="python-egraph"`` is the
explicit opt-out.  Two halves, mirroring ``test_native_engine.py``:

* **Seam tests** (always run): ``_resolve_engine("eqsat")`` materialises
  an :class:`~mlopt_core.ports.Engine`-conforming
  :class:`mlopt_orchestrator.eqsat.EqsatEngine`; a missing ``eqsat-run``
  binary is a named :class:`EqsatUnavailableError`, never a silent
  fallback; unsupported ``stop``/``policy``/extraction-steering
  requests are refused by name.
* **Live tests** (self-skip when no ``eqsat-run`` binary resolves —
  build ``engine-eqsat`` with ``cargo build --bin eqsat-run`` or set
  ``MLOPT_EQSAT_RUN``): opt-in ``search`` vs the default Python
  engine on a small IR, honest law-coverage stats, and the
  ``lower(verify_oracle=...)`` gate proving the searched term still
  verifies.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mlopt_core.cost import flops_cost
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var
from mlopt_core.laws.tensor import FACTOR_MUL_ADD
from mlopt_core.oracle import FloatOracle
from mlopt_core.ports import Engine
from mlopt_orchestrator import Optimizer
from mlopt_orchestrator.eqsat import (
    EqsatEngine,
    EqsatError,
    EqsatUnavailableError,
    _kterm,
    _pyterm,
    _record_synops,
    eqsat_binary,
    portable_laws,
)
from mlopt_orchestrator.optimize import _resolve_engine, search

try:
    _ENGINE = EqsatEngine()
    _HAS_EQSAT = True
except EqsatUnavailableError:  # pragma: no cover — binary absent
    _ENGINE = None
    _HAS_EQSAT = False

requires_eqsat = pytest.mark.skipif(
    not _HAS_EQSAT,
    reason="eqsat-run binary not found (build engine-eqsat or set "
    "MLOPT_EQSAT_RUN)",
)


# ---------------------------------------------------------------------------
#  Sources / sinks — the torch-free doubles from the sibling suites
# ---------------------------------------------------------------------------


class _FakeSource:
    """Source port double: the model IS the ``(ir, params)`` pair."""

    def to_ir(self, model: Any, x: Any) -> tuple[IR, dict]:
        return model, {}


def _simplifying_model() -> IR:
    """``add(mul(x, 1), mul(y, 1))`` — ``id_mul`` is portable."""
    x = Var("x", TensorType((2,)))
    y = Var("y", TensorType((2,)))
    root = Op.make(
        "add",
        Op.make("mul", x, Const(1)),
        Op.make("mul", y, Const(1)),
    )
    return IR(
        root=root, inputs=[x, y], input_names={"x", "y"}, params={}
    )


class _Model:
    """The 'model' the oracle source exports — identity on lists."""

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
        from mlopt_core.ops import OpTable
        from mlopt_core.ports import ExecutorSpec

        self._oracle = oracle or FloatOracle()
        self._ops = OpTable.core()
        self._spec = ExecutorSpec(
            name="generic",
            accepts=lambda ir, term: True,
            lower=self.lower,
            engaged=lambda mod: True,
        )

    @property
    def ops(self):
        return self._ops

    @property
    def executors(self):
        return {"generic": self._spec}

    def lower(self, ir, params=None):
        return _OracleModule(ir, params, self._oracle)

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        return self._oracle.compare(
            ref(*args), opt(*args), rtol=rtol, atol=atol
        )


class _OracleSource:
    """Source — ``add(mul(x, 1), mul(y, 1))`` over two inputs."""

    def to_ir(self, model, example_inputs):
        x = Var("x", TensorType((2,)))
        y = Var("y", TensorType((2,)))
        root = Op.make(
            "add",
            Op.make("mul", x, Const(1)),
            Op.make("mul", y, Const(1)),
        )
        ir = IR(
            root=root,
            inputs=[x, y],
            input_names={"x", "y"},
            params={},
        )
        return ir, {}


def _opt(engine=None):
    sink = _OracleSink()
    return Optimizer(
        source=_OracleSource(), sink=sink, engine=engine
    ), sink


# ---------------------------------------------------------------------------
#  Seam — no binary required
# ---------------------------------------------------------------------------


def test_engine_protocol_conformance() -> None:
    """``EqsatEngine`` satisfies the ``Engine`` port structurally."""
    eng = object.__new__(EqsatEngine)
    assert isinstance(eng, Engine)
    assert eng.engine_name == "eqsat"
    assert eng.truncation_level == 1


def test_resolve_engine_eqsat_string() -> None:
    """``engine="eqsat"`` materialises an EqsatEngine — or declines."""
    try:
        eng = _resolve_engine("eqsat")
    except EqsatUnavailableError:
        pytest.skip("eqsat-run binary not found")
    assert isinstance(eng, EqsatEngine)


def test_resolve_engine_default_is_eqsat() -> None:
    """``engine=None`` resolves to the WP4-v4 default — ``"eqsat"``.

    With no ``eqsat-run`` binary the default declines by name (never
    a silent Python fallback); ``engine="python-egraph"`` is the
    explicit opt-out.
    """
    from mlopt_core.egraph import EGraph

    assert isinstance(_resolve_engine("python-egraph"), EGraph)
    try:
        eng = _resolve_engine(None)
    except EqsatUnavailableError:
        pytest.skip("eqsat-run binary not found")
    assert isinstance(eng, EqsatEngine)
    assert eng.engine_name == "eqsat"


def test_missing_binary_is_a_named_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No eqsat-run anywhere -> EqsatUnavailableError, never fallback."""
    import mlopt_orchestrator.eqsat as eqsat_mod

    def _none(*_a, **_kw):
        raise EqsatUnavailableError("no eqsat-run on this host")

    monkeypatch.setattr(eqsat_mod, "eqsat_binary", _none)
    with pytest.raises(EqsatUnavailableError, match="no eqsat-run"):
        EqsatEngine()
    # And through the search seam: the decline propagates, it does not
    # silently degrade to the Python engine — for the explicit
    # ``engine="eqsat"`` AND for the ``None`` default alike.
    for eng in ("eqsat", None):
        with pytest.raises(EqsatUnavailableError):
            search(
                _simplifying_model(),
                None,
                source=_FakeSource(),
                cost_fn=flops_cost,
                max_iterations=2,
                engine=eng,
            )


def test_env_override_bad_path_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLOPT_EQSAT_RUN", "/nonexistent/eqsat-run")
    with pytest.raises(EqsatUnavailableError, match="MLOPT_EQSAT_RUN"):
        eqsat_binary()


def test_portable_laws_snapshot() -> None:
    """The shipped set is the full non-demo corpus, demo excluded."""
    laws = portable_laws()
    assert len(laws) == 488
    names = {r["name"] for r in laws}
    assert "reshape_flatten_demo" not in names
    assert {"comm_add", "id_mul", "factor_mul_add", "assoc_matmul"} <= (
        names
    )
    # LAW_FORMAT=2 records carry the pat authoring fields.
    assert all(r.get("version") == 2 for r in laws)


def test_record_synops_census() -> None:
    """The 60 ``prim:`` records each name minted ``synop*`` ops."""
    prim = [r for r in portable_laws() if r["name"].startswith("prim:")]
    assert len(prim) == 60
    assert all(_record_synops(r) for r in prim)
    comm_add = next(
        r for r in portable_laws() if r["name"] == "comm_add"
    )
    assert _record_synops(comm_add) == frozenset()


def test_prim_records_withheld_until_synops_seated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unseated ``synop*`` denotations keep ``prim:`` records off the wire.

    Regression for the extraction crash: the minted ``prim:synop*``
    records fired under the ``size`` CostSpec and extraction could
    pick a ``synop_*`` member with no torch binding — ``lower`` then
    died on ``ValueError: No torch binding``.  The bridge now
    withholds any record naming an unseated ``synop*`` op — the same
    census ``supported_ops`` gives the Python engine's
    ``backend_cost`` — and reports it under
    ``stats["eqsat"]["laws"]["unbound"]``.
    """
    import mlopt_core.ops as ops_mod

    eng = EqsatEngine(binary="/bin/true")
    requested = {
        "comm_add",
        "prim:synop_03889d8fe1",
        "prim:synop3_04df5fd446",
    }

    # No backend ambient table -> nothing seated -> every prim record
    # withheld; manual-op records are unaffected.
    monkeypatch.setattr(ops_mod, "_AMBIENT_BINDINGS", None)
    applied, uncovered, unbound = eng._select_laws(requested)
    assert [r["name"] for r in applied] == ["comm_add"]
    assert uncovered == []
    assert unbound == [
        "prim:synop3_04df5fd446",
        "prim:synop_03889d8fe1",
    ]

    # Seating only the depth-1 op frees ITS record — the depth-3
    # fold still names unseated embeds and stays withheld.
    monkeypatch.setattr(
        ops_mod,
        "_AMBIENT_BINDINGS",
        {"synop_03889d8fe1": object()},
    )
    applied, _, unbound = eng._select_laws(requested)
    assert {r["name"] for r in applied} == {
        "comm_add",
        "prim:synop_03889d8fe1",
    }
    assert unbound == ["prim:synop3_04df5fd446"]

    # The full shipped census seats every minted op -> all applied.
    monkeypatch.setattr(
        ops_mod,
        "_AMBIENT_BINDINGS",
        {
            op: object()
            for r in portable_laws()
            for op in _record_synops(r)
        },
    )
    applied, _, unbound = eng._select_laws(requested)
    assert {r["name"] for r in applied} == requested
    assert unbound == []


def test_extract_best_refuses_unbound_synop_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached extraction naming an unseated ``synop*`` falls back.

    The record-level filter keeps unbound synops out of the wire law
    set, but a custom ``laws=`` file (or a census that changed between
    ``run`` and ``extract_best``) can still surface one — the
    extract-side floor returns the recorded input term instead, the
    same never-deliver-an-unlowerable-member bound the cost floor
    gives.
    """
    import mlopt_core.ops as ops_mod

    monkeypatch.setattr(ops_mod, "_AMBIENT_BINDINGS", None)
    x = Var("x", TensorType((2,)))
    manual = Op.make("add", x, x)
    minted = Op.make("synop_03889d8fe1", x, x)
    eng = object.__new__(EqsatEngine)
    eng._terms = [manual]
    eng._results = {0: {"term": minted, "cost": 0.0}}
    eng._n_enodes = eng._n_classes = 0
    eng._rule_fires = {}

    got = eng.extract_best(0, flops_cost)
    assert got is manual
    assert eng._results[0]["fell_back_to_input"] is True
    assert eng._results[0]["unbound_synops"] == ["synop_03889d8fe1"]

    # Once the denotation is seated the same pick is honoured —
    # cost_fn=None skips the pricing floor entirely.
    monkeypatch.setattr(
        ops_mod,
        "_AMBIENT_BINDINGS",
        {"synop_03889d8fe1": object()},
    )
    eng._results = {0: {"term": minted, "cost": 0.0}}
    assert eng.extract_best(0, None) is minted
    assert "fell_back_to_input" not in eng._results[0]


def test_codec_roundtrip() -> None:
    """``_kterm``/``_pyterm`` preserve ops, attrs, and typed leaves."""
    x = Var("x", TensorType((2, 3)))
    w = Param("w", TensorType((3, 4)))
    t = Op.make(
        "transpose",
        Op.make("add", x, x),
        dim0=0,
        dim1=1,
    )
    wire = _kterm(t, [0, 1000])
    back = _pyterm(wire)
    assert back == t
    # Leaf records carry name AND TensorType shape.
    lx = _pyterm(_kterm(x, [0, 1000]))
    assert lx == x and lx.typ.shape == (2, 3)
    lp = _pyterm(_kterm(w, [0, 1000]))
    assert lp == w and lp.typ.shape == (3, 4)


def test_codec_nonfinite_and_attrs() -> None:
    """NaN/inf ride the __nonfinite__ tag; tuple attrs restore."""
    t = Op.make(
        "clamp",
        Const(float("nan")),
        Const(float("inf")),
        lo=(1, 2),
        hi=[3, 4],
    )
    wire = json.dumps(_kterm(t, [0, 1000]))
    assert "__nonfinite__" in wire
    back = _pyterm(json.loads(wire))
    assert back.attrs["lo"] == (1, 2)
    assert back.attrs["hi"] == [3, 4]
    vals = [a.value for a in back.args]
    assert vals[0] != vals[0]  # nan
    assert vals[1] == float("inf")


def test_kdag_pyterm_roundtrip() -> None:
    """``_terms_wire``'s dag.program record decodes via ``_pyterm``.

    The doubling DAG shares every subtree — the record is ~22 nodes
    where a tree spelling is 2**20 — and the decoded term keeps the
    sharing (``Op.make`` interning makes it object sharing again).
    """
    from mlopt_orchestrator.eqsat import _terms_wire

    x = Var("x", TensorType((2,)))
    t = x
    for _ in range(20):
        t = Op.make("add", t, t)
    wire = _terms_wire([t], 100_000)
    rec = wire[0]
    assert rec["Opaque"]["__tag__"] == "dag.program"
    assert len(rec["Opaque"]["nodes"]) < 40
    back = _pyterm(rec)
    assert back == t


def test_pyterm_dag_declines_bad_records() -> None:
    """Malformed dag.program payloads raise EqsatError, not garbage."""
    with pytest.raises(EqsatError, match="nodes"):
        _pyterm({"Opaque": {"__tag__": "dag.program", "root": 0}})
    with pytest.raises(EqsatError, match="root index"):
        _pyterm(
            {
                "Opaque": {
                    "__tag__": "dag.program",
                    "nodes": [{"Const": {"value": 1}}],
                    "root": 5,
                }
            }
        )
    with pytest.raises(EqsatError, match="earlier node index"):
        _pyterm(
            {
                "Opaque": {
                    "__tag__": "dag.program",
                    "nodes": [
                        {"Const": {"value": 1}},
                        {
                            "Op": {
                                "dim": 0,
                                "name": "add",
                                "args": [1],
                                "source": "Native",
                            }
                        },
                    ],
                    "root": 1,
                }
            }
        )
    with pytest.raises(EqsatError, match="Opaque term"):
        _pyterm({"Opaque": {"__tag__": "unrelated", "x": 1}})


def test_wire_node_budget() -> None:
    """The dag.program node-table bound is a named EqsatError.

    The 20-level doubling DAG encodes as ~22 *distinct* nodes (the
    table shares every repeated subtree — the tree spelling would
    have emitted 2**20), so the budget now bounds distinct nodes:
    10 is under the table size and must refuse by name, not hang.
    """
    x = Var("x", TensorType((2,)))
    t = x
    for _ in range(20):  # DAG: each level doubles the tree
        t = Op.make("add", t, t)
    eng = object.__new__(EqsatEngine)
    eng._terms = [t]
    eng._wire_node_budget = 10
    with pytest.raises(EqsatError, match="wire_node_budget"):
        eng._invoke(0)


@requires_eqsat
def test_run_refuses_improving_stop() -> None:
    eng = EqsatEngine()
    eng.add_term(_simplifying_model().root)
    with pytest.raises(ValueError, match="stop"):
        eng.run([], 0, stop="improving", cost_fn=flops_cost)


@requires_eqsat
def test_extract_best_refuses_steering() -> None:
    eng = EqsatEngine()
    eng.add_term(_simplifying_model().root)
    eng.run([], 0, max_iterations=4)
    with pytest.raises(TypeError, match="overrides"):
        eng.extract_best(0, flops_cost, overrides={0: None})
    with pytest.raises(TypeError, match="fusion_epsilon"):
        eng.extract_best(0, flops_cost, fusion_epsilon=0.5)
    # The plain call returns the cached term.
    assert eng.extract_best(0, flops_cost) is not None


@requires_eqsat
def test_engine_bad_binary_report() -> None:
    """A non-eqsat executable surfaces its refusal, not a verdict."""
    eng = EqsatEngine(binary="/bin/true")
    eng.add_term(_simplifying_model().root)
    with pytest.raises(EqsatError):
        eng.run([], 0, max_iterations=2)


def test_binary_explicit_and_env_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``binary=`` and a valid ``MLOPT_EQSAT_RUN`` resolve."""
    import pathlib

    assert eqsat_binary("/bin/true") == pathlib.Path("/bin/true")
    eng = EqsatEngine(binary="/bin/true")
    assert eng._binary == pathlib.Path("/bin/true")
    monkeypatch.setenv("MLOPT_EQSAT_RUN", "/bin/true")
    assert eqsat_binary() == pathlib.Path("/bin/true")
    with pytest.raises(
        EqsatUnavailableError, match="not an executable"
    ):
        eqsat_binary("/nonexistent/eqsat-run")


def test_laws_argument_forms(tmp_path) -> None:
    """``laws=`` accepts the snapshot default, a JSON path, records."""
    eng = EqsatEngine(binary="/bin/true", laws=[])
    assert eng._laws == ()
    f = tmp_path / "laws.json"
    f.write_text(json.dumps([{"name": "x", "version": 2}]))
    eng2 = EqsatEngine(binary="/bin/true", laws=str(f))
    assert [r["name"] for r in eng2._laws] == ["x"]
    eng3 = EqsatEngine(binary="/bin/true", laws=[{"name": "y"}])
    assert [r["name"] for r in eng3._laws] == ["y"]
    with pytest.raises(EqsatError, match="unreadable"):
        EqsatEngine(
            binary="/bin/true", laws=str(tmp_path / "none.json")
        )
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "not-a-list"}))
    with pytest.raises(EqsatError, match="record array"):
        EqsatEngine(binary="/bin/true", laws=str(bad))
    with pytest.raises(EqsatError, match="string 'name'"):
        EqsatEngine(binary="/bin/true", laws=[{"noname": 1}])


def test_cost_spec_forms() -> None:
    """``cost=`` normalises kind strings and CostSpec records."""
    assert EqsatEngine(binary="/bin/true")._cost == {"kind": "size"}
    assert EqsatEngine(binary="/bin/true", cost="count")._cost == {
        "kind": "count"
    }
    spec = {"kind": "op_table", "default": 1.0, "ops": {"add": 2.0}}
    assert EqsatEngine(binary="/bin/true", cost=spec)._cost == spec
    with pytest.raises(EqsatError, match="cost kind"):
        EqsatEngine(binary="/bin/true", cost="flops-v2")
    with pytest.raises(EqsatError, match="CostSpec"):
        EqsatEngine(binary="/bin/true", cost=42)


def test_codec_declines() -> None:
    """Malformed wire records and undecodable terms raise EqsatError."""
    with pytest.raises(EqsatError, match="leaf record"):
        _pyterm({"Const": {"value": "__mlopt_leaf__:{bad json"}})
    with pytest.raises(EqsatError, match="leaf record"):
        _pyterm({"Const": {"value": "__mlopt_leaf__:[1,2]"}})
    with pytest.raises(EqsatError, match="leaf kind"):
        _pyterm(
            {
                "Const": {
                    "value": "__mlopt_leaf__:"
                    + json.dumps({"k": "?", "n": "x", "s": None})
                }
            }
        )
    with pytest.raises(EqsatError, match="undecodable"):
        _pyterm({"Seq": {}})
    with pytest.raises(TypeError, match="wireable"):
        _kterm(object(), [0, 10])
    # A Const string that merely *resembles* a leaf prefix is still a
    # Const — only the exact prefix + valid record decodes as a leaf.
    back = _pyterm({"Const": {"value": "__mlopt_leaf__x"}})
    assert isinstance(back, Const) and back.value == "__mlopt_leaf__x"


def test_extract_best_never_worse_than_input() -> None:
    """WP4 F3: a CostSpec pick priced worse under ``cost_fn`` falls back.

    The engine's ``{"kind": "size"}`` extraction can rank a member the
    caller's cost axis dislikes (the ``select_mul`` regression).  The
    bridge restores the Python extractor's floor — min-cost under the
    handed ``cost_fn``, input term inside the e-class — by pricing
    both sides with ``dag_cost`` and keeping the input when the pick
    is strictly worse.
    """
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    cheap = Op.make("add", x, y)  # the "input" the engine recorded
    # A bigger-but-equivalent-looking pick: size-cost would prefer it
    # only if smaller — here we fake the cached extraction outright.
    pricey = Op.make("mul", Op.make("add", x, y), Const(2))
    eng = object.__new__(EqsatEngine)
    eng._terms = [cheap]
    eng._results = {0: {"term": pricey, "cost": 0.0}}
    eng._n_enodes = eng._n_classes = 0
    eng._rule_fires = {}

    # cost_fn: 1.0 per op node + 0.5 per leaf — pricey has 2 ops.
    def two_ops_worse(t):
        from mlopt_core.ir import Op as _Op

        if isinstance(t, _Op):
            return 1.0 + sum(two_ops_worse(a) for a in t.args)
        return 0.5

    # pricey (add+mul = 2 ops) vs cheap (add = 1 op): strictly worse.
    got = eng.extract_best(0, two_ops_worse)
    assert got is cheap
    assert eng._results[0]["fell_back_to_input"] is True

    # And the reverse: a cheaper pick is kept, no flag.
    eng._results = {0: {"term": cheap, "cost": 0.0}}
    got = eng.extract_best(0, two_ops_worse)
    assert got is cheap
    assert "fell_back_to_input" not in eng._results[0]

    # cost_fn=None skips the floor entirely (no pricing requested).
    eng._results = {0: {"term": pricey, "cost": 0.0}}
    assert eng.extract_best(0, None) is pricey


@requires_eqsat
def test_extract_best_before_run_declines() -> None:
    eng = EqsatEngine()
    eng.add_term(_simplifying_model().root)
    with pytest.raises(EqsatError, match="before run"):
        eng.extract_best(0, flops_cost)
    assert eng.n_enodes == 0 and eng.n_classes == 0


# ---------------------------------------------------------------------------
#  Live — the opt-in search, coverage honesty, verify gate
# ---------------------------------------------------------------------------


@requires_eqsat
def test_search_eqsat_engine_end_to_end() -> None:
    """``engine="eqsat"`` searches, records itself, skips non-local."""
    res = search(
        _simplifying_model(),
        None,
        source=_FakeSource(),
        cost_fn=flops_cost,
        max_iterations=6,
        engine="eqsat",
    )
    assert res.stats["engine"] == "eqsat"
    assert res.stats["nonlocal_passes"] == (
        "skipped (engine is not an EGraph)"
    )
    # id_mul fired on both mul(x,1) sites -> the extracted term is x+y.
    assert res.term == Op.make(
        "add", Var("x", TensorType((2,))), Var("y", TensorType((2,)))
    )
    eq = res.stats["eqsat"]
    assert eq["laws"]["applied"]
    assert isinstance(eq["laws"]["uncovered"], list)
    assert eq["cost"] == {"kind": "size"}
    assert "not portable" in eq["cost_fn"]
    assert res.stats["rule_fires"]
    assert res.stats["n_enodes"] > 0


@requires_eqsat
def test_search_eqsat_coverage_is_honest() -> None:
    """Requested rules with no portable record are named, not dropped."""
    res = search(
        _simplifying_model(),
        None,
        source=_FakeSource(),
        cost_fn=flops_cost,
        rules=[FACTOR_MUL_ADD],
        max_iterations=4,
        engine=EqsatEngine(),
    )
    eq = res.stats["eqsat"]
    # factor_mul_add IS portable — applied; everything else requested
    # (none here) uncovered.  Pin the positive direction:
    assert "factor_mul_add" in eq["laws"]["applied"]
    assert eq["laws"]["uncovered"] == []

    # And the negative: a procedural-hook law name stays uncovered.
    res2 = search(
        _simplifying_model(),
        None,
        source=_FakeSource(),
        cost_fn=flops_cost,
        rules=[FACTOR_MUL_ADD, _rewrite_named("sdpa_fold")],
        max_iterations=2,
        engine=EqsatEngine(),
    )
    unc = res2.stats["eqsat"]["laws"]["uncovered"]
    assert "sdpa_fold" in unc
    assert "factor_mul_add" not in unc


def _rewrite_named(name: str) -> Rewrite:
    """A minimal Rewrite carrying only a name — enough for coverage."""
    return Rewrite(name=name, lhs="A", rhs="A")


@requires_eqsat
def test_search_eqsat_vs_python_same_verified_outcome() -> None:
    """Opt-in vs default: both deliver verify-passing terms."""
    x = ([1.0, 2.0], [3.0, 4.0])
    opt_eq, _ = _opt(engine="eqsat")
    res_eq = opt_eq.search(_Model(), x)
    lr_eq = opt_eq.lower(res_eq, x, verify_oracle=FloatOracle())
    assert lr_eq.verified is not None and lr_eq.verified.passed

    opt_py, _ = _opt(engine="python-egraph")
    res_py = opt_py.search(_Model(), x)
    lr_py = opt_py.lower(res_py, x, verify_oracle=FloatOracle())
    assert lr_py.verified is not None and lr_py.verified.passed
    assert res_eq.stats["engine"] == "eqsat"
    assert res_py.stats["engine"] == "python"


@requires_eqsat
def test_lower_verify_gate_still_enforced() -> None:
    """The substrate verify gate does not care which engine searched."""
    from mlopt_orchestrator import VerifyDeclinedError

    class Liar(FloatOracle):
        def compare(self, ref, opt, *, rtol=1e-4, atol=None):
            from mlopt_core.oracle import OracleReport

            return OracleReport(
                max_abs=1.0, max_rel=1.0, passed=False, rtol=rtol
            )

    opt, _sink = _opt(engine="eqsat")
    x = ([1.0, 2.0], [3.0, 4.0])
    res = opt.search(_Model(), x)
    with pytest.raises(VerifyDeclinedError):
        opt.lower(res, x, verify_oracle=Liar())
