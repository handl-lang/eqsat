"""Plan 0007 — the orchestrator is backend-neutral.

Two complements to :mod:`tests.test_core_torch_free` (the core half):

* **static** — an AST walk over ``packages/mlopt-orchestrator/src``
  rejects any direct ``import``/``from`` of the backend roots
  (``torch``, ``mlopt_torch``, ``kit_cuda``).  The
  ``kit_carriers`` imports the regime/carrier helpers defer to
  call time are *not* forbidden at the statement level — carriers
  are optional rewrite machinery — but the runtime proof below
  blocks them too and the whole package must still import.
* **runtime** — ``mlopt_orchestrator`` is imported in a subprocess with
  ``torch``/``numpy``/``mlopt_torch``/``kit_carriers``/
  ``kit_cuda`` all blocked by a meta-path finder; every
  submodule walks, the lazy carrier fallbacks degrade cleanly, and
  a full ``Optimizer`` run on a *pure-Python* fake backend still
  optimizes end-to-end — search, lower, autotune — without a tensor
  library in ``sys.modules``.

The fake backend exercises the plan-0007 port surface as a whole:
:class:`~mlopt_core.pipeline.Backend` is a plain value of four
ports; :class:`~mlopt_core.ports.Source` /
:class:`~mlopt_core.ports.Sink` (+ ``executors``) /
:class:`~mlopt_core.ports.Composer` /
:class:`~mlopt_core.ports.Meter` drive every phase.
"""

from __future__ import annotations

import ast
import copy
import pathlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import NamedTuple

import pytest
from mlopt_core.ir import IR, Op, Param, TensorType, Var
from mlopt_core.ops import OpTable
from mlopt_core.pipeline import Backend
from mlopt_core.ports import (
    Composer,
    ExecutorSpec,
    Meter,
    Sink,
    Source,
    TimingResult,
)
from mlopt_orchestrator import Compositional, Optimizer
from mlopt_orchestrator.optimize import Autotuned, _lower_extracted

_OPT_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "packages"
    / "mlopt-orchestrator"
    / "src"
)

#: Import roots the orchestrator must never name directly.
_FORBIDDEN = frozenset({"torch", "mlopt_torch", "kit_cuda"})


# ---------------------------------------------------------------------------
#  Static proof — no direct backend import statement in mlopt_orchestrator
# ---------------------------------------------------------------------------


def _imported_roots(tree: ast.AST):
    """Yield ``(root_name, lineno)`` for every absolute import in *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_optimize_source_has_no_backend_imports():
    offenders: list[str] = []
    files = sorted(_OPT_SRC.rglob("*.py"))
    assert files, f"no orchestrator sources under {_OPT_SRC}"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for root, lineno in _imported_roots(tree):
            if root in _FORBIDDEN:
                rel = path.relative_to(_OPT_SRC)
                offenders.append(f"{rel}:{lineno} imports {root}")
    assert offenders == [], (
        "mlopt-orchestrator must import no backend directly; found:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
#  Runtime proof — the package loads and degrades with backends blocked
# ---------------------------------------------------------------------------

_BLOCKED_SCRIPT = r"""
import sys

_BLOCKED = ("torch", "numpy", "mlopt_torch", "kit_carriers",
            "kit_cuda")


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError(f"blocked import: {name}")
        return None


sys.meta_path.insert(0, _Blocker())

import importlib
import pkgutil

import mlopt_orchestrator

# Walk EVERY orchestrator submodule — none may need a blocked package
# at import time.
for _info in pkgutil.walk_packages(
    mlopt_orchestrator.__path__, "mlopt_orchestrator."
):
    importlib.import_module(_info.name)

assert "torch" not in sys.modules
assert "mlopt_torch" not in sys.modules
assert "kit_carriers" not in sys.modules

# The lazy carrier fallbacks degrade to empty structures, never raise.
import mlopt_orchestrator.optimize as O
import mlopt_orchestrator.regime as R

assert O._carrier_plans() == {}
assert O._carrier_lifts(object(), {}, {}, False) == []
assert len(R._xc_rules()) == 0
# Core scan laws compose in; blocked carriers contribute nothing.
from mlopt_core.laws import RuleSet
assert isinstance(R.CARRIER_LAWS, RuleSet)
assert len(R.CARRIER_LAWS) == len(R.core_laws.CARRIER_SEARCH)

print("optimize-imported-torch-free")
"""


def test_optimize_imports_without_backends():
    """Import the whole orchestrator with every backend blocked."""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"backend-free orchestrator import failed:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "optimize-imported-torch-free" in proc.stdout


# ---------------------------------------------------------------------------
#  A pure-Python backend — the ports as plain values
# ---------------------------------------------------------------------------


class _ListModel:
    """A pure-Python 'model': ``x -> x + x`` on a list of floats."""

    def __init__(self, inner=None):
        self.inner = inner

    def __call__(self, x):
        return [a + a for a in x]

    forward = __call__


class FakeModule:
    """A lowered IR term evaluated over Python lists."""

    def __init__(self, ir, params):
        self._root = ir.root
        self._inputs = list(ir.inputs)
        self._param_map = dict(params)

    def forward(self, *xs):
        env = {
            v.name: x for v, x in zip(self._inputs, xs, strict=False)
        }
        return self._eval(self._root, env)

    __call__ = forward

    def _eval(self, t, env):
        if isinstance(t, Var):
            return env[t.name]
        if isinstance(t, Param):
            return self._param_map[t.name]
        args = [self._eval(a, env) for a in t.args]
        if t.op == "add":
            return [a + b for a, b in zip(*args, strict=False)]
        raise KeyError(t.op)


class _VerifyOut(NamedTuple):
    max_abs: float
    max_rel: float
    passed: bool


class FakeSink:
    """``Sink`` whose runtime is plain Python."""

    supported_ops = frozenset({"add"})

    @property
    def ops(self):
        return OpTable.core()

    @property
    def executors(self):
        return MappingProxyType(
            {
                "generic": ExecutorSpec(
                    name="generic",
                    accepts=lambda ir, term: True,
                    lower=self.lower,
                    engaged=lambda mod: True,
                ),
                "twin": ExecutorSpec(
                    name="twin",
                    accepts=lambda ir, term: False,
                    lower=self.lower,
                    engaged=lambda mod: True,
                ),
            }
        )

    def lower(self, ir, params=None):
        return FakeModule(ir, params or {})

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        a = list(ref(*args))
        b = list(opt(*args))
        diffs = [abs(u - v) for u, v in zip(a, b, strict=False)]
        max_abs = max(diffs, default=0.0)
        max_rel = max_abs / (max(abs(x) for x in a) + 1e-8)
        return _VerifyOut(max_abs, max_rel, max_rel < rtol)


class FakeComposer:
    """The structural port — a plain-Python compositional surface."""

    def blocks(self, model, predicate=None):
        out = []
        if getattr(model, "inner", None) is not None:
            out.append(("inner", model.inner))
        return out

    def capture_inputs(self, model, blocks, example_input):
        captured = {n: ((example_input,), {}) for n, _ in blocks}
        io = {n: {"calls": 1} for n, _ in blocks}
        return captured, io

    def perturbed(self, example_input):
        return [v + 1.0 for v in example_input]

    def clone_sharing(self, model):
        return copy.deepcopy(model)

    def graft(self, model, replacements):
        for name, mod in replacements.items():
            setattr(model, name, mod)
        return model

    def boundary(self, *_a, **_k):
        return None

    # Deliberately NO ``cross_pairs`` / ``param_report`` — the
    # optional composer hooks must degrade, not require.


class FakeMeter:
    """``Meter`` — a fixed deterministic measurement."""

    def time(
        self, runnable, inputs, *, warmup=5, n_calls=30, timeout_s=None
    ):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        runnable(*args)
        return TimingResult(median_s=0.001, iqr_s=0.0, n_calls=n_calls)


class FakeSource:
    """``Source`` — emits a one-op IR whatever the model is."""

    def to_ir(self, model, example_input):
        x = Var("x", TensorType((4,)))
        w = Param("w", TensorType((4,)))
        root = Op.make("add", x, x)
        ir = IR(
            root=root,
            inputs=[x],
            input_names={"x"},
            params={"w": w},
        )
        # One non-tensor leaf — ``_param_map`` material for the
        # autotune re-lowering pass (exercises the copy.copy arm).
        return ir, {"w": 1.5}


def _fake_backend() -> Backend:
    return Backend(
        source=FakeSource(),
        sink=FakeSink(),
        composer=FakeComposer(),
        meter=FakeMeter(),
    )


# ---------------------------------------------------------------------------
#  The Backend value
# ---------------------------------------------------------------------------


def test_backend_is_a_frozen_port_tuple():
    b = _fake_backend()
    assert isinstance(b.source, Source)
    assert isinstance(b.sink, Sink)
    assert isinstance(b.composer, Composer)
    assert isinstance(b.meter, Meter)
    assert "generic" in b.sink.executors
    with pytest.raises(FrozenInstanceError):
        b.sink = FakeSink()  # type: ignore[misc]


def test_optimizer_requires_source_and_sink():
    with pytest.raises(TypeError, match="source/sink"):
        Optimizer()
    with pytest.raises(TypeError, match="source/sink"):
        Optimizer(source=FakeSource())


def test_optimizer_backend_vs_explicit_ports():
    b = _fake_backend()
    # Explicit ports override the backend's — the backend supplies
    # only what is missing.
    s2, k2, c2, m2 = (
        FakeSource(),
        FakeSink(),
        FakeComposer(),
        FakeMeter(),
    )
    opt = Optimizer(
        backend=b, source=s2, sink=k2, composer=c2, meter=m2
    )
    assert opt.source is s2 and opt.sink is k2
    assert opt.composer is c2 and opt.meter is m2
    # … and the backend still fills whatever is not given.
    opt_b = Optimizer(backend=b)
    assert opt_b.composer is b.composer and opt_b.meter is b.meter
    # A bare explicit-port optimizer still works for Monolithic.
    opt2 = Optimizer(source=FakeSource(), sink=FakeSink())
    assert opt2.composer is None and opt2.meter is None


# ---------------------------------------------------------------------------
#  Fake backend end-to-end — no torch anywhere in the run
# ---------------------------------------------------------------------------


def test_fake_backend_monolithic():
    model = _ListModel()
    x = [1.0, 2.0, 3.0, 4.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    lr = opt.optimize(model, x, max_iterations=2)
    assert lr.module(x) == model(x)
    assert lr.stats["runner"] == "identity"
    assert lr.stats["lowering"] == "generic"
    assert lr.verified is not None and lr.verified.passed


def test_fake_backend_phase_verbs_and_verify():
    model = _ListModel()
    x = [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    res = opt.search(model, x, max_iterations=2)
    lr = opt.lower(res, x, verify=True)
    assert lr.verified is not None and lr.verified.passed
    lr2 = opt.lower(res, x, verify=False)
    assert lr2.verified is None


# ---------------------------------------------------------------------------
#  The verify gate is substrate-enforced — a failed/missing report declines
# ---------------------------------------------------------------------------


class _FailingSink(FakeSink):
    """A sink whose equivalence gate always fails."""

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        return _VerifyOut(1.0, 1.0, False)


def _failing_backend() -> Backend:
    b = _fake_backend()
    return Backend(
        source=b.source,
        sink=_FailingSink(),
        composer=b.composer,
        meter=b.meter,
    )


def test_failed_verify_declines_lower_and_optimize():
    """``lower(verify=True)`` refuses a failed operative verdict."""
    from mlopt_orchestrator import VerifyDeclinedError

    model, x = _ListModel(), [1.0, 2.0]
    opt = Optimizer(backend=_failing_backend(), engine="python-egraph")
    res = opt.search(model, x, max_iterations=2)
    with pytest.raises(
        VerifyDeclinedError, match="delivery declined"
    ) as ei:
        opt.lower(res, x)
    declined = ei.value.result
    # The declined LowerResult stays inspectable on the exception.
    assert declined.verified is not None
    assert not declined.verified.passed
    assert declined.stats["accepted_by"] == "declined"
    assert declined.module is not None
    # The same gate guards the composed verb.
    with pytest.raises(VerifyDeclinedError):
        opt.optimize(model, x, max_iterations=2)
    # The explicit opt-out ships unverified — verified stays None.
    lr = opt.lower(res, x, verify=False)
    assert lr.verified is None and lr.module is not None


def test_missing_verify_report_declines():
    """A ``verify=True`` run that yields no report is a decline."""
    import mlopt_orchestrator.optimize as O
    from mlopt_orchestrator import VerifyDeclinedError

    model, x = _ListModel(), [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    res = opt.search(model, x, max_iterations=2)

    def no_report(*a, **kw):
        return None

    orig = O._gate_delivery
    O._gate_delivery = no_report
    try:
        with pytest.raises(
            VerifyDeclinedError, match="no verify report"
        ):
            opt.lower(res, x)
    finally:
        O._gate_delivery = orig


class _AlwaysFar:
    """A task metric whose distance always exceeds tolerance."""

    name = "always_far"
    tolerance = 0.5

    def distance(self, ref, opt):
        return 1.0


def test_task_gate_verdict_is_the_operative_one():
    """Under ``task=`` the task distance — not the pointwise report —
    decides ship/decline."""
    from mlopt_orchestrator import VerifyDeclinedError

    model, x = _ListModel(), [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    res = opt.search(model, x, max_iterations=2)
    # Pointwise verify passes; the task contract fails → decline.
    with pytest.raises(VerifyDeclinedError, match="task 'always_far'"):
        opt.lower(res, x, task=_AlwaysFar())

    # A passing task metric ships even though it replaced the verdict.
    class _AlwaysNear(_AlwaysFar):
        name = "always_near"

        def distance(self, ref, opt):
            return 0.0

    lr = opt.lower(res, x, task=_AlwaysNear())
    assert lr.verified is not None and lr.verified.passed
    assert lr.stats["accepted_by"] == "task"


def test_fake_backend_autotuned():
    model = _ListModel()
    x = [1.0, 2.0, 3.0, 4.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    lr = opt.optimize(
        model,
        x,
        strategy=Autotuned(
            candidates=("generic", "eager", "twin", "bogus"),
            n_calls=2,
            warmup=0,
        ),
        max_iterations=2,
    )
    at = lr.stats["autotune"]
    assert at["winner"] in {"generic", "eager", "twin"}
    assert at["candidates"]["bogus"]["status"] == "unknown"
    timed = [
        n
        for n, r in at["candidates"].items()
        if r.get("status") == "timed"
    ]
    assert len(timed) == 3


def test_autotuned_requires_meter():
    opt = Optimizer(
        source=FakeSource(), sink=FakeSink(), engine="python-egraph"
    )
    with pytest.raises(TypeError, match="Meter"):
        opt.optimize(
            _ListModel(),
            [1.0],
            strategy=Autotuned(n_calls=1, warmup=0),
        )


def test_fake_backend_compositional():
    model = _ListModel(inner=_ListModel())
    x = [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    lr = opt.optimize(
        model,
        x,
        strategy=Compositional(max_cross_pairs=0),
        max_iterations=2,
        verbose=False,
    )
    stats = lr.stats
    assert stats["n_blocks"] == 1
    assert stats["blocks"]["inner"]["status"] == "optimized"
    # The composer without a param_report hook yields no audit.
    assert stats["param_report"]["original_params"] == 0


def test_compositional_verify_decline_labels_block(capsys):
    """F10: a verify-gate refusal is ``declined``, not ``failed`` —
    under BOTH raise sites: ``lower(verify=True)`` under verbose, and
    the per-block gate's ``_block_decline`` when verify is off."""
    model = _ListModel(inner=_ListModel())
    x = [1.0, 2.0]
    for verbose in (False, True):
        opt = Optimizer(
            backend=_failing_backend(), engine="python-egraph"
        )
        lr = opt.optimize(
            model,
            x,
            strategy=Compositional(max_cross_pairs=0),
            max_iterations=2,
            verbose=verbose,
        )
        rep = lr.stats["blocks"]["inner"]
        assert rep["status"] == "declined", (verbose, rep)
        assert rep["error"].startswith("VerifyDeclinedError")
        assert lr.stats["n_declined"] == 1
        assert lr.stats["n_failed"] == 0
        capsys.readouterr()


def test_compositional_requires_composer():
    opt = Optimizer(
        source=FakeSource(), sink=FakeSink(), engine="python-egraph"
    )
    with pytest.raises(TypeError, match="Composer"):
        opt.optimize(
            _ListModel(inner=_ListModel()),
            [1.0],
            strategy=Compositional(max_cross_pairs=0),
        )


def test_lower_extracted_routes_to_sink_executor():
    """A carrier spec's ``accepts`` probe wins routing; else generic."""
    sink = FakeSink()
    x = Var("x", TensorType((2,)))
    term = Op.make("add", x, x)
    ir = IR(root=term, inputs=[x], input_names={"x"}, params={})
    routed = _lower_extracted(term, ir, {}, sink)
    assert isinstance(routed, FakeModule)


# ---------------------------------------------------------------------------
#  corpus_probe — the injected per-model coverage hook (backend-neutral)
# ---------------------------------------------------------------------------


def test_corpus_probe_hook_runs_and_records():
    """``corpus_probe=`` is invoked once as ``(name, model, x)`` and its
    report lands in ``stats["corpus_probe"]`` — through ``optimize``."""
    calls = []

    def probe(name, model, x):
        calls.append((name, model, x))
        return {"coverage": {"n_novel_op_tuples": 3}}

    model = _ListModel()
    x = [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    lr = opt.optimize(model, x, corpus_probe=probe, max_iterations=2)
    assert calls == [("_ListModel", model, x)]
    assert lr.stats["corpus_probe"] == {
        "coverage": {"n_novel_op_tuples": 3}
    }


def test_corpus_probe_default_is_inert():
    """No probe passed → nothing runs, nothing recorded (both verbs)."""
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    lr = opt.optimize(_ListModel(), [1.0], max_iterations=2)
    assert "corpus_probe" not in lr.stats
    res = opt.search(_ListModel(), [1.0], max_iterations=2)
    assert "corpus_probe" not in res.stats


def test_corpus_probe_compositional_runs_once_on_whole_model():
    """The Compositional strategy scopes the probe to the whole model —
    once, never per block."""
    calls = []

    def probe(name, model, x):
        calls.append((name, model, x))
        return {"ok": True}

    model = _ListModel(inner=_ListModel())
    x = [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend(), engine="python-egraph")
    lr = opt.optimize(
        model,
        x,
        strategy=Compositional(max_cross_pairs=0),
        corpus_probe=probe,
        max_iterations=2,
        verbose=False,
    )
    assert calls == [("_ListModel", model, x)]
    assert lr.stats["corpus_probe"] == {"ok": True}


# ---------------------------------------------------------------------------
#  No-compat state — moved names do NOT resolve through the orchestrator
# ---------------------------------------------------------------------------


def test_optimize_module_has_no_compat_attrs():
    """Plan 0008: the torch-facing names moved to their real homes."""
    import mlopt_orchestrator.optimize as O

    for gone in (
        "optimize_model",
        "optimize_compositional",
        "optimize_model_autotuned",
        "param_report",
        "_select_blocks",
        "save_optimized_weights",
        "torch",
        "_CARRIER_PLANS",
    ):
        assert not hasattr(O, gone), gone
    with pytest.raises(AttributeError):
        _ = O.bogus_name


def test_orchestrator_package_surface_is_neutral():
    import mlopt_orchestrator as pkg

    assert pkg.Optimizer is not None
    assert pkg.Autotuned is not None
    for gone in (
        "optimize_model",
        "optimize_compositional",
        "optimize_model_autotuned",
        "TorchCompileRunner",
        "CudaGraphRunner",
        "export_optimized",
    ):
        with pytest.raises(AttributeError):
            getattr(pkg, gone)


def test_runners_module_has_no_backend_names():
    import mlopt_orchestrator.runners as R

    assert R.IdentityRunner is not None
    for gone in ("TorchCompileRunner", "CudaGraphRunner"):
        with pytest.raises(AttributeError):
            getattr(R, gone)


def test_regime_module_lazy_laws_only():
    import mlopt_orchestrator.regime as R

    # CARRIER_LAWS / XC_LAWS materialise once, then cache.
    laws = R.CARRIER_LAWS
    assert R.CARRIER_LAWS is laws
    xc = R.XC_LAWS
    assert R.XC_LAWS is xc
    with pytest.raises(AttributeError):
        _ = R.RegimeDispatch
    with pytest.raises(AttributeError):
        _ = R.bogus_name


def test_shim_modules_are_gone():
    """The calibrate/export redirect shims were deleted (plan 0008)."""
    import importlib

    for mod in (
        "mlopt_orchestrator.calibrate",
        "mlopt_orchestrator.export",
    ):
        sys.modules.pop(mod, None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


# ---------------------------------------------------------------------------
#  Regime / adapter error + fallback arms
# ---------------------------------------------------------------------------


def test_register_regime_backend_partial_args(monkeypatch):
    import mlopt_orchestrator.regime as R

    sentinel = object()
    monkeypatch.setattr(R, "_DISPATCH_CLS", None)
    # dispatch-only registration: executors untouched, cls recorded.
    R.register_regime_backend(dispatch=sentinel)
    assert R._DISPATCH_CLS is sentinel


def test_frontier_build_without_dispatch(monkeypatch):
    import mlopt_orchestrator.regime as R

    monkeypatch.setattr(R, "_DISPATCH_CLS", None)
    frontier = R.RegimeFrontier(
        eg=None,
        root_eid=0,
        src_term=None,
        ir=None,
        regimes=[],
        choices={},
        executors={},
    )
    with pytest.raises(RuntimeError, match="dispatch backend"):
        frontier.build()


def test_runner_candidate_requires_ir():
    from mlopt_orchestrator.autotune import (
        AutotuneContext,
        CandidateUnavailableError,
    )
    from mlopt_orchestrator.runners import (
        IdentityRunner,
        runner_candidate,
    )

    ctx = AutotuneContext(
        model=None,
        example_input=None,
        ir=None,
        term=None,
        param_values={},
        sink=None,
        delivered=None,
        lowering="generic",
    )
    with pytest.raises(CandidateUnavailableError):
        runner_candidate(IdentityRunner())(ctx)


def test_carrier_helpers_degrade_without_carriers(monkeypatch):
    """The lazy carrier fallbacks return empty in-process too."""
    import sys as _sys

    import mlopt_orchestrator.optimize as O
    import mlopt_orchestrator.regime as R
    from mlopt_core.egraph import EGraph

    for name in (
        "kit_carriers.om_lower",
        "kit_carriers.omd_lower",
        "kit_carriers.scan_lower",
        "kit_carriers.trace_lift",
        "kit_carriers.xcarrier",
    ):
        monkeypatch.setitem(_sys.modules, name, None)
    assert O._carrier_plans() == {}
    # A real (empty) e-graph — the share/tying passes tolerate it.
    assert isinstance(O._carrier_lifts(EGraph(), {}, {}, False), list)
    assert len(R._xc_rules()) == 0
    # build_egraph with blocked carriers runs the plain core tier.
    eg, _root, ir, _src, _st = R.build_egraph(
        _ListModel(), [1.0, 2.0], source=FakeSource(), rules=[], xc=True
    )
    assert eg is not None and ir is not None


def test_carrier_registry_absent_paths(monkeypatch):
    """The ``get_carriers() is None`` arms, exercised in-process.

    ``kit_carriers`` installs its machinery at import, so the
    carrier-free arms are otherwise reachable only in a backend-free
    process (the subprocess proof above).  Neutralising the registry
    directly drives the same arms here — real coverage of the
    fallbacks, not a spawned interpreter the tracer never sees.
    """
    import mlopt_orchestrator.carriers as C
    import mlopt_orchestrator.optimize as O
    import mlopt_orchestrator.regime as R

    monkeypatch.setattr(C, "_MACHINERY", None)

    # ``_carrier_search`` composes the core preset alone.
    assert len(R._carrier_search()) == len(R.core_laws.CARRIER_SEARCH)
    # ``_xc_rules`` degrades to an empty seam set.
    assert len(R._xc_rules()) == 0
    # ``build_egraph`` runs the plain core tier: no lifts, no XC.
    _eg, _root, _ir, _src, stats = R.build_egraph(
        _ListModel(), [1.0, 2.0], source=FakeSource(), rules=[], xc=True
    )
    assert "nonlocal_lifts" not in stats
    assert "xc_rounds" not in stats
    # The pipeline default composes without the carrier preset.  The
    # composed set is cached, so reset it (monkeypatch restores it).
    monkeypatch.setattr(O, "_DEFAULT_RULES", None)
    assert len(O.default_rules()) == len(R.core_laws.DEFAULT)


def test_register_regime_backend_executors_only():
    import mlopt_orchestrator.regime as R

    R.register_regime_backend(executors={"_probe": object()})
    assert "_probe" in R.EXECUTORS
    del R.EXECUTORS["_probe"]


def test_lower_extracted_empty_executor_table():
    """A sink whose table is empty routes through ``sink.lower``."""

    class BareSink(FakeSink):
        @property
        def executors(self):
            return {}

    x = Var("x", TensorType((2,)))
    term = Op.make("add", x, x)
    ir = IR(root=term, inputs=[x], input_names={"x"}, params={})
    mod = _lower_extracted(term, ir, {}, BareSink())
    assert isinstance(mod, FakeModule)


# ---------------------------------------------------------------------------
#  The real Backend — TorchBackend is a plain value
# ---------------------------------------------------------------------------
