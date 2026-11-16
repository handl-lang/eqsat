"""Plan 0037 phase 1 — the backend registry.

``mlopt_core.backends`` records *which model backends exist* — the
declaration half of the ports layer.  These tests pin:

* the registration surface (``BackendRegistration`` value +
  ``BackendRegistry`` table + the ``DEFAULT_REGISTRY`` module
  functions),
* the lazy-factory contract — a registration stores callables and
  resolves them only when a consumer asks for the port, so seating
  a backend never imports its runtime,
* the duplicate-name policy (refuse by default, ``replace=True``
  to supersede),
* the capability filter (``find_by_ops`` over ``supported_ops``),
* the honest partial-backend path — a source+sink-only
  registration (the ONNX shape) drives ``Optimizer`` port-by-port
  while ``build_backend()`` names what it cannot assemble.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from types import MappingProxyType
from typing import NamedTuple

import pytest
from mlopt_core.backends import (
    DEFAULT_REGISTRY,
    BackendPortsMissing,
    BackendRegistration,
    BackendRegistry,
    backend_names,
    find_backends_by_ops,
    get_backend,
    register_backend,
)
from mlopt_core.ir import IR, Op, Param, TensorType, Var
from mlopt_core.ops import OpTable
from mlopt_core.pipeline import Backend
from mlopt_core.ports import (
    Composer,
    Meter,
    Sink,
    Source,
    TimingResult,
)

# ---------------------------------------------------------------------------
#  Minimal fakes — the port surface as plain Python
# ---------------------------------------------------------------------------


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
    """``Sink`` over plain Python — ops: ``{"add"}``."""

    supported_ops = frozenset({"add"})

    @property
    def ops(self):
        return OpTable.core()

    @property
    def executors(self):
        return MappingProxyType({})

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


class WideSink(FakeSink):
    """A second sink with a wider op surface."""

    supported_ops = frozenset({"add", "mul", "sub"})


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
        return ir, {"w": 1.5}


class FakeComposer:
    """The structural port — minimal presence conformance."""

    def blocks(self, model, *, predicate=None):
        return []

    def capture_inputs(self, model, blocks, example_input):
        return {}, {}

    def graft(self, model, replacements):
        return model

    def clone_sharing(self, model):
        return model

    def boundary(self, *_a, **_k):
        return None

    def perturbed(self, example_input):
        return example_input


class FakeMeter:
    """``Meter`` — a fixed deterministic measurement."""

    def time(
        self, runnable, inputs, *, warmup=5, n_calls=30, timeout_s=None
    ):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        runnable(*args)
        return TimingResult(median_s=0.001, iqr_s=0.0, n_calls=n_calls)


def _registration(name="fake", **kw) -> BackendRegistration:
    args = {
        "name": name,
        "source": FakeSource,
        "targets": {"fake": FakeSink},
    }
    args.update(kw)
    return BackendRegistration(**args)


# ---------------------------------------------------------------------------
#  Registration surface
# ---------------------------------------------------------------------------


def test_register_get_names_contains():
    reg = BackendRegistry()
    r = reg.register(_registration())
    assert r.name == "fake"
    assert reg.get("fake") is r
    assert reg["fake"] is r
    assert "fake" in reg
    assert reg.names() == ("fake",)
    assert reg.items() == (("fake", r),)
    assert list(reg) == [r]
    assert len(reg) == 1


def test_unregistered_name_raises_keyerror():
    reg = BackendRegistry()
    with pytest.raises(KeyError, match="no backend"):
        reg.get("nope")
    with pytest.raises(KeyError):
        reg["nope"]
    with pytest.raises(KeyError):
        reg.unregister("nope")
    assert "nope" not in reg


def test_unregister_removes_and_returns():
    reg = BackendRegistry()
    r = reg.register(_registration())
    assert reg.unregister("fake") is r
    assert len(reg) == 0
    assert "fake" not in reg


def test_clear_empties():
    reg = BackendRegistry()
    reg.register(_registration())
    reg.register(_registration("other"))
    reg.clear()
    assert reg.names() == ()
    assert len(reg) == 0


def test_register_rejects_non_registration():
    reg = BackendRegistry()
    with pytest.raises(TypeError, match="BackendRegistration"):
        reg.register(object())  # type: ignore[arg-type]


def test_registration_order_is_preserved():
    reg = BackendRegistry()
    reg.register(_registration("b"))
    reg.register(_registration("a"))
    assert reg.names() == ("b", "a")


# ---------------------------------------------------------------------------
#  Duplicate policy
# ---------------------------------------------------------------------------


def test_duplicate_name_refused_by_default():
    reg = BackendRegistry()
    reg.register(_registration())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_registration())
    # The first registration survives untouched.
    assert reg.get("fake").source is FakeSource


def test_replace_true_supersedes():
    reg = BackendRegistry()
    first = reg.register(_registration())
    second = reg.register(
        _registration(source=FakeSource), replace=True
    )
    assert reg.get("fake") is second
    assert reg.get("fake") is not first
    assert len(reg) == 1


def test_register_backend_module_function_uses_default_registry():
    try:
        before = set(backend_names())
        r = register_backend(
            "zz_test_backend",
            source=FakeSource,
            targets={"fake": FakeSink},
        )
        assert r.name == "zz_test_backend"
        assert "zz_test_backend" in DEFAULT_REGISTRY
        assert "zz_test_backend" in backend_names()
        assert get_backend("zz_test_backend") is r
    finally:
        DEFAULT_REGISTRY.unregister("zz_test_backend")
    assert set(backend_names()) == before


def test_register_backend_honours_explicit_registry_and_replace():
    reg = BackendRegistry()
    register_backend(
        "x", source=FakeSource, targets={"t": FakeSink}, registry=reg
    )
    assert DEFAULT_REGISTRY.names() == () or "x" not in DEFAULT_REGISTRY
    with pytest.raises(ValueError):
        register_backend(
            "x",
            source=FakeSource,
            targets={"t": FakeSink},
            registry=reg,
        )
    register_backend(
        "x",
        source=FakeSource,
        targets={"t": WideSink},
        registry=reg,
        replace=True,
    )
    assert isinstance(reg.get("x").sink("t"), WideSink)


# ---------------------------------------------------------------------------
#  Validation at declaration time — no factories resolved
# ---------------------------------------------------------------------------


def test_registration_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        _registration(name="")


def test_registration_rejects_no_targets():
    with pytest.raises(ValueError, match="no sink targets"):
        _registration(targets={})


def test_registration_rejects_bad_target_value():
    with pytest.raises(TypeError, match="Sink or a zero-arg"):
        _registration(targets={"bad": 42})


def test_registration_rejects_bad_source():
    with pytest.raises(TypeError, match="Source or a zero-arg"):
        _registration(source=42)


def test_registration_rejects_bad_composer_meter():
    with pytest.raises(TypeError, match="composer"):
        _registration(composer=42)
    with pytest.raises(TypeError, match="meter"):
        _registration(meter=42)


# ---------------------------------------------------------------------------
#  Lazy resolution — factories are stored, not called
# ---------------------------------------------------------------------------


def test_factories_are_not_called_at_registration():
    calls = []

    def source_factory():
        calls.append("source")
        return FakeSource()

    def sink_factory():
        calls.append("sink")
        return FakeSink()

    reg = BackendRegistry()
    reg.register(
        BackendRegistration(
            name="lazy",
            source=source_factory,
            targets={"t": sink_factory},
        )
    )
    assert calls == []
    # Lookup alone must not resolve either.
    assert reg.get("lazy").name == "lazy"
    assert calls == []
    # Resolution happens on demand, per port.
    assert isinstance(reg.get("lazy").source_instance(), FakeSource)
    assert calls == ["source"]
    assert isinstance(reg.get("lazy").sink("t"), FakeSink)
    assert calls == ["source", "sink"]


def test_class_objects_resolve_to_instances():
    """``source=FakeSource`` means ``FakeSource()``, not the class."""
    reg = BackendRegistry()
    r = reg.register(_registration())
    assert isinstance(r.source_instance(), FakeSource)
    assert not inspect.isclass(r.source_instance())
    assert isinstance(r.sink(), FakeSink)


def test_sink_requires_target_name_when_ambiguous():
    r = _registration(targets={"a": FakeSink, "b": WideSink})
    with pytest.raises(KeyError, match="pass a target name"):
        r.sink()
    assert isinstance(r.sink("a"), FakeSink)
    assert isinstance(r.sink("b"), WideSink)
    with pytest.raises(KeyError, match="no target"):
        r.sink("zzz")


def test_supported_ops_unions_targets():
    r = _registration(targets={"a": FakeSink, "b": WideSink})
    assert r.supported_ops() == frozenset({"add", "mul", "sub"})
    assert r.supported_ops("a") == frozenset({"add"})


def test_bad_factory_result_raises_typeerror():
    r = _registration(source=lambda: 42)
    with pytest.raises(TypeError, match="not a Source"):
        r.source_instance()


# ---------------------------------------------------------------------------
#  Capability filter
# ---------------------------------------------------------------------------


def test_find_by_ops_filters_on_supported_ops():
    reg = BackendRegistry()
    reg.register(_registration("narrow", targets={"t": FakeSink}))
    reg.register(_registration("wide", targets={"t": WideSink}))
    # {"add"} is covered by both; {"mul"} only by wide.
    both = reg.find_by_ops({"add"})
    assert {r.name for r in both} == {"narrow", "wide"}
    wide_only = reg.find_by_ops({"mul"})
    assert [r.name for r in wide_only] == ["wide"]
    # Nothing covers an op nobody declares.
    assert reg.find_by_ops({"sdpa"}) == []


def test_find_by_ops_target_narrows():
    reg = BackendRegistry()
    reg.register(
        BackendRegistration(
            name="multi",
            source=FakeSource,
            targets={"narrow": FakeSink, "wide": WideSink},
        )
    )
    hits = reg.find_by_ops({"mul"}, target="wide")
    assert [r.name for r in hits] == ["multi"]
    assert reg.find_by_ops({"mul"}, target="narrow") == []
    # An undeclared target name matches nothing.
    assert reg.find_by_ops({"add"}, target="zzz") == []


def test_find_backends_by_ops_module_function():
    reg = BackendRegistry()
    register_backend(
        "zz_cap",
        source=FakeSource,
        targets={"t": WideSink},
        registry=reg,
    )
    assert [r.name for r in reg.find_by_ops({"mul"})] == ["zz_cap"]
    assert find_backends_by_ops({"mul"}) == []  # default registry empty


# ---------------------------------------------------------------------------
#  build_backend — the bundled value, honestly partial
# ---------------------------------------------------------------------------


def test_build_backend_assembles_full_value():
    r = _registration(composer=FakeComposer, meter=FakeMeter)
    b = r.build_backend()
    assert isinstance(b, Backend)
    assert isinstance(b.source, Source)
    assert isinstance(b.sink, Sink)
    assert isinstance(b.composer, Composer)
    assert isinstance(b.meter, Meter)
    assert isinstance(b.sink, FakeSink)


def test_build_backend_names_missing_ports():
    """The ONNX shape: source+sink only — build_backend says what
    it cannot assemble rather than inventing composer/meter."""
    r = _registration()
    with pytest.raises(BackendPortsMissing, match="composer, meter"):
        r.build_backend()
    r2 = _registration(composer=FakeComposer)
    with pytest.raises(BackendPortsMissing, match="meter"):
        r2.build_backend()


def test_build_backend_passes_device_policy():
    sentinel = object()
    r = _registration(
        composer=FakeComposer,
        meter=FakeMeter,
        device_policy=sentinel,
    )
    assert r.build_backend().device_policy is sentinel


# ---------------------------------------------------------------------------
#  End-to-end honesty — a registered source+sink drives Optimizer
# ---------------------------------------------------------------------------


def test_registered_source_sink_drive_optimizer():
    """The registry's output is usable port-by-port: ``Optimizer``
    takes ``source_instance()``/``sink()`` directly — no composer or
    meter needed for the monolithic path."""
    from mlopt_orchestrator import Optimizer

    reg = BackendRegistry()
    reg.register(_registration())
    r = reg.get("fake")
    opt = Optimizer(
        source=r.source_instance(),
        sink=r.sink(),
        engine="python-egraph",
    )
    result = opt.optimize(
        object(), [1.0, 2.0, 3.0, 4.0], max_iterations=2
    )
    assert result.verified is not None
    assert result.verified.passed


# ---------------------------------------------------------------------------
#  Lazy-import proof — the module seats nothing and imports no backend
# ---------------------------------------------------------------------------

_LAZY_SCRIPT = r"""
import sys
import mlopt_core.backends as B

assert B.backend_names() == ()
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
assert "kit_torch" not in sys.modules
assert "kit_onnx" not in sys.modules
print("registry-import-clean")
"""


def test_backends_module_imports_with_no_backend_loaded():
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"registry import pulled in a backend:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "registry-import-clean" in proc.stdout
