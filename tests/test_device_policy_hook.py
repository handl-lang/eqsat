"""Phase B — the backend-neutral ``device_policy`` hook.

``search``/``lower``/``Optimizer`` accept a
:class:`~mlopt_core.ports.DevicePolicy`-shaped policy: ``search``
calls ``resolve(model, x)`` once before export and records the
answer under ``stats["device"]``; ``lower`` runs the delivered
module through ``check_module`` (when provided) and records
``stats["delivery_device"]``.  Refusals propagate — never degrade.
The substrate stores no device type, so a fake policy returning a
plain string exercises the whole seam without torch.
"""

from __future__ import annotations

import pytest
from mlopt_core.ports import DevicePolicy
from mlopt_orchestrator import Optimizer
from mlopt_orchestrator.optimize import discover_alternatives

from tests.test_backend_neutral import (
    FakeComposer,
    FakeMeter,
    FakeSink,
    FakeSource,
    _fake_backend,
    _ListModel,
)


class RecordingPolicy:
    """A DevicePolicy-shaped fake — resolves to a string, records calls."""

    def __init__(self, device: str = "fake:0") -> None:
        self.device = device
        self.resolved: list[tuple] = []
        self.checked: list = []

    def resolve(self, module=None, inputs=None):
        self.resolved.append((module, inputs))
        return self.device

    def check_module(self, module):
        self.checked.append(module)


class ResolveOnlyPolicy:
    """No ``check_module`` — the duck-typed delivery hook must skip it."""

    def resolve(self, module=None, inputs=None):
        return "resolve-only"


class RefusingPolicy:
    """Raises on resolve — the refusal must propagate, not degrade."""

    def resolve(self, module=None, inputs=None):
        raise RuntimeError("mixed-device operands: refuse")

    def check_module(self, module):
        raise AssertionError(
            "check_module must not run — resolve raised"
        )


def _opt(**kw):
    return Optimizer(
        source=FakeSource(),
        sink=FakeSink(),
        composer=FakeComposer(),
        meter=FakeMeter(),
        **kw,
    )


def test_policy_is_a_runtime_checkable_port():
    assert isinstance(RecordingPolicy(), DevicePolicy)
    # A resolve-only object does NOT satisfy the full protocol — but
    # ``lower`` duck-types ``check_module`` anyway, so it still works
    # as a call-site value (see test_resolve_only_policy_skips_check_module).
    assert not isinstance(ResolveOnlyPolicy(), DevicePolicy)
    assert not isinstance(object(), DevicePolicy)


def test_search_records_resolved_device():
    pol = RecordingPolicy()
    model, x = _ListModel(), [1.0, 2.0]
    res = _opt().search(model, x, max_iterations=2, device_policy=pol)
    # resolve ran once, before export, on (model, x)
    assert pol.resolved == [(model, x)]
    assert res.stats["device"] == "fake:0"


def test_lower_checks_delivered_module_and_records_device():
    pol = RecordingPolicy()
    model, x = _ListModel(), [1.0, 2.0]
    opt = _opt()
    res = opt.search(model, x, max_iterations=2)
    lr = opt.lower(res, x, verify=False, device_policy=pol)
    # check_module saw the DELIVERED module (post-runner), resolve saw
    # it plus the verify input.
    assert pol.checked == [lr.module]
    assert pol.resolved[-1][0] is lr.module
    assert lr.stats["delivery_device"] == "fake:0"
    # the search record is untouched — lower's stats are a fresh dict
    assert "delivery_device" not in res.stats


def test_optimize_end_to_end_forwards_policy():
    pol = RecordingPolicy()
    opt = _opt()
    lr = opt.optimize(
        _ListModel(), [1.0, 2.0], max_iterations=2, device_policy=pol
    )
    assert lr.stats["device"] == "fake:0"
    assert lr.stats["delivery_device"] == "fake:0"
    # search-side resolve on the model, lower-side on the delivered
    # module — two resolves, one check.
    assert len(pol.resolved) == 2
    assert len(pol.checked) == 1


def test_optimizer_device_policy_field_and_backend():
    pol = RecordingPolicy()
    # explicit field wins outright
    opt = _opt(device_policy=pol)
    assert opt.device_policy is pol
    # backend-carried policy resolves when no explicit field is set
    backend = _fake_backend()
    object.__setattr__(backend, "device_policy", pol)
    opt2 = Optimizer(backend=backend)
    assert opt2.device_policy is pol
    # explicit field overrides the backend's
    other = RecordingPolicy("other:0")
    opt3 = Optimizer(backend=backend, device_policy=other)
    assert opt3.device_policy is other


def test_refusal_propagates_from_search():
    with pytest.raises(RuntimeError, match="mixed-device"):
        _opt().search(
            _ListModel(),
            [1.0],
            max_iterations=2,
            device_policy=RefusingPolicy(),
        )


def test_refusal_propagates_from_lower():
    class LowerRefuser(RecordingPolicy):
        def check_module(self, module):
            raise RuntimeError("sharded params: refuse")

    opt = _opt()
    res = opt.search(_ListModel(), [1.0], max_iterations=2)
    with pytest.raises(RuntimeError, match="sharded params"):
        opt.lower(
            res, [1.0], verify=False, device_policy=LowerRefuser()
        )


def test_resolve_only_policy_skips_check_module():
    opt = _opt()
    res = opt.search(_ListModel(), [1.0], max_iterations=2)
    lr = opt.lower(
        res, [1.0], verify=False, device_policy=ResolveOnlyPolicy()
    )
    assert lr.stats["delivery_device"] == "resolve-only"


def test_no_policy_is_the_historical_path():
    opt = _opt()
    lr = opt.optimize(_ListModel(), [1.0, 2.0], max_iterations=2)
    assert "device" not in lr.stats
    assert "delivery_device" not in lr.stats


def test_discover_alternatives_forwards_policy():
    pol = RecordingPolicy()
    res = discover_alternatives(
        _ListModel(),
        [1.0],
        source=FakeSource(),
        max_iterations=2,
        device_policy=pol,
    )
    assert res.stats["device"] == "fake:0"
