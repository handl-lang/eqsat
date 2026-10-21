"""``mlopt_core.profile`` — the torch-free half of calibration.

The measurement half lives in ``mlopt_torch.calibrate``; this pins
the data half — the ``TargetProfile`` record, JSON persistence, the
``shape_bucket`` keying, the measured-feedback channel
(``record_measured`` / ``measured_price_ns``) and the learned
correction table (``corrected_price_ns``) — without naming a tensor
library (``shape_bucket`` only needs ``numel()``/``device``).
"""

from __future__ import annotations

import json

import pytest

from mlopt_core.profile import (
    PROFILE_DIR_ENV,
    TargetProfile,
    corrected_price_ns,
    list_profiles,
    load_profile,
    measured_price_ns,
    profile_graph_overhead_us,
    profiles_dir,
    record_measured,
    save_profile,
    shape_bucket,
)

from tests.faketensor import FakeTensor


def _prof(**kw):
    base = {
        "name": "test-gpu",
        "tflops": 40.0,
        "gbps": 800.0,
        "launch_us": 4.0,
        "device": "cuda:0",
        "measured_at": "2026-10-21T00:00:00",
    }
    return TargetProfile(**{**base, **kw})


def test_profile_json_round_trip_and_unknown_keys():
    p = _prof(meta={"dtype": "fp32"}, op_kernel_ns={"matmul": {"8x8x8": 123.0}})
    p2 = TargetProfile.from_json(p.to_json())
    assert p2 == p
    # unknown keys are dropped; a non-dict is a loud error
    p3 = TargetProfile.from_json({**json.loads(p.to_json()), "bogus": 1})
    assert p3 == p
    assert TargetProfile.from_json(p.to_json().encode()) == p
    with pytest.raises(TypeError):
        TargetProfile.from_json(42)


def test_profile_persistence_round_trip(tmp_path):
    p = _prof()
    path = save_profile(p, tmp_path)
    assert path.name == "test-gpu.json"
    # load by name, by safe-name file, and by explicit path
    assert load_profile("test-gpu", tmp_path) == p
    assert load_profile(path) == p
    assert list_profiles(tmp_path) == ["test-gpu"]
    # a second profile lists alongside; bad files are skipped
    save_profile(_prof(name="other gpu"), tmp_path)
    (tmp_path / "junk.json").write_text("not json")
    assert sorted(list_profiles(tmp_path)) == ["other gpu", "test-gpu"]
    with pytest.raises(FileNotFoundError):
        load_profile("missing", tmp_path)


def test_profiles_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path / "prof"))
    d = profiles_dir()
    assert d.name == "prof"
    # persistence honours the override
    save_profile(_prof())
    assert (d / "test-gpu.json").is_file()


def test_safe_name_sanitises(tmp_path):
    p = _prof(name="a/b c:d")
    path = save_profile(p, tmp_path)
    assert "/" not in path.name and path.suffix == ".json"
    assert load_profile("a/b c:d", tmp_path).name == "a/b c:d"


def test_shape_bucket_order_of_magnitude():
    # numel() and device are the whole contract — duck-typed
    t = FakeTensor([[0.0] * 16] * 8)  # 128 elems -> 2^7
    assert shape_bucket(t) == "cpu:2^7"
    t2 = FakeTensor([0.0] * 200)  # ceil(log2 200) = 8
    assert shape_bucket(t2) == "cpu:2^8"

    class _Dev(FakeTensor):
        @property
        def device(self):
            return "cuda:0"

    assert shape_bucket(_Dev([[0.0]])) == "cuda:0:2^0"
    # tuples/lists count every element's numel
    assert shape_bucket((FakeTensor([0.0] * 8), FakeTensor([0.0] * 8))) == "cpu:2^4"
    # a non-tensor input still buckets (numel 1 floor)
    assert shape_bucket(None) == "cpu:2^0"


def test_graph_overhead_us_sources_and_fallbacks():
    assert profile_graph_overhead_us({}) > 0
    assert profile_graph_overhead_us({"graph_overhead_us": 12.5}) == 12.5
    assert profile_graph_overhead_us({"graph_overhead_us": -1}) > 0
    assert profile_graph_overhead_us(_prof(graph_overhead_us=7.5)) == 7.5
    assert profile_graph_overhead_us(None) > 0


def test_record_and_measured_price_ns():
    prof: dict = {}
    record_measured(prof, "batched", "cuda:0:2^7", median_ns=900.0, model_ns=600.0)
    # additive residual: model + (measured - recorded model)
    assert measured_price_ns(prof, "batched", "cuda:0:2^7", 700.0) == 1000.0
    # unknown pair -> model price unchanged
    assert measured_price_ns(prof, "batched", "cuda:0:2^9", 700.0) == 700.0
    assert measured_price_ns(prof, "generic", "cuda:0:2^7", 700.0) == 700.0
    # no recorded model_ns -> the measured median IS the price
    record_measured(prof, "eager", "cuda:0:2^7", median_ns=1500.0)
    assert measured_price_ns(prof, "eager", "cuda:0:2^7", 700.0) == 1500.0
    assert measured_price_ns(prof, "eager", "cuda:0:2^7", None) == 1500.0
    # object profiles are updated via replace()/attribute set too
    p = record_measured(
        _prof(), "batched", "cuda:0:2^7", median_ns=900.0, model_ns=600.0
    )
    assert measured_price_ns(p, "batched", "cuda:0:2^7", 700.0) == 1000.0


def test_corrected_price_ns_learns_factor():
    prof: dict = {}
    # two observations reach _CORRECTION_MIN_SAMPLES
    record_measured(prof, "batched", "cpu:2^7", median_ns=1200.0, model_ns=600.0)
    record_measured(prof, "batched", "cpu:2^7", median_ns=1400.0, model_ns=700.0)
    # learned factor ~= 2x -> corrected price scales the model
    got = corrected_price_ns(prof, "batched", "cpu:2^7", 500.0)
    assert got == pytest.approx(1000.0, rel=0.15)
    # a candidate with too few samples falls back to the model price
    assert corrected_price_ns(prof, "generic", "cpu:2^7", 500.0) == 500.0
    assert corrected_price_ns({}, "batched", "cpu:2^7", 500.0) == 500.0


def test_cost_fn_binds_roofline():
    p = _prof()
    fn = p.cost_fn()
    assert callable(fn)
