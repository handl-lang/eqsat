"""Substrate-pure tests ported from ``graphopt:tests/test_resilience.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def _toy_rule(name):
    from mlopt_core.egraph import Rewrite
    from mlopt_core.ir import Const, Op, TensorType, Var

    x = Var("x", TensorType((4,)))
    return Rewrite(
        name, Op.make("add", x, x), Op.make("mul", x, Const(2.0))
    )

def test_rulecache_concurrent_stores(tmp_path):
    """Parallel writers share no tmp file; every key loads cleanly."""
    from mlopt_core.rulecache import RuleCache

    cache = RuleCache(tmp_path / "rc")

    def store(k):
        RuleCache(tmp_path / "rc").store(
            f"key{k}", [_toy_rule(f"r{k}")]
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        # distinct keys plus same-key writers racing identical content
        list(pool.map(store, [*range(8), 0, 1, 2]))

    for k in range(8):
        loaded = cache.load(f"key{k}")
        assert loaded is not None and loaded[0].name == f"r{k}"
    assert not list(cache.dir.glob(".rules-*.tmp-*"))

def test_rulecache_sweeps_orphan_tmps(tmp_path):
    """Only *old* tmp files are swept — an in-flight write is safe."""
    import os

    from mlopt_core.rulecache import RuleCache

    cache = RuleCache(tmp_path / "rc")
    cache.dir.mkdir(parents=True)
    old = cache.dir / ".rules-dead.tmp-999-aa"
    old.write_text("partial")
    os.utime(old, (0, 0))  # crash leftover, not an in-flight write
    # An old directory under the tmp pattern raises OSError on
    # unlink — the sweep must swallow it, not fail the store.
    old_dir = cache.dir / ".rules-dead.tmp-888-bb"
    old_dir.mkdir()
    os.utime(old_dir, (0, 0))
    # A fresh tmp is a live writer's file — the sweep leaves it.
    fresh = cache.dir / ".rules-live.tmp-111-cc"
    fresh.write_text("in flight")

    cache.store("k", [_toy_rule("r")])
    assert not old.exists()
    assert old_dir.is_dir()
    assert fresh.exists()
    assert cache.load("k") is not None
