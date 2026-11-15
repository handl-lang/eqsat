"""The Compositional strategy's per-block driver.

:func:`_optimize_compositional` delegates every backend-native step
to the optimizer's ports — block selection, input capture,
per-block ``search``/``lower``, the pairwise cross-block pass, the
param-sharing clone + graft, and the fail-closed end-to-end verify
(:func:`_e2e_or_rollback` /
:func:`_roll_back_delivery` — a verified-failing graft never
ships).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

from mlopt_core.laws import RuleSet
from mlopt_core.pipeline import PipelineStats
from mlopt_core.ports import CostFn, Sink, Source, TaskMetric

from mlopt_orchestrator.cache import (
    _block_key,
    _cache_entry,
    _cache_replay,
)
from mlopt_orchestrator.pricing import _default_cost_fn
from mlopt_orchestrator.resources import (
    OptimizationResourceError,
    _looks_like_oom,
)
from mlopt_orchestrator.runners import IdentityRunner
from mlopt_orchestrator.taskgate import (
    VerifyDeclinedError,
    _block_decline,
    _block_gate,
    _task_contract,
)

if TYPE_CHECKING:
    from mlopt_orchestrator.optimize import Optimizer

logger = logging.getLogger("mlopt_orchestrator.compositional")


def _roll_back_delivery(
    model: Any,
    new_model: Any,
    composer: Any,
    replacements: Mapping[str, Any],
) -> Any:
    """Return the un-grafted composition after a failed e2e verify.

    A graft that breaks the whole-model forward must never ship
    (fail-closed): the delivery reverts to the original composition
    with NO replacement installed.  The per-block gate cannot catch
    this — it only ever sees each block's FIRST captured call, so a
    shape-specialized graft survives it and then breaks a second call
    at another resolution (the BiRefNet ``bb`` trap).

    ``new_model`` is the already-built clone; when nothing was grafted
    it IS the un-grafted composition (an empty graft is a no-op), so
    it is returned as-is.  Otherwise a fresh param-sharing clone keeps
    the "returned model is not the caller's live object" invariant the
    successful path holds; if that clone fails the input is returned
    unmodified — the same honest fallback ``clone_sharing`` uses.
    """
    if not replacements:
        return new_model
    try:
        return composer.clone_sharing(model)
    except Exception:
        return model


def _e2e_or_rollback(
    model: Any,
    new_model: Any,
    composer: Any,
    replacements: dict[str, Any],
    sink: Any,
    example_input: Any,
    verbose: bool,
    label: str = "Compositional",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run the whole-model check; revert the graft on failure.

    Returns ``(model_to_return, replacements_that_shipped, end_to_end)``.
    Fail-closed: a raising verify OR a non-passing report reverts the
    delivery to the un-grafted composition (:func:`_roll_back_delivery`)
    and empties ``replacements`` — a verified-failing graft never
    ships.  The per-block gate cannot catch this: it only ever sees
    each block's FIRST captured call, so a shape-specialized graft
    passes it and then breaks a second call at another resolution (the
    BiRefNet ``bb`` trap).

    ``label`` prefixes the verbose lines; the compositional, morphism
    and diagram drivers all deliver through this one gate.
    """
    args2 = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    e2e: dict[str, Any] = {}
    ok = False
    try:
        vr = sink.verify(model, new_model, args2)
        e2e = {
            "max_abs_diff": vr.max_abs,
            "max_rel_diff": vr.max_rel,
        }
        ok = vr.passed
        if verbose:
            print(  # stdout-compat: verbose-mode progress
                f"[{label}] end-to-end rel diff: {vr.max_rel:.3e}"
            )
    except Exception as e:
        e2e = {"error": f"{type(e).__name__}: {e}"}
        if verbose:
            print(  # stdout-compat: verbose-mode progress
                f"[{label}] end-to-end check failed: {e}"
            )
    if ok:
        return new_model, replacements, e2e
    e2e["rolled_back"] = True
    if verbose:
        print(  # stdout-compat: verbose-mode progress
            f"[{label}] end-to-end check failed — rolled back to the "
            "un-grafted composition"
        )
    return (
        _roll_back_delivery(model, new_model, composer, replacements),
        {},
        e2e,
    )


def _optimize_compositional(
    model: Any,
    example_input: Any,
    *,
    optimizer: Optimizer,
    block_pred: Callable | None = None,
    cost_fn: CostFn | None = None,
    rules: RuleSet | str | Iterable | None = None,
    max_iterations: int = 100,
    max_enodes: int | None = 100_000,
    max_memory_mb: float | None = None,
    verify_tol: float = 1e-4,
    max_cross_pairs: int = 8,
    cache: dict | None = None,
    error_budget: float | None = None,
    detect_specials: bool = False,
    detect_factors: bool = False,
    detect_headshare: bool = False,
    task: TaskMetric | None = None,
    task_tol: float | None = None,
    corpus_probe: Callable[[str, Any, Any], Any] | None = None,
    verbose: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Optimize a stacked/multi-block model one block at a time.

    Whole-model equality saturation is monolithic: the e-graph grows with
    the product of block structures, so deep stacks saturate slowly.
    This backend-neutral driver delegates every backend-native step to
    the optimizer's ports and

    1. selects *blocks* via the composer port — leaf sub-objects, plus
       any child where ``block_pred(parent, name, child)`` holds,
    2. runs the ORIGINAL model once (``composer.capture_inputs``) to
       capture each block's real input (a block's input is not the
       model input),
    3. runs the optimizer's ``search``/``lower`` phases on each block
       with its captured input, verifying the lowered block against
       the original on that input — a block that fails to export,
       saturate, lower, or verify keeps its original implementation;
       a block that crosses a resource bound (``max_enodes``,
       ``max_memory_mb``, or a caught OOM) fails with
       ``reason == "resource_limit"``,
    4. runs the pairwise cross-block pass (the composer's optional
       ``cross_pairs`` hook): for each *adjacent* pair whose boundary
       is a simple value flow (``composer.boundary``), build the
       joint micro-model, optimize it, verify it against the eager
       pair, and graft it in place of both blocks when it is verified
       AND cheaper than the separate results — capped at
       ``max_cross_pairs`` joint runs (``0`` disables),
    5. clones the model — structure deep-copied, parameter/buffer
       values *shared* with the original
       (``composer.clone_sharing``; no second copy of the weights, so
       recompose does not double device memory) — and grafts the
       optimized executors back in place
       (``composer.graft``), preserving the original forward
       structure, then verifies end-to-end equivalence on
       ``example_input`` through ``sink.verify``.

    ``error_budget`` / ``detect_specials`` / ``detect_factors`` /
    ``detect_headshare``
    forward to each per-block :meth:`Optimizer.search` — with a
    budget set, a block's certified weight-space bound propagates to
    an output bound (:func:`_propagate_bounds`, evaluated on the
    captured input), the sink verify runs with its relative
    tolerance open, and the measured error must honor the propagated
    bound (:func:`_bounded_gate`) or the block keeps its original
    implementation.

    ``task`` / ``task_tol`` forward the same way: under a task
    contract each block's verify gates on the task distance evaluated
    on the block's *captured* input (the metric is conditioned on
    that calibration input), and the block report records
    ``task = {name, tolerance, distance, passed}``.

    ``corpus_probe`` is the same injected hook :func:`search` takes,
    but scoped to the WHOLE model — it runs once, before the
    per-block loop, as ``corpus_probe(name, model, example_input)``
    (never per block: a block's captured input is not the model's
    feed), and its report lands in ``stats["corpus_probe"]``.

    ``cache`` (the :class:`Compositional` strategy resolves its
    ``cache=`` flag into this mapping): when a dict is given, each
    block's exported IR is canonicalised by :func:`structural_key` —
    same op-DAG, same constants, Params/Vars differing at most by
    name (positions, shapes and dtypes are pinned) — and a block whose
    key already has a template replays the cached extracted term under
    its own leaf names/values instead of re-running the search.  The
    replay re-derives the template's value-dependent assumptions
    (exact-tie clusters and ``__heads`` dedup stacks recorded in
    ``stats["param_sharing"]``; the causal-mask fold re-evaluated on
    the hit block's tensors) and the delivered module is verified
    numerically like any other — an unservable or failing replay
    falls back to the ordinary search.

    Returns ``(recomposed_model, stats)`` where ``stats["blocks"]`` maps
    each block's dotted name to ``{"status", "stats", "param_report",
    "time_s", ...}`` and ``stats["param_report"]`` aggregates the
    per-block parameter diffs (eliminated/derived names are prefixed by
    block name for auditability) — the composer's ``param_report`` hook
    supplies the per-block diff when it exists.  ``stats["cross_pairs"]``
    maps each attempted pair ``"a+b"`` to ``{"status", "boundary", ...}``
    — ``"grafted"`` / ``"declined"`` / ``"skipped"``.  ``stats["shared_
    params"]`` is True when the recomposed model shares the original's
    storage (the normal path); ``stats["in_place"]`` True means cloning
    failed and the input was returned unmodified.  With caching enabled,
    ``stats["cache"]`` reports ``{"hits", "misses", "fallbacks"}`` and
    each block's report carries ``"cache"``: ``"hit"`` / ``"miss"`` /
    ``"replay_failed"``.
    """
    t_start = time.time()
    composer = optimizer.composer
    if composer is None:
        raise TypeError(
            "Compositional strategy needs a Composer port — "
            "pass composer= (or backend=) to Optimizer"
        )
    # ``Optimizer.__post_init__`` already rejects a missing source/sink,
    # so by construction both ports are present here.
    source = cast(Source, optimizer.source)
    sink = cast(Sink, optimizer.sink)
    if cost_fn is None:
        cost_fn = _default_cost_fn()
    contract = _task_contract(task, task_tol)

    # Whole-model corpus probe — once, before any per-block search
    # (the hook contract is ``search``'s: ``(name, model, x)``).
    probe_report: Any = None
    if corpus_probe is not None:
        probe_report = corpus_probe(
            getattr(model, "__name__", None) or type(model).__name__,
            model,
            example_input,
        )

    blocks = composer.blocks(model, predicate=block_pred)
    if verbose:
        names = [n for n, _ in blocks]
        print(  # stdout-compat: verbose-mode progress
            f"[Compositional] {len(blocks)} candidate blocks: {names}"
        )

    captured, io = composer.capture_inputs(model, blocks, example_input)
    # A second capture on a perturbed probe input arms the residual
    # boundary check against a coincidence of values (it only runs when
    # the pair pass is enabled; a probe failure fails closed — residual
    # pairs decline, identity-proven chain pairs still work).
    captured2: dict[str, Any] = {}
    io2: dict[str, Any] = {}
    if max_cross_pairs:
        import contextlib

        with contextlib.suppress(Exception):
            captured2, io2 = composer.capture_inputs(
                model, blocks, composer.perturbed(example_input)
            )

    replacements: dict[str, Any] = {}
    block_reports: dict[str, dict[str, Any]] = {}
    agg = {
        "original_params": 0,
        "optimized_params": 0,
        "original_bytes": 0,
        "optimized_bytes": 0,
        "eliminated": [],
        "derived": [],
    }
    # The weight-file diff is an optional composer hook — a backend
    # without one reports no parameter audit.
    param_diff = getattr(composer, "param_report", None)
    cache_hits = 0
    cache_misses = 0
    cache_fallbacks = 0

    for name, block in blocks:
        rep: dict[str, Any] = {"status": "not_executed"}
        block_reports[name] = rep
        cap = captured.get(name)
        if cap is None:
            continue
        args, kwargs = cap
        if kwargs:
            rep["status"] = "skipped"
            rep["reason"] = f"non-positional kwargs {sorted(kwargs)}"
            continue
        ex = args[0] if len(args) == 1 else args
        t0 = time.time()
        try:
            opt_mod = st = None
            key = None
            entry = None
            if cache is not None:
                # Export once up-front for the structural signature —
                # a cache hit skips the search phase entirely.
                kir, ktensors = source.to_ir(block, ex)
                key = _block_key(kir, ktensors, args)
                entry = cache.get(key)
                res2 = _cache_replay(
                    entry,
                    kir,
                    ktensors,
                    sink=sink,
                    source=source,
                    model=block,
                )
                if res2 is not None:
                    try:
                        lr2 = optimizer.lower(
                            res2,
                            ex,
                            runner=IdentityRunner(),
                            verify=verbose,
                            verbose=verbose,
                        )
                        vh, _out2, gate_ok = _block_gate(
                            sink,
                            block,
                            lr2.module,
                            args,
                            verify_tol,
                            lr2.stats,
                            res2,
                            rep,
                            task,
                            contract,
                        )
                        if gate_ok:
                            opt_mod, st = lr2.module, lr2.stats
                            cache_hits += 1
                            rep["cache"] = "hit"
                            if verbose:
                                print(  # stdout-compat: verbose-mode progress
                                    f"[Compositional] {name}: "
                                    f"cache hit ({vh.max_rel:.2e})"
                                )
                    except Exception as exc:
                        # A failed replay simply reverts to search.
                        logger.debug(
                            "[Compositional] %s: cache replay failed: %s",
                            name,
                            exc,
                        )
            if opt_mod is None:
                if entry is not None:
                    # An entry existed but could not serve this block
                    # (unmappable leaf, violated value assumption, or a
                    # failed replay verify) — fall back to the search.
                    cache_fallbacks += 1
                    rep["cache"] = "replay_failed"
                elif cache is not None:
                    rep["cache"] = "miss"
                # Per-block search+lower — the same phases the monolithic
                # pipeline runs, through the optimizer's ports.
                res = optimizer.search(
                    block,
                    ex,
                    rules=rules,
                    max_iterations=max_iterations,
                    max_enodes=max_enodes,
                    max_memory_mb=max_memory_mb,
                    cost_fn=cost_fn,
                    delivers_compiled=False,
                    error_budget=error_budget,
                    detect_specials=detect_specials,
                    detect_factors=detect_factors,
                    detect_headshare=detect_headshare,
                    task=task,
                    task_tol=task_tol,
                    verbose=verbose,
                )
                lr = optimizer.lower(
                    res,
                    ex,
                    runner=IdentityRunner(),
                    verify=verbose,
                    verbose=verbose,
                )
                opt_mod, st = lr.module, lr.stats
                if cache is not None:
                    cache_misses += 1
                    # ``key`` is always set here: a cache-enabled block
                    # that failed its export exited through the outer
                    # ``except`` before reaching the miss path.
                    # ``setdefault``: a fallback block's fresh template
                    # must not evict the existing one — an entry that
                    # declined THIS block may still serve later blocks
                    # whose values honour its assumptions.
                    cache.setdefault(key, _cache_entry(res))
                # Per-block verification on the captured input —
                # soundness gate independent of the pipeline's own
                # (verbose-gated) check.  Any mismatch or eval failure
                # falls back.  Under a bounded search the certified
                # bound is in weight space, so it propagates to an
                # output bound on the captured input first
                # (``_propagate_bounds``) and the measured error must
                # honor THAT — same units — while the sink's relative
                # tolerance opens (the bound is the tolerance).
                vr, out_b, gate_ok = _block_gate(
                    sink,
                    block,
                    opt_mod,
                    args,
                    verify_tol,
                    st,
                    res,
                    rep,
                    task,
                    contract,
                )
                if not gate_ok:
                    _block_decline(rep, vr, out_b)
            replacements[name] = opt_mod
            rep["status"] = "optimized"
            rep["stats"] = st
            if param_diff is not None:
                pr = param_diff(block, opt_mod)
                rep["param_report"] = pr
                agg["original_params"] += pr["original_params"]
                agg["optimized_params"] += pr["optimized_params"]
                agg["original_bytes"] += pr["original_bytes"]
                agg["optimized_bytes"] += pr["optimized_bytes"]
                agg["eliminated"] += [
                    f"{name}:{n}" for n in pr["eliminated"]
                ]
                agg["derived"] += [f"{name}:{n}" for n in pr["derived"]]
            if verbose:
                print(  # stdout-compat: verbose-mode progress
                    f"[Compositional] {name}: optimized "
                    f"({rep['rel_diff']:.2e})"
                )
        except Exception as e:
            # A verify-gate refusal (``VerifyDeclinedError`` — raised
            # by ``lower(verify=True)`` under verbose, or by
            # ``_block_decline`` on a failed block gate) is a decline,
            # not a failure: the block keeps its original either way,
            # but the gate's own taxonomy label stays honest.
            rep["status"] = (
                "declined"
                if isinstance(e, VerifyDeclinedError)
                else "failed"
            )
            rep["error"] = f"{type(e).__name__}: {e}"
            if isinstance(
                e, OptimizationResourceError
            ) or _looks_like_oom(e):
                rep["reason"] = "resource_limit"
            if verbose:
                print(  # stdout-compat: verbose-mode progress
                    f"[Compositional] {name}: keeping original ({e})"
                )
        rep["time_s"] = time.time() - t0

    # -- Pairwise cross-block pass --------------------------------------
    # Per-block optimization is blind across the boundary: adjacent
    # blocks can share transforms a per-block search cannot see (block
    # i's output projection composing with block i+1's input
    # projections; a residual add absorbing a shared affine).  The
    # pass is adapter machinery — verified and cost-gated; every
    # decline keeps the separate results.
    cross_pairs: dict[str, dict[str, Any]] = {}
    cross = getattr(composer, "cross_pairs", None)
    if max_cross_pairs and callable(cross):
        cross_pairs = cross(
            blocks,
            captured,
            io,
            captured2,
            io2,
            replacements,
            block_reports,
            agg,
            rules=rules,
            max_iterations=max_iterations,
            max_enodes=max_enodes,
            max_memory_mb=max_memory_mb,
            cost_fn=cost_fn,
            verify_tol=verify_tol,
            source=source,
            sink=sink,
            max_cross_pairs=max_cross_pairs,
            verbose=verbose,
        )

    # -- Recompose -------------------------------------------------------
    # The clone grafts structure, never values, so it shares the
    # original's parameter/buffer storage — recomposing costs no extra
    # weight bytes (the old plain deepcopy doubled the footprint and
    # OOMed at ~0.5B fp16 on small GPUs).
    in_place = False
    try:
        new_model = composer.clone_sharing(model)
    except Exception:
        # Never graft into the caller's live model: the replacements
        # carry shape-specialized state baked by the export for the
        # example input, and a mutated caller fails at the NEXT input
        # shape (and the e2e check degenerates to self-comparison).
        new_model = model
        in_place = True
        replacements = {}
    composer.graft(new_model, replacements)

    # -- End-to-end verification (fail-closed) ---------------------------
    # The per-block gate only ever sees each block's FIRST captured call,
    # so a shape-specialized graft can pass it and still break a second
    # call at another resolution (the BiRefNet ``bb`` trap: ``mul_scl_
    # ipt='cat'`` calls the backbone twice per forward).  A failed
    # whole-model check therefore rolls the delivery BACK to the
    # un-grafted composition — a verified-failing graft never ships.
    if in_place:
        # new_model IS the input model — a verify would be a
        # self-comparison that always reports 0.0.  Record the
        # degenerate case honestly instead of a false pass.
        e2e: dict[str, Any] = {
            "skipped": "in_place",
            "reason": "param-sharing clone failed — returned model "
            "is the input module, unmodified",
        }
        logger.warning(
            "optimize_compositional: clone failed — returning the "
            "input module unmodified (no blocks grafted; "
            "report.in_place=True)"
        )
    else:
        new_model, replacements, e2e = _e2e_or_rollback(
            model,
            new_model,
            composer,
            replacements,
            sink,
            example_input,
            verbose,
        )

    agg["bytes_saved"] = agg["original_bytes"] - agg["optimized_bytes"]
    agg["ratio"] = (
        agg["optimized_bytes"] / agg["original_bytes"]
        if agg["original_bytes"]
        else 1.0
    )
    stats: PipelineStats = PipelineStats(
        {
            "compositional": True,
            "n_blocks": len(blocks),
            # Post-rollback: a failed e2e cleared ``replacements``, so
            # this counts what actually shipped (0 when the graft was
            # reverted).
            "n_optimized": len(replacements),
            "n_failed": sum(
                1
                for r in block_reports.values()
                if r["status"] == "failed"
            ),
            "n_declined": sum(
                1
                for r in block_reports.values()
                if r["status"] == "declined"
            ),
            "n_skipped": sum(
                1
                for r in block_reports.values()
                if r["status"] in ("skipped", "not_executed")
            ),
            "blocks": block_reports,
            "in_place": in_place,
            "shared_params": not in_place,
            "param_report": agg,
            "cross_pairs": cross_pairs,
            "end_to_end": e2e,
        }
    )
    if cache is not None:
        stats["cache"] = {
            "hits": cache_hits,
            "misses": cache_misses,
            "fallbacks": cache_fallbacks,
        }
    if probe_report is not None:
        stats["corpus_probe"] = probe_report

    stats["wall_time_s"] = time.time() - t_start
    return new_model, stats
