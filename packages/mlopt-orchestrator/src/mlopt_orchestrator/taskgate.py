"""Verify gates — the task-metric contract and delivery verdicts.

The ``task=`` / ``task_tol=`` contract normalisation
(:func:`_task_contract` / :func:`_resolve_task`), the task-distance
verify (:func:`_task_verify` / :func:`_block_verify_task`), the
operative-gate composition (:func:`_gate_delivery` /
:func:`_block_gate`) that runs the pointwise/bound machinery first
and lets a task contract replace the verdict, and the acceptance
labels (:func:`_verify_metric` / :func:`_accepted_by`) recorded in
``stats`` — never silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlopt_core.ports import Sink

from mlopt_orchestrator.bounds import (
    _block_bound_gate,
    _bound_verify,
    _bounded_gate,
    _rel_bound,
)

# ---------------------------------------------------------------------------
#  Task-metric verify gate (plan 0015) — the certificate's task contract
# ---------------------------------------------------------------------------


def _task_contract(
    task: Any, task_tol: float | None
) -> tuple[str, float] | None:
    """Normalise a ``task=`` / ``task_tol=`` pair to ``(name, tol)``.

    ``None`` when no task metric is configured — the pointwise/bound
    gate is then unchanged.  A ``task_tol`` with no metric, a metric
    without a callable ``distance``, and a metric with no
    ``tolerance`` member and no explicit ``task_tol`` are all loud
    errors — a gate's parameters are never guessed.
    """
    if task is None:
        if task_tol is not None:
            raise TypeError("task_tol= requires a task= metric")
        return None
    if not callable(getattr(task, "distance", None)):
        raise TypeError(
            "task metric must be a TaskMetric — an object with "
            "distance(ref, opt), name and tolerance; "
            f"got {type(task).__name__} (no callable distance)"
        )
    name = str(getattr(task, "name", type(task).__name__))
    tol = (
        float(task_tol)
        if task_tol is not None
        else getattr(task, "tolerance", None)
    )
    if tol is None:
        raise TypeError(
            f"task metric {name!r} declares no tolerance — "
            "pass task_tol= explicitly"
        )
    return name, float(tol)


def _run_exec(mod: Any, args: tuple) -> Any:
    """Run a lowered executor on positional ``args`` (duck-typed).

    The :class:`~mlopt_core.ports.Executor` contract is ``forward``;
    a runner-delivered wrapper that only defines ``__call__`` still
    works.  Tensor-ish args are cloned — the same guard
    ``verify_module`` uses against in-place forwards.
    """
    fn = getattr(mod, "forward", None)
    if not callable(fn):
        fn = mod
    return fn(*[a.clone() if hasattr(a, "clone") else a for a in args])


def _task_verify(
    stats: dict[str, Any],
    report: Any,
    task: Any,
    contract: tuple[str, float],
    ref: Any,
    mod: Any,
    args: tuple,
) -> _TaskVerify:
    """Evaluate the task metric on the verify args; gate on it.

    Under a task contract the task distance is the verdict — that is
    what the caller opted into.  The pointwise measurements still ride
    the returned report (``max_abs`` / ``max_rel``) and the bound
    ledger keeps its ``error_bounds_honored`` verdict, so a
    task-accepted delivery records BOTH numbers: certified under the
    task tolerance, drift measured.  ``stats["task"]`` gains the
    measured ``distance`` and the verdict — never silent.

    The metric evaluates on the *verify* input — the certificate is
    calibration-conditioned on exactly this input distribution.
    """
    name, tol = contract
    dist = float(
        task.distance(_run_exec(ref, args), _run_exec(mod, args))
    )
    passed = bool(dist <= tol)
    stats["task"] = {
        "name": name,
        "tolerance": tol,
        "distance": dist,
        "passed": passed,
        "evaluated_on": "verify_input",
    }
    return _TaskVerify(
        max_abs=report.max_abs,
        max_rel=report.max_rel,
        passed=passed,
        task_metric=name,
        task_distance=dist,
        task_tolerance=tol,
    )


def _verify_metric(
    contract: tuple[str, float] | None, bound_total: float
) -> str:
    """Name the metric that produced the verify verdict.

    ``"max_rel"`` — the plain pointwise gate; ``"bound"`` — the
    propagated output bound (a bounded member was delivered); the
    task metric's name when a ``task=`` contract gated.
    """
    if contract is not None:
        return contract[0]
    return "bound" if bound_total > 0.0 else "max_rel"


def _accepted_by(stats: dict[str, Any], verified: Any) -> str:
    """Which contract accepted the delivery — or ``"declined"``.

    ``"task"`` when a task metric gated acceptance, ``"bound"`` when
    the propagated output bound did, ``"pointwise"`` when the plain
    tolerance gate did.  The same label lands on each
    ``stats["error_bounds"]`` entry — a task-accepted bounded member
    records ``accepted_by="task"`` (and its pointwise bound is still
    reported).
    """
    if not verified.passed:
        return "declined"
    if stats.get("task", {}).get("passed"):
        return "task"
    return (
        "bound"
        if float(stats.get("error_bound_total", 0.0)) > 0.0
        else "pointwise"
    )


@dataclass(frozen=True)
class _TaskVerify:
    """A ``VerifyResult``-shaped report carrying the task verdict.

    ``max_abs`` / ``max_rel`` keep the pointwise measurements —
    reported either way — while ``passed`` is the task gate's verdict.
    The extra fields expose which metric gated, its tolerance and the
    measured distance.
    """

    max_abs: float
    max_rel: float
    passed: bool
    task_metric: str = ""
    task_distance: float = 0.0
    task_tolerance: float = 0.0


def _resolve_task(
    task: Any, task_tol: float | None, result: Any
) -> tuple[Any, tuple[str, float] | None]:
    """Resolve the effective ``(task, (name, tol))`` contract pair.

    ``lower`` precedence: an explicit ``task=`` wins outright; absent
    one, the contract recorded on the :class:`SearchResult` by
    :func:`search` governs (metric AND resolved tolerance).  An
    explicit ``task=`` without ``task_tol`` falls back to the
    metric's own ``tolerance`` — never to a tolerance recorded for a
    different metric.  ``_task_contract`` loudly rejects the bad
    combinations (tolerance without a metric, a metric without
    ``distance``, a tolerance-less metric without an override).
    """
    if task is None:
        task = getattr(result, "task", None)
        if task is not None and task_tol is None:
            task_tol = getattr(result, "task_tol", None)
    return task, _task_contract(task, task_tol)


def _contract_tol(contract: tuple[str, float] | None) -> float | None:
    """Return the resolved tolerance of a task contract, else None."""
    return contract[1] if contract is not None else None


def _task_declared(
    stats: dict[str, Any], contract: tuple[str, float] | None
) -> None:
    """Record the declared task contract in the search stats.

    The contract is declared at search and *evaluated* at lower's
    verify — ``distance``/``passed`` stay absent until then, so a
    promised-but-unevaluated contract is visible as such.
    """
    if contract is not None:
        stats["task"] = {"name": contract[0], "tolerance": contract[1]}


def _gate_delivery(
    stats: dict[str, Any],
    report: Any,
    bound_total: float,
    output_bound: float | None,
    task: Any,
    contract: tuple[str, float] | None,
    ref: Any,
    mod: Any,
    x: Any,
) -> Any:
    """Apply the operative verify gate and record the verdict.

    The pointwise/bound machinery always runs first (``stats``
    carries ``error_bounds_honored`` and the measured fields either
    way); under a task contract the task distance then replaces the
    verdict — the bound ledger stays on the record, never silent.
    ``verify_metric`` names the gating metric and ``accepted_by``
    which contract accepted — the same label lands on each
    ``error_bounds`` entry.
    """
    verified = _bound_verify(stats, report, bound_total, output_bound)
    if contract is not None:
        verified = _task_verify(
            stats,
            report,
            task,
            contract,
            ref,
            mod,
            x if isinstance(x, tuple) else (x,),
        )
    stats["verify_metric"] = _verify_metric(contract, bound_total)
    stats["accepted_by"] = _accepted_by(stats, verified)
    for e in stats.get("error_bounds", []):
        e["accepted_by"] = stats["accepted_by"]
    return verified


def _block_verify_task(
    task: Any,
    contract: tuple[str, float] | None,
    block: Any,
    mod: Any,
    args: tuple,
) -> dict[str, Any] | None:
    """Task-gate record for one compositional block verify.

    ``None`` when no task contract is configured (the block's
    pointwise/bound gate is then unchanged).  Otherwise runs both
    sides on the block's *captured* input — the metric is conditioned
    on that calibration input — and returns the serializable record
    (``name`` / ``tolerance`` / ``distance`` / ``passed``) that lands
    in the block report and its stats.
    """
    if contract is None:
        return None
    name, tol = contract
    dist = float(
        task.distance(_run_exec(block, args), _run_exec(mod, args))
    )
    return {
        "name": name,
        "tolerance": tol,
        "distance": dist,
        "passed": bool(dist <= tol),
        "evaluated_on": "captured_input",
    }


def _block_gate(
    sink: Sink,
    block: Any,
    mod: Any,
    args: tuple,
    verify_tol: float,
    st: dict[str, Any],
    res: Any,
    rep: dict[str, Any],
    task: Any,
    contract: tuple[str, float] | None,
) -> tuple[Any, float | None, bool]:
    """Verify one delivered block module; return the gate record.

    Runs ``_block_bound_gate`` first — the pointwise report and the
    bound propagation land in ``st`` either way (the ledger stays
    honest) — then the operative verdict: under a task contract the
    task distance measured on the captured input decides and the
    record lands in ``rep["task"]`` / ``st["task"]`` /
    ``st["verify_metric"]`` / ``st["accepted_by"]``; otherwise the
    propagated-bound gate decides.  Returns ``(report,
    output_bound, ok)``.
    """
    vr, out_b = _block_bound_gate(
        sink, block, mod, args, verify_tol, st, res
    )
    rep["rel_diff"] = vr.max_rel
    trec = _block_verify_task(task, contract, block, mod, args)
    if trec is None:
        return vr, out_b, _bounded_gate(vr, out_b)
    rep["task"] = trec
    st["task"] = trec
    st["verify_metric"] = trec["name"]
    st["accepted_by"] = "task" if trec["passed"] else "declined"
    return vr, out_b, bool(trec["passed"])


def _block_decline(
    rep: dict[str, Any], vr: Any, out_b: float | None
) -> None:
    """Raise the per-block verify failure — task or bound wording."""
    t = rep.get("task")
    if t is not None:
        raise RuntimeError(
            f"block verification failed: task {t['name']} distance "
            f"{t['distance']:.3e} exceeds tolerance "
            f"{t['tolerance']:.3e}"
        )
    raise RuntimeError(
        f"block verification failed: rel diff {vr.max_rel:.3e} "
        f"(accepted bound {_rel_bound(out_b, vr):.3e})"
    )
