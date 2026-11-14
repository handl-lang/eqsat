"""Resource bounds for the optimization pipeline.

The ``max_enodes`` / ``max_memory_mb`` caps and the
allocator-failure -> :class:`OptimizationResourceError` translation
shared by every pipeline entry point (:func:`search`,
:func:`lower`, :func:`discover_alternatives`, the compositional
driver).  Backend-agnostic: host RSS is read portably and device
bytes arrive through the optional ``Meter.device_memory_mb`` hook.
"""

from __future__ import annotations

from typing import Any


class OptimizationResourceError(RuntimeError):
    """Raised when the optimizer crosses a resource bound.

    Sources: the e-graph reached ``max_enodes`` (checked once per
    saturation iteration inside ``EGraph.run`` and again at phase
    boundaries here), the process/device memory footprint crossed
    ``max_memory_mb``, or an allocator-failure exception — host
    ``MemoryError``, the backend's ``OutOfMemoryError`` (torch's
    ``torch.cuda.OutOfMemoryError`` is one), or the RuntimeError
    variants allocator failures surface as — anywhere in the
    export → saturation → lowering pipeline.

    The :class:`Compositional` strategy records these as ordinary
    per-block failures with ``reason == "resource_limit"``; a
    standalone ``Optimizer.optimize`` caller gets this dedicated type
    instead of a raw OOM.
    """


def _looks_like_oom(exc: BaseException) -> bool:
    """Return True for allocator-failure exceptions — backend-agnostic.

    Host ``MemoryError``, a backend-named ``*OutOfMemoryError*`` class
    (``torch.cuda.OutOfMemoryError`` and any future backend's
    equivalent), and the ``RuntimeError`` variants allocator failures
    surface as ("CUDA out of memory", DefaultCPUAllocator's
    "can't allocate memory").
    """
    if isinstance(exc, MemoryError) or (
        isinstance(exc, RuntimeError)
        and "outofmemoryerror" in type(exc).__name__.lower()
    ):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "can't allocate memory" in msg
            or "cannot allocate memory" in msg
        )
    return False


def _oom_to_resource_error(fn):
    """Wrap an optimizer entry point against allocator failures.

    They surface as :class:`OptimizationResourceError` instead of a raw
    OOM.  ``functools.wraps`` keeps the public signature and docstring.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OptimizationResourceError:
            raise
        except Exception as e:
            if _looks_like_oom(e):
                raise OptimizationResourceError(
                    f"{type(e).__name__}: {e}"
                ) from e
            raise

    return wrapper


def _current_memory_mb(meter: Any = None) -> float:
    """Return the current process memory footprint in MiB.

    Host RSS (read from ``/proc/self/status``, backend-agnostic) plus
    backend-device bytes the adapter reports — the meter's optional
    ``device_memory_mb()`` hook (torch's implementation reads
    ``torch.cuda.memory_allocated``; device memory lives outside RSS).
    A meter without the hook contributes zero.
    """
    rss = 0.0
    try:
        with open("/proc/self/status") as fh:
            for line in (
                fh
            ):  # pragma: no branch — VmRSS always present on Linux
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
                    break
    except OSError:  # pragma: no cover — non-Linux only
        import resource

        rss = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        )
    dev_fn = getattr(meter, "device_memory_mb", None)
    dev = float(dev_fn()) if callable(dev_fn) else 0.0
    return rss + dev


def _check_resources(
    eg, max_enodes, max_memory_mb, meter: Any = None
) -> None:
    """Cheap watermark check, called at phase boundaries.

    ``max_enodes`` is already enforced inside ``EGraph.run`` once per
    saturation iteration — and mid-iteration for budgeted rules via the
    match-loop ``enode_budget`` — so a crossing observed here means the
    run truncated against the cap: fail fast rather than spending
    extraction/lowering effort on an over-budget graph.  The memory
    check catches tensor pressure an e-node count cannot see
    (materialised weight folds, lowered parameters).
    """
    if (
        max_enodes is not None
        and eg is not None
        and eg.n_enodes >= max_enodes
    ):
        raise OptimizationResourceError(
            f"e-graph reached {eg.n_enodes} e-nodes "
            f"(max_enodes={max_enodes})"
        )
    if max_memory_mb is not None:
        used = _current_memory_mb(meter)
        if used > max_memory_mb:
            raise OptimizationResourceError(
                f"memory footprint {used:.0f} MiB exceeds "
                f"max_memory_mb={max_memory_mb}"
            )
