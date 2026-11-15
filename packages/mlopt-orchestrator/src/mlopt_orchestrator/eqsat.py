"""``engine-eqsat`` bridge — the ``engine="eqsat"`` opt-in (ADR-0178 WP3).

The search interpretation of ADR-0174/0177 lives in the Rust crate
``hnl-engine-eqsat`` (repo ``handl-lang/engine-eqsat``).  This module is
the Python half of the bridge: it adapts that engine to the
:class:`~mlopt_core.ports.Engine` seam so ``search`` /
``Optimizer.search`` can select it explicitly::

    res = search(model, x, source=..., sink=..., engine="eqsat")
    # or, configured:
    res = search(model, x, source=..., sink=...,
                 engine=EqsatEngine(cost={"kind": "op_table", ...}))

**Bridge shape: CLI subprocess, not PyO3.**  ``eqsat-run``
(``engine-eqsat/src/bin/eqsat-run.rs``) is a JSON-in/JSON-out driver —
one stdin spec saturates the submitted kernel ``Term``s under a
``LAW_FORMAT = 2`` law set and prints the extracted best term, the
saturation stats, and per-rule fire provenance.  That wire protocol is
exactly what the differential harness
(``graphopt:bench/drivers/audit/eqsat_differential.py``) proved at
174/174 semantic parity, so the bridge reuses it verbatim rather than
adding a second FFI surface.  A PyO3 binding remains the WP4
latency upgrade path (ADR-0178 D2 prefers it for a per-search hot
loop); the subprocess keeps this package stdlib-only and the engine
binary an ordinary build artifact.

**The ``Engine`` port is batch-shaped, the CLI is one-shot.**  The
port's ``add_term``/``run``/``extract_best`` split maps onto the CLI's
single saturate+extract call: ``add_term`` records the term,
``run`` performs the one subprocess invocation over every recorded
term (the CLI saturates them in a shared e-graph, exactly like the
Python engine's shared interning), and ``extract_best`` returns the
cached decoded member for the requested root.  Consequences, all
named rather than silent:

* ``stop="improving"`` and ``policy=`` are refused — the CLI has no
  per-iteration extraction hook and no action ordering;
* ``extract_best`` cannot honour ``overrides``/``bans``/
  ``fusion_epsilon`` (the extraction already happened under the
  engine's ``CostSpec``);
* the Python ``cost_fn`` does not cross the wire — extraction is
  priced by the engine's ``cost`` record (default ``{"kind":
  "size"}``); the mismatch is recorded in ``stats["eqsat"]`` on every
  run.  The verify gate downstream is unaffected: every extracted
  member is e-class-equivalent by construction;
* non-local passes (pairing/lifts), certificates and bounded
  ``error_budget`` extraction are Python-``EGraph`` machinery —
  ``search`` already skips/refuses them for non-``EGraph`` engines.

**Law coverage is reported, never silently narrowed.**  The engine
runs the *portable* law set — the 111 ``LAW_FORMAT = 2`` records of
the full non-demo ``kit-eqsat`` corpus (simplification,
categorical, attention, pointwise_factor, prior_art; the unsound
``reshape_flatten_demo`` matcher demo excluded; snapshot shipped as
``data/eqsat_laws.json``, **generated** from the canonical
``handl-lang/kit-eqsat`` checkout by
``scripts/emit_eqsat_laws.py`` and drift-guarded by
``tests/test_eqsat_laws_corpus.py`` — ADR-0177 §1).
``run(rules)`` intersects the caller's requested rule names with that
set: requested names with no engine record are listed verbatim in
``stats["eqsat"]["laws"]["uncovered"]`` — the 282-law procedural-hook /
conv / attention gap the census measured stays named.

**Term wire format.**  mlopt terms cross as kernel ``Term`` JSON:
``Op`` nodes carry ``dim = Const(1)``, ``source = "Native"``, and the
attrs payload under ``attrs`` encoded with ``attr_to_data`` +
the ``__nonfinite__`` scalar tag (the hnl_codec conventions);
``Var``/``Param`` leaves ride as string-valued ``Const`` records under
the ``__mlopt_leaf__`` prefix carrying name + ``TensorType`` shape, so
the extracted term decodes back to fully-typed leaves.  The kernel
``Term`` is a tree — a shared-subtree (DAG) input is tree-expanded on
the wire, bounded by ``wire_node_budget``; exceeding it is a named
``EqsatError``, not an exponential surprise.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

from mlopt_core.ir import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    attr_from_data,
    attr_to_data,
)

__all__ = [
    "EqsatEngine",
    "EqsatError",
    "EqsatUnavailableError",
    "eqsat_binary",
    "portable_laws",
]


class EqsatError(RuntimeError):
    """The eqsat bridge failed — wire, subprocess, or engine refusal."""


class EqsatUnavailableError(EqsatError):
    """No ``eqsat-run`` binary could be resolved.

    The opt-in is explicit: this error names the miss and the fix
    (build ``engine-eqsat`` or set ``MLOPT_EQSAT_RUN``) rather than
    silently falling back to the Python engine.
    """


#: Reserved prefix marking a leaf ``Const`` as an mlopt ``Var`` /
#: ``Param`` — the kernel ``Term`` has no variable leaf, so typed
#: leaves ride as string consts (the differential driver's proven
#: encoding, extended with the ``TensorType`` shape payload).
_LEAF_PREFIX = "__mlopt_leaf__:"

#: ``{"__nonfinite__": "nan"|"inf"|"-inf"}`` — the hnl_codec /
#: host-tensor wire tag; ``serde_json::Value`` cannot hold NaN/±inf.
_NONFINITE_TAG = "__nonfinite__"
_NONFINITE_VALUES = ("nan", "inf", "-inf")

#: serde defaults pinned by the spike: ``Term::Op.dim`` defaults to
#: ``Const(1)`` and serializes; ``source`` is the ``tensor`` module
#: boundary — the search wire pins ``Native`` (the driver's spelling).
_DIM1 = {"Const": {"value": 1}}


# ---------------------------------------------------------------------------
#  eqsat-run binary resolution
# ---------------------------------------------------------------------------

#: Engine-eqsat checkout locations a ``cargo build`` lands in — the
#: repo-local target dir first, then the user-global shared target.
_ENGINE_EQSAT = Path.home() / "GitHub" / "handl-lang" / "engine-eqsat"
_BIN_CANDIDATES = (
    _ENGINE_EQSAT / "target" / "release" / "eqsat-run",
    _ENGINE_EQSAT / "target" / "debug" / "eqsat-run",
    Path.home() / ".cargo" / "shared-target" / "release" / "eqsat-run",
    Path.home() / ".cargo" / "shared-target" / "debug" / "eqsat-run",
)


def eqsat_binary(binary: str | os.PathLike | None = None) -> Path:
    """Resolve the ``eqsat-run`` executable.

    Order: an explicit ``binary`` argument, the ``MLOPT_EQSAT_RUN``
    environment override, ``eqsat-run`` on ``PATH``, then the
    engine-eqsat checkout target dirs.  Raises
    :class:`EqsatUnavailableError` naming every probe when nothing
    resolves — never a silent fallback.
    """
    if binary is not None:
        p = Path(binary)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise EqsatUnavailableError(
            f"eqsat-run binary {p} is not an executable file"
        )
    env = os.environ.get("MLOPT_EQSAT_RUN")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise EqsatUnavailableError(
            f"MLOPT_EQSAT_RUN={env} is not an executable file"
        )
    on_path = shutil.which("eqsat-run")
    if on_path is not None:
        return Path(on_path)
    for cand in _BIN_CANDIDATES:
        if cand.is_file():
            return cand
    raise EqsatUnavailableError(
        'engine="eqsat" needs the eqsat-run binary: build it with '
        "`cargo build --bin eqsat-run` in the handl-lang/engine-eqsat "
        "checkout, put it on PATH, or set MLOPT_EQSAT_RUN=<path>.  "
        f"Probed: {', '.join(map(str, _BIN_CANDIDATES))}"
    )


# ---------------------------------------------------------------------------
#  The portable law set — generated snapshot of the kit-eqsat corpus
# ---------------------------------------------------------------------------

_LAWS_PATH = Path(__file__).with_name("data") / "eqsat_laws.json"
_LAWS_CACHE: tuple[dict, ...] | None = None


def portable_laws() -> tuple[dict, ...]:
    """Return the shipped portable ``LAW_FORMAT = 2`` law records.

    The snapshot is generated from the canonical ``kit-eqsat``
    corpus (ADR-0177 §1) — all 111 non-demo records across the
    simplification / categorical / attention /
    pointwise_factor / prior_art packs (the original 55 were the
    differential-proven parity subset;
    ``graphopt:bench/drivers/audit/eqsat_differential.py``).  Records
    are dicts in the ``pat`` authoring spelling; pass them to
    ``EqsatEngine(laws=...)`` or a custom JSON file of the same shape.
    """
    global _LAWS_CACHE
    if _LAWS_CACHE is None:
        try:
            data = json.loads(_LAWS_PATH.read_text())
        except OSError as e:
            raise EqsatError(
                f"shipped eqsat law set unreadable: {_LAWS_PATH} ({e})"
            ) from e
        if not isinstance(data, list) or not all(
            isinstance(r, dict) and "name" in r for r in data
        ):
            raise EqsatError(
                f"shipped eqsat law set malformed: {_LAWS_PATH}"
            )
        _LAWS_CACHE = tuple(data)
    return _LAWS_CACHE


def _load_laws(laws: Any) -> tuple[dict, ...]:
    """Normalise the ``laws`` constructor argument to records."""
    if laws is None:
        return portable_laws()
    if isinstance(laws, (str, os.PathLike)):
        path = Path(laws)
        try:
            data = json.loads(path.read_text())
        except OSError as e:
            raise EqsatError(
                f"law file unreadable: {path} ({e})"
            ) from e
        if not isinstance(data, list):
            raise EqsatError(
                f"law file must hold a record array: {path}"
            )
        return _check_records(data, str(path))
    return _check_records(list(laws), "laws argument")


def _check_records(records: list, what: str) -> tuple[dict, ...]:
    for r in records:
        if not isinstance(r, dict) or not isinstance(
            r.get("name"), str
        ):
            raise EqsatError(
                f"{what}: every law record needs a string 'name'"
            )
    return tuple(records)


# ---------------------------------------------------------------------------
#  Non-finite scalars — the __nonfinite__ wire tag
# ---------------------------------------------------------------------------


def _wire_value(v: Any) -> Any:
    """Encode *v* (post-``attr_to_data`` shape) as JSON-safe data."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        if math.isnan(v):
            return {_NONFINITE_TAG: "nan"}
        return {_NONFINITE_TAG: "inf" if v > 0 else "-inf"}
    if isinstance(v, (list, tuple)):
        return [_wire_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _wire_value(x) for k, x in v.items()}
    raise TypeError(f"unserializable wire value: {v!r}")


def _unwire_value(v: Any) -> Any:
    """Decode :func:`_wire_value` output — tags back to floats."""
    if isinstance(v, dict):
        if set(v) == {_NONFINITE_TAG}:
            s = v[_NONFINITE_TAG]
            if s not in _NONFINITE_VALUES:
                raise EqsatError(
                    f"bad {_NONFINITE_TAG} payload {s!r} — expected "
                    "'nan', 'inf' or '-inf'"
                )
            if s == "nan":
                return float("nan")
            return float("inf") if s == "inf" else float("-inf")
        return {k: _unwire_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unwire_value(x) for x in v]
    return v


# ---------------------------------------------------------------------------
#  mlopt term -> kernel Term JSON (the eqsat-run wire)
# ---------------------------------------------------------------------------


def _leaf_payload(t: Var | Param) -> str:
    """Return the ``Const`` string carrying a typed leaf over the wire."""
    shape = None if t.typ.shape is None else list(t.typ.shape)
    kind = "var" if isinstance(t, Var) else "param"
    return _LEAF_PREFIX + json.dumps(
        {"k": kind, "n": t.name, "s": shape}, separators=(",", ":")
    )


def _shape_of(v: Any) -> Any:
    """Return ``v`` in ``TensorType.shape``'s runtime-legal form.

    ``None`` is the wholly-unknown wildcard the annotation does not
    admit — the same cast ``hnl_codec`` uses for the contract.
    """
    if v is None:
        return cast("tuple[int | None, ...]", None)
    return cast("tuple[int | None, ...]", tuple(v))


def _kterm(t: Any, state: list[int]) -> dict:
    """Mlopt term -> ``hnl_engine::Term`` serde JSON.

    ``state`` is ``[emitted, budget]`` shared across the whole term
    list — the kernel ``Term`` is a *tree*, so a DAG input
    tree-expands on the wire; ``wire_node_budget`` bounds that
    expansion and overage is a named :class:`EqsatError`.
    """
    state[0] += 1
    if state[0] > state[1]:
        raise EqsatError(
            f"term wire expansion exceeds wire_node_budget "
            f"({state[1]} nodes) — shared-subtree blowup; the eqsat "
            "wire is a tree until a dag.program record lands"
        )
    if isinstance(t, Op):
        node: dict[str, Any] = {
            "dim": _DIM1,
            "name": t.op,
            "args": [_kterm(a, state) for a in t.args],
            "source": "Native",
        }
        if t.attrs:
            node["attrs"] = {
                k: _wire_value(attr_to_data(v))
                for k, v in t.attrs.items()
            }
        return {"Op": node}
    if isinstance(t, (Var, Param)):
        return {"Const": {"value": _leaf_payload(t)}}
    if isinstance(t, Const):
        return {"Const": {"value": _wire_value(t.value)}}
    raise TypeError(f"not a wireable leaf: {t!r}")


def _terms_wire(terms: list, budget: int) -> list:
    """Encode a term list, bounding TOTAL wire nodes at *budget*."""
    state = [0, budget]
    return [_kterm(t, state) for t in terms]


def _leaf_from_payload(value: str) -> Any:
    """Decode a ``_LEAF_PREFIX`` const back to ``Var``/``Param``."""
    try:
        rec = json.loads(value[len(_LEAF_PREFIX) :])
    except ValueError as e:
        raise EqsatError(f"bad eqsat leaf record {value!r}: {e}") from e
    if not isinstance(rec, dict):
        raise EqsatError(f"bad eqsat leaf record {value!r}")
    try:
        name = rec["n"]
        shape = _shape_of(rec.get("s"))
    except (KeyError, TypeError) as e:
        raise EqsatError(f"bad eqsat leaf record {value!r}: {e}") from e
    if rec.get("k") == "var":
        return Var(name, TensorType(shape))
    if rec.get("k") == "param":
        return Param(name, TensorType(shape))
    raise EqsatError(f"bad eqsat leaf kind in {value!r}")


def _pyterm(j: Any) -> Any:
    """``hnl_engine::Term`` serde JSON -> mlopt term (best-term decode).

    Inverse of :func:`_kterm`: ``attrs`` decodes through
    ``attr_from_data`` (restoring tuple/list spellings), ``lit`` is
    accepted for forward compatibility, and ``__mlopt_leaf__`` consts
    restore the typed ``Var``/``Param`` leaves — name *and* shape.
    """
    if "Op" in j:
        node = j["Op"]
        attrs = node.get("attrs")
        kw: dict[str, Any] = {}
        if isinstance(attrs, dict):
            kw = {
                k: attr_from_data(_unwire_value(v))
                for k, v in attrs.items()
            }
        lit = node.get("lit")
        if isinstance(lit, dict) and "__mlopt_attrs__" in lit:
            kw = {
                k: attr_from_data(_unwire_value(v))
                for k, v in lit["__mlopt_attrs__"].items()
            }
        return Op.make(
            node["name"],
            *[_pyterm(a) for a in node["args"]],
            validate=False,
            **kw,
        )
    if "Const" in j:
        v = j["Const"]["value"]
        if isinstance(v, str) and v.startswith(_LEAF_PREFIX):
            return _leaf_from_payload(v)
        return Const(_unwire_value(v))
    raise EqsatError(f"undecodable term record: {j!r}")


# ---------------------------------------------------------------------------
#  Stop-reason spelling — Rust StopReason -> the Python stats vocabulary
# ---------------------------------------------------------------------------

_STOP_NAMES = {
    "FixedPoint": "fixed_point",
    "Improving": "improving",
    "MaxNodes": "max_nodes",
    "MaxIterations": "max_iterations",
}


class EqsatEngine:
    """The ``engine-eqsat`` saturation engine behind ``engine=``.

    Satisfies :class:`mlopt_core.ports.Engine` over the ``eqsat-run``
    JSON CLI: ``add_term`` records terms, ``run`` performs the one
    saturate+extract subprocess call, ``extract_best`` returns the
    cached decoded member.  ``engine_name = "eqsat"`` is what
    ``stats["engine"]`` records; ``truncation_level = 1`` honestly
    declares that no proof witnesses cross the wire.

    Parameters
    ----------
    binary
        Explicit ``eqsat-run`` path; ``None`` resolves via
        :func:`eqsat_binary` (``MLOPT_EQSAT_RUN`` → ``PATH`` → the
        engine-eqsat checkout target dirs).  Resolution happens at
        construction — a missing binary is an immediate
        :class:`EqsatUnavailableError`, not a mid-search surprise.
    laws
        The ``LAW_FORMAT = 2`` record set: ``None`` → the shipped
        portable snapshot (:func:`portable_laws`); a path to a JSON
        record array; or an iterable of records.  ``run(rules)``
        intersects requested rule names with this set and reports the
        uncovered remainder in ``stats["eqsat"]["laws"]``.
    cost
        The extraction ``CostSpec``: ``None`` → ``{"kind": "size"}``;
        a kind string (``"size"`` / ``"count"``); or a full record
        (``{"kind": "op_table", ...}`` / ``{"kind": "flops", ...}``
        entries).  The Python ``cost_fn`` does not cross the wire —
        every run records the spec actually used.
    options
        Extra ``SaturateOptions`` fields merged under the ``run``
        arguments (``max_iterations`` / ``max_nodes`` /
        ``rule_budgets`` always win — they are the call's contract).
    timeout
        Subprocess wall-clock bound in seconds.
    wire_node_budget
        Cap on total emitted wire nodes across ``add_term``ed terms —
        the tree-encoding bound for shared-subtree inputs.

    """

    engine_name = "eqsat"
    #: Honest scope marker — the CLI carries saturation provenance
    #: (``fired`` / ``rule_fires``) but no level-2 proof witnesses.
    truncation_level = 1

    def __init__(
        self,
        *,
        binary: str | os.PathLike | None = None,
        laws: Any = None,
        cost: Any = None,
        options: dict | None = None,
        timeout: float = 300.0,
        wire_node_budget: int = 200_000,
    ) -> None:
        """Resolve the binary and law/cost configuration eagerly.

        A missing ``eqsat-run`` is an immediate
        :class:`EqsatUnavailableError` — the opt-in fails loudly at
        engine construction, never mid-search and never by silent
        fallback.
        """
        self._binary = eqsat_binary(binary)
        self._laws = _load_laws(laws)
        self._law_names = {r["name"] for r in self._laws}
        self._cost = _cost_spec(cost)
        self._options = dict(options or {})
        self._timeout = timeout
        self._wire_node_budget = wire_node_budget
        self._terms: list = []
        self._results: dict[int, dict[str, Any]] = {}
        self._n_enodes = 0
        self._n_classes = 0
        self._rule_fires: dict[str, int] = {}
        self._run_laws: list[dict] = []
        self._run_options: dict[str, Any] = {}

    # -- internals -----------------------------------------------------

    def _invoke(self, root_eid: int) -> dict[str, Any]:
        """Run the CLI once over every recorded term; return the report."""
        terms_json = _terms_wire(self._terms, self._wire_node_budget)
        spec = {
            "name": "mlopt-search",
            "terms": terms_json,
            "root": root_eid,
            "laws": list(self._run_laws),
            "options": self._run_options,
            "cost": self._cost,
        }
        try:
            proc = subprocess.run(
                [str(self._binary)],
                input=json.dumps(spec),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise EqsatError(
                f"eqsat-run exceeded {self._timeout}s — raise "
                "EqsatEngine(timeout=...) or lower max_iterations/"
                "max_nodes"
            ) from e
        except OSError as e:
            raise EqsatError(f"eqsat-run spawn failed: {e}") from e
        try:
            report = json.loads(proc.stdout)
        except ValueError as e:
            raise EqsatError(
                f"eqsat-run emitted non-JSON (exit {proc.returncode}): "
                f"{proc.stdout[:400]!r} stderr={proc.stderr[:400]!r}"
            ) from e
        if not report.get("ok"):
            raise EqsatError(
                f"eqsat-run refused the spec: {report.get('error')} "
                f"(exit {proc.returncode}, stderr={proc.stderr[:400]!r})"
            )
        return report

    # -- Engine port ----------------------------------------------------

    def add_term(
        self,
        term: Any,
        _memo: dict | None = None,
        provenance: str = "input",
    ) -> int:
        """Record *term*; return its root id (the wire ``terms`` index)."""
        eid = len(self._terms)
        self._terms.append(term)
        return eid

    def find(self, eid: int) -> int:
        """Canonical id — the CLI exposes no union-find; ids are stable."""
        return eid

    def rebuild(self, classes: Any = None) -> bool:
        """No incremental graph exists between runs — nothing to close."""
        return False

    def run(
        self,
        rules: Any,
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
        stop: str = "fixed_point",
        patience: int = 3,
        cost_fn: Any = None,
    ) -> dict[str, Any]:
        """Saturate + extract in one ``eqsat-run`` call.

        ``rules`` selects by name from the engine's law set — the
        requested names with no portable record are reported under
        ``stats["eqsat"]["laws"]["uncovered"]``.  ``stop="improving"``
        and ``policy=`` are refused by name (the CLI has neither a
        per-iteration extraction hook nor an action ordering).
        """
        if stop != "fixed_point":
            raise ValueError(
                f"engine-eqsat does not support stop={stop!r} — the "
                "CLI saturates to its fixed point / caps only "
                "(stop='improving' needs per-iteration extraction)"
            )
        requested = {getattr(r, "name", str(r)) for r in rules}
        applied = [r for r in self._laws if r["name"] in requested]
        uncovered = sorted(requested - self._law_names)
        self._run_laws = applied
        self._run_options = {
            **self._options,
            "max_iterations": max_iterations,
            "max_nodes": max_nodes,
            "rule_budgets": dict(rule_budgets or {}),
        }
        report = self._invoke(root_eid)

        stats = dict(report.get("stats") or {})
        stats["stop"] = _STOP_NAMES.get(
            str(stats.get("stop")), stats.get("stop")
        )
        if stats.get("improved") is None:
            # EGraph.run records "improved" only under
            # stop="improving" — the CLI reports null otherwise.
            stats.pop("improved", None)
        stats["eqsat"] = {
            "binary": str(self._binary),
            "laws": {
                "requested": len(requested),
                "applied": [r["name"] for r in applied],
                "uncovered": uncovered,
            },
            "cost": self._cost,
            "cost_fn": (
                "not portable across the wire — extraction priced "
                "by the CostSpec above"
                if cost_fn is not None
                else None
            ),
            "fired": list(report.get("fired") or []),
        }
        term_json = report.get("term")
        if term_json is None:
            raise EqsatError(
                "eqsat-run extracted no term for the root — the "
                "saturated e-class has no CostSpec-priced member"
            )
        self._results[root_eid] = {
            "term": _pyterm(term_json),
            "cost": report.get("cost"),
        }
        self._n_enodes = int(stats.get("n_enodes") or 0)
        self._n_classes = int(stats.get("n_classes") or 0)
        self._rule_fires = dict(report.get("rule_fires") or {})
        return stats

    def extract_best(
        self,
        eid: int,
        cost_fn: Any,
        **kw: Any,
    ) -> Any:
        """Return the member ``run`` extracted for *eid*'s root.

        The CLI extracts once, under the engine's ``CostSpec`` — the
        ``cost_fn`` argument cannot re-price it, and member steering
        (``overrides`` / ``bans`` / ``fusion_epsilon`` /
        ``_cache_out``) has no wire form.  Non-default values are
        refused by name rather than silently ignored.
        """
        for key in ("overrides", "bans", "_cache_out"):
            if kw.get(key):
                raise TypeError(
                    f"engine-eqsat cannot honour extract_best({key}=…) "
                    "— extraction is fixed at run time under the "
                    "engine's CostSpec"
                )
        if kw.get("fusion_epsilon"):
            raise TypeError(
                "engine-eqsat cannot honour extract_best("
                "fusion_epsilon=…) — the near-tie band is a "
                "Python-extractor feature"
            )
        try:
            return self._results[self.find(eid)]["term"]
        except KeyError as e:
            raise EqsatError(
                f"extract_best({eid}) before run() or for an "
                "unrecorded root — the eqsat engine has no "
                "incremental e-classes to extract from"
            ) from e

    @property
    def n_enodes(self) -> int:
        """E-nodes in the last saturation (0 before ``run``)."""
        return self._n_enodes

    @property
    def n_classes(self) -> int:
        """E-classes in the last saturation (0 before ``run``)."""
        return self._n_classes

    @property
    def rule_fires(self) -> dict[str, int]:
        """``{law_name: merge_count}`` from the last run's merge log."""
        return dict(self._rule_fires)


def _cost_spec(cost: Any) -> dict:
    """Normalise the ``cost`` argument to a ``CostSpec`` record."""
    if cost is None:
        return {"kind": "size"}
    if isinstance(cost, str):
        if cost not in ("count", "size"):
            raise EqsatError(
                f"unknown eqsat cost kind {cost!r} — 'count', 'size', "
                "or a full CostSpec record"
            )
        return {"kind": cost}
    if isinstance(cost, dict) and isinstance(cost.get("kind"), str):
        return dict(cost)
    raise EqsatError(
        f"bad eqsat cost spec {cost!r} — a kind string or a "
        "CostSpec record dict"
    )
