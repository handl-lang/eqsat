"""``engine-eqsat`` bridge — the ``"eqsat"`` engine (ADR-0178 WP3+).

Since the WP4 parity sweep v4 GO, ``"eqsat"`` is the *default*
search interpretation (``engine=None`` resolves to it);
``engine="python-egraph"`` is the explicit opt-out.

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
  member is e-class-equivalent by construction.  Because the
  ``CostSpec`` axis can disagree with the caller's ``cost_fn`` (a
  size-minimal member may be *worse* on the requested axis — the
  WP4-F3 ``select_mul`` regression), ``extract_best`` restores the
  invariant the Python extractor gets for free — the input term is a
  member of its own e-class, so a min-cost extraction can never
  deliver worse-than-input *under the pricing it was asked for* —
  by pricing the extracted member and the recorded input term with
  the ``cost_fn`` it is handed and falling back to the input term
  when the extraction is strictly worse.  The fallback is recorded
  on the cached result (``fell_back_to_input``); it can only ever
  *lose* ground vs the Python arm's pick, never invent a worse
  delivery;
* non-local passes (pairing/lifts), certificates and bounded
  ``error_budget`` extraction are Python-``EGraph`` machinery —
  ``search`` already skips/refuses them for non-``EGraph`` engines.

**Law coverage is reported, never silently narrowed.**  The engine
runs the *portable* law set — the 524 ``LAW_FORMAT = 2`` records of
the full non-demo ``kit-eqsat`` corpus (simplification,
categorical, prior_art, attention, pointwise_factor, batch_norm,
conv_folds, scan, layout, prim_folds; the unsound
``reshape_flatten_demo`` matcher demo excluded; snapshot shipped as
``data/eqsat_laws.json``, **generated** from the canonical
``handl-lang/kit-eqsat`` checkout by
``scripts/emit_eqsat_laws.py`` and drift-guarded by
``tests/test_eqsat_laws_corpus.py`` — ADR-0177 §1).
``run(rules)`` intersects the caller's requested rule names with that
set: requested names with no engine record are listed verbatim in
``stats["eqsat"]["laws"]["uncovered"]`` — the residual
procedural-hook / conv gap the census measured stays named.  Records
whose ``synop*`` ops have no seated lowering in this process (the
60 ``prim_folds`` minted records — their denotations are
discovery-side torch bindings, ``mlopt_core.ops.ambient_bindings`` is
the census) are withheld per run and listed under
``stats["eqsat"]["laws"]["unbound"]``: the Rust ``CostSpec`` cannot
price a denotation it cannot see, so shipping them would let
extraction pick an unlowerable member — the eqsat-side analogue of
the Python engine's ``+inf`` ``supported_ops`` bound.

**Term wire format.**  mlopt terms cross as kernel ``Term`` JSON
records — either a tree (``Op`` nodes carry ``dim = Const(1)``,
``source = "Native"``, and the attrs payload under ``attrs`` encoded
with ``attr_to_data`` + the ``__nonfinite__`` scalar tag, the hnl_codec
conventions; ``Var``/``Param`` leaves ride as canonical ``Arr``
records carrying name + ``TensorType`` shape) or, since the
``dag.program`` landing, the
ADR-0173 shared-subtree envelope: ``{"Opaque": {"__tag__":
"dag.program", "nodes": [...], "root": n}}`` — a post-order node table
where every distinct node appears once and an ``Op``'s ``dim``/``args``
are *indices* into ``nodes``.  ``eqsat-run`` decodes the table
bottom-up into e-class ids (``engine-eqsat/src/dag.rs``,
``saturate_json``), so hash-consed DAG inputs cross the wire in
linear size — the tree spelling tree-expanded shared subtrees per
occurrence and blew ``wire_node_budget`` (or ``serde_json``'s
recursion limit) on every real model.  The budget now bounds
*distinct emitted nodes*; exceeding it is a named ``EqsatError``,
not an exponential surprise.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mlopt_core import term_json
from mlopt_core.term import Op

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

    ``"eqsat"`` is the default engine (the WP4 parity sweep v4 GO),
    so this error names the miss *and* the fix — build
    ``engine-eqsat`` or set ``MLOPT_EQSAT_RUN`` — plus the explicit
    opt-out (``engine="python-egraph"``) rather than silently
    falling back to the Python engine.
    """


#: The term codec moved to ``mlopt_core.term_json`` (ADR-0178 D5
#: option B, Step E-3) — the ONE kernel-Term-JSON codec shared with
#: the ``kit_discovery.hnl_codec`` successor.  This bridge emits the
#: canonical ``leaf="arr"`` profile (``{"Arr": {"value":
#: {"__mlopt_leaf__": …}}}`` records) — safe since engine-eqsat
#: ``dd4815e`` made the ``arr`` enode spell its leaves back; decode
#: accepts both spellings permanently.


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
        'engine="eqsat" (the default) needs the eqsat-run binary: '
        "build it with `cargo build --bin eqsat-run` in the "
        "handl-lang/engine-eqsat checkout, put it on PATH, or set "
        "MLOPT_EQSAT_RUN=<path>.  To use the pure-Python reference "
        'engine instead, pass engine="python-egraph" explicitly.  '
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
    corpus (ADR-0177 §1) — all 524 non-demo records across the
    simplification / categorical / prior_art / attention /
    pointwise_factor / batch_norm / conv_folds / scan / layout /
    prim_folds packs
    (the original 55 were the
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
#  Minted-primitive (``synop*``) binding census
# ---------------------------------------------------------------------------
#
# The ``prim_folds`` pack ships the 60 minted ``prim:synop_*`` /
# ``prim:synop2_*`` / ``prim:synop3_*`` records.  They are *pattern*
# data: a ``synop_*`` call is a name to the substrate, and the op's
# denotation is the torch binding the discovery side seats
# (``kit_discovery.shipped_primitives.register()`` writes
# ``_IR_TO_TORCH[synop] = fn``).  In a process where that seat never
# ran — every plain ``engine="eqsat"`` search — the Rust engine's
# ``size`` CostSpec has no way to know the op is unlowerable, so
# saturation can extract a ``synop_*`` member and ``lower`` dies on
# ``ValueError: No torch binding for op 'synop_…'``.  The Python
# engine never hits this: ``backend_cost`` prices any op outside the
# sink's ``supported_ops`` at ``+inf``, and ``supported_ops`` IS the
# seated-binding key set.  The bridge restores the same bound by
# withholding records whose ``synop*`` ops are not seated in the
# ambient binding table (:func:`mlopt_core.ops.ambient_bindings`) —
# the exact census a default ``OpTable.full()`` sink exposes — and by
# refusing an extracted member that still names an unseated ``synop*``
# op.  Both checks run per ``run``/``extract_best`` call, so a
# ``register()`` that lands after engine construction is honoured.

#: Minted-primitive op names: ``synop_<sha>``, ``synop2_<sha>``,
#: ``synop3_<sha>`` (``mlopt_core.laws.minted``'s naming family).
_SYNOP_OP_RE = re.compile(r"synop\d*_")


def _seated_binding_names() -> frozenset[str]:
    """Op names with a seated lowering in this process's ambient table.

    ``None`` ambient (no backend adapter imported) seats nothing —
    the same empty ``supported_ops`` a binding-less environment
    gives the Python engine's ``backend_cost`` bound.
    """
    from mlopt_core.ops import ambient_bindings

    ambient = ambient_bindings()
    if ambient is None:
        return frozenset()
    # Iterating the dict yields materialized keys only — the same
    # census ``Sink.supported_ops`` (``frozenset(torch_bindings)``)
    # computes; lazy carrier resolution is never triggered.
    return frozenset(ambient)


def _record_synops(record: dict) -> frozenset[str]:
    """``synop*`` op names a law record's ``lhs``/``rhs`` mention."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            op = node.get("op")
            if isinstance(op, str) and _SYNOP_OP_RE.match(op):
                found.add(op)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(record.get("lhs"))
    walk(record.get("rhs"))
    return frozenset(found)


def _pyterm_synops(term: Any) -> frozenset[str]:
    """``synop*`` op names an mlopt term uses (DAG-safe)."""
    found: set[str] = set()
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Op):
            if _SYNOP_OP_RE.match(node.op):
                found.add(node.op)
            for a in node.args:
                walk(a)

    walk(term)
    return frozenset(found)


# ---------------------------------------------------------------------------
#  mlopt term -> kernel Term JSON — delegated to mlopt_core.term_json
# ---------------------------------------------------------------------------
#
# The codec quartet that lived here (``_kterm`` / ``_kdag`` /
# ``_pyterm`` / ``_pyterm_dag`` plus the ``__nonfinite__`` /
# ``__mlopt_leaf__`` helpers — ~330 lines) is now
# ``mlopt_core.term_json``: one codec for the whole stack, reachable
# because it sits in ``mlopt-core`` (the only dep this package has —
# ``kit_discovery`` was never importable here without a cycle).
# ``TermJsonError`` surfaces as ``EqsatError`` — the same refusal
# channel the quartet raised.  The private names below stay as thin
# delegates: the test-suite pins them.


def _kterm(t: Any, state: list[int]) -> dict:
    """Mlopt term -> ``hnl_engine::Term`` serde JSON (tree spelling).

    Delegate over ``term_json.term_to_json(dag="tree", leaf="arr")``
    — ``state`` is the shared ``[emitted,
    budget]`` cell exactly as the deleted walker used it (the tree
    spelling expands sharing; the budget bounds the expansion).
    """
    try:
        return term_json.term_to_json(
            t, dag="tree", leaf="arr", budget=state
        )
    except term_json.TermJsonError as e:
        raise EqsatError(str(e)) from e


def _kdag(t: Any, state: list[int]) -> dict:
    """Mlopt term -> ``dag.program`` record (kept for private callers)."""
    try:
        return term_json.term_to_json(
            t, dag=True, leaf="arr", budget=state
        )
    except term_json.TermJsonError as e:
        raise EqsatError(str(e)) from e


def _terms_wire(terms: list, budget: int) -> list:
    """Encode a term list as ``dag.program`` records.

    Delegate over ``term_json.terms_to_json(leaf="arr")`` —
    bounds TOTAL distinct wire nodes at *budget*, shared across the
    whole list, exactly as the deleted ``_kdag`` loop did.
    """
    try:
        return term_json.terms_to_json(terms, leaf="arr", budget=budget)
    except term_json.TermJsonError as e:
        raise EqsatError(str(e)) from e


def _pyterm(j: Any) -> Any:
    """``hnl_engine::Term`` serde JSON -> mlopt term (best-term decode).

    Delegate over ``term_json.term_from_json(validate=False)`` — the
    bridge decodes law-extracted terms without ``Op.make`` attr
    validation, the contract the deleted quartet kept.  Both leaf
    spellings and both ``dim`` spellings decode; ``source`` is
    ignored (law patterns pin ``"Native"`` on the way OUT — decode
    direction carries no source semantics for mlopt terms).
    """
    try:
        return term_json.term_from_json(j, validate=False)
    except term_json.TermJsonError as e:
        raise EqsatError(str(e)) from e


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
        the ``dag.program`` node-table bound (distinct nodes, so
        shared-subtree inputs encode linearly).

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

    def _select_laws(
        self, requested: set[str]
    ) -> tuple[list[dict], list[str], list[str]]:
        """Split *requested* into applied / uncovered / unbound.

        ``applied`` are the records that go on the wire; ``uncovered``
        are requested names with no record at all (the residual
        procedural-hook gap); ``unbound`` are requested names whose
        record exists but names a ``synop*`` op with no seated
        lowering in this process — withheld because the Rust
        ``CostSpec`` cannot price a denotation it cannot see, exactly
        the bound ``backend_cost`` gives the Python engine through
        ``supported_ops``.  The census is read per call so a
        ``shipped_primitives.register()`` that lands after engine
        construction is honoured.
        """
        seated = _seated_binding_names()
        eligible = [
            r for r in self._laws if _record_synops(r) <= seated
        ]
        eligible_names = {r["name"] for r in eligible}
        applied = [r for r in eligible if r["name"] in requested]
        uncovered = sorted(requested - self._law_names)
        unbound = sorted((requested & self._law_names) - eligible_names)
        return applied, uncovered, unbound

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
        applied, uncovered, unbound = self._select_laws(requested)
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
                # Requested records withheld because their ``synop*``
                # denotations are not seated in this process — the
                # eqsat-side analogue of the Python engine's
                # ``+inf``-priced ``supported_ops`` bound.
                "unbound": unbound,
            },
            "cost": self._cost,
            "cost_fn": (
                "not portable across the wire — extraction priced "
                "by the CostSpec above; extract_best falls back to "
                "the input term when the pick prices worse under "
                "this cost_fn (never-worse-than-input)"
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

        What *does* cross back is the never-worse-than-input floor:
        the Python extractor's guarantee — min-cost under the
        caller's ``cost_fn``, with the input term inside the root
        e-class — has no ``CostSpec`` analogue, so when a ``cost_fn``
        is supplied the extracted member and the recorded input term
        are both priced under it (one shared ``dag_cost`` memo) and
        the input term is returned when the extraction is strictly
        worse.  The fallback is recorded on the cached result as
        ``fell_back_to_input`` and can only surrender a *better*
        pick the ``CostSpec`` happened to find, never deliver a
        worse-than-input term.
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
        eid = self.find(eid)
        try:
            rec = self._results[eid]
        except KeyError as e:
            raise EqsatError(
                f"extract_best({eid}) before run() or for an "
                "unrecorded root — the eqsat engine has no "
                "incremental e-classes to extract from"
            ) from e
        term = rec["term"]
        if eid < len(self._terms):
            # Hard capability floor, checked before pricing: an
            # extracted member naming a ``synop*`` op with no seated
            # lowering can never ``lower`` in this process (the
            # ``ValueError: No torch binding`` crash) — the same bound
            # ``_select_laws`` applies to the records, enforced here
            # for extra law sets (``laws=``) or a census that changed
            # between ``run`` and ``extract_best``.
            unbound = _pyterm_synops(term) - _seated_binding_names()
            if unbound:
                rec["fell_back_to_input"] = True
                rec["unbound_synops"] = sorted(unbound)
                term = self._terms[eid]
        if cost_fn is not None and eid < len(self._terms):
            # Never-worse-than-input: the input term IS a member of
            # the saturated root e-class, so under the caller's own
            # pricing it is the floor a faithful extraction cannot
            # go below.  ``CostSpec`` extraction has no such floor
            # across axes — price both sides and keep the input when
            # the pick is strictly worse (WP4 F3).
            from mlopt_core.cost import dag_cost

            memo: dict = {}
            in_cost = dag_cost(self._terms[eid], cost_fn, memo=memo)
            out_cost = dag_cost(term, cost_fn, memo=memo)
            # ``not <=`` not ``>``: a term pricing ``+inf`` under
            # ``backend_cost`` (an op the sink cannot lower — e.g. a
            # minted ``synop_*`` whose binding was never seated)
            # decomposes to ``inf - inf = NaN`` inside ``dag_cost``,
            # and ``NaN > in_cost`` is False — the naive comparison
            # would wave the unbindable term through.  NaN is never
            # a better price, so anything not strictly <= the input
            # falls back.
            if not out_cost <= in_cost:
                rec["fell_back_to_input"] = True
                rec["input_cost"] = in_cost
                rec["extracted_cost"] = out_cost
                term = self._terms[eid]
        return term

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
