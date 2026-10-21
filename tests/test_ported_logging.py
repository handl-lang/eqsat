"""Substrate-pure tests ported from ``graphopt:tests/test_logging.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import ast

import logging

from pathlib import Path

from mlopt_core.egraph import EGraph

from mlopt_core.ir import Op, Param, TensorType, Var

from mlopt_core.laws import ASSOC_LINEAR_BIAS


_REPO = Path(__file__).resolve().parents[1]

_OWNED = sorted(
    str(p.relative_to(_REPO))
    for p in (_REPO / "packages").rglob("*.py")
    # the owned set is the phase-3c logging-migration files; optimize.py
    # keeps its deliberate verbose-mode stdout prints
    if "src" in p.parts and p.name != "optimize.py"
)

def _print_call_lines(path: Path) -> list[int]:
    """Line numbers of ``print(...)`` *calls* in *path*.

    AST-based: docstring examples (``regime.py``) and prose cannot
    register, only real call nodes.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

def _small_graph() -> tuple[EGraph, int]:
    x = Var("x", TensorType((4, 8)))
    A = Param("A", TensorType((16, 8)))
    B = Param("B", TensorType((8, 16)))
    b1 = Param("b1", TensorType((16,)))
    b2 = Param("b2", TensorType((8,)))
    src = Op.make("linear", Op.make("linear", x, A, b1), B, b2)
    eg = EGraph()
    return eg, eg.add_term(src)

def test_no_print_calls_in_owned_files():
    offenders = []
    for rel in _OWNED:
        path = _REPO / rel
        lines = path.read_text().splitlines()
        for lineno in _print_call_lines(path):
            # A kept print must justify itself on its own line.
            if "stdout-compat" not in lines[lineno - 1]:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"print() calls outside compat: {offenders}"

def test_saturation_logs_iteration(caplog):
    eg, eid = _small_graph()
    with caplog.at_level(logging.INFO):
        stats = eg.run(
            [ASSOC_LINEAR_BIAS], eid, max_iterations=8, max_nodes=2000
        )
    assert stats["iterations"] >= 1
    hits = [
        r
        for r in caplog.records
        if r.name.startswith("mlopt")
        and r.levelno == logging.INFO
        and "saturation at iteration" in r.getMessage()
    ]
    assert hits, "no INFO saturation-iteration line logged"

def test_max_nodes_stop_logs_warning(caplog):
    eg, eid = _small_graph()
    with caplog.at_level(logging.WARNING):
        eg.run([ASSOC_LINEAR_BIAS], eid, max_nodes=1)
    assert any(
        r.name.startswith("mlopt")
        and r.levelno == logging.WARNING
        and "max_nodes" in r.getMessage()
        for r in caplog.records
    )
