"""The torch-free invariant — mlopt-core imports no backend.

``mlopt-core`` is the declared zero-dependency engine: it must import
neither ``torch``/``numpy`` nor any sibling domain package
(``mlopt_torch``/``mlopt_carriers``/``mlopt_orchestrator``)
on ANY code path.  Two complementary proofs:

* **static** — an AST walk over every ``.py`` under
  ``packages/mlopt-core/src`` rejects any ``import``/``from`` of a
  forbidden root, at module level or inside a function body (the
  ``TYPE_CHECKING`` block included — a stronger reading than the
  runtime contract, and the reason core names no ``torch.Tensor``);
* **runtime** — ``mlopt_core`` is imported in a subprocess with
  ``torch``/``numpy`` blocked by a meta-path finder, and every
  submodule is walked; the import succeeds and the adapter-injected
  hooks degrade cleanly.

The wiring that keeps this true (core is a *sink* for adapter-pushed
state, never a puller) is documented in :mod:`mlopt_core.ops`
(*Backend wiring*) and :mod:`mlopt_core.meta`.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import mlopt_core.meta as meta
import pytest
from mlopt_core.ops import OpTable

_CORE_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "packages"
    / "mlopt-core"
    / "src"
)

#: Import roots core must never name.
_FORBIDDEN = frozenset(
    {
        "torch",
        "numpy",
        "mlopt_torch",
        "mlopt_carriers",
        "mlopt_orchestrator",
    }
)


# ---------------------------------------------------------------------------
#  Static proof — no forbidden import statement anywhere in core
# ---------------------------------------------------------------------------


def _imported_roots(tree: ast.AST):
    """Yield ``(root_name, lineno)`` for every absolute import in *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import — intra-core, fine
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_core_source_has_no_forbidden_imports():
    offenders: list[str] = []
    files = sorted(_CORE_SRC.rglob("*.py"))
    assert files, f"no core sources found under {_CORE_SRC}"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for root, lineno in _imported_roots(tree):
            if root in _FORBIDDEN:
                rel = path.relative_to(_CORE_SRC)
                offenders.append(f"{rel}:{lineno} imports {root}")
    assert offenders == [], (
        "mlopt-core must import no backend/tensor library; found:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
#  Runtime proof — import the whole of mlopt_core with torch blocked
# ---------------------------------------------------------------------------

_BLOCKED_SCRIPT = r"""
import sys

_BLOCKED = ("torch", "numpy")


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError(f"blocked import: {name}")
        return None


sys.meta_path.insert(0, _Blocker())

import importlib
import pkgutil

import mlopt_core
import mlopt_core.meta as meta
import mlopt_core.ops as ops

# Walk EVERY core submodule — none may need a blocked package.
for _info in pkgutil.walk_packages(mlopt_core.__path__, "mlopt_core."):
    importlib.import_module(_info.name)

assert "torch" not in sys.modules, "torch was imported"
assert "numpy" not in sys.modules, "numpy was imported"

# With no adapter imported the core table composes shape rules + attr
# schema but carries no lowering bindings (no backend can lower).
core = ops.OpTable.core()
assert core.torch_bindings == {}, core.torch_bindings
assert core.attr_schemas and core.shape_rules

# The concrete-eval hooks degrade to "skip the numeric check".
assert meta._tensor_env({"x": object()}) == {}
assert meta._eval_allclose(1, 2) is False
try:
    meta._eval_term(object(), {})
except RuntimeError:
    pass
else:
    raise AssertionError("expected RuntimeError without a backend")

print("core-imported-torch-free")
"""


def test_core_imports_without_torch():
    """Import the whole core in a subprocess with torch/numpy blocked."""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"torch-free core import failed:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "core-imported-torch-free" in proc.stdout


# ---------------------------------------------------------------------------
#  The adapter-push wiring — core-owned registries + degraded branches
# ---------------------------------------------------------------------------






def test_meta_hooks_degrade_without_backend(monkeypatch):
    """``_eval_term`` / ``_eval_allclose`` / ``_tensor_env`` skip the
    numeric check when no concrete-eval backend is registered."""
    monkeypatch.setattr(meta, "_concrete_eval", None)
    with pytest.raises(RuntimeError):
        meta._eval_term(object(), {})
    assert meta._eval_allclose(1, 2) is False
    assert meta._tensor_env({"x": object()}) == {}




