"""Substrate-pure tests ported from ``graphopt:tests/test_public_api.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from __future__ import annotations

import importlib

import re

import pytest


_PUBLIC_SURFACE: dict[str, tuple[str, ...]] = {
    "mlopt_core": ("API_VERSION",),
    "mlopt_core.ir": (
        "IR",
        "Op",
        "Var",
        "Param",
        "Const",
        "TensorType",
        "op_repr",
        "op_repr_dag",
        "term_to_data",
        "term_from_data",
    ),
    "mlopt_core.egraph": (
        "EGraph",
        "ENode",
        "EClass",
        "UnionFind",
        "Rewrite",
        "Certificate",
        "CertStep",
        "ProofEdge",
        "CertificateVerificationError",
        "cert_to_data",
        "cert_from_data",
        "verify_certificate",
    ),
    "mlopt_core.cost": (
        "dag_cost",
        "count_cost",
        "flops_cost",
        "param_bytes_cost",
        "param_bytes_cost_for",
        "backend_cost",
        "executor_cost_for",
        "fused_cost_for",
        "lowering_aware_cost_for",
        "depth_cost",
        "depth_cost_for",
        "roofline_cost",
        "roofline_cost_for",
        "launch_aware_cost",
        "fusion_regions",
        "fusion_member_key",
        "CostModel",
        "LOWERINGS",
    ),
    "mlopt_core.ports": (
        "Source",
        "Sink",
        "Capabilities",
        "Engine",
        "Composer",
        "Executor",
        "ExecutorSpec",
        "Runner",
        "Meter",
        "Verifier",
        "VerifyResult",
        "CostFn",
        "Criterion",
        "Policy",
        "Strategy",
        "TaskMetric",
        "PerformanceModel",
        "Profiler",
        "OpRegistry",
        "RuleSetLike",
        "RuleLike",
        "RuleSetProvider",
        "Binding",
        "TimingResult",
        "signature_conforms",
    ),
    "mlopt_core.laws": (
        "RuleSet",
        "preset",
        "DEFAULT",
        "FULL",
        "SYMMETRY",
        "ALL_RULES",
        "tags",
        "provenance",
    ),
    "mlopt_core.laws.provenance": (
        "MACHINE_SHIPPED",
        "MACHINE_STORED",
        "MACHINE_LAWS",
        "law_provenance",
    ),
    "mlopt_orchestrator": (
        "Optimizer",
        "search",
        "lower",
        "SearchResult",
        "LowerResult",
        "Backend",
        "discover_alternatives",
        "DEFAULT_RULES",
        "OptimizationResourceError",
        "Monolithic",
        "Compositional",
        "Autotuned",
        "IdentityRunner",
        "ChainedRunner",
        "runner_candidate",
        "Criterion",
        "Criteria",
        "Blend",
        "criteria_cost",
        "peak_bytes_cost",
        "LatencyCriterion",
        "MemoryCriterion",
        "FlopsCriterion",
        "DepthCriterion",
        "CompiledCriterion",
        "PredictedCriterion",
        "delivered_cost_for",
        "structural_key",
    ),
    "gamekit.engine": (
        "Board",
        "Report",
        "Trajectory",
        "run_episode",
        "evaluate",
        "train_policy",
    ),
    "gamekit.players": (
        "PlayerSpec",
        "LinearPolicy",
        "DomainSpec",
        "LearnedPlayer",
        "PlayHist",
        "Bias",
        "StateBroadcast",
        "PlayRates",
        "OpOneHot",
        "ActionFeats",
        "RelFeats",
        "compose_featurizer",
        "register_part",
        "register_legal",
        "register_rank",
        "spec_from_record",
        "PARTS",
        "LEGALS",
        "RANKS",
        "registry",
    ),
    "gamekit.players.registry": (
        "PlayerSpec",
        "BOARDS",
        "EPISODE_BOARDS",
        "names",
        "register_player",
        "spec",
        "specs",
        "for_board",
        "module",
        "make",
    ),
    "gamekit.scoring": (
        "AXES",
        "DIRECTIONS",
        "ScoreGate",
        "ScoringSpec",
        "GateVerdict",
        "ScoreVerdict",
        "score",
    ),
    "mlopt_discovery.verifier": ("verify_law", "LawResult"),
    "mlopt_discovery.play": (
        "DOMAINS",
        "Domain",
        "play",
        "spec_domain",
    ),
}

# The contract's domain seats (``mlopt_torch``, ``mlopt_carriers``,
# ``mlopt_discovery``, ``mlopt_onnx``, ``gamekit``) are checked in
# graphopt where those packages live; this repo holds the substrate
# slice of the same frozen table.
_PUBLIC_SURFACE = {
    m: names
    for m, names in _PUBLIC_SURFACE.items()
    if m.split(".")[0] in {"mlopt_core", "mlopt_orchestrator"}
}


class TestFrozenSurface:
    @pytest.mark.parametrize("mod_name", sorted(_PUBLIC_SURFACE))
    def test_module_imports(self, mod_name):
        mod = importlib.import_module(mod_name)
        assert mod is not None

    @pytest.mark.parametrize(
        "mod_name,name",
        [
            (m, n)
            for m, names in _PUBLIC_SURFACE.items()
            for n in names
        ],
    )
    def test_frozen_name_resolves(self, mod_name, name):
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, name), (
            f"CONTRACT BREAK: {mod_name}.{name} — a frozen public name "
            "no longer resolves (docs/public_api.md)"
        )

    @pytest.mark.parametrize("mod_name", sorted(_PUBLIC_SURFACE))
    def test_frozen_names_hold_all_seats(self, mod_name):
        """Where a module declares ``__all__``, frozen names sit in it."""
        mod = importlib.import_module(mod_name)
        declared = getattr(mod, "__all__", None)
        if declared is None:
            pytest.skip(f"{mod_name} declares no __all__")
        missing = set(_PUBLIC_SURFACE[mod_name]) - set(declared)
        assert not missing, (
            f"CONTRACT BREAK: {mod_name} frozen names missing from "
            f"__all__: {sorted(missing)}"
        )

class TestApiVersion:
    def test_version_shape(self):
        from mlopt_core import API_VERSION

        assert isinstance(API_VERSION, str)
        assert re.fullmatch(r"\d+\.\d+\.\d+", API_VERSION), (
            f"API_VERSION must be semver-shaped, got {API_VERSION!r}"
        )

    def test_version_is_exported(self):
        import mlopt_core

        assert "API_VERSION" in mlopt_core.__all__
