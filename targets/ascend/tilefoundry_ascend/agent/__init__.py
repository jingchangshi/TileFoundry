"""Neuro-symbolic agent subsystem for TileFoundry Ascend kernel synthesis."""

from __future__ import annotations

from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    BayesianRuleTracker,
    RulePosterior,
    UnsafePredicateError,
    build_predicate_context,
    default_seed_rules_path,
    load_seed_rules,
    safe_eval_predicate,
    save_rules_yaml,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator import (
    AgyCliLLMBackend,
    BeamCandidate,
    EvolutionRoundRecord,
    EvolutionTrajectory,
    GateFunnel,
    GenerationRequest,
    MockLLMBackend,
    PortfolioRunner,
    StaticDynamicResidual,
    TileConfigCandidate,
    TwoLaneBeamSearch,
    prescreen_tile_configs,
    run_e2e_evolution,
)
from tilefoundry_ascend.agent.router import (
    RouteDecision,
    StructuralFeatures,
    extract_structural_features,
    route_execution_track,
)

__all__ = [
    "AgyCliLLMBackend",
    "AscendCausalRule",
    "BayesianRuleTracker",
    "BeamCandidate",
    "EvolutionRoundRecord",
    "EvolutionTrajectory",
    "GateFunnel",
    "GenerationRequest",
    "MockLLMBackend",
    "PortfolioRunner",
    "RouteDecision",
    "RulePosterior",
    "StaticDynamicResidual",
    "StructuralFeatures",
    "TileConfigCandidate",
    "TwoLaneBeamSearch",
    "UnsafePredicateError",
    "build_predicate_context",
    "default_seed_rules_path",
    "extract_structural_features",
    "load_seed_rules",
    "prescreen_tile_configs",
    "route_execution_track",
    "run_e2e_evolution",
    "safe_eval_predicate",
    "save_rules_yaml",
    "validate_rule_predicate",
]
