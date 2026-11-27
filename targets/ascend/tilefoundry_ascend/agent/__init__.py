"""Neuro-symbolic agent subsystem for TileFoundry Ascend kernel synthesis."""

from __future__ import annotations

from tilefoundry_ascend.agent.campaign import (
    CampaignConfig,
    CampaignReport,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.contracts import (
    CampaignRoundRecord,
    EvolutionRoundRecord,
    EvolutionTrajectory,
    WorkerStepRecord,
)
from tilefoundry_ascend.agent.flywheel import (
    FlywheelDistillationReport,
    FlywheelDistiller,
)
from tilefoundry_ascend.agent.gates import (
    GateFunnel,
    extract_ordered_tensor_shapes,
    prescreen_dsl_tile_kwargs,
)
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
from tilefoundry_ascend.agent.npu import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
    run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.orchestrator import (
    AgyCliLLMBackend,
    BeamCandidate,
    GenerationRequest,
    MockLLMBackend,
    PortfolioRunner,
    StreamWatchdogConfig,
    TileConfigCandidate,
    TwoLaneBeamSearch,
    build_progressive_llm_constraints,
    format_blueprint_delta_section,
    format_hardware_and_sil_layer,
    format_prescreen_summary,
    format_target_dialect_layer,
    generate_aligned_neighborhood,
    prescreen_tile_configs,
)
from tilefoundry_ascend.agent.router import (
    CapabilityGate,
    RouteDecision,
    route_execution_track,
)
from tilefoundry_ascend.agent.specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
    StaticDynamicResidual,
    append_dynamic_anchor,
    get_default_anchors,
    get_default_perf_target,
)

__all__ = [
    "AgyCliLLMBackend",
    "AnchorCaseSpec",
    "AnchorEvalMetrics",
    "AscendCausalRule",
    "BayesianRuleTracker",
    "BeamCandidate",
    "CampaignConfig",
    "CampaignReport",
    "CampaignRoundRecord",
    "CandidateSurrogateResult",
    "CapabilityGate",
    "EvolutionRoundRecord",
    "EvolutionTrajectory",
    "FlywheelDistillationReport",
    "FlywheelDistiller",
    "GateFunnel",
    "GenerationRequest",
    "L3AcceptanceResult",
    "L4CannbenchResult",
    "MockLLMBackend",
    "PerfTargetCampaignRunner",
    "PerfTargetSpec",
    "PortfolioRunner",
    "RegimeTargetSpec",
    "RouteDecision",
    "RulePosterior",
    "StaticDynamicResidual",
    "StreamWatchdogConfig",
    "TileConfigCandidate",
    "TwoLaneBeamSearch",
    "UnsafePredicateError",
    "WorkerStepRecord",
    "append_dynamic_anchor",
    "build_predicate_context",
    "build_progressive_llm_constraints",
    "extract_ordered_tensor_shapes",
    "format_blueprint_delta_section",
    "format_hardware_and_sil_layer",
    "format_target_dialect_layer",
    "default_seed_rules_path",
    "format_prescreen_summary",
    "generate_aligned_neighborhood",
    "get_default_anchors",
    "get_default_perf_target",
    "load_seed_rules",
    "prescreen_dsl_tile_kwargs",
    "prescreen_tile_configs",
    "route_execution_track",
    "run_surrogate_batch_via_agent_gates",
    "safe_eval_predicate",
    "save_rules_yaml",
    "validate_rule_predicate",
]
