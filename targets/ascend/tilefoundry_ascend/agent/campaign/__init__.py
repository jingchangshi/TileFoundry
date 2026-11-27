"""Campaign runner, marathon harness, and NPU surrogate evaluation exports."""

from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig, CampaignReport, PerfTargetCampaignRunner, ResolvedRegimeConfig,
    extract_regime_snippet, patch_regime_into_module, resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.cli import build_arg_parser, main
from tilefoundry_ascend.agent.campaign.marathon import (
    ExtremeCampaignRunner, FastCreativeAgyBackend, MarathonConfig,
    discover_marathon_environment, ensure_marathon_env_file, run_operator_marathon,
    run_operators_marathon,
)
from tilefoundry_ascend.agent.npu import (
    AnchorEvalMetrics, CandidateSurrogateResult, L0L1CandidateCheck, L3AcceptanceResult,
    L4CannbenchResult, aggregate_surrogate_result, bottleneck_anchor_for_regime,
    classify_l3_observation_failure, compute_ring_burst_k,
    prewarm_surrogate_tilelang_cache, run_l3_full_contract_via_agent_gates,
    run_l4_official_cannbench_via_agent_gates, run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, PerfTargetSpec, RegimeTargetSpec, build_anchor_var_bindings,
    get_default_anchors, get_default_perf_target, get_default_regime_specs,
    get_operator_module_relpath, get_operator_task, lookup_case_dims, lookup_case_metadata,
    resolve_cannbench_repo, resolve_cannbench_tasks_root, resolve_regime_for_case,
)

__all__ = [
    "AnchorCaseSpec", "AnchorEvalMetrics", "CampaignConfig", "CampaignReport",
    "CandidateSurrogateResult", "ExtremeCampaignRunner",
    "FastCreativeAgyBackend", "L0L1CandidateCheck", "L3AcceptanceResult", "L4CannbenchResult",
    "MarathonConfig", "PerfTargetCampaignRunner", "PerfTargetSpec", "RegimeTargetSpec",
    "ResolvedRegimeConfig", "aggregate_surrogate_result", "bottleneck_anchor_for_regime",
    "build_anchor_var_bindings", "build_arg_parser", "classify_l3_observation_failure",
    "compute_ring_burst_k", "discover_marathon_environment", "ensure_marathon_env_file",
    "extract_regime_snippet", "get_default_anchors",
    "get_default_perf_target", "get_default_regime_specs", "get_operator_module_relpath",
    "get_operator_task", "lookup_case_dims", "lookup_case_metadata", "main",
    "patch_regime_into_module", "prewarm_surrogate_tilelang_cache", "resolve_cannbench_repo",
    "resolve_cannbench_tasks_root", "resolve_regime_config", "resolve_regime_for_case",
    "run_l3_full_contract_via_agent_gates", "run_l4_official_cannbench_via_agent_gates",
    "run_operator_marathon", "run_operators_marathon", "run_surrogate_batch_via_agent_gates",
]
