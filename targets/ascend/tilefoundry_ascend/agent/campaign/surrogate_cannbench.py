"""Thin pure-L4 re-export facade over tilefoundry_ascend.agent.npu."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from tilefoundry_ascend.agent import npu as _npu
from tilefoundry_ascend.agent.npu import (
    AGENT_GATES_ADAPTER, AGENT_GATES_DEVICE_FAST_GATE, AGENT_GATES_L3_PROMOTION_GATE,
    AGENT_GATES_L4_CANNBENCH_GATE, AGENT_GATES_L4_CANNBENCH_PREP_GATE,
    DEFAULT_AGENT_GATES_BACKOFF_SEC, DEFAULT_AGENT_GATES_MAX_ATTEMPTS,
    DEFAULT_L3_PROMOTION_WAIT_SEC, DEFAULT_L4_CANNBENCH_WAIT_SEC,
    FAILURE_KIND_CANDIDATE, FAILURE_KIND_INFRASTRUCTURE, FAST_REQUEST_ENV,
    FAST_REQUEST_SCHEMA, FAST_RESPONSE_SCHEMA, FAST_SOURCE_ROOT_ENV,
    FAST_STATUS_CANDIDATE_FAILURE, FAST_STATUS_INFRASTRUCTURE_ERROR, FAST_STATUS_OK,
    FAST_STATUS_REQUEST_ERROR, _LLM_API_KEY_ENV, _NON_AGENT_GATES_ENV_KEYS,
    AgentGatesDeviceFaultError, AnchorEvalMetrics, CandidateSurrogateResult,
    L0L1CandidateCheck, L0L1RoundResults, L2BatchResults, L3AcceptanceResult,
    L4CannbenchResult, LocalHardwareProbePort, MicroProbeSources, OccupancyExhaustedError,
    PrewarmTileLangResult, _V2GateExecution, _agent_gates_evidence_root,
    _agent_gates_source_override, _agent_gates_workspace_name,
    _call_order_candidates_by_idle_preflight, _clean_agent_gates_subprocess_env,
    _default_surrogate_cache_dir, _default_surrogate_tl_cache_dir, _format_env_line,
    _git_common_dir, _is_transient_blocked_env, _normalize_device_pool,
    _normalized_module_ast_signature, _prewarm_micro_probe_cache, _probe_host_card_idle,
    _probe_host_card_load, _resolve_agent_gates_bin, _resolve_env_file_path,
    _run_agent_gates_cmd, _run_agent_gates_v2_cmd, _run_v2_gate,
    _source_identity_matches, _submit_fast_gate, _validate_llm_api_key_env,
    _write_temp_env_file, aggregate_surrogate_result, allocate_rotating_ring_params,
    bottleneck_anchor_for_regime, build_anchor_metrics, classify_l3_observation_failure,
    collect_bottleneck_msprof_via_agent_gates, collect_p2_deep_dive_via_agent_gates,
    compute_adaptive_poison_pill_timeout, compute_ring_burst_k, ensure_cann_include_cpath,
    ensure_npu_compare_shim_dir, estimate_params_io_bytes,
    l3_failure_result_from_observation, lookup_case_static_spec,
    measure_npu_ring_burst_stats,
    measure_npu_ring_burst_us, measure_paired_interleaved_micro_burst,
    order_candidates_by_idle_preflight, prewarm_anchor_reference_cache,
    prewarm_surrogate_tilelang_cache, prewarm_surrogate_tilelang_cache_with_stats,
    run_l0_static_precheck,
    run_l3_full_contract_via_agent_gates, run_l4_official_cannbench_via_agent_gates,
    run_micro_probe_via_agent_gates, run_msprof_detail,
    run_multi_operator_surrogate_batch_via_agent_gates, run_poison_pill_subprocess,
    run_surrogate_batch_via_agent_gates, select_idle_campaign_device, wait_for_npu_quiescent,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import extract_ordered_tensor_shapes
from tilefoundry_ascend.agent.orchestrator.stage_b_gate import (
    run_l1_compile_check, validate_prewarm_manifest_v1,
)
from tilefoundry_ascend.agent.specs.anchor_specs import get_operator_task, resolve_cannbench_repo

_npu._bind_facade_globals(globals())


__all__ = [
    "AGENT_GATES_ADAPTER", "AGENT_GATES_DEVICE_FAST_GATE", "AGENT_GATES_L3_PROMOTION_GATE",
    "AGENT_GATES_L4_CANNBENCH_GATE", "AGENT_GATES_L4_CANNBENCH_PREP_GATE",
    "AgentGatesDeviceFaultError", "AnchorEvalMetrics", "CandidateSurrogateResult",
    "DEFAULT_AGENT_GATES_BACKOFF_SEC", "DEFAULT_AGENT_GATES_MAX_ATTEMPTS",
    "DEFAULT_L3_PROMOTION_WAIT_SEC", "DEFAULT_L4_CANNBENCH_WAIT_SEC",
    "FAILURE_KIND_CANDIDATE", "FAILURE_KIND_INFRASTRUCTURE", "FAST_REQUEST_ENV",
    "FAST_REQUEST_SCHEMA", "FAST_RESPONSE_SCHEMA", "FAST_SOURCE_ROOT_ENV",
    "FAST_STATUS_CANDIDATE_FAILURE", "FAST_STATUS_INFRASTRUCTURE_ERROR", "FAST_STATUS_OK",
    "FAST_STATUS_REQUEST_ERROR", "L0L1CandidateCheck", "L0L1RoundResults", "L2BatchResults",
    "L3AcceptanceResult", "L4CannbenchResult", "LocalHardwareProbePort", "MicroProbeSources",
    "OccupancyExhaustedError", "PrewarmTileLangResult", "aggregate_surrogate_result",
    "allocate_rotating_ring_params", "bottleneck_anchor_for_regime", "build_anchor_metrics",
    "classify_l3_observation_failure", "collect_bottleneck_msprof_via_agent_gates",
    "collect_p2_deep_dive_via_agent_gates", "compute_adaptive_poison_pill_timeout",
    "compute_ring_burst_k", "ensure_cann_include_cpath", "ensure_npu_compare_shim_dir",
    "estimate_params_io_bytes", "extract_ordered_tensor_shapes", "get_operator_task",
    "l3_failure_result_from_observation", "lookup_case_static_spec",
    "measure_npu_ring_burst_stats", "measure_npu_ring_burst_us",
    "measure_paired_interleaved_micro_burst", "order_candidates_by_idle_preflight",
    "prewarm_anchor_reference_cache", "prewarm_surrogate_tilelang_cache",
    "prewarm_surrogate_tilelang_cache_with_stats", "resolve_cannbench_repo",
    "run_l0_static_precheck", "run_l1_compile_check",
    "run_l3_full_contract_via_agent_gates", "run_l4_official_cannbench_via_agent_gates",
    "run_micro_probe_via_agent_gates", "run_msprof_detail",
    "run_multi_operator_surrogate_batch_via_agent_gates", "run_poison_pill_subprocess",
    "run_surrogate_batch_via_agent_gates", "select_idle_campaign_device",
    "validate_prewarm_manifest_v1", "wait_for_npu_quiescent",
]
