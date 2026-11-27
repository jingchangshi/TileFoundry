"""Canonical L2 static, Stage-B CPU compile, and Track-B slicing gate exports."""

from __future__ import annotations

from tilefoundry_ascend.agent.gates.cpp_slice_protocol import (
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    select_active_trackb_kernel,
)
from tilefoundry_ascend.agent.gates.gate_funnel import (
    FunnelEvaluation,
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
    evaluate_ascendc_candidate,
    extract_ordered_tensor_shapes,
    prescreen_dsl_tile_kwargs,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
    run_stage_c_agent_gates,
    synthesize_pybind_extension_wrapper,
)
from tilefoundry_ascend.agent.gates.stage_b_gate import (
    ModuleSelectorPlanExtractor,
    StageBCompileGate,
    build_post_selector_regime_plan,
    run_l1_compile_check,
    validate_prewarm_manifest_v1,
)

__all__ = [
    "FunnelEvaluation",
    "GateFunnel",
    "ModuleSelectorPlanExtractor",
    "StageAResult",
    "StageBCompileGate",
    "StageBResult",
    "StageCResult",
    "apply_scoped_trackb_patch",
    "build_post_selector_regime_plan",
    "build_trackb_context_view",
    "evaluate_ascendc_candidate",
    "extract_ordered_tensor_shapes",
    "prescreen_dsl_tile_kwargs",
    "run_l1_compile_check",
    "run_stage_a_static",
    "run_stage_b_ascendc_compile",
    "run_stage_b_compile",
    "run_stage_c_agent_gates",
    "select_active_trackb_kernel",
    "synthesize_pybind_extension_wrapper",
    "validate_prewarm_manifest_v1",
]
