"""Ascend scheduling and modulo pipeline constraint solver."""

from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    BufferSnapshot,
    EnvelopeInspectionResult,
    inspect_tilelang_primfunc,
    inspect_tir_envelope,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    AscendPipelineResult,
    bitwise_conflict_pairs,
    extract_tir_loop_problem,
    format_pipe_key,
    masks_conflict,
    pipe_cycles_from_problem,
    read_from_sil,
    solve_ascend_pipeline,
    solve_loop_with_ledger,
    solve_pipeline_with_monotone_backoff,
    task_resource_mask,
    validate_pipeline_spec,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    PiecewiseSchedulePlan,
    RegimePiecewiseSolver,
    RegimePlanResult,
    compute_critical_cube_tile_m,
    recompute_regime_for_live_buffers,
)

__all__ = [
    "AscendPipelineResult",
    "BufferSnapshot",
    "EnvelopeInspectionResult",
    "PiecewiseSchedulePlan",
    "RegimePiecewiseSolver",
    "RegimePlanResult",
    "bitwise_conflict_pairs",
    "compute_critical_cube_tile_m",
    "extract_tir_loop_problem",
    "format_pipe_key",
    "inspect_tilelang_primfunc",
    "inspect_tir_envelope",
    "masks_conflict",
    "pipe_cycles_from_problem",
    "read_from_sil",
    "recompute_regime_for_live_buffers",
    "solve_ascend_pipeline",
    "solve_loop_with_ledger",
    "solve_pipeline_with_monotone_backoff",
    "task_resource_mask",
    "validate_pipeline_spec",
]
