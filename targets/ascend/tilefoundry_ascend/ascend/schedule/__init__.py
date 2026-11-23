"""Ascend scheduling and modulo pipeline constraint solver."""

from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    BufferSnapshot,
    EnvelopeInspectionResult,
    ReadOnlyAscendPassTap,
    inspect_tilelang_primfunc,
    specialize_primfunc_for_anchor,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    AscendPipelineResult,
    PipelineError,
    bitwise_conflict_pairs,
    format_pipe_key,
    masks_conflict,
    pipe_cycles_from_problem,
    read_from_sil,
    solve_ascend_pipeline,
    solve_pipeline_with_monotone_backoff,
    task_resource_mask,
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
    "PipelineError",
    "ReadOnlyAscendPassTap",
    "RegimePiecewiseSolver",
    "RegimePlanResult",
    "bitwise_conflict_pairs",
    "compute_critical_cube_tile_m",
    "format_pipe_key",
    "inspect_tilelang_primfunc",
    "masks_conflict",
    "pipe_cycles_from_problem",
    "read_from_sil",
    "recompute_regime_for_live_buffers",
    "solve_ascend_pipeline",
    "solve_pipeline_with_monotone_backoff",
    "specialize_primfunc_for_anchor",
    "task_resource_mask",
]
