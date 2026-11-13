"""Ascend execution mode declarations."""

from __future__ import annotations

from tilefoundry.ir.core.execution import ExecutionKind, register_execution_kind
from tilefoundry.ir.types import DType, StorageKind

SIMTVF_KIND = ExecutionKind(
    name="ascend.simtvf",
    surface="SimtVF",
    register_storages=(StorageKind.RMEM, StorageKind.CARRIER),
    mask_width_bits=32,
    conversion_mask_width_bits=16,
    lane_count=None,
    element_dtypes=(DType.f32,),
    required_topologies=("aiv",),
)

SimtVF = register_execution_kind(SIMTVF_KIND, dialect="T")

__all__ = ["SIMTVF_KIND", "SimtVF"]
