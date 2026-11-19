"""Ascend compiler passes."""

from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    HirRooflineSummary,
    compile_hir_blueprint,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
    build_objective_physical_ledger,
    build_scope_waterline,
    compute_dominant_pipe,
)
from tilefoundry_ascend.ascend.passes.pipeline import (
    ResolveAscendPipelines,
    ValidateAscendPipelines,
)

__all__ = [
    "HirBlueprintPacket",
    "HirRooflineSummary",
    "ObjectivePhysicalLedger",
    "ResolveAscendPipelines",
    "ScopeWaterline",
    "ValidateAscendPipelines",
    "build_objective_physical_ledger",
    "build_scope_waterline",
    "compile_hir_blueprint",
    "compute_dominant_pipe",
]
