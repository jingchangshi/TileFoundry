"""Ascend compiler passes."""

from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirCandidateOpEntry,
    HirRooflineSummary,
    HirTirNavigationPacket,
    compile_hir_navigation_packet,
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
    "HirCandidateOpEntry",
    "HirRooflineSummary",
    "HirTirNavigationPacket",
    "ObjectivePhysicalLedger",
    "ResolveAscendPipelines",
    "ScopeWaterline",
    "ValidateAscendPipelines",
    "build_objective_physical_ledger",
    "build_scope_waterline",
    "compile_hir_navigation_packet",
    "compute_dominant_pipe",
]
