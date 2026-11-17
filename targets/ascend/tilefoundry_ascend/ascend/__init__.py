"""Ascend target and compiler registrations for TileFoundry."""

from tilefoundry_ascend.ascend import ir, passes, schedule, visitor_registry
from tilefoundry_ascend.ascend.passes import (
    HirCandidateOpEntry,
    HirRooflineSummary,
    HirTirNavigationPacket,
    compile_hir_navigation_packet,
)
from tilefoundry_ascend.ascend.target import (
    PURE_SIMDVF_UBUF_FLAGS,
    Ascend950PR,
    AscendSpaceFacts,
    AscendTarget,
    DaVinci950,
    NodeResourceMask,
)

__all__ = [
    "PURE_SIMDVF_UBUF_FLAGS",
    "Ascend950PR",
    "AscendSpaceFacts",
    "AscendTarget",
    "DaVinci950",
    "HirCandidateOpEntry",
    "HirRooflineSummary",
    "HirTirNavigationPacket",
    "NodeResourceMask",
    "compile_hir_navigation_packet",
    "ir",
    "passes",
    "schedule",
    "visitor_registry",
]
