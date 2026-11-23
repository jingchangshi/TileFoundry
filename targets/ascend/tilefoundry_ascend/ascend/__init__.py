"""Ascend target and compiler registrations for TileFoundry."""

from tilefoundry_ascend.ascend import passes, schedule
from tilefoundry_ascend.ascend.passes import (
    HirBlueprintPacket,
    HirRooflineSummary,
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
    "HirBlueprintPacket",
    "HirRooflineSummary",
    "NodeResourceMask",
    "passes",
    "schedule",
]
