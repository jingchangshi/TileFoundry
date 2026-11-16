"""Ascend target and compiler registrations for TileFoundry."""

from tilefoundry_ascend.ascend import ir, passes, schedule, visitor_registry
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
    "NodeResourceMask",
    "ir",
    "passes",
    "schedule",
    "visitor_registry",
]
