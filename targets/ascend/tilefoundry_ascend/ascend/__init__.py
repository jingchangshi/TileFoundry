"""Ascend target and compiler registrations for TileFoundry."""

from tilefoundry_ascend.ascend import ir, passes, schedule, visitor_registry
from tilefoundry_ascend.ascend.target import Ascend950PR, AscendSpaceFacts, AscendTarget, DaVinci950

__all__ = [
    "Ascend950PR",
    "AscendSpaceFacts",
    "AscendTarget",
    "DaVinci950",
    "ir",
    "passes",
    "schedule",
    "visitor_registry",
]
