"""Ascend target hardware and fact projections."""

from tilefoundry_ascend.ascend.target.architecture import DaVinci950
from tilefoundry_ascend.ascend.target.device import Ascend950PR
from tilefoundry_ascend.ascend.target.facts import AscendSpaceFacts
from tilefoundry_ascend.ascend.target.target import AscendTarget

__all__ = ["Ascend950PR", "AscendSpaceFacts", "AscendTarget", "DaVinci950"]
