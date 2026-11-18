"""Ascend target hardware and fact projections."""

from tilefoundry_ascend.ascend.target.architecture import DaVinci950
from tilefoundry_ascend.ascend.target.device import Ascend950PR, detect_runtime_core_counts
from tilefoundry_ascend.ascend.target.facts import (
    PURE_SIMDVF_UBUF_FLAGS,
    AscendSpaceFacts,
    cube_macs_per_cycle,
)
from tilefoundry_ascend.ascend.target.spec import (
    ASCEND_950DT_ID,
    ASCEND_950PR_ID,
    DAVINCI_950_ID,
    NodeResourceMask,
)
from tilefoundry_ascend.ascend.target.target import AscendTarget

__all__ = [
    "ASCEND_950DT_ID",
    "ASCEND_950PR_ID",
    "DAVINCI_950_ID",
    "PURE_SIMDVF_UBUF_FLAGS",
    "Ascend950PR",
    "AscendSpaceFacts",
    "AscendTarget",
    "DaVinci950",
    "NodeResourceMask",
    "cube_macs_per_cycle",
    "detect_runtime_core_counts",
]
