"""Declare the Ascend950PR Reg-SIMD vector-fetch execution region.

The AIV Reg-SIMD pipeline runs in a mode, not on a participant set: within one
vector-fetch region the vector registers and the predicate register are the
working set, and neither has a meaning outside it. This module states that mode
once, so the parser, the analyses and the diagnostics all read one declaration
instead of restating the boundary.

Scope: f32 lanes. pset(32) sets the element width of the predicate and is not a
lane count; the compute domain is the declared 64 lanes of the AIV vector unit.
Instructions that name a different element width address lanes this mode does
not reduce to, so the width is stated here rather than inferred from a dtype.
"""

from __future__ import annotations

from tilefoundry.ir.core.execution import ExecutionKind, register_execution_kind
from tilefoundry.ir.types.storage import StorageKind

SIMD_VF_KIND = "ascend.simdvf"
"""Registry name of the AIV vector-fetch execution mode."""

SIMD_VF_SURFACE = "SimdVF"
"""Surface spelling an authored function writes to enter the mode."""

SIMD_VF_LANES = 64
"""Lanes the AIV vector unit computes over in one issue."""

SIMD_VF_ELEMENT_WIDTH_BITS = 32
"""Element width pset(32) selects for the f32 subset; not a lane count."""


def _declare() -> ExecutionKind:
    return register_execution_kind(
        ExecutionKind(
            name=SIMD_VF_KIND,
            surface=SIMD_VF_SURFACE,
            # The vector and predicate registers are the mode private working set.
            register_storages=(StorageKind.RMEM,),
            mask_width_bits=SIMD_VF_ELEMENT_WIDTH_BITS,
            required_topologies=("aiv",),
            # One scheduling unit per region: a nested region would state a
            # second mode where the first is still in force.
            nestable=False,
        )
    )


SIMD_VF = _declare()
"""The registered surface marker a function enters the mode through."""

SIMD_VF_DECLARATION = SIMD_VF.execution_kind
"""The declaration the marker resolves to."""

__all__ = [
    "SIMD_VF",
    "SIMD_VF_DECLARATION",
    "SIMD_VF_ELEMENT_WIDTH_BITS",
    "SIMD_VF_KIND",
    "SIMD_VF_LANES",
    "SIMD_VF_SURFACE",
]