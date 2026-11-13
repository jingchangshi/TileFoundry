"""Ascend HIR-to-TIR target-scoped candidate pairings."""

from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.visitor_registry.candidates import register_target_candidates
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    AscendGmToL1,
    AscendGmToUb,
    AscendL0cToGm,
    AscendL1ToL0A,
    AscendL1ToL0B,
    AscendMmad,
    AscendRegToUb,
    AscendUbToGm,
    AscendUbToReg,
    AscendVcast,
)

register_target_candidates("ascend", MatMul, (AscendMmad,))
register_target_candidates("ascend", Cast, (AscendVcast,))
register_target_candidates(
    "ascend",
    Reshard,
    (
        AscendGmToUb,
        AscendUbToGm,
        AscendGmToL1,
        AscendL1ToL0A,
        AscendL1ToL0B,
        AscendL0cToGm,
        AscendUbToReg,
        AscendRegToUb,
    ),
)
