"""Ascend HIR-to-TIR target-scoped candidate pairings."""

from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.where import Where as HirWhere
from tilefoundry.visitor_registry.candidates import register_target_candidates
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    AscendDualCopyL0cToUb,
    AscendDualCopyUbToL1,
    AscendGmToL1,
    AscendGmToL1Nz,
    AscendGmToUb,
    AscendGmToUbPad,
    AscendL0cToGm,
    AscendL0cToGmAtomic,
    AscendL1ToL0A,
    AscendL1ToL0B,
    AscendMmad,
    AscendRegToUb,
    AscendUbToGm,
    AscendUbToGmAtomic,
    AscendUbToReg,
    AscendVbinary,
    AscendVcast,
    AscendVclamp,
    AscendVldDist,
    AscendVreduce,
    AscendVstsDist,
    AscendVunary,
    AscendVwhere,
)

register_target_candidates("ascend", MatMul, (AscendMmad,))
register_target_candidates("ascend", Cast, (AscendVcast,))
register_target_candidates("ascend", HirBinary, (AscendVbinary,))
register_target_candidates("ascend", HirUnary, (AscendVunary,))
register_target_candidates("ascend", HirReduce, (AscendVreduce,))
register_target_candidates("ascend", HirWhere, (AscendVwhere,))
register_target_candidates("ascend", HirClamp, (AscendVclamp,))
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
        AscendDualCopyL0cToUb,
        AscendDualCopyUbToL1,
        AscendUbToReg,
        AscendRegToUb,
        AscendGmToL1Nz,
        AscendGmToUbPad,
        AscendVldDist,
        AscendVstsDist,
        AscendUbToGmAtomic,
        AscendL0cToGmAtomic,
    ),
)
