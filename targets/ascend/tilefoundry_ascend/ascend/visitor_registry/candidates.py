"""Ascend HIR-to-TIR target-scoped candidate pairings."""

from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.softplus import Softplus as HirSoftplus
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.conv2d import Conv2D as HirConv2D
from tilefoundry.ir.hir.nn.gelu import Gelu as HirGelu
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm as HirLayerNorm
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm as HirRMSNorm
from tilefoundry.ir.hir.nn.rope import RoPE as HirRoPE
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid as HirSigmoid
from tilefoundry.ir.hir.nn.silu import Silu as HirSilu
from tilefoundry.ir.hir.nn.softmax import SoftMax as HirSoftMax
from tilefoundry.ir.hir.nn.tanh import Tanh as HirTanh
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.argmax import ArgMax as HirArgMax
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.index_add import IndexAdd as HirIndexAdd
from tilefoundry.ir.hir.tensor.index_copy import IndexCopy as HirIndexCopy
from tilefoundry.ir.hir.tensor.index_select import IndexSelect as HirIndexSelect
from tilefoundry.ir.hir.tensor.quant import Quant as HirQuant
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.topk import TopK as HirTopK
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
    AscendMxQuant,
    AscendRegToUb,
    AscendSortTopK,
    AscendUbToGm,
    AscendUbToGmAtomic,
    AscendUbToReg,
    AscendVadd,
    AscendVbinary,
    AscendVcast,
    AscendVclamp,
    AscendVexp,
    AscendVgather,
    AscendVldDist,
    AscendVmul,
    AscendVmula,
    AscendVreduce,
    AscendVreduceMax,
    AscendVreduceSum,
    AscendVrsqrt,
    AscendVstsDist,
    AscendVsub,
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

# Extended Ascend 950 HIR-to-TIR target-scoped candidates.
register_target_candidates("ascend", HirConv2D, (AscendMmad,))
register_target_candidates("ascend", HirArgMax, (AscendSortTopK,))
register_target_candidates("ascend", HirTopK, (AscendSortTopK,))
register_target_candidates("ascend", HirIndexSelect, (AscendVgather,))
register_target_candidates("ascend", HirIndexAdd, (AscendUbToGmAtomic, AscendL0cToGmAtomic))
register_target_candidates("ascend", HirIndexCopy, (AscendUbToGm, AscendL0cToGm, AscendVstsDist))
register_target_candidates("ascend", HirQuant, (AscendMxQuant,))
register_target_candidates("ascend", HirRoPE, (AscendVmula, AscendVmul, AscendVadd, AscendVsub))
register_target_candidates("ascend", HirRMSNorm, (AscendVreduce, AscendVrsqrt, AscendVmul))
register_target_candidates(
    "ascend",
    HirSoftMax,
    (AscendVexp, AscendVreduceMax, AscendVreduceSum),
)
register_target_candidates(
    "ascend",
    HirLayerNorm,
    (AscendVreduce, AscendVrsqrt, AscendVmul, AscendVadd),
)
register_target_candidates("ascend", HirGelu, (AscendVexp,))
register_target_candidates("ascend", HirSilu, (AscendVexp,))
register_target_candidates("ascend", HirSigmoid, (AscendVexp,))
register_target_candidates("ascend", HirTanh, (AscendVexp,))
register_target_candidates("ascend", HirSoftplus, (AscendVexp,))
