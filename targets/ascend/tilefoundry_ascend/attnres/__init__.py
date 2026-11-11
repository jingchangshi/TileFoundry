"""P5-01/G50: attention_residual as ONE logical Ascend kernel.

WHAT THIS PACKAGE IS. The operator fuses five steps -- RMSNorm along D, a pseudo-query score
along D, a depth softmax along N, a weighted sum over the ORIGINAL V, and an alpha residual
blend -- into a single kernel launch. The package models that schedule, emits the kernel, and
records what was measured.

WHY IT IS NOT PART OF THE MIXEDKERNEL PACKAGE. attention_residual uses ONE execution domain
(AIV). The MixedKernel package exists to model TWO, so re-using it here would add an empty Cube
side purely to satisfy a shape, which the backend spec forbids.

THE STRUCTURAL / DEVICE SPLIT IS PRESERVED. attention_residual is AIV-only, so every dependency
between its tasks is pipe ordering within one core and there is no cross-core flag to emit. The
model and checker can say whether a described kernel is internally consistent; only device
execution can say whether the hardware produces the right numbers, so the evidence record keeps
those in separate fields and requires BOTH for a gate pass.
"""
from .model import (
    AttentionResidualError,
    AttentionResidualSpec,
    BLOCKED_ONLINE,
    BLOCKED_ENV,
    FACTORED_GW,
    FAMILIES,
    FP32State,
    LITERAL_FP32,
    NOT_RUN,
    ONLINE,
    REDUCTIONS,
    RESIDENT_N,
    ReductionAxis,
    SCORE_MODES,
    UNVERIFIED,
    VERIFIED,
    AttentionEvidence,
    build_attention_residual,
    measured_attention_evidence,
)
from .hir import (
    AIV,
    BLOCK_COMPUTE,
    Edge,
    FINALIZE,
    LOAD_V,
    LOAD_X,
    MTE2,
    MTE3,
    STORE_Y,
    ScheduledHIR,
    Task,
    VECTOR,
    build_scheduled_hir,
    check_scheduled_hir,
)
from .twin import (
    AUTO_SCHEDULE_FLAG,
    AttnResTwin,
    ConformanceReport,
    FAILED,
    MATCHED,
    build_attnres_twin,
    check_attnres_conformance,
    parse_attnres_twin,
)

__all__ = [
    "AIV",
    "AUTO_SCHEDULE_FLAG",
    "AttentionEvidence",
    "AttentionResidualError",
    "AttentionResidualSpec",
    "AttnResTwin",
    "BLOCKED_ENV",
    "BLOCKED_ONLINE",
    "BLOCK_COMPUTE",
    "ConformanceReport",
    "Edge",
    "FACTORED_GW",
    "FAMILIES",
    "FAILED",
    "FINALIZE",
    "FP32State",
    "LITERAL_FP32",
    "LOAD_V",
    "LOAD_X",
    "MATCHED",
    "MTE2",
    "MTE3",
    "NOT_RUN",
    "ONLINE",
    "REDUCTIONS",
    "RESIDENT_N",
    "ReductionAxis",
    "SCORE_MODES",
    "STORE_Y",
    "ScheduledHIR",
    "Task",
    "UNVERIFIED",
    "VECTOR",
    "VERIFIED",
    "build_attention_residual",
    "build_attnres_twin",
    "build_scheduled_hir",
    "check_attnres_conformance",
    "check_scheduled_hir",
    "measured_attention_evidence",
    "parse_attnres_twin",
]
