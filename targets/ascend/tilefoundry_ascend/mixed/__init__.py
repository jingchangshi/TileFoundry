"""P4-04 / G43: collaborative AIC + AIV MixedKernel modelling.

WHY THIS PACKAGE EXISTS. Every stage before this one described a VECTOR-ONLY program: one
execution domain (`aiv`), one transfer topology (GM <-> UB), and completion edges that never
left that domain. A MixedKernel breaks all three assumptions at once:

  * TWO core domains run concurrently, an AIC (Cube) and its paired AIV(s) (Vector), inside
    ONE logical kernel launch;
  * data crosses a domain boundary twice in each direction -- L0C -> UB for scores, and
    UB -> L1 for the packed probabilities -- and those transfers are performed by the VECTOR
    core reading a CUBE-owned buffer, so neither domain alone owns the buffer being moved;
  * the handoff between the two domains is a CROSS-CORE synchronisation with a different
    mechanism from the intra-core pipe ordering the completion layer already models.

WHAT THIS MODULE DOES, AND WHAT IT DOES NOT. The modelling and the checker below are
STRUCTURAL: they say whether a described MixedKernel is internally consistent and whether it
satisfies the contract. They cannot themselves say whether the hardware produces the right
numbers -- that comes from device execution, recorded in evidence/G43_DEVICE_EVIDENCE.json
(17/17 cases on Ascend950PR, FP16 and BF16, CANN 9.3.0). The two are kept in SEPARATE fields
on the record type so neither can be read as the other, and `is_gate_pass()` requires BOTH.

Note on history: this docstring previously asserted that no device evidence could exist,
because TileLang 0.1.15 appeared unable to compile against the host CANN. That conclusion was
wrong -- it checked only CANN 9.1.0 at /usr/local/Ascend and generalised to the host. A
matching CANN 9.3.0 was installed at /data/pri/9.3.0-weekly. evidence/G43_TOOLCHAIN_BLOCKER.json
is retained as the record of that error rather than deleted. The distinction is preserved in
the record type itself
(MixedEvidence), where structural success and device success are separate fields that cannot
be confused for one another.
"""

from .model import (
    measured_evidence,
    AIC,
    AIV,
    BLOCKED_ENV,
    UNVERIFIED,
    VERIFIED,
    CrossCoreEdge,
    Domain,
    MixedEvidence,
    MixedKernelError,
    MixedKernelSpec,
    MixedRegion,
    MixedTransfer,
    L0C_TO_UB,
    TransferRoute,
    UB_SOFTMAX,
    UB_TO_L1,
    build_mixed_kernel,
    check_mixed_kernel,
)

__all__ = [
    "AIC",
    "AIV",
    "BLOCKED_ENV",
    "UNVERIFIED",
    "VERIFIED",
    "CrossCoreEdge",
    "Domain",
    "MixedEvidence",
    "MixedKernelError",
    "MixedKernelSpec",
    "MixedRegion",
    "MixedTransfer",
    "L0C_TO_UB",
    "TransferRoute",
    "UB_SOFTMAX",
    "UB_TO_L1",
    "build_mixed_kernel",
    "measured_evidence",
    "check_mixed_kernel",
]