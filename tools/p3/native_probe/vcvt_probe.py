#!/usr/bin/env python3
"""03-19 VCVT single-instruction native microtest. DESIGN ONLY (NOT EXECUTED THIS ROUND).

Source anchors: simd_inst.h:291-294 (::vcvt), simd.py:1209-1215 (round/part/mode),
simd.py:1270-1274 (mask granularity = min(src,target) = 16 bit for bf16<->f32).
HANDOFF §5.4/§5.5: UNPK_B16 belongs to VLD and is NOT a numeric cast; the carrier is a
DISTINCT register class; SimdVF element_dtypes stays (f32,); vcvt uses a DECLARED 16-bit
mask produced by a pset in the SAME region instance and lexically BEFORE the consumer;
part=0 is explicit and part=1 / 32-bit mask / plain-f32 source are NAMED refusals.

Vectors (expected_bits.json): bf16->f32 widening with inf/NaN/-0/subnormal, ROUND_R
tie boundary, FTZ/DAZ separation, inactive-lane preservation with a distinct dst
prefill, and an explicit part=1 lane-mapping contrast. All predictions are CPU models:
STATUS UNSOURCED / UNVALIDATED until WBS 03-20 observes the selected build.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_base import run_probe  # noqa: E402


def build_kernel_widening(T):
    """bf16 -> f32. UNPK_B16 belongs to VLD and is transport, NOT a cast."""
    @T.prim_func
    def main(IN: T.Tensor((64,), "bfloat16"),
             PREFILL: T.Tensor((64,), "float32"),
             AUX: T.Tensor((64,), "float32"),
             OUT: T.Tensor((64,), "float32")):
        with T.Kernel(1) as core:
            src = T.alloc_shared((64,), "bfloat16")
            dst = T.alloc_shared((64,), "float32")
            T.copy(IN[:], src[:])
            T.copy(PREFILL[:], dst[:])
            with T.SimdVF():
                mask16 = T.simd.pset(16)        # DECLARED 16-bit granularity
                carrier = T.simd.vld(src[0], dist="UNPK_B16")   # transport, NOT a cast
                # <<< THE SINGLE INSTRUCTION UNDER TEST (part=0, ROUND_R)
                r = T.simd.vcvt(carrier, "float32", mask16, part=0, round="ROUND_R")
                T.simd.vsts(dst[0], r, mask16)
            T.copy(dst[:], OUT[:])
    return main


def build_kernel_narrowing(T):
    """f32 -> bf16. This direction had NO kernel at all before.

    The harness sent every vcvt case to the widening kernel, so a narrowing case
    could not have run even with its tensors built correctly: the signature is the
    other way round. vld uses dist="NORM" because the source is plain f32 and
    UNPK_B16 is a bf16-lane transport, not a numeric cast.

    part=0 (PART_EVEN) is explicit. Measured on device 1 with 64 distinct inputs,
    input lane i lands at output slot 2*i and the odd slots keep the prefill, i.e.
    one 16-bit result per 32-bit destination word. That is observed, not assumed.

    KNOWN OPEN GAP: the mask is a hardcoded full pset(16). A case that declares a
    PARTIAL mask (vcvt_inactive_lane_preservation declares lanes 0..31) is therefore
    NOT exercised: every lane is written and the prefill is overwritten. That case
    establishes nothing about MODE_MERGING vs MODE_ZEROING and must not be cited as
    inactive-lane evidence until the mask is plumbed in from the case data.
    """
    @T.prim_func
    def main(IN: T.Tensor((64,), "float32"),
             PREFILL: T.Tensor((64,), "bfloat16"),
             AUX: T.Tensor((64,), "float32"),
             OUT: T.Tensor((64,), "bfloat16")):
        with T.Kernel(1) as core:
            src = T.alloc_shared((64,), "float32")
            dst = T.alloc_shared((64,), "bfloat16")
            T.copy(IN[:], src[:])
            T.copy(PREFILL[:], dst[:])
            with T.SimdVF():
                mask16 = T.simd.pset(16)        # 16 bit = min(src, target)
                carrier = T.simd.vld(src[0], dist="NORM")   # plain f32 transport
                # <<< THE SINGLE INSTRUCTION UNDER TEST (part=0, ROUND_R)
                r = T.simd.vcvt(carrier, "bfloat16", mask16, part=0, round="ROUND_R")
                T.simd.vsts(dst[0], r, mask16)
            T.copy(dst[:], OUT[:])
    return main


def build_kernel(T, case=None):
    """Choose the kernel by the case DECLARED direction, never by probe identity.

    Before this existed there was ONE kernel - widening - and every vcvt case was
    compiled through it. A narrowing case could not run even with correct tensors,
    because the tensor signature itself was the other way round.
    """
    if case is None:
        return build_kernel_widening(T)
    return (build_kernel_widening(T) if case.get("source_dtype") == "bfloat16"
            else build_kernel_narrowing(T))


#: Cases whose `mask` field is NOT plumbed into the kernel. Their partial-mask
#: claims are UNEXERCISED and must not be cited as evidence. An explicit list so a
#: future test can fail loudly instead of trusting the case name.
MASK_NOT_PLUMBED = ("vcvt_inactive_lane_preservation",)


if __name__ == "__main__":
    raise SystemExit(run_probe("vcvt_probe", "vcvt", "vcvt_bf16_to_f32_part0", build_kernel))
