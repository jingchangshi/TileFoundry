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


def build_kernel(T):
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


if __name__ == "__main__":
    raise SystemExit(run_probe("vcvt_probe", "vcvt", "vcvt_bf16_to_f32_part0", build_kernel))
