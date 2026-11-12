#!/usr/bin/env python3
"""03-19 VMULA single-instruction native microtest. DESIGN ONLY (NOT EXECUTED THIS ROUND).

Source anchors: simd_inst.h:404-406 (::vmula), simd.py:635-646 (dst is a READ|WRITE
buffer access, "Fused multiply-add: dst = dst + src0 * src1"), simd_inst.h:651-657
(vmula used inside a Newton iteration, so single-rounding behaviour matters).

FMA constructive discriminator (NOT a random search):
  a=0x3F7FFFF8, b=0x3F800004 => a*b = 1-2^-42, rounded product = 0x3F800000;
  acc=0x34400000 = 1.5 ulp. Unfused (mul then add) hits the exact tie -> ties-to-even
  0x3F800002; a fused single rounding is just BELOW the tie -> 0x3F800001. ONE BIT
  decides whether vmula is an FMA. Also covered: inactive lanes with a distinct dst
  prefill, loop-carried READ|WRITE acc over 3 iterations, FTZ/DAZ on subnormal sources,
  and a constant-0 reducer mutation (a dead accumulator must be killed).
All predictions are CPU models: STATUS UNSOURCED / UNVALIDATED until WBS 03-20.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_base import run_probe  # noqa: E402


def build_kernel(T, repeats=1):
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"),
             B: T.Tensor((64,), "float32"),
             ACCIN: T.Tensor((64,), "float32"),
             OUT: T.Tensor((64,), "float32")):
        with T.Kernel(1) as core:
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            dst = T.alloc_shared((64,), "float32")
            T.copy(A[:], a[:])
            T.copy(B[:], b[:])
            T.copy(ACCIN[:], dst[:])
            with T.SimdVF():
                mask = T.simd.pset(32)
                va = T.simd.vld(a[0])
                vb = T.simd.vld(b[0])
                acc = T.simd.alloc_var(T.float32)
                acc = T.simd.vld(dst[0])
                for _ in T.serial(repeats):
                    T.simd.vmula(acc, va, vb, mask)  # <<< THE SINGLE INSTRUCTION UNDER TEST
                T.simd.vsts(dst[0], acc, mask)
            T.copy(dst[:], OUT[:])
    return main


if __name__ == "__main__":
    raise SystemExit(run_probe("vmula_probe", "vmula", "vmula_fma_fusion", build_kernel))
