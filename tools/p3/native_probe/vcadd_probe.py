#!/usr/bin/env python3
"""03-19 VCADD single-instruction native microtest. DESIGN ONLY (NOT EXECUTED THIS ROUND).

Selected build: tilelang 0.1.15+cuda.git8e2d3c82, simd_inst.h sha 979f4ff7...
Source anchor: simd_inst.h:688-702 (vcadd -> asc_reduce_sum; merge form uses PAT_VL1),
simd.py:957-966. The native REDUCTION TREE and OUTPUT LAYOUT are UNSOURCED / UNVALIDATED:
this probe exists to OBSERVE them, not to assert them.

Constructive discrimination (expected_bits.json):
  [2^24,1,-2^24,0] at lanes 0-3 -> adjacent-first trees 0x00000000, strided/2-chain
  0x3F800000. Random search cannot substitute for this exact f32 cancellation.
  Moving the same four values to lanes 0,1,32,33 flips the signature (lane position).
  A distinct per-lane dst PREFILL reveals PAT_VL1 / broadcast / group output layout and
  inactive-lane preservation (MODE_MERGING vs MODE_ZEROING).
  FTZ and rounding cases are separate vectors in the same file.
Six-call variant: --six-calls uses EXACTLY six vcadd calls plus an EXPLICIT
vdupv(POS_LOWEST); the broadcast is never baked into vcadd (HANDOFF §5.7).

Device preconditions are enforced by device_guard.py: root, TF_DEVICE_AUTH=1,
ASCEND_RT_VISIBLE_DEVICES=0, one non-blocking /home/.tf_device0.lock, npu-smi before+after,  # no-machine-path: allow
bounded timeout, selected-build hash verification. --plan is CPU-only; --execute is
required to touch a device and is NOT RUN in this round.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_base import run_probe  # noqa: E402


def build_kernel(T, six=False):
    @T.prim_func
    def main(IN: T.Tensor((64,), "float32"),
             PREFILL: T.Tensor((64,), "float32"),
             AUX: T.Tensor((64,), "float32"),
             OUT: T.Tensor((64,), "float32")):
        with T.Kernel(1) as core:
            src = T.alloc_shared((64,), "float32")
            dst = T.alloc_shared((64,), "float32")
            T.copy(IN[:], src[:])
            T.copy(PREFILL[:], dst[:])          # distinct sentinel in EVERY lane
            with T.SimdVF():
                mask = T.simd.pset(32)
                v = T.simd.vld(src[0])
                if six:
                    # SIX SOURCE CALLS, then an EXPLICIT broadcast. The instruction
                    # under test is the six-call VCADD SEQUENCE; vdupv is separate.
                    a1 = T.simd.vcadd(v, mask)
                    a2 = T.simd.vcadd(a1, mask)
                    a3 = T.simd.vcadd(a2, mask)
                    a4 = T.simd.vcadd(a3, mask)
                    a5 = T.simd.vcadd(a4, mask)
                    a6 = T.simd.vcadd(a5, mask)
                    r = T.simd.vdupv(a6, mask, pos="POS_LOWEST")
                else:
                    r = T.simd.vcadd(v, mask)   # <<< THE SINGLE INSTRUCTION UNDER TEST
                T.simd.vsts(dst[0], r, mask)    # transport; observe which lanes change
            T.copy(dst[:], OUT[:])
    return main


if __name__ == "__main__":
    raise SystemExit(run_probe("vcadd_probe", "vcadd", "vcadd_cancel_adjacent",
                               build_kernel, lambda T: build_kernel(T, six=True)))
