#!/usr/bin/env python3
"""P2 microtests: each pinned to a numeric expectation from the independent reference.

The expectation is computed by tools/p2/bf16_reference.py (exact rationals), never by the
implementation under test. The implementation here is torch.bfloat16, the nearest
available stand-in for the TF side; the same comparison applies once TF exposes a
converter. A deviation FAILS the case and the tool exits non-zero.

Run: PYTHONDONTWRITEBYTECODE=1 python tools/p2/microtests.py
"""
from __future__ import annotations

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import struct
import sys

from bf16_reference import (bf16_from_f32_bits, bf16_bits_to_fraction,
                            f32_bits_to_fraction)

# Hand-derived constants: these check the REFERENCE itself, independent of any code path
# that produced them (1.0, 0.5, 2.0, -1.0 are unambiguous).
HAND_DERIVED = [
    ("one", 0x3F800000, 0x3F80),
    ("half", 0x3F000000, 0x3F00),
    ("two", 0x40000000, 0x4000),
    ("neg_one", 0xBF800000, 0xBF80),
]

# Cases whose expectation is computed by the reference; each targets a boundary.
CASES = [
    ("tie_down_even", 0x3F808000),
    ("tie_up_even", 0x3F818000),
    ("round_up", 0x3F80C000),
    ("max_f32_overflows", 0x7F7FFFFF),
    ("neg_max_f32_overflows", 0xFF7FFFFF),
    ("overflow_midpoint_ties_to_inf", 0x7F7F8000),
    ("just_below_overflow_midpoint", 0x7F7F7FFF),
    ("min_f32_subnormal_to_zero", 0x00000001),
    ("positive_zero", 0x00000000),
    ("negative_zero", 0x80000000),
    ("infinity", 0x7F800000),
    ("nan", 0x7FC00000),
    ("bf16_max_finite", 0x7F7F0000),
    ("bf16_min_normal", 0x00800000),
    ("bf16_min_subnormal", 0x00010000),
    ("subnormal_half_ties_to_zero", 0x00008000),
    ("subnormal_1p5_ties_to_even", 0x00018000),
    ("just_below_min_normal_rounds_up", 0x007F8000),
]


def is_nan(bits16: int) -> bool:
    return (bits16 & 0x7F80) == 0x7F80 and (bits16 & 0x7F) != 0


def torch_bf16_bits(bits32: int) -> int:
    import torch
    x = struct.unpack("<f", struct.pack("<I", bits32 & 0xFFFFFFFF))[0]
    t = torch.tensor([x], dtype=torch.float32).to(torch.bfloat16)
    return t.view(torch.int16).item() & 0xFFFF


def main() -> int:
    failures = 0
    print("== reference self-check (hand-derived) ==")
    for name, f32, expected in HAND_DERIVED:
        got = bf16_from_f32_bits(f32)
        ok = got == expected
        failures += 0 if ok else 1
        print("%-28s ref=0x%04X expect=0x%04X %s" % (name, got, expected, "PASS" if ok else "FAIL"))

    print("== reference vs implementation (expectation from reference) ==")
    for name, f32 in CASES:
        ref = bf16_from_f32_bits(f32)
        try:
            impl = torch_bf16_bits(f32)
        except Exception as e:  # noqa: BLE001
            print("%-32s impl raised %s: %s" % (name, type(e).__name__, e))
            failures += 1
            continue
        if is_nan(ref) or is_nan(impl):
            ok = is_nan(ref) and is_nan(impl)
        else:
            ok = ref == impl
        failures += 0 if ok else 1
        print("%-32s ref=0x%04X impl=0x%04X %s" % (name, ref, impl, "PASS" if ok else "FAIL"))

    print("RESULT:", "ALL PASS" if failures == 0 else ("FAILURES=%d" % failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
