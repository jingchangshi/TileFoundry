#!/usr/bin/env python3
"""Two-independent-paths cross-check: tools/p2 reference vs the target-side bf16 module.

Read-only: this file imports tilefoundry_ascend.bf16 and never writes to it. A divergence
is REPORTED with the exact input, both bit patterns, and which side is right plus the
reason; it is not patched here.

Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/p2:targets/ascend python tools/p2/cross_check_impl.py
Exit: 0 agreement (encode/decode/bf16-fold), 1 divergence, 2 error.
"""
from __future__ import annotations

import os
import random
import struct
import sys
from fractions import Fraction

import bf16_reference as ref

try:
    from tilefoundry_ascend import bf16 as impl
except Exception as e:  # noqa: BLE001
    print("ERROR: cannot import tilefoundry_ascend.bf16:", type(e).__name__, e)
    raise SystemExit(2)


def f32_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def is_nan(bits16: int) -> bool:
    return (bits16 & 0x7F80) == 0x7F80 and (bits16 & 0x7F) != 0


def phase_encode(n: int = 20000) -> dict:
    """Compare exact-f32 input encoding. Input is always exactly representable in f32."""
    rng = random.Random(0xC0FFEE)
    exps = (0, 1, 2, 125, 126, 127, 128, 253, 254, 255)
    checked = 0
    for _ in range(n):
        if rng.random() < 0.5:
            bits = (rng.getrandbits(1) << 31) | (rng.choice(exps) << 23) | rng.getrandbits(23)
        else:
            bits = rng.getrandbits(32)
        r = ref.bf16_from_f32_bits(bits)
        i = impl.encode(f32_to_float(bits))
        checked += 1
        if is_nan(r) or is_nan(i):
            if is_nan(r) and is_nan(i):
                continue
            return {"phase": "encode", "input_f32_bits": "0x%08X" % bits,
                    "reference": "0x%04X" % r, "implementation": "0x%04X" % i,
                    "reason": "NaN class disagreement"}
        if r != i:
            return {"phase": "encode", "input_f32_bits": "0x%08X" % bits,
                    "input_value": str(ref.f32_bits_to_fraction(bits)),
                    "reference": "0x%04X" % r, "implementation": "0x%04X" % i,
                    "reason": "both are exact-rational RNE; reference rounds the f32 value directly"}
    return {"phase": "encode", "checked": checked}


def phase_decode() -> dict:
    """Compare every one of the 65536 bf16 patterns."""
    for bits in range(0x10000):
        r = ref.bf16_bits_to_fraction(bits)
        i = impl.decode(bits)
        if r is None:
            if not (i != i or i in (float("inf"), float("-inf"))):
                return {"phase": "decode", "input_bf16": "0x%04X" % bits,
                        "reference": "inf/nan", "implementation": repr(i)}
            continue
        if Fraction(i) != r:
            return {"phase": "decode", "input_bf16": "0x%04X" % bits,
                    "reference": str(r), "implementation": repr(i),
                    "reason": "reference is the exact rational value"}
    return {"phase": "decode", "checked": 0x10000}


def _finite_bf16(rng) -> int:
    while True:
        value = rng.getrandbits(16)
        if ((value >> 7) & 0xFF) != 0xFF:  # exclude inf/NaN patterns
            return value


def _bf16_linear_fold(values) -> int:
    """Independent bf16-step linear fold, using only the reference.

    Stops (returning the inf bits) if a step overflows the bf16 range.
    """
    acc = Fraction(0)
    for v in values:
        bits = ref.fraction_to_bf16_bits(acc + ref.bf16_bits_to_fraction(v))
        acc = ref.bf16_bits_to_fraction(bits)
        if acc is None:          # overflowed to inf
            return bits
    return ref.fraction_to_bf16_bits(acc)


def phase_reduction() -> dict:
    """His ORDER_LINEAR is a bf16-after-every-step fold; check it against our own fold."""
    rng = random.Random(0xADD)
    values = [_finite_bf16(rng) for _ in range(64)]
    got = impl.vcadd(values, impl.ORDER_LINEAR)
    want = _bf16_linear_fold(values)
    if got != want:
        return {"phase": "reduction-linear", "input": ["0x%04X" % v for v in values],
                "reference": "0x%04X" % want, "implementation": "0x%04X" % got,
                "reason": "reference folds the exact rational sum then rounds to bf16 per step"}
    return {"phase": "reduction-linear", "checked": "64 lanes"}


def phase_precision_choice() -> dict:
    """Report (not fail) where bf16-step vs f32-step accumulation differs: a CHOICE.

    First searches a rounding-prone family (all terms in [1,2)) so the difference is
    demonstrated rather than merely not found.
    """
    rng = random.Random(0xF32)
    for _ in range(20000):
        values = [(127 << 7) | rng.getrandbits(7) for _ in range(64)]
        bf16_step = _bf16_linear_fold(values)
        total = ref.dot_bf16_reference([float(ref.bf16_bits_to_fraction(v)) for v in values])
        f32_step = ref.fraction_to_bf16_bits(Fraction(total))
        if bf16_step != f32_step:
            return {"phase": "precision-choice", "divergence_found": True,
                    "crafted_vector": ["0x%04X" % v for v in values],
                    "bf16_step_accumulate": "0x%04X" % bf16_step,
                    "f32_step_accumulate": "0x%04X" % f32_step,
                    "note": "same inputs, different accumulation precision -> different bits; "
                            "the spec does not fix which one the device does."}
    for _ in range(2000):
        values = [_finite_bf16(rng) for _ in range(64)]
        bf16_step = _bf16_linear_fold(values)
        try:
            total = ref.dot_bf16_reference([float(ref.bf16_bits_to_fraction(v)) for v in values])
            f32_step = ref.fraction_to_bf16_bits(Fraction(total))
        except (OverflowError, ValueError):
            continue  # squared magnitude overflowed f32; not a comparable vector
        if bf16_step != f32_step:
            return {"phase": "precision-choice", "first_vector": ["0x%04X" % v for v in values],
                    "bf16_step_accumulate": "0x%04X" % bf16_step,
                    "f32_step_accumulate": "0x%04X" % f32_step}
    return {"phase": "precision-choice", "note": "no difference in 2000 vectors"}


def main() -> int:
    results = [phase_encode(), phase_decode(), phase_reduction()]
    hard_fail = [r for r in results if "reference" in r and "implementation" in r]
    for r in results:
        print(r)
    print("PRECISION CHOICE (documented, not a failure):", phase_precision_choice())
    if hard_fail:
        print("RESULT: DIVERGENCE")
        return 1
    print("RESULT: AGREE (encode/decode/reduction-linear)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
