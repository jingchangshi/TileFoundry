#!/usr/bin/env python3
"""Independent bit-level BF16/f32 reference (P2 reference side).

This module is deliberately NOT built on torch, numpy or the implementation under test.
Every value is an exact rational (fractions.Fraction); rounding is implemented from the
IEEE-754 definitions (round-to-nearest, ties-to-even) on integer arithmetic.

Formats:
  f32: 1 sign | 8 exponent (bias 127) | 23 mantissa
  bf16:1 sign | 8 exponent (bias 127) | 7 mantissa   (same exponent range as f32)

Consequences of the shared exponent range: f32 and bf16 have identical normal/subnormal
boundaries; only mantissa precision (7 vs 23 bits) differs, so f32->bf16 never changes the
binade except by rounding up at a boundary.

Documented reduction order (REFERENCE_REDUCTION_ORDER) is the order this module assumes;
which parts of it the SPEC fixes is discussed in tools/p2/REPORT.md.
"""
from __future__ import annotations

from fractions import Fraction

F32_SIGN = 0x80000000
F32_EXP = 0x7F800000
F32_MANT = 0x007FFFFF
BF16_SIGN = 0x8000
BF16_EXP = 0x7F80
BF16_MANT = 0x007F

# bf16 grid boundaries as exact rationals
TWO = Fraction(2)
BF16_MIN_NORMAL = TWO ** -126
BF16_MAX_NORMAL = Fraction(255, 128) * TWO ** 127
BF16_MIN_SUBNORMAL = TWO ** -133
BF16_SUBNORMAL_HALF = TWO ** -134
BF16_OVERFLOW_MIDPOINT = BF16_MAX_NORMAL + TWO ** 119  # max + ulp/2


def round_half_even(x: Fraction) -> int:
    """Round an exact rational to the nearest integer, ties to even."""
    n, d = x.numerator, x.denominator
    q, r = divmod(n, d)          # q = floor for x >= 0
    twice = 2 * r
    if twice > d or (twice == d and q % 2 == 1):
        q += 1
    return q


def f32_bits_to_fraction(bits: int) -> Fraction | None:
    """Exact value of an f32 bit pattern; None for inf/NaN."""
    sign = -1 if (bits & F32_SIGN) else 1
    exp = (bits >> 23) & 0xFF
    mant = bits & F32_MANT
    if exp == 0xFF:
        return None
    if exp == 0:
        return sign * Fraction(mant, 2 ** 23) * TWO ** -126
    return sign * Fraction(2 ** 23 + mant, 2 ** 23) * TWO ** (exp - 127)


def fraction_to_bf16_bits(value: Fraction) -> int:
    """Round an exact rational to the nearest bf16 (ties to even), inf on overflow."""
    sign = BF16_SIGN if value < 0 else 0
    a = -value if value < 0 else value
    if a == 0:
        return sign
    if a >= BF16_MIN_NORMAL:
        # normal: value = m * 2^(e-7) with m in [128, 256)
        e = 0
        lo, hi = -126, 128
        # exact floor(log2(a)) by binary search
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if TWO ** mid <= a:
                lo = mid
            else:
                hi = mid - 1
        e = lo
        m = a / TWO ** (e - 7)
        q = round_half_even(m)
        if q == 256:
            q = 128
            e += 1
        if e > 127:
            return sign | BF16_EXP  # overflow -> inf
        return sign | ((e + 127) << 7) | (q - 128)
    # subnormal grid: multiples of 2^-133
    q = round_half_even(a / BF16_MIN_SUBNORMAL)
    if q >= 128:
        return sign | (1 << 7)   # rounds up to min normal
    return sign | q


def bf16_bits_to_fraction(bits: int) -> Fraction | None:
    """Exact value of a bf16 bit pattern; None for inf/NaN."""
    sign = -1 if (bits & BF16_SIGN) else 1
    exp = (bits >> 7) & 0xFF
    mant = bits & BF16_MANT
    if exp == 0xFF:
        return None
    if exp == 0:
        return sign * Fraction(mant, 128) * TWO ** -126
    return sign * Fraction(128 + mant, 128) * TWO ** (exp - 127)


def bf16_from_f32_bits(bits: int) -> int:
    """Reference f32 -> bf16 conversion (RNE), preserving inf/NaN class."""
    exp = (bits >> 23) & 0xFF
    mant = bits & F32_MANT
    sign = bits & F32_SIGN
    if exp == 0xFF:
        if mant == 0:
            return (BF16_SIGN if sign else 0) | BF16_EXP        # inf
        return (BF16_SIGN if sign else 0) | 0x7FC0              # canonical quiet NaN
    if exp == 0 and mant == 0:
        return BF16_SIGN if sign else 0                          # preserve -0.0
    value = f32_bits_to_fraction(bits)
    return fraction_to_bf16_bits(value)


def f32_round(x: float) -> float:
    """Round a Python float (double) to f32, RNE; overflow saturates to +/-inf."""
    import math
    import struct
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


# Documented accumulation order used by the reference model.
REFERENCE_REDUCTION_ORDER = {
    "accumulators": 2,
    "accumulator_assignment": "round-robin over chunk index (acc0 even, acc1 odd)",
    "per_accumulator_order": "increasing chunk index",
    "accumulator_combine": "acc0 + acc1 in index order",
    "cross_lane": "vcadd chain applied left-to-right",
    "accumulate_precision": "binary32 after every multiply and add",
}


def dot_bf16_reference(values, accumulators: int = 2):
    """Reference sum of x*x over bf16-decoded chunk vectors, in binary32."""
    acc = [0.0] * accumulators
    for i, v in enumerate(values):
        x = f32_round(v)
        acc[i % accumulators] = f32_round(acc[i % accumulators] + f32_round(x * x))
    total = acc[0]
    for a in acc[1:]:
        total = f32_round(total + a)
    return total


__all__ = ["round_half_even", "f32_bits_to_fraction", "fraction_to_bf16_bits",
           "bf16_bits_to_fraction", "bf16_from_f32_bits", "f32_round",
           "dot_bf16_reference", "REFERENCE_REDUCTION_ORDER",
           "BF16_MIN_NORMAL", "BF16_MAX_NORMAL", "BF16_MIN_SUBNORMAL",
           "BF16_SUBNORMAL_HALF", "BF16_OVERFLOW_MIDPOINT"]
