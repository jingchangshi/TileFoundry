"""P2: BF16/f32 elementwise numeric surface, implemented independently.

The review names BF16 ENCODING and REDUCTION ORDER as the next real technical risk, so
this module is a pure-Python reference with no Torch and no numpy. Encoding rounds the
EXACT rational value to BF16 (single round-to-nearest-even, never via f32 first); lane
semantics round every elementwise result to BF16 and leave inactive predicated lanes bit
unchanged; reduction order is an explicit, documented fold with per-step BF16 rounding,
and the exact-then-round order is provided as a deliberate CONTRAST so a microtest can
show the order is observable rather than assumed.

BF16 layout: 1 sign, 8 exponent (bias 127), 7 mantissa. Two BF16 lanes pack into one
32-bit carrier with the low lane in bits 0..15.
"""
from __future__ import annotations

import math
from fractions import Fraction

SIGN_SHIFT = 15
EXP_SHIFT = 7
MANTISSA_BITS = 7
EXP_MASK = 0xFF
MANTISSA_MASK = (1 << MANTISSA_BITS) - 1
BIAS = 127
SUBNORMAL_QUANTUM = Fraction(1, 1 << (126 + MANTISSA_BITS))
LANES = 64
TWO = Fraction(2)


def _round_half_even(value: Fraction) -> int:
    """Nearest integer, ties to even, on a non-negative exact rational."""
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder > Fraction(1, 2):
        return floor + 1
    if remainder < Fraction(1, 2):
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _floor_log2(value: Fraction) -> int:
    """Largest e with 2**e <= value, for a positive exact rational."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    while TWO ** (exponent + 1) <= value:
        exponent += 1
    while TWO ** exponent > value:
        exponent -= 1
    return exponent


def _encode_magnitude(exact: Fraction) -> int:
    """BF16 bits for a non-negative exact rational, single RNE (no sign bit)."""
    if exact == 0:
        return 0
    if exact < Fraction(1, 1 << 126):
        steps = _round_half_even(exact / SUBNORMAL_QUANTUM)
        if steps >= (1 << MANTISSA_BITS):
            return 1 << EXP_SHIFT  # rounded up into the smallest normal
        return steps
    exponent = _floor_log2(exact)
    if exponent > BIAS:
        return EXP_MASK << EXP_SHIFT  # overflow to infinity
    quantum = TWO ** (exponent - MANTISSA_BITS)
    steps = _round_half_even(exact / quantum)
    if steps == (1 << (MANTISSA_BITS + 1)):  # 256 -> carry into the next exponent
        steps = 1 << MANTISSA_BITS
        exponent += 1
        if exponent > BIAS:
            return EXP_MASK << EXP_SHIFT
    mantissa = steps - (1 << MANTISSA_BITS)
    return ((exponent + BIAS) << EXP_SHIFT) | mantissa


def encode_fraction(exact: Fraction) -> int:
    """Encode an EXACT rational to BF16 bits with a single round-to-nearest-even."""
    if exact < 0:
        return (1 << SIGN_SHIFT) | _encode_magnitude(-exact)
    return _encode_magnitude(exact)


def encode(value: float) -> int:
    """Encode a Python float to BF16 bits with a single round-to-nearest-even."""
    if math.isnan(value):
        return (1 << SIGN_SHIFT) | (EXP_MASK << EXP_SHIFT) | (1 << 6)
    sign = 1 << SIGN_SHIFT if math.copysign(1.0, value) < 0 else 0
    if math.isinf(value):
        return sign | (EXP_MASK << EXP_SHIFT)
    return sign | _encode_magnitude(Fraction(abs(value)))


def decode(bits: int) -> float:
    """Decode BF16 bits to the exact Python float they denote (no rounding)."""
    sign = -1.0 if bits & (1 << SIGN_SHIFT) else 1.0
    exponent = (bits >> EXP_SHIFT) & EXP_MASK
    mantissa = bits & MANTISSA_MASK
    if exponent == EXP_MASK:
        return sign * (math.inf if mantissa == 0 else math.nan)
    if exponent == 0:
        return sign * float(Fraction(mantissa) * SUBNORMAL_QUANTUM)
    return sign * float(Fraction(1) + Fraction(mantissa, 1 << MANTISSA_BITS)) * (2.0 ** (exponent - BIAS))


def round_value(value: float) -> float:
    return decode(encode(value))


def pack_pair(low: int, high: int) -> int:
    """Two BF16 lanes into one 32-bit carrier: low lane in bits 0..15."""
    return (low & 0xFFFF) | ((high & 0xFFFF) << 16)


def unpack_pair(carrier: int) -> tuple[int, int]:
    return carrier & 0xFFFF, (carrier >> 16) & 0xFFFF


def lane_add(a, b, mask, dest):
    """dst[L] = bf16(a[L] + b[L]) where mask[L] is active, else dest[L] bits unchanged."""
    out = []
    for index in range(LANES):
        if mask[index]:
            out.append(encode(decode(a[index]) + decode(b[index])))
        else:
            out.append(dest[index])
    return out


def lane_mul(a, b, mask, dest):
    out = []
    for index in range(LANES):
        if mask[index]:
            out.append(encode(decode(a[index]) * decode(b[index])))
        else:
            out.append(dest[index])
    return out


def lane_scale(a, scalar: float, mask, dest):
    out = []
    for index in range(LANES):
        if mask[index]:
            out.append(encode(decode(a[index]) * scalar))
        else:
            out.append(dest[index])
    return out


# Reduction orders. The order is part of the CONTRACT: it changes the BF16 result.
ORDER_LINEAR = "linear"              # acc = bf16(acc + lane[L]) for L = 0..n-1
ORDER_PAIRWISE = "pairwise"          # halve: bf16(a+b) pairwise, then combine
ORDER_EXACT = "exact-then-round"     # exact rational sum, rounded ONCE to BF16


def vcadd(values, order: str) -> int:
    """Reduce *values* (BF16 bit patterns) to one BF16 accumulator in a stated order."""
    lanes = [decode(value) for value in values]
    if order == ORDER_LINEAR:
        accumulator = 0.0
        for lane in lanes:
            accumulator = decode(encode(accumulator + lane))
        return encode(accumulator)
    if order == ORDER_PAIRWISE:
        level = list(lanes)
        while len(level) > 1:
            nxt = []
            for index in range(0, len(level) - 1, 2):
                nxt.append(decode(encode(level[index] + level[index + 1])))
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return encode(level[0] if level else 0.0)
    if order == ORDER_EXACT:
        return encode_fraction(sum((Fraction(lane) for lane in lanes), Fraction(0)))
    raise ValueError(f"unknown reduction order {order!r}; known: {ORDER_LINEAR}, {ORDER_PAIRWISE}, {ORDER_EXACT}")


__all__ = [
    "LANES", "encode", "encode_fraction", "decode", "round_value", "pack_pair",
    "unpack_pair", "lane_add", "lane_mul", "lane_scale", "vcadd",
    "ORDER_LINEAR", "ORDER_PAIRWISE", "ORDER_EXACT",
]
