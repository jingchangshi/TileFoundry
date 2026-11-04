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
import struct
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

#: The credit THIS evidence supports. It is a CPU-side software quantizer reference; it is
#: not proof of the native reduction tree, output layout or device behaviour.
CREDIT = "CPU-only software quantizer; full P2 native-ISA claim NOT established"


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
    """dst[L] = bf16(a[L] * scalar) for active lanes; inactive lanes keep dest bits.

    SCALAR PRECISION IS DECLARED, not implicit: the product is evaluated EXACTLY from the
    bf16 lane and the exact binary value of *scalar* (Fraction(scalar)), then rounded ONCE
    to BF16. Multiplying in binary64 first and rounding that product to BF16 is DOUBLE
    rounding and can differ by one ulp (observed: lane 0x3F81 with scalar
    0x1.02fa0be82fa0cp+0 would give 0x3F82 instead of the exact 0x3F83). The ABI does not
    establish whether the device scalar is pre-rounded to f32/f16, so this reference does
    not pretend it is: it evaluates the exact product of the values it is handed.
    """
    out = []
    for index in range(LANES):
        if mask[index]:
            out.append(encode_fraction(Fraction(decode(a[index])) * Fraction(scalar)))
        else:
            out.append(dest[index])
    return out


def _f32(value: float) -> float:
    """Round to IEEE-754 binary32. v33 accumulates in FP32 after BF16->FP32 conversion."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


MERGE_BALANCED_4 = "((0+1)+(2+3))"
MERGE_PAIR_2 = "(0+1)"


def vcadd_v33(values, chains: int, merge: str) -> int:
    """ABSTRACT SOFTWARE SUM: BF16 in, FP32 accumulate, BF16 out. NOT full RMSNorm.

    Scope is deliberately narrow: this reduces 64 lane VALUES. It does NOT model v33's
    strided chunk sequence of 64-lane vmula/square operations, its load/unpack of bf16
    chunks, or any RMSNorm row structure. Calling it "RMSNorm semantics" would overclaim;
    it is a reference for the SUM SHAPE only (per-case chain count, merge tree, FP32
    accumulation position).

    reference/kernel_v33.py (SHA c6c37fb7...) converts each UNPK_B16 lane to float32 and
    accumulates with FP32 vmula chains; only the final scalar is stored as BF16. So the
    per-step rounding here is to FP32, NOT to BF16. *chains* and *merge* are the PER-CASE
    accumulator contract from the algorithm (rows=4 -> 4 chains, ((0+1)+(2+3)); rows<=2 ->
    2 chains, (0+1)), not a universal ISA constant.

    Chain c accumulates lanes c, c+chains, c+2*chains, ... in FP32; the chains are then
    merged in the stated balanced tree, in FP32; the merged scalar is rounded to BF16.
    The SIX-CALL baseline sequence is kept as the reduction sequence; this function does
    not claim the native tree shape or output layout, which are UNSOURCED against the
    current build (isa.NATIVE_BUILD). This is a SOFTWARE reference.
    """
    if chains not in (2, 4):
        raise ValueError(f"v33 declares 2 or 4 accumulator chains per case, got {chains}")
    if len(values) % chains:
        raise ValueError(f"{len(values)} lanes do not divide into {chains} chains")
    per_chain = len(values) // chains
    accumulators = [0.0] * chains
    for chain in range(chains):
        for step in range(per_chain):
            lane = _f32(decode(values[chain + step * chains]))
            accumulators[chain] = _f32(accumulators[chain] + lane)
    if chains == 4:
        if merge != MERGE_BALANCED_4:
            raise ValueError(f"four chains must merge {MERGE_BALANCED_4}, got {merge!r}")
        merged = _f32(
            _f32(accumulators[0] + accumulators[1]) + _f32(accumulators[2] + accumulators[3])
        )
    else:
        if merge != MERGE_PAIR_2:
            raise ValueError(f"two chains must merge {MERGE_PAIR_2}, got {merge!r}")
        merged = _f32(accumulators[0] + accumulators[1])
    return encode(merged)


# RETRACTED numeric model (kept for the record and for a CONTRAST microtest, NOT used):
# these orders round every step to BF16. That is NOT v33 accumulation semantics -- v33
# accumulates in FP32 after the BF16 conversion -- so a per-step-BF16 reducer would make a
# plausible evaluator disagree with the actual sequence. Use vcadd_v33 instead.
ORDER_LINEAR = "RETRACTED:linear-per-step-bf16"
ORDER_PAIRWISE = "RETRACTED:pairwise-per-step-bf16"
ORDER_EXACT = "RETRACTED:exact-then-round"


def vcadd(values, order: str) -> int:
    """RETRACTED per-step-BF16 reducer (contrast only). Use :func:`vcadd_v33`.

    Kept so a microtest can show the difference from v33 FP32 accumulation; NOT v33
    semantics and not used by any reference or report path.
    """
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
    "LANES", "CREDIT", "encode", "encode_fraction", "decode", "round_value", "pack_pair",
    "unpack_pair", "lane_add", "lane_mul", "lane_scale", "vcadd", "vcadd_v33",
    "MERGE_BALANCED_4", "MERGE_PAIR_2",
    "ORDER_LINEAR", "ORDER_PAIRWISE", "ORDER_EXACT",
]
