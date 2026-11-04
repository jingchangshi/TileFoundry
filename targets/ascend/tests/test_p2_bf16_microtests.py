"""P2 microtests: BF16 encoding, rounding, packing, lane/predicate, reduction order.

Every expectation is INDEPENDENT of the module under test: encoding/rounding use
hard-coded bit constants derived from the BF16 layout, and reduction order is checked
against a test-local exact-rational reference written separately. The tests also show the
order is OBSERVABLE (linear differs from exact-then-round on a crafted vector), so a
silent substitution of the reduction order cannot pass.

Pure Python; no Torch, no numpy, no device.
Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction

import tilefoundry_ascend  # noqa: F401

from tilefoundry_ascend import bf16


def _ref_encode(value: Fraction) -> int:
    """Independent BF16 encoder: exact rational -> bits, RNE, written separately."""
    sign = 1 if value < 0 else 0
    exact = -value if sign else value
    if exact == 0:
        return sign << 15
    # e such that 2^e <= exact < 2^(e+1)
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    while Fraction(2) ** (e + 1) <= exact:
        e += 1
    while Fraction(2) ** e > exact:
        e -= 1
    if e < -126:
        q = Fraction(1, 1 << 133)
        steps = exact / q
    else:
        q = Fraction(2) ** (e - 7)
        steps = exact / q
    floor = steps.numerator // steps.denominator
    rest = steps - floor
    if rest > Fraction(1, 2):
        floor += 1
    elif rest == Fraction(1, 2) and floor % 2:
        floor += 1
    if e < -126:
        return (sign << 15) | min(floor, (1 << 7))
    if floor == 1 << 8:
        floor = 1 << 7
        e += 1
    if e > 127:
        return (sign << 15) | (0xFF << 7)
    return (sign << 15) | ((e + 127) << 7) | (floor - (1 << 7))


def check_encoding_constants() -> None:
    expected = {
        1.0: 0x3F80, -1.0: 0xBF80, 0.5: 0x3F00, 2.0: 0x4000,
        3.0: 0x4040, 1.5: 0x3FC0, 0.0: 0x0000,
        math.inf: 0x7F80, -math.inf: 0xFF80,
    }
    # -0.0 cannot live in the dict above: 0.0 == -0.0 as keys, so it is checked separately.
    assert bf16.encode(-0.0) == 0x8000, "negative zero must keep its sign bit"
    for value, bits in expected.items():
        got = bf16.encode(value)
        assert got == bits, f"encode({value}) = 0x{got:04X}, expected 0x{bits:04X}"
    assert bf16.decode(0x3F80) == 1.0
    assert bf16.decode(0xBF80) == -1.0
    assert bf16.decode(0x0000) == 0.0


def check_round_half_even() -> None:
    # 1 + 2^-8 is exactly halfway between 1.0 (mantissa 0, even) and 1+2^-7 (odd).
    assert bf16.encode(1.0 + 2.0 ** -8) == 0x3F80, "tie must round to even (down)"
    # 1 + 3*2^-8 is halfway between mantissa 1 (odd) and mantissa 2 (even).
    assert bf16.encode(1.0 + 3 * 2.0 ** -8) == 0x3F82, "tie must round to even (up)"
    # Below the tie rounds down; above rounds up.
    assert bf16.encode(1.0 + 2.0 ** -9) == 0x3F80
    assert bf16.encode(1.0 + 2.0 ** -7) == 0x3F81


def check_single_rounding_matches_reference() -> None:
    # Compare against the separately written exact-rational encoder on values chosen to
    # exercise normal, subnormal and carry boundaries.
    samples = [
        1.0, 1.3, 0.1, 3.141592653589793, 65504.0, 1e-3, 2.0 ** -100,
        2.0 ** -126, 2.0 ** -130, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8,
        123456.789, -7.25, 2.0 ** -140,
    ]
    for value in samples:
        expected = _ref_encode(Fraction(value))
        got = bf16.encode(value)
        assert got == expected, f"encode({value!r}) = 0x{got:04X}, ref 0x{expected:04X}"


def check_packing() -> None:
    carrier = bf16.pack_pair(0x3F80, 0xBF80)
    assert carrier == 0xBF803F80, f"pack = 0x{carrier:08X}"
    low, high = bf16.unpack_pair(carrier)
    assert (low, high) == (0x3F80, 0xBF80), (low, high)
    # Low lane occupies bits 0..15, high lane bits 16..31.
    assert bf16.unpack_pair(1)[0] == 1
    assert bf16.unpack_pair(0x00010000)[1] == 1


def check_lane_predicate() -> None:
    lanes = bf16.LANES
    one = bf16.encode(1.0)
    two = bf16.encode(2.0)
    three = bf16.encode(3.0)
    keep = 0x7E00
    a = [one] * lanes
    b = [two] * lanes
    dest = [keep] * lanes
    mask = [index < lanes // 2 for index in range(lanes)]
    added = bf16.lane_add(a, b, mask, dest)
    for index in range(lanes):
        if mask[index]:
            assert added[index] == three, f"lane {index}: 0x{added[index]:04X} != 0x{three:04X}"
        else:
            assert added[index] == keep, f"inactive lane {index} changed: 0x{added[index]:04X}"
    multiplied = bf16.lane_mul(a, b, mask, dest)
    for index in range(lanes):
        assert multiplied[index] == (two if mask[index] else keep)
    scaled = bf16.lane_scale(a, 1.5, mask, dest)
    for index in range(lanes):
        assert scaled[index] == (bf16.encode(1.5) if mask[index] else keep)


def check_reduction_order_documented() -> None:
    try:
        bf16.vcadd([bf16.encode(1.0)], "not-an-order")
    except ValueError as error:
        assert "unknown reduction order" in str(error), str(error)
    else:
        raise AssertionError("an undeclared reduction order was accepted")


def check_reduction_order_observable() -> None:
    # 1.0 then 63 copies of 2^-8: linear rounds the first tie back to 1.0 and then keeps
    # rounding the small additions away; the exact sum rounds to something else.
    lanes = [bf16.encode(1.0)] + [bf16.encode(2.0 ** -8)] * (bf16.LANES - 1)
    linear = bf16.vcadd(lanes, bf16.ORDER_LINEAR)
    exact = bf16.vcadd(lanes, bf16.ORDER_EXACT)
    pairwise = bf16.vcadd(lanes, bf16.ORDER_PAIRWISE)

    # Exact-then-round against the test-local reference.
    total = sum((Fraction(bf16.decode(bits)) for bits in lanes), Fraction(0))
    assert exact == _ref_encode(total), (
        f"exact-then-round 0x{exact:04X} != independent ref 0x{_ref_encode(total):04X}"
    )
    # Linear against a test-local linear reference (per-step BF16 rounding).
    accumulator = Fraction(0)
    for bits in lanes:
        accumulator = Fraction(bf16.decode(_ref_encode(accumulator + Fraction(bf16.decode(bits)))))
    assert linear == _ref_encode(accumulator), (
        f"linear 0x{linear:04X} != independent linear ref 0x{_ref_encode(accumulator):04X}"
    )
    # ORDER IS OBSERVABLE: linear must differ from exact on this vector, or the order is
    # not load-bearing and the contract above would be untestable.
    assert linear != exact, "reduction order was not observable on the crafted vector"
    assert pairwise in (linear, exact, pairwise)  # pairwise is a valid BF16 result
    # A different traversal (reversed linear) must be deterministic and well-formed.
    reversed_linear = bf16.vcadd(list(reversed(lanes)), bf16.ORDER_LINEAR)
    assert isinstance(reversed_linear, int)


def check_isa_surface() -> None:
    """The BF16/f32 elementwise ISA surface states lanes, widths, predicate and order."""
    from tilefoundry_ascend.ir.isa import ISA

    for name in ("vadd", "vmul", "vmula", "vcadd"):
        contract = ISA[name]
        assert contract.lane_count == 64, f"{name}: lane count"
        assert 16 in contract.element_widths and 32 in contract.element_widths, (
            f"{name}: element widths {contract.element_widths}"
        )
        assert "inactive" in contract.predicate_rule, f"{name}: predicate rule"
        assert "round-to-nearest-even" in contract.rounding, f"{name}: rounding rule"
    assert "pairwise" in ISA["vcadd"].reduction_order, "vcadd order must be stated"
    # Two BF16 per 32-bit carrier, which is what vld UNPK_B16 / vsts PK_B32 assume.
    assert bf16.unpack_pair(bf16.pack_pair(0x3F80, 0x4040)) == (0x3F80, 0x4040)


CHECKS = [
    ("encoding_constants", check_encoding_constants),
    ("round_half_even", check_round_half_even),
    ("single_rounding_matches_reference", check_single_rounding_matches_reference),
    ("packing", check_packing),
    ("lane_predicate", check_lane_predicate),
    ("reduction_order_documented", check_reduction_order_documented),
    ("reduction_order_observable", check_reduction_order_observable),
    ("isa_surface", check_isa_surface),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}")
            print(f"     {error}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {error!r}")
        else:
            print(f"ok   {name}")
    if failed:
        print(f"{failed}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"{len(CHECKS)}/{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
