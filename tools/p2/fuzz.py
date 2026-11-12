#!/usr/bin/env python3
"""P2 fuzz/property check over random bit patterns at the format boundaries.

Reports the FIRST divergence between the independent reference and the implementation,
with the input bits and both decodings, so a disagreement is diagnosable rather than
averaged away.

Run: PYTHONDONTWRITEBYTECODE=1 python tools/p2/fuzz.py [iterations]
"""
from __future__ import annotations

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import random
import sys

from bf16_reference import (bf16_bits_to_fraction, bf16_from_f32_bits,
                            f32_bits_to_fraction)


def is_nan(bits16: int) -> bool:
    return (bits16 & 0x7F80) == 0x7F80 and (bits16 & 0x7F) != 0


def torch_bf16_bits(bits32: int) -> int:
    import struct
    import torch
    x = struct.unpack("<f", struct.pack("<I", bits32 & 0xFFFFFFFF))[0]
    t = torch.tensor([x], dtype=torch.float32).to(torch.bfloat16)
    return t.view(torch.int16).item() & 0xFFFF


BOUNDARY_EXPONENTS = (0, 1, 2, 125, 126, 127, 128, 253, 254, 255)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(0xB1F16)
    first = None
    checked = 0
    for _ in range(n):
        if rng.random() < 0.5:
            exp = rng.choice(BOUNDARY_EXPONENTS)
            bits = (rng.getrandbits(1) << 31) | (exp << 23) | rng.getrandbits(23)
        else:
            bits = rng.getrandbits(32)
        ref = bf16_from_f32_bits(bits)
        impl = torch_bf16_bits(bits)
        checked += 1
        if is_nan(ref) or is_nan(impl):
            ok = is_nan(ref) and is_nan(impl)
        else:
            ok = ref == impl
        if not ok:
            first = {"input_f32_bits": "0x%08X" % bits,
                     "reference_bf16": "0x%04X" % ref,
                     "implementation_bf16": "0x%04X" % impl,
                     "input_value": str(f32_bits_to_fraction(bits)),
                     "reference_value": str(bf16_bits_to_fraction(ref)),
                     "implementation_value": str(bf16_bits_to_fraction(impl))}
            break

    print("checked:", checked, "of", n)
    if first:
        print("FIRST DIVERGENCE:", first)
        return 1
    # property: re-encoding a decoded bf16 reproduces the same bits (no drift)
    bad = 0
    for _ in range(n):
        b = rng.getrandbits(16)
        frac = bf16_bits_to_fraction(b)
        if frac is None:
            continue
        import struct
        x = float(frac)
        # compare via f32 path only where the value is f32-representable
        try:
            f32 = struct.unpack("<f", struct.pack("<f", x))[0]
        except (OverflowError, struct.error):
            continue
        if f32 != x:
            continue
        if bf16_from_f32_bits(struct.unpack("<I", struct.pack("<f", f32))[0]) != b:
            bad += 1
    print("roundtrip drift cases:", bad)
    print("RESULT:", "NO DIVERGENCE" if bad == 0 else "DRIFT")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
