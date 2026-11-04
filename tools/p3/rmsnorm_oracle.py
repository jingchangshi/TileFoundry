#!/usr/bin/env python3
"""P3 independent numeric oracle for BF16 RMSNorm, modelling the pinned v33 semantics.

CORRECTED per the Lead's retraction (tools/p2/REPORT.md section 8). This oracle does NOT
use one global accumulator count, and it does NOT round each reduction step to BF16.

v33 semantics it models (from reference/kernel_v33.py, SHA c6c37fb7... == P0 pin):
  * inputs BF16, converted to FP32 (vld UNPK_B16 -> vcvt "float32");
  * ALL accumulation in FP32, PER LANE, chain j takes chunk c with c % chains == j;
  * chains per case from _rows_per_core: prefill 8192x4096 -> rows=4 -> FOUR chains,
    merged BALANCED ((a0+a1)+(a2+a3)); decode 64x4096 -> rows=1 -> rows<=2 fallback ->
    TWO chains, merged (a0+a1);
  * then a lane reduction to 1 (mean = b*(1/d), scale = 1/sqrt(mean+eps)), all FP32;
    THE LANE-REDUCTION ORDER IS UNRESOLVED -- the pairwise-halving code below is a
    PLACEHOLDER, unsourced against simd_inst.h 688-702 (asc_reduce_sum, PAT_VL1) and
    681-684 (vcpadd asc_pair_reduce_sum); pin it with an exact selected-build microtest.
  * y = FP32(x*scale*w) and a SINGLE RNE to BF16 on store.

Assumptions (not fixed by the spec; see the report): vcadd as pairwise adjacent halving;
SCALAR SEMANTICS: this oracle multiplies by a FLOAT32-ROUNDED scalar (v33's scale is FP32).
An exact-scalar multiply would differ: 0x3f81 * 0x1.02fa0be82fa0cp+0 gives bf16 0x3f82 with
the f32-rounded scalar but 0x3f83 with the exact scalar (demonstrated by --scalar-demo).
vmula as two roundings (mul then add) rather than a fused single-rounding FMA.

Outputs: <out>/<case>/<pattern>.y_bf16.npy (uint16) + MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "p2"))
import bf16_reference as p2  # noqa: E402

EPS = 1e-6
CH = 64
SHAPES = {"prefill": (8192, 4096), "decode": (64, 4096)}
# Per-case chains from _rows_per_core: prefill rows=4 (four), decode rows=1 (fallback two).
CHAINS = {"prefill": p2.ACCUMULATOR_CHAINS["prefill_8192x4096"],
          "decode": p2.ACCUMULATOR_CHAINS["decode_64x4096"]}
PATTERNS = ("zero", "ones", "random", "alternating")
BLOCK = 256


def bf16_from_f64(a: np.ndarray) -> np.ndarray:
    """Single RNE f64 -> bf16, from the f64 significand (no intermediate f32)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    u = a.view(np.uint64)
    sign = (((u >> np.uint64(63)) & np.uint64(1)).astype(np.uint32) << np.uint32(15))
    exp = ((u >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    mant = u & np.uint64((1 << 52) - 1)
    out = sign.copy()
    is_nan_inf = exp == 0x7FF
    e = exp - 1023
    sig = mant | np.uint64(1 << 52)

    # normal path: round the 53-bit significand to 8 bits
    shift = np.uint64(45)
    q = sig >> shift
    rem = sig & ((np.uint64(1) << shift) - np.uint64(1))
    half = np.uint64(1) << (shift - np.uint64(1))
    q = q + ((rem > half) | ((rem == half) & ((q & np.uint64(1)) == np.uint64(1)))).astype(np.uint64)
    carry = q == np.uint64(256)
    q = np.where(carry, np.uint64(128), q)
    e = np.where(carry, e + 1, e)
    normal = (e >= -126) & (exp != 0)

    # bf16 subnormal range: values in [2^-134, 2^-126)
    sub = (exp != 0) & (e < -126) & (e >= -134)
    k = np.clip(-(e + 81), 46, 53).astype(np.uint64)
    q2 = sig >> k
    rem2 = sig & ((np.uint64(1) << k) - np.uint64(1))
    half2 = np.uint64(1) << (k - np.uint64(1))
    q2 = q2 + ((rem2 > half2) | ((rem2 == half2) & ((q2 & np.uint64(1)) == np.uint64(1)))).astype(np.uint64)
    q2 = np.where(q2 >= np.uint64(128), np.uint64(128), q2)

    exp_bits = np.where(normal, (e + 127), 0).astype(np.uint32)
    mant_bits = np.where(normal, (q.astype(np.int64) - 128), q2.astype(np.int64)).astype(np.uint32)
    bits = out | (exp_bits << np.uint32(7)) | mant_bits
    bits = np.where(sub, out | q2.astype(np.uint32), bits)

    tiny = (exp == 0) | ((exp != 0) & (e < -134))
    bits = np.where(tiny, out, bits)
    bits = np.where(is_nan_inf, out | np.uint32(0x7F80), bits)
    bits = np.where(is_nan_inf & (mant != 0), out | np.uint32(0x7FC0), bits)
    bits = np.where(normal & (e > 127) & (~is_nan_inf), out | np.uint32(0x7F80), bits)
    return bits.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    sign = np.where((bits & np.uint16(0x8000)) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((bits >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64)
    mant = (bits & np.uint16(0x7F)).astype(np.int64)
    val = np.where(exp == 0,
                   mant.astype(np.float32) * np.float32(2.0 ** -133),
                   (128 + mant).astype(np.float32) * np.float32(2.0) ** (exp - 127 - 7))
    return (sign * val.astype(np.float32)).astype(np.float32)


def gen_inputs(pattern, r0, r1, d, seed):
    n = (r1 - r0) * d
    if pattern == "zero":
        x = np.zeros(n); w = np.ones(n)
    elif pattern == "ones":
        x = np.ones(n); w = np.ones(n)
    elif pattern == "alternating":
        x = np.where(np.arange(n) % 2 == 0, 1.0, -1.0); w = np.ones(n)
    elif pattern == "random":
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n); w = rng.standard_normal(n)
    else:
        raise ValueError("unknown pattern " + repr(pattern))
    return x.reshape(r1 - r0, d), w.reshape(r1 - r0, d)


def run_case(case, pattern, seed, out_dir):
    rows, d = SHAPES[case]
    chains = CHAINS[case]
    nch = d // CH
    y_bits = np.empty((rows, d), dtype=np.uint16)
    scale_lo, scale_hi = float("inf"), float("-inf")
    max_x = 0.0
    sum_x2 = 0.0
    for r0 in range(0, rows, BLOCK):
        r1 = min(rows, r0 + BLOCK)
        x, w = gen_inputs(pattern, r0, r1, d, seed)
        xb = bf16_to_f32(bf16_from_f64(x))
        wb = bf16_to_f32(bf16_from_f64(w))
        max_x = max(max_x, float(np.max(np.abs(xb.astype(np.float64)))))
        sum_x2 += float(np.sum(xb.astype(np.float64) ** 2))
        n = r1 - r0
        acc = [np.zeros((n, CH), dtype=np.float32) for _ in range(chains)]
        for c in range(nch):
            v = xb[:, c * CH:(c + 1) * CH]
            j = c % chains
            acc[j] = (acc[j] + (v * v)).astype(np.float32)   # FP32, per lane, in chunk order
        if chains == 4:
            merged = ((acc[0] + acc[1]) + (acc[2] + acc[3])).astype(np.float32)
        elif chains == 2:
            merged = (acc[0] + acc[1]).astype(np.float32)
        else:
            raise ValueError("unsupported chain count %r" % chains)
        # PLACEHOLDER lane reduction; ORDER IS UNRESOLVED (see module docstring / manifest)
        while merged.shape[1] > 1:
            merged = (merged[:, 0::2] + merged[:, 1::2]).astype(np.float32)
        total = merged[:, 0]
        mean = (total * np.float32(1.0 / d)).astype(np.float32)
        scale = (np.float32(1.0) / np.sqrt((mean + np.float32(EPS)).astype(np.float32))).astype(np.float32)
        scale_lo = min(scale_lo, float(scale.min())); scale_hi = max(scale_hi, float(scale.max()))
        y = ((xb * scale[:, None]).astype(np.float32) * wb).astype(np.float32)
        y_bits[r0:r1] = bf16_from_f64(y.astype(np.float64))
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / (pattern + ".y_bf16.npy"), y_bits)
    y = bf16_to_f32(y_bits).astype(np.float64)
    return {"case": case, "pattern": pattern, "seed": seed, "shape": [rows, d],
            "chains": chains, "max_abs_x": max_x, "sum_x2": sum_x2,
            "max_abs_y": float(np.max(np.abs(y))), "sum_y": float(np.sum(y)),
            "scale_min": scale_lo, "scale_max": scale_hi,
            "y_file": str((case_dir / (pattern + ".y_bf16.npy")).relative_to(out_dir))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--scalar-demo", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        from fractions import Fraction
        import random
        rng = random.Random(7)
        bad = 0
        for _ in range(5000):
            v = np.frombuffer(rng.getrandbits(64).to_bytes(8, "little"), dtype=np.float64)[0]
            got = int(bf16_from_f64(np.array([v]))[0])
            if math.isnan(v):
                if (got & 0x7F80) != 0x7F80 or (got & 0x7F) == 0:
                    bad += 1
                continue
            if math.isinf(v):
                if (got & 0x7F80) != 0x7F80 or (got & 0x7F) != 0:
                    bad += 1
                continue
            if got != p2.fraction_to_bf16_bits(Fraction(v)):
                bad += 1
        print("self-check bad:", bad)
        return 0 if bad == 0 else 1
    if args.scalar_demo:
        from fractions import Fraction
        scalar = float.fromhex("0x1.02fa0be82fa0cp+0")
        lane = p2.bf16_bits_to_fraction(0x3f81)
        exact_bits = p2.fraction_to_bf16_bits(lane * Fraction(scalar))
        p32 = np.float32(np.float32(float(lane)) * np.float32(scalar))
        f32_bits = p2.fraction_to_bf16_bits(Fraction(float(p32)))
        print("scalar exact:", repr(scalar))
        print("exact-scalar bf16:    0x%04X" % exact_bits)
        print("f32-rounded-scalar bf16: 0x%04X" % f32_bits)
        return 0 if (exact_bits, f32_bits) == (0x3F83, 0x3F82) else 1
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in SHAPES:
        for pattern in PATTERNS:
            for loop_mode in ("dyn", "recompute"):
                r = run_case(case, pattern, args.seed, out)
                r["loop_mode"] = loop_mode
                results.append(r)
    manifest = {
        "protocol_sha256": "a1676239e1436aa9d897db3f5145afc668532995dd6f84e4b3045785462a78d0",
        "reference_kernel_sha256": "c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909",
        "eps": EPS, "chunk": CH, "seed": args.seed,
        "accumulation": "FP32 after BF16->FP32 conversion",
        "chains_per_case": CHAINS,
        "merge": {"prefill": "((a0+a1)+(a2+a3))", "decode": "(a0+a1)"},
        "vcadd_model": "UNRESOLVED: pairwise-halving PLACEHOLDER only; must be pinned by a selected-build microtest against simd_inst.h 688-702 (asc_reduce_sum PAT_VL1) / 681-684 (vcpadd asc_pair_reduce_sum)",
        "vmula_model": "mul then add, two FP32 roundings (ASSUMED)",
        "scalar_semantics": "FLOAT32-ROUNDED scalar (v33 scale is FP32); exact scalar differs: 0x3f81*0x1.02fa0be82fa0cp+0 -> 0x3f82 (f32 scalar) vs 0x3f83 (exact)",
        "patterns": list(PATTERNS), "shapes": {k: list(v) for k, v in SHAPES.items()},
        "results": results,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    for r in results:
        print("%-8s %-11s %-9s chains=%d max|x|=%.6g sum(x^2)=%.6g max|y|=%.6g sum(y)=%.6g scale=[%.6g,%.6g]"
              % (r["case"], r["pattern"], r["loop_mode"], r["chains"], r["max_abs_x"],
                 r["sum_x2"], r["max_abs_y"], r["sum_y"], r["scale_min"], r["scale_max"]))
    print("manifest:", out / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
