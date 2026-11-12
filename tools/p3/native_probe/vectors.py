#!/usr/bin/env python3
"""03-19 DESIGN-TIME predictor: full 64-lane input/output bits for candidate MODELS.

STATUS: UNSOURCED / UNVALIDATED. Nothing in this file is a device observation. It computes
what each *hypothesised* reduction tree / conversion / FMA model WOULD produce, so that a
future selected-build microtest (WBS 03-20, V09 device window) can discriminate them.
The three probe scripts (vcadd_probe.py / vcvt_probe.py / vmula_probe.py) consume these
vectors; THIS ROUND THEY ARE NOT EXECUTED on any device.

Run (CPU only, no device): python tools/p3/native_probe/vectors.py --emit expected_bits.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct

import numpy as np

LANES = 64
f32 = np.float32

PINNED = {
    "tilelang_version_file": "0.1.15",
    "tilelang_build": "0.1.15+cuda.git8e2d3c82",
    "tilelang_git_head": "8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b",
    "simd_inst_h_sha256": "979f4ff76eee82bf092addcc069b238e3084af890f11660498f00640a00f20a1",
    "simd_py_sha256": "89483579ddde5e36a352a853720fc02ba2b06badeab9eca74b31bcde6a04b57f",
    "source_anchors": {
        "vcadd": "src/tl_templates/ascend/simd_inst.h:688-702 (asc_reduce_sum, merge form PAT_VL1)",
        "vcpadd": "src/tl_templates/ascend/simd_inst.h:681-684 (asc_pair_reduce_sum)",
        "vmula": "src/tl_templates/ascend/simd_inst.h:404-406 (::vmula)",
        "vcvt": "src/tl_templates/ascend/simd_inst.h:291-294 (::vcvt)",
        "vcvt_mask_granularity": "tilelang/ascend/language/simd.py:1270-1274 (min(src,target))",
        "vcvt_api": "tilelang/ascend/language/simd.py:1209-1215 (round/part/mode)",
        "vmula_api": "tilelang/ascend/language/simd.py:635-646 (dst READ|WRITE)",
        "vcadd_api": "tilelang/ascend/language/simd.py:957-966",
        "vdupv_api": "tilelang/ascend/language/simd.py:933-939 (POS_LOWEST)",
    },
    "status": "UNSOURCED / UNVALIDATED (design only; no native observation)",
}


def bits(x) -> int:
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def hexbits(x) -> str:
    return "0x%08X" % bits(x)


def from_bits(b: int):
    return np.frombuffer(struct.pack("<I", b & 0xFFFFFFFF), dtype=np.float32)[0]


def bf16_to_f32(b: int) -> np.float32:
    b &= 0xFFFF
    sign = 0x80000000 if b & 0x8000 else 0
    exp = (b >> 7) & 0xFF
    mant = b & 0x7F
    if exp == 0xFF:
        return from_bits(sign | 0x7F800000 | (0x00400000 if mant else 0))
    if exp == 0:
        # subnormal bf16: exact in f32 (2^-133 .. 2^-127)
        val = mant * (2.0 ** -133)
    else:
        val = (128 + mant) * (2.0 ** (exp - 127 - 7))
    u = bits(np.float32(val))
    return from_bits(u | sign)


def f32_to_bf16_rne(v: np.float32) -> int:
    u = bits(v)
    sign = (u >> 16) & 0x8000
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF
    if exp == 0xFF:
        return sign | 0x7F80 | (0x40 if mant else 0)
    if exp == 0 and mant == 0:
        return sign
    lo = u & 0xFFFF
    half = 0x8000
    rounded = (u + half + ((u >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16) & 0xFFFF if not sign else ((rounded >> 16) & 0xFFFF) | 0x8000


def lane_vector(spec: dict[int, int], fill: int = 0) -> list[int]:
    return [spec.get(i, fill) for i in range(LANES)]


def mask_vector(active: list[int], n: int = LANES) -> list[int]:
    return [1 if i in set(active) else 0 for i in range(n)]


def model_seq(v: list[int]) -> int:
    a = f32(0)
    for b in v:
        a = f32(a + from_bits(b))
    return bits(a)


def model_adjacent(v: list[int]) -> int:
    cur = [from_bits(b) for b in v]
    while len(cur) > 1:
        cur = [f32(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)]
    return bits(cur[0])


def model_strided(v: list[int]) -> int:
    cur = [from_bits(b) for b in v]
    while len(cur) > 1:
        h = len(cur) // 2
        cur = [f32(cur[i] + cur[i + h]) for i in range(h)]
    return bits(cur[0])


def model_rr2(v: list[int]) -> int:
    a = [f32(0), f32(0)]
    for i, b in enumerate(v):
        a[i % 2] = f32(a[i % 2] + from_bits(b))
    return bits(f32(a[0] + a[1]))


def model_rr4(v: list[int]) -> int:
    a = [f32(0)] * 4
    for i, b in enumerate(v):
        a[i % 4] = f32(a[i % 4] + from_bits(b))
    return bits(f32(f32(a[0] + a[1]) + f32(a[2] + a[3])))


TREE_MODELS = {"seq": model_seq, "adjacent": model_adjacent, "strided": model_strided,
               "rr2": model_rr2, "rr4": model_rr4}


def case(name: str, kind: str, spec: dict[int, int], *, fill: int = 0,
         active=None, dst_fill: int = 0x7FC00000, models=(), **extra) -> dict:
    v = lane_vector(spec, fill)
    out = {"name": name, "kind": kind, "input_bits": ["0x%08X" % b for b in v],
           "mask": mask_vector(active if active is not None else list(range(LANES))),
           "dst_prefill_bits": ["0x%08X" % ((dst_fill + i) & 0xFFFFFFFF) for i in range(LANES)],
           "predicted": {}, "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    for m in models:
        out["predicted"][m] = "0x%08X" % TREE_MODELS[m](v)
    out.update(extra)
    return out


def build_cases() -> dict:
    P24 = 0x4B800000       # 2^24
    N24 = 0xCB800000       # -2^24
    ONE = 0x3F800000
    ZERO = 0x00000000
    cases = []

    # ---- VCADD: constructive cancellation. The whole point: a RANDOM search would
    # essentially never hit an exact f32 cancellation at 2^24.
    cases.append(case("vcadd_cancel_adjacent", "vcadd", {0: P24, 1: ONE, 2: N24, 3: ZERO},
                      models=list(TREE_MODELS),
                      note="[2^24,1,-2^24,0] at lanes 0-3: adjacent-first trees lose the 1 "
                           "(2^24+1 ties to even); strided/rr2 group 2^24 with -2^24 first "
                           "and keep the 1. Six vcadd calls remain the SOURCE baseline."))
    cases.append(case("vcadd_cancel_strided_position", "vcadd",
                      {0: P24, 1: N24, 32: ONE, 33: ZERO}, models=list(TREE_MODELS),
                      note="SAME four values, different LANE POSITIONS; distinguishes "
                           "adjacent-pair from half-vector-stride pairing."))
    cases.append(case("vcadd_rounding_smallfirst", "vcadd",
                      {0: ONE, 1: P24, 32: ONE, 33: ZERO}, models=list(TREE_MODELS),
                      note="1+1=2 must be formed before adding 2^24 only in some trees; "
                           "f32 result 2^24+2 (0x4B800001) vs 2^24 (0x4B800000)."))
    cases.append(case("vcadd_cancel_high_lanes", "vcadd",
                      {60: P24, 61: ONE, 62: N24, 63: ZERO}, models=list(TREE_MODELS),
                      note="lane-position control: cancels in the LAST four lanes."))
    cases.append(case("vcadd_mask_partial", "vcadd", {0: P24, 1: ONE, 2: N24, 3: ZERO},
                      active=[0, 1, 2, 3], models=list(TREE_MODELS),
                      note="INACTIVE LANES: dst prefilled with a distinct sentinel per lane; "
                           "MODE_ZEROING zeroes them, MODE_MERGING preserves them. The C++ "
                           "merge form simd_inst.h:697-702 uses merge_masked + PAT_VL1."))
    cases.append(case("vcadd_output_layout", "vcadd", {0: P24, 1: ONE, 2: N24, 3: ZERO},
                      dst_fill=0x40400000, models=list(TREE_MODELS),
                      note="OUTPUT LAYOUT: a distinct finite sentinel in every dst lane "
                           "reveals whether the reduction result lands in lane 0 only "
                           "(PAT_VL1), in every lane (broadcast), or in a group; an "
                           "explicit vdupv(POS_LOWEST) must NOT be baked into vcadd."))
    cases.append(case("vcadd_ftz_subnormal", "vcadd",
                      {0: 0x00000001, 1: 0x007FFFFF, 2: 0x00800000, 3: 0x80000001},
                      models=list(TREE_MODELS),
                      note="FTZ/DAZ: IEEE-754 keeps f32 subnormals; a flush-to-zero device "
                           "changes the sum. Compare observed bits to the model column."))
    cases.append(case("vcadd_six_calls", "vcadd", {0: P24, 1: ONE, 2: N24, 3: ZERO},
                      models=list(TREE_MODELS),
                      note="SIX SOURCE CALLS (twin_rmsnorm.py:96-97) then an EXPLICIT "
                           "vdupv(POS_LOWEST) at :98; this case must be run BOTH with and "
                           "without the explicit broadcast to separate the two behaviours."))

    # ---- VCVT: 16-bit mask granularity, part, RNE, inactive, FTZ
    bf16_cases = {0: 0x3F80, 1: 0x3F81, 2: 0xBF80, 3: 0x0001, 4: 0x7F80,
                  5: 0x7FC0, 6: 0x8000, 7: 0x0080}
    # carrier window: two bf16 per 32-bit word. A 64-lane window is 32 words; store the
    # bf16 payload as an f32 bit pattern for transport, documenting the packing explicitly.
    carrier = []
    for i in range(LANES):
        carrier.append(bf16_cases.get(i, 0x0000))
    c = {"name": "vcvt_bf16_to_f32_part0", "kind": "vcvt",
         "source_dtype": "bfloat16", "dest_dtype": "float32",
         "input_bf16_bits": ["0x%04X" % b for b in carrier],
         "carrier_packing": "two bf16 per 32-bit word; UNPK_B16 is a VLD dist, NOT a numeric cast",
         "part": 0, "mask_granularity_bits": 16,
         "mask": mask_vector(list(range(LANES))),
         "dst_prefill_bits": ["0x%08X" % ((0x7FC00000 + i) & 0xFFFFFFFF) for i in range(LANES)],
         "predicted": {"part0_widen": ["0x%08X" % bits(bf16_to_f32(b)) for b in carrier],
                       "part1_widen": ["0x%08X" % bits(bf16_to_f32(b)) for b in carrier[::2] + carrier[1::2]],
                       "note": "part=1 is a DIFFERENT lane mapping and is NOT the TF path; "
                               "a wrong part must be refused by name (HANDOFF §5.4/§5.5)."},
         # RULED observation from the device-1 run (2026-11-05), kept in the artifact
         # so it survives context loss. It is NEITHER a pass NOR a protocol failure.
         "native_observations": {
             "nan_payload_not_preserved": {
                 "lane": 5,
                 "source_bf16_bits": "0x7FC0",
                 "model_predicted": "0x7FC00000",
                 "device_observed": "0x7FFFFFFF",
                 "statement": "bf16 NaN payload is not preserved: the device produces "
                              "an all-ones quiet NaN (0x7FFFFFFF) rather than keeping "
                              "the payload (0x7FC00000).",
                 "counts_as_pass": False,
                 "counts_as_protocol_failure": False,
                 "ruling": "recorded as a native OBSERVATION FACT by architect/Lead "
                           "ruling 2026-11-05; tightening it into a hard rule requires "
                           "an ARCHITECT ruling and must not be self-judged as a failure",
                 "scope": "this case and this instruction only: vcvt bf16->f32, ROUND_R, "
                          "part=0, 16-bit mask; not generalisable to any other "
                          "instruction or direction",
                 "measured_on": "device 1, 2026-11-05",
             },
         },
         "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    cases.append(c)
    tie_vals = [0x3F802000, 0x3F806000, 0x3F80A000, 0x3F80E000,  # around bf16 ties
                0x00400000, 0x00800000, 0x00000001, 0x80000001]
    # NARROWING. dest_dtype is bfloat16, so predictions are 16-bit codewords in
    # DESTINATION-SLOT space, not the contiguous 16-bit packing of the sources.
    # Layout MEASURED on device 1 (2026-11-05, 64 distinct f32 inputs): source lane
    # i lands at destination slot 2*i and odd slots keep the prefill, because part=0
    # (PART_EVEN) writes one 16-bit result per 32-bit destination word. Source lanes
    # 32..63 therefore have no slot in a 64-slot bf16 destination; that is recorded,
    # not silently truncated.
    rne_src = [f32_to_bf16_rne(from_bits(b)) for b in (tie_vals * 8)]
    even_slot = ["0x0000"] * LANES
    for i in range(LANES // 2):
        even_slot[2 * i] = "0x%04X" % rne_src[i]
    c2 = {"name": "vcvt_f32_to_bf16_rne_ftz", "kind": "vcvt",
          "source_dtype": "float32", "dest_dtype": "bfloat16",
          "input_bits": ["0x%08X" % b for b in (tie_vals * 8)],
          "mask_granularity_bits": 16, "round": "ROUND_R", "part": 0,
          "layout": {
              "rule": "destination slot 2*i receives the conversion of source lane i",
              "odd_slots": "keep the prefill; part=0 does not write them",
              "source_lanes_32_to_63": "no destination slot in a 64-slot bf16 window",
              "basis": "OBSERVED on device 1 2026-11-05 with 64 distinct f32 inputs; "
                       "NOT inferred from the instruction name or the case name"},
          # Only 0x00400000 is a DECISIVE FTZ probe: it is an f32 subnormal that
          # converts to a non-zero bf16 (0x0040) unless it is flushed. 0x00000001 is
          # NOT usable as an FTZ probe - it is 2^-149, below the smallest bf16
          # subnormal (2^-133), so it rounds to 0x0000 under plain RNE whether or not
          # FTZ is active. Declaring it as evidence would prove nothing.
          "subnormal_probe": [
              {"src_lane": 4, "src_f32_bits": "0x00400000", "dst_slot": 8,
               "if_flushed": "0x0000", "if_not_flushed": "0x0040"},
          ],
          "predicted": {"rne_bf16_even_slots": even_slot,
                        "rne_bf16_contiguous_reference": [
                            "0x%04X" % b for b in rne_src],
                        "odd_slots_rule": "preserved prefill, not written by part=0",
                        "ftz_daz_if_flushed": "subnormal f32 inputs (0x00400000, 0x00000001) "
                                              "become bf16 0x0000 instead of the RNE subnormal/normal",
                        "note": "ROUND_R ties-to-even at the bf16 mantissa boundary. "
                                "FTZ/DAZ is a device property to be OBSERVED, never assumed: "
                                "the case NAME says ftz and is not evidence."},
          "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    cases.append(c2)
    c3 = {"name": "vcvt_inactive_lane_preservation", "kind": "vcvt",
          "source_dtype": "bfloat16", "dest_dtype": "float32",
          "input_bf16_bits": ["0x%04X" % b for b in carrier],
          "mask": mask_vector(list(range(0, 32))), "mask_granularity_bits": 16,
          "dst_prefill_bits": ["0x%08X" % ((0x40400000 + i) & 0xFFFFFFFF) for i in range(LANES)],
          "predicted": {"masked_lanes_active": "lanes 0..31 written with widened values",
                        "inactive_lanes_MODEL_MERGING": "dst_prefill_bits[32..63] unchanged",
                        "inactive_lanes_MODEL_ZEROING": "0x00000000 in lanes 32..63",
                        "note": "declared 16-bit mask (pset16), produced in the SAME region "
                                "instance and lexically before the vcvt; a 32-bit mask is "
                                "a NAMED refusal, not a shape inference."},
          "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    cases.append(c3)

    # ---- VMULA: FMA fusion, loop carry, inactive, FTZ, rounding
    A, B, C = 0x3F7FFFF8, 0x3F800004, 0x34400000   # constructive single-rounding discriminator
    mul = bits(f32(from_bits(A) * from_bits(B)))
    unfused = bits(f32(from_bits(mul) + from_bits(C)))
    exact = float(from_bits(A)) * float(from_bits(B)) + float(from_bits(C))
    fused = bits(np.float32(exact))
    m1 = {"name": "vmula_fma_fusion", "kind": "vmula",
          "src0_bits": ["0x%08X" % A] * LANES,
          "src1_bits": ["0x%08X" % B] * LANES,
          "acc_in_bits": ["0x%08X" % C] * LANES,
          "mask": mask_vector(list(range(LANES))),
          "predicted": {"fused_single_rounding": "0x%08X" % fused,
                        "mul_then_add_two_roundings": "0x%08X" % unfused,
                        "rounded_product": "0x%08X" % mul,
                        "note": "a*b = 1-2^-42 rounds to 1.0; acc = 3*2^-24 = 1.5 ulp. "
                                "Unfused hits the exact tie -> ties-to-even 2 ulp; fused is "
                                "just BELOW the tie -> 1 ulp. A ONE-BIT difference decides "
                                "whether vmula is an FMA. This is CONSTRUCTIVE, not a random search."},
          "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    cases.append(m1)
    m2 = {"name": "vmula_inactive_lanes", "kind": "vmula",
          "src0_bits": ["0x%08X" % A] * LANES, "src1_bits": ["0x%08X" % B] * LANES,
          "acc_in_bits": ["0x%08X" % ((0x40400000 + i) & 0xFFFFFFFF) for i in range(LANES)],
          "mask": mask_vector(list(range(0, 32))),
          "predicted": {"MODE_ZEROING": "lanes 32..63 = 0x00000000",
                        "MODE_MERGING": "lanes 32..63 = acc_in_bits (unchanged)",
                        "note": "inactive lanes must not be silently written; acc is READ|WRITE "
                                "(simd.py:635-646, simd_inst.h:404-406)."},
          "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    cases.append(m2)
    m3 = {"name": "vmula_loop_carry_ftz", "kind": "vmula",
          "src0_bits": ["0x%08X" % b for b in ([ONE] * 32 + [0x00000001] * 32)],
          "src1_bits": ["0x%08X" % b for b in ([ONE] * 32 + [0x00000001] * 32)],
          "acc_in_bits": ["0x%08X" % ZERO] * LANES,
          "mask": mask_vector(list(range(LANES))),
          "predicted": {"after_1_vmula_ieee": "0x3F800000 in lanes 0..31; 0x00000000 in lanes 32..63 "
                                             "(2^-149 squared underflows to zero even in IEEE fp32)",
                        "ftz_daz": "DAZ on the f32 subnormal src also gives zero; add a case "
                                   "with src=0x00800000 (smallest normal) to separate DAZ from FTZ",
                        "loop_carry": "repeat vmula 3 times; acc must be READ then WRITE each "
                                      "iteration (a constant-0 reducer would be killed)"},
          "status": "UNSOURCED / UNVALIDATED (model prediction, NOT device observation)"}
    cases.append(m3)
    return {"pinned_build": PINNED, "lanes": LANES,
            "disclaimer": ("All 'predicted' values are CPU MODEL outputs for HYPOTHESES about the "
                           "selected build. None is a native contract, a device observation, or "
                           "a validation result. Status: UNSOURCED / UNVALIDATED until WBS 03-20 "
                           "runs them in an approved V09 window and architect rules."),
            "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    args = ap.parse_args()
    data = build_cases()
    text = json.dumps(data, indent=1)
    with open(args.emit, "w") as fh:
        fh.write(text + "\n")
    print("cases:", len(data["cases"]), "lanes:", LANES,
          "status:", data["pinned_build"]["status"])
    print("json sha256:", hashlib.sha256((text + "\n").encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
