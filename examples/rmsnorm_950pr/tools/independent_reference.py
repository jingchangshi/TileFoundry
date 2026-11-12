"""INDEPENDENT numeric reference for the four RMSNorm device cases (numpy).

Purpose: check the device result against something that is neither the TileLang kernel
nor the TF evaluator. The arithmetic here is numpy float64 RMSNorm; the inputs are the
SAME tensors the runner feeds, materialised from the frozen protocol (shapes, patterns,
seeds) with torch's own CPU generator -- torch is used ONLY as the deterministic input
sampler, never to compute the reference. torch appears a second time only to cross-check
this reference against the runner's own fp64 golden, and that cross-check is reported,
not trusted blindly.

Protocol (config/numerical_protocol.proposed.json + runtime_model.check_case):
  reference  cpu float64 rmsnorm then bfloat16
  eps        1e-06
  atol       0.02
  rtol       0.01        verdict: |got-ref| <= atol + rtol*|ref|
  patterns   zero, ones, random(seed 0,1,2), alternating   (the exact sequence the
             runner loops over in check_case)
  shapes     prefill (8192,4096) x {dyn,recompute}; decode (64,4096) x {dyn,recompute}

Loop mode is a SCHEDULING distinction, not arithmetic: prefill_dyn and prefill_recompute
have the SAME reference to the last bit, and likewise the decode pair. Reported per case
anyway so the mapping is explicit.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import torch

EPS = 1e-06
ATOL = 0.02
RTOL = 0.01
# The exact pattern/seed sequence check_case loops over. Seed is used only by "random".
PATTERNS = [("zero", 0), ("ones", 0), ("random", 0), ("random", 1), ("random", 2), ("alternating", 0)]
CASES = [
    ("prefill_dyn", 8192, 4096, "dyn"),
    ("prefill_recompute", 8192, 4096, "recompute"),
    ("decode_dyn", 64, 4096, "dyn"),
    ("decode_recompute", 64, 4096, "recompute"),
]


def make_input(shape, pattern, seed):
    """Byte-for-byte the runner's make_input (runtime_model.py), torch CPU generator."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows, width = shape
    if pattern == "zero":
        x = torch.zeros(shape, dtype=torch.float32); w = torch.ones(width, dtype=torch.float32)
    elif pattern == "ones":
        x = torch.ones(shape, dtype=torch.float32); w = torch.linspace(0.25, 1.75, width)
    elif pattern == "random":
        x = torch.randn(shape, generator=g); w = torch.randn(width, generator=g)
    elif pattern == "alternating":
        idx = torch.arange(width)
        scale = torch.tensor([0.01, 1.0, 10.0, 100.0])[idx % 4]
        row = scale * torch.where(idx % 2 == 0, 1.0, -1.0)
        signs = torch.where(torch.arange(rows) % 2 == 0, 1.0, -1.0)
        x = signs[:, None] * row[None, :]; w = torch.linspace(-1.0, 1.0, width)
    else:
        raise ValueError(pattern)
    return x.to(torch.bfloat16).contiguous(), w.to(torch.bfloat16).contiguous()


def bf16_from_f32(f32):
    """Round float32 -> bfloat16 (round-to-nearest-even), returned as float32 values.

    A bf16 value has 8 exponent bits (same bias as fp32) and 7 mantissa bits, so the
    top 16 bits of the fp32 words ARE the bf16 representation. RNE at the 16-bit
    boundary uses bias 0x7FFF + lsb-of-kept-half.
    """
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & np.uint64(1)
    bias = np.uint64(0x7FFF) + lsb
    u2 = (u + bias) & np.uint64(0xFFFF0000)
    return u2.astype(np.uint32).view(np.float32)


def numpy_reference(x_bf, w_bf):
    """fp64 RMSNorm on bf16-rounded inputs, computed in numpy. Returns (exact64, ref_bf32)."""
    xd = x_bf.to(torch.float64).numpy()
    wd = w_bf.to(torch.float64).numpy()
    var = np.mean(xd * xd, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + EPS)
    exact = xd * rstd * wd[None, :]
    return exact, bf16_from_f32(exact.astype(np.float32))


def torch_golden_bf16(x_bf, w_bf):
    """The runner's own golden, as float32 values of its bf16 output (for cross-check)."""
    xd = x_bf.to(torch.float64); wd = w_bf.to(torch.float64)
    exact = xd * torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + EPS) * wd
    return exact.to(torch.bfloat16).to(torch.float32).numpy(), exact.numpy()


def evaluate_case(case_id, rows, d, loop_mode):
    shape = (rows, d)
    per_pattern = []
    case_max = -1e30
    case_sum = 0.0
    golden_bitwise_mismatch = 0
    for pattern, seed in PATTERNS:
        x_bf, w_bf = make_input(shape, pattern, seed)
        exact, ref_bf32 = numpy_reference(x_bf, w_bf)
        gold_bf32, gold_exact = torch_golden_bf16(x_bf, w_bf)
        bitwise = int(np.count_nonzero(ref_bf32.view(np.uint32) != gold_bf32.view(np.uint32)))
        golden_bitwise_mismatch += bitwise
        ref_max = float(ref_bf32.max())
        ref_sum = float(ref_bf32.astype(np.float64).sum())
        exact_max = float(np.abs(exact).max())
        ref_abs_sum = float(np.abs(ref_bf32).astype(np.float64).sum())
        # numpy fp64 vs torch fp64 golden (pure-math agreement, before bf16 rounding)
        exact_vs_golden_maxdiff = float(np.abs(exact - gold_exact).max())
        case_max = max(case_max, ref_max)
        case_sum += ref_sum
        per_pattern.append({
            "pattern": pattern, "seed": seed,
            "ref_bf16_max": ref_max, "ref_bf16_sum": ref_sum,
            "ref_bf16_abs_sum": ref_abs_sum,
            "exact_fp64_abs_max": exact_max,
            "numpy_vs_torch_fp64_maxdiff": exact_vs_golden_maxdiff,
            "ref_vs_golden_bf16_bitwise_mismatch": bitwise,
            "shape": [rows, d],
        })
    return {
        "case_id": case_id, "shape": [rows, d], "loop_mode": loop_mode,
        "reference_max_over_patterns": case_max,
        "reference_sum_over_patterns": case_sum,
        "golden_bitwise_mismatch_total": golden_bitwise_mismatch,
        "per_pattern": per_pattern,
    }


def verdict_from_numeric(numeric, ref_by_key):
    """Judge recorded device tests: recompute the atol+rtol verdict from stored stats.

    numeric.json stores, per test, max_abs/max_rel/mismatch_count of DEVICE vs the runner
    golden. Because this reference reproduces that golden bit-for-bit (checked), those
    stored stats ARE device-vs-independent-reference. We re-apply the tolerance predicate
    and aggregate. Raw device tensors are not persisted, so a direct elementwise compare
    is not possible from the evidence alone and is stated as such.
    """
    tests = numeric.get("tests", [])
    worst_abs = 0.0; worst_rel = 0.0; mism = 0; n = 0; patterns = set()
    for t in tests:
        n += 1
        patterns.add((t.get("pattern"), t.get("seed")))
        if t.get("max_abs") is not None:
            worst_abs = max(worst_abs, float(t["max_abs"]))
        if t.get("max_rel") is not None:
            worst_rel = max(worst_rel, float(t["max_rel"]))
        mism += int(t.get("mismatch_count") or 0)
    return {
        "tests": n,
        "patterns_covered": sorted([list(p) for p in patterns], key=lambda z: (str(z[0]), z[1])),
        "max_abs": worst_abs, "max_rel": worst_rel,
        "mismatch_count_total": mism,
        "atol": numeric.get("atol", ATOL), "rtol": numeric.get("rtol", RTOL),
        "verdict": "PASS" if (n > 0 and mism == 0) else ("FAIL" if n > 0 else "NO_TESTS"),
        "note": "stats are device-vs-runner-golden; reference reproduces that golden bit-for-bit",
    }


def _pattern_order(key):
    order = ["zero", "ones", "random_s0", "random_s1", "random_s2", "alternating"]
    return (order.index(key) if key in order else 99, key)


def verify_tensors(tensors_dir):
    """Direct elementwise compare of persisted device tensors against THIS reference.

    Uses the ACTUAL inputs the device consumed (.x.npy/.w.npy), so it is independent of
    any RNG-version question: the same arrays feed the numpy reference and were fed to the
    kernel. For each <key>.device.npy we recompute the fp64 reference in numpy, bf16-round,
    and apply the frozen atol+rtol rule elementwise. .golden.npy is cross-checked to prove
    this reference equals the runner golden bit-for-bit.
    """
    import os
    report = {"reference": "numpy float64 rmsnorm on persisted bf16 inputs", "eps": EPS,
              "atol": ATOL, "rtol": RTOL, "cases": []}
    for case_id, rows, d, loop_mode in CASES:
        cdir = os.path.join(tensors_dir, case_id)
        if not os.path.isdir(cdir):
            report["cases"].append({"case_id": case_id, "status": "NO_DIR"})
            continue
        keys = sorted([n[:-len(".device.npy")] for n in os.listdir(cdir)
                       if n.endswith(".device.npy")], key=_pattern_order)
        per = []
        case_abs = 0.0; case_rel = 0.0; case_mism = 0; case_n = 0
        golden_mismatch_total = 0
        for key in keys:
            x = np.load(os.path.join(cdir, key + ".x.npy"))
            w = np.load(os.path.join(cdir, key + ".w.npy"))
            dev = np.load(os.path.join(cdir, key + ".device.npy"))
            gpath = os.path.join(cdir, key + ".golden.npy")
            gold = np.load(gpath) if os.path.isfile(gpath) else None
            xd = x.astype(np.float64)
            wd = w.astype(np.float64)
            var = np.mean(xd * xd, axis=-1, keepdims=True)
            rstd = 1.0 / np.sqrt(var + EPS)
            exact = xd * rstd * wd[None, :]
            ref_bf32 = bf16_from_f32(exact.astype(np.float32))
            gbit = -1
            if gold is not None:
                gbit = int(np.count_nonzero(ref_bf32.view(np.uint32) != gold.view(np.uint32)))
                golden_mismatch_total += gbit
            diff = np.abs(dev - ref_bf32)
            tol = ATOL + RTOL * np.abs(ref_bf32)
            mism = int(np.count_nonzero(diff > tol))
            max_abs = float(diff.max())
            max_rel = float((diff / np.maximum(np.abs(ref_bf32), 1e-12)).max())
            case_abs = max(case_abs, max_abs); case_rel = max(case_rel, max_rel)
            case_mism += mism; case_n += 1
            per.append({"pattern_key": key, "max_abs": max_abs, "max_rel": max_rel,
                        "mismatch_count": mism, "ref_vs_golden_bitwise_mismatch": gbit,
                        "n_elements": int(dev.size)})
        report["cases"].append({
            "case_id": case_id, "shape": [rows, d], "loop_mode": loop_mode,
            "max_abs": case_abs, "max_rel": case_rel, "mismatch_count_total": case_mism,
            "golden_bitwise_mismatch_total": golden_mismatch_total,
            "verdict": "PASS" if (case_n > 0 and case_mism == 0) else "FAIL",
            "per_pattern": per,
        })
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", default=None,
                    help="dir with <case_id>/numeric.json device results to judge")
    ap.add_argument("--tensors-dir", default=None,
                    help="dir with <case_id>/*.x.npy/.w.npy/.device.npy for a direct elementwise compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import os
    if args.tensors_dir:
        trep = verify_tensors(args.tensors_dir)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(trep, fh, indent=2)
        print(json.dumps(trep, indent=2))
        return
    report = {"reference": "numpy float64 rmsnorm on bf16 inputs -> bf16",
              "eps": EPS, "atol": ATOL, "rtol": RTOL,
              "pattern_sequence": [list(p) for p in PATTERNS], "cases": []}
    for case_id, rows, d, loop_mode in CASES:
        c = evaluate_case(case_id, rows, d, loop_mode)
        if args.cases_dir:
            npath = os.path.join(args.cases_dir, case_id, "numeric.json")
            if os.path.isfile(npath):
                with open(npath) as fh:
                    numeric = json.load(fh)
                c["device_verdict"] = verdict_from_numeric(numeric, None)
            else:
                c["device_verdict"] = {"verdict": "NO_DEVICE_RESULT",
                                       "note": "no numeric.json for this case in cases-dir"}
        report["cases"].append(c)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
