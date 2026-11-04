#!/usr/bin/env python3
"""Direct ELEMENTWISE diff of persisted device tensors against the INDEPENDENT numpy
reference (independent_reference.py), per case and per pattern.

The runner saves <key>.device.npy (device output, float32 of bf16) per pattern. This
loads those, rebuilds the numpy fp64->bf16 reference for the same inputs, saves
<key>.ref_numpy.npy, and reports max_abs, max_rel, mismatch count, whether max_abs <=
atol (explicitly, and never claimed when false), and whether deterministic patterns are
bit-exact.

Exit: 0 all cases pass, 1 any mismatch, 2 usage/IO error.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import independent_reference as IR

ATOL, RTOL = IR.ATOL, IR.RTOL
DETERMINISTIC = {"zero", "ones", "alternating"}


def key_of(pattern, seed):
    return pattern if pattern != "random" else "random_s%d" % seed


def diff_case(cases_dir, case_id, rows, d, loop_mode):
    cdir = os.path.join(cases_dir, case_id)
    out = {"case_id": case_id, "shape": [rows, d], "loop_mode": loop_mode,
           "atol": ATOL, "rtol": RTOL, "patterns": []}
    if not os.path.isdir(cdir):
        out["error"] = "missing case dir"
        return out
    for pattern, seed in IR.PATTERNS:
        key = key_of(pattern, seed)
        dev_path = os.path.join(cdir, key + ".device.npy")
        if not os.path.isfile(dev_path):
            out["patterns"].append({"pattern": pattern, "seed": seed, "key": key,
                                    "error": "no " + key + ".device.npy"})
            continue
        device = np.load(dev_path).astype(np.float64)
        x_bf, w_bf = IR.make_input((rows, d), pattern, seed)
        exact, ref = IR.numpy_reference(x_bf, w_bf)   # ref is float32 of bf16
        ref64 = ref.astype(np.float64)
        np.save(os.path.join(cdir, key + ".ref_numpy.npy"), ref.astype(np.float32))
        if device.shape != ref64.shape:
            out["patterns"].append({"pattern": pattern, "seed": seed, "key": key,
                                    "error": "shape %s vs %s" % (device.shape, ref64.shape)})
            continue
        diff = np.abs(device - ref64)
        tol = ATOL + RTOL * np.abs(ref64)
        mism = int(np.count_nonzero(diff > tol))
        max_abs = float(diff.max())
        max_rel = float((diff / np.maximum(np.abs(ref64), 1e-12)).max())
        det = pattern in DETERMINISTIC
        rec = {"pattern": pattern, "seed": seed, "key": key,
               "max_abs": max_abs, "max_rel": max_rel, "mismatch_count": mism,
               "atol": ATOL, "rtol": RTOL,
               "max_abs_le_atol": bool(max_abs <= ATOL),
               "deterministic": det,
               "pass": (mism == 0) and ((max_abs == 0.0) if det else True),
               "pass_basis": ("max_abs == 0 (deterministic, exact required)" if det
                              else ("within atol alone (max_abs <= atol)" if max_abs <= ATOL
                                    else "PASS on rtol wing only: max_abs > atol"))}
        out["patterns"].append(rec)
    ok = all(p.get("pass") for p in out["patterns"] if "error" not in p) and \
        all("error" not in p for p in out["patterns"])
    out["case_pass"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = {"reference": "numpy float64 rmsnorm on bf16 inputs -> bf16 (independent_reference)",
              "atol": ATOL, "rtol": RTOL, "cases": []}
    all_ok = True
    for case_id, rows, d, loop_mode in IR.CASES:
        c = diff_case(args.cases_dir, case_id, rows, d, loop_mode)
        all_ok = all_ok and bool(c.get("case_pass"))
        report["cases"].append(c)
    report["all_cases_pass"] = all_ok
    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w").write(text)
    print(text)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
