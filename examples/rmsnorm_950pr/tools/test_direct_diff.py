#!/usr/bin/env python3
"""06-06 discriminating negatives for the hardened direct_diff tool.

Positive control first, then single-defect mutants that MUST fail/refuse:
  det_one_ulp     one bf16 codeword (0.0078, INSIDE atol) on a deterministic pattern
                  -> mismatch 0 but DETERMINISTIC_BITS_FAILED (bit exactness != tolerance)
  reuse_bad       numeric.output_buffer_reuse.values_equal=false -> REUSE_VALUES_MISMATCH
  reuse_same_ptr  storage not distinct -> REUSE_STORAGE_NOT_DISTINCT
  stability_bad   stability.all_bitwise_equal_first=false -> STABILITY_FAILED
  repeats_bad     one repeat_bitwise_equal=false -> REPEATS_NOT_BITWISE_EQUAL
  nonfinite       device element = inf -> NONFINITE_OUTPUT
  empty_set       tensors=[] -> REFUSED EMPTY_PATTERN_SET (exit 2)
  missing_file    device.npy removed -> REFUSED TENSOR_FILE_MISSING
  unfrozen        protocol status DRAFT -> REFUSED PROTOCOL_NOT_FROZEN
  rtol_wing       random pattern with error > atol but < tol -> PASS, basis rtol wing only

Run: python examples/rmsnorm_950pr/tools/test_direct_diff.py --out-dir <dir>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TF_ROOT = HERE.parents[2]
sys.path.insert(0, str(TF_ROOT / "tools" / "p3"))
from rmsnorm_oracle import bf16_from_f64, bf16_to_f32  # noqa: E402

TOOL = HERE / "direct_diff.py"
PROTOCOL = TF_ROOT.parent / "TileFoundry_Ascend950PR_AgentTeam/artifacts/run01/p0/precision_protocol.json"
SHAPE = (4, 128)
FAILS = 0
COUNTS = {"positive": 0, "negative": 0}


def check(cond, label, group="negative"):
    global FAILS
    COUNTS[group] += 1
    if not cond:
        FAILS += 1
    print("%-96s %s" % (label, "PASS" if cond else "FAIL"))


def bits_of(arr):
    return (np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def make_case(base: Path, name="decode_selftest"):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    rows, width = SHAPE
    eps = 1e-6
    entries = []
    rng = np.random.default_rng(0)
    for pattern, seed in (("zero", 0), ("ones", 0), ("alternating", 0), ("random", 0)):
        if pattern == "zero":
            x = np.zeros((rows, width), np.float32); w = np.ones(width, np.float32)
        elif pattern == "ones":
            x = np.ones((rows, width), np.float32); w = np.ones(width, np.float32)
        elif pattern == "alternating":
            idx = np.arange(width)
            x = np.where(idx % 2 == 0, 1.0, -1.0).astype(np.float32)
            x = np.tile(x, (rows, 1)) * np.where(np.arange(rows) % 2 == 0, 1.0, -1.0)[:, None].astype(np.float32)
            w = np.linspace(0.25, 1.75, width).astype(np.float32)
        else:
            x = rng.standard_normal((rows, width)).astype(np.float32)
            w = rng.standard_normal(width).astype(np.float32)
        x = bf16_to_f32(bits_of(x)); w = bf16_to_f32(bits_of(w))
        xd = x.astype(np.float64); wd = w.astype(np.float64)
        mean = np.sum(xd * xd, axis=-1, keepdims=True) / width
        y = xd * (1.0 / np.sqrt(mean + eps)) * wd
        ref_bits = bf16_from_f64(y)
        dev = bf16_to_f32(ref_bits)
        key = "random_s0" if pattern == "random" else pattern
        np.save(d / (key + ".x.npy"), x); np.save(d / (key + ".w.npy"), w)
        np.save(d / (key + ".golden.npy"), dev.copy())
        np.save(d / (key + ".device.npy"), dev.copy())
        entries.append({"key": key, "pattern": pattern, "seed": seed,
                        "files": [key + ".device.npy", key + ".golden.npy",
                                  key + ".x.npy", key + ".w.npy"]})
    (d / "tensors.json").write_text(json.dumps({"case_id": name, "shape": list(SHAPE),
                                                "tensors": entries}, indent=2))
    tests = []
    for e in entries:
        for rep in range(4):
            tests.append({"pattern": e["pattern"], "seed": e["seed"], "repetition": rep,
                          "repeat_bitwise_equal": (None if rep == 0 else True),
                          "passed": True, "mismatch_count": 0})
    numeric = {"case_id": name, "shape": list(SHAPE),
               "tests": tests,
               "stability": {"runs": 64, "all_bitwise_equal_first": True, "fixed_input": True},
               "output_buffer_reuse": {"device_ptr_first": 1000, "device_ptr_second": 2000,
                                        "device_storage_distinct": True, "values_equal": True}}
    (d / "numeric.json").write_text(json.dumps(numeric, indent=2))
    return d


def run(cdir, protocol=PROTOCOL, out=None):
    cmd = [sys.executable, str(TOOL), "--case-dir", str(cdir), "--protocol", str(protocol)]
    if out:
        cmd += ["--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env={"PYTHONPATH": str(TF_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                            "PATH": "/usr/bin:/bin"})
    data = {}
    if out and Path(out).is_file():
        data = json.loads(Path(out).read_text())
    return p, data


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        pos = make_case(base, "decode_pos")
        p, data = run(pos, out=outdir / "pos.json")
        check(p.returncode == 0 and data.get("verdict") == "PASS",
              "positive control: clean fixture -> exit 0 PASS (got %d)" % p.returncode, "positive")
        check(all(x["protocol"]["mismatch_count"] == 0 for x in data["cases"][0]["patterns"]),
              "positive control: zero protocol mismatches", "positive")
        check(data["cases"][0]["repeats_stability_reuse"]["pass"],
              "positive control: repeats/stability/reuse pass", "positive")

        def mutated(name, fn):
            d = make_case(base, "decode_" + name)
            fn(d)
            p, data = run(d, out=outdir / (name + ".json"))
            text = p.stdout + p.stderr + json.dumps(data)
            return p.returncode, text, data

        def edit_npy(key, suffix, fn):
            def mut(d):
                q = d / (key + suffix); a = np.load(q); np.save(q, fn(a))
            return mut

        def det_one_ulp(a):
            b = a.copy(); b[0, 0] = np.float32(b[0, 0] + np.float32(2.0 ** -7)); return b
        rc, text, data = mutated("det_one_ulp", edit_npy("ones", ".device.npy", det_one_ulp))
        check(rc == 1 and "DETERMINISTIC_BITS_FAILED" in text,
              "det 1-bf16-ULP inside atol -> FAIL DETERMINISTIC_BITS_FAILED (rc=%d)" % rc)

        def reuse_bad(d):
            n = json.loads((d / "numeric.json").read_text()); n["output_buffer_reuse"]["values_equal"] = False
            (d / "numeric.json").write_text(json.dumps(n))
        rc, text, _ = mutated("reuse_bad", reuse_bad)
        check(rc == 1 and "REUSE_VALUES_MISMATCH" in text, "reuse bad value -> FAIL REUSE_VALUES_MISMATCH (rc=%d)" % rc)

        def reuse_same(d):
            n = json.loads((d / "numeric.json").read_text())
            n["output_buffer_reuse"]["device_storage_distinct"] = False
            n["output_buffer_reuse"]["device_ptr_second"] = n["output_buffer_reuse"]["device_ptr_first"]
            (d / "numeric.json").write_text(json.dumps(n))
        rc, text, _ = mutated("reuse_same", reuse_same)
        check(rc == 1 and "REUSE_STORAGE_NOT_DISTINCT" in text, "reuse same ptr -> FAIL REUSE_STORAGE_NOT_DISTINCT (rc=%d)" % rc)

        def stability_bad(d):
            n = json.loads((d / "numeric.json").read_text()); n["stability"]["all_bitwise_equal_first"] = False
            (d / "numeric.json").write_text(json.dumps(n))
        rc, text, _ = mutated("stability_bad", stability_bad)
        check(rc == 1 and "STABILITY_FAILED" in text, "stability false -> FAIL STABILITY_FAILED (rc=%d)" % rc)

        def repeats_bad(d):
            n = json.loads((d / "numeric.json").read_text()); n["tests"][1]["repeat_bitwise_equal"] = False
            (d / "numeric.json").write_text(json.dumps(n))
        rc, text, _ = mutated("repeats_bad", repeats_bad)
        check(rc == 1 and "REPEATS_NOT_BITWISE_EQUAL" in text, "repeat not bitwise equal -> FAIL (rc=%d)" % rc)

        rc, text, _ = mutated("nonfinite", edit_npy("zero", ".device.npy",
                                                    lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.float32(np.inf), a)))
        check(rc == 1 and "NONFINITE_OUTPUT" in text, "nonfinite device output -> FAIL NONFINITE_OUTPUT (rc=%d)" % rc)

        def empty(d):
            t = json.loads((d / "tensors.json").read_text()); t["tensors"] = []
            (d / "tensors.json").write_text(json.dumps(t))
        rc, text, _ = mutated("empty", empty)
        check(rc == 2 and "EMPTY_PATTERN_SET" in text, "empty pattern set -> REFUSED exit2 (rc=%d)" % rc)

        rc, text, _ = mutated("missing_file", lambda d: (d / "ones.device.npy").unlink())
        check(rc == 2 and "TENSOR_FILE_MISSING" in text, "missing device file -> REFUSED exit2 (rc=%d)" % rc)

        bad_proto = base / "unfrozen.json"
        bad_proto.write_text(json.dumps(dict(json.loads(PROTOCOL.read_text()), status="DRAFT")))
        p, data = run(make_case(base, "decode_unfrozen"), protocol=bad_proto, out=outdir / "unfrozen.json")
        check(p.returncode == 2 and "PROTOCOL_NOT_FROZEN" in (p.stdout + p.stderr + json.dumps(data)),
              "unfrozen protocol -> REFUSED exit2 (rc=%d)" % p.returncode)

        # rtol wing: random element error > atol but < tol
        d = make_case(base, "decode_rtol")
        a = np.load(d / "random_s0.device.npy").copy()
        idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
        bumped = np.float32(a[idx] + np.float32(0.04))
        a[idx] = bf16_to_f32(bits_of(np.array([bumped], dtype=np.float32)))[0]
        np.save(d / "random_s0.device.npy", a)
        p, data = run(d, out=outdir / "rtol.json")
        rec = next(x for x in data["cases"][0]["patterns"] if x["key"] == "random_s0")
        check(p.returncode == 0 and rec["protocol"]["max_abs"] > 0.02
              and rec["protocol"]["mismatch_count"] == 0 and "rtol wing" in rec["pass_basis"],
              "rtol wing: max_abs>atol, mismatch 0, basis names rtol wing (rc=%d)" % p.returncode)

    total = COUNTS["positive"] + COUNTS["negative"]
    print("positive=%d negative=%d total=%d" % (COUNTS["positive"], COUNTS["negative"], total))
    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
