#!/usr/bin/env python3
"""Discriminating negatives for the 03-21 actual-input ABI.

A run-dir fixture is built in the runner's own layout (tensors.json + X[M,D]/W[D]/
golden/device .npy). The UNMODIFIED fixture must exit 0. Every negative changes ONE
thing and must exit NON-zero with its NAMED code in MANIFEST.json; if the check under
test were removed (mutated to a no-op), that same negative would become a pass. The
positive control is therefore run first and its exit is asserted, so a broken oracle
that fails everything cannot be mistaken for a working set of negatives.

Run: python tools/p3/test_input_abi.py   (exit 0 iff the positive passes and every
negative fails with the code named below)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF_ROOT = HERE.parents[1]
ORACLE = HERE / "rmsnorm_oracle.py"
TEAM = TF_ROOT.parent / "TileFoundry_Ascend950PR_AgentTeam"
PROTOCOL = TEAM / "artifacts/run01/p0/precision_protocol.json"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
           PYTHONPATH=str(HERE))
KEYS = ("zero", "ones", "random_s0")
SHAPE = (8, 128)
FAILS = 0
# Counters are incremented by check() ITSELF in this run: the tail line is produced by
# the object under test, not copied from a report (HANDOFF §6: a self-description is
# not evidence). "positive" = assertions on the untouched control fixture;
# "negative" = single-defect fault injections that must be refused/failed by name.
COUNTS = {"positive": 0, "negative": 0}


def check(cond: bool, label: str, group: str = "negative") -> None:
    global FAILS
    COUNTS[group] += 1
    if not cond:
        FAILS += 1
    print("%-86s %s" % (label, "PASS" if cond else "FAIL"))


def runner_arrays(torch, pattern, rows, width):
    import numpy as np
    if pattern == "zero":
        x = torch.zeros(rows, width); w = torch.ones(width)
    elif pattern == "ones":
        x = torch.ones(rows, width); w = torch.ones(width)
    else:
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((rows, width)).astype(np.float32))
        w = torch.from_numpy(rng.standard_normal(width).astype(np.float32))
    x = x.to(torch.bfloat16).to(torch.float32)
    w = w.to(torch.bfloat16).to(torch.float32)
    xd = x.to(torch.float64); wd = w.to(torch.float64)
    eps = 1e-6
    exact = xd * torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + eps) * wd
    golden = exact.to(torch.bfloat16).to(torch.float32)
    return x.numpy(), w.numpy(), golden.numpy()


def make_fixture(base: Path, case_id="decode_selftest", shape=SHAPE, keys=KEYS):
    import numpy as np
    import torch
    d = base / case_id
    d.mkdir(parents=True, exist_ok=True)
    entries = []
    for key in keys:
        pat = "random" if key.startswith("random_") else key
        x, w, g = runner_arrays(torch, pat, shape[0], shape[1])
        np.save(d / (key + ".x.npy"), x)
        np.save(d / (key + ".w.npy"), w)
        np.save(d / (key + ".golden.npy"), g)
        np.save(d / (key + ".device.npy"), g.copy())
        entries.append({"key": key, "pattern": pat, "seed": 0,
                        "files": [key + ".device.npy", key + ".golden.npy",
                                  key + ".x.npy", key + ".w.npy"]})
    (d / "tensors.json").write_text(json.dumps(
        {"case_id": case_id, "shape": list(shape),
         "convention": "<key>.device.npy = device output as float32 of bf16; "
                       "<key>.golden.npy = runner fp64->bf16 reference; "
                       "<key>.x/.w = bf16 inputs as float32",
         "tensors": entries}, indent=2))
    return d


def invoke(run_dir: Path, out: Path, protocol: Path = PROTOCOL):
    cmd = [sys.executable, str(ORACLE), "--run-dir", str(run_dir),
           "--out", str(out), "--protocol", str(protocol)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    mf = out / "MANIFEST.json"
    data = json.loads(mf.read_text()) if mf.is_file() else {}
    text = proc.stdout + proc.stderr + json.dumps(data)
    return proc.returncode, data, text


def main() -> int:
    global FAILS
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # 1. POSITIVE CONTROL: the untouched runner-shaped fixture must pass.
        pos = make_fixture(base)
        rc, mf, _ = invoke(pos, base / "out_pos")
        check(rc == 0 and mf.get("verdict") == "PASS",
              "positive control: real runner-shaped inputs -> exit 0 (got %s)" % rc,
              group="positive")
        check(len(mf.get("runs", [{}])[0].get("patterns", [])) == len(KEYS),
              "positive control: all %d patterns consumed, no empty-set pass" % len(KEYS),
              group="positive")
        check(all("sha256" in p["files"]["x"] for p in mf["runs"][0]["patterns"]),
              "positive control: every input file carries its sha256", group="positive")
        check(all(p["input_identity"]["golden_reproduced_bit_exact"]
                  for p in mf["runs"][0]["patterns"]),
              "positive control: golden reproduced bit-exactly from persisted X/W",
              group="positive")

        def negative(name, mutate, expected_code, expected_rc=(1, 2)):
            d = make_fixture(base, case_id="decode_" + name)
            mutate(d)
            rc, mf, text = invoke(d, base / ("out_" + name))
            check(rc in expected_rc and expected_code in text,
                  "%s -> exit in %r with %s (got %d, manifest %r)"
                  % (name, expected_rc, expected_code, rc,
                     mf.get("verdict") or mf.get("failures")))
            return d

        def edit(key, filename, fn):
            def mut(d):
                p = d / (key + filename)
                a = np.load(p)
                np.save(p, fn(a))
            return mut

        def rewrite_tj(fn):
            def mut(d):
                p = d / "tensors.json"
                t = json.loads(p.read_text())
                fn(t)
                p.write_text(json.dumps(t, indent=2))
            return mut

        # 2. The exact pre-03-21 defect: a ROW-WISE w is a different input.
        negative("row_wise_w",
                 edit("ones", ".w.npy", lambda a: np.repeat(a[None, :], SHAPE[0], axis=0)),
                 "ROW_WISE_W", (2,))
        negative("w_shape", edit("ones", ".w.npy", lambda a: a[:-1].copy()),
                 "W_SHAPE_MISMATCH", (2,))
        negative("x_shape", edit("ones", ".x.npy", lambda a: a.T.copy()),
                 "X_SHAPE_MISMATCH", (2,))
        negative("not_bf16",
                 edit("ones", ".x.npy", lambda a: np.where(
                     np.arange(a.size).reshape(a.shape) == 0,
                     np.float32(1.0 + 2.0 ** -9), a).astype(np.float32)),
                 "INPUT_NOT_BF16", (2,))

        def rm(name, key, suffix):
            def mut(d):
                (d / (key + suffix)).unlink()
            return negative(name, mut, {".device.npy": "DEVICE_OUTPUT_MISSING",
                                        ".golden.npy": "GOLDEN_MISSING"}[suffix], (2,))
        rm("no_device", "ones", ".device.npy")
        rm("no_golden", "ones", ".golden.npy")

        def sha_wrong(t):
            t["tensors"][0]["sha256"] = {"x": "0" * 64}
        negative("sha_mismatch", rewrite_tj(sha_wrong), "INPUT_SHA_MISMATCH", (2,))

        def empty(t):
            t["tensors"] = []
        negative("empty_set", rewrite_tj(empty), "EMPTY_TENSOR_SET", (2,))

        def unknown_case(t):
            t["case_id"] = "mystery_case"
        negative("chains_unknown", rewrite_tj(unknown_case), "CHAINS_UNKNOWN", (2,))

        # 3. INPUT IDENTITY: golden from DIFFERENT inputs cannot pass as "same input".
        negative("identity_failed",
                 edit("ones", ".golden.npy", lambda a: np.zeros_like(a)),
                 "INPUT_IDENTITY_FAILED", (1,))

        # 4. Wrong device target: one bf16 ULP on a DETERMINISTIC pattern. This is within
        # atol (0.02) but MUST still fail on raw bits, so the det check is not decorative.
        def one_ulp(a):
            b = a.copy()
            # one full bf16 ULP at 1.0 is 2^-7 (7 mantissa bits); 2^-8 is only
            # HALF an ulp and would be refused as a non-bf16 float32 by the ABI.
            b[0, 0] = np.float32(b[0, 0] + np.float32(2.0 ** -7))
            return b
        negative("det_one_ulp", edit("ones", ".device.npy", one_ulp),
                 "DETERMINISTIC_BITS_FAILED", (1,))

        def big_error(a):
            b = a.copy()
            # Set an exactly-bf16-representable value far outside the tolerance; a
            # non-bf16 float32 would be refused by the ABI instead, which is a
            # different (and intentional) check.
            b[0, 0] = np.float32(-1.0) if b[0, 0] >= 0 else np.float32(1.0)
            return b
        negative("protocol_failed", edit("random_s0", ".device.npy", big_error),
                 "DEVICE_PROTOCOL_FAILED", (1,))

        def no_tj(d):
            (d / "tensors.json").unlink()
        negative("no_tensors_json", no_tj, "TENSORS_JSON_MISSING", (2,))

        # 5. Unfrozen protocol must not be used to pass or fail a candidate.
        bad_protocol = base / "unfrozen.json"
        p = json.loads(PROTOCOL.read_text()); p["status"] = "DRAFT"
        bad_protocol.write_text(json.dumps(p))
        rc, mf, text = invoke(make_fixture(base, case_id="decode_protocol"),
                              base / "out_protocol", bad_protocol)
        check(rc == 2 and "PROTOCOL_NOT_FROZEN" in text,
              "unfrozen protocol -> REFUSED PROTOCOL_NOT_FROZEN (got %d)" % rc)

        # 6. Missing run-dir: no inputs means no judgement, NOT a vacuous pass.
        rc, mf, text = invoke(base / "does_not_exist", base / "out_missing")
        check(rc == 2 and "RUN_DIR_MISSING" in text,
              "missing run-dir -> REFUSED RUN_DIR_MISSING (got %d)" % rc)

    total = COUNTS["positive"] + COUNTS["negative"]
    print("positive=%d negatives=%d total=%d" % (COUNTS["positive"], COUNTS["negative"], total))
    # Guards against a vacuous/empty run being reported as a pass.
    if COUNTS["positive"] < 4 or COUNTS["negative"] < 15:
        FAILS += 1
    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
