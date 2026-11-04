#!/usr/bin/env python3
"""Mutation tests for check_06_05.py (PLAN_ONLY). Each stage must be able to fail."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TF = Path(__file__).resolve().parents[3]
ARCH = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/device/archive/cases_audited_20261104T202316/decode_dyn")
AIBIN = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run03/w15/06-04/run/decode_dyn.cce.aibin")
ENV_OK = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run03/w16/06-05/fixtures/env_ok.json")
DRIVER = TF / "examples/rmsnorm_950pr/tools/check_06_05.py"
PY = sys.executable
ROOT = Path(tempfile.mkdtemp(prefix="check0605_"))
results = []


def bump_bf16(path: Path):
    a = np.load(path)
    b = (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    b.reshape(-1)[0] = (int(b.reshape(-1)[0]) + 1) & 0xFFFF
    out = (b.astype(np.uint32) << np.uint32(16)).view(np.float32)
    np.save(path, out)


def run(name, mutate=None, extra=None, env_json=ENV_OK):
    cdir = ROOT / name / "case"
    shutil.copytree(ARCH, cdir)
    for p in cdir.rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    if mutate:
        mutate(cdir)
    out = ROOT / name / "out"
    cmd = [PY, str(DRIVER), "--mode", "plan", "--case-dir", str(cdir), "--aibin", str(AIBIN),
           "--env-json", str(env_json), "--out", str(out)] + (extra or [])
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    phases = json.loads((out / "phases.json").read_text())
    first_fail = next((s for s, v in phases["stages"].items()
                       if v["status"] in ("FAIL", "REFUSED")), None)
    code = phases["stages"][first_fail]["code"] if first_fail else None
    results.append({"name": name, "rc": p.returncode, "fail_stage": first_fail, "code": code})
    print("%-16s rc=%s stage=%s code=%s" % (name, p.returncode, first_fail, code))
    return p.returncode, first_fail, code


def m_golden(cdir):
    bump_bf16(cdir / "zero.golden.npy")


def m_det(cdir):
    bump_bf16(cdir / "zero.device.npy")


def m_reuse(cdir):
    p = cdir / "numeric.json"
    d = json.loads(p.read_text())
    d.setdefault("output_buffer_reuse", {})["values_equal"] = False
    p.write_text(json.dumps(d, indent=2))


def m_empty(cdir):
    p = cdir / "tensors.json"
    d = json.loads(p.read_text())
    d["tensors"] = []
    p.write_text(json.dumps(d, indent=2))


run("baseline")
run("bad_aibin_sha", extra=["--expected-aibin-sha", "deadbeef" * 8])
run("bad_golden", m_golden)
run("bad_det", m_det)
run("bad_reuse", m_reuse)
run("empty_set", m_empty)
run("device_gate", extra=["--mode", "device"])

expect = {"baseline": (0, None), "bad_aibin_sha": (2, "CCA_SHA_MISMATCH"),
          "bad_golden": (1, "INDEPENDENT_FAILED_UNKNOWN"), "bad_det": (1, "DETERMINISTIC_BITS_FAILED"),
          "bad_reuse": (1, "REUSE_VALUES_MISMATCH"), "empty_set": (2, "EMPTY_PATTERN_SET"),
          "device_gate": (2, "DEVICE_GATE")}
ok = 0
for r in results:
    want_rc, want_code = expect[r["name"]]
    if r["name"] == "bad_golden":
        good = r["rc"] == 1 and r["code"] in ("INPUT_IDENTITY_FAILED", "INPUT_IDENTITY_UNCHECKED")
    elif r["name"] == "baseline":
        good = r["rc"] == 0
    else:
        good = r["rc"] == want_rc and r["code"] == want_code
    r["ok"] = good
    ok += good
print("RESULT positive=1 negative=%d total=%d all_ok=%s" %
      (len(results) - 1, len(results), ok == len(results)))
sys.exit(0 if ok == len(results) else 1)
