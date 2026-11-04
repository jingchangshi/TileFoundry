#!/usr/bin/env python3
"""Mutation tests for plan_device_protocol.py (PLAN_ONLY, no device)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TF = Path(__file__).resolve().parents[3]
TOOL = TF / "examples/rmsnorm_950pr/tools/plan_device_protocol.py"
BASE = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run03/w18/06-07_10_prep/cases/decode_dyn")
PROTO = TF.parent / "TileFoundry_Ascend950PR_AgentTeam/artifacts/run01/p0/precision_protocol.json"
ROOT = Path(tempfile.mkdtemp(prefix="plan0607_"))
PY = sys.executable


def run(name, mutate=None, busy_policy=None, expect_code=None, expect_pass=False,
        expect_busy=False):
    d = ROOT / name
    shutil.copytree(BASE, d)
    for p in d.rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    if mutate:
        mutate(d)
    cmd = [PY, str(TOOL), "--validate-only", "--case", str(d), "--protocol", str(PROTO)]
    if busy_policy:
        cmd += ["--busy-policy", busy_policy]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    res = json.loads(p.stdout) if p.stdout.strip() else {}
    ok = ((p.returncode == 0 and res.get("status") == "PASS"
           and bool((res.get("busy") or {}).get("before")) == expect_busy)
          if expect_pass else (p.returncode == 2 and res.get("code") == expect_code))
    print("%-16s rc=%s status=%s code=%s busy_before=%s ok=%s" %
          (name, p.returncode, res.get("status"), res.get("code"),
           (res.get("busy") or {}).get("before"), ok))
    return ok


def m_missing_smi(d):
    (d / "npu-smi.before.txt").unlink()


def m_busy(d):
    p = d / "npu-smi.before.txt"
    t = p.read_text()
    t = t.replace("| No running processes found in NPU 0                                                             |",
                  "| 0                          | 12345         | python                   | 100                      |")
    p.write_text(t)


def m_timer(d):
    p = d / "timing/windows.json"
    w = json.loads(p.read_text())
    w["windows"] = w["windows"][:-1]
    p.write_text(json.dumps(w, indent=2))


def m_same_ptr(d):
    p = d / "numeric.json"
    n = json.loads(p.read_text())
    n["output_buffer_reuse"]["device_ptr_second"] = n["output_buffer_reuse"]["device_ptr_first"]
    p.write_text(json.dumps(n, indent=2))


def m_bad_value(d):
    p = d / "numeric.json"
    n = json.loads(p.read_text())
    n["output_buffer_reuse"]["values_equal"] = False
    p.write_text(json.dumps(n, indent=2))


def m_stability(d):
    p = d / "numeric.json"
    n = json.loads(p.read_text())
    n["stability"]["all_bitwise_equal_first"] = False
    p.write_text(json.dumps(n, indent=2))


def m_missing_numeric(d):
    (d / "numeric.json").unlink()


# unique-dir refusal: build the same case dir twice
ARCH = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/device/archive/cases_audited_20261104T202316/decode_dyn")
udir = ROOT / "unique"
cmd = [PY, str(TOOL), "--case", str(ARCH), "--out-root", str(udir), "--protocol", str(PROTO)]
subprocess.run(cmd, capture_output=True, text=True, timeout=300)
p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
res2 = json.loads(p2.stdout)
unique_ok = (p2.returncode == 2 and res2.get("code") == "CASE_DIR_NOT_NEW")
print("%-16s rc=%s status=%s code=%s ok=%s" %
      ("unique_dir", p2.returncode, res2.get("status"), res2.get("code"), unique_ok))

checks = [
    unique_ok,
    run("baseline", expect_pass=True),
    run("missing_smi", m_missing_smi, expect_code="SMI_MISSING"),
    run("busy_recorded", m_busy, expect_pass=True, expect_busy=True),
    run("busy_refused", m_busy, busy_policy="refuse", expect_code="BUSY_MISJUDGED"),
    run("timer_missing", m_timer, expect_code="TIMER_WINDOW_MISSING"),
    run("reuse_same_ptr", m_same_ptr, expect_code="REUSE_STORAGE_NOT_DISTINCT"),
    run("reuse_bad_value", m_bad_value, expect_code="REUSE_VALUES_MISMATCH"),
    run("stability_false", m_stability, expect_code="STABILITY_FAILED"),
    run("missing_numeric", m_missing_numeric, expect_code="LAYOUT_MISSING"),
]
print("RESULT positive=2 negative=%d total=%d all_ok=%s" %
      (len(checks) - 2, len(checks), all(checks)))
sys.exit(0 if all(checks) else 1)
