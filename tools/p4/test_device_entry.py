#!/usr/bin/env python3
"""06-01/06-12 local (no device) tests for the unified device-entry policy.

Covers:
  A. run_case._device_identity_errors: root / device0 / ffi 0.1.12 assertions
     (0.1.15.post2 CPU ffi is refused).
  B. run_case._ensure_device_lock: free -> acquire; held by another process -> exit 75;
     TF_DEVICE_LOCK_HELD=1 -> no re-acquire (parent holder).
  C. device_precondition.sh: BUSY occupancy is ALLOWED; missing/empty occupancy,
     wrong device id, unreadable path are REFUSED (exit 78).
  D. two real concurrent flock attempts on /home/.tf_device0.lock: the second is blocked.

Run: python tools/p4/test_device_entry.py --out-dir <dir>   (exit 0 iff all pass)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TF = Path(__file__).resolve().parents[2]
RUN_CASE = TF / "examples/rmsnorm_950pr/tools/run_case.py"
PRECOND = TF / "targets/ascend/tests/device_precondition.sh"
LOCK = "/home/.tf_device0.lock"
FAILS = 0
COUNTS = {"positive": 0, "negative": 0, "concurrency": 0}


def check(cond, label, group="positive"):
    global FAILS
    COUNTS[group] += 1
    if not cond:
        FAILS += 1
    print("%-92s %s" % (label, "PASS" if cond else "FAIL"))


def env(**kw):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    e.update(kw)
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # A. identity errors are pure and testable
    code = (
        "import importlib.util,sys,json;"
        "spec=importlib.util.spec_from_file_location('rc', %r);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "print(json.dumps({"
        "'ok':m._device_identity_errors(0,'0','0.1.12',0),"
        "'nonroot':m._device_identity_errors(1071,'0','0.1.12',0),"
        "'wrong_visible':m._device_identity_errors(0,'1','0.1.12',0),"
        "'cpu_ffi':m._device_identity_errors(0,'0','0.1.15.post2',0),"
        "'unknown_ffi':m._device_identity_errors(0,'0',None,0)}))"
    ) % str(RUN_CASE)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env())
    if p.returncode != 0:
        print("identity probe FAILED:", p.stderr[-500:]); check(False, "identity probe runs", "positive")
    else:
        d = json.loads(p.stdout)
        check(d["ok"] == [], "identity: correct env yields no errors", "positive")
        check(any("root" in x for x in d["nonroot"]), "identity: non-root refuses", "negative")
        check(any("VISIBLE" in x for x in d["wrong_visible"]), "identity: non-0 visible device refuses", "negative")
        check(any("0.1.12" in x for x in d["cpu_ffi"]), "identity: CPU ffi 0.1.15.post2 refuses", "negative")
        check(any("unknown" in x for x in d["unknown_ffi"]), "identity: unknown ffi refuses", "negative")

    # B. lock: hold in a background process, then ask a fresh process to acquire.
    holder = subprocess.Popen(["flock", "-x", LOCK, "-c", "sleep 4"], env=env())
    time.sleep(0.6)
    code_lock = (
        "import importlib.util,sys;"
        "spec=importlib.util.spec_from_file_location('rc', %r);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "m._ensure_device_lock();print('ACQUIRED')"
    ) % str(RUN_CASE)
    p2 = subprocess.run([sys.executable, "-c", code_lock], capture_output=True, text=True, env=env())
    check(p2.returncode == 75, "lock: second standalone entry refused 75 while held (got %d)" % p2.returncode, "concurrency")
    check("holds" in (p2.stderr or ""), "lock: refusal names the held canonical lock", "concurrency")
    p3 = subprocess.run([sys.executable, "-c", code_lock], capture_output=True, text=True,
                        env=env(TF_DEVICE_LOCK_HELD="1"))
    check(p3.returncode == 0 and "ACQUIRED" in p3.stdout,
          "lock: TF_DEVICE_LOCK_HELD=1 child does NOT re-acquire (parent holder) (got %d)" % p3.returncode,
          "concurrency")
    holder.wait(timeout=10)
    p4 = subprocess.run([sys.executable, "-c", code_lock], capture_output=True, text=True, env=env())
    check(p4.returncode == 0 and "ACQUIRED" in p4.stdout,
          "lock: after release, standalone entry acquires (got %d)" % p4.returncode, "concurrency")

    # C. device_precondition policy
    occ = out / "occupancy_busy.txt"; occ.write_text("Device 0 BUSY: 3 processes\n")
    occ_empty = out / "occupancy_empty.txt"; occ_empty.write_text("")
    def prec(env_extra, label, want_rc, want_text=None, group="negative"):
        cmd = ('. %s; tf_require_device; echo GATE_RC=$?' % str(PRECOND))
        p = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, env=env(**env_extra))
        ok = ("GATE_RC=%d" % want_rc) in p.stdout and (want_text is None or want_text in p.stderr)
        check(ok, label + " (rc=%s)" % (p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "?"), group)
    good = dict(TF_DEVICE_AUTH="1", TF_DEVICE_DEVICE_ID="0", TF_DEVICE_OCCUPANCY=str(occ))
    prec(good, "precondition: BUSY card is ALLOWED (exit 0)", 0, "ALLOWED", "positive")
    prec(dict(good, TF_DEVICE_OCCUPANCY=str(occ_empty)), "precondition: empty occupancy refuses 78", 78, "empty")
    prec({k: v for k, v in good.items() if k != "TF_DEVICE_OCCUPANCY"},
         "precondition: missing occupancy refuses 78", 78, "TF_DEVICE_OCCUPANCY")
    prec(dict(good, TF_DEVICE_DEVICE_ID="1"), "precondition: non-0 device refuses 78", 78, "pinned to device 0")
    prec(dict(good, TF_DEVICE_OCCUPANCY=str(out / "nope.txt")), "precondition: unreadable occupancy refuses 78", 78, "unreadable")

    # D. real concurrent flock: host uid1071 + container root each attempt while host holds
    h = subprocess.Popen(["flock", "-x", LOCK, "-c", "sleep 4"])
    time.sleep(0.6)
    p = subprocess.run(["flock", "-xn", LOCK, "-c", "echo GOT"], capture_output=True, text=True)
    check(p.returncode != 0 and "GOT" not in p.stdout,
          "concurrency(host uid1071): second flock blocked while first holds", "concurrency")
    docker = ["docker", "exec", "-u", "root", "tf-ascend950pr", "flock", "-xn", LOCK, "-c", "echo GOT"]
    try:
        d = subprocess.run(docker, capture_output=True, text=True, timeout=30)
        check(d.returncode != 0 and "GOT" not in d.stdout,
              "concurrency(container root): second flock blocked while host holds (rc=%d)" % d.returncode,
              "concurrency")
    except Exception as exc:  # noqa: BLE001
        print("container concurrency UNRUN:", exc)
        check(False, "concurrency(container root) UNRUN is not PASS", "concurrency")
    h.wait(timeout=10)

    total = sum(COUNTS.values())
    print("positive=%d negative=%d concurrency=%d total=%d"
          % (COUNTS["positive"], COUNTS["negative"], COUNTS["concurrency"], total))
    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
