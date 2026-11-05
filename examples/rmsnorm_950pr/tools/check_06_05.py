#!/usr/bin/env python3
"""06-05 single driver: input ABI -> binary/env -> launch (PLAN assemble) -> readback ->
direct_diff (06-06) -> independent FP64 (03-21) -> verdict, with per-stage phases.json.

Modes:
  plan   (default) NO device, NO torch import, NO lock. Uses ARCHIVED run2 case products and
         the current .aibin for a rehearsal; "readback" is the archived device.npy and is
         labelled archived_not_fresh.
  device requires ALL of: euid 0, TF_DEVICE_AUTH=1, ASCEND_RT_VISIBLE_DEVICES as a
         non-empty comma list within TF_DEVICE_ALLOW whose ids all exist in the container,
         a NEW empty --cases-dir, and the canonical lock acquired non-blocking. Any missing
         condition is a named REFUSE[DEVICE_GATE] exit 2 and nothing is launched.
         A busy npu-smi process table is recorded, never refused or waited on.

Verdict: mismatch_count==0 for every pattern; deterministic (zero/ones/alternating) bf16
codewords exact; repeats/stability/reuse pass; nonfinite output fails; empty pattern set
refuses. Exit 0 PASS, 1 FAIL, 2 REFUSED.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
TF = HERE.parents[3]
sys.path.insert(0, str(TF / "tools" / "p3"))
sys.path.insert(0, str(TF / "tools" / "p3" / "native_probe"))
sys.path.insert(0, str(TF / "examples/rmsnorm_950pr/tools"))
import rmsnorm_oracle as oracle  # noqa: E402
from device_guard import GuardError, check_device_identity  # noqa: E402
import direct_diff as dd  # noqa: E402

FROZEN = TF.parent / "TileFoundry_Ascend950PR_AgentTeam/artifacts/run01/p0/precision_protocol.json"
E_ASCEND_AIV = 0x1029
# The CANONICAL team lock: the fixed path every agent on this host meets at. It is a
# literal ON PURPOSE -- it is not a per-machine location to be derived, and deriving
# it would let two processes disagree about where the lock is and both hold "the" one.
LOCK = Path("/home/.tf_device0.lock")  # no-machine-path: allow


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class StageFail(RuntimeError):
    def __init__(self, code, msg, exit_code=1):
        super().__init__("%s: %s" % (code, msg))
        self.code, self.msg, self.exit_code = code, msg, exit_code


class Phases:
    def __init__(self, out: Path):
        self.out = out
        self.path = out / "phases.json"
        self.data = {"started": now(), "stages": {}, "order": []}

    def add(self, name, status, code=None, detail=None, extra=None):
        rec = {"status": status, "time": now(), "code": code, "detail": detail}
        if extra:
            rec.update(extra)
        self.data["stages"][name] = rec
        self.data["order"].append(name)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def finish(self, status):
        self.data["finished"] = now()
        self.data["status"] = status
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


def elf_machine(p: Path):
    head = p.read_bytes()[:64]
    if len(head) < 20 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
        return None
    return int.from_bytes(head[18:20], "little")


def env_check(env):
    bad = []
    home = env.get("ASCEND_HOME_PATH", "")
    if not home or not Path(home).is_dir():
        bad.append("ASCEND_HOME_PATH")
    elif not any((Path(home) / s).is_dir() for s in ("lib64", "runtime/lib64")):
        bad.append("ASCEND_HOME_PATH/lib64")
    ldp = [Path(x) for x in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if x]
    if not any(p.is_dir() and "lib64" in str(p) for p in ldp):
        bad.append("LD_LIBRARY_PATH(lib64)")
    pp = [Path(x) for x in env.get("PYTHONPATH", "").split(os.pathsep) if x]
    if not any((p / "tilelang").is_dir() for p in pp):
        bad.append("PYTHONPATH(tilelang)")
    if not any((p / "tilefoundry").is_dir() for p in pp):
        bad.append("PYTHONPATH(TF src)")
    if not any(p.name == "ascend" and p.is_dir() for p in pp):
        bad.append("PYTHONPATH(TF targets/ascend)")
    try:
        check_device_identity(env)
    except GuardError as exc:
        bad.append("device identity %s" % exc.code)
    if env.get("TF_DEVICE_AUTH") != "1":
        bad.append("TF_DEVICE_AUTH=1")
    return bad


def device_gate(args, env):
    reasons = []
    if args.mode != "device":
        return reasons
    if os.geteuid() != 0:
        reasons.append("NOT_ROOT")
    if env.get("TF_DEVICE_AUTH") != "1":
        reasons.append("TF_DEVICE_AUTH!=1")
    cd = Path(args.cases_dir) if args.cases_dir else None
    if cd is None or not cd.is_dir():
        reasons.append("CASES_DIR_MISSING")
    elif any(cd.iterdir()):
        reasons.append("CASES_DIR_NOT_NEW(not empty)")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "device"], default="plan")
    ap.add_argument("--case-dir", required=True, help="archived (plan) or empty output dir (device)")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--aibin", required=True)
    ap.add_argument("--expected-aibin-sha", default=None)
    ap.add_argument("--expected-arch", default="dav-3510")
    ap.add_argument("--cases-dir", default=None, help="device mode: NEW empty dir for run_case")
    ap.add_argument("--run-case", default=str(TF / "examples/rmsnorm_950pr/tools/run_case.py"))
    ap.add_argument("--protocol", default=str(FROZEN))
    ap.add_argument("--env-json", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env = json.loads(Path(args.env_json).read_text()) if args.env_json else dict(os.environ)
    phases = Phases(out)
    lock_fh = None
    try:
        # stage 1: input ABI (03-21)
        try:
            protocol = dd.load_protocol(Path(args.protocol))
            tj = json.loads((Path(args.case_dir) / "tensors.json").read_text())
            shape = tuple(int(v) for v in tj["shape"])
            entries = tj.get("tensors") or []
            if not entries:
                raise StageFail("EMPTY_PATTERN_SET", "tensors.json lists zero patterns", 2)
            abi = {"shape": list(shape), "patterns": []}
            for e in entries:
                rec = oracle.read_inputs(Path(args.case_dir), e, shape)
                abi["patterns"].append({"key": rec["key"], "pattern": rec["pattern"],
                                        "input_digest": rec["input_digest"],
                                        "files": rec["files"]})
            (out / "abi.json").write_text(json.dumps(abi, indent=2) + "\n")
            phases.add("abi", "PASS", detail="%d patterns, shape %s" % (len(entries), shape))
        except (dd.Refusal, oracle.AbiError, StageFail) as exc:
            code = getattr(exc, "code", "ABI_FAILED")
            phases.add("abi", "REFUSED" if getattr(exc, "exit_code", 2) == 2 else "FAIL", code, str(exc))
            raise

        # stage 2: aibin / ELF / arch / env assertions
        try:
            p = Path(args.aibin)
            if not p.is_file():
                raise StageFail("CCA_FILE_MISSING", str(p), 2)
            got = sha256_file(p)
            want = args.expected_aibin_sha
            if want and got != want:
                raise StageFail("CCA_SHA_MISMATCH", "expected %s got %s" % (want, got), 2)
            mach = elf_machine(p)
            if mach != E_ASCEND_AIV:
                raise StageFail("E_MACHINE_MISMATCH",
                                "expected 0x1029 got %s" % (None if mach is None else "0x%04X" % mach), 2)
            if args.expected_arch != "dav-3510":
                raise StageFail("ARCH_MISMATCH", "expected arch %s" % args.expected_arch, 2)
            bad = env_check(env)
            if bad:
                raise StageFail("ENV_MISSING", ", ".join(bad), 2)
            if args.mode == "device":
                reasons = device_gate(args, env)
                if reasons:
                    raise StageFail("DEVICE_GATE", ", ".join(reasons), 2)
                check_device_identity(env)
                lock_fh = open(LOCK, "a+")
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_fh.close(); lock_fh = None
                    raise StageFail("DEVICE_GATE", "LOCK_HELD_BY_OTHER", 2)
            binary = {"aibin": str(p), "sha256": got, "e_machine": "0x%04X" % mach,
                      "arch": "dav-3510", "mode": args.mode}
            (out / "binary.json").write_text(json.dumps(binary, indent=2) + "\n")
            phases.add("binary", "PASS", detail="sha %s machine 0x1029" % got[:16])
        except StageFail as exc:
            phases.add("binary", "REFUSED" if exc.exit_code == 2 else "FAIL", exc.code, exc.msg)
            raise

        # stage 3: launch (PLAN only assembles) / device runs run_case
        try:
            if args.mode == "plan":
                plan = {"status": "PLAN_ONLY_NOT_EXECUTED", "aibin": args.aibin,
                        "aibin_sha256": sha256_file(Path(args.aibin)),
                        "case_id": args.case_id or Path(args.case_dir).name,
                        "shape": list(shape), "patterns": [e["key"] for e in entries],
                        "device": 0, "lock": str(LOCK), "stream": "torch.npu.current_stream()",
                        "readback_source": "archived device.npy (NOT a fresh launch)"}
                (out / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
                phases.add("launch", "PLAN_ONLY", detail="assembled, not executed")
            else:
                # The case id must be the REAL case name: the independent FP64 oracle derives the
                # per-case accumulator count from it, and a placeholder makes it refuse.
                case_id = args.case_id or Path(args.case_dir).name
                cmd = [sys.executable, args.run_case, "--case-id", case_id,
                       "--shape", str(shape[0]), str(shape[1]), "--rows",
                       str(int(tj.get("rows", 1))), "--loop-mode", str(tj.get("loop_mode", "dyn")),
                       "--stages", str(int(tj.get("num_stages", 2))), "--xversions", "2",
                       "--yversions", "1", "--device", "0", "--out", args.cases_dir]
                # The driver ALREADY holds the one canonical lock (taken in stage 2), so the child
                # must not take it again: a second flock on the same file from a child blocks
                # forever and every device run would fail with a false LOCK_HELD_BY_OTHER. The
                # child's documented contract is TF_DEVICE_LOCK_HELD=1 = "a parent holds it".
                child_env = dict(env, TF_DEVICE_LOCK_HELD="1")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                                   env=child_env)
                (out / "launch_stdout.txt").write_text(r.stdout)
                if r.returncode != 0:
                    raise StageFail("DEVICE_LAUNCH_FAILED", r.stderr[-800:], 1)
                phases.add("launch", "EXECUTED", detail="run_case rc=0")
        except StageFail as exc:
            phases.add("launch", "FAIL", exc.code, exc.msg)
            raise

        # stage 4: readback
        cdir = Path(args.cases_dir) if args.mode == "device" else Path(args.case_dir)
        try:
            if args.mode == "plan":
                files = {e["key"]: sha256_file(cdir / (e["key"] + ".device.npy")) for e in entries}
                rb = {"source": "archived_device_npy", "files": files}
                phases.add("readback", "ARCHIVED", detail="%d archived device tensors" % len(files))
            else:
                files = {e["key"]: sha256_file(cdir / (e["key"] + ".device.npy")) for e in entries}
                rb = {"source": "fresh_device_output", "files": files}
                phases.add("readback", "PASS", detail="%d fresh device tensors" % len(files))
            (out / "readback.json").write_text(json.dumps(rb, indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001
            phases.add("readback", "FAIL", "READBACK_MISSING", str(exc))
            raise StageFail("READBACK_MISSING", str(exc), 1)

        # stage 5: direct_diff (06-06)
        try:
            diff = dd.diff_case(cdir, protocol)
            (out / "direct_diff.json").write_text(json.dumps(diff, indent=2) + "\n")
            if not diff.get("case_pass"):
                failed = [f for pat in diff["patterns"] for f in pat["failures"]]
                failed += [k for k, v in diff["repeats_stability_reuse"]["rules"].items() if v]
                phases.add("direct_diff", "FAIL", failed[0], ",".join(failed))
                raise StageFail(failed[0] if failed else "DIRECT_DIFF_FAILED", ",".join(failed), 1)
            phases.add("direct_diff", "PASS", detail="mismatch 0, det exact, repeats/reuse ok")
        except dd.Refusal as exc:
            phases.add("direct_diff", "REFUSED", exc.code, str(exc))
            raise StageFail(exc.code, str(exc), 2)
        except StageFail:
            raise
        except Exception as exc:  # noqa: BLE001
            phases.add("direct_diff", "FAIL", "DIRECT_DIFF_ERROR", str(exc))
            raise StageFail("DIRECT_DIFF_ERROR", str(exc), 1)

        # stage 6: independent FP64 + input identity (03-21)
        try:
            indep = oracle.consume_run_dir(cdir, protocol)
            (out / "independent_fp64.json").write_text(json.dumps(indep, indent=2) + "\n")
            if not indep.get("passed"):
                code = indep["failures"][0].split(":")[-1].strip()
                phases.add("independent_fp64", "FAIL", code, "; ".join(indep["failures"][:2]))
                raise StageFail(code or "INDEPENDENT_FAILED", "; ".join(indep["failures"][:2]), 1)
            phases.add("independent_fp64", "PASS", detail="identity + independent FP64 pass")
        except StageFail:
            raise
        except Exception as exc:  # noqa: BLE001
            phases.add("independent_fp64", "FAIL", "INDEPENDENT_ERROR", str(exc))
            raise StageFail("INDEPENDENT_ERROR", str(exc), 1)

        # stage 7: verdict
        verdict = {"case_id": diff["case_id"],
                   "mismatch_total": sum(p["protocol"]["mismatch_count"] for p in diff["patterns"]),
                   "det_exact": all((not p["deterministic"]["required"]) or
                                    p["deterministic"]["bf16_codewords_exact"]
                                    for p in diff["patterns"]),
                   "repeats_stability_reuse": diff["repeats_stability_reuse"]["pass"],
                   "independent_fp64": indep["passed"],
                   "readback_source": rb["source"],
                   "mode": args.mode}
        ok = (verdict["mismatch_total"] == 0 and verdict["det_exact"]
              and verdict["repeats_stability_reuse"] and verdict["independent_fp64"])
        verdict["pass"] = bool(ok)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        phases.add("verdict", "PASS" if ok else "FAIL", detail=json.dumps(verdict))
        phases.finish("PASS" if ok else "FAIL")
        print(("PASS " if ok else "FAIL ") + json.dumps(verdict))
        return 0 if ok else 1
    except StageFail as exc:
        phases.finish("REFUSED" if exc.exit_code == 2 else "FAIL")
        print("REFUSE " + exc.code + ": " + exc.msg if exc.exit_code == 2 else
              "FAIL " + exc.code + ": " + exc.msg)
        return exc.exit_code
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
