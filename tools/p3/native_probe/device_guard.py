#!/usr/bin/env python3
"""03-19 device-window guard for the native microtests (DESIGN, NOT EXECUTED THIS ROUND).

Every native probe must run under ONE approved V09 window with the following PRECONDITIONS,
enforced here rather than described in prose:

  * ROOT (device ops require root in this environment);
  * TF_DEVICE_AUTH=1 supplied for this invocation only, no default;
  * ASCEND_RT_VISIBLE_DEVICES=0;
  * ONE host flock /home/.tf_device0.lock, non-blocking: a second concurrent team
    measurement job must NOT enter (parent holds the lock; children do not re-acquire);
  * npu-smi snapshot BEFORE and AFTER, both saved as raw files;
  * every device-side subprocess has a BOUNDED timeout (no unbounded hang);
  * the selected build is verified by VERSION / git HEAD / file sha256 before ANY launch.

STATUS: UNSOURCED / UNVALIDATED. Nothing here is a device observation.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

LOCK = Path("/home/.tf_device0.lock")
PINNED = {
    "tilelang_version": "0.1.15",
    "tilelang_build": "0.1.15+cuda.git8e2d3c82",
    "tilelang_git_head": "8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b",
    "simd_inst_h_sha256": "979f4ff76eee82bf092addcc069b238e3084af890f11660498f00640a00f20a1",
    "simd_py_sha256": "89483579ddde5e36a352a853720fc02ba2b06badeab9eca74b31bcde6a04b57f",
}


class GuardError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_build(tl_root: Path) -> dict:
    """Bind the probe to the SELECTED build. A different source is a refusal, not a result."""
    tl_root = Path(tl_root)
    v = (tl_root / "VERSION").read_text().strip()
    h = (tl_root / "src/tl_templates/ascend/simd_inst.h")
    s = (tl_root / "tilelang/ascend/language/simd.py")
    # root in the container cannot read a user-owned repo without safe.directory; that is
    # an ownership guard, not a source difference.
    head = subprocess.run(["git", "-c", "safe.directory=" + str(tl_root),
                          "-C", str(tl_root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=30).stdout.strip()
    got = {"tilelang_version": v, "tilelang_git_head": head,
           "simd_inst_h_sha256": sha256_file(h), "simd_py_sha256": sha256_file(s)}
    for k, want in PINNED.items():
        if k in got and got[k] != want:
            raise GuardError("BUILD_SHA_MISMATCH", "%s: expected %s, got %s" % (k, want, got[k]))
    return got


class DeviceWindow:
    """Context manager: one lock, root/auth/device identity, before+after npu-smi."""

    def __init__(self, out: Path, timeout_s: int = 180):
        self.out = Path(out)
        self.timeout_s = timeout_s
        self.fh = None
        self.report = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "lock": str(LOCK), "timeout_s": timeout_s,
                       "status": "UNSOURCED / UNVALIDATED (design)"}

    def __enter__(self):
        if os.geteuid() != 0:
            raise GuardError("NOT_ROOT", "device microtests require root")
        if os.environ.get("TF_DEVICE_AUTH", "").strip() != "1":
            raise GuardError("DEVICE_AUTH_MISSING", "TF_DEVICE_AUTH=1 is required for this invocation")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip() != "0":
            raise GuardError("DEVICE_ID_MISMATCH", "ASCEND_RT_VISIBLE_DEVICES must be 0")
        self.out.mkdir(parents=True, exist_ok=True)
        if os.environ.get("TF_DEVICE_LOCK_HELD", "").strip() == "1":
            # ONE holder per run: the parent shell (run_all_cases.sh or the 03-20 wrapper)
            # holds the canonical flock. A child must NOT take a second lock.
            self.report["lock_held_by_parent"] = True
            self.report["lock"] = str(LOCK) + " (held by parent)"
            self._snapshot("npu-smi.before.txt")
            return self
        import fcntl
        self.fh = LOCK.open("a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.fh.close(); self.fh = None
            raise GuardError("DEVICE_LOCK_HELD",
                             "another team measurement job holds %s; do NOT enter" % LOCK) from exc
        self.report["lock_held_by_parent"] = False
        self._snapshot("npu-smi.before.txt")
        return self

    def _snapshot(self, name: str):
        try:
            p = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                               timeout=min(30, self.timeout_s))
            (self.out / name).write_text("exit=%s\n%s\n%s" % (p.returncode, p.stdout, p.stderr))
            self.report[name] = {"exit": p.returncode}
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            (self.out / name).write_text("SNAPSHOT UNAVAILABLE: %r\n" % (exc,))
            self.report[name] = {"exit": None, "error": repr(exc)}

    def run(self, argv: list[str], name: str):
        """Bounded device-side subprocess; raw stdout/stderr/exit are persisted."""
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s)
            (self.out / (name + ".stdout")).write_text(p.stdout)
            (self.out / (name + ".stderr")).write_text(p.stderr)
            (self.out / (name + ".exit")).write_text(str(p.returncode))
            self.report[name] = {"exit": p.returncode}
            return p
        except subprocess.TimeoutExpired as exc:
            (self.out / (name + ".timeout")).write_text(repr(exc))
            self.report[name] = {"exit": None, "timeout_s": self.timeout_s}
            raise GuardError("DEVICE_TIMEOUT", "%s exceeded %ds" % (name, self.timeout_s)) from exc

    def __exit__(self, exc_type, exc, tb):
        self._snapshot("npu-smi.after.txt")
        self.report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.report["exception"] = None if exc is None else repr(exc)
        (self.out / "device_window.json").write_text(json.dumps(self.report, indent=2))
        if self.fh is not None:
            import fcntl
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
        return False


def plan_text(probe: str, case: str, out: Path, timeout_s: int) -> dict:
    """The --plan output: everything a 03-20 operator needs, with NO device access."""
    return {
        "probe": probe, "case": case, "status": "UNSOURCED / UNVALIDATED (DESIGN ONLY)",
        "executed_this_round": False,
        "preconditions": ["root (euid 0)", "TF_DEVICE_AUTH=1", "ASCEND_RT_VISIBLE_DEVICES=0",
                          "one non-blocking flock /home/.tf_device0.lock (V09 window)",
                          "npu-smi info before AND after saved to raw files",
                          "bounded timeout %ds on every device subprocess" % timeout_s,
                          "tilelang VERSION / git HEAD / simd_inst.h + simd.py sha256 verified"],
        "command": ("TF_DEVICE_AUTH=1 ASCEND_RT_VISIBLE_DEVICES=0 "
                    "PYTHONDONTWRITEBYTECODE=1 python tools/p3/native_probe/%s "
                    "--case %s --out %s --execute --timeout %d"
                    % (probe, case, out, timeout_s)),
        "expected_bits": "tools/p3/native_probe/expected_bits.json (CPU model predictions ONLY)",
        "deliverables": ["observed_bits.json (full 64 lanes, raw)", "device_window.json",
                         "npu-smi.before.txt", "npu-smi.after.txt",
                         "compile stdout/stderr/exit", "launch stdout/stderr/exit"],
        "acceptance": ("architect 03-20 must rule on the observed bits and the selected-build "
                       "source/binary hash; a CPU model match is NOT a native contract"),
    }
