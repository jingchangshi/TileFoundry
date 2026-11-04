"""P3 staging: bf16 GM<->UB accepted, f32 unchanged, mismatched dtype refused.

Runs the real CLI (analyze/finalize) so the claim is about the compiled path, not a
pattern unit test. Exit codes 0 PASS, 1 FAIL.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_staging")
OUT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "p3_staging_out")


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    failed = 0
    # bf16 staging outside the mode: accepted and finalized.
    r = cli("analyze", os.path.join(FIX, "bf16_copy.py") + ":Bf16Copy")
    if r.returncode != 0:
        failed += 1; print("FAIL bf16_copy_analyze:", r.stderr.strip()[-200:])
    else:
        print("ok   bf16_copy_analyze")
    r = cli("schedule", "finalize", os.path.join(FIX, "bf16_copy.py") + ":Bf16Copy", os.path.join(OUT, "copy.py"))
    if r.returncode != 0:
        failed += 1; print("FAIL bf16_copy_finalize:", r.stderr.strip()[-200:])
    else:
        print("ok   bf16_copy_finalize")
    # f32 path unchanged.
    r = cli("schedule", "finalize", os.path.join(ROOT, "examples", "rmsnorm_950pr", "vf_vadd.py") + ":VfVadd", os.path.join(OUT, "vadd.py"))
    if r.returncode != 0:
        failed += 1; print("FAIL f32_vadd_finalize:", r.stderr.strip()[-200:])
    else:
        print("ok   f32_vadd_finalize")
    # bf16 -> f32 mismatch refused with the named constraint.
    r = cli("analyze", os.path.join(FIX, "bf16_mismatch.py") + ":Bf16Mismatch")
    if r.returncode == 0:
        failed += 1; print("FAIL bf16_mismatch_analyze: accepted")
    elif "src.dtype = dst.dtype" not in r.stderr:
        failed += 1; print("FAIL bf16_mismatch_analyze: wrong reason:", r.stderr.strip()[-200:])
    else:
        print("ok   bf16_mismatch_refused")
    if failed:
        print(f"{failed}/4 checks FAILED")
        return 1
    print("4/4 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
