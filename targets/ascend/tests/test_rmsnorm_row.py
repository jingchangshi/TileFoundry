"""03-22 step 1 (path a): single-row RMSNorm STRUCTURAL realization at D=256.

One row of D=256 = FOUR 64-lane tiles, named explicitly (x0..x3, w0..w3). The full row's
rstd is computed over all four tiles: four vmula chains -> balanced ((0+1)+(2+3)) merge ->
six vcadd 64->1 -> explicit vdupv(POS_LOWEST) -> vmuls(1/D) -> vadds(eps) -> exact vsqrt ->
vdiv(1, sqrt); then tile 0 is scaled by rstd and its weight and written through the reverse
path (reverse vcvt -> carrier -> vsts PK_B32 -> bf16 UB -> GM). Weight is converted in a
SEPARATE SimdVF to f32 in UB and re-loaded by the main VF.

This is NOT the full 8192x4096 kernel: it is the structural realization of one row, four
tiles, and it does not claim the two production shapes or 56-AIV/ragged. It exists to prove
the chain/merge/vcadd6/vdupv/vsqrt->vdiv/reverse-vsts composition end-to-end through the
public pipeline with an independent numeric reference.

Two real errors found while building it (kept in the record):
  1. a SimdVF region must COMPUTE A VALUE TO LEAVE; the weight VF had to hand its f32 tile
     (wst) to the main VF instead of writing an unused UB buffer;
  2. reading the raw dst Var of a schedule (f0) gives zeros; the schedule RESULT (v0) is the
     bound value. Using f0 silently produced a zero output.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys

import tilefoundry_ascend  # noqa: F401

import torch  # noqa: E402

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu is loaded; this check is CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "rmsnorm_row.py")
ENTRY = "RMSNormRow"
L, D, EPS = 64, 256, 1e-6
ATOL, RTOL = 0.02, 0.01


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def bf(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def main() -> int:
    from tilefoundry.cli.source import checked_program, load_authored_ir
    from tilefoundry.evaluator import evaluate
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function

    checks = {}
    directory = os.environ.get("TMPDIR", "/tmp")
    tir = os.path.join(directory, "rmsnorm_row_tir.py")
    r = cli("analyze", f"{FIX}:{ENTRY}")
    print(f"child analyze exit={r.returncode}")
    checks["public_analyze_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    r = cli("schedule", "finalize", f"{FIX}:{ENTRY}", tir)
    print(f"child finalize exit={r.returncode}")
    checks["public_finalize_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    if os.path.isfile(tir):
        module = load_authored_ir(f"{tir}:{ENTRY}")
        verify_module(module)
        verify_prim_function(module.entry_function())
        text = open(tir, encoding="utf-8").read()
        checks["public_reload_verified"] = True
        checks["six_vcadd"] = text.count("ascend_vcadd(") == 6
        checks["explicit_vdupv"] = "ascend_vdupv(" in text
        checks["exact_sqrt_then_div"] = "ascend_vsqrt(" in text and "ascend_vdiv(" in text and "rsqrt" not in text
        checks["reverse_and_pk_b32"] = 'direction="reverse"' in text and 'pack="PK_B32"' in text
        checks["two_simd_vf_regions"] = text.count("with T.SimdVF()") == 2
    else:
        for key in ("public_reload_verified", "six_vcadd", "explicit_vdupv", "exact_sqrt_then_div",
                    "reverse_and_pk_b32", "two_simd_vf_regions"):
            checks[key] = False

    xs = [[0.5, -1.25, 2.0, -3.5, 0.75, 1.0, -2.25, 4.0] + [((i % 7) - 3) * 0.5 for i in range(56)] for _ in range(4)]
    ws = [[1.0, 0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5] + [((i % 5) - 2) * 0.25 for i in range(56)] for _ in range(4)]

    def as_bf16(v):
        return torch.tensor(v, dtype=torch.float32).to(torch.bfloat16)

    program = checked_program(f"{FIX}:{ENTRY}", None, command="rmsnorm-row")
    args = [as_bf16(xs[i]) for i in range(4)] + [as_bf16(ws[i]) for i in range(4)]
    out = getattr(evaluate(program.entry, *args), "data", None)
    got = [b & 0xFFFF for b in out.reshape(-1).view(torch.int16).tolist()]

    allx = [bf(float(v)) for tile in xs for v in tile]
    rstd = 1.0 / math.sqrt(sum(v * v for v in allx) / D + EPS)
    ref = [bf(bf(bf(float(xs[0][i]) * rstd) if False else bf(float(xs[0][i])) * rstd) * bf(float(ws[0][i]))) for i in range(L)]
    ref_f = [bf(bf(float(xs[0][i]) * rstd) * bf(float(ws[0][i]))) for i in range(L)]
    # The program computes f32 throughout; compare values with the frozen protocol.
    got_f = [struct.unpack("<f", struct.pack("<I", (g & 0xFFFF) << 16))[0] for g in got]
    diffs = [abs(got_f[i] - ref_f[i]) for i in range(L)]
    mismatch = [i for i in range(L) if diffs[i] > ATOL + RTOL * abs(ref_f[i])]
    checks["dtype_bf16"] = out.dtype == torch.bfloat16
    checks["mismatch_zero"] = len(mismatch) == 0
    checks["not_constant_zero"] = any(g != 0 for g in got)
    # input dependence: scale all x tiles by 3 -> output changes
    xs2 = [[v * 3.0 for v in tile] for tile in xs]
    args2 = [as_bf16(xs2[i]) for i in range(4)] + [as_bf16(ws[i]) for i in range(4)]
    got2 = getattr(evaluate(program.entry, *args2), "data", None).reshape(-1).view(torch.int16).tolist()
    checks["input_dependent"] = got2 != got

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"mismatch={len(mismatch)} max_abs={max(diffs):.6g} atol={ATOL} rtol={RTOL}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} checks FAILED")
        return 1
    print(f"{len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(2)
