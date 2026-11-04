"""BOUNDED ROW SAMPLING numeric evidence for the prefill shape (8192 x 4096).

The full prefill program (BATCH=8192) is evaluated only structurally (analyze/finalize/reload/
verify); its full eval is NOT run. Here the SAME compliant generator body, instantiated at
BATCH=1 (rmsnorm_row4096.py), is evaluated on a DECLARED SAMPLE of rows of a deterministic
8192x4096 input. Each sampled row gets a full 64-tile coverage check and an independent float64
reference. This is NOT full 8192-row numeric coverage and must not be read as such.

Discriminative evidence (a sample that cannot fail is meaningless):
  - row-shift: feed row r+1's data, compare to row r's reference -> MUST mismatch;
  - one-tile corruption: zero one output tile in the comparison -> MUST mismatch / lose coverage.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import argparse, math, os, struct, subprocess, sys, time
import tilefoundry_ascend  # noqa: F401
import torch

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu loaded; CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIXDIR = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm")
ENTRY = "RMSNormD"
L, EPS, ATOL, RTOL = 64, 1e-6, 0.02, 0.01
PREFILL_ROWS, D = 8192, 4096
SAMPLES = [0, 1, 2048, 4095, 4096, 8190, 8191]


def bf(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="rmsnorm_row4096.py")
    args = ap.parse_args()
    from tilefoundry.cli.source import checked_program
    from tilefoundry.evaluator import evaluate

    checks = {}
    torch.manual_seed(1234)
    X = (torch.rand(PREFILL_ROWS * D) * 4 - 2).to(torch.bfloat16)   # deterministic prefill input
    w = (torch.rand(D) * 2 - 1).to(torch.bfloat16)
    wf = w.to(torch.float32).tolist()
    prog = checked_program(os.path.join(FIXDIR, args.fixture) + ":" + ENTRY, None, command="rmsnorm-sample")

    def run_row(r):
        x = X[r * D:(r + 1) * D]
        out = getattr(evaluate(prog.entry, x, w), "data", None)
        return [b & 0xFFFF for b in out.reshape(-1).view(torch.int16).tolist()]

    def reference(r):
        row = X[r * D:(r + 1) * D].to(torch.float32).view(-1).tolist()
        rstd = 1.0 / math.sqrt(sum(v * v for v in row) / D + EPS)
        return [bf(bf(row[i] * rstd) * bf(wf[i])) for i in range(D)]

    mism_total = 0; tiles_ok = 0; t0 = time.time(); worst = 0.0
    for r in SAMPLES:
        got = run_row(r); ref = reference(r)
        mism = 0; nonzero = [False] * (D // L)
        for i in range(D):
            gv = struct.unpack("<f", struct.pack("<I", (got[i] & 0xFFFF) << 16))[0]
            diff = abs(gv - ref[i]); worst = max(worst, diff)
            if diff > ATOL + RTOL * abs(ref[i]): mism += 1
            if gv != 0: nonzero[i // L] = True
        mism_total += mism
        if mism == 0 and all(nonzero): tiles_ok += 1
        print(f"row {r}: mismatch={mism} tiles_nonzero={sum(nonzero)}/{D // L}")
    elapsed = time.time() - t0
    checks["sampled_rows_numeric_exact"] = mism_total == 0
    checks["sampled_rows_full_tile_coverage"] = tiles_ok == len(SAMPLES)

    # discriminative 1: row-shift must mismatch
    r = SAMPLES[2]
    got_shift = run_row(r + 1); ref_r = reference(r)
    shifted_mism = 0
    for i in range(D):
        gv = struct.unpack("<f", struct.pack("<I", (got_shift[i] & 0xFFFF) << 16))[0]
        if abs(gv - ref_r[i]) > ATOL + RTOL * abs(ref_r[i]): shifted_mism += 1
    checks["row_shift_detected"] = shifted_mism > 0

    # discriminative 2: one-tile corruption must be detected by coverage/mismatch
    got = run_row(SAMPLES[0]); ref0 = reference(SAMPLES[0])
    corrupt = list(got); corrupt[63 * L:64 * L] = [0] * L
    bad_mism = 0; bad_tiles = [False] * (D // L)
    for i in range(D):
        gv = struct.unpack("<f", struct.pack("<I", (corrupt[i] & 0xFFFF) << 16))[0]
        if abs(gv - ref0[i]) > ATOL + RTOL * abs(ref0[i]): bad_mism += 1
        if gv != 0: bad_tiles[i // L] = True
    checks["one_tile_corruption_detected"] = bad_mism > 0 and not bad_tiles[63]

    for name, ok in checks.items(): print(("ok   " if ok else "FAIL ") + name)
    print(f"SCOPE: sampled rows {SAMPLES} of {PREFILL_ROWS} (NOT full coverage); "
          f"per-row eval {elapsed / len(SAMPLES):.2f}s, projected full 8192 rows ~{elapsed / len(SAMPLES) * PREFILL_ROWS / 60:.1f} min")
    print(f"mismatch={mism_total} worst_abs={worst:.6g} atol={ATOL} rtol={RTOL}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}"); raise SystemExit(2)
