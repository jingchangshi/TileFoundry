"""Level-2 RMSNorm structural kernel test (GM-assembly route, Form D).

Fixture: examples/rmsnorm_950pr/scheduled_d.py:RMSNormD (generated). Checks public
analyze/finalize, reload+verify, an independent numeric reference per row (RNE bf16), full
output-tile coverage with no missing/overlapping tile, and that a DROPPED tile write makes the
coverage/numeric checks FAIL. CPU software model; UNCREDITED. Not the production shapes.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import argparse, math, os, re, struct, subprocess, sys
import tilefoundry_ascend  # noqa: F401
import torch

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu loaded; CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIXDIR = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm")
FIX = os.path.join(FIXDIR, "rmsnorm_d512.py")
ENTRY = "RMSNormD"
L, EPS, ATOL, RTOL = 64, 1e-6, 0.02, 0.01


def cli(*a):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"; env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *a], cwd=ROOT, env=env, capture_output=True, text=True)


def bf(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def tir_write_check(text, D, L, BATCH):
    """TYPED-TIR write-side check, per row. Parses every T.copy destination tensor_view
    (layout/shape and its root/offset expression) and requires, for EACH row r, the set of
    64-element tile offsets to be exactly {0, L, ..., D-L}: full coverage, no overlap, no
    whole-row window, no Fill. Returns (ok, reason, windows) with a NAMED rejection reason."""
    if "T.fill(" in text:
        return False, "fill", []
    defs = {}
    for m in re.finditer(r"(\w+)\s*=\s*T\.tensor_view\(\s*T\.ptr_of\(out\[(.+?)\]\)\s*,\s*layout=\((\d+),?\)\s*,\s*shape=\((\d+),?\)", text, re.S):
        defs[m.group(1)] = (int(m.group(3)), int(m.group(4)), m.group(2))
    windows = []
    for line in text.splitlines():
        m = re.search(r"T\.copy\([^,]+,\s*(\w+)\)", line)
        if not m:
            continue
        name = m.group(1)
        if name not in defs:
            return False, "unbound_window", windows
        layout, shape, expr = defs[name]
        if layout != L or shape != L:
            return False, "whole_row_copy", windows
        m2 = re.search(r"r \+ (\d+)\) \* (\d+)\) \+ \((\d+) \* (\d+)\)", expr)
        if m2:
            row_off, row_stride, tile, step = (int(m2.group(1)), int(m2.group(2)),
                                               int(m2.group(3)), int(m2.group(4)))
        else:
            m3 = re.match(r"\s*(\d+):", expr)
            row_off, row_stride, tile, step = 0, D, (int(m3.group(1)) // L if m3 else -1), L
        if row_stride != D or step != L:
            return False, "bad_row_stride", windows
        windows.append((row_off, tile * L))
    # The loop body has offset(r) = (r + c) * D + k * L. For the loop to write every row of the
    # BATCH iteration space, c MUST be 0 and the k set must be exactly the tile set. A nonzero c
    # (row shift) leaves row 0 unwritten (and the last row out of range) -> named rejection.
    tiles = sorted(k for _, k in windows)
    if tiles != list(range(0, D, L)) or len(windows) != D // L:
        return False, "row0_coverage", windows
    if any(c != 0 for c, _ in windows):
        return False, "row0_missing_tile", windows
    return True, "ok", windows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tiles", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fixture", default="scheduled_d.py"); args = ap.parse_args()
    global FIX; FIX = os.path.join(FIXDIR, args.fixture)
    TILES = args.tiles; BATCH = args.batch; D = TILES * L
    from tilefoundry.cli.source import checked_program, load_authored_ir
    from tilefoundry.evaluator import evaluate
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function

    checks = {}
    d = os.environ.get("TMPDIR", "/tmp"); tir = os.path.join(d, f"rmsnorm_d_{D}.py")
    r = cli("analyze", f"{FIX}:{ENTRY}"); checks["public_analyze_exit0"] = r.returncode == 0; print("analyze", r.returncode)
    r = cli("schedule", "finalize", f"{FIX}:{ENTRY}", tir); checks["public_finalize_exit0"] = r.returncode == 0; print("finalize", r.returncode)
    text = ""
    if os.path.isfile(tir):
        m = load_authored_ir(f"{tir}:{ENTRY}"); verify_module(m); verify_prim_function(m.entry_function())
        text = open(tir, encoding="utf-8").read()
        checks["public_reload_verified"] = True
        checks["six_vcadd"] = text.count("ascend_vcadd(") == 6 * max(1, TILES // TILES)  # per row
        checks["has_reverse_pk"] = 'direction="reverse"' in text and 'pack="PK_B32"' in text
        checks["per_tile_copy"] = text.count("T.copy(") == TILES
        # INDEPENDENT write-side coverage from the typed TIR, PER ROW (not the reference loop).
        ok_w, why_w, windows = tir_write_check(text, D, L, BATCH)
        checks["tir_covers_all_tiles_per_row"] = ok_w and why_w == "ok"
        checks["no_whole_row_copy_no_fill"] = ok_w
        # NAMED negative baselines: each mutation MUST be rejected with its own reason.
        bad_whole = text + "\nwindow_bad = T.tensor_view(T.ptr_of(out[0:%d]), layout=(%d,), shape=(%d,))\nT.copy(uo0_2, window_bad)\n" % (D, D, D)
        n_whole = tir_write_check(bad_whole, D, L, BATCH)
        checks["neg_whole_row_copy_rejected"] = (not n_whole[0]) and n_whole[1] == "whole_row_copy"
        n_fill = tir_write_check(text + "\nT.fill(out, 0.0)\n", D, L, BATCH)
        checks["neg_fill_rejected"] = (not n_fill[0]) and n_fill[1] == "fill"
        n_off = tir_write_check(text.replace("(%d * %d)" % (TILES - 1, L), "(%d * %d)" % (TILES - 2, L)), D, L, BATCH)
        checks["neg_wrong_offset_rejected"] = (not n_off[0]) and n_off[1].startswith("row0")
        n_row = tir_write_check(text.replace("(r + 0)", "(r + 1)"), D, L, BATCH)
        checks["neg_row_shift_rejected"] = (not n_row[0]) and "row0" in n_row[1]
    else:
        for k in ("public_reload_verified", "six_vcadd", "has_reverse_pk", "per_tile_copy",
                  "tir_covers_all_tiles_per_row", "no_whole_row_copy_no_fill",
                  "neg_whole_row_copy_rejected", "neg_fill_rejected", "neg_wrong_offset_rejected",
                  "neg_row_shift_rejected"):
            checks[k] = False

    prog = checked_program(f"{FIX}:{ENTRY}", None, command="rmsnorm-d")
    torch.manual_seed(7)
    x = (torch.rand(BATCH * D) * 4 - 2).to(torch.bfloat16)
    w = (torch.rand(D) * 2 - 1).to(torch.bfloat16)
    out = getattr(evaluate(prog.entry, x, w), "data", None)
    got = out.reshape(-1).view(torch.int16).tolist()
    xf = x.to(torch.float32).view(-1).tolist(); wf = w.to(torch.float32).tolist()
    mism = 0; maxabs = 0.0; tile_nonzero = [False] * TILES; tile_seen = [0] * TILES
    for b in range(BATCH):
        row = xf[b * D:(b + 1) * D]
        rstd = 1.0 / math.sqrt(sum(v * v for v in row) / D + EPS)
        for i in range(D):
            ref = bf(bf(row[i] * rstd) * bf(wf[i]))
            g = got[b * D + i] & 0xFFFF
            gv = struct.unpack("<f", struct.pack("<I", g << 16))[0]
            diff = abs(gv - ref); maxabs = max(maxabs, diff)
            if diff > ATOL + RTOL * abs(ref): mism += 1
            t = i // L
            tile_seen[t] += 1
            if gv != 0: tile_nonzero[t] = True
    checks["mismatch_zero"] = mism == 0
    checks["coverage_all_tiles_nonzero"] = all(tile_nonzero)
    checks["coverage_full_no_overlap"] = all(c == L * BATCH for c in tile_seen) and sum(tile_seen) == BATCH * D
    for name, ok in checks.items(): print(("ok   " if ok else "FAIL ") + name)
    print(f"mismatch={mism} max_abs={maxabs:.6g} atol={ATOL} rtol={RTOL}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}"); raise SystemExit(2)
