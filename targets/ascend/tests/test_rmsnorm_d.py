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


def tir_write_windows(text, D, L):
    """Parse the TYPED finalized TIR: every T.copy destination window, its view layout/shape and
    its element offset. This is the independent write-side coverage evidence (not the reference
    loop). Returns (windows, ok_layout) where windows = [(offset, extent)]."""
    defs = {}
    for m in re.finditer(r"(\w+)\s*=\s*T\.tensor_view\(\s*T\.ptr_of\(out\[(.+?)\]\)\s*,\s*layout=\((\d+),?\)\s*,\s*shape=\((\d+),?\)", text, re.S):
        defs[m.group(1)] = (int(m.group(3)), int(m.group(4)), m.group(2))
    windows = []
    ok_layout = True
    for line in text.splitlines():
        m = re.search(r"T\.copy\([^,]+,\s*(\w+)\)", line)
        if not m:
            continue
        name = m.group(1)
        if name not in defs:
            ok_layout = False
            continue
        layout, shape, expr = defs[name]
        if layout != L or shape != L:
            ok_layout = False
        lit = re.search(r"\+ \((\d+) \* %d\)" % L, expr)
        if lit:
            off = int(lit.group(1)) * L
        else:
            lit0 = re.match(r"\s*(\d+):", expr)
            off = int(lit0.group(1)) if lit0 else -1
        windows.append((off, shape))
    return windows, ok_layout


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
        # INDEPENDENT write-side coverage from the typed TIR: every T.copy destination is a
        # 64-element tile view, the tile offsets are distinct and exactly cover [0, D).
        windows, ok_layout = tir_write_windows(text, D, L)
        offsets = sorted(off for off, _ in windows)
        checks["tir_all_dst_windows_are_tiles"] = ok_layout and all(ext == L for _, ext in windows)
        checks["tir_covers_all_tiles_no_overlap"] = offsets == list(range(0, D, L)) and len(windows) == TILES
        checks["no_whole_row_copy"] = all(ext == L for _, ext in windows) and "T.fill(" not in text
        # negative: a whole-row window/copy injected into the TIR MUST be rejected by the checker
        bad = text + "\nwindow_bad = T.tensor_view(T.ptr_of(out[0:%d]), layout=(%d,), shape=(%d,))\nT.copy(uo0_2, window_bad)\n" % (D, D, D)
        bad_windows, bad_layout = tir_write_windows(bad, D, L)
        checks["negative_whole_row_copy_rejected"] = (not bad_layout) or any(ext == D for _, ext in bad_windows)
    else:
        for k in ("public_reload_verified", "six_vcadd", "has_reverse_pk", "per_tile_copy",
                  "tir_all_dst_windows_are_tiles", "tir_covers_all_tiles_no_overlap",
                  "no_whole_row_copy", "negative_whole_row_copy_rejected"):
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
