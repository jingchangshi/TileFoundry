"""03-24: 56-AIV ownership / ragged partition and cross-core byte conservation (CPU, UNCREDITED).

Two required raggings are derived from the SAME partition policy:
  nbase = tiles // 56, rem = tiles % 56
  start(c) = c*nbase + min(c, rem);  count(c) = nbase + (1 if c < rem else 0);  stop = start+count
  - 32x37 + 24x36  (tiles = 2048)  -> nbase=36, rem=32
  - 8x2  + 48x1    (tiles = 64)    -> nbase=1,  rem=8
A synthetic zero-work case (tiles = 40 -> 16 cores with count 0) checks the minimal-loop rule.

Checks: per-core start/count/stop; intervals disjoint, strictly increasing, union = [0,tiles);
total bytes conserved; zero-work cores flagged for a minimal (guarded, 0-iteration) loop.
Fail-able mutations: missing last tile, core start off by one, duplicate interval, deleted core.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import argparse, sys

CORES = 56


def plan(tiles):
    nbase, rem = divmod(tiles, CORES)
    out = []
    for c in range(CORES):
        start = c * nbase + min(c, rem)
        count = nbase + (1 if c < rem else 0)
        out.append((start, count, start + count))
    return out


def conserve(p, tiles, d_bytes):
    """Return (ok, reason): disjoint, strictly increasing, union [0,tiles), bytes conserved."""
    if any(e < s for s, _, e in p):
        return False, "negative_count"
    # zero-work cores own an EMPTY interval; they may share a start (the end of the work range)
    # and are excluded from the ordering/overlap checks, but must stay inside [0, tiles].
    work = [(s, e) for s, c, e in p if c > 0]
    empty = [(s, e) for s, c, e in p if c == 0]
    if any(s < 0 or e > tiles for s, e in work + empty):
        return False, "out_of_range"
    if any(work[i][1] > work[i + 1][0] for i in range(len(work) - 1)):
        return False, "overlap"
    if any(work[i + 1][0] <= work[i][0] for i in range(len(work) - 1)):
        return False, "not_increasing"
    if not work or work[0][0] != 0:
        return False, "gap_at_start"
    if work[-1][1] != tiles:
        return False, "gap_at_end"
    covered = sum(c for _, c, _ in p)
    if covered != tiles:
        return False, "count_mismatch"
    # byte conservation: each core writes [start*d_bytes, stop*d_bytes); total = tiles*d_bytes
    byte_total = sum((e - s) * d_bytes for s, _, e in p)
    if byte_total != tiles * d_bytes:
        return False, "byte_mismatch"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    D_BYTES = 4096 * 2
    checks = {}
    cases = [("32x37+24x36", 2048), ("8x2+48x1", 64), ("zero_work_40", 40)]
    for name, tiles in cases:
        p = plan(tiles)
        ok, why = conserve(p, tiles, D_BYTES)
        counts = [c for _, c, _ in p]
        # the required ragged shape must actually appear
        if name == "32x37+24x36":
            shape_ok = sorted(counts) == sorted([37] * 32 + [36] * 24)
        elif name == "8x2+48x1":
            shape_ok = sorted(counts) == sorted([2] * 8 + [1] * 48)
        else:
            shape_ok = sorted(counts) == sorted([1] * 40 + [0] * 16)
        checks[f"{name}_partition_shape"] = shape_ok
        checks[f"{name}_disjoint_union_byte_conserved"] = ok and why == "ok"
        zero = [c for c, n in enumerate(counts) if n == 0]
        checks[f"{name}_zero_work_cores_minimal_loop"] = all(counts[c] == 0 for c in zero) and (
            len(zero) == (16 if name == "zero_work_40" else 0))
        # ---- mutations (each must FAIL the same conservation check) ----
        mut = [list(x) for x in p]
        mut[-1][1] -= 1; mut[-1][2] = mut[-1][0] + mut[-1][1]
        checks[f"{name}_mut_missing_last_tile_rejected"] = not conserve([tuple(x) for x in mut], tiles, D_BYTES)[0]
        mut = [list(x) for x in p]; mut[10][0] += 1
        checks[f"{name}_mut_start_off_by_one_rejected"] = not conserve([tuple(x) for x in mut], tiles, D_BYTES)[0]
        mut = [list(x) for x in p]; mut[11][0] = mut[10][0]
        checks[f"{name}_mut_duplicate_interval_rejected"] = not conserve([tuple(x) for x in mut], tiles, D_BYTES)[0]
        dropped = [tuple(x) for i, x in enumerate(p) if i != 20]
        checks[f"{name}_mut_deleted_core_rejected"] = not conserve(dropped, tiles, D_BYTES)[0]
    for k, v in checks.items(): print(("ok   " if v else "FAIL ") + k)
    print(f"cases={[c[0] for c in cases]} cores={CORES} d_bytes={D_BYTES}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}"); raise SystemExit(2)
