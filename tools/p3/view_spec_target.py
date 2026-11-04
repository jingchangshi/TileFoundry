#!/usr/bin/env python3
"""Adapt I1's DtypeViewSpec objects into this spec's GmemRoot/OffsetView (READ-ONLY).

Reports, for I1's REAL view objects, which of the five rules fire, and the two valid
controls. Also reports where HIS validation and THIS spec DISAGREE, so the drift is visible
rather than papered over. This never writes to his file and never mutates his classes.

Run: PYTHONPATH=tools/p3 python tools/p3/view_spec_target.py [path-to-dtype_view.py]
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from view_spec import DTYPE_WIDTH, GmemRoot, OffsetView, check

DEFAULT_I1 = Path(__file__).resolve().parents[2] / "src/tilefoundry/ir/types/dtype_view.py"


def load_i1(path: Path):
    spec = importlib.util.spec_from_file_location("i1_dtype_view", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i1_dtype_view"] = mod
    spec.loader.exec_module(mod)
    return mod


def adapt(view, generation=0) -> OffsetView:
    return OffsetView(name="%s:%s@%d" % (view.root, view.dtype, view.byte_offset),
                      root=view.root, generation=generation,
                      byte_offset=view.byte_offset, dtype=view.dtype, count=view.extent)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_I1
    if not path.is_file():
        print("no I1 dtype_view.py at", path); return 75
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    i1 = load_i1(path)
    Spec = i1.DtypeViewSpec
    print("I1 dtype_view.py sha256:", sha)
    print("(state: file is UNTRACKED in git; results are valid for THIS sha only)\n")

    root = GmemRoot("buf", 0, 1024, generation=0)
    roots = {"buf": root}
    big_roots = {"big": GmemRoot("big", 0, 1 << 40, generation=0)}

    def v(dtype, off, extent):
        return Spec(root="buf", dtype=dtype, element_bytes=DTYPE_WIDTH[dtype],
                    byte_offset=off, extent=extent, alignment=DTYPE_WIDTH[dtype],
                    root_bytes=1024)

    cases = []
    # two valid controls
    cases.append(("valid_full_bf16_to_f32", [v("f32", 0, 256)], {"0": "read"}, None, "read"))
    cases.append(("valid_read_read_overlap_diff_dtype",
                  [v("bf16", 0, 256), v("f32", 256, 64)], {"0": "read", "1": "read"}, None, "read"))
    # five rules
    cases.append(("misaligned", [v("f32", 2, 8)], {"0": "read"}, "VIEW_MISALIGNED_OFFSET", "read"))
    cases.append(("range_exceeds_parent", [v("f32", 768, 128)], {"0": "read"},
                  "VIEW_RANGE_OVERFLOWS_PARENT", "read"))
    cases.append(("overlap_write_read_same_dtype",
                  [v("bf16", 0, 256), v("bf16", 256, 128)], {"0": "write", "1": "read"},
                  "VIEW_OVERLAP_INCOMPATIBLE", "read"))
    extent_view = Spec(root="big", dtype="u8", element_bytes=1, byte_offset=0,
                       extent=(1 << 32) + 1, alignment=1, root_bytes=1 << 40)
    cases.append(("extent_overflow", [extent_view], {"0": "read"},
                  "VIEW_EXTENT_OVERFLOW", "read"))
    cases.append(("stale_after_realloc", [v("f32", 0, 8)], {"0": "read"},
                  "VIEW_STALE_AFTER_REALLOC", "read_generation_1"))

    print("%-34s %-34s %-22s %s" % ("case", "my rules on HIS objects", "his validate_view", "verdict"))
    print("-" * 118)
    fails = 0
    for name, his_views, accesses, expected, mode in cases:
        rts = big_roots if name == "extent_overflow" else roots
        views = [adapt(x, 1 if mode == "read_generation_1" else 0) for x in his_views]
        acc = {views[i].name: kind for i, kind in ((int(k), vv) for k, vv in accesses.items())}
        res = check(rts, views, acc)
        fired = sorted(r for r, m in res.items() if m)
        try:
            i1.validate_view(his_views[0]) if len(his_views) == 1 else i1.validate_views(tuple(his_views))
            his = "ACCEPT"
        except Exception as e:  # noqa: BLE001
            his = getattr(e, "reason", type(e).__name__)
        ok = (fired == [expected]) if expected else (not fired)
        fails += 0 if ok else 1
        print("%-34s %-34s %-22s %s" % (name, fired or "[]", his, "PASS" if ok else "DRIFT/FAIL"))
    print("-" * 118)

    # explicit drift probes
    print("\nDRIFT PROBES (my spec vs his validator):")
    same = [v("bf16", 0, 256), v("bf16", 256, 128)]
    try:
        i1.validate_views(tuple(same)); his_same = "ACCEPT"
    except Exception as e:  # noqa: BLE001
        his_same = getattr(e, "reason", "error")
    rr = [v("bf16", 0, 256), v("f32", 256, 64)]
    try:
        i1.validate_views(tuple(rr)); his_rr = "ACCEPT"
    except Exception as e:  # noqa: BLE001
        his_rr = getattr(e, "reason", "error")
    print("  same-dtype overlapping W x R: my spec=FORBID, his=%s (NON-CONFORMING if ACCEPT)" % his_same)
    print("  diff-dtype overlapping R x R: my spec=ALLOW, his=%s (NON-CONFORMING if rejected)" % his_rr)
    # PINNING ASSERTIONS under the Lead ruling (access-based)
    pin_same = [adapt(x) for x in same]
    res_same = check(roots, pin_same, {pin_same[0].name: "write", pin_same[1].name: "read"})
    if not res_same["VIEW_OVERLAP_INCOMPATIBLE"]:
        print("  PIN FAIL: same-dtype W x R must be FORBIDDEN"); fails += 1
    pin_rr = [adapt(x) for x in rr]
    res_rr = check(roots, pin_rr, {pin_rr[0].name: "read", pin_rr[1].name: "read"})
    if any(res_rr.values()):
        print("  PIN FAIL: cross-dtype R x R must be ALLOWED"); fails += 1
    if fails == 0:
        print("  PINNING CASES: HOLD")
    print("  allocation generation on his objects: ABSENT -> my STALE rule cannot be evaluated")
    print("  count cap on his objects: ABSENT -> my EXTENT rule cannot be evaluated")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
