#!/usr/bin/env python3
"""Adapt I1's committed DtypeViewSpec into this spec's GmemRoot/OffsetView (READ-ONLY).

Reports per-case CONFORMANCE of the target-side validator against the Lead ruling
(access-based overlap): each case says what this spec decides, what his validator decides,
and whether they conform. Two scope gaps are expected and labelled (no generation, no
element cap). This never writes to his file.

Run: PYTHONPATH=tools/p3 python tools/p3/view_spec_target.py [path]
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
        print("no dtype_view.py at", path); return 75
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    i1 = load_i1(path)
    Spec = i1.DtypeViewSpec
    print("I1 dtype_view.py sha256:", sha)
    print("(committed at 81bf059; this run is valid for THIS sha only)\n")

    root = GmemRoot("buf", 0, 1024, generation=0)
    roots = {"buf": root}
    big_roots = {"big": GmemRoot("big", 0, 1 << 40, generation=0)}

    def v(dtype, off, extent, access, root_name="buf", root_bytes=1024):
        return Spec(root=root_name, dtype=dtype, element_bytes=DTYPE_WIDTH[dtype],
                    byte_offset=off, extent=extent, alignment=DTYPE_WIDTH[dtype],
                    root_bytes=root_bytes, access=access)

    extent_view = v("u8", 0, (1 << 32) + 1, "read", root_name="big", root_bytes=1 << 40)

    # (name, roots, his_views, access_map, spec_rule, generation, REQUIRED target outcome)
    # The REQUIRED outcome is the Lead ruling restated as an assertion about HIS validator.
    # None means "no target-side requirement for this row" (single-view rules and the two
    # declared scope gaps), and is printed as such rather than counted as conforming.
    cases = [
        ("valid_full_bf16_to_f32", roots, [v("f32", 0, 256, "read")], {0: "read"}, None, 0, None),
        ("valid_read_read_overlap_diff_dtype", roots,
         [v("bf16", 0, 256, "read"), v("f32", 256, 64, "read")], {0: "read", 1: "read"}, None, 0,
         "ACCEPT"),
        ("misaligned", roots, [v("f32", 2, 8, "read")], {0: "read"}, "VIEW_MISALIGNED_OFFSET", 0,
         None),
        ("range_exceeds_parent", roots, [v("f32", 768, 128, "read")], {0: "read"},
         "VIEW_RANGE_OVERFLOWS_PARENT", 0, None),
        ("overlap_write_read_same_dtype", roots,
         [v("bf16", 0, 256, "write"), v("bf16", 256, 128, "read")], {0: "write", 1: "read"},
         "VIEW_OVERLAP_INCOMPATIBLE", 0, "overlapping-write"),
        ("extent_overflow", big_roots, [extent_view], {0: "read"}, "VIEW_EXTENT_OVERFLOW", 0, None),
        ("stale_after_realloc", roots, [v("f32", 0, 8, "read")], {0: "read"},
         "VIEW_STALE_AFTER_REALLOC", 1, None),
    ]

    print("%-34s %-30s %-22s %s" % ("case", "my spec", "his validator", "conformance"))
    print("-" * 112)
    fails = 0
    nonconforming = []
    for name, rts, his_views, access_map, expected, gen, expect_target in cases:
        views = [adapt(x, gen) for x in his_views]
        acc = {views[i].name: kind for i, kind in access_map.items()}
        res = check(rts, views, acc)
        fired = sorted(r for r, m in res.items() if m)
        try:
            i1.validate_view(his_views[0]) if len(his_views) == 1 else i1.validate_views(tuple(his_views))
            his = "ACCEPT"
        except Exception as e:  # noqa: BLE001
            his = getattr(e, "reason", type(e).__name__)
        ok = (fired == [expected]) if expected else (not fired)
        fails += 0 if ok else 1
        if expect_target is None:
            # Declared as NOT checked here (single-view rule, or the two scope gaps stated in
            # the I1 module docstring). Printed, never silently counted as CONFORMS.
            conf = ("SCOPE-GAP"
                    if name in ("extent_overflow", "stale_after_realloc")
                    else "N/A-SINGLE-VIEW")
        elif his == expect_target:
            conf = "CONFORMS"
        else:
            # AUDIT HOLE FIXED: a non-conforming target is a FAILURE of this cross-check, so
            # it must move the exit code. Printing NON-CONFORMING and exiting 0 would let a
            # wrong target validator (or a later regression in the real one) pass unnoticed.
            conf = "NON-CONFORMING"
            nonconforming.append("%s: target said %r, Lead ruling requires %r"
                                 % (name, his, expect_target))
            fails += 1
        print("%-34s %-30s %-22s %s" % (name, fired or "[]", his, conf))
    print("-" * 112)
    print("target conformance is part of THIS exit code: %d NON-CONFORMING row(s)"
          % len(nonconforming))
    for line in nonconforming:
        print("  NON-CONFORMING:", line)

    print("\nPINNING CASES (Lead ruling: access-based)")
    same = [adapt(v("bf16", 0, 256, "write")), adapt(v("bf16", 256, 128, "read"))]
    r_same = check(roots, same, {same[0].name: "write", same[1].name: "read"})
    rr = [adapt(v("bf16", 0, 256, "read")), adapt(v("f32", 256, 64, "read"))]
    r_rr = check(roots, rr, {rr[0].name: "read", rr[1].name: "read"})
    pin_ok = bool(r_same["VIEW_OVERLAP_INCOMPATIBLE"]) and not any(r_rr.values())
    print("  same-dtype overlapping W x R -> spec %s" %
          ("FORBIDS" if r_same["VIEW_OVERLAP_INCOMPATIBLE"] else "ALLOWS"))
    print("  cross-dtype overlapping R x R -> spec %s" %
          ("FORBIDS" if any(r_rr.values()) else "ALLOWS"))
    print("  PINNING CASES:", "HOLD" if pin_ok else "FAIL")
    fails += 0 if pin_ok else 1
    print("  allocation generation: his object carries NONE -> staleness is spec/plan only")
    print("  element-count cap:     his object carries NONE -> extent is spec/plan only")
    print("\nOVERALL:", "PASS (every checked row conforms)" if fails == 0
          else "FAIL (%d failed row(s), including %d NON-CONFORMING)" % (fails, len(nonconforming)))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
