#!/usr/bin/env python3
"""Prove that view_spec.check() decides overlap ONLY through alias_conflicts.

The refactor that removes a second range-intersection loop is worthless if nothing can
detect its return. Three discriminating assertions, each of which fails if the unique
entry point is bypassed:

  1. the module carries no second overlap implementation (no _overlaps);
  2. replacing alias_conflicts with a stub that returns [] makes the overlap matrix cases
     FAIL (so the matrix is load-bearing on that call, not on duplicated logic);
  3. check() actually invokes alias_conflicts exactly once per check and forwards exactly
     the views whose parent root is known.

Run: python tools/p3/test_view_spec_single_entry.py   (exit 0 iff all three hold)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_spec as vs  # noqa: E402

FAILS = 0


def check(cond: bool, label: str) -> None:
    global FAILS
    if not cond:
        FAILS += 1
    print("%-70s %s" % (label, "PASS" if cond else "FAIL"))


def main() -> int:
    check(not hasattr(vs, "_overlaps"),
          "no second overlap implementation (_overlaps absent)")

    # Baseline: the real matrix is discriminating.
    base_rc = vs.main()
    check(base_rc == 0, "unpatched matrix exits 0 (real alias_conflicts)")

    # Mutation 1: alias_conflicts becomes a no-op. The overlap cases must now fail.
    real = vs.alias_conflicts
    vs.alias_conflicts = lambda views, accesses: []
    try:
        mut_rc = vs.main()
    finally:
        vs.alias_conflicts = real
    check(mut_rc != 0, "stubbed alias_conflicts makes the matrix exit NON-zero (mutation killed)")

    # Mutation 2: trace the call and the argument it receives.
    calls = []

    def tracing(views, accesses):
        calls.append((tuple(v.name for v in views), dict(accesses)))
        return real(views, accesses)

    vs.alias_conflicts = tracing
    try:
        roots = {"buf": vs.GmemRoot("buf", 512, 1024, generation=3),
                 "other": vs.GmemRoot("other", 0, 64, generation=0)}
        wr = [vs.OffsetView("w", "buf", 3, 0, "bf16", 256),
              vs.OffsetView("r", "buf", 3, 256, "f32", 64)]
        res = vs.check(roots, wr, {"w": "write", "r": "read"})
        rr = [vs.OffsetView("a", "buf", 3, 0, "bf16", 256),
              vs.OffsetView("b", "buf", 3, 256, "f32", 64)]
        res_rr = vs.check(roots, rr, {"a": "read", "b": "read"})
        # A view with an unknown parent must NOT be forwarded with a bogus root.
        unknown = [vs.OffsetView("x", "nope", 0, 0, "f32", 4)]
        res_unknown = vs.check(roots, unknown, {"x": "read"})
    finally:
        vs.alias_conflicts = real

    check(len(calls) == 3, "check() calls alias_conflicts once per check (got %d)" % len(calls))
    check(calls[0][0] == ("w", "r"), "write/read pair forwarded: %r" % (calls[0][0],))
    check(bool(res["VIEW_OVERLAP_INCOMPATIBLE"]),
          "overlap decided by alias_conflicts is reported as VIEW_OVERLAP_INCOMPATIBLE")
    check(not any(res_rr.values()), "cross-dtype read/read overlap remains ALLOWED")
    check(calls[1][0] == ("a", "b"), "read/read pair forwarded (rule, not early exit)")
    check(calls[2][0] == (), "unknown-root views are not forwarded to alias_conflicts")
    check(bool(res_unknown["VIEW_STALE_AFTER_REALLOC"]),
          "unknown root still reported as its own rule")

    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
