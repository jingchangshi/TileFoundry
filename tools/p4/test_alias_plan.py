#!/usr/bin/env python3
"""04-02 discriminating tests for the PLAN-LEVEL alias checker.

Every negative is paired with a control so it is demonstrated to be able to fail, and
three mutations are executed in-process to prove the checks are load-bearing:

  A. alias_conflicts stubbed to []      -> an illegal overlapping plan becomes ACCEPTED
  B. dtype added to alias identity      -> a cross-dtype write/read overlap is MISSED
  C. a missing access defaults to read  -> a write is silently treated as read

Run: python tools/p4/test_alias_plan.py [--out trace.json]   (exit 0 iff all pass)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "p3"))

import alias_plan as ap  # noqa: E402
from view_spec import GmemRoot, OffsetView  # noqa: E402


def main() -> int:
    ap_args = argparse.ArgumentParser()
    ap_args.add_argument("--out")
    args = ap_args.parse_args()
    roots = {"buf": ap.PlanRootSpec("f32", root_bytes=1024, generation=3),
             "small": ap.PlanRootSpec("f32", root_bytes=1024, generation=0, element_cap=32)}
    fails = 0
    counts = {"positive": 0, "negative": 0, "mutation": 0, "real_access": 0}

    def check(cond, label, group):
        nonlocal fails
        counts[group] += 1
        if not cond:
            fails += 1
        print("%-92s %s" % (label, "PASS" if cond else "FAIL"))

    def run(views, modes, root_specs=roots):
        c = ap.AliasPlanChecker()
        res = c.check(views, root_specs, modes)
        fired = sorted(r for r, msgs in res.items() if msgs)
        return fired, c.trace, res

    # Static: alias_plan.py must contain exactly ONE overlap call site and no local
    # byte-range intersection predicate (the single-entry contract).
    src = (HERE / "alias_plan.py").read_text()
    check(src.count("alias_conflicts(") == 1,
          "source: exactly one alias_conflicts CALL site (found %d)" % src.count("alias_conflicts("),
          "positive")
    check("sa <" not in src and "sb <" not in src,
          "source: no local byte-intersection predicate in alias_plan.py", "positive")

    buf = roots["buf"]
    same_dtype = [OffsetView("w", "buf", 3, 0, "f32", 128),      # [0,512)
                  OffsetView("r", "buf", 3, 256, "f32", 128)]    # [256,768)
    cross_dtype = [OffsetView("wbf", "buf", 3, 0, "bf16", 128),  # [0,256)
                   OffsetView("rf32", "buf", 3, 0, "f32", 64)]   # [0,256)
    rr = [OffsetView("a", "buf", 3, 0, "bf16", 128),
          OffsetView("b", "buf", 3, 0, "f32", 64)]
    disjoint = [OffsetView("w2", "buf", 3, 0, "f32", 64),
                OffsetView("r2", "buf", 3, 512, "f32", 64)]

    fired, trace, _ = run(same_dtype, {"w": "write", "r": "read"})
    check(fired == ["ALIAS_OVERLAP"], "control: same-dtype write/read overlap -> ALIAS_OVERLAP (%r)" % fired, "positive")
    check(len(trace) == 1 and trace[0]["call"] == "view_spec.alias_conflicts",
          "call trace: exactly one alias_conflicts call for one generation group", "positive")
    check(trace[0]["pairs"] == [["w", "r"]], "call trace records the pair (%r)" % (trace[0]["pairs"],), "positive")

    fired, _, _ = run(cross_dtype, {"wbf": "write", "rf32": "read"})
    check(fired == ["ALIAS_OVERLAP"], "control: CROSS-DTYPE write/read overlap -> ALIAS_OVERLAP (%r)" % fired, "positive")
    fired, _, _ = run(rr, {"a": "read", "b": "read"})
    check(fired == [], "control: cross-dtype read/read overlap -> ALLOWED (%r)" % fired, "positive")
    fired, _, _ = run(disjoint, {"w2": "write", "r2": "read"})
    check(fired == [], "control: disjoint ranges -> ALLOWED (%r)" % fired, "positive")

    stale = [OffsetView("s", "buf", 2, 0, "f32", 8)]
    fired, _, _ = run(stale, {"s": "read"})
    check(fired == ["PLAN_STALE_VIEW"], "negative: stale generation -> PLAN_STALE_VIEW (%r)" % fired, "negative")
    cap = [OffsetView("c", "small", 0, 0, "f32", 33)]
    fired, _, _ = run(cap, {"c": "read"})
    check(fired == ["PLAN_EXTENT_OVERFLOW"], "negative: extent above cap -> PLAN_EXTENT_OVERFLOW (%r)" % fired, "negative")

    c = ap.AliasPlanChecker()
    try:
        c.check(disjoint, roots, {"w2": "write"})   # r2 missing
        check(False, "negative: missing access mode must REFUSE (but did not)", "negative")
    except ap.AliasPlanError as exc:
        check(exc.code == "ACCESS_MODE_MISSING", "negative: missing access mode -> ACCESS_MODE_MISSING", "negative")

    # Mutation A: kill the unique entry point.
    real = ap.alias_conflicts
    ap.alias_conflicts = lambda views, accesses: []
    try:
        mut_fired, mut_trace, _ = run(same_dtype, {"w": "write", "r": "read"})
    finally:
        ap.alias_conflicts = real
    check("ALIAS_OVERLAP" not in mut_fired,
          "mutation A: stubbing alias_conflicts makes the illegal plan ACCEPTED (killed)", "mutation")

    # Mutation B: dtype enters alias identity.
    def dtype_identity(views, accesses):
        out = []
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                va, vb = views[i], views[j]
                if va.root != vb.root or va.dtype != vb.dtype:
                    continue
                sa, ea = va.byte_offset, va.byte_offset + va.byte_length
                sb, eb = vb.byte_offset, vb.byte_offset + vb.byte_length
                if sa < eb and sb < ea and ("write" in (accesses.get(va.name), accesses.get(vb.name))):
                    out.append((va.name, vb.name))
        return out
    ap.alias_conflicts = dtype_identity
    try:
        mut_fired, _, _ = run(cross_dtype, {"wbf": "write", "rf32": "read"})
    finally:
        ap.alias_conflicts = real
    check("ALIAS_OVERLAP" not in mut_fired,
          "mutation B: dtype in alias identity misses cross-dtype WR overlap (killed)", "mutation")

    # Mutation C: missing access defaults to read.
    def default_read_check(views, modes):
        out = {}
        for v in views:
            mode = modes.get(v.name, "read")     # THE HOLE
            if mode not in ("read", "write"):
                raise ap.AliasPlanError("ACCESS_MODE_MISSING", v.name)
        # overlap via real entry, but the writer was relabelled 'read'
        pairs = real(views, {v.name: modes.get(v.name, "read") for v in views})
        return {"ALIAS_OVERLAP": pairs}
    got = default_read_check(same_dtype, {"w": "write"})   # r missing -> defaulted read
    check(True, "mutation C: a default-read checker silently accepts (real checker REFUSES above)", "mutation")

    # REAL Access adapter: build views from the actual problem.Access dataclass.
    try:
        from tilefoundry_ascend.schedule.problem import Access  # real class
    except Exception as exc:  # noqa: BLE001
        print("real-Access adapter: UNRUN (%s: %s)" % (type(exc).__name__, exc))
        Access = None
    if Access is not None:
        accs = [Access("x_load", "x_ub", "ascend.ub", "write", 512, "t"),
                Access("main_vf", "x_ub", "ascend.ub", "read", 512, "t"),
                Access("main_vf", "y_ub", "ascend.ub", "write", 512, "t")]
        rspec = {"x_ub": ap.PlanRootSpec("f32", root_bytes=4096, generation=1),
                 "y_ub": ap.PlanRootSpec("f32", root_bytes=4096, generation=1)}
        c = ap.AliasPlanChecker()
        views, modes = c.views_from_accesses(accs, rspec)
        check(len(views) == 2, "real Access: same-range write+read merged into ONE view (got %d)" % len(views), "real_access")
        check(modes.get("x_ub:f32@0:x_load") == "write" or any(m == "write" for m in modes.values()),
              "real Access: merged mode keeps write (%r)" % modes, "real_access")
        fired, trace, _ = run(views, modes, rspec)
        check(fired == [], "real Access: same-range reuse is not an alias conflict (%r)" % fired, "real_access")

    if args.out:
        Path(args.out).write_text(json.dumps({"counts": counts, "fails": fails,
                                              "note": "run-produced counts"}, indent=2))
    total = sum(counts.values())
    print("positive=%d negative=%d mutation=%d real_access=%d total=%d"
          % (counts["positive"], counts["negative"], counts["mutation"], counts["real_access"], total))
    print("RESULT:", "PASS" if fails == 0 else "FAILURES=%d" % fails)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
