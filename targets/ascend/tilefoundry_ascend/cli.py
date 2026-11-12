"""Command-line surface for the Ascend950PR compiler bundle.

Usage:
  python -m tilefoundry_ascend.cli facts --out DIR
  python -m tilefoundry_ascend.cli microtests --out DIR
  python -m tilefoundry_ascend.cli case --case-id prefill_dyn --shape 8192 4096 --rows 4 --loop-mode dyn --stages 4 --xversions 4 --yversions 2 --out DIR
  python -m tilefoundry_ascend.cli cases --cases path/to/cases.json --out DIR
  python -m tilefoundry_ascend.cli trip-tests --case-id decode_dyn --shape 64 4096 --rows 1 --loop-mode dyn --stages 2 --xversions 2 --yversions 1 --out DIR
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

from .pipeline import compile_case, trip_count_tests
from .evaluator.evalref import microtests
from .target.target import ascend950pr_profile
from .memory import SPACE_FACTS, transfer_contract, MemorySpaceRef
from .ir.isa import ISA


def _write(dirpath, name, obj):
    p = Path(dirpath); p.mkdir(parents=True, exist_ok=True)
    (p / name).write_text(json.dumps(obj, indent=2) if not isinstance(obj, str) else obj)
    return str(p / name)


def cmd_facts(args):
    prof = ascend950pr_profile()
    _write(args.out, "target_profile.json", prof.as_dict())
    _write(args.out, "memory_spaces.json", {k: v.as_dict() for k, v in SPACE_FACTS.items()})
    transfers = {}
    for src in ("ascend.gm", "ascend.ub", "ascend.l1", "ascend.l0c"):
        for dst in ("ascend.gm", "ascend.ub", "ascend.l1", "ascend.l0a", "ascend.l0b", "ascend.l0c"):
            tc = transfer_contract(src, dst)
            transfers[f"{src}->{dst}"] = {"legal": tc.legal, "pipe": tc.issuer, "note": tc.note}
    _write(args.out, "transfers.json", transfers)
    _write(args.out, "isa.json", {k: {
        "category": v.category, "compute_domain": v.compute_domain,
        "required_region": v.required_region, "core_affinity": v.core_affinity,
        "resource": v.resource, "completion": v.completion, "semantics": v.semantics,
    } for k, v in ISA.items()})
    # alias demo: UB of different cores same offset must not alias
    a = MemorySpaceRef("ascend.ub", "aiv:0").alias_key()
    b = MemorySpaceRef("ascend.ub", "aiv:1").alias_key()
    g1 = MemorySpaceRef("ascend.gm", "aiv:0").alias_key()
    g2 = MemorySpaceRef("ascend.gm", "aiv:1").alias_key()
    _write(args.out, "alias_demo.json", {
        "ub_core0": a, "ub_core1": b, "ub_alias": a == b,
        "gm_core0": g1, "gm_core1": g2, "gm_alias": g1 == g2})
    print(json.dumps({"facts_written": args.out}))


def cmd_microtests(args):
    mt = microtests()
    _write(args.out, "microtests.json", mt)
    fails = [k for k, v in mt.items() if not v.get("ok")]
    print(json.dumps({"microtests": len(mt), "fails": fails}))
    return 0 if not fails else 1


def cmd_case(args):
    case = {"case_id": args.case_id, "shape": [args.shape[0], args.shape[1]], "rows": args.rows,
            "loop_mode": args.loop_mode, "requested_num_stages": args.stages,
            "x_versions": args.xversions, "y_versions": args.yversions, "eps": args.eps}
    r = compile_case(case)
    d = Path(args.out) / args.case_id; d.mkdir(parents=True, exist_ok=True)
    _write(d, "program.json", r["program"])
    _write(d, "capacity.json", r["capacity"])
    _write(d, "problem.json", r["problem"])
    _write(d, "solve.json", r["solve"])
    _write(d, "plan.json", r["plan"])
    _write(d, "resolved_tir.txt", r["tir_text"])
    _write(d, "verify.json", r["verify"])
    _write(d, "negative_tests.json", r["negative_tests"])
    _write(d, "summary.json", {k: r[k] for k in ("case_id", "capacity", "verify", "all_negative_rejected", "plan_digest", "verify_steps")})
    print(json.dumps({"case": args.case_id, "status": r["solve"]["status"], "verify_ok": r["verify"]["ok"],
                      "negatives_rejected": r["all_negative_rejected"], "out": str(d)}))
    return 0 if (r["solve"]["status"] == "solved" and r["verify"]["ok"] and r["all_negative_rejected"]) else 1


def cmd_cases(args):
    data = json.loads(Path(args.cases).read_text())
    rc = 0
    for c in data["cases"]:
        ns = argparse.Namespace(case_id=c["case_id"], shape=c["shape"], rows=c["rows"],
                                loop_mode=c["loop_mode"], stages=c["requested_num_stages"],
                                xversions=c["x_versions"], yversions=c["y_versions"],
                                eps=c.get("eps", 1e-6), out=args.out)
        rc |= cmd_case(ns)
    return rc


def cmd_trip(args):
    case = {"case_id": args.case_id, "shape": [args.shape[0], args.shape[1]], "rows": args.rows,
            "loop_mode": args.loop_mode, "requested_num_stages": args.stages,
            "x_versions": args.xversions, "y_versions": args.yversions, "eps": args.eps}
    r = trip_count_tests(case)
    _write(args.out, "trip_tests.json", r)
    print(json.dumps({"trips": list(r.keys()), "all_verify_ok": all(v["verify_ok"] for v in r.values())}))
    return 0 if all(v["verify_ok"] for v in r.values()) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("facts"); p.add_argument("--out", required=True); p.set_defaults(func=cmd_facts)
    p = sub.add_parser("microtests"); p.add_argument("--out", required=True); p.set_defaults(func=cmd_microtests)
    p = sub.add_parser("case")
    p.add_argument("--case-id", required=True); p.add_argument("--shape", type=int, nargs=2, required=True)
    p.add_argument("--rows", type=int, required=True); p.add_argument("--loop-mode", required=True)
    p.add_argument("--stages", type=int, required=True); p.add_argument("--xversions", type=int, required=True)
    p.add_argument("--yversions", type=int, required=True); p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_case)
    p = sub.add_parser("cases"); p.add_argument("--cases", required=True); p.add_argument("--out", required=True); p.set_defaults(func=cmd_cases)
    p = sub.add_parser("trip-tests")
    p.add_argument("--case-id", required=True); p.add_argument("--shape", type=int, nargs=2, required=True)
    p.add_argument("--rows", type=int, required=True); p.add_argument("--loop-mode", required=True)
    p.add_argument("--stages", type=int, required=True); p.add_argument("--xversions", type=int, required=True)
    p.add_argument("--yversions", type=int, required=True); p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_trip)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
