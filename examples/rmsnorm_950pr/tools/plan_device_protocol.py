#!/usr/bin/env python3
"""06-07..06-10 PLAN-side device-protocol rehearsal (NO device).

For each case it builds a UNIQUE new case directory with the frozen layout:
  device_info.json, numeric.json, patterns.json, tensors.json,
  npu-smi.before.txt, npu-smi.after.txt, exit_code.txt, phases.json,
  timing/windows.json (raw), timing/aggregate.json (computed separately).
It then validates: layout, npu-smi health + busy semantics (busy is RECORDED, never a
refusal), raw timing-window completeness, and the repeats/stability/reuse rules.

PLAN mode is a REHEARSAL over archived run2 products: tensors/numeric/smi/exit_code are
copies, timing windows are synthesized (paired windows reuse archived host timings) and are
labelled synthetic; there is no device output and no P6 credit.

Exit: 0 PASS, 1 FAIL, 2 REFUSED. Failure codes are named per stage.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

TF = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TF / "tools" / "p3"))
import direct_diff as dd  # noqa: E402

FROZEN = TF.parent / "TileFoundry_Ascend950PR_AgentTeam/artifacts/run01/p0/precision_protocol.json"
# npu-smi is checked in its own stage so a missing smi is SMI_MISSING, not LAYOUT_MISSING.
LAYOUT = ("device_info.json", "numeric.json", "patterns.json", "tensors.json",
          "exit_code.txt", "phases.json", "timing/windows.json")
SMI_FILES = ("npu-smi.before.txt", "npu-smi.after.txt")


class Refuse(RuntimeError):
    def __init__(self, code, msg):
        super().__init__("%s: %s" % (code, msg))
        self.code, self.msg = code, msg


def sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_smi(text: str) -> dict:
    """Health + occupied. Busy is data, never an error by itself."""
    out = {"readable": True, "health_ok": None, "busy": None, "process_rows": 0,
           "no_running_marker": "No running processes found" in text}
    for line in text.splitlines():
        if "Ascend950PR" in line and "|" in line:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3 and parts[2] == "OK":
                out["health_ok"] = True
            elif len(parts) >= 3 and parts[2] not in ("", "Health"):
                out["health_ok"] = False
        if line.strip().startswith("| 0") or line.strip().startswith("|0"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 4 and parts[1].isdigit() and parts[2] not in ("Process id",):
                out["process_rows"] += 1
    if not out["no_running_marker"]:
        out["busy"] = out["process_rows"] > 0
    else:
        out["busy"] = False
    if out["health_ok"] is None:
        out["health_ok"] = "Ascend950PR" not in text
    return out


def synth_windows(case_dir: Path, tj: dict, numeric: dict, paired: dict | None) -> dict:
    patterns = [e["key"] for e in tj["tensors"]]
    reps = int(numeric.get("tests") and 4 or 4)
    stab_runs = int((numeric.get("stability") or {}).get("runs", 64))
    windows = []
    wid = 0
    t = 1_000_000_000
    for p in patterns:
        for r in range(reps):
            wid += 1; t += 1000
            windows.append({"window_id": wid, "kind": "repeat", "pattern": p, "repeat_idx": r,
                            "npu_event_pair": [wid * 2, wid * 2 + 1],
                            "t_start_ns": t, "t_end_ns": t + 900,
                            "host_wall": "2026-11-05T00:00:00+0800", "synthetic": True,
                            "source": "PLAN_REHEARSAL_SYNTHETIC"})
    for r in range(stab_runs):
        wid += 1; t += 1000
        windows.append({"window_id": wid, "kind": "stability", "pattern": "fixed_input",
                        "repeat_idx": r, "npu_event_pair": [wid * 2, wid * 2 + 1],
                        "t_start_ns": t, "t_end_ns": t + 800,
                        "host_wall": "2026-11-05T00:00:00+0800", "synthetic": True,
                        "source": "PLAN_REHEARSAL_SYNTHETIC"})
    if paired:
        for i, (base_ms, cand_ms) in enumerate(zip(paired.get("baseline_ms") or [],
                                                    paired.get("candidate_ms") or [])):
            wid += 1; t += 1000
            windows.append({"window_id": wid, "kind": "paired", "pattern": "paired_%d" % i,
                            "repeat_idx": i, "npu_event_pair": [wid * 2, wid * 2 + 1],
                            "t_start_ns": t, "t_end_ns": t + int(base_ms * 1e6),
                            "host_wall": "2026-11-05T00:00:00+0800", "synthetic": False,
                            "source": "ARCHIVED_HOST_WINDOW_MS",
                            "baseline_ms": base_ms, "candidate_ms": cand_ms})
    expected = (len(patterns) * reps) + stab_runs + (len(paired.get("baseline_ms") or []) if paired else 0)
    return {"schema": "raw_event_paired_windows_v1", "synthetic_rehearsal": True,
            "window_count": len(windows), "expected_window_count": expected, "windows": windows}


def aggregate(windows: dict) -> dict:
    """Computed SEPARATELY from the raw windows."""
    durs = [w["t_end_ns"] - w["t_start_ns"] for w in windows.get("windows", [])]
    paired = [w.get("candidate_ms") for w in windows.get("windows", []) if w.get("kind") == "paired"]
    paired = [x for x in paired if x is not None]
    out = {"raw_windows_sha256": None, "count": len(durs), "synthetic": True,
           "mean_ns": statistics.fmean(durs) if durs else None,
           "median_ns": statistics.median(durs) if durs else None,
           "p95_ns": (sorted(durs)[max(0, int(round(0.95 * (len(durs) - 1))))] if durs else None),
           "paired_candidate_ms_median": statistics.median(paired) if paired else None,
           "note": "aggregate computed from the raw windows file, not from device output"}
    return out


def build_case(archived: Path, out_case: Path, protocol: dict, *, busy_policy="record") -> dict:
    """Create the unique case dir layout and run the PLAN validations."""
    if out_case.exists() and any(out_case.iterdir()):
        raise Refuse("CASE_DIR_NOT_NEW", "%s exists and is not empty" % out_case)
    out_case.mkdir(parents=True, exist_ok=True)
    phases = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {}, "status": None}

    def phase(name, status, code=None, detail=None):
        phases["stages"][name] = {"status": status, "code": code, "detail": detail,
                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (out_case / "phases.json").write_text(json.dumps(phases, indent=2) + "\n")

    # layout: copy archived inputs, label rehearsal
    tj = json.loads((archived / "tensors.json").read_text())
    numeric = json.loads((archived / "numeric.json").read_text())
    for name in ("tensors.json", "numeric.json"):
        shutil.copy2(archived / name, out_case / name)
    shutil.copy2(archived / "device_info.json", out_case / "device_info.json")
    shutil.copy2(archived / "exit_code.txt", out_case / "exit_code.txt")
    for name in ("npu-smi.before.txt", "npu-smi.after.txt"):
        src = archived / name
        if src.is_file():
            shutil.copy2(src, out_case / name)
    patterns = [{"key": e["key"], "pattern": e.get("pattern", e["key"].split("_")[0]),
                 "seed": e.get("seed", 0), "deterministic": e.get("pattern") in ("zero", "ones", "alternating")}
                for e in tj["tensors"]]
    (out_case / "patterns.json").write_text(json.dumps(patterns, indent=2) + "\n")
    (out_case / "timing").mkdir(exist_ok=True)
    paired_path = archived / "paired_benchmark.json"
    paired = json.loads(paired_path.read_text()) if paired_path.is_file() else None
    windows = synth_windows(out_case, tj, numeric, paired)
    (out_case / "timing/windows.json").write_text(json.dumps(windows, indent=2) + "\n")
    agg = aggregate(windows)
    agg["raw_windows_sha256"] = sha(out_case / "timing/windows.json")
    (out_case / "timing/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    phase("layout", "PASS", detail="required files written")
    try:
        # layout validation
        missing = [n for n in LAYOUT if not (out_case / n).is_file()]
        if missing:
            raise Refuse("LAYOUT_MISSING", ",".join(missing))
        phase("layout_validate", "PASS", detail="all required files present")
        # npu-smi:
        for tag in ("before", "after"):
            p = out_case / ("npu-smi.%s.txt" % tag)
            if not p.is_file() or not p.read_text().strip():
                raise Refuse("SMI_MISSING", "npu-smi.%s missing/empty" % tag)
            s = parse_smi(p.read_text())
            if s["health_ok"] is not True:
                raise Refuse("DEVICE_HEALTH", "%s health not OK" % tag)
            if s["busy"] and busy_policy == "refuse":
                raise Refuse("BUSY_MISJUDGED",
                             "busy device was treated as a failure; busy must be recorded")
            phase("smi_%s" % tag, "PASS", detail="busy=%s process_rows=%d (recorded)" %
                  (s["busy"], s["process_rows"]))
        # timing completeness
        got = len(windows.get("windows") or [])
        if got != windows.get("expected_window_count"):
            raise Refuse("TIMER_WINDOW_MISSING",
                         "raw windows %d != expected %d" % (got, windows["expected_window_count"]))
        ids = [w.get("window_id") for w in windows["windows"]]
        if len(set(ids)) != len(ids):
            raise Refuse("TIMER_WINDOW_DUPLICATE", "duplicate window ids")
        phase("timing_windows", "PASS", detail="%d raw windows; aggregate separate" % got)
        # numeric repeats/stability/reuse
        r = dd.check_repeats_stability_reuse(numeric, protocol)
        (out_case / "numeric_verdict.json").write_text(json.dumps(r, indent=2) + "\n")
        if not r["pass"]:
            code = next(k for k, v in r["rules"].items() if v)
            raise Refuse(code, json.dumps(r["rules"])[:300])
        phase("numeric", "PASS", detail="repeats/stability/reuse pass")
        phases["status"] = "PASS"
        (out_case / "phases.json").write_text(json.dumps(phases, indent=2) + "\n")
        return {"case": out_case.name, "status": "PASS", "busy_policy": busy_policy,
                "windows": got, "aggregate": agg, "phases": phases}
    except Refuse as exc:
        phase("verdict", "REFUSED", exc.code, exc.msg)
        phases["status"] = "REFUSED"
        (out_case / "phases.json").write_text(json.dumps(phases, indent=2) + "\n")
        return {"case": out_case.name, "status": "REFUSED", "code": exc.code,
                "detail": exc.msg, "phases": phases}


def validate_case(case_dir: Path, protocol: dict, *, busy_policy="record") -> dict:
    """Validate an EXISTING case dir (used by PLAN mutation tests)."""
    phases = {"stages": {}, "status": None}

    def phase(name, status, code=None, detail=None):
        phases["stages"][name] = {"status": status, "code": code, "detail": detail}
        (case_dir / "phases.json").write_text(json.dumps(phases, indent=2) + chr(10))
    try:
        missing = [n for n in LAYOUT if not (case_dir / n).is_file()]
        if missing:
            raise Refuse("LAYOUT_MISSING", ",".join(missing))
        phase("layout_validate", "PASS")
        busy = {}
        for tag in ("before", "after"):
            p = case_dir / ("npu-smi.%s.txt" % tag)
            if not p.is_file() or not p.read_text().strip():
                raise Refuse("SMI_MISSING", "npu-smi.%s missing/empty" % tag)
            s = parse_smi(p.read_text())
            busy[tag] = s["busy"]
            if s["health_ok"] is not True:
                raise Refuse("DEVICE_HEALTH", "%s health not OK" % tag)
            if s["busy"] and busy_policy == "refuse":
                raise Refuse("BUSY_MISJUDGED", "busy must be recorded, not refused")
            phase("smi_%s" % tag, "PASS", detail="busy=%s" % s["busy"])
        wpath = case_dir / "timing/windows.json"
        win = json.loads(wpath.read_text())
        if len(win.get("windows") or []) != win.get("expected_window_count"):
            raise Refuse("TIMER_WINDOW_MISSING", "%d != %d" % (len(win.get("windows") or []),
                                                              win.get("expected_window_count")))
        phase("timing_windows", "PASS", detail="%d" % len(win["windows"]))
        numeric = json.loads((case_dir / "numeric.json").read_text())
        r = dd.check_repeats_stability_reuse(numeric, protocol)
        (case_dir / "numeric_verdict.json").write_text(json.dumps(r, indent=2) + chr(10))
        if not r["pass"]:
            code = next(k for k, v in r["rules"].items() if v)
            raise Refuse(code, json.dumps(r["rules"])[:300])
        phase("numeric", "PASS")
        phases["status"] = "PASS"
        (case_dir / "phases.json").write_text(json.dumps(phases, indent=2) + chr(10))
        return {"case": case_dir.name, "status": "PASS", "busy": busy, "windows": len(win["windows"])}
    except Refuse as exc:
        phase("verdict", "REFUSED", exc.code, exc.msg)
        phases["status"] = "REFUSED"
        (case_dir / "phases.json").write_text(json.dumps(phases, indent=2) + chr(10))
        return {"case": case_dir.name, "status": "REFUSED", "code": exc.code, "detail": exc.msg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archived-root", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--protocol", default=str(FROZEN))
    ap.add_argument("--busy-policy", choices=["record", "refuse"], default="record")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--case", default=None, help="single archived case dir (mutation mode)")
    ap.add_argument("--validate-only", action="store_true",
                    help="validate an existing case dir (no rebuild)")
    args = ap.parse_args()
    protocol = dd.load_protocol(Path(args.protocol))
    out_root = Path(args.out_root) if args.out_root else None
    if args.validate_only:
        case_dir = Path(args.case) if args.case else out_root
        res = validate_case(case_dir, protocol, busy_policy=args.busy_policy)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "PASS" else 2
    if args.case:
        archived = Path(args.case)
        out_case = out_root / archived.name
        try:
            res = build_case(archived, out_case, protocol, busy_policy=args.busy_policy)
        except Refuse as exc:
            print(json.dumps({"case": out_case.name, "status": "REFUSED", "code": exc.code,
                              "detail": exc.msg}, indent=2))
            return 2
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "PASS" else (2 if res["status"] == "REFUSED" else 1)
    names = args.cases or sorted(p.name for p in Path(args.archived_root).iterdir() if p.is_dir())
    results = []
    for n in names:
        res = build_case(Path(args.archived_root) / n, out_root / n, protocol,
                         busy_policy=args.busy_policy)
        results.append({"case": n, "status": res["status"], "code": res.get("code")})
    (out_root / "PREP_REPORT.json").write_text(json.dumps({"mode": "plan_rehearsal",
                                                          "results": results}, indent=2) + "\n")
    ok = all(r["status"] == "PASS" for r in results)
    print("PLAN cases=%d pass=%d" % (len(results), sum(r["status"] == "PASS" for r in results)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
