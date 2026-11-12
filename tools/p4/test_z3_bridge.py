#!/usr/bin/env python3
"""04-07/08 mutation runner: each named mutation in its OWN process, with exit file.

Process exit 0 means the mutation was KILLED (expected named refusal / state / change).
The runner writes <out>/<name>.exit plus raw stdout/stderr and MUTATIONS.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF = HERE.parents[1]
BRIDGE = HERE / "z3_bridge.py"
sys.path.insert(0, str(HERE))
import z3_bridge  # noqa: E402


def write(path: Path, text: str) -> None:
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--problem-source", required=True, choices=["ledger", "template"],
                    help="EXPLICIT source; ledger for 04-09, template only 05dcfcea old-domain regression")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        for child in out.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        out.mkdir(parents=True)
    env = dict(os.environ,
               PYTHONPATH=str(TF / "src") + os.pathsep + str(TF / "targets" / "ascend"),
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1")
    rows = []
    for name in z3_bridge.MUTATIONS:
        p = subprocess.run([sys.executable, str(BRIDGE), "--out", str(out),
                            "--mutation", name, "--problem-source", args.problem_source],
                           capture_output=True, text=True, env=env, timeout=600)
        write(out / (name + ".exit"), str(p.returncode) + chr(10))
        write(out / (name + ".stdout"), p.stdout)
        write(out / (name + ".stderr"), p.stderr)
        jf = out / (name + ".json")
        report = json.loads(jf.read_text()) if jf.is_file() else {}
        killed = bool(report.get("killed")) and p.returncode == 0
        rows.append({"name": name, "process_exit": p.returncode,
                     "killed": killed, "named_code": report.get("named_code"),
                     "detail": report.get("detail")})
        print("%-30s exit=%s killed=%s code=%s" %
              (name, p.returncode, killed, report.get("named_code")))
    summary = {"total": len(rows), "killed": sum(r["killed"] for r in rows),
               "all_ok": all(r["killed"] for r in rows), "rows": rows,
               "problem_source": args.problem_source,
               "old_domain_05dcfcea_regression": args.problem_source == "template",
               "not_04_09_evidence": args.problem_source == "template",
               "explicit_source_required": True,
               "auto_template_fallback_forbidden": True,
               "mode": "CPU_PLANNING_ONLY", "no_device": True}
    write(out / "MUTATIONS.json", json.dumps(summary, indent=2, sort_keys=True) + chr(10))
    print("RESULT killed=%d/%d all_ok=%s" %
          (summary["killed"], summary["total"], summary["all_ok"]))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
