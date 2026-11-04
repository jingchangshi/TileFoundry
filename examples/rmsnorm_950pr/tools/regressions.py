"""A regression harness whose every check is a subprocess with its own evidence.

Each check states what it runs, under which interpreter and environment, and
keeps the exit code, duration and captured streams. A failure therefore names
the command that produced it instead of surfacing as a traceback from inside
this harness.

A check that cannot run is NOT_RUN with its reason; a check whose command ran
and returned non-zero is FAIL. Neither is upgraded to PASS because a
neighbouring check passed.

Checks
    import_dsl    the DSL imports and a minimal program loads
    analyze       tiled_mlp type-checks and prints its typed HIR
    m2b_copy      copy_slice finalizes to TIR, and that TIR reloads and verifies

Exit codes
    0  every check PASSed
    1  a check FAILed
    2  ERROR: a check could not be started, or the report could not be written
    3  NOT_RUN: a required input is absent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PASS, FAIL, ERROR, NOT_RUN = 0, 1, 2, 3
_STATUS_EXIT = {"PASS": PASS, "FAIL": FAIL, "ERROR": ERROR, "NOT_RUN": NOT_RUN}

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/home/s00653124/workspace/tfpy312/bin/python"

# Set from --python in main(). Module-level so a check can be called on its own
# for a self-test, which is only meaningful if it uses the same interpreter.
PYTHON = os.environ.get("TF_PYTHON", DEFAULT_PYTHON)
DEFAULT_WORKSPACE = REPO_ROOT.parent

#: Written by a check into its scratch directory and then imported as a file,
#: never handed to python -c: the DSL reads a function's source lines, and a
#: program passed on the command line has none to read.
DSL_PROBE = """
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="ident", target=CpuTarget())
class Ident:
    @func
    def ident(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return x


fn = Ident.entry_function()
print("dsl_ok", fn.name, type(fn).__name__)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env() -> dict:
    """The environment every check runs under: TF source plus the Ascend target."""
    env = dict(os.environ)
    parts = [str(REPO_ROOT / "src"), str(REPO_ROOT / "targets" / "ascend")]
    prefix = env.get("PYTHONPATH")
    if prefix:
        parts.append(prefix)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_command(name: str, argv, timeout: float) -> dict:
    """Run one subprocess, keeping its exact exit status and streams."""
    record = {
        "check": name,
        "command": " ".join(str(part) for part in argv),
        "cwd": str(REPO_ROOT),
    }
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            env=_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        record.update(
            status="ERROR",
            error="interpreter not found: " + str(exc),
            duration_s=round(time.monotonic() - started, 3),
        )
        return record
    except subprocess.TimeoutExpired as exc:
        record.update(
            status="FAIL",
            error="timed out after %ss" % timeout,
            stdout=(exc.stdout or "")[-4000:],
            stderr=(exc.stderr or "")[-4000:],
            duration_s=round(time.monotonic() - started, 3),
        )
        return record
    record["duration_s"] = round(time.monotonic() - started, 3)
    record["exit_code"] = completed.returncode
    record["stdout"] = completed.stdout[-8000:]
    record["stderr"] = completed.stderr[-8000:]
    return record


def check_import_dsl(scratch: Path) -> dict:
    """The DSL must import and build a program without a backend installed."""
    probe = scratch / "dsl_probe.py"
    probe.write_text(DSL_PROBE, encoding="utf-8")
    record = run_command("import_dsl", [PYTHON, str(probe)], timeout=180.0)
    if record.get("status") == "ERROR":
        return record
    checks = {
        "exit_zero": record.get("exit_code") == 0,
        "dsl_loaded": "dsl_ok ident Function" in record.get("stdout", ""),
    }
    record["assertions"] = checks
    record["status"] = "PASS" if all(checks.values()) else "FAIL"
    return record


def check_analyze() -> dict:
    """Analyze must type-check tiled_mlp and emit its typed HIR."""
    fixture = REPO_ROOT / "tests/core/fixtures/dsl/tiled_mlp.py"
    if not fixture.is_file():
        return {
            "check": "analyze",
            "status": "NOT_RUN",
            "reason": "fixture absent: " + str(fixture),
        }
    record = run_command(
        "analyze",
        [PYTHON, "-m", "tilefoundry.cli", "analyze", str(fixture) + ":TiledMLP.tiled_mlp"],
        timeout=900.0,
    )
    if record.get("status") == "ERROR":
        return record
    out = record.get("stdout", "")
    checks = {
        "exit_zero": record.get("exit_code") == 0,
        "typed_hir": "Tensor[" in out,
        "entry_named": "tiled_mlp" in out,
        "matmul_present": "tf.matmul" in out,
    }
    record["assertions"] = checks
    record["status"] = "PASS" if all(checks.values()) else "FAIL"
    return record


def check_m2b_copy(scratch: Path) -> dict:
    """copy_slice finalizes to TIR; that TIR reloads and verifies from disk."""
    artifact = REPO_ROOT / "examples/rmsnorm_950pr/copy_slice.py"
    if not artifact.is_file():
        return {
            "check": "m2b_copy",
            "status": "NOT_RUN",
            "reason": "artifact absent: " + str(artifact),
        }
    tir = scratch / "copy_resolved_tir.py"
    finalize = run_command(
        "m2b_finalize",
        [
            PYTHON,
            "-m",
            "tilefoundry.cli",
            "schedule",
            "finalize",
            str(artifact) + ":CopySlice",
            str(tir),
        ],
        timeout=900.0,
    )
    checks = {
        "finalize_exit_zero": finalize.get("exit_code") == 0,
        "tir_written": tir.is_file(),
    }
    record = {
        "check": "m2b_copy",
        "finalize": finalize,
        "tir_path": str(tir),
    }
    if not all(checks.values()):
        record["assertions"] = checks
        record["status"] = "FAIL"
        record["exit_code"] = FAIL
        return record

    text = tir.read_text(encoding="utf-8")
    structure = {
        "copy_count": text.count("T.copy("),
        "smem_view": 'storage="smem"' in text,
        "aiv_topology": "Topology(" in text,
    }
    record["structure"] = structure
    checks["two_copies"] = structure["copy_count"] == 2
    checks["smem_staging"] = structure["smem_view"]
    checks["aiv_topology"] = structure["aiv_topology"]

    # The reload is the real M2b closure: the emitted text is a program, and
    # reloading it proves the emission round-trips. verify_tir.py then states
    # what the loaded PrimFunction verifies against.
    reload_check = run_command(
        "m2b_reload_verify",
        [PYTHON, str(TOOLS_DIR / "verify_tir.py"), "--source", str(tir)],
        timeout=300.0,
    )
    record["reload_verify"] = reload_check
    checks["tir_reloads_and_verifies"] = reload_check.get("exit_code") == 0
    record["assertions"] = checks
    record["status"] = "PASS" if all(checks.values()) else "FAIL"
    # This record is composite: its exit_code is the harness exit the status
    # maps to, while finalize and reload_verify keep their own codes above.
    record["exit_code"] = _STATUS_EXIT[record["status"]]
    return record


def check_no_silent_fallback() -> dict:
    """check_copy_slice.py refuses an unregistered source/entry instead of
    reporting the default artifact's verdict.

    This is a regression guard for a real defect: the tool used to resolve a
    custom case and then forward the DEFAULT artifact anyway, so a caller who
    named a source that was not registered still got exit 0 and a PASS about
    copy_slice.py. A verdict about a program the caller never named is worse
    than no verdict, so the refusal is now asserted rather than assumed.
    """
    tool = TOOLS_DIR / "check_copy_slice.py"
    if not tool.is_file():
        return {
            "check": "no_silent_fallback",
            "status": "NOT_RUN",
            "reason": "tool absent: " + str(tool),
        }

    real = REPO_ROOT / "examples/rmsnorm_950pr/copy_slice.py"
    cases = [
        # (label, argv, expected non-zero because)
        (
            "absent_entry",
            [str(real), "NotARealEntry"],
            "the entry does not exist in the artifact",
        ),
        (
            "absent_source",
            [str(REPO_ROOT / "examples/rmsnorm_950pr/does_not_exist.py"), "CopySlice"],
            "the source does not exist",
        ),
        (
            "registered_source_wrong_entry",
            [str(real), "CopySliceWrong"],
            "the pair is not registered, though the source is",
        ),
    ]

    results = {}
    ok = True
    for label, argv, why in cases:
        record = run_command(
            "fallback_" + label,
            [PYTHON, str(tool)] + argv,
            timeout=300.0,
        )
        refused = record.get("exit_code") != 0
        # A refusal that passed the default artifact would still be a silent
        # fallback, so the emitted report is read for the fallback marker too.
        fell_back = "delegated_to_default_case" in (
            record.get("stdout_tail") or ""
        )
        results[label] = {
            "argv": argv,
            "expectation": why,
            "exit_code": record.get("exit_code"),
            "refused": refused,
            "fell_back_to_default": fell_back,
        }
        ok = ok and refused and not fell_back

    return {
        "check": "no_silent_fallback",
        "cases": results,
        "status": "PASS" if ok else "FAIL",
        "exit_code": PASS if ok else FAIL,
    }

CHECK_NAMES = ["import_dsl", "analyze", "m2b_copy", "no_silent_fallback"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="P0 CPU regressions for the 950PR work package."
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("TF_PYTHON", DEFAULT_PYTHON),
        help="interpreter every check runs under",
    )
    parser.add_argument("--out", help="write regressions.json here")
    parser.add_argument("--log-dir", help="also write one .log per check here")
    parser.add_argument(
        "--only",
        action="append",
        choices=CHECK_NAMES,
        help="run only this check; repeatable",
    )
    args = parser.parse_args(argv)

    global PYTHON
    PYTHON = args.python

    report = {
        "tool": "regressions",
        "generated_at": _now(),
        "repo_root": str(REPO_ROOT),
        "python": PYTHON,
        "python_exists": Path(PYTHON).is_file(),
        "checks": [],
    }
    if not Path(PYTHON).is_file():
        report["status"] = "ERROR"
        report["error"] = "interpreter not found: " + PYTHON
        print(json.dumps(report, indent=2))
        return ERROR

    wanted = args.only or CHECK_NAMES
    with tempfile.TemporaryDirectory(prefix="regressions_") as scratch:
        scratch_path = Path(scratch)
        if "import_dsl" in wanted:
            report["checks"].append(check_import_dsl(scratch_path))
        if "analyze" in wanted:
            report["checks"].append(check_analyze())
        if "m2b_copy" in wanted:
            report["checks"].append(check_m2b_copy(scratch_path))
        if "no_silent_fallback" in wanted:
            report["checks"].append(check_no_silent_fallback())

    # A requested check that produced no record would otherwise vanish and the
    # run would still report PASS on whatever checks remained. That is the same
    # silent-substitution failure this harness exists to catch, so a missing
    # record is an ERROR naming the check.
    produced = {check.get("check") for check in report["checks"]}
    missing = [name for name in wanted if name not in produced]
    if missing:
        report["status"] = "ERROR"
        report["error"] = "no record produced for: " + ", ".join(missing)
        report["exit_code"] = ERROR
        print(json.dumps(report, indent=2))
        return ERROR

    for check in report["checks"]:
        if "status" not in check:
            check["status"] = "ERROR"
    statuses = [check["status"] for check in report["checks"]]
    if "FAIL" in statuses:
        report["status"] = "FAIL"
    elif "ERROR" in statuses:
        report["status"] = "ERROR"
    elif "NOT_RUN" in statuses:
        report["status"] = "NOT_RUN"
    else:
        report["status"] = "PASS"
    report["exit_code"] = _STATUS_EXIT[report["status"]]

    text = json.dumps(report, indent=2)
    print(text)

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for check in report["checks"]:
            lines = [
                "check: " + str(check.get("check")),
                "status: " + str(check.get("status")),
                "command: " + str(check.get("command")),
                "exit_code: " + str(check.get("exit_code")),
                "duration_s: " + str(check.get("duration_s")),
                "assertions: " + json.dumps(check.get("assertions", {})),
            ]
            if check.get("error"):
                lines.append("error: " + str(check["error"]))
            if check.get("stderr"):
                lines.append("--- stderr ---")
                lines.append(str(check["stderr"]))
            if check.get("stdout"):
                lines.append("--- stdout (tail) ---")
                lines.append(str(check["stdout"])[-3000:])
            (log_dir / (str(check.get("check")) + ".log")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print("regressions: wrote " + str(destination), file=sys.stderr)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())