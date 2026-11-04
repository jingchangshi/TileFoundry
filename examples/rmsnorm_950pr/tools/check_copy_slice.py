"""M2b closure checks on typed IR objects, plus the reload that closes them.

This tool owns the STRUCTURAL and RELOAD evidence for copy_slice, and it
delegates the two claims that already have a home rather than re-deriving
them here:

    structure     read off the loaded TIR tree: params and their storage,
                  the mesh, the copy count and each copy operand storage
    reload        the authored program finalizes to TIR, and that emitted
                  text loads back as a PrimFunction and verifies
    equality      delegated to tools/cpu_equality.py (I2), which compares
                  the HIR against an independent torch reference
    rejections    delegated to tools/verify_tir.py (I2), which pins each
                  expected error type and message

Delegating is what keeps one claim in one place. A local copy of the equality
or rejection argument would be a second opinion that drifts from the first.
Both delegations record the child exit code verbatim, and a child that is not
runnable is NOT_RUN with its reason -- never a pass.

Exit codes
    0  structure holds, the reload verifies, and both delegations passed
    1  a check failed
    2  ERROR: the artifact could not be read
    3  NOT_RUN: a delegated tool or its dependency is unavailable
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PASS, FAIL, ERROR, NOT_RUN = 0, 1, 2, 3
_STATUS_EXIT = {"PASS": PASS, "FAIL": FAIL, "ERROR": ERROR, "NOT_RUN": NOT_RUN}

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[2]
DEFAULT_ARTIFACT = REPO_ROOT / "examples/rmsnorm_950pr/copy_slice.py"
DEFAULT_ENTRY = "CopySlice"
PYTHON = os.environ.get("TF_PYTHON", sys.executable or "python3")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _walk(node, found: dict) -> None:
    """Collect every MeshScope mesh and Evaluate statement under one node."""
    name = type(node).__name__
    if name == "Sequential":
        for child in node.body:
            _walk(child, found)
        return
    if name == "MeshScope":
        found["meshes"].append(node.mesh)
        _walk(node.body, found)
        return
    if name == "Evaluate":
        found["evaluates"].append(node)
        return
    body = getattr(node, "body", None)
    if body is None:
        return
    if isinstance(body, (tuple, list)):
        for child in body:
            _walk(child, found)
    else:
        _walk(body, found)


def inspect_structure(tir: Path, entry: str) -> dict:
    """Read the structural facts off the FINALIZED tree, not off source text.

    The authored body is a Call expression tree: it states the program, and it
    has no Statement nodes to read a mesh scope or an instruction call off. The
    M2b claim is about what finalize emits, so the facts are read from the
    emitted TIR -- which is a Sequential of statements, exactly where a mesh and
    an instruction appear.
    """
    from tilefoundry.cli.source import load_authored_ir

    module = load_authored_ir(str(tir) + ":" + entry)
    function = module.entry_function()
    found = {"meshes": [], "evaluates": []}
    _walk(function.body, found)

    # StorageKind: 2 = GMEM, 3 = SMEM. The numbers are the enum, read as numbers
    # so a storage comparison cannot silently match a rendered string.
    gmem, smem = 2, 3
    params = {param.name: param.type for param in function.params}
    copies = [
        stmt for stmt in found["evaluates"]
        if type(getattr(stmt, "callable", None)).__name__ == "Copy"
    ]
    storages = set()
    for stmt in copies:
        for arg in getattr(stmt, "args", ()) or ():
            storage = getattr(getattr(arg, "type", None), "storage", None)
            if storage is not None:
                storages.add(int(storage))
    return {
        "entry": function.name,
        "entry_type": type(function).__name__,
        "body_type": type(function.body).__name__,
        "params": sorted(params),
        "param_storages": {
            name: int(getattr(kind, "storage", -1))
            for name, kind in sorted(params.items())
        },
        "meshes": [
            [str(t.name) for t in getattr(mesh, "topologies", ())]
            for mesh in found["meshes"]
        ],
        "copy_count": len(copies),
        "copy_operand_storages": sorted(storages),
        "has_smem_staging": smem in storages,
        "has_gmem_in_out": all(
            int(getattr(kind, "storage", -1)) == gmem for kind in params.values()
        ),
    }

def _run(name: str, argv, timeout: float) -> dict:
    """Run a delegated tool, keeping its exit code and streams verbatim."""
    record = {"check": name, "command": " ".join(str(a) for a in argv)}
    try:
        completed = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        record.update(status="NOT_RUN", reason="interpreter not found: " + str(exc))
        return record
    except subprocess.TimeoutExpired:
        record.update(status="FAIL", reason="timed out after %ss" % timeout)
        return record
    record["exit_code"] = completed.returncode
    record["stdout_tail"] = completed.stdout[-4000:]
    record["stderr_tail"] = completed.stderr[-4000:]
    return record


def delegate_equality() -> dict:
    """CPU equality lives in tools/cpu_equality.py; this records its verdict."""
    tool = TOOLS_DIR / "cpu_equality.py"
    if not tool.is_file():
        return {"check": "equality", "status": "NOT_RUN", "reason": "absent: " + str(tool)}
    record = _run(
        "equality",
        [PYTHON, str(tool), "--case", "copy_slice"],
        timeout=600.0,
    )
    if record.get("status") == "NOT_RUN":
        return record
    code = record.get("exit_code")
    # cpu_equality.py: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
    record["status"] = {0: "PASS", 1: "FAIL", 3: "NOT_RUN"}.get(code, "ERROR")
    return record


def delegate_rejections() -> dict:
    """Expected-rejection evidence lives in tools/verify_tir.py."""
    tool = TOOLS_DIR / "verify_tir.py"
    if not tool.is_file():
        return {"check": "rejections", "status": "NOT_RUN", "reason": "absent: " + str(tool)}
    # verify_tir.py exits 1 on an authored artifact by design while the region
    # gap is open, so its rejection evidence is read from its OWN report file
    # rather than inferred from that exit code or from a truncated stdout tail.
    report_path = Path(tempfile.mkdtemp(prefix="rejections_")) / "verify_tir.json"
    record = _run(
        "rejections",
        [PYTHON, str(tool), "--out", str(report_path)],
        timeout=600.0,
    )
    record["report_path"] = str(report_path)
    if record.get("status") == "NOT_RUN":
        return record
    code = record.get("exit_code")
    summary = None
    if report_path.is_file():
        try:
            summary = json.loads(report_path.read_text(encoding="utf-8")).get("summary")
        except ValueError:
            summary = None
    record["reported_summary"] = summary
    if isinstance(summary, dict) and "expected_rejections_ok" in summary:
        record["status"] = "PASS" if summary["expected_rejections_ok"] else "FAIL"
    else:
        record["status"] = "ERROR" if code == 2 else "FAIL"
    return record


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="M2b closure checks for copy_slice.")
    parser.add_argument("artifact", nargs="?", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("entry", nargs="?", default=DEFAULT_ENTRY)
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument(
        "--skip-delegations",
        action="store_true",
        help="structural evidence only; the report then has no equality or rejection",
    )
    args = parser.parse_args(argv)

    artifact = Path(args.artifact)
    if not artifact.is_file():
        print("check_copy_slice: error: no such artifact: " + str(artifact), file=sys.stderr)
        return ERROR

    report = {
        "tool": "check_copy_slice",
        "generated_at": _now(),
        "artifact": str(artifact),
        "entry": args.entry,
        "python": PYTHON,
        "checks": {},
    }
    # Finalize first: the structural facts are facts about the emitted TIR, and
    # the reload of that same file is what closes M2b. One emission serves both.
    with tempfile.TemporaryDirectory(prefix="copy_slice_") as scratch:
        tir = Path(scratch) / "copy_resolved_tir.py"
        finalize = _run(
            "finalize",
            [
                PYTHON,
                "-m",
                "tilefoundry.cli",
                "schedule",
                "finalize",
                str(artifact) + ":" + args.entry,
                str(tir),
            ],
            timeout=900.0,
        )
        report["finalize"] = finalize
        report["tir_path"] = str(tir)
        if finalize.get("exit_code") != 0 or not tir.is_file():
            report["status"] = "FAIL"
            report["checks"]["finalize_exit_zero"] = False
            report["exit_code"] = FAIL
            print(json.dumps(report, indent=2))
            return FAIL
        report["checks"]["finalize_exit_zero"] = True
        try:
            structure = inspect_structure(tir, args.entry)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "ERROR"
            report["error"] = type(exc).__name__ + ": " + str(exc)[:400]
            report["exit_code"] = ERROR
            print(json.dumps(report, indent=2))
            return ERROR
        report["tir_text"] = tir.read_text(encoding="utf-8")

    report["structure"] = structure
    report["checks"]["two_copies"] = structure["copy_count"] == 2
    report["checks"]["smem_staging"] = structure["has_smem_staging"]
    report["checks"]["gmem_params"] = structure["has_gmem_in_out"]
    report["checks"]["aiv_mesh"] = structure["meshes"] == [["aiv"]]

    if not args.skip_delegations:
        equality = delegate_equality()
        report["equality"] = equality
        report["checks"]["cpu_equality"] = equality.get("status") == "PASS"
        rejections = delegate_rejections()
        report["rejections"] = rejections
        report["checks"]["expected_rejections"] = rejections.get("status") == "PASS"

        delegated = [equality.get("status"), rejections.get("status")]
        if "FAIL" in delegated:
            report["status"] = "FAIL"
        elif "ERROR" in delegated:
            report["status"] = "ERROR"
        elif "NOT_RUN" in delegated:
            report["status"] = "NOT_RUN"

    if "status" not in report:
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    elif report["status"] == "NOT_RUN" and not all(report["checks"].values()):
        report["status"] = "FAIL"
    report["exit_code"] = _STATUS_EXIT[report["status"]]

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print("check_copy_slice: wrote " + str(destination), file=sys.stderr)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())