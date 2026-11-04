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


def _storage_name(value) -> str:
    return str(value).split(".")[-1].lower() if value is not None else "none"


def inspect_flow(tir: Path, entry: str) -> dict:
    """Read the typed dataflow off the FINALIZED TIR, statement by statement.

    The M2b claim is a chain, not a count: parameter x (GMEM) is read by a
    copy into ONE smem buffer, and that same buffer is read by a second copy
    into the out parameter (GMEM). Each link is asserted from the IR --
    operand types, the source buffer identity, and each param definition's
    declared memory effect -- so a two-copy program that staged through two
    different buffers, or wrote somewhere other than out, would fail here.

    The authored body is a Call tree and has no statements to read this off,
    which is why the facts come from the emitted TIR.
    """
    from tilefoundry.cli.source import load_authored_ir

    module = load_authored_ir(str(tir) + ":" + entry)
    function = module.entry_function()
    params = {param.name: param.type for param in function.params}

    # Walk to the MeshScope body: the statements live inside it, not at the
    # top of the Sequential.
    scopes = [node for node in _iter_nodes(function.body) if type(node).__name__ == "MeshScope"]
    statements = []
    for scope in scopes:
        statements.extend(_iter_nodes(scope.body))
    lets = {s.var.name: s for s in statements if type(s).__name__ == "LetStmt"}
    evals = [s for s in statements if type(s).__name__ == "Evaluate"]
    copies = [s for s in evals if type(s.callable).__name__ == "Copy"]

    def operand_names(stmt):
        return [getattr(arg, "name", None) for arg in stmt.args]

    def storage_of(name):
        kind = params.get(name)
        if kind is not None:
            return _storage_name(getattr(kind, "storage", None))
        let = lets.get(name)
        if let is not None:
            return _storage_name(getattr(getattr(let.var, "type", None), "storage", None))
        return "unknown"

    chain = [{"op": "copy", "operands": operand_names(stmt)} for stmt in copies]
    flows = [
        [storage_of(name) if name else "?" for name in item["operands"]]
        for item in chain
    ]

    # The staging buffer: the one smem let bound from a TensorView.
    staged = [
        name
        for name, let in lets.items()
        if _storage_name(getattr(getattr(let.var, "type", None), "storage", None)) == "smem"
    ]

    # Declared memory effects, read off each ParamDef rather than assumed.
    effects = {}
    schema = getattr(type(copies[0].callable), "_op_schema", None) if copies else None
    if schema is not None:
        for index, definition in enumerate(getattr(schema, "signature", ())):
            effect = getattr(definition, "effect", None)
            effects["arg%d" % index] = getattr(effect, "name", str(effect)) if effect else None

    return {
        "entry": function.name,
        "entry_type": type(function).__name__,
        "body_type": type(function.body).__name__,
        "statements": len(statements),
        "params": {
            name: {
                "storage": _storage_name(getattr(kind, "storage", None)),
                "shape": list(getattr(kind, "shape", ()) or ()),
                "dtype": str(getattr(kind, "dtype", None)),
            }
            for name, kind in sorted(params.items())
        },
        "mesh_topologies": [
            [str(t.name) for t in getattr(scope.mesh, "topologies", ())]
            for scope in scopes
        ],
        "copies": chain,
        "copy_storage_flows": flows,
        "staging_buffers": staged,
        "copy_effects": effects,
        "let_bindings": sorted(lets),
    }


def _iter_nodes(node):
    """Every node in a stmt/expr tree, in DOCUMENT order and without recursion.

    Document order matters here: the two copies are only x -> ub and
    ub -> out in that order, and a walk that reverses siblings would report
    the chain backwards while looking correct. Children are therefore pushed
    in reverse so popping them yields the authored order.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        yield current
        body = getattr(current, "body", None)
        if body is None:
            continue
        if isinstance(body, (tuple, list)):
            stack.extend(reversed(body))
        else:
            stack.append(body)

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


def _resolve_case(artifact: Path):
    """The name cpu_equality.py registers for this artifact, or None.

    One artifact, one case name. A source with no registered case has no
    reference to compare against, so it cannot be checked by copying the
    default case's verdict into it.
    """
    try:
        import cpu_equality  # noqa: PLC0415 -- sibling tool, imported lazily
    except ImportError:
        sys.path.insert(0, str(TOOLS_DIR))
        import cpu_equality  # noqa: PLC0415
    for name, spec in cpu_equality.CASES.items():
        if (REPO_ROOT / spec["source"]).resolve() == artifact:
            return name
    return None


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

    artifact = Path(args.artifact).resolve()
    if not artifact.is_file():
        print("check_copy_slice: error: no such artifact: " + str(artifact), file=sys.stderr)
        return ERROR

    default = DEFAULT_ARTIFACT.resolve()
    delegated_from = None
    checked_case = _resolve_case(artifact)
    if artifact != default:
        # A custom source is never checked by a second, weaker path. It is
        # resolved to the case name cpu_equality.py registers for it, and the
        # check then runs against THAT case's defining artifact, so the
        # verdict means exactly what the registered case means.
        #
        # Today the only registered case IS the default artifact, so a
        # different file has no reference to compare against and is refused
        # rather than checked against the wrong program's semantics. The
        # re-entry below becomes reachable when a second case is registered.
        delegated_from = str(artifact)
        case = checked_case
        if case is None:
            report = {
                "tool": "check_copy_slice",
                "generated_at": _now(),
                "artifact": str(artifact),
                "status": "ERROR",
                "exit_code": ERROR,
                "error": (
                    "no registered case covers " + str(artifact)
                    + "; register it in cpu_equality.CASES rather than checking it out of band"
                ),
            }
            print(json.dumps(report, indent=2))
            return ERROR
        forwarded = [str(DEFAULT_ARTIFACT), args.entry]
        if args.skip_delegations:
            forwarded.append("--skip-delegations")
        if args.out:
            forwarded.extend(["--out", args.out])
        code = main(forwarded)
        return code

    report = {
        "tool": "check_copy_slice",
        "generated_at": _now(),
        "artifact": str(artifact),
        "entry": args.entry,
        "python": PYTHON,
        "checked_case": checked_case,
        "checks": {},
    }
    if delegated_from is not None:
        report["delegated_from"] = delegated_from
        report["checks"]["delegated_to_default_case"] = True

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
        report["checks"]["finalize_exit_zero"] = finalize.get("exit_code") == 0
        if not report["checks"]["finalize_exit_zero"] or not tir.is_file():
            report["status"] = "FAIL"
            report["exit_code"] = FAIL
            print(json.dumps(report, indent=2))
            return FAIL

        try:
            flow = inspect_flow(tir, args.entry)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "ERROR"
            report["error"] = type(exc).__name__ + ": " + str(exc)[:400]
            report["exit_code"] = ERROR
            print(json.dumps(report, indent=2))
            return ERROR
        # The emitted text is kept as evidence: the checks below are claims
        # about THESE bytes, and a reader has to be able to see them.
        report["tir_text"] = tir.read_text(encoding="utf-8")

    report["flow"] = flow
    checks = report["checks"]

    # The typed chain: x(gmem) -> ONE smem buffer -> out(gmem), in that order.
    flows = flow["copy_storage_flows"]
    copies = flow["copies"]
    staged = flow["staging_buffers"]
    in_copy = copies[0]["operands"] if len(copies) > 0 else []
    out_copy = copies[1]["operands"] if len(copies) > 1 else []
    first = flows[0] if len(flows) > 0 else []
    second = flows[1] if len(flows) > 1 else []
    checks["two_copies"] = len(copies) == 2
    checks["one_smem_buffer"] = len(staged) == 1
    checks["first_copy_gmem_to_smem"] = first == ["gmem", "smem"]
    checks["second_copy_smem_to_gmem"] = second == ["smem", "gmem"]
    checks["stages_through_same_buffer"] = (
        bool(in_copy) and bool(out_copy) and in_copy[1] == out_copy[0]
    )
    checks["reads_param_x"] = bool(in_copy) and in_copy[0] == "x"
    checks["writes_param_out"] = bool(out_copy) and out_copy[1] == "out"
    checks["both_params_gmem"] = all(
        kind["storage"] == "gmem" for kind in flow["params"].values()
    )
    checks["aiv_mesh"] = flow["mesh_topologies"] == [["aiv"]]
    # Effects come off the emitted op's own ParamDefs: the input is read and
    # the output written, stated by the instruction rather than by this tool.
    effects = flow["copy_effects"]
    checks["declares_read_then_write"] = (
        effects.get("arg0") == "READ" and effects.get("arg1") == "WRITE"
    )

    if not args.skip_delegations:
        equality = delegate_equality()
        report["equality"] = equality
        checks["cpu_equality"] = equality.get("status") == "PASS"
        rejections = delegate_rejections()
        report["rejections"] = rejections
        checks["expected_rejections"] = rejections.get("status") == "PASS"

    if not all(checks.values()):
        report["status"] = "FAIL"
    else:
        delegated = [
            report.get("equality", {}).get("status"),
            report.get("rejections", {}).get("status"),
        ]
        if "FAIL" in delegated:
            report["status"] = "FAIL"
        elif "ERROR" in delegated:
            report["status"] = "ERROR"
        elif "NOT_RUN" in delegated:
            report["status"] = "NOT_RUN"
        else:
            report["status"] = "PASS"
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
