"""Verify an artifact through the published TF verifiers, recording each verdict.

The verifiers an artifact is asked of depend on what it is, and the kind is
read off the loaded entry rather than assumed from its filename:

    hir   entry is a Function whose body is an HIR expression (MeshRegion)
          verify_module, verify_function
    tir   entry is a PrimFunction whose body is a Sequential of statements
          verify_module, verify_prim_function

verify_prim_function states the PrimFunction rule list and is not asked of an
authored HIR Function; verify_function states the HIR Function rules and is not
asked of a PrimFunction. The verifier that does not apply is recorded
NOT_APPLICABLE with that reason, which is a statement about the tool, not a
pass.

Expected-rejection cases mutate the artifact and state, per case, the error
type it must raise and a substring of its message. A rejection for the wrong
reason FAILs: it is not the evidence the case was written to collect, and an
accepted bad program is a defect.

The region gap
    verify_prim_function walks Statement nodes through verify_stmt_registry,
    and no verify_stmt is registered for MeshRegion. An authored HIR body IS a
    MeshRegion, so asking verify_prim_function of one raises instead of
    verifying. That is a defect in the tracked core, not in the artifact, and
    this tool reports it as the FAIL it is: an authored HIR artifact does not
    pass every verifier today and its exit code is 1 until the core is fixed.

    Two pinned cases state the boundary, so the fix cannot land silently:
      MeshRegion_diagnostic      must raise VerifyError with the exact text
      MeshRegion_region_verifies must pass once the registry can walk it
    Pre-fix the first holds and the second does not; post-fix the opposite.
    The report names the epoch it observed, and only the post-fix epoch lets
    an authored HIR artifact exit 0.

Exit codes
    0  every in-protocol verifier passed, every expected rejection matched,
       and the region gap is closed
    1  a verifier, a rejection, or the region gap disagrees with what this
       tool predicts
    2  ERROR: usage, or an artifact that could not be read
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PASS, FAIL, ERROR = 0, 1, 2

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SOURCE = "examples/rmsnorm_950pr/copy_slice.py"
DEFAULT_ENTRY = "CopySlice"

#: The finalized TIR of the same program, kept for contrast: a PrimFunction
#: body the statement registry can already walk. Absent means that contrast is
#: NOT_RUN, never that it passed.
DEFAULT_TIR = (
    "TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/r1/copy_resolved_tir.py:CopySlice"
)

#: The pinned diagnostic. When the tracked core starts walking a MeshRegion
#: body this text stops appearing, and MeshRegion_diagnostic is what notices.
REGION_GAP_ERROR = "no verify_stmt registered for MeshRegion"
REGION_GAP_TYPE = "VerifyError"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _attempt(record: dict, name: str, call) -> bool:
    """Run one verifier and record its exact verdict and error type."""
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 -- the verdict is the evidence
        record[name] = {
            "verdict": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
        }
        return False
    record[name] = {"verdict": "PASS", "result_type": type(result).__name__}
    return True


def verify_artifact(source: str, entry: str, record: dict) -> str:
    """Load one selection and ask the verifiers that apply to its kind.

    Returns the artifact kind ("hir" or "tir"), or "" when loading failed.
    """
    from tilefoundry.cli.source import load_authored_ir
    from tilefoundry.visitor_registry.verify import (
        verify_function,
        verify_module,
        verify_prim_function,
    )

    state = {"module": None, "function": None}

    def load():
        module = load_authored_ir(source + ":" + entry)
        state["module"] = module
        state["function"] = module.entry_function()
        return module

    record["artifact"] = {"source": str(source), "entry": entry}
    if not _attempt(record, "load_authored_ir", load):
        record["skipped"] = ["verify_module", "verify_function", "verify_prim_function"]
        return ""

    module = state["module"]
    function = state["function"]
    kind = "tir" if type(function).__name__ == "PrimFunction" else "hir"
    record["kind"] = kind
    record["entry_function"] = {
        "name": function.name,
        "type": type(function).__name__,
        "body_type": type(getattr(function, "body", None)).__name__,
    }

    _attempt(record, "verify_module", lambda: verify_module(module))
    if kind == "hir":
        _attempt(record, "verify_function", lambda: verify_function(function))
        record["verify_prim_function"] = {
            "verdict": "NOT_APPLICABLE",
            "reason": (
                "entry is an HIR Function; verify_prim_function states the "
                "PrimFunction rules, and the region gap is reported separately"
            ),
        }
    else:
        _attempt(
            record,
            "verify_prim_function",
            lambda: verify_prim_function(function, module_fns=module),
        )
        record["verify_function"] = {
            "verdict": "NOT_APPLICABLE",
            "reason": "entry is a PrimFunction; verify_function states HIR Function rules",
        }
    return kind


#: One rejection set per artifact kind. Each case states the exact mutation,
#: the error type it must raise, and a substring its message must carry; "why"
#: is the argument for that pairing.
EXPECTED_REJECTIONS = {
    "hir": [
        {
            "name": "wrong_storage",
            "why": (
                "the storage vocabulary is closed at the IR type boundary; l2 is "
                "not a StorageKind, so the surface form is refused while parsing, "
                "before any type exists to verify."
            ),
            "replace": [("(N,), \"smem\"", "(N,), \"l2\"")],
            "error_type": "ParseError",
            "contains": "unknown storage",
        },
        {
            "name": "wrong_layout",
            "why": (
                "a layout must cover exactly its logical extent; a (N // 2,) "
                "layout over a (N,) tensor is well formed for the wrong tensor, "
                "so the verifier must refuse it, not the parser."
            ),
            "replace": [
                (
                    "Tensor[(N,), \"f32\", (N,), \"smem\"]",
                    "Tensor[(N,), \"f32\", (N // 2,), \"smem\"]",
                )
            ],
            "error_type": "VerifyError",
            "contains": "cross logical axes",
        },
        {
            "name": "unknown_flag",
            "why": (
                "an instruction takes no options; naming one is an authoring "
                "error the parser must catch off the call, not something that "
                "can be silently ignored."
            ),
            "replace": [("op=T.copy()", "op=T.copy(bogus_flag=True)")],
            "error_type": "ParseError",
            "contains": "has no attribute",
        },
    ],
    "tir": [
        {
            "name": "wrong_storage",
            "why": (
                "a finalized tensor view names its residency; l2 is not a "
                "StorageKind, so the emitted TIR must not parse back."
            ),
            "replace": [("storage=\"smem\"", "storage=\"l2\"")],
            "error_type": "ParseError",
            "contains": "unknown storage",
        },
        {
            "name": "wrong_layout",
            "why": (
                "the view physical layout must cover its shape; halving it "
                "leaves a well-formed view of the wrong extent, which only a "
                "verifier can catch."
            ),
            "replace": [("layout=(64,), shape=(64,)", "layout=(32,), shape=(64,)")],
            "error_type": "VerifyError",
            "contains": "cross logical axes",
        },
        {
            "name": "unknown_flag",
            "why": (
                "T.copy takes no options in the emitted TIR either; naming one "
                "must fail rather than be dropped."
            ),
            "replace": [("T.copy(x, ub)", "T.copy(x, ub, bogus_flag=True)")],
            "error_type": "ParseError",
            "contains": "has no attribute",
        },
    ],
}


def run_rejections(source_text: str, entry: str, kind: str, record: dict) -> bool:
    """Every expected rejection raises the stated type carrying the stated text."""
    from tilefoundry.cli.source import load_authored_ir
    from tilefoundry.ir.core.errors import VerifyError
    from tilefoundry.parser.ast_pattern import ParseError

    types = {"ParseError": ParseError, "VerifyError": VerifyError}
    cases = EXPECTED_REJECTIONS.get(kind)
    if cases is None:
        record["expected_rejections"] = {"verdict": "NOT_RUN", "reason": "no set for " + kind}
        return True

    results = []
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="verify_tir_") as scratch:
        for case in cases:
            mutated = source_text
            for old, new in case["replace"]:
                mutated = mutated.replace(old, new)
            outcome = {"name": case["name"], "why": case["why"]}
            if mutated == source_text:
                outcome.update(verdict="FAIL", error="mutation changed nothing")
                results.append(outcome)
                all_ok = False
                continue
            path = Path(scratch) / (case["name"] + ".py")
            path.write_text(mutated, encoding="utf-8")
            try:
                load_authored_ir(str(path) + ":" + entry)
            except Exception as exc:  # noqa: BLE001 -- the rejection is the point
                expected = types[case["error_type"]]
                type_matched = isinstance(exc, expected)
                text_matched = case["contains"] in str(exc)
                matched = type_matched and text_matched
                outcome.update(
                    expected_type=case["error_type"],
                    expected_contains=case["contains"],
                    observed_type=type(exc).__name__,
                    observed=str(exc)[:300],
                    type_matched=type_matched,
                    text_matched=text_matched,
                    verdict="PASS" if matched else "FAIL",
                )
                all_ok = all_ok and matched
            else:
                outcome.update(
                    expected_type=case["error_type"],
                    verdict="FAIL",
                    error="artifact was accepted; the predicted rejection never fired",
                )
                all_ok = False
            results.append(outcome)
    record["expected_rejections"] = results
    return all_ok


def _ask_prim_function(path: Path, entry: str) -> dict:
    """Ask verify_prim_function of one artifact and keep what it answered."""
    from tilefoundry.cli.source import load_authored_ir
    from tilefoundry.visitor_registry.verify import verify_prim_function

    if not path.is_file():
        return {"verdict": "NOT_RUN", "reason": "artifact absent: " + str(path)}
    try:
        module = load_authored_ir(str(path) + ":" + entry)
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "LOAD_FAILED", "error": str(exc)[:300]}
    try:
        verify_prim_function(module.entry_function(), module_fns=module)
    except Exception as exc:  # noqa: BLE001 -- the raising IS the observation
        return {
            "verdict": "RAISES",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }
    return {"verdict": "VERIFIES"}


def probe_region_gap(record: dict) -> dict:
    """Pin the region gap with cases that straddle the fix.

    Pre-fix: asking verify_prim_function of an authored HIR body raises the
    pinned diagnostic; the finalized TIR of the same program verifies.
    Post-fix: the authored body verifies too. Naming both cases means the
    transition is observed rather than assumed, and a half-fix that changes the
    message without making the walk work matches neither.
    """
    authored_source = REPO_ROOT / DEFAULT_SOURCE
    tir_source, _, tir_entry = DEFAULT_TIR.partition(":")
    tir_path = REPO_ROOT.parent / tir_source

    authored = _ask_prim_function(authored_source, DEFAULT_ENTRY)
    resolved = _ask_prim_function(tir_path, tir_entry)

    pre_fix = (
        authored.get("verdict") == "RAISES"
        and authored.get("error_type") == REGION_GAP_TYPE
        and REGION_GAP_ERROR in authored.get("error", "")
    )
    post_fix = authored.get("verdict") == "VERIFIES"

    cases = {
        "MeshRegion_diagnostic": {
            "observed": authored,
            "pinned": {
                "error_type": REGION_GAP_TYPE,
                "contains": REGION_GAP_ERROR,
            },
            "verdict": "PASS" if pre_fix else "FAIL",
            "why": (
                "the walk must raise the pinned diagnostic while the statement "
                "registry cannot walk a MeshRegion body; a different error, or "
                "no error, is a different situation than the one pinned."
            ),
        },
        "MeshRegion_region_verifies": {
            "observed": authored,
            "verdict": "PASS" if post_fix else "NOT_RUN",
            "why": (
                "once the registry walks a MeshRegion body this must verify; it "
                "is NOT_RUN before the fix and cannot be mistaken for a pass."
            ),
        },
        "finalized_tir_contrast": {
            "observed": resolved,
            "verdict": "PASS" if resolved.get("verdict") == "VERIFIES" else "FAIL",
            "why": (
                "the same program finalized to a Sequential body does verify, "
                "which is what isolates the gap to the region node rather than "
                "to the program."
            ),
        },
    }

    if pre_fix:
        epoch = "GAP"
    elif post_fix:
        epoch = "CLOSED"
    else:
        epoch = "INCONCLUSIVE"
    region = {
        "statement": (
            "verify_prim_function dispatches Statement nodes through "
            "verify_stmt_registry, and none is registered for MeshRegion. An "
            "authored HIR body IS a MeshRegion, so the walk raises instead of "
            "verifying; a finalized TIR body is a Sequential of primitive "
            "statements, which does verify."
        ),
        "pinned_diagnostic": {
            "error_type": REGION_GAP_TYPE,
            "contains": REGION_GAP_ERROR,
        },
        "epoch": epoch,
        "cases": cases,
    }
    record["region_gap"] = region
    return region


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify an artifact and its expected rejections."
    )
    parser.add_argument("--source", default=str(REPO_ROOT / DEFAULT_SOURCE))
    parser.add_argument("--entry", default=DEFAULT_ENTRY)
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument(
        "--skip-rejections",
        action="store_true",
        help="verify only; the report then carries no rejection evidence",
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_file():
        print("verify_tir: error: no such artifact: " + str(source), file=sys.stderr)
        return ERROR
    record = {
        "tool": "verify_tir",
        "generated_at": _now(),
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
    }
    try:
        kind = verify_artifact(str(source), args.entry, record)
        rejections_ok = True
        if kind and not args.skip_rejections:
            text = source.read_text(encoding="utf-8")
            rejections_ok = run_rejections(text, args.entry, kind, record)
        region = probe_region_gap(record)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "ERROR"
        record["error"] = type(exc).__name__ + ": " + str(exc)[:500]
        record["exit_code"] = ERROR
        print(json.dumps(record, indent=2))
        return ERROR

    in_protocol = ["load_authored_ir", "verify_module"]
    in_protocol.append("verify_function" if kind == "hir" else "verify_prim_function")
    verifiers_ok = all(record[name]["verdict"] == "PASS" for name in in_protocol)
    gap_cases = region["cases"]
    gap_ok = (
        gap_cases["MeshRegion_diagnostic"]["verdict"] == "PASS"
        and gap_cases["finalized_tir_contrast"]["verdict"] == "PASS"
    )

    # An authored HIR artifact does not pass every verifier today: the core
    # cannot walk its region node. That stays FAIL with its cause named until
    # the registry is fixed -- not a pass with a footnote.
    blocked = kind == "hir" and region["epoch"] == "GAP"
    if blocked:
        status, exit_code = "FAIL", FAIL
    elif verifiers_ok and rejections_ok and gap_ok and region["epoch"] != "INCONCLUSIVE":
        status, exit_code = "PASS", PASS
    else:
        status, exit_code = "FAIL", FAIL

    record["summary"] = {
        "kind": kind,
        "in_protocol_verifiers": {name: record[name]["verdict"] for name in in_protocol},
        "in_protocol_ok": verifiers_ok,
        "not_applicable": {
            name: record[name]["verdict"]
            for name in ("verify_function", "verify_prim_function")
            if record.get(name, {}).get("verdict") == "NOT_APPLICABLE"
        },
        "expected_rejections_ok": rejections_ok,
        "region_gap_epoch": region["epoch"],
        "blocked_by_region_gap": blocked,
    }
    record["status"] = status
    record["exit_code"] = exit_code
    text = json.dumps(record, indent=2)
    print(text)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print("verify_tir: wrote " + str(destination), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
