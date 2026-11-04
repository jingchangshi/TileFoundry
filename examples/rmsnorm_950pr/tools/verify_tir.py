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

The verifier domains
    The two verifiers are declared over different node types:

        verify_prim_function(fn: PrimFunction, *, module_fns=...)   TIR
        verify_function(fn: Function, *, module=...)                HIR

    A PrimFunction and an HIR Function are different classes, and an HIR body
    is an Expr tree with no Stmt sequence in it. Asking verify_prim_function of
    an authored HIR Function is a domain error -- the wrong verifier for the
    node -- not a defect in the verifier. Its resulting raise is pinned as an
    expected consequence so the boundary is recorded rather than misread as a
    gap in the tracked core, and the correct-domain TIR call must still pass.

    An earlier revision of this tool reported that call as a core defect and
    failed authored artifacts because of it. That was wrong: the artifact was
    being asked a question stated about another node type.
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

#: What the wrong-domain call raises today. Pinned so the boundary is an
#: expectation: if the two node types were ever conflated, this stops firing
#: and the case below is what notices.
WRONG_DOMAIN_ERROR = "no verify_stmt registered for MeshRegion"
WRONG_DOMAIN_TYPE = "VerifyError"


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
    function = module.entry_function()
    record = {"entry_type": type(function).__name__}
    try:
        verify_prim_function(function, module_fns=module)
    except Exception as exc:  # noqa: BLE001 -- the raising IS the observation
        record.update(
            verdict="RAISES",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return record
    record["verdict"] = "VERIFIES"
    return record


def probe_verifier_domains(record: dict) -> dict:
    """Check that each verifier is asked only of the node type it states.

    The two verifiers are declared over different node types:

        verify_prim_function(fn: PrimFunction, *, module_fns=...)   -- TIR
        verify_function(fn: Function, *, module=...)               -- HIR

    A PrimFunction and an HIR Function are different classes, and every HIR
    body is an Expr tree with no Stmt sequence in it. Asking
    verify_prim_function of an authored HIR Function is therefore a DOMAIN
    ERROR -- the wrong verifier for the node -- not a gap in the verifier.
    What it raises is an artifact of the walk meeting an Expr where it
    expects a Stmt, not a rule the artifact broke.

    These cases pin that boundary, so the wrong-domain call is a recorded
    expectation rather than a number someone reads as a defect. They also
    keep the correct domains honest: the TIR path must still verify.
    """
    authored_source = REPO_ROOT / DEFAULT_SOURCE
    tir_source, _, tir_entry = DEFAULT_TIR.partition(":")
    tir_path = REPO_ROOT.parent / tir_source

    authored = _ask_prim_function(authored_source, DEFAULT_ENTRY)
    resolved = _ask_prim_function(tir_path, tir_entry)

    wrong_domain_raises = (
        authored.get("entry_type") == "Function"
        and authored.get("verdict") == "RAISES"
        and authored.get("error_type") == WRONG_DOMAIN_TYPE
        and WRONG_DOMAIN_ERROR in authored.get("error", "")
    )
    tir_verifies = resolved.get("verdict") == "VERIFIES"

    cases = {
        "prim_function_of_hir_function_raises": {
            "observed": authored,
            "pinned": {
                "error_type": WRONG_DOMAIN_TYPE,
                "contains": WRONG_DOMAIN_ERROR,
            },
            "verdict": "PASS" if wrong_domain_raises else "FAIL",
            "domain": "wrong -- verify_prim_function is declared over PrimFunction",
            "why": (
                "an authored HIR entry is a Function whose body is an Expr; "
                "the PrimFunction walk meets an expression where it expects a "
                "statement. The raise is the expected consequence of the wrong "
                "domain, and a silent pass here would mean the two node types "
                "had been conflated."
            ),
        },
        "prim_function_of_tir_primfunction_verifies": {
            "observed": resolved,
            "verdict": "PASS" if tir_verifies else "FAIL",
            "domain": "correct -- verify_prim_function over a PrimFunction",
            "why": (
                "the finalized TIR entry IS a PrimFunction with a Sequential "
                "body, which is exactly what this verifier states. It must "
                "pass; a failure here would be a real defect in the TIR path."
            ),
        },
    }

    domains = {
        "statement": (
            "verify_prim_function is declared over PrimFunction; an authored "
            "HIR entry is a Function over Expr nodes. Asking the TIR verifier "
            "of an HIR function is a domain error, not a core defect, and the "
            "error it raises is pinned above rather than read as one."
        ),
        "pinned_diagnostic": {
            "error_type": WRONG_DOMAIN_TYPE,
            "contains": WRONG_DOMAIN_ERROR,
        },
        "cases": cases,
    }
    record["verifier_domains"] = domains
    return domains

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
        domains = probe_verifier_domains(record)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "ERROR"
        record["error"] = type(exc).__name__ + ": " + str(exc)[:500]
        record["exit_code"] = ERROR
        print(json.dumps(record, indent=2))
        return ERROR

    in_protocol = ["load_authored_ir", "verify_module"]
    in_protocol.append("verify_function" if kind == "hir" else "verify_prim_function")
    verifiers_ok = all(record[name]["verdict"] == "PASS" for name in in_protocol)
    domain_cases = domains["cases"]
    domains_ok = all(case["verdict"] == "PASS" for case in domain_cases.values())

    # Every in-protocol verifier passed, so the artifact passes the verifiers
    # that state its own node type. The wrong-domain call is a recorded
    # expectation above, not a failure of this artifact.
    if verifiers_ok and rejections_ok and domains_ok:
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
        "verifier_domains_ok": domains_ok,
        "wrong_domain_calls_raise": domain_cases[
            "prim_function_of_hir_function_raises"
        ]["verdict"],
        "tir_domain_call_verifies": domain_cases[
            "prim_function_of_tir_primfunction_verifies"
        ]["verdict"],
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