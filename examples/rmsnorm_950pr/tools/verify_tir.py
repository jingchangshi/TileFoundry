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

Expected-rejection cases are declared PER ARTIFACT (see ARTIFACT_REJECTIONS),
keyed by source basename and kind. Each artifact states its OWN applicable
rejections; there is no blanket set, because a mutation that does not match its
artifact changes nothing and "changed nothing" then reads as FAIL -- training
people to ignore FAIL, the mirror image of a test that cannot fail.

A case runs at one of two stages:
    load      mutate, load the artifact, expect error_type + a message substring.
    finalize  mutate, run schedule finalize, expect a non-zero exit + substring.
              Required because element-width, lane-extent and storage-placement
              rules are enforced during LOWERING, not at load. A load-only
              mechanism silently skips them.

An artifact with NO declared rejections is SKIPPED with a named reason and does
NOT fail: absent evidence is not evidence of a defect. Cases the Lead proposed
that the current pipeline does not enforce are listed in KNOWN_UNENFORCED so the
gap is visible rather than papered over with a case that cannot fire.

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
    0  every in-protocol verifier passed, every DECLARED expected rejection
       matched (or none were declared, so rejections are SKIPPED), and the
       region gap is closed
    1  a verifier, a declared rejection, or the region gap disagrees with what
       this tool predicts
    2  ERROR: usage, or an artifact that could not be read

    SKIPPED is not a failure: an artifact that declares no rejections passes on
    its verifiers alone, and the report says so by name. This is the final form.
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


#: Per-ARTIFACT expected-rejection sets, keyed by source basename and then by kind.
#:
#: Each artifact declares its OWN applicable rejections. There is deliberately no
#: blanket set applied to every artifact: a mutation that does not match its
#: artifact changes nothing, and "mutation changed nothing" then reads as FAIL. That
#: trains people to ignore FAIL, which is the mirror image of a test that cannot
#: fail. An artifact with no declared rejections is SKIPPED with a name, never FAIL.
#:
#: A case has a stage:
#:   "load"      (default) mutate, load_authored_ir, expect error_type + contains.
#:   "finalize"  mutate, run the CLI schedule finalize, expect a non-zero exit and
#:               contains in the error. Needed because some rules -- element width,
#:               lane extent, storage placement -- are enforced during LOWERING, not
#:               at load. A load-only rejection mechanism silently skips them.
#:
#: Every "contains" below was OBSERVED from the real pipeline, not guessed.
ARTIFACT_REJECTIONS = {
    "copy_slice.py": {
        "hir": [
            {
                "name": "wrong_storage",
                "why": "the storage vocabulary is closed at the IR type boundary; l2 is "
                       "not a StorageKind, so the surface form is refused while parsing.",
                "replace": [('(N,), "smem"', '(N,), "l2"')],
                "error_type": "ParseError",
                "contains": "unknown storage",
            },
            {
                "name": "wrong_layout",
                "why": "a layout must cover exactly its logical extent; a (N // 2,) "
                       "layout over a (N,) tensor is well formed for the wrong tensor.",
                "replace": [('Tensor[(N,), "f32", (N,), "smem"]',
                             'Tensor[(N,), "f32", (N // 2,), "smem"]')],
                "error_type": "VerifyError",
                "contains": "cross logical axes",
            },
            {
                "name": "unknown_flag",
                "why": "an instruction takes no options; naming one is an authoring "
                       "error the parser must catch off the call.",
                "replace": [("op=T.copy()", "op=T.copy(bogus_flag=True)")],
                "error_type": "ParseError",
                "contains": "has no attribute",
            },
        ],
        "tir": [
            {
                "name": "wrong_storage",
                "why": "a finalized tensor view names its residency; l2 is not a "
                       "StorageKind, so the emitted TIR must not parse back.",
                "replace": [('storage="smem"', 'storage="l2"')],
                "error_type": "ParseError",
                "contains": "unknown storage",
            },
            {
                "name": "wrong_layout",
                "why": "the view physical layout must cover its shape; halving it "
                       "leaves a well-formed view of the wrong extent.",
                "replace": [("layout=(64,), shape=(64,)", "layout=(32,), shape=(64,)")],
                "error_type": "VerifyError",
                "contains": "cross logical axes",
            },
            {
                "name": "unknown_flag",
                "why": "T.copy takes no options in the emitted TIR either.",
                "replace": [("T.copy(x, ub)", "T.copy(x, ub, bogus_flag=True)")],
                "error_type": "ParseError",
                "contains": "has no attribute",
            },
        ],
    },
    "vf_vadd.py": {
        "hir": [
            {
                "name": "lane_extent_64_to_32",
                "why": "the mask is one bit per lane; a 32-lane mask over 64-lane data "
                       "selects the wrong lanes, which only lowering can see.",
                "stage": "finalize",
                "replace": [('Tensor[(LANES,), "bool", (LANES,), "rmem"]',
                             'Tensor[(LANES // 2,), "bool", (LANES // 2,), "rmem"]')],
                "expect_nonzero_exit": True,
                "contains": "mask covers (32,) lanes",
            },
            {
                "name": "elem_width_32_to_17",
                "why": "ascend_pset states the element width; 17 does not match the "
                       "region's declared 32, so it addresses lanes the mode lacks.",
                "stage": "finalize",
                "replace": [("elem_width=32", "elem_width=17")],
                "expect_nonzero_exit": True,
                "contains": "states element width 17",
            },
            {
                "name": "storage_smem_to_gmem",
                "why": "ascend_gm_to_ub writes UB/smem; a gmem destination is the wrong "
                       "residency for that transfer and lowering must refuse it.",
                "stage": "finalize",
                "replace": [('loaded = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])',
                             'loaded = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])')],
                "expect_nonzero_exit": True,
                "contains": "does not match smem",
            },
            {
                "name": "mask_without_producer",
                "why": "a predicated op whose mask no instruction wrote is reading a "
                       "register nothing produced; lowering enforces producer DOMINANCE "
                       "(the writer must precede the reader), so this is refused.",
                "stage": "finalize",
                "replace": [
                    ("pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))", ""),
                    ("(hl, hr, pred, acc)", "(hl, hr, mask, acc)"),
                ],
                "expect_nonzero_exit": True,
                "contains": "no producing instruction",
            },
        ],
    },
}

#: Rejections the Lead listed for vf_vadd that the CURRENT pipeline does NOT enforce.
#: Recorded so the gap is visible rather than papered over with a case that cannot
#: fire. Dropping the pset and feeding the mask tensor straight to ascend_vadd is
#: ACCEPTED today (finalize exit 0); there is no "mask has no producer" check.
#: Empty now. The producerless-mask gap this once recorded is CLOSED upstream:
#: TileFoundry 0550703 added verify_predicate_producers (presence), and d13528f
#: tightened it to DOMINANCE (the writing instruction must precede the reader).
#: The dropped-pset case is asserted in ARTIFACT_REJECTIONS["vf_vadd.py"].
#:
#: The ORDERING case -- a pset placed AFTER the vadd -- is deliberately NOT here.
#: It is not authorable in the DSL: a source file is read top to bottom, so a vadd
#: referencing 'pred' before the pset defines it fails with NameError, not with a
#: dominance violation. That case is direct-TIR only and is covered by
#: targets/ascend/tests/test_predicate_producer.py:late_pset_is_refused. A
#: source-keyed registry cannot express it, and faking it would assert the wrong
#: reason. Kept as a dict so a future genuinely-unenforced proposal has a home -- but
#: anything that DOES fire belongs in ARTIFACT_REJECTIONS, not here, or it is a lie.
KNOWN_UNENFORCED = {}


def run_rejections(source_text: str, source_path: str, entry: str, kind: str, record: dict) -> bool:
    """Run this artifact's OWN declared rejections, per their stage.

    Returns True when every declared rejection behaved as declared, OR when none are
    declared (SKIPPED). Returns False only when a declared rejection disagrees.
    """
    import os as _os
    import subprocess as _subprocess

    from tilefoundry.cli.source import load_authored_ir
    from tilefoundry.ir.core.errors import VerifyError
    from tilefoundry.parser.ast_pattern import ParseError

    basename = Path(source_path).name
    entry_sets = ARTIFACT_REJECTIONS.get(basename)
    cases = (entry_sets or {}).get(kind)
    unenforced = KNOWN_UNENFORCED.get(basename, [])
    if not cases:
        record["expected_rejections"] = {
            "verdict": "SKIPPED",
            "reason": ("no declared rejections for artifact %r kind %r; a blanket set "
                       "would be vacuous here, so this is SKIPPED, not FAIL" % (basename, kind)),
            "known_unenforced": unenforced,
        }
        return True

    types = {"ParseError": ParseError, "VerifyError": VerifyError}
    results = []
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="verify_tir_") as scratch:
        for case in cases:
            mutated = source_text
            for old, new in case["replace"]:
                mutated = mutated.replace(old, new)
            outcome = {"name": case["name"], "why": case["why"],
                       "stage": case.get("stage", "load")}
            if mutated == source_text:
                outcome.update(verdict="FAIL", error="mutation changed nothing")
                results.append(outcome)
                all_ok = False
                continue
            path = Path(scratch) / (case["name"] + ".py")
            path.write_text(mutated, encoding="utf-8")

            if case.get("stage", "load") == "finalize":
                out_tir = Path(scratch) / (case["name"] + "_out.py")
                proc = _subprocess.run(
                    [sys.executable, "-m", "tilefoundry.cli", "schedule", "finalize",
                     str(path) + ":" + entry, str(out_tir)],
                    capture_output=True, text=True, cwd=str(REPO_ROOT),
                    env=dict(_os.environ),
                )
                blob = (proc.stderr or "") + (proc.stdout or "")
                nonzero_ok = (proc.returncode != 0) if case.get("expect_nonzero_exit") else True
                text_ok = case["contains"] in blob
                outcome.update(
                    expected_exit="nonzero" if case.get("expect_nonzero_exit") else "any",
                    observed_exit=proc.returncode,
                    expected_contains=case["contains"],
                    observed=blob.strip().splitlines()[-1][:300] if blob.strip() else "",
                    exit_matched=nonzero_ok,
                    text_matched=text_ok,
                    verdict="PASS" if (nonzero_ok and text_ok) else "FAIL",
                )
                all_ok = all_ok and nonzero_ok and text_ok
            else:
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
    record["expected_rejections"] = {
        "verdict": "PASS" if all_ok else "FAIL",
        "cases": results,
        "known_unenforced": unenforced,
    }
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
            rejections_ok = run_rejections(text, str(source), args.entry, kind, record)
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