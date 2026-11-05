"""task-80: predicate provenance independence + executable POC dependency gate.

Item 1: a WIRED predicate passes iff required (bounds/domain) and actual (materialised TIR)
have INDEPENDENT provenance; same source is refused even if the sets differ; independent and
EQUAL is a PASS. Missing provenance is refused.
Item 2: an executable POC gate requiring a source-proven load->VF and VF->store RAW each, and
zero UNKNOWN within the supported (GMEM/SMEM) domain. It passes on a synthetic supported case;
on the real vf_vadd carrier it records the current NAMED gap (root/alias lineage does not yet
unify SMEM across region boundaries, so no source-proven load->VF RAW exists).

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import functools
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
VF = os.path.join(ROOT, "examples", "rmsnorm_950pr", "vf_vadd.py")


def _refusal(call):
    from tilefoundry_ascend.schedule.ledger import LedgerError

    try:
        call()
    except LedgerError as error:
        return str(error).split(":", 1)[0]
    return None


def _problem_with_predicate(predicate):
    return types.SimpleNamespace(policy={"predicate": predicate}, accesses=[], dependencies=[])


TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


@_test
def test_same_provenance_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    predicate = {
        "status": "WIRED",
        "required_active": frozenset({0, 1, 2}),
        "actual_active": frozenset({0, 1, 2}),
        "required_provenance": {"kind": "tir", "ref": "actual"},
        "actual_provenance": {"kind": "tir", "ref": "actual"},
    }
    assert _refusal(lambda: verify_predicate_wiring(_problem_with_predicate(predicate))) == "PREDICATE_SELF_CERTIFIED"


@_test
def test_independent_and_equal_passes():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    predicate = {
        "status": "WIRED",
        "required_active": frozenset({0, 1, 2}),
        "actual_active": frozenset({0, 1, 2}),
        "required_provenance": {"kind": "bounds", "ref": "N70/base64"},
        "actual_provenance": {"kind": "tir", "ref": "pset@loop0"},
    }
    report = verify_predicate_wiring(_problem_with_predicate(predicate))
    assert report["independent"] is True and report["equal"] is True, report


@_test
def test_independent_and_different_passes():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    predicate = {
        "status": "WIRED",
        "required_active": frozenset({0, 1, 2, 3, 4, 5}),
        "actual_active": frozenset({0, 1, 2}),
        "required_provenance": {"kind": "bounds", "ref": "N70/base64"},
        "actual_provenance": {"kind": "tir", "ref": "pset@loop0"},
    }
    report = verify_predicate_wiring(_problem_with_predicate(predicate))
    assert report["independent"] is True and report["equal"] is False, report


@_test
def test_missing_provenance_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    predicate = {"status": "WIRED", "required_active": frozenset({0}), "actual_active": frozenset({0})}
    assert _refusal(lambda: verify_predicate_wiring(_problem_with_predicate(predicate))) == "PREDICATE_SOURCE_UNBOUND"


@_test
def test_mutation_required_reuses_actual_is_caught():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    # A broken implementation that returns actual as required records the actual provenance on
    # both sides -> the provenance check must catch it.
    predicate = {
        "status": "WIRED",
        "required_active": frozenset({0, 1}),
        "actual_active": frozenset({0, 1}),
        "required_provenance": {"kind": "tir", "ref": "pset@loop0"},
        "actual_provenance": {"kind": "tir", "ref": "pset@loop0"},
    }
    assert _refusal(lambda: verify_predicate_wiring(_problem_with_predicate(predicate))) == "PREDICATE_SELF_CERTIFIED"


@_test
def test_unknown_predicate_still_none():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    predicate = {"status": "UNKNOWN", "required_active": None, "actual_active": None}
    assert verify_predicate_wiring(_problem_with_predicate(predicate))["status"] == "UNKNOWN"


# --- POC gate ---------------------------------------------------------------


def _access(task, root, mode, off, footprint, space):
    from tilefoundry_ascend.schedule.ledger import ProblemAccess

    return ProblemAccess(
        task=task, instruction=task, root=root, root_kind="real", space=space, mode=mode,
        byte_footprint=footprint, byte_offset=off, source_offset="[]", attrs=(),
        offset_structure=(("literal", off),), guard=(), normalized_offset=off,
        offset_status=("BYTES" if off is not None else "OFFSET_UNKNOWN"),
        logical_iteration="outside", iteration_path="outside", iteration_extent="1", iteration_domains=(),
    )


@_test
def test_poc_gate_passes_on_synthetic_supported_case():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies, verify_poc_dependency_gate

    accesses = [
        _access("p_load", "ub", "write", 0, 64, "SMEM"),
        _access("c_vld", "ub", "read", 0, 64, "SMEM"),
        _access("p_vsts", "ub2", "write", 0, 64, "SMEM"),
        _access("c_gm", "ub2", "read", 0, 64, "SMEM"),
    ]
    problem = types.SimpleNamespace(accesses=accesses, dependencies=derive_dependencies(accesses))
    ops = {"p_load": "GmToUb", "c_vld": "Vld", "p_vsts": "Vsts", "c_gm": "UbToGm"}
    report = verify_poc_dependency_gate(problem, op_by_task=ops)
    assert report["load_to_vf_raw"] >= 1 and report["vf_to_store_raw"] >= 1, report
    assert report["supported_unknown"] == 0 and report["critical_unknown"] == 0, report


@functools.lru_cache(maxsize=1)
def _vf_problem():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    prog = checked_program(VF + ":VfVadd", None, command="task80-gate")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


@_test
def test_poc_gate_on_real_carrier_records_named_gap():
    from tilefoundry_ascend.schedule.ledger import LedgerError, verify_poc_dependency_gate

    ledger, problem = _vf_problem()
    try:
        report = verify_poc_dependency_gate(problem, ledger=ledger)
    except LedgerError as error:
        message = str(error)
        assert message.startswith("POC_DEPENDENCY_GATE_FAILED"), message
        # The real carrier currently lacks a source-proven load->VF RAW because the SMEM root
        # lineage does not unify across the Mesh/ExecutionScope boundary.
        assert "MISSING_SOURCE_PROVEN_LOAD_TO_VF_RAW" in message, message
        print("POC GAP (real carrier):", message)
    else:
        # If a future fix unifies the lineage, the gate must then fully pass.
        assert report["load_to_vf_raw"] >= 1 and report["vf_to_store_raw"] >= 1, report
        assert report["supported_unknown"] == 0 and report["critical_unknown"] == 0, report


def main() -> int:
    passed = failed = errored = 0
    for name, case in TESTS.items():
        try:
            case()
        except AssertionError as error:
            failed += 1
            print("FAIL %s: %s" % (name, error))
        except Exception as error:  # noqa: BLE001
            errored += 1
            print("ERROR %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("PASS %s" % name)
    print("%d/%d passed, %d failed, %d errored" % (passed, len(TESTS), failed, errored))
    if errored:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
