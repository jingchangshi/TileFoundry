"""task-77 provenance: L-required (bounds) vs P-actual (materialised) independence + controls."""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import dataclasses
import functools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")
KEY = "d0.predicate_provenance"


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import extract_ledger

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="task77-prov")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, module.metadata.get(KEY)


def _refusal(call):
    from tilefoundry_ascend.schedule.ledger import LedgerError

    try:
        call()
    except LedgerError as error:
        return str(error).split(":", 1)[0]
    return None


def test_l_required_independent_from_actual_is_wired_and_passes():
    from tilefoundry_ascend.schedule.ledger import build_problem, verify_predicate_wiring

    ledger, metadata = _built()
    assert metadata and metadata["actual"]["kind"] == "materialised", metadata
    problem = build_problem(ledger, predicate_metadata=metadata)
    report = verify_predicate_wiring(problem)
    assert report["status"] == "WIRED" and report["independent"] is True and report["equal"] is True, report
    required = problem.policy["predicate"]["required_provenance"]
    actual = problem.policy["predicate"]["actual_provenance"]
    assert required["kind"] == "bounds" and actual["kind"] == "materialised"
    assert required["ref"] != actual["ref"]


def test_forged_same_ref_is_self_certified():
    from tilefoundry_ascend.schedule.ledger import build_problem, verify_predicate_wiring

    ledger, metadata = _built()
    problem = build_problem(ledger, predicate_metadata=metadata)
    required_ref = problem.policy["predicate"]["required_provenance"]["ref"]
    forged_meta = {"actual": {"kind": "materialised", "ref": required_ref, "active": metadata["actual"]["active"]}}
    forged = build_problem(ledger, predicate_metadata=forged_meta)
    assert _refusal(lambda: verify_predicate_wiring(forged)) == "PREDICATE_SELF_CERTIFIED"


def test_missing_metadata_stays_unknown_and_is_allowed():
    from tilefoundry_ascend.schedule.ledger import build_problem, verify_predicate_wiring

    ledger, _metadata = _built()
    problem = build_problem(ledger)
    assert problem.policy["predicate"]["status"] == "UNKNOWN"
    assert verify_predicate_wiring(problem)["status"] == "UNKNOWN"


def main() -> int:
    tests = (test_l_required_independent_from_actual_is_wired_and_passes,
             test_forged_same_ref_is_self_certified,
             test_missing_metadata_stays_unknown_and_is_allowed)
    failed = 0
    for case in tests:
        try:
            case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (case.__name__, type(error).__name__, error))
        else:
            print("PASS %s" % case.__name__)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
