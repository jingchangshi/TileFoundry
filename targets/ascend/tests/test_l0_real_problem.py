"""L0: real-problem/1 schema, canonical digest, bidirectional coverage.

The Problem handed to S/R/C/T must state its schema, hash its own canonical payload, and
cover every dataflow statement with an action and every action with a statement AND an
access. These controls prove the positive path, then each decisive defect by name:
a mutated field or a dropped policy changes the digest; a dropped action loses its access;
a statement with no action is named.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import dataclasses
import functools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "examples", "rmsnorm_950pr", "vf_vadd.py")


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    prog = checked_program(FIX + ":VfVadd", None, command="l0-real-problem")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


def _refusal(call):
    from tilefoundry_ascend.schedule.ledger import LedgerError

    try:
        call()
    except LedgerError as error:
        return str(error).split(":", 1)[0]
    return None


TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


@_test
def test_schema_and_verify_pass():
    from tilefoundry_ascend.schedule.ledger import REAL_PROBLEM_SCHEMA, verify_real_problem

    ledger, problem = _built()
    assert problem.schema == REAL_PROBLEM_SCHEMA, problem.schema
    report = verify_real_problem(problem, ledger)
    assert report["coverage"]["bidirectional"] is True
    assert report["tasks"]["bidirectional"] is True
    assert report["statements"]["bidirectional"] is True


@_test
def test_statement_action_coverage_is_bidirectional():
    from tilefoundry_ascend.schedule.ledger import statement_coverage

    ledger, _problem = _built()
    report = statement_coverage(ledger)
    assert report["statements"] > 0 and report["statements"] == report["actions"], report


@_test
def test_bad_dropped_policy_changes_digest():
    from tilefoundry_ascend.schedule.ledger import verify_real_problem

    ledger, problem = _built()
    assert _refusal(lambda: verify_real_problem(dataclasses.replace(problem, policy={}), ledger)) == "PROBLEM_DIGEST_MISMATCH"


@_test
def test_bad_mutated_field_changes_digest():
    from tilefoundry_ascend.schedule.ledger import verify_real_problem

    ledger, problem = _built()
    assert _refusal(lambda: verify_real_problem(dataclasses.replace(problem, loop_domain={"mutated": 1}), ledger)) == "PROBLEM_DIGEST_MISMATCH"


@_test
def test_bad_unknown_schema_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_real_problem

    ledger, problem = _built()
    assert _refusal(lambda: verify_real_problem(dataclasses.replace(problem, schema="real-problem/0"), ledger)) == "PROBLEM_SCHEMA_UNSUPPORTED"


@_test
def test_bad_action_without_access_is_refused():
    from tilefoundry_ascend.schedule.ledger import coverage

    ledger, problem = _built()
    assert _refusal(lambda: coverage(dataclasses.replace(ledger, entries=ledger.entries[1:]), problem)) == "COVERAGE_MISMATCH"


@_test
def test_bad_statement_without_action_is_refused():
    from tilefoundry_ascend.schedule.ledger import statement_coverage

    ledger, _problem = _built()
    forged = dataclasses.replace(ledger, statement_paths=ledger.statement_paths + ["s999/fake"])
    assert _refusal(lambda: statement_coverage(forged)) == "STATEMENT_COVERAGE_MISMATCH"


@_test
def test_bad_action_without_statement_is_refused():
    from tilefoundry_ascend.schedule.ledger import statement_coverage

    ledger, _problem = _built()
    assert _refusal(lambda: statement_coverage(dataclasses.replace(ledger, statement_paths=[]))) == "STATEMENT_COVERAGE_MISMATCH"


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
