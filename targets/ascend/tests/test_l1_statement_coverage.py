"""L1: real stmt/action/access coverage on the F0 SURVIVING vadd carrier.

The expectation is re-derived from the finalized TIR by an independent walker, not from
the ledger's own recorded paths. Positive path: every dataflow statement has an action,
every action has a statement and an access. Bad controls: drop all actions of one
statement -> stmt_without_action; forge an action with no statement -> action_without_stmt.

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
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="l1-statement-coverage")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    function = module.entry_function()
    ledger = extract_ledger(function)
    problem = build_problem(ledger)
    return function, ledger, problem


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


def _tir_for_count(node) -> int:
    from tilefoundry.ir.tir.stmts import ExecutionScope, For, If, MeshScope, Sequential

    if isinstance(node, (list, tuple)):
        return sum(_tir_for_count(item) for item in node)
    if isinstance(node, For):
        return 1 + _tir_for_count(node.body)
    if isinstance(node, If):
        return _tir_for_count(node.then_body) + _tir_for_count(node.else_body)
    if isinstance(node, (Sequential, MeshScope, ExecutionScope)):
        return _tir_for_count(node.body)
    return 0


@_test
def test_carrier_is_live_and_extracts_actions():
    function, ledger, _problem = _built()

    assert _tir_for_count(function.body) >= 1, "the carrier loop must survive to a real TIR For"
    assert len(ledger.entries) > 0
    assert len(ledger.statement_paths) > 0


@_test
def test_statement_action_access_coverage_bidirectional():
    function, ledger, problem = _built()
    from tilefoundry_ascend.schedule.ledger import verify_real_problem

    report = verify_real_problem(problem, ledger, function=function)
    assert report["statements"]["bidirectional"] is True, report
    assert report["coverage"]["bidirectional"] is True, report
    assert report["tasks"]["bidirectional"] is True, report
    assert report["statements"]["statements"] == report["statements"]["actions"], report


@_test
def test_bad_statement_without_action_is_refused():
    function, ledger, _problem = _built()
    from tilefoundry_ascend.schedule.ledger import verify_statement_coverage

    first_path = ledger.statement_paths[0]
    dropped = dataclasses.replace(
        ledger, entries=[e for e in ledger.entries if e.path != first_path]
    )
    assert _refusal(lambda: verify_statement_coverage(function, dropped)) == "STATEMENT_COVERAGE_MISMATCH"


@_test
def test_bad_action_without_statement_is_refused():
    function, ledger, _problem = _built()
    from tilefoundry_ascend.schedule.ledger import verify_statement_coverage

    first = ledger.entries[0]
    forged = dataclasses.replace(ledger, entries=ledger.entries + [dataclasses.replace(first, path="s999/fake")])
    assert _refusal(lambda: verify_statement_coverage(function, forged)) == "STATEMENT_COVERAGE_MISMATCH"


@_test
def test_bad_action_without_access_is_refused():
    function, ledger, problem = _built()
    from tilefoundry_ascend.schedule.ledger import coverage

    assert _refusal(lambda: coverage(dataclasses.replace(ledger, entries=ledger.entries[1:]), problem)) == "COVERAGE_MISMATCH"


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
