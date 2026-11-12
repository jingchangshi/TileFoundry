"""L3: budget (PipelineSpec.num_stages) and buffer versions (Empty.buffers) wired independently.

Proves the two numbers are DIFFERENT dimensions and are not copied from one another, that a
forged budget/version source is refused by name, and that wiring policy does NOT accidentally
mark the predicate WIRED (P output is not materialised).

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
    from tilefoundry_ascend.schedule.ledger import (
        build_problem, capture_problem_sources, extract_ledger,
    )

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="l3-budget")
    # Capture the typed sources BEFORE lowering mutates the module in place.
    sources = capture_problem_sources(prog.module)
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return sources, build_problem(ledger, sources), build_problem(ledger)


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
def test_budget_from_pipeline_spec_and_versions_from_empty_buffers():
    from tilefoundry_ascend.schedule.ledger import verify_policy_sources

    sources, problem, _unwired = _built()
    assert problem.policy["status"] == "WIRED"
    assert problem.policy["requested_budget"] == {"loop[0]:i": 2}, problem.policy["requested_budget"]
    assert set(problem.policy["buffer_versions"].values()) == {1}
    # INDEPENDENT: budget is 2, versions are 1 -- they are not the same number.
    assert set(problem.policy["requested_budget"].values()) != set(problem.policy["buffer_versions"].values())
    assert verify_policy_sources(sources, problem)["budgets"] == {"loop[0]:i": 2}


@_test
def test_predicate_stays_unknown_when_policy_is_wired():
    _sources, problem, _unwired = _built()
    predicate = problem.policy["predicate"]
    assert predicate["status"] == "UNKNOWN"
    assert predicate["required_active"] is None and predicate["actual_active"] is None


@_test
def test_bad_versions_copied_from_budget_are_refused():
    from tilefoundry_ascend.schedule.ledger import verify_policy_sources

    sources, problem, _unwired = _built()
    loop = next(iter(problem.policy["requested_budget"]))
    budget = problem.policy["requested_budget"][loop]
    forged = dataclasses.replace(
        problem,
        policy={**problem.policy, "buffer_versions": {k: budget for k in problem.policy["buffer_versions"]}},
    )
    assert _refusal(lambda: verify_policy_sources(sources, forged)) == "BUFFER_VERSION_DECLARATION_CHANGED"


@_test
def test_bad_budget_constant_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_policy_sources

    sources, problem, _unwired = _built()
    loop = next(iter(problem.policy["requested_budget"]))
    forged = dataclasses.replace(
        problem, policy={**problem.policy, "requested_budget": {loop: 99}}
    )
    assert _refusal(lambda: verify_policy_sources(sources, forged)) == "PIPELINE_BUDGET_SOURCE_MISMATCH"


@_test
def test_bad_forged_source_object_is_refused():
    from tilefoundry_ascend.schedule.ledger import (
        PROBLEM_SOURCES_SCHEMA, ProblemSources, validate_problem_sources,
    )

    # num_stages punched in as a constant string instead of the typed int from PipelineSpec.
    forged = ProblemSources(PROBLEM_SOURCES_SCHEMA, {"loop[0]:i": "2"}, {"empty[2]:empty": 1})
    assert _refusal(lambda: validate_problem_sources(forged)) == "PIPELINE_BUDGET_SOURCE_MISMATCH"


@_test
def test_bad_version_above_budget_is_exceeds():
    from tilefoundry_ascend.schedule.ledger import (
        PROBLEM_SOURCES_SCHEMA, ProblemSources, validate_problem_sources,
    )

    forged = ProblemSources(PROBLEM_SOURCES_SCHEMA, {"loop[0]:i": 2}, {"b": 4})
    assert _refusal(lambda: validate_problem_sources(forged)) == "BUFFER_VERSION_EXCEEDS_PIPELINE_BUDGET"


@_test
def test_bad_predicate_marked_wired_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_policy_sources

    sources, problem, _unwired = _built()
    forged = dataclasses.replace(
        problem,
        policy={**problem.policy, "predicate": {"status": "WIRED", "required_active": frozenset({0}), "actual_active": frozenset({1})}},
    )
    assert _refusal(lambda: verify_policy_sources(sources, forged)) == "PREDICATE_STATUS_INVALID"


@_test
def test_unwired_problem_remains_unwired_without_sources():
    _sources, _wired, unwired = _built()
    assert unwired.policy["status"] == "UNWIRED_L0"
    assert unwired.policy["requested_budget"] is None
    assert unwired.policy["buffer_versions"] == {}


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
