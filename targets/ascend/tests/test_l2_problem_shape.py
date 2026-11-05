"""L2: root/window, whole-VF task, final GM return chain, predicate UNKNOWN, escape.

Positives on the surviving vadd carrier, then each decisive defect by name:
removing the final GM store, a wrong escape endpoint, a self-certifying predicate, an
UNKNOWN predicate filled with 0, and a split whole-VF task. Plus a cross-module check:
a program whose loop is DCE'd must show ZERO dataflow loop statements and must not pass
against a STALE ledger that still holds the old actions.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import dataclasses
import functools
import pathlib
import tempfile
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")

DEAD_SOURCE = '''from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

@module(entry="k", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class M:
    @func
    def k(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:
        acc = x
        for i in range(4):
            acc = acc + 1.0
        return x
'''


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="l2-problem-shape")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    function = module.entry_function()
    ledger = extract_ledger(
        function,
        escape_records=module.metadata.get("escape_records"),
        declaration_sites=module.metadata.get("escape_declaration_sites"),
    )
    return function, ledger, build_problem(ledger)


def _dead_function():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize

    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_l2_dead_"))
    path = directory / "prog.py"
    path.write_text(DEAD_SOURCE)
    prog = checked_program("%s:M" % path, None, command="l2-dead")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    return finalize(analyzed, authored=prog.authored).entry_function()


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
def test_positive_l2_shape():
    from tilefoundry_ascend.schedule.ledger import verify_l2_problem

    function, ledger, problem = _built()
    report = verify_l2_problem(function, ledger, problem)
    assert report["final_gm"]["outputs"] == ["out"], report
    assert report["final_gm"]["final_gm_writes"] >= 1
    assert report["vf_tasks"]["vf_tasks"] >= 1
    assert report["predicate"]["status"] == "UNKNOWN"
    predicate = problem.policy["predicate"]
    assert predicate["required_active"] is None and predicate["actual_active"] is None


@_test
def test_bad_missing_final_gm_store_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_final_gm_return_chain

    function, ledger, _problem = _built()
    outputs = {p.name for p in function.params[-function.output_count:]}
    dropped = dataclasses.replace(
        ledger,
        entries=[e for e in ledger.entries if not (e.mode == "write" and e.space == "GMEM" and e.root in outputs)],
    )
    assert _refusal(lambda: verify_final_gm_return_chain(function, dropped)) == "FINAL_GM_RETURN_MISSING"


@_test
def test_bad_wrong_escape_endpoint_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_escape_endpoints

    _function, ledger, _problem = _built()
    edge = ledger.escape_edges[0]
    forged = dataclasses.replace(
        ledger, escape_edges=[dataclasses.replace(edge, inner_root="smem@WRONG_ENDPOINT")]
    )
    assert _refusal(lambda: verify_escape_endpoints(forged)) == "ESCAPE_EDGE_MISMATCH"


@_test
def test_bad_predicate_self_certification_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    _function, _ledger, problem = _built()
    forged = dataclasses.replace(
        problem,
        policy={
            **problem.policy,
            "predicate": {"status": "WIRED", "required_active": frozenset(range(6)), "actual_active": frozenset(range(6))},
        },
    )
    assert _refusal(lambda: verify_predicate_wiring(forged)) == "PREDICATE_SELF_CERTIFIED"


@_test
def test_bad_unknown_predicate_filled_with_zero_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_predicate_wiring

    _function, _ledger, problem = _built()
    forged = dataclasses.replace(
        problem,
        policy={**problem.policy, "predicate": {"status": "UNKNOWN", "required_active": 0, "actual_active": None}},
    )
    assert _refusal(lambda: verify_predicate_wiring(forged)) == "DEFAULTED_UNKNOWN_VALUE"


@_test
def test_bad_split_whole_vf_task_is_refused():
    from tilefoundry_ascend.schedule.ledger import verify_whole_vf_tasks

    _function, ledger, problem = _built()
    split = [
        {
            "scope": problem.vf_tasks[0]["scope"],
            "action_ids": problem.vf_tasks[0]["action_ids"][:-1],
            "instructions": [],
        }
    ]
    assert _refusal(lambda: verify_whole_vf_tasks(ledger, dataclasses.replace(problem, vf_tasks=split))) == "VF_TASK_COVERAGE_MISMATCH"


@_test
def test_bad_dce_loop_is_zero_and_stale_ledger_is_refused():
    from tilefoundry_ascend.schedule.ledger import _iter_effect_statement_paths, verify_statement_coverage

    _function, good_ledger, _problem = _built()
    dead = _dead_function()
    rows = _iter_effect_statement_paths(dead)
    assert [p for p, _s, _n in rows if "for:" in p] == [], rows
    assert rows == [], "a DCE'd loop must produce ZERO dataflow statements in the TIR"
    # A stale ledger (from the live carrier) must NOT make this pass.
    assert _refusal(lambda: verify_statement_coverage(dead, good_ledger)) == "STATEMENT_COVERAGE_MISMATCH"


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
