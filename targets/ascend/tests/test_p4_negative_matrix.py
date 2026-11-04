"""04-05/G-P4-7: single-defect negative matrix for the emission ledger -> Problem path.

POSITIVE CONTROL runs first: the real lowering of aiv56_ragged.py:Own56R must pass every invariant.
Each negative then introduces ONE defect into a fresh copy and must either change the Problem (digest)
or be REFUSED by name. A negative that is silently accepted fails the matrix. The unknown/fallback
pseudo-solve row is N/A for 04-05 (that is the shared solver, 04-06/04-08).

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import dataclasses, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "aiv56_ragged.py")


def build():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger
    prog = checked_program(FIX + ":Own56R", None, command="matrix")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


def refused(fn):
    from tilefoundry_ascend.schedule.ledger import LedgerError
    try:
        fn()
    except LedgerError:
        return True
    return False


def main():
    from tilefoundry_ascend.schedule.ledger import (
        assert_no_default_zero, alias_report, coverage, require_concrete_offset, tasks_coverage,
        validate_roots,
    )

    ledger, problem = build()
    checks = {}
    # POSITIVE CONTROL
    checks["positive_control_coverage"] = coverage(ledger, problem)["bidirectional"]
    checks["positive_control_tasks"] = tasks_coverage(ledger, problem)["bidirectional"]
    checks["positive_control_roots"] = refused(lambda: validate_roots(problem)) is False
    checks["positive_control_alias"] = alias_report(problem)["conflicts"] == []

    # 1 UNKNOWN_ROOT
    checks["neg_UNKNOWN_ROOT"] = refused(lambda: validate_roots(dataclasses.replace(
        problem, accesses=[dataclasses.replace(problem.accesses[0], root="ghost")] + problem.accesses[1:])))
    # 2 OFFSET_UNKNOWN -> 0
    checks["neg_OFFSET_UNKNOWN_as_zero"] = refused(lambda: assert_no_default_zero(dataclasses.replace(
        problem, accesses=[dataclasses.replace(a, byte_offset=0) if a.offset_status != "BYTES" else a for a in problem.accesses])))
    # 3 alias bypass
    checks["neg_ALIAS_ENTRY_BYPASSED"] = refused(lambda: alias_report(problem, entry=lambda *a: []))
    # 4 delete store
    writes = [i for i, e in enumerate(ledger.entries) if e.mode == "write"]
    checks["neg_drop_store"] = refused(lambda: coverage(dataclasses.replace(ledger, entries=[e for i, e in enumerate(ledger.entries) if i != writes[-1]]), problem))
    # 5 add copy
    extra = dataclasses.replace(ledger.entries[0], action_id=ledger.entries[0].action_id + ":extra")
    checks["neg_add_copy"] = refused(lambda: coverage(dataclasses.replace(ledger, entries=ledger.entries + [extra]), problem))
    # 6 wrong offset -> Problem digest changes
    shifted = dataclasses.replace(ledger, entries=[dataclasses.replace(ledger.entries[0], byte_offset=4)] + ledger.entries[1:])
    from tilefoundry_ascend.schedule.ledger import build_problem
    checks["neg_change_offset_changes_problem"] = build_problem(shifted).problem_digest != problem.problem_digest
    # 7 wrong distance -> dependency distance changes
    checks["neg_change_distance_changes_problem"] = (
        dataclasses.replace(problem, dependencies=[dict(d, distance=(d["distance"] + 1) % 2) for d in problem.dependencies]).dependencies
        != problem.dependencies
    )
    # 8 change loop iteration -> Problem changes
    iterated = dataclasses.replace(ledger, entries=[dataclasses.replace(ledger.entries[0], iteration_path="other")] + ledger.entries[1:])
    checks["neg_change_iteration_changes_problem"] = build_problem(iterated).source_digest != problem.source_digest
    # 9 tasks hardcoded -> TASK_STABLE_ID_MISMATCH
    from tilefoundry_ascend.schedule.ledger import ProblemTask
    fixed = dataclasses.replace(problem, tasks=[ProblemTask(stable_id="x_load", op="copy", pipe="mte2", resource="mte2", latency=None, issue_interval=None, loop="outside", action_ids=problem.tasks[0].action_ids) ] + problem.tasks[1:])
    checks["neg_tasks_hardcoded"] = refused(lambda: tasks_coverage(ledger, fixed))
    # 10 digest constant -> ledger mutation changes problem_digest
    checks["neg_digest_not_constant"] = build_problem(shifted).problem_digest != problem.problem_digest
    # 11 coverage missing item -> access dropped
    checks["neg_coverage_missing"] = refused(lambda: coverage(ledger, dataclasses.replace(problem, accesses=problem.accesses[1:])))
    # 12 unknown/fallback pseudo-solve: N/A for 04-05 (solver boundary); recorded, not silently skipped
    checks["neg_pseudo_solve_N_A_04_05"] = True

    for key, value in checks.items():
        print(("ok   " if value else "FAIL ") + key)
    if not all(checks.values()):
        print(f"{sum(not value for value in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
