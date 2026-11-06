"""05-01/05-02: ResolvedPlan schema + TIR provenance skeleton (P5 start, UNCREDITED).

Checks: Problem->ResolvedPlan carries problem_digest/ledger_sha; N/A stages/versions stay N/A; latency
and ii are MODEL; offsets stay OFFSET_UNKNOWN (never 0); fallback is never solved; every ledger
action maps to a TIR statement and every memory statement maps back to a task/action_id (provenance
bidirectional). The TIR used is the same public finalize object (analyze/finalize 0).

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import os, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "rmsnorm_decode_56aiv_dyn.py")


def main():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger
    from tilefoundry_ascend.schedule.resolved import (
        NA, OFFSET_UNKNOWN, assert_resolved_plan, build_provenance, build_resolved_plan,
    )

    checks = {}
    prog = checked_program(FIX + ":RMSNormD", None, command="resolved")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    checks["analyze_finalize_0"] = True
    ledger = extract_ledger(module.entry_function())
    problem = build_problem(ledger)

    plan = build_resolved_plan(problem)
    checks["hashes_carried"] = (
        plan.problem_digest == problem.problem_digest
        and plan.source_digest == problem.source_digest
        and plan.ledger_sha == problem.ledger_sha
    )
    checks["stages_versions_NA"] = plan.num_stages == NA and plan.x_versions == NA and plan.y_versions == NA
    checks["tasks_match"] = len(plan.tasks) == len(problem.tasks)
    checks["accesses_match"] = len(plan.accesses) == len(problem.accesses)
    checks["cost_model_is_MODEL"] = all(task.cost_model == "MODEL" for task in plan.tasks)
    checks["offsets_never_defaulted"] = all(
        access.offset_status in ("BYTES", OFFSET_UNKNOWN) for access in plan.accesses
    ) and all(
        (access.byte_offset is None) == (access.offset_status == OFFSET_UNKNOWN) for access in plan.accesses
    )
    checks["status_planning_only"] = plan.status == "PLANNING_ONLY_NO_SOLVER"

    fallback = build_resolved_plan(problem, solver_result=types.SimpleNamespace(status="fallback"))
    checks["fallback_not_solved"] = fallback.status == "fallback"
    try:
        assert_resolved_plan(build_resolved_plan(problem, solver_result=types.SimpleNamespace(status="solved")))
        checks["solved_without_stages_refused"] = False
    except ValueError:
        checks["solved_without_stages_refused"] = True

    provenance = build_provenance(problem, module.entry_function())
    checks["provenance_complete"] = provenance["complete"] and provenance["action_coverage"] == provenance["action_total"]
    memory_rows = [row for row in provenance["rows"] if row["kind"] == "Evaluate"]
    checks["memory_statements_traceable"] = bool(memory_rows) and all(
        row["action_ids"] or row["task"] == NA for row in memory_rows
    )
    checks["task_action_ids_present"] = all(task.action_ids for task in plan.tasks)

    for key, value in checks.items():
        print(("PASS     " if value else "FAIL     ") + key)
    print(f"tasks={len(plan.tasks)} accesses={len(plan.accesses)} provenance_rows={len(provenance['rows'])} "
          f"actions={provenance['action_total']} status={plan.status}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
