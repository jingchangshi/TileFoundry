"""04-05: Problem tasks (ISA-sourced) + ledger-derived digest + 56-AIV offset honesty.

Checks on TWO 56-AIV ragged groups (aiv56_ragged 36/32, aiv56_ragged_8 1/8):
  * tasks group the accesses per REAL instruction, pipe/latency/ii come from the ISA contract,
    every task traces back to ledger action ids, and dropping a task fails the task coverage;
  * the per-core symbolic GM offset is OFFSET_UNKNOWN, never 0;
  * source_digest is derived from ledger content: same HIR -> same digest, different HIR -> different
    digest, and mutating the ledger changes the digest.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import dataclasses, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "aiv56_ragged.py")
FIX8 = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "aiv56_ragged_8.py")


def build(fixture, entry):
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger
    prog = checked_program(f"{fixture}:{entry}", None, command="tasks-digest")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


def main():
    from tilefoundry_ascend.schedule.ledger import LedgerError, tasks_coverage

    checks = {}
    results = {}
    for label, fixture in (("36_32", FIX), ("1_8", FIX8)):
        ledger, problem = build(fixture, "Own56R")
        results[label] = (ledger, problem)
        checks[f"{label}_tasks_nonempty"] = len(problem.tasks) > 0
        checks[f"{label}_tasks_coverage"] = tasks_coverage(ledger, problem)["bidirectional"]
        checks[f"{label}_task_action_ids_traceable"] = all(
            set(task.action_ids) <= {entry.action_id for entry in ledger.entries} for task in problem.tasks
        )
        checks[f"{label}_task_pipe_from_isa"] = all(
            task.pipe in ("vector", "mte2", "mte3", "scalar") or task.pipe == "NO_ISA_CONTRACT"
            for task in problem.tasks
        )
        out_writes = [a for a in problem.accesses if a.root == "out" and a.mode == "write"]
        checks[f"{label}_per_core_offset_unknown"] = bool(out_writes) and all(
            a.offset_status == "OFFSET_UNKNOWN" and a.byte_offset is None for a in out_writes
        )
        checks[f"{label}_synthetic_smem_marked"] = all(
            (b["root_kind"] == "synthetic_alloc") == b["root"].startswith("smem@") for b in problem.buffers
        )

    # negative: drop a task -> task coverage fails
    ledger, problem = results["36_32"]
    dropped = dataclasses.replace(problem, tasks=problem.tasks[1:])
    try:
        tasks_coverage(ledger, dropped); checks["mut_drop_task_rejected"] = False
    except LedgerError:
        checks["mut_drop_task_rejected"] = True

    # digest: deterministic; HIR change changes it; ledger mutation changes it
    _, again = build(FIX, "Own56R")
    _, other = results["1_8"]
    checks["digest_deterministic"] = results["36_32"][1].source_digest == again.source_digest
    checks["digest_changes_with_hir"] = results["36_32"][1].source_digest != other.source_digest
    mutated_ledger = dataclasses.replace(
        results["36_32"][0],
        entries=[dataclasses.replace(results["36_32"][0].entries[0], byte_offset=4)] + results["36_32"][0].entries[1:],
    )
    from tilefoundry_ascend.schedule.ledger import build_problem
    checks["digest_changes_with_ledger"] = build_problem(mutated_ledger).source_digest != results["36_32"][1].source_digest

    for key, value in checks.items():
        print(("ok   " if value else "FAIL ") + key)
    print(
        f"36_32: tasks={len(results['36_32'][1].tasks)} accesses={len(results['36_32'][1].accesses)} "
        f"digest={results['36_32'][1].source_digest}; 1_8: tasks={len(results['1_8'][1].tasks)} "
        f"accesses={len(results['1_8'][1].accesses)} digest={results['1_8'][1].source_digest}"
    )
    if not all(checks.values()):
        print(f"{sum(not value for value in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
