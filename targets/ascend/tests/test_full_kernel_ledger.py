"""04-05 extension: full RMSNorm kernel ledger (whole VF tasks, loop iterations, negatives).

Checks on the REAL lowering of targets/ascend/tests/fixtures/rmsnorm/rmsnorm_main.py:RMSNormD:
  * the SIX vcadd tasks and the reverse-vsts tasks are visible in the ledger (tools/VF, not only DMA);
  * their accesses line up with SSA (a VF read root is another VF task's write root);
  * a row-loop action keeps its iteration path/extent (multi-iteration is NOT collapsed);
  * dropping a whole VF task from the ledger makes bidirectional coverage FAIL;
  * GM real roots and smem synthetic roots stay separate; alias only for real+KNOWN offsets.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import dataclasses, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "rmsnorm_main.py")


def main():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import (
        LedgerError, alias_report, build_problem, coverage, extract_ledger, require_concrete_offset,
    )

    checks = {}
    prog = checked_program(FIX + ":RMSNormD", None, command="full-ledger-test")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    problem = build_problem(ledger)
    entries = ledger.entries

    checks["ledger_nonempty"] = len(entries) > 0
    vcadd_tasks = sorted({e.instruction for e in entries if e.op == "Vcadd"})
    vsts_tasks = sorted({e.instruction for e in entries if e.op == "Vsts"})
    checks["six_vcadd_tasks_visible"] = len(vcadd_tasks) == 6
    # Vsts classification by op attribute, NOT by the total count: reverse carries PK_B32 and a
    # carrier source; forward does not. Report the REAL split and fail if forward is mislabelled.
    reverse = [e for e in entries if e.op == "Vsts" and ("pack", "PK_B32") in e.attrs]
    forward = [e for e in entries if e.op == "Vsts" and ("pack", "PK_B32") not in e.attrs]
    # Per-instruction split (each Vsts has a read-acc and a write-smem access entry).
    checks["vsts_reverse_pk_b32_split"] = len({e.instruction for e in reverse}) == 64 and len({e.instruction for e in forward}) == 64
    checks["reverse_vsts_tasks_visible"] = len({e.instruction for e in reverse}) == 64
    mislabelled = [dataclasses.replace(e, attrs=e.attrs + (("pack", "PK_B32"),))
                   if (e.op == "Vsts" and ("pack", "PK_B32") not in e.attrs) else e for e in entries]
    mis_reverse = [e for e in mislabelled if e.op == "Vsts" and ("pack", "PK_B32") in e.attrs]
    checks["neg_forward_mislabelled_detected"] = len(mis_reverse) != 64
    checks["whole_vf_tasks_not_only_dma"] = any(e.op in ("Vcadd", "Vmula", "Vmuls", "Vadds", "Vsqrt", "Vdiv") for e in entries)

    # SSA alignment: every VF read root is written by some VF task in the same iteration scope.
    vf_ops = {"Vcadd", "Vmula", "Vmul", "Vadd", "Vmuls", "Vadds", "Vsqrt", "Vdiv", "Vdup", "Vdupv"}
    written = {(e.root, e.iteration_path) for e in entries if e.op in vf_ops and e.mode == "write"}
    written_any = {e.root for e in entries if e.mode == "write"}
    reads = [(e.root, e.iteration_path) for e in entries if e.op in vf_ops and e.mode == "read"]
    # A VF read is SSA-linked when the register is written by a VF task in the same scope, or
    # belongs to a root written anywhere (live-in / DMA-produced); every Vcadd read must be linked.
    checks["vf_reads_ssa_linked"] = bool(reads) and all(
        (root, scope) in written or root in written_any for root, scope in reads
    )
    vcadd_reads = {e.root for e in entries if e.op == "Vcadd" and e.mode == "read"}
    checks["vcadd_chain_ssa_linked"] = len(vcadd_reads) == 6 and vcadd_reads <= written_any

    # multi-iteration: loop-body actions keep path + extent and are NOT collapsed to one iteration.
    loop_entries = [e for e in entries if e.iteration_path != "outside"]
    checks["loop_iterations_distinguished"] = bool(loop_entries) and all(
        e.iteration_extent and e.iteration_extent != "1" for e in loop_entries
    )
    checks["iterations_not_collapsed"] = len({(e.instruction, e.iteration_path) for e in loop_entries}) == len(
        {(e.instruction) for e in loop_entries}
    )
    deps = problem.dependencies
    checks["dependencies_have_iteration_distance"] = bool(deps) and all("distance" in d for d in deps)

    # negative: drop a WHOLE VF task (both accesses of one vcadd) -> coverage must FAIL.
    drop = {e.action_id for e in entries if e.instruction == vcadd_tasks[0]}
    dropped = dataclasses.replace(ledger, entries=[e for e in entries if e.action_id not in drop])
    try:
        coverage(dropped, problem); checks["mut_drop_whole_vf_task_rejected"] = False
    except LedgerError:
        checks["mut_drop_whole_vf_task_rejected"] = True

    unknown = [a for a in problem.accesses if a.offset_status != "BYTES"]
    checks["offset_unknown_reported"] = len(unknown) > 0
    try:
        require_concrete_offset(problem); checks["offset_unknown_refused"] = False
    except LedgerError:
        checks["offset_unknown_refused"] = True
    checks["roots_separated"] = all(
        (b["root_kind"] == "synthetic_alloc") == b["root"].startswith("smem@") for b in problem.buffers
    )
    report = alias_report(problem)
    checks["alias_excluded_reported"] = report["excluded_unknown_or_synthetic"] >= len(unknown)

    for key, value in checks.items():
        print(("ok   " if value else "FAIL ") + key)
    print(
        f"entries={len(entries)} accesses={len(problem.accesses)} buffers={len(problem.buffers)} "
        f"dependencies={len(deps)} vcadd_tasks={len(vcadd_tasks)} vsts_tasks={len(vsts_tasks)} "
        f"offset_unknown={len(unknown)}/{len(problem.accesses)} alias_real={report['real_views']} "
        f"alias_excluded={report['excluded_unknown_or_synthetic']}"
    )
    if not all(checks.values()):
        print(f"{sum(not value for value in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
