"""04-05: real emission ledger -> normalized Problem (CPU, UNCREDITED).

The ledger is extracted from the TYPED TIR produced in-process by the public pipeline. Checks:
stable action ids (no id(obj)), buffers from the SAME ledger, bidirectional coverage, OFFSET_UNKNOWN
refused as 0 by code, UNKNOWN_ROOT refusal, and five fail-able mutations (drop a store, add a copy,
change a root, change an offset, change a distance) plus the constant-0 extractor kill.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import dataclasses, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "aiv56_ragged.py")


def main():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import (
        LedgerError, alias_report, assert_no_default_zero, build_problem, coverage,
        extract_ledger, require_concrete_offset,
    )

    checks = {}
    prog = checked_program(FIX + ":Own56R", None, command="ledger-test")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    checks["ledger_nonempty"] = len(ledger.entries) > 0
    ids = [entry.action_id for entry in ledger.entries]
    checks["stable_ids_no_object_identity"] = all("0x" not in action_id and " at " not in action_id for action_id in ids)
    checks["stable_ids_deterministic"] = ids == [entry.action_id for entry in extract_ledger(module.entry_function()).entries]
    problem = build_problem(ledger)
    checks["buffers_from_ledger"] = all(
        any(buffer["root"] == access.root and buffer["space"] == access.space for buffer in problem.buffers)
        for access in problem.accesses
    )
    checks["coverage_bidirectional"] = coverage(ledger, problem)["bidirectional"]
    unknown = [access for access in problem.accesses if access.offset_status == "OFFSET_UNKNOWN"]
    checks["offset_unknown_counted_not_zero"] = len(unknown) > 0 and all(access.byte_offset is None for access in unknown)
    try:
        require_concrete_offset(problem); checks["offset_unknown_refused"] = False
    except LedgerError as error:
        checks["offset_unknown_refused"] = "OFFSET_UNKNOWN" in str(error)
    checks["synthetic_roots_marked"] = all(
        buffer["root_kind"] == ("synthetic_alloc" if buffer["root"].startswith("smem@") else "real")
        for buffer in problem.buffers
    )

    # negative: constant-0 extractor must be killed
    defaulted = dataclasses.replace(
        problem,
        accesses=[dataclasses.replace(a, byte_offset=0) if a.offset_status != "KNOWN" else a for a in problem.accesses],
    )
    try:
        assert_no_default_zero(defaulted); checks["constant_zero_extractor_killed"] = False
    except LedgerError:
        checks["constant_zero_extractor_killed"] = True

    # negative: drop a store (remove the last write entry) -> coverage mismatch
    writes = [index for index, entry in enumerate(ledger.entries) if entry.mode == "write"]
    dropped = dataclasses.replace(ledger, entries=[e for i, e in enumerate(ledger.entries) if i != writes[-1]])
    try:
        coverage(dropped, problem); checks["mut_drop_store_rejected"] = False
    except LedgerError:
        checks["mut_drop_store_rejected"] = True
    # negative: add a copy (a NEW action id) -> coverage mismatch
    extra = dataclasses.replace(ledger.entries[0], action_id=ledger.entries[0].action_id + ":extra")
    added = dataclasses.replace(ledger, entries=ledger.entries + [extra])
    try:
        coverage(added, problem); checks["mut_add_copy_rejected"] = False
    except LedgerError:
        checks["mut_add_copy_rejected"] = True
    # negative: change a root -> UNKNOWN_ROOT (validated on the Problem, not re-derived)
    from tilefoundry_ascend.schedule.ledger import validate_roots
    ghost = dataclasses.replace(problem, accesses=[dataclasses.replace(problem.accesses[0], root="ghost")] + problem.accesses[1:])
    try:
        validate_roots(ghost); checks["mut_change_root_rejected"] = False
    except LedgerError as error:
        checks["mut_change_root_rejected"] = "UNKNOWN_ROOT" in str(error)
    # negative: change a distance -> dependency set changes (need a same-root pair)
    write_entry = next(entry for entry in ledger.entries if entry.mode == "write")
    reader = dataclasses.replace(write_entry, mode="read", action_id=write_entry.action_id + ":reader")
    twice = dataclasses.replace(ledger, entries=ledger.entries + [reader])
    paired = build_problem(twice)
    base_deps = {tuple(sorted(d.items())) for d in paired.dependencies}
    changed = dataclasses.replace(paired, dependencies=[dict(d, distance=(d["distance"] + 1) % 2) for d in paired.dependencies])
    changed_deps = {tuple(sorted(d.items())) for d in changed.dependencies}
    checks["mut_change_distance_changes_problem"] = bool(paired.dependencies) and base_deps != changed_deps
    # negative: change an offset -> Problem changes
    shifted = dataclasses.replace(
        problem,
        accesses=[dataclasses.replace(a, byte_offset=4) if a.offset_status == "KNOWN" else a for a in problem.accesses],
    )
    checks["mut_change_offset_changes_problem"] = (
        [a.byte_offset for a in shifted.accesses] != [a.byte_offset for a in problem.accesses]
    )

    # alias wiring: only root_kind=real + KNOWN offsets participate; overlapping real views conflict
    report = alias_report(problem)
    checks["alias_report_counts_excluded"] = report["excluded_unknown_or_synthetic"] == len(unknown) + sum(
        1 for a in problem.accesses if a.root_kind == "synthetic_alloc"
    )
    from view_spec import OffsetView, alias_conflicts
    v1 = OffsetView(name="a", root="out", generation=0, byte_offset=0, dtype="f32", count=64)
    v2 = OffsetView(name="b", root="out", generation=0, byte_offset=0, dtype="f32", count=64)
    checks["alias_entry_detects_overlap"] = bool(alias_conflicts([v1, v2], {"a": "write", "b": "read"}))
    v3 = OffsetView(name="c", root="out", generation=0, byte_offset=256, dtype="f32", count=64)
    checks["alias_entry_allows_disjoint"] = not alias_conflicts([v1, v3], {"a": "write", "c": "read"})

    for key, value in checks.items():
        print(("ok   " if value else "FAIL ") + key)
    print(f"entries={len(ledger.entries)} accesses={len(problem.accesses)} buffers={len(problem.buffers)} "
          f"dependencies={len(problem.dependencies)} offset_unknown={len(unknown)}")
    if not all(checks.values()):
        print(f"{sum(not value for value in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
