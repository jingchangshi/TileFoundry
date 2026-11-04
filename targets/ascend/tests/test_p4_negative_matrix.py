"""04-05/G-P4-7: single-defect negative matrix for the emission ledger -> Problem path.

Categories are NOT collapsed into PASS:
  PASS    - the positive control (the real lowering passes every invariant)
  REFUSED - a single defect is rejected by a NAMED refusal (LedgerError)
  CHANGED - a defect in the LEDGER, re-run through the extractor, changes the Problem
            (new/removed dependency or a changed distance/digest)
  N/A     - out of 04-05 scope (the solver's unknown/fallback row, 04-06/04-08); NOT counted.

A digest probe is pinned: with the digest hardwired to a constant, the offset-mutation probe must
FAIL, proving the probe can fail.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import dataclasses, os, sys, types

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


def main():
    from tilefoundry_ascend.schedule import ledger as ledger_mod
    from tilefoundry_ascend.schedule.ledger import (
        LedgerError, ProblemTask, alias_report, assert_no_default_zero, build_problem, coverage,
        tasks_coverage, validate_roots,
    )

    ledger, problem = build()
    rows = []
    def add(category, name, ok): rows.append((category, name, bool(ok)))
    def refused(fn):
        try:
            fn()
        except LedgerError:
            return True
        return False

    # PASS: positive control
    add("PASS", "positive_control_coverage", coverage(ledger, problem)["bidirectional"])
    add("PASS", "positive_control_tasks", tasks_coverage(ledger, problem)["bidirectional"])
    add("PASS", "positive_control_roots", not refused(lambda: validate_roots(problem)))
    add("PASS", "positive_control_alias", alias_report(problem)["conflicts"] == [])

    # REFUSED: named refusals
    add("REFUSED", "neg_UNKNOWN_ROOT", refused(lambda: validate_roots(dataclasses.replace(
        problem, accesses=[dataclasses.replace(problem.accesses[0], root="ghost")] + problem.accesses[1:]))))
    add("REFUSED", "neg_OFFSET_UNKNOWN_as_zero", refused(lambda: assert_no_default_zero(dataclasses.replace(
        problem, accesses=[dataclasses.replace(a, byte_offset=0) if a.offset_status != "BYTES" else a for a in problem.accesses]))))
    add("REFUSED", "neg_ALIAS_ENTRY_BYPASSED", refused(lambda: alias_report(problem, entry=lambda *a: [])))
    writes = [i for i, e in enumerate(ledger.entries) if e.mode == "write"]
    add("REFUSED", "neg_drop_store", refused(lambda: coverage(dataclasses.replace(
        ledger, entries=[e for i, e in enumerate(ledger.entries) if i != writes[-1]]), problem)))
    extra = dataclasses.replace(ledger.entries[0], action_id=ledger.entries[0].action_id + ":extra")
    add("REFUSED", "neg_add_copy", refused(lambda: coverage(dataclasses.replace(ledger, entries=ledger.entries + [extra]), problem)))
    fixed = dataclasses.replace(problem, tasks=[ProblemTask(stable_id="x_load", op="copy", pipe="mte2", resource="mte2",
                                                           latency=None, issue_interval=None, loop="outside",
                                                           action_ids=problem.tasks[0].action_ids)] + problem.tasks[1:])
    add("REFUSED", "neg_tasks_hardcoded", refused(lambda: tasks_coverage(ledger, fixed)))
    add("REFUSED", "neg_coverage_missing", refused(lambda: coverage(ledger, dataclasses.replace(problem, accesses=problem.accesses[1:]))))

    # CHANGED: mutate the LEDGER and RE-RUN the extractor
    writer = ledger.entries[writes[0]]
    reader = dataclasses.replace(writer, mode="read", action_id=writer.action_id + ":reader",
                                 iteration_path=writer.iteration_path + "/second", logical_iteration="second")
    paired = dataclasses.replace(ledger, entries=ledger.entries + [reader])
    paired_problem = build_problem(paired)
    moved_reader = dataclasses.replace(reader, iteration_path=writer.iteration_path, logical_iteration=writer.logical_iteration)
    same_problem = build_problem(dataclasses.replace(paired, entries=ledger.entries + [moved_reader]))
    def dep_signature(p):
        return {(d["producer"], d["consumer"], d["distance"], d["kind"]) for d in p.dependencies}
    add("CHANGED", "changed_dependency_distance_rebuilt",
        bool(paired_problem.dependencies) and dep_signature(paired_problem) != dep_signature(same_problem))
    add("CHANGED", "changed_delete_dependency_rebuilt",
        dep_signature(build_problem(dataclasses.replace(ledger, entries=[e for e in paired.entries if e.action_id != reader.action_id])))
        != dep_signature(paired_problem))
    first = ledger.entries[0]
    add("CHANGED", "changed_root_digest",
        build_problem(dataclasses.replace(ledger, entries=[dataclasses.replace(first, root="x2")] + ledger.entries[1:])).source_digest
        != problem.source_digest)
    # The digest follows the typed offset STRUCTURE, not the human string: mutating the structure
    # changes the digest, while a string-only edit must NOT change it (the old broken probe).
    symbolic = next((e for e in ledger.entries if e.byte_offset is None), ledger.entries[0])
    symbolic_index = next(i for i, e in enumerate(ledger.entries) if e is symbolic)
    shifted_ledger = dataclasses.replace(
        ledger, entries=ledger.entries[:symbolic_index] + [dataclasses.replace(symbolic, offset_structure=symbolic.offset_structure + (("const", 64),))] + ledger.entries[symbolic_index + 1:])
    add("CHANGED", "changed_symbolic_offset_digest", build_problem(shifted_ledger).problem_digest != problem.problem_digest)
    string_only_ledger = dataclasses.replace(
        ledger, entries=ledger.entries[:symbolic_index] + [dataclasses.replace(symbolic, source_offset="hand-edited")] + ledger.entries[symbolic_index + 1:])
    add("PASS", "neg_string_only_offset_rejected", build_problem(string_only_ledger).problem_digest == problem.problem_digest)
    add("CHANGED", "changed_iteration_digest",
        build_problem(dataclasses.replace(ledger, entries=[dataclasses.replace(first, iteration_path="other")] + ledger.entries[1:])).source_digest
        != problem.source_digest)
    attr_entry = next((e for e in ledger.entries if e.attrs), None)
    if attr_entry is not None:
        ai = next(i for i, e in enumerate(ledger.entries) if e is attr_entry)
        wide = dataclasses.replace(attr_entry, attrs=attr_entry.attrs + (("elem_width", 32),))
        add("CHANGED", "changed_attr_range_digest",
            build_problem(dataclasses.replace(ledger, entries=ledger.entries[:ai] + [wide] + ledger.entries[ai + 1:])).problem_digest
            != problem.problem_digest)
    else:
        add("CHANGED", "changed_attr_range_digest", False)

    # PASS: digest probe can fail (hardwire the digest; the offset probe must go False)
    original_hashlib = ledger_mod.hashlib
    class _Const:
        def hexdigest(self): return "0" * 64
    try:
        ledger_mod.hashlib = types.SimpleNamespace(sha256=lambda *a, **k: _Const())
        stub_base = build_problem(ledger)
        stub_shifted = build_problem(shifted_ledger)
    finally:
        ledger_mod.hashlib = original_hashlib
    # Under a hardwired digest both builds must agree, i.e. the offset probe CAN fail.
    add("PASS", "digest_probe_can_fail", stub_shifted.problem_digest == stub_base.problem_digest)

    # N/A: solver row (04-06/04-08), excluded from the valid count
    add("N/A", "pseudo_solve_solver_row", True)

    valid = [row for row in rows if row[0] != "N/A"]
    for category, name, ok in rows:
        verdict = category if ok else "FAIL"
        print(f"{verdict:<8} {name}")
    print(f"valid={sum(ok for _, _, ok in valid)}/{len(valid)} N/A={sum(1 for c, _, _ in rows if c == 'N/A')}")
    return 0 if all(ok for _, _, ok in valid) else 1


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
