"""task-82 step 1: root/alias lineage — same buffer across scopes, distinct buffers distinct.

Paired controls:
  (a) a buffer written outside and read inside an ExecutionScope must resolve to ONE canonical
      root and produce a source-proven RAW (GmToUb -> Vld, Vsts -> UbToGm);
  (b) two genuinely different buffers must stay different canonical roots (no over-merge).

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import functools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
VF = os.path.join(ROOT, "examples", "rmsnorm_950pr", "vf_vadd.py")


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    prog = checked_program(VF + ":VfVadd", None, command="task82-root")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


@_test
def test_same_buffer_across_scope_gets_one_canonical_root_and_raw():
    ledger, problem = _built()
    alias = ledger.root_alias
    # The inner ExecutionScope re-view of 'loaded' maps to the outer allocation site.
    assert any(src.endswith("ExecutionScope/s1") and dst.endswith("MeshScope/s0") for src, dst in alias.items()), alias

    by_task = {a.task: a for a in problem.accesses}
    ops = {e.action_id: e.op for e in ledger.entries}
    load_to_vf = 0
    vf_to_store = 0
    for edge in problem.dependencies:
        if edge["kind"] != "RAW" or edge["status"] != "BYTES":
            continue
        p, c = by_task[edge["producer"]], by_task[edge["consumer"]]
        if p.space != "SMEM" or c.space != "SMEM":
            continue
        if ops[edge["producer"]] == "GmToUb" and ops[edge["consumer"]] == "Vld":
            load_to_vf += 1
        if ops[edge["producer"]] == "Vsts" and ops[edge["consumer"]] == "UbToGm":
            vf_to_store += 1
    assert load_to_vf >= 1, "same-buffer cross-scope load->VF RAW missing after root fix"
    assert vf_to_store >= 1, "same-buffer cross-scope VF->store RAW missing after root fix"


@_test
def test_canonical_roots_are_fewer_but_distinct_buffers_stay_distinct():
    ledger, problem = _built()
    local_roots = {e.root for e in ledger.entries if e.root.startswith("smem@")}
    canonical_roots = {problem.root_alias_fix(e.root) if hasattr(problem, "root_alias_fix") else ledger.root_alias.get(e.root, e.root) for e in ledger.entries if e.root.startswith("smem@")}
    # Alias collapses at least one scope-local root, so canonical set is strictly smaller.
    assert len(canonical_roots) < len(local_roots), (sorted(local_roots), sorted(canonical_roots))
    # ... but two genuinely different buffers must remain two distinct canonical roots.
    assert len(canonical_roots) == 2, sorted(canonical_roots)


@_test
def test_poc_gate_passes_after_root_fix():
    from tilefoundry_ascend.schedule.ledger import verify_poc_dependency_gate

    ledger, problem = _built()
    report = verify_poc_dependency_gate(problem, ledger=ledger)
    assert report["load_to_vf_raw"] >= 1 and report["vf_to_store_raw"] >= 1, report
    assert report["supported_unknown"] == 0 and report["critical_unknown"] == 0, report


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
