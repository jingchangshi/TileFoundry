"""task-82: POC gate G0-G9 on pipe_vadd_carrier + all seven bad controls."""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import copy
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
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="task82-g0g9")
    hir_module = copy.deepcopy(prog.module)
    hir_function = hir_module.entry_function()
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    # The CLEAN serial loop, for the same reason as test_l2_problem_shape: entry_function()
    # is now unrolled and completion-carrying, and its injected FlagSet/FlagWait statements
    # have no ledger action. hir_function above is deliberately LEFT as-is -- it comes from
    # the pre-finalize HIR module and is not the same program.
    from tilefoundry_ascend.passes.p3_chain import pre_completion_tir

    tir_function = pre_completion_tir(module)
    ledger = extract_ledger(tir_function)
    problem = build_problem(ledger)
    expansion = expand_loop_instances(ledger)
    return hir_module, hir_function, tir_function, ledger, problem, expansion


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


def _gate(**overrides):
    from tilefoundry_ascend.schedule.poc_gate import verify_poc_gate

    hir_module, hir_function, tir_function, ledger, problem, expansion = _built()
    kwargs = dict(hir_module=hir_module, hir_function=hir_function, tir_function=tir_function)
    kwargs.update(overrides)
    return verify_poc_gate(
        kwargs.pop("ledger", ledger), kwargs.pop("problem", problem), kwargs.pop("expansion", expansion), **kwargs
    )


@_test
def test_g0_g9_all_pass_with_full_denominator():
    report = _gate()
    assert report["raw_count"] == 12, report
    assert report["ssa_count"] >= 16, report
    assert report["reuse_count"] >= 9, report
    assert report["final_count"] == 1
    assert report["expected_required_count"] == 12 + report["ssa_count"] + report["reuse_count"] + 1
    assert report["required_unknown_count"] == 0 and report["critical_unknown_count"] == 0
    assert len(report["gm_tiles"]) == 4 and report["gm_tiles"][0][0] == 0 and report["gm_tiles"][-1][1] == 1024


@_test
def test_g0_requires_live_pipeline():
    from tilefoundry_ascend.schedule.poc_gate import verify_g0_pipeline_live

    hir_module, hir_function, tir_function, _l, _p, _e = _built()
    assert verify_g0_pipeline_live(hir_module, hir_function, tir_function)["hir_loops"] >= 1


@_test
def test_bad_delete_a_load_branch():
    _hm, _hf, _tf, ledger, problem, expansion = _built()
    mutant = dataclasses.replace(ledger, entries=[e for e in ledger.entries if not (e.op == "GmToUb" and "/s5" in e.path)])
    from tilefoundry_ascend.schedule.poc_gate import verify_poc_gate
    code = _refusal(lambda: verify_poc_gate(mutant, __import__("tilefoundry_ascend.schedule.ledger", fromlist=["build_problem"]).build_problem(mutant), expansion, hir_module=_hm, hir_function=_hf, tir_function=_tf))
    assert code in {"POC_RAW_LOAD_VF_MISSING", "POC_REQUIRED_SOURCE_SET_MISMATCH"}, code


@_test
def test_bad_drop_last_store():
    _hm, _hf, _tf, ledger, problem, expansion = _built()
    mutant = dataclasses.replace(ledger, entries=[e for e in ledger.entries if e.op != "copy"])
    from tilefoundry_ascend.schedule.poc_gate import verify_poc_gate
    from tilefoundry_ascend.schedule.ledger import build_problem
    code = _refusal(lambda: verify_poc_gate(mutant, build_problem(mutant), expansion, hir_module=_hm, hir_function=_hf, tir_function=_tf))
    assert code in {"POC_RAW_VF_STORE_MISSING", "FINAL_GM_RETURN_MISSING", "POC_STATEMENT_COVERAGE_MISMATCH"}, code


@_test
def test_bad_window_plus_64_out_of_range():
    _hm, _hf, _tf, ledger, problem, expansion = _built()
    mutated = copy.deepcopy(expansion)
    space = {e.action_id: e.space for e in ledger.entries}
    for task, window in list(mutated.instances[-1]["windows"].items()):
        if space.get(task) == "GMEM":
            mutated.instances[-1]["windows"][task] = (window[0] + 64, window[1] + 64)
    from tilefoundry_ascend.schedule.poc_gate import verify_poc_gate
    code = _refusal(lambda: verify_poc_gate(ledger, problem, mutated, hir_module=_hm, hir_function=_hf, tir_function=_tf))
    assert code == "POC_BYTE_WINDOW_MISMATCH", code


@_test
def test_bad_unknown_offset_is_critical_not_zero():
    _hm, _hf, _tf, ledger, problem, expansion = _built()
    stored = next(e for e in ledger.entries if e.op == "Vsts" and e.mode == "write")
    mutant = dataclasses.replace(ledger, entries=[
        (dataclasses.replace(e, byte_offset=None) if e is stored else e) for e in ledger.entries
    ])
    from tilefoundry_ascend.schedule.poc_gate import verify_poc_gate
    from tilefoundry_ascend.schedule.ledger import build_problem
    code = _refusal(lambda: verify_poc_gate(mutant, build_problem(mutant), expansion, hir_module=_hm, hir_function=_hf, tir_function=_tf))
    assert code in {"POC_CRITICAL_DEPENDENCY_UNKNOWN", "POC_RAW_VF_STORE_MISSING"}, code


@_test
def test_bad_delete_a_proof_keeps_denominator():
    from tilefoundry_ascend.schedule.poc_gate import build_expected_obligations, assert_expected_matches

    _hm, _hf, _tf, ledger, problem, expansion = _built()
    authoritative = build_expected_obligations(ledger, problem, expansion)["all"]
    provided = [o for o in authoritative if not o.oid.startswith("RAW:X-in:i0")]
    assert _refusal(lambda: assert_expected_matches(provided, authoritative)) == "POC_REQUIRED_COVERAGE_MISMATCH"


@_test
def test_bad_skip_free_war():
    from tilefoundry_ascend.schedule.poc_gate import build_expected_obligations, assert_expected_matches

    _hm, _hf, _tf, ledger, problem, expansion = _built()
    authoritative = build_expected_obligations(ledger, problem, expansion)["all"]
    provided = [o for o in authoritative if o.kind not in ("WAR", "FREE")]
    assert _refusal(lambda: assert_expected_matches(provided, authoritative)) == "POC_REQUIRED_COVERAGE_MISMATCH"


@_test
def test_bad_g9_same_source_required():
    _hm, _hf, _tf, ledger, problem, expansion = _built()
    forged = dataclasses.replace(problem, policy={**problem.policy, "predicate": {
        "status": "WIRED",
        "required_active": frozenset({0, 1}),
        "actual_active": frozenset({0, 1}),
        "required_provenance": {"kind": "tir", "ref": "pset"},
        "actual_provenance": {"kind": "tir", "ref": "pset"},
    }})
    from tilefoundry_ascend.schedule.poc_gate import verify_poc_gate
    code = _refusal(lambda: verify_poc_gate(ledger, forged, expansion, hir_module=_hm, hir_function=_hf, tir_function=_tf, predicate_status="WIRED"))
    assert code == "PREDICATE_SELF_CERTIFIED", code


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
