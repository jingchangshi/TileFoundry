"""task-83 vertical L->S->R(->C) single run on the POC carrier (test-only glue)."""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import json
import os
import sys
import types
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")
KEY = "d0.predicate_provenance"


def _l():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import (
        build_problem, build_s_problem_document, capture_problem_sources, extract_ledger,
    )
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances
    from tilefoundry_ascend.schedule.poc_gate import build_expected_obligations

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="task83")
    sources = capture_problem_sources(prog.module)
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    tir = module.entry_function()
    ledger = extract_ledger(tir)
    problem = build_problem(ledger, sources, predicate_metadata=module.metadata.get(KEY))
    expansion = expand_loop_instances(ledger)
    expected = build_expected_obligations(ledger, problem, expansion)
    return sources, tir, problem, expansion, expected, build_s_problem_document(problem)


def test_vertical_run():
    from tilefoundry_ascend.solver.g10 import classify_poc_result, consume_poc_expected
    from tilefoundry_ascend.solver import bridge as B
    from tilefoundry_ascend.plan import resolved_plan as RP
    from tilefoundry_ascend.plan import materialize as MAT
    from poc_vertical_adapter import project_problem, nominal_solve_result, VerticalError

    sources, tir, problem, expansion, expected, s_problem = _l()
    run = {"problem_digest": problem.problem_digest, "steps": {}, "device_ready": False}

    run["steps"]["L"] = {"status": "OK", "obligations": len(expected["all"]),
                         "predicate": problem.policy["predicate"]["status"], "kind": "real"}
    poc_in = consume_poc_expected(expected, s_problem)
    run["steps"]["S_input"] = {"status": poc_in.status, "required": poc_in.required_count,
                               "unknown": poc_in.supported_required_unknown, "kind": "real"}

    r_problem, projected = project_problem(problem, tir, expansion.count, sources)
    run["projected_fields"] = projected

    core = None
    model_facts = {"model_id": "ascend-poc-nominal", "model_version": "v1",
                   "memory_groups": [{"id": "group0",
                                      "capacity_bytes": max(1, sum(int(b["bytes"]) for b in r_problem["buffers"])),
                                      "buffers": [b["root"] for b in r_problem["buffers"]],
                                      "cost_model": "MODEL"}], "cost_model": "MODEL"}
    try:
        core = B.solve_real_problem(r_problem, model_facts, {})
    except Exception:  # noqa: BLE001
        core = None

    raw = None
    if core is not None:
        raw = types.SimpleNamespace(status=core.raw_core_status, assignment=core.assignment,
                                    core_source_hash=core.core_source_hash)
    sres = classify_poc_result(expected, s_problem, raw_core=raw, authoritative=expected)
    run["steps"]["S"] = {"state": sres.state, "sat_scope": "NOMINAL_MODEL_ONLY",
                         "postcheck": "SOLVER_POSTCHECK_NOT_IMPLEMENTED",
                         "r_usable_nominal": bool(sres.r_usable),
                         "core_source_hash": getattr(core, "core_source_hash", None), "kind": "real+nominals"}

    trimmed = {"all": expected["all"][:-1]}
    run["controls"] = {
        "delete_obligation": classify_poc_result(trimmed, s_problem, raw_core=types.SimpleNamespace(status="sat", assignment={}), authoritative=expected).code,
        "fallback_not_sat": classify_poc_result(expected, s_problem, raw_core=types.SimpleNamespace(status="fallback", fallback_candidate={}), authoritative=expected).state,
    }

    run["steps"]["R"] = {"status": "NOT_RUN", "kind": "unknown"}
    run["steps"]["C"] = {"status": "NOT_RUN", "code": "COMPLETION_NOT_VERIFIED", "kind": "unknown"}

    plan = None
    if sres.state == "SAT" and core is not None:
        nominal = nominal_solve_result(core, r_problem)
        original_status = RP._solve_status
        with unittest.mock.patch.object(RP, "_solve_status",
                                        lambda r: RP.SAT_POSTCHECKED if getattr(r, "status", "") == "NOMINAL_SAT" else original_status(r)):
            try:
                plan = RP.build_resolved_plan(r_problem, nominal, {"L": "l", "S": "s", "C": "c", "F": "f", "P": "p"})
                base = 0
                facts = {}
                for buf in r_problem["buffers"]:
                    facts[buf["root"]] = {"versions": 1, "alignment": 64, "arena_id": "arena0",
                                          "base_offset_bytes": base}
                    base += ((buf["bytes"] + 63) // 64) * 64
                plan = RP.resolve_buffer_layout(plan, facts)
                run["steps"]["R"] = {"status": "PLAN_LAYOUT_RESOLVED", "kind": "real(adapter view)"}
                rewritten, matmap = MAT.materialize_plan(tir, plan)
                report = MAT.verify_plan_application(tir, plan, rewritten, matmap)
                run["steps"]["R"] = {"status": "MATERIALIZED", "digest_before": matmap.original_digest,
                                     "digest_after": matmap.rewritten_digest,
                                     "verify": report.get("status"), "verify_code": report.get("code"),
                                     "verify_detail": report.get("detail"), "kind": "real"}
                if report.get("status") == "PASS":
                    rows = MAT.replay_plan_addresses(plan)
                    run["steps"]["R"]["replay_rows"] = len(rows)
                    run["steps"]["R"]["replay_sample"] = rows[:2]
                    try:
                        from tilefoundry_ascend.sync import derive_completion_obligations, verify_completion_coverage
                        obligations = derive_completion_obligations(problem, plan, completion_capabilities={})
                        coverage = verify_completion_coverage(problem, plan, rewritten,
                                                              {"placements": (), "channels": ()}, {})
                        run["steps"]["C"] = {"status": getattr(coverage, "status", None),
                                             "code": getattr(coverage, "code", None),
                                             "required": getattr(coverage, "required", None),
                                             "covered": getattr(coverage, "covered", None),
                                             "kind": "real"}
                    except Exception as error:  # noqa: BLE001
                        run["steps"]["C"] = {"status": "NOT_RUN", "code": getattr(error, "code", type(error).__name__),
                                             "detail": str(error)[:140], "kind": "unknown"}
            except Exception as error:  # noqa: BLE001
                run["steps"]["R"] = {"status": "NOT_RUN", "code": getattr(error, "code", type(error).__name__),
                                      "detail": str(error)[:160], "kind": "unknown"}

    if plan is not None:
        try:
            RP.attach_completion(plan, None)
            run["controls"]["c_missing"] = "UNEXPECTED_PASS"
        except Exception as error:  # noqa: BLE001
            run["controls"]["c_missing"] = getattr(error, "code", type(error).__name__)

    gm_task = next((a.task for a in problem.accesses if not str(a.root).startswith("smem@")), None)
    if gm_task is not None:
        try:
            project_problem(problem, tir, expansion.count, sources, obligation_access_ids={gm_task})
            run["controls"]["filter_obligation_access"] = "UNEXPECTED_PASS"
        except VerticalError as error:
            run["controls"]["filter_obligation_access"] = error.code

    run["device_ready"] = False
    print(json.dumps(run, indent=1, default=str)[:3000])
    assert run["steps"]["L"]["obligations"] == 47
    assert run["steps"]["S"]["sat_scope"] == "NOMINAL_MODEL_ONLY"
    assert "SAT_POSTCHECKED" not in json.dumps(run["steps"]["S"])
    assert run["controls"]["delete_obligation"] == "POC_SOLVER_INPUT_NOT_READY"
    assert run["controls"]["fallback_not_sat"] == "FALLBACK"
    assert run["controls"]["filter_obligation_access"] == "POC_ADAPTER_UNSUPPORTED_ACCESS"
    assert run["device_ready"] is False


def main() -> int:
    test_vertical_run()
    print("PASS test_vertical_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
