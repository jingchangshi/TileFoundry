"""task-83: vertical L->S(->R/C) single-run harness on the POC carrier.

POC_GLUE_NOT_PRODUCTION: R/S views are adapter projections (see poc_vertical_adapter.py);
real-problem/1 is NOT changed. Honest per-step status is recorded; device_ready is false if any
step is UNKNOWN/not VERIFIED. S SAT (if any) is NOMINAL_MODEL_ONLY, never a physical proof.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import copy
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")


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
    problem = build_problem(ledger, sources, predicate_metadata=module.metadata.get("d0.predicate_provenance"))
    expansion = expand_loop_instances(ledger)
    expected = build_expected_obligations(ledger, problem, expansion)
    s_problem = build_s_problem_document(problem)
    return sources, tir, problem, expansion, expected, s_problem


def test_vertical_single_run():
    from tilefoundry_ascend.solver.g10 import classify_poc_result, consume_poc_expected
    from tilefoundry_ascend.solver import bridge as B
    from poc_vertical_adapter import project_problem, nominal_solve_result

    sources, tir, problem, expansion, expected, s_problem = _l()
    run = {"problem_digest": problem.problem_digest, "steps": {}}

    # L
    run["steps"]["L"] = {"status": "OK", "obligations": len(expected["all"]),
                         "predicate": problem.policy["predicate"]["status"]}

    # S input
    poc_in = consume_poc_expected(expected, s_problem)
    run["steps"]["S_input"] = {"status": poc_in.status, "required": poc_in.required_count,
                               "unknown": poc_in.supported_required_unknown, "code": poc_in.code}

    r_problem, projected = project_problem(problem, tir, expansion.count, sources)
    run["projected_fields"] = projected

    core = None
    model_facts = {"model_id": "ascend-poc-nominal", "model_version": "v1",
                   "memory_groups": [{"id": "group0",
                                      "capacity_bytes": max(1, sum(int(b["bytes"]) for b in r_problem["buffers"])),
                                      "buffers": [b["root"] for b in r_problem["buffers"]],
                                      "cost_model": "MODEL"}],
                   "cost_model": "MODEL"}
    run["steps"]["S_model_facts"] = {"cost_model": "MODEL", "capacity": "sum of planned buffer bytes (nominal)"}
    try:
        core = B.solve_real_problem(r_problem, model_facts, {})
        run["steps"]["S_core"] = {"state": core.state, "raw": core.raw_core_status}
    except Exception as error:  # noqa: BLE001
        run["steps"]["S_core"] = {"state": "UNAVAILABLE", "error": type(error).__name__ + ": " + str(error)[:160]}

    raw = None
    if core is not None:
        raw = types.SimpleNamespace(status=core.raw_core_status, assignment=core.assignment,
                                    core_source_hash=core.core_source_hash)
    sres = classify_poc_result(expected, s_problem, raw_core=raw, authoritative=expected)
    run["steps"]["S"] = {"state": sres.state, "r_usable": sres.r_usable, "code": sres.code,
                         "sat_scope": "NOMINAL_MODEL_ONLY", "core_source_hash": getattr(core, "core_source_hash", None)}

    # G10 controls
    from tilefoundry_ascend.solver.g10 import classify_poc_result as classify
    trimmed = {"all": expected["all"][:-1]}
    assert classify(trimmed, s_problem, raw_core=types.SimpleNamespace(status="sat", assignment={}), authoritative=expected).code == "POC_SOLVER_INPUT_NOT_READY"
    fb = classify(expected, s_problem, raw_core=types.SimpleNamespace(status="fallback", fallback_candidate={}), authoritative=expected)
    assert fb.state == "FALLBACK" and fb.r_usable is False

    # R (only if S gave a nominal SAT)
    run["steps"]["R"] = {"status": "NOT_RUN", "code": "POC_SOLVER_INPUT_NOT_READY"}
    run["steps"]["C"] = {"status": "NOT_RUN", "code": "COMPLETION_NOT_VERIFIED"}
    if sres.state == "SAT" and sres.r_usable and core is not None:
        from tilefoundry_ascend.plan import resolved_plan as RP
        nominal = nominal_solve_result(core, r_problem)
        try:
            plan = RP.build_resolved_plan(r_problem, nominal, {"L": "l", "S": "s", "C": "c", "F": "f", "P": "p"})
            run["steps"]["R"] = {"status": "PLAN_BUILT", "plan_digest": plan.digest if hasattr(plan, "digest") else None}
        except Exception as error:  # noqa: BLE001
            run["steps"]["R"] = {"status": "REFUSED", "code": getattr(error, "code", type(error).__name__),
                                 "detail": str(error)[:160]}

    run["device_ready"] = False
    print(json.dumps(run, indent=1, default=str)[:2500])
    assert run["steps"]["L"]["obligations"] == 47
    assert run["steps"]["S_input"]["status"] == "READY" and run["steps"]["S_input"]["required"] == 47
    assert run["steps"]["S"]["sat_scope"] == "NOMINAL_MODEL_ONLY"
    assert run["device_ready"] is False


def main() -> int:
    test_vertical_single_run()
    print("PASS test_vertical_single_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
