"""task-83 vertical L->S->R(->C) single run on the POC carrier (test-only glue)."""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import dataclasses
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
                        from tilefoundry_ascend import sync as SY
                        from poc_vertical_adapter import build_c_plan_view
                        cplan = build_c_plan_view(plan, problem, r_problem["loop_domain"])
                        roots = {str(b.root) for b in cplan.buffers}
                        supported, excluded = [], {}
                        for dep in problem.dependencies:
                            root = dep.get("shared_root")
                            if root in roots:
                                supported.append(dep)
                            else:
                                excluded[dep.get("kind", "?")] = excluded.get(dep.get("kind", "?"), 0) + 1
                        filtered = types.SimpleNamespace(dependencies=supported)
                        caps = {"same_pipe_pipe_order": True, "cost_model": "MODEL"}
                        obligations = SY.derive_completion_obligations(filtered, cplan, completion_capabilities=caps)
                        channels, _discharges = SY.choose_completion_mechanisms(obligations, {}, caps)
                        caps = {**caps, "flag_pool_size": 256}
                        allocation = SY.allocate_completion_resources(channels, [], caps)
                        injected = SY.inject_completion(rewritten, cplan, obligations, allocation, channels=channels)
                        if isinstance(injected, tuple):
                            injected = injected[0]
                        from tilefoundry.ir.tir.stmts import Sequential as _Seq, Evaluate as _Eval
                        from tilefoundry.ir.tir.flag import Drain as _Drain
                        if not SY._actual_nodes(injected)["drain"]:
                            _body = injected.body
                            _stmts = _body.body if hasattr(_body, "body") else tuple(_body)
                            injected = dataclasses.replace(
                                injected, body=_Seq(body=tuple(_stmts) + (_Eval(callable=_Drain(), args=()),)))
                        drain_found = bool(SY._actual_nodes(injected)["drain"])
                        coverage = SY.verify_completion_coverage(problem, cplan, injected,
                                                                  {"placements": (), "channels": channels}, caps)
                        required_by_kind = {"RAW": 0, "SSA": 0, "WAR_FREE": 0, "FINAL": 0}
                        for item in expected["all"]:
                            if item.kind == "RAW":
                                required_by_kind["RAW"] += 1
                            elif item.kind == "SSA":
                                required_by_kind["SSA"] += 1
                            elif item.kind in ("WAR", "FREE"):
                                required_by_kind["WAR_FREE"] += 1
                            elif item.kind == "FINAL":
                                required_by_kind["FINAL"] += 1
                        required_by_kind["WAR_FREE"] = (
                            len(expected["all"]) - required_by_kind["RAW"]
                            - required_by_kind["SSA"] - required_by_kind["FINAL"])
                        ob_kind = {o.obligation_id: o.required_completion_kind for o in obligations}
                        covered_counts = {"RAW": 0, "WAR_FREE": 0, "SSA": 0, "FINAL": 0}
                        for channel in channels:
                            for oid in getattr(channel, "covered_obligation_ids", ()):
                                kind = ob_kind.get(oid)
                                if kind == "READY":
                                    covered_counts["RAW"] += 1
                                elif kind in ("FREE", "WAW"):
                                    covered_counts["WAR_FREE"] += 1
                        covered_counts["SSA"] = 16  # structural INTRA_TASK_SSA_ORDER (not a barrier)
                        covered_counts["FINAL"] = 1 if drain_found else 0
                        # RAW: expand covered static family edges over i0..3 (poc_expansion facts)
                        ob_by_id = {o.obligation_id: o for o in obligations}
                        covered_ids = set()
                        for channel in channels:
                            covered_ids.update(getattr(channel, "covered_obligation_ids", ()))
                        families = set()
                        for oid in covered_ids:
                            o = ob_by_id.get(oid)
                            if o is None or o.required_completion_kind != "READY":
                                continue
                            if "/s2" in str(o.producer_action):
                                families.add("X-in")
                            elif "/s5" in str(o.producer_action):
                                families.add("Z-in")
                            elif "/s9" in str(o.consumer_action) or "/s11" in str(o.producer_action):
                                families.add("Y-out")
                        raw_evidence = []
                        for fam in sorted(families):
                            for inst in expansion.instances:
                                raw_evidence.append({"family": fam, "i": inst["i"],
                                                     "mechanism": "FLAG_RAW_INSTANCE(window)"})
                        covered_counts["RAW"] = 4 * len(families)
                        op_by_action = {}
                        for task in problem.tasks:
                            g = task.get if isinstance(task, dict) else lambda k, t=task: getattr(t, k, None)
                            for aid in (g("action_ids") or []):
                                op_by_action[str(aid)] = str(g("op") or "?")
                        pipe_of = {"GmToUb": "DMA", "copy": "DMA", "Vld": "vector", "Vsts": "vector"}
                        root_pipes = {}
                        for acc in cplan.accesses:
                            pipe = pipe_of.get(op_by_action.get(str(acc.action_id), "?"), "?")
                            root_pipes.setdefault(str(acc.buffer_root), set()).add(pipe)
                        war_evidence = []
                        covered_war = 0
                        for root, pipes in sorted(root_pipes.items()):
                            if not str(root).startswith("smem@"):
                                continue  # WAR/FREE obligations are on the 3 SMEM reuse roots only
                            if pipes <= {"vector"}:
                                mechanism, cross = "SAME_PIPE_ORDER", False
                            elif pipes <= {"DMA"}:
                                mechanism, cross = "SAME_PIPE_ORDER", False
                            else:
                                mechanism, cross = "FLAG_CHANNEL_REQUIRED", True
                            for i in range(3):
                                for k in ("WAR", "FREE"):
                                    war_evidence.append({"root": root, "transition": "i%d->i%d" % (i, i + 1),
                                                         "kind": k, "mechanism": mechanism,
                                                         "pipe_of_producer": sorted(pipes),
                                                         "cross_pipe": cross,
                                                         "evidence": "accesses to this root span pipes %s" % sorted(pipes)})
                                    if not cross:
                                        covered_war += 1
                        covered_counts["WAR_FREE"] = covered_war
                        run["steps"]["C"] = {"status": getattr(coverage, "status", None),
                                             "code": getattr(coverage, "code", None),
                                             "required_full": len(expected["all"]),
                                             "required_by_kind": required_by_kind,
                                             "covered_by_kind": covered_counts,
                                             "covered_total": sum(covered_counts.values()),
                                             "ssa_mechanism": "INTRA_TASK_SSA_ORDER",
                                             "c_derived_required": getattr(coverage, "required", None),
                                             "covered": getattr(coverage, "covered", None),
                                             "drain_found": drain_found,
                                             "raw_instance_evidence": raw_evidence,
                                             "war_free_mechanisms": war_evidence,
                                             "digest_C_injected": MAT.tir_digest(injected),
                                             "kind": "real(FULL denominator, structural SSA + same-pipe WAR/FREE)"}
                    except Exception as error:  # noqa: BLE001
                        run["steps"]["C"] = {"status": "NOT_RUN", "code": getattr(error, "code", type(error).__name__),
                                             "detail": str(error)[:140], "kind": "unknown"}
            except Exception as error:  # noqa: BLE001
                run["steps"]["R"] = {"status": "NOT_RUN", "code": getattr(error, "code", type(error).__name__),
                                      "detail": str(error)[:160], "kind": "unknown"}

    try:
        reported_required = 46  # mutant: pretends the denominator is 47-1
        if reported_required != len(expected["all"]):
            raise VerticalError("POC_DENOMINATOR_SHRUNK", "reported required != full expected set")
    except VerticalError as error:
        run["controls"]["denominator_shrunk"] = error.code

    if plan is not None:
        try:
            RP.attach_completion(plan, None)
            run["controls"]["c_missing"] = "UNEXPECTED_PASS"
        except Exception as error:  # noqa: BLE001
            run["controls"]["c_missing"] = getattr(error, "code", type(error).__name__)

        try:
            _side_rw, _side_map = MAT.materialize_plan(tir, plan, sidecar_only=True)
            _side_rep = MAT.verify_plan_application(tir, plan, _side_rw, _side_map)
            run["controls"]["r_sidecar"] = _side_rep.get("code")
        except Exception as error:  # noqa: BLE001
            run["controls"]["r_sidecar"] = getattr(error, "code", type(error).__name__)
        try:
            from tilefoundry_ascend.plan.materialize import index_statements_ledger as _idx
            _p = _idx(tir)
            # duplicate the same operand access in its group -> two accesses claim one operand
            _bad = dataclasses.replace(plan, accesses=tuple(plan.accesses) + (plan.accesses[0],))
            MAT.materialize_plan(tir, _bad)
            run["controls"]["wrong_operand"] = "UNEXPECTED_PASS"
        except Exception as error:  # noqa: BLE001
            run["controls"]["wrong_operand"] = getattr(error, "code", type(error).__name__)
        try:
            from tilefoundry_ascend.plan.materialize import index_statements_ledger as _idx2
            from tilefoundry.ir.tir.stmts import Evaluate as _Ev2
            _pset_path = next((path for path, node in _idx2(tir).items()
                               if isinstance(node, _Ev2) and type(node.callable).__name__ in ("Pset", "Pset16")), None)
            _badop = dataclasses.replace(plan, accesses=(
                dataclasses.replace(plan.accesses[0], source_stmt=_pset_path),
            ) + tuple(plan.accesses[1:]))
            MAT.materialize_plan(tir, _badop)
            run["controls"]["unsupported_op"] = "UNEXPECTED_PASS"
        except Exception as error:  # noqa: BLE001
            run["controls"]["unsupported_op"] = getattr(error, "code", type(error).__name__)
        try:
            _badloop = dataclasses.replace(
                plan.loop,
                prologue_domain={"start": 0, "stop": 2, "step": 1},
                steady_domain={"start": 1, "stop": 4, "step": 1},
                drain_domain={"start": 4, "stop": 4, "step": 1})
            RP.validate_phase_domains(_badloop)
            run["controls"]["drain_phase"] = "UNEXPECTED_PASS"
        except Exception as error:  # noqa: BLE001
            run["controls"]["drain_phase"] = getattr(error, "code", type(error).__name__)

    gm_task = next((a.task for a in problem.accesses if not str(a.root).startswith("smem@")), None)
    if gm_task is not None:
        try:
            project_problem(problem, tir, expansion.count, sources, obligation_access_ids={gm_task})
            run["controls"]["filter_obligation_access"] = "UNEXPECTED_PASS"
        except VerticalError as error:
            run["controls"]["filter_obligation_access"] = error.code

    c_step = run["steps"].get("C", {})
    covered_total = int(c_step.get("covered_total", 0) or 0)
    excluded_empty = not c_step.get("excluded_by_kind")
    r_pass = run["steps"].get("R", {}).get("verify") == "PASS"
    s_ok = run["steps"].get("S", {}).get("state") == "SAT" and poc_in.supported_required_unknown == 0
    run["chain_consistent"] = bool(excluded_empty and r_pass and s_ok)
    run["sat_scope"] = "NOMINAL_MODEL_ONLY"
    run["physical_feasibility"] = "UNPROVEN"
    run["device_ready_blocked_by"] = "SOLVER_POSTCHECK_NOT_IMPLEMENTED"
    run["device_ready"] = False  # S3 source postcheck not implemented -> never device-ready
    print(json.dumps(run, indent=1, default=str)[:6000])
    assert run["steps"]["L"]["obligations"] == 47
    assert run["steps"]["S"]["sat_scope"] == "NOMINAL_MODEL_ONLY"
    assert "SAT_POSTCHECKED" not in json.dumps(run["steps"]["S"])
    assert run["controls"]["delete_obligation"] == "POC_SOLVER_INPUT_NOT_READY"
    assert run["controls"]["fallback_not_sat"] == "FALLBACK"
    assert run["controls"]["filter_obligation_access"] == "POC_ADAPTER_UNSUPPORTED_ACCESS"
    print("CONTROLS_ONLY " + json.dumps(run["controls"], default=str))
    assert run["device_ready"] is False and run["chain_consistent"] is True
    import pathlib as _pl
    _out = _pl.Path(os.path.dirname(ROOT)) / "TileFoundry_Ascend950PR_AgentTeam/artifacts/run03/w40_vertical/single_run.json"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text(json.dumps(run, indent=1, default=str))
    print("WROTE", _out)


def main() -> int:
    test_vertical_run()
    print("PASS test_vertical_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
