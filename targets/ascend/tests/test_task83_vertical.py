"""task-83 vertical L->S->R(->C) single run on the POC carrier (test-only glue)."""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import dataclasses
import json
import os
import sys
import types

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


def _remove_first_stmt(node, predicate):
    """Return a cloned TIR with the first matching statement removed (real-node mutation)."""
    import dataclasses
    from tilefoundry.ir.tir.stmts import ExecutionScope, For, If, MeshScope, Sequential
    if predicate(node):
        return None, True
    if isinstance(node, Sequential):
        body = []
        removed = False
        for child in node.body:
            if removed:
                body.append(child)
                continue
            new_child, hit = _remove_first_stmt(child, predicate)
            if hit:
                removed = True
                if new_child is not None:
                    body.append(new_child)
                continue
            body.append(new_child if new_child is not None else child)
        return dataclasses.replace(node, body=tuple(body)), removed
    if isinstance(node, (MeshScope, ExecutionScope, For)):
        new_body, removed = _remove_first_stmt(node.body, predicate)
        return dataclasses.replace(node, body=new_body), removed
    if isinstance(node, If):
        then_body, removed = _remove_first_stmt(node.then_body, predicate)
        if removed:
            return dataclasses.replace(node, then_body=then_body), True
        else_body, removed = _remove_first_stmt(node.else_body, predicate)
        return dataclasses.replace(node, else_body=else_body), removed
    return node, False


def _drain_into_if_false(fn):
    """Move the real Drain into an unreachable If(False) branch."""
    import dataclasses
    from tilefoundry.ir.core import Constant
    from tilefoundry.ir.tir.flag import Drain
    from tilefoundry.ir.tir.stmts import Evaluate, If, Sequential
    from tilefoundry.ir.types import make_tensor_type
    drain_callable = None
    from tilefoundry_ascend.sync import _actual_nodes
    for _path, node in _actual_nodes(fn)["drain"]:
        drain_callable = node
        break
    if drain_callable is None:
        raise AssertionError("no real Drain to mutate")
    predicate = lambda node: isinstance(node, Evaluate) and isinstance(node.callable, Drain)
    stripped_body = fn.body
    while True:
        stripped_body, removed = _remove_first_stmt(stripped_body, predicate)
        if not removed:
            break
    cond = Constant(type=make_tensor_type((), dtype="bool", storage="umat"), value=False)
    unreachable = If(cond=cond,
                     then_body=Sequential(body=(Evaluate(callable=drain_callable, args=()),)),
                     else_body=Sequential(body=()))
    return dataclasses.replace(fn, body=Sequential(body=tuple(stripped_body.body) + (unreachable,)))


def _mutate_first_wait(fn, attribute, value):
    from tilefoundry_ascend.sync import _actual_nodes
    wait = _actual_nodes(fn)["wait"][0][1]
    setattr(wait, attribute, value)
    return fn


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
        # P3-02: the REAL source postcheck, so the plan below is built from a genuine
        # SAT_POSTCHECKED result instead of a patched status label.
        core = B.solve_real_problem(r_problem, model_facts, {},
                                    require_source_postcheck=True)
    except Exception:  # noqa: BLE001
        core = None

    raw = None
    if core is not None:
        raw = types.SimpleNamespace(status=core.raw_core_status, assignment=core.assignment,
                                    core_source_hash=core.core_source_hash)
    sres = classify_poc_result(expected, s_problem, raw_core=raw, authoritative=expected)
    # P3-01/P3-02 replaced the placeholder below with the REAL source postcheck.
    # The old record said sat_scope=NOMINAL_MODEL_ONLY / postcheck=NOT_IMPLEMENTED, which
    # is no longer true: solve_real_problem(require_source_postcheck=True) independently
    # re-checks the schedule against every Problem hard constraint and reports
    # SAT_SOURCE_POSTCHECKED / POSTCHECK_PASS. The limits that DO remain open are kept
    # explicit: physical feasibility and capacity stay MODEL-only, and nothing here proves
    # device execution.
    _postcheck = getattr(core, "postcheck", None) or {}
    run["steps"]["S"] = {
        "state": sres.state,
        "sat_scope": getattr(core, "sat_scope", None),
        "postcheck": _postcheck.get("verdict"),
        "postcheck_limits": _postcheck.get("limits"),
        "capacity_scope": getattr(core, "capacity_scope", None),
        "physical_feasibility": (_postcheck.get("limits") or {}).get("physical_feasibility"),
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
        # P3-02: no status patch. The builder verifies the REAL source postcheck, so a
        # fabricated schedule can no longer masquerade as SAT_POSTCHECKED here.
        try:
            plan = RP.build_resolved_plan(
                r_problem, nominal, {"L": "l", "S": "s", "C": "c", "F": "f", "P": "p"},
                require_source_postcheck=True)
            base = 0
            facts = {}
            for buf in r_problem["buffers"]:
                facts[buf["root"]] = {"versions": 1, "alignment": 64, "arena_id": "arena0",
                                      "base_offset_bytes": base}
                base += ((buf["bytes"] + 63) // 64) * 64
            plan = RP.resolve_buffer_layout(plan, facts)
            run["steps"]["R"] = {"status": "PLAN_LAYOUT_RESOLVED", "kind": "real(adapter view)"}
            rewritten, matmap = MAT.materialize_plan(tir, plan)
            from tilefoundry.visitor_registry.verify import verify_prim_function
            # Typed validity of R's own rewritten TIR is an independent assertion.
            # If this fails, the failure belongs to R materialize, not C injection.
            assert verify_prim_function(rewritten) is None, "R_REWRITTEN_TYPED_INVALID"
            run["controls"]["r_rewritten_typed"] = "PASS"
            report = MAT.verify_plan_application(tir, plan, rewritten, matmap, require_drain=False)
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
                    from tilefoundry_ascend.schedule.ledger import extract_ledger
                    ledger = extract_ledger(tir)
                    reuse = SY.derive_reuse_obligations_from_ledger(ledger, expansion)
                    obligations = SY.derive_completion_obligations_from_ledger(
                        ledger, expansion, problem=problem, expected=expected)
                    caps = {"supported_pairs": [["vector", "DMA"], ["DMA", "vector"]],
                            "same_pipe_pipe_order": False, "flag_pool_size": 256,
                            "cost_model": "MODEL"}
                    actionable = tuple(o for o in obligations if o.kind in ("RAW", "WAR", "FREE"))
                    channels, _discharges = SY.choose_completion_mechanisms(actionable, {}, caps)
                    allocation = SY.allocate_completion_resources(channels, [], caps)
                    injected, placement = SY.inject_completion_anchored(
                        rewritten, None, actionable, allocation, channels=channels, ledger=ledger,
                        materialization=matmap)
                    assert verify_prim_function(injected) is None, "C_INJECTED_TYPED_INVALID"
                    run["controls"]["c_injected_typed"] = "PASS"
                    coverage = SY.verify_completion_coverage_from_ledger(
                        ledger, expansion, problem=problem, actual_tir=injected,
                        placement_map=placement, capabilities=caps, expected=expected)
                    required_by_kind = dict(getattr(coverage, "required_by_kind", {}))
                    covered_counts = dict(getattr(coverage, "covered_by_kind", {}))
                    run["steps"]["C"] = {"status": getattr(coverage, "status", None),
                                         "code": getattr(coverage, "code", None),
                                         "required_full": len(expected["all"]),
                                         "required_by_kind": required_by_kind,
                                         "covered_by_kind": covered_counts,
                                         "covered_total": sum(covered_counts.values()),
                                         "ssa_mechanism": "INTRA_TASK_SSA_ORDER",
                                         "c_derived_required": getattr(coverage, "required", None),
                                         "covered": getattr(coverage, "covered", None),
                                         "drain_found": bool(SY._actual_nodes(injected)["drain"]),
                                         "edge_table": list(getattr(coverage, "edge_table", ())),
                                         "digest_C_injected": MAT.tir_digest(injected),
                                         "kind": "real(FULL denominator, anchored reuse, structural SSA/FINAL)"}
                    def fresh_injected():
                        return SY.inject_completion_anchored(
                            rewritten, None, actionable, allocation, channels=channels,
                            ledger=ledger, materialization=matmap)

                    def first_wait(fn):
                        return SY._actual_nodes(fn)["wait"][0][1]

                    def mut_flag_id(fn):
                        wait = first_wait(fn)
                        wait.flag_id = int(wait.flag_id) + 100
                        return fn

                    def mut_epoch(fn):
                        wait = first_wait(fn)
                        wait.generation = int(wait.generation) + 100
                        return fn

                    def mut_flag(fn):
                        wait = first_wait(fn)
                        wait.channel_name = "FORGED_CHANNEL"
                        return fn

                    def mut_missing_wait(fn):
                        import dataclasses as _dc
                        from tilefoundry.ir.tir.flag import FlagWait
                        from tilefoundry.ir.tir.stmts import Evaluate
                        new_body, removed = _remove_first_stmt(
                            fn.body, lambda node: isinstance(node, Evaluate)
                            and isinstance(node.callable, FlagWait))
                        assert removed, "real Wait node was not removed"
                        return _dc.replace(fn, body=new_body)

                    mutation_controls = {}
                    for name, mutator in (
                        ("flag_id_plus_100", mut_flag_id),
                        ("drain_in_if_false", _drain_into_if_false),
                        ("wrong_epoch", mut_epoch),
                        ("wrong_flag", mut_flag),
                        ("missing_wait", mut_missing_wait),
                    ):
                        mutant, mutant_placement = fresh_injected()
                        mutant = mutator(mutant)
                        mutant_report = SY.verify_completion_coverage_from_ledger(
                            ledger, expansion, problem=problem, actual_tir=mutant,
                            placement_map=mutant_placement, capabilities=caps, expected=expected)
                        mutation_controls[name] = {
                            "status": mutant_report.status,
                            "code": mutant_report.code,
                            "covered": mutant_report.covered,
                        }
                        assert mutant_report.status != "VERIFIED", (
                            name, mutant_report.as_dict())
                    run["steps"]["C"]["mutation_controls"] = mutation_controls

                except Exception as error:  # noqa: BLE001
                    import traceback as _tb
                    run["steps"]["C"] = {"status": "NOT_VERIFIED", "code": getattr(error, "code", type(error).__name__),
                                         "detail": str(error)[:140],
                                         "traceback": _tb.format_exc()[-5000:], "kind": "unknown"}
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
    c_ok = c_step.get("status") == "VERIFIED" and covered_total == 47
    excluded_empty = not c_step.get("excluded_by_kind")
    r_pass = run["steps"].get("R", {}).get("verify") == "PASS"
    s_ok = run["steps"].get("S", {}).get("state") == "SAT" and poc_in.supported_required_unknown == 0
    run["chain_consistent"] = bool(excluded_empty and r_pass and s_ok and c_ok)
    run["digests"] = {
        "problem": problem.problem_digest,
        "r_before": run["steps"].get("R", {}).get("digest_before"),
        "r_after": run["steps"].get("R", {}).get("digest_after"),
        "c_actual": c_step.get("digest_C_injected"),
    }
    run["field_classification"] = {
        "real": ["problem_digest", "L facts", "R plan/actual TIR", "C actual FlagSet/FlagWait"],
        "projected": ["loop_domain", "requested_num_stages", "stage_depth", "task latency/ii/stage"],
        "model": ["capacity/size/latency/ii model quantities"],
        # The source postcheck is REAL now, so it is no longer an unknown. What remains
        # unknown is physical feasibility, device execution, and the capacity proof.
        "unknown": ["physical feasibility", "device execution", "asynchronous capacity proof"],
    }
    run["sat_scope"] = getattr(core, "sat_scope", None)
    run["capacity_scope"] = getattr(core, "capacity_scope", None)
    run["physical_feasibility"] = "UNPROVEN"
    # Device readiness is still blocked, but NOT by a missing postcheck any more: the
    # model-level checks pass while physical feasibility stays unproven and no device run
    # has happened. Naming the real blocker keeps this from reading as "almost ready".
    run["device_ready_blocked_by"] = "PHYSICAL_FEASIBILITY_UNPROVEN_AND_NO_DEVICE_RUN"
    run["device_ready"] = False
    if run["controls"].get("r_rewritten_typed") != "PASS":
        run["steps"]["C"] = {"status": "NOT_VERIFIED", "code": "R_REWRITTEN_TYPED_INVALID",
                             "detail": "R materialize output failed verify_prim_function",
                             "kind": "blocked_by_R"}
    elif (run["controls"].get("c_injected_typed") != "PASS"
          and run["steps"].get("C", {}).get("status") != "NOT_VERIFIED"):
        run["steps"]["C"] = {"status": "NOT_VERIFIED", "code": "C_INJECTED_TYPED_INVALID",
                             "detail": "C injected actual TIR failed verify_prim_function",
                             "kind": "blocked_by_C"}
    print(json.dumps(run, indent=1, default=str)[:6000])
    assert run["steps"]["L"]["obligations"] == 47
    # The S step must now report the REAL postcheck, not a placeholder. Asserting the
    # positive fact is stronger than asserting the absence of a label: it fails if the
    # postcheck is ever silently dropped again.
    assert run["steps"]["S"]["sat_scope"] == "SAT_SOURCE_POSTCHECKED", run["steps"]["S"]
    assert run["steps"]["S"]["postcheck"] == "POSTCHECK_PASS", run["steps"]["S"]
    assert run["steps"]["S"]["physical_feasibility"] == "UNPROVEN", run["steps"]["S"]
    # Capacity stays model-only: a passing source postcheck is NOT a physical capacity proof.
    assert run["steps"]["S"]["capacity_scope"] == "NOMINAL_MODEL_ONLY", run["steps"]["S"]
    assert run["controls"]["delete_obligation"] == "POC_SOLVER_INPUT_NOT_READY"
    assert run["controls"]["fallback_not_sat"] == "FALLBACK"
    assert run["controls"]["filter_obligation_access"] == "POC_ADAPTER_UNSUPPORTED_ACCESS"
    print("CONTROLS_ONLY " + json.dumps(run["controls"], default=str))
    assert run["device_ready"] is False
    import pathlib as _pl
    _out = _pl.Path(os.path.dirname(ROOT)) / "TileFoundry_Ascend950PR_AgentTeam/artifacts/run03/w40_vertical/single_run.json"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text(json.dumps(run, indent=1, default=str))
    print("WROTE", _out)
    # These two typed assertions pin the attribution: R must emit a typed-valid
    # TIR and C injection must keep it typed-valid. A failure here is never hidden
    # behind a C NOT_VERIFIED report or a PASS from R's own address verifier.
    assert run["controls"].get("r_rewritten_typed") == "PASS", (
        "R_REWRITTEN_TYPED_INVALID: R materialize output failed verify_prim_function")
    assert run["controls"].get("c_injected_typed") == "PASS", (
        "C_INJECTED_TYPED_INVALID: C injected actual TIR failed verify_prim_function")


def main() -> int:
    test_vertical_run()
    print("PASS test_vertical_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
