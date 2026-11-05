"""R4/R5 tests: phase domains + stage-shifted access mapping + C attach/X boundary.

Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_plan_r4_r5.py
Static CPU only; no device.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))
sys.path.insert(0, str(REPO / "targets" / "ascend" / "tests"))

from tilefoundry.ir.tir.flag import Drain
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential

from tilefoundry_ascend.plan import materialize as MAT
from tilefoundry_ascend.plan import resolved_plan as RP
from tilefoundry_ascend import sync as C
import test_plan_r3 as R3


def stage_problem():
    problem = R3.make_problem(rows=5)
    problem["stage_depth"] = 2
    # The consumer view is stage-shifted: its ordinal is i+1, not current i/k.
    problem["accesses"][1]["execution_ordinal"] = RP.Add(RP.V("i"), RP.C(1))
    return problem


def build_stage():
    plan = RP.build_resolved_plan(stage_problem(), R3.make_solve(rows=5), R3.SOURCES)
    return RP.resolve_buffer_layout(plan, R3.FACTS)


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def test_r4_phase_domains_and_stage_shift_are_explicit_and_disjoint():
    plan = build_stage()
    loop = plan.loop
    assert loop.stage_depth == 2 and loop.stage_shift == 1
    assert loop.prologue_domain == {"start": 0, "stop": 1, "step": 1}
    assert loop.steady_domain == {"start": 1, "stop": 4, "step": 1}
    assert loop.drain_domain == {"start": 4, "stop": 5, "step": 1}
    assert RP.validate_phase_domains(loop)["valid"] is True
    # non-overlap and in-loop bounds
    assert loop.prologue_domain["stop"] <= loop.steady_domain["start"]
    assert loop.steady_domain["stop"] <= loop.drain_domain["start"]
    assert loop.drain_domain["stop"] <= loop.domain["stop"]


def test_r4_access_k_comes_from_accessplan_not_current_loop_i():
    plan = build_stage()
    rows = MAT.replay_plan_addresses(plan)
    # c1-dst is shifted by +1: at loop k=0 it consumes ordinal 1 / slot 1 / 96..128,
    # NOT the current loop slot 0 / 32..64.
    k0 = [row for row in rows if row["access_id"] == "acc-c1-dst" and row["k"] == 0][0]
    assert k0["generation"] == 1 and k0["slot"] == 1
    assert (k0["start"], k0["end"]) == (96, 128)
    k2 = [row for row in rows if row["access_id"] == "acc-c1-dst" and row["k"] == 2][0]
    assert k2["generation"] == 3 and k2["slot"] == 1


def test_r4_materialization_has_phase_controls_and_terminal_drain():
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "PASS", report
    drains = [stmt for stmt in MAT.index_statements(rewritten).values()
              if isinstance(stmt, Evaluate) and isinstance(stmt.callable, Drain)]
    assert len(drains) == 1
    # top-level phase control nodes: prologue, steady, drain
    top = [stmt for stmt in rewritten.body.body if hasattr(stmt, "then_body")]
    assert len(top) >= 3


def test_r4_missing_drain_is_plan_drain_missing():
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    kept = [stmt for stmt in rewritten.body.body
            if not (isinstance(stmt, Evaluate) and isinstance(stmt.callable, Drain))]
    broken = PrimFunction(name=rewritten.name, params=rewritten.params,
                          body=Sequential(body=tuple(kept)), output_count=rewritten.output_count,
                          target=rewritten.target)
    report = MAT.verify_plan_application(original, plan, broken, materialization)
    assert report["status"] == "FAIL" and report["code"] == "PLAN_DRAIN_MISSING", report


def test_r4_bad_stage_mapping_and_overlap_are_named():
    plan = build_stage()
    bad_access = dataclasses.replace(plan.accesses[1], execution_ordinal=None)
    bad_plan = dataclasses.replace(plan, accesses=(plan.accesses[0], bad_access) + plan.accesses[2:])
    expect_code("PLAN_STAGE_MAPPING_INVALID",
                lambda: MAT.materialize_plan(R3.make_tir_fixture(), bad_plan))
    overlap = dataclasses.replace(plan.loop,
                                  prologue_domain={"start": 0, "stop": 3, "step": 1},
                                  steady_domain={"start": 2, "stop": 4, "step": 1})
    expect_code("PLAN_PHASE_DOMAIN_OVERLAP", lambda: RP.validate_phase_domains(overlap))


def test_r5_attach_requires_real_verified_C_coverage():
    plan = build_stage()
    failed = C.CoverageReport("FAILED", "COMPLETION_NOT_VERIFIED", 1, 1, 0, ("ob-1",), 0)
    expect_code("COMPLETION_NOT_VERIFIED", lambda: RP.attach_completion(plan, failed))
    expect_code("COMPLETION_NOT_VERIFIED", lambda: RP.attach_completion(plan, None))
    verified = C.CoverageReport("VERIFIED", None, 1, 1, 1, (), 7)
    attached = RP.attach_completion(plan, verified)
    assert attached.state.completion_status == RP.COMPLETION_VERIFIED
    assert all(sync.status == RP.COMPLETION_VERIFIED for sync in attached.syncs)


def test_r5_x_interface_is_named_unimplemented_not_fake_success():
    plan = build_stage()
    expect_code("COMPLETION_NOT_VERIFIED", lambda: RP.emit_for_codegen(plan))
    verified = C.CoverageReport("VERIFIED", None, 1, 1, 1, (), 7)
    attached = RP.attach_completion(plan, verified)
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: RP.emit_for_codegen(attached))
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED",
                lambda: RP.emit_for_codegen(attached, {"implemented": True}))


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
