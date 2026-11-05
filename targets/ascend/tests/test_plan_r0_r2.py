"""R0-R2 tests for resolved-plan/1 and actual-TIR materialization (CPU, no device).

Run with the CPU interpreter plus the repo on PYTHONPATH:
    PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_plan_r0_r2.py
or via pytest in an interpreter that has pytest.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, Sequential
from tilefoundry.ir.types import Layout, PointerType, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind

from tilefoundry_ascend.plan import materialize as MAT
from tilefoundry_ascend.plan import resolved_plan as RP

ROWS, D = 4, 8
HALF_BYTES = D * 4            # 32 bytes: one half of the arena
ALLOC_BYTES = 2 * HALF_BYTES  # 64 bytes after materialization (padded from 32)
PROBLEM_DIGEST = "sha256:" + "1" * 64


def _scalar(dtype):
    return make_tensor_type((), dtype=dtype, storage="umat")


def _view(offset, tensor_type):
    pointer = PointerType(tensor_type.dtype, tensor_type.storage)
    return Call(type=tensor_type,
                target=TensorView(storage=tensor_type.storage, dtype=str(tensor_type.dtype),
                                  layout=Layout((D,)), shape=(D,)),
                args=(Constant(type=pointer, value=int(offset)),))


def make_tir_fixture():
    """A REAL typed TIR module: alloc(8 f32 smem) + for i in [0,4): two Copy sites."""
    arena_type = make_tensor_type((D,), dtype="f32", storage=StorageKind.SMEM)
    arena = Var(name="arena", type=arena_type)
    alloc = LetStmt(var=arena,
                    value=Call(type=arena_type, target=AllocTensor(tensor_type=arena_type),
                               args=()))
    loop_var = Var(name="i", type=_scalar("i32"))
    zero = Constant(type=_scalar("i32"), value=0)
    stop = Constant(type=_scalar("i32"), value=ROWS)
    step = Constant(type=_scalar("i32"), value=1)
    copy1 = Evaluate(callable=Copy(),
                     args=(_view(0, arena_type), _view(HALF_BYTES, arena_type)))
    copy2 = Evaluate(callable=Copy(),
                     args=(_view(HALF_BYTES, arena_type), _view(0, arena_type)))
    body = Sequential(body=(alloc, For(induction_var=loop_var, start=zero, stop=stop,
                                       step=step, body=Sequential(body=(copy1, copy2)))))
    return PrimFunction(name="serial_fixture", params=(), body=body, output_count=0)


def make_problem():
    access = lambda aid, action, stmt, mode, offset: {
        "access_id": aid, "task": "main_vf", "action_id": action, "source_stmt": stmt,
        "root": "arena", "space": "ascend.ub", "mode": mode,
        "relative_byte_offset": offset, "byte_footprint": HALF_BYTES,
        "logical_index": RP.V("i"),
    }
    return {
        "problem_digest": PROBLEM_DIGEST, "source_digest": "sha256:" + "2" * 64,
        "ledger_sha": "sha256:" + "3" * 64, "loop_id": "row_pipeline",
        "loop_domain": {"start": 0, "stop": ROWS, "step": 1},
        "requested_num_stages": 1, "ii": 1, "latency": 1,
        "tasks": [{"stable_id": "main_vf", "action_ids": ["a0", "a1"],
                   "pipe": "vector", "resource": "vector", "source": "main_compute"}],
        "buffers": [{"root": "arena", "source_decl": "arena", "owner": "row_pipeline",
                     "space": "ascend.ub", "dtype": "f32", "shape": [D * 2],
                     "bytes": ALLOC_BYTES, "alignment": 64, "versions": 1, "fixed": True}],
        "accesses": [
            access("acc-c1-src", "a0", "s1/for/s0", "read", 0),
            access("acc-c1-dst", "a0", "s1/for/s0", "write", HALF_BYTES),
            access("acc-c2-src", "a1", "s1/for/s1", "read", HALF_BYTES),
            access("acc-c2-dst", "a1", "s1/for/s1", "write", 0),
        ],
    }


def make_solve(status=RP.SAT_POSTCHECKED, schedule=None):
    return {"status": status, "core_digest": "sha256:" + "4" * 64,
            "result_digest": "sha256:" + "5" * 64,
            "schedule": schedule if schedule is not None else {
                "main_vf": {"stage": 0, "order": 0, "start_time": 0, "iteration_shift": 0}}}


SOURCES = {"L": "sha256:" + "a" * 64, "S": "sha256:" + "b" * 64,
           "C": "sha256:" + "c" * 64, "F": "sha256:" + "d" * 64,
           "P": "sha256:" + "e" * 64}
FACTS = {"arena": {"versions": 1, "alignment": 64, "arena_id": "arena0",
                   "base_offset_bytes": 0}}


def build_ready():
    plan = RP.build_resolved_plan(make_problem(), make_solve(), SOURCES)
    return RP.resolve_buffer_layout(plan, FACTS)


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


# ------------------------------------------------------------------ R0 / R1
def test_r0_schema_is_strict_and_starts_skeleton_not_applied():
    plan = build_ready()
    assert plan.identity.schema == "resolved-plan/1"
    assert plan.state.planning_status == RP.SAT_POSTCHECKED
    assert plan.state.application_status == RP.STATE_SKELETON
    assert plan.state.completion_status == RP.COMPLETION_PENDING
    assert plan.loop.application_mode == "SERIAL" and plan.loop.static
    for task in plan.tasks:
        assert task.stage is not None and task.order is not None and task.start_time
    for access in plan.accesses:
        assert access.action_id and access.action_id != RP.NA
        assert access.relative_byte_offset != 0, "offset must be typed, never a zero filler"
    buffer = plan.buffers[0]
    assert buffer.version_stride_bytes == 64 and buffer.allocation_bytes == 64
    # serialisation is schema-tagged and JSON-safe
    assert plan.to_dict()["schema"] == "resolved-plan/1"


def test_r1_fallback_or_unknown_solve_is_refused():
    expect_code("PLAN_REQUIRES_SAT",
                lambda: RP.build_resolved_plan(make_problem(), make_solve("fallback"), SOURCES))
    expect_code("PLAN_REQUIRES_SAT",
                lambda: RP.build_resolved_plan(make_problem(), make_solve("unknown"), SOURCES))


def test_r1_missing_or_foreign_ids_are_source_mismatch():
    expect_code("PLAN_SOURCE_MISMATCH",
                lambda: RP.build_resolved_plan(make_problem(), make_solve(schedule={}),
                                               SOURCES))
    problem = make_problem()
    problem["accesses"][0]["action_id"] = "not-an-action"
    expect_code("PLAN_SOURCE_MISMATCH",
                lambda: RP.build_resolved_plan(problem, make_solve(), SOURCES))
    expect_code("PLAN_SOURCE_MISMATCH",
                lambda: RP.build_resolved_plan(make_problem(), make_solve(),
                                               {**SOURCES, "C": ""}))


def test_r1_verify_sources_rejects_a_drifted_problem():
    plan = build_ready()
    other = make_problem()
    other["problem_digest"] = "sha256:" + "9" * 64
    expect_code("PLAN_SOURCE_MISMATCH", lambda: RP.verify_plan_sources(plan, other, SOURCES))


# --------------------------------------------------------------- R0 layout
def test_r0_fixed_version_cannot_be_rebudgeted():
    plan = RP.build_resolved_plan(make_problem(), make_solve(), SOURCES)
    expect_code("FIXED_VERSION_MISMATCH",
                lambda: RP.resolve_buffer_layout(
                    plan, {"arena": {"versions": 2, "alignment": 64,
                                     "arena_id": "arena0", "base_offset_bytes": 0}}))


def test_r0_alignment_and_overlap_are_named_refusals():
    plan = RP.build_resolved_plan(make_problem(), make_solve(), SOURCES)
    expect_code("ALIGNMENT_INVALID",
                lambda: RP.resolve_buffer_layout(
                    plan, {"arena": {"versions": 1, "alignment": 48,
                                     "arena_id": "arena0", "base_offset_bytes": 0}}))
    # Two buffers in one arena whose allocations overlap.
    two = dataclasses.replace(plan, buffers=plan.buffers + (
        dataclasses.replace(plan.buffers[0], root="arena_b", shape=(D * 2,)),))
    expect_code("PLAN_ADDRESS_OVERLAP",
                lambda: RP.resolve_buffer_layout(two, {
                    "arena": {"versions": 1, "alignment": 64, "arena_id": "a",
                              "base_offset_bytes": 0},
                    "arena_b": {"versions": 1, "alignment": 64, "arena_id": "a",
                                "base_offset_bytes": 32}}))


def test_r0_unknown_offset_is_not_zero():
    plan = build_ready()
    unknown = dataclasses.replace(plan, accesses=(
        dataclasses.replace(plan.accesses[0],
                            relative_byte_offset=RP.OFFSET_UNKNOWN),) + plan.accesses[1:])
    env = {"i": 0, "k": 0, "slot": 0}
    expect_code("PLAN_ADDRESS_OFFSET_UNKNOWN",
                lambda: RP.address_range(unknown, unknown.accesses[0], env))


# ---------------------------------------------------- R1 ordinal/generation
def test_generation_token_is_the_full_execution_ordinal_not_slot_or_tile():
    plan = build_ready()
    assert RP.generation_token(plan.loop, 3) == 3
    assert RP.generation_token(plan.loop, 3, supplied=3) == 3
    expect_code("GENERATION_MAPPING_INVALID",
                lambda: RP.generation_token(plan.loop, 3, supplied=0))          # slot 0
    expect_code("GENERATION_MAPPING_INVALID",
                lambda: RP.generation_token(plan.loop, 3, supplied=1))          # recompute min(tile)
    expect_code("LOOP_ITERATION_OUT_OF_DOMAIN",
                lambda: RP.execution_ordinal(plan.loop, ROWS))


def test_address_replay_covers_every_iteration_and_range():
    plan = build_ready()
    rows = MAT.replay_plan_addresses(plan)
    assert len(rows) == ROWS * len(plan.accesses)
    for row in rows:
        assert 0 <= row["start"] < row["end"] <= ALLOC_BYTES
        assert row["generation"] == row["k"]


# --------------------------------------------------------- R2 actual apply
def test_r2_positive_materialization_changes_real_tir_and_verifies():
    plan = build_ready()
    original = make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    assert rewritten is not original
    assert MAT.tir_digest(rewritten) != MAT.tir_digest(original)
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "PASS", report
    assert report["allocations_applied"] == 1
    assert report["controls_applied"] == ROWS
    assert report["applied_accesses"] == 2 * ROWS and report["expected_applications"] == 8
    # actual node inspection: the original arena alloc was widened (8 -> 16 elements).
    allocs = [s for s in MAT.index_statements(rewritten).values() if isinstance(s, LetStmt)]
    shapes = [tuple(s.var.type.shape) for s in allocs]
    assert (D * 2,) in shapes, shapes
    # the rewritten copies carry typed TensorView operands with Constant byte addresses.
    copies = [s for s in MAT.index_statements(rewritten).values()
              if isinstance(s, Evaluate) and isinstance(s.callable, Copy)]
    assert len(copies) == 2 * ROWS
    offsets = {arg.args[0].value for copy in copies for arg in copy.args}
    assert offsets == {0, HALF_BYTES}


def test_r2_sidecar_only_is_plan_not_applied():
    plan = build_ready()
    original = make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan, sidecar_only=True)
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "FAIL" and report["code"] == "PLAN_NOT_APPLIED", report


def test_r2_missing_one_store_is_application_coverage_mismatch():
    plan = build_ready()
    original = make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(
        original, plan, skip_access_ids={"acc-c2-dst"})
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "FAIL"
    assert report["code"] == "APPLICATION_COVERAGE_MISMATCH", report


def test_r2_stale_map_is_materialization_map_mismatch():
    plan = build_ready()
    original = make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    forged = dataclasses.replace(
        materialization,
        entries=tuple({**entry, "new_fingerprint": "0" * 64}
                      for entry in materialization.entries))
    report = MAT.verify_plan_application(original, plan, rewritten, forged)
    assert report["status"] == "FAIL"
    assert report["code"] == "MATERIALIZATION_MAP_MISMATCH", report


def test_r2_non_serial_or_multiversion_is_refused():
    plan = build_ready()
    serial_single = plan
    # R3 owns V2; R2's guard still refuses >2 versions and any non-SERIAL mode.
    multi = dataclasses.replace(plan, buffers=(
        dataclasses.replace(plan.buffers[0], slot_count=3),))
    expect_code("PLAN_MATERIALIZATION_UNSUPPORTED",
                lambda: MAT.materialize_plan(make_tir_fixture(), multi))
    dynamic = dataclasses.replace(plan, loop=dataclasses.replace(
        plan.loop, application_mode="PIPELINED"))
    expect_code("PLAN_MATERIALIZATION_UNSUPPORTED",
                lambda: MAT.materialize_plan(make_tir_fixture(), dynamic))
    assert serial_single.loop.application_mode == "SERIAL"


def test_r5_completion_hook_refuses_unverified_C():
    plan = build_ready()
    expect_code("COMPLETION_NOT_VERIFIED",
                lambda: RP.attach_completion(plan, {"verified": False, "digest": None}))
    done = RP.attach_completion(plan, {"verified": True, "digest": "sha256:" + "f" * 64})
    assert done.state.completion_status == RP.COMPLETION_VERIFIED
    assert all(s.status == RP.COMPLETION_VERIFIED for s in done.syncs)


def test_apply_plan_marks_applied_only_after_verification():
    plan = build_ready()
    original = make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "PASS"
    applied = MAT.apply_plan(plan, materialization)
    assert applied.state.application_status == RP.STATE_APPLIED


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
