"""R3 tests: V2 ring, slot(k)=k%V, full generation ordinal, >2V/static guard.

Run:
  PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_plan_r3.py
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
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, Sequential
from tilefoundry.ir.types import Layout, PointerType, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind

sys.path.insert(0, str(HERE.parent))

from tilefoundry_ascend.plan import materialize as MAT
from tilefoundry_ascend.plan import resolved_plan as RP
import solve_fixture

ROWS, D = 5, 8
HALF_BYTES = D * 4
STRIDE = 64
ALLOC_BYTES = 2 * STRIDE
PROBLEM_DIGEST = "sha256:" + "a" * 64
SOURCES = {k: "sha256:" + v * 64 for k, v in (("L", "a"), ("S", "b"), ("C", "c"),
                                               ("F", "d"), ("P", "e"))}
FACTS = {"arena": {"versions": 2, "alignment": 64, "arena_id": "arena0",
                   "base_offset_bytes": 0}}


def _scalar(dtype):
    return make_tensor_type((), dtype=dtype, storage="umat")


def _view(offset, tensor_type):
    pointer = PointerType(tensor_type.dtype, tensor_type.storage)
    return Call(type=tensor_type,
                target=TensorView(storage=tensor_type.storage, dtype=str(tensor_type.dtype),
                                  layout=Layout((D,)), shape=(D,)),
                args=(Constant(type=pointer, value=int(offset)),))


def make_tir_fixture(rows=ROWS):
    arena_type = make_tensor_type((D,), dtype="f32", storage=StorageKind.SMEM)
    arena = Var(name="arena", type=arena_type)
    alloc = LetStmt(var=arena,
                    value=Call(type=arena_type, target=AllocTensor(tensor_type=arena_type),
                               args=()))
    iv = Var(name="i", type=_scalar("i32"))
    zero = Constant(type=_scalar("i32"), value=0)
    stop = Constant(type=_scalar("i32"), value=rows)
    step = Constant(type=_scalar("i32"), value=1)
    c1 = Evaluate(callable=Copy(), args=(_view(0, arena_type), _view(HALF_BYTES, arena_type)))
    c2 = Evaluate(callable=Copy(), args=(_view(HALF_BYTES, arena_type), _view(0, arena_type)))
    body = Sequential(body=(alloc, For(induction_var=iv, start=zero, stop=stop, step=step,
                                       body=Sequential(body=(c1, c2)))))
    return PrimFunction(name="ring_v2_fixture", params=(), body=body, output_count=0)


def make_problem(rows=ROWS):
    def access(aid, action, stmt, mode, offset):
        return {"access_id": aid, "task": "main_vf", "action_id": action,
                "source_stmt": stmt, "root": "arena", "space": "ascend.ub",
                "mode": mode, "relative_byte_offset": offset,
                "byte_footprint": HALF_BYTES, "logical_index": RP.V("i")}
    return {
        "problem_digest": PROBLEM_DIGEST, "source_digest": "sha256:" + "b" * 64,
        "ledger_sha": "sha256:" + "c" * 64, "loop_id": "row_pipeline",
        "loop_domain": {"start": 0, "stop": rows, "step": 1},
        "requested_num_stages": 1, "ii": 1, "latency": 1,
        "tasks": [{"stable_id": "main_vf", "action_ids": ["a0", "a1"],
                   "pipe": "vector", "resource": "vector", "source": "main_compute"}],
        "buffers": [{"root": "arena", "source_decl": "arena", "owner": "row_pipeline",
                     "space": "ascend.ub", "dtype": "f32", "shape": [D * 2],
                     "bytes": STRIDE, "alignment": 64, "versions": 2, "fixed": True}],
        "accesses": [
            access("acc-c1-src", "a0", "s1/for/s0", "read", 0),
            access("acc-c1-dst", "a0", "s1/for/s0", "write", HALF_BYTES),
            access("acc-c2-src", "a1", "s1/for/s1", "read", HALF_BYTES),
            access("acc-c2-dst", "a1", "s1/for/s1", "write", 0),
        ],
    }


def make_solve(rows=ROWS, schedule=None, problem=None):
    """A REAL solve-result/1 shape; the plan builder no longer reads a "schedule"."""
    placed = schedule if schedule is not None else {
        "main_vf": {"stage": 0, "order": 0, "start_time": 0, "iteration_shift": 0}}
    target = problem if problem is not None else make_problem(rows)
    return solve_fixture.solve_result(
        placed, problem_digest=target["problem_digest"], ii=target.get("ii", 1),
        buffers=solve_fixture.fixed_buffer_versions(target))


def build_ready(rows=ROWS, facts=None):
    plan = RP.build_resolved_plan(make_problem(rows), make_solve(rows), SOURCES)
    return RP.resolve_buffer_layout(plan, facts or FACTS)


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def row_for(rows, access_id, k):
    matches = [row for row in rows if row["access_id"] == access_id and row["k"] == k]
    assert len(matches) == 1, (access_id, k, matches)
    return matches[0]


def test_r3_v2_slot_and_generation_full_ordinal():
    plan = build_ready()
    rows = MAT.replay_plan_addresses(plan)
    assert len(rows) == ROWS * len(plan.accesses) == 20
    # V=2 ring: k0 and k2 share slot 0 but have different generation tokens.
    a0 = row_for(rows, "acc-c1-src", 0)
    a2 = row_for(rows, "acc-c1-src", 2)
    assert a0["slot"] == a2["slot"] == 0
    assert a0["generation"] == 0 and a2["generation"] == 2
    assert a0["start"] == a2["start"] == 0  # same physical slot, different generation
    b1 = row_for(rows, "acc-c1-src", 1)
    b3 = row_for(rows, "acc-c1-src", 3)
    assert b1["slot"] == b3["slot"] == 1 and b1["generation"] == 1 and b3["generation"] == 3
    assert b1["start"] == b3["start"] == STRIDE
    # k4 wraps to slot 0 again with generation 4; ranges never leave [0, ALLOC_BYTES)
    for row in rows:
        assert 0 <= row["start"] < row["end"] <= ALLOC_BYTES, row
    assert row_for(rows, "acc-c1-src", 4)["slot"] == 0
    assert row_for(rows, "acc-c1-src", 4)["generation"] == 4


def test_r3_v2_materialization_rewrites_real_alloc_and_addresses():
    plan = build_ready()
    original = make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "PASS", report
    assert report["applied_accesses"] == 2 * ROWS  # 2 write accesses x 5 iterations
    assert report["expected_applications"] == 2 * ROWS
    allocs = [s for s in MAT.index_statements(rewritten).values() if isinstance(s, LetStmt)]
    assert (D * 4,) in [tuple(s.var.type.shape) for s in allocs]  # 128B / 4
    copies = [s for s in MAT.index_statements(rewritten).values()
              if isinstance(s, Evaluate) and isinstance(s.callable, Copy)]
    offsets = {arg.args[0].value for copy in copies for arg in copy.args}
    assert offsets == {0, HALF_BYTES, STRIDE, STRIDE + HALF_BYTES}, offsets


def test_r3_bad_stride_or_overlap_is_named():
    plan = RP.build_resolved_plan(make_problem(), make_solve(), SOURCES)
    expect_code("ALIGNMENT_INVALID",
                lambda: RP.resolve_buffer_layout(plan, {
                    "arena": {"versions": 2, "alignment": 48, "arena_id": "arena0",
                              "base_offset_bytes": 0}}))
    two = dataclasses.replace(plan, buffers=plan.buffers + (
        dataclasses.replace(plan.buffers[0], root="arena_b"),))
    expect_code("PLAN_ADDRESS_OVERLAP",
                lambda: RP.resolve_buffer_layout(two, {
                    "arena": {"versions": 2, "alignment": 64, "arena_id": "a",
                              "base_offset_bytes": 0},
                    "arena_b": {"versions": 2, "alignment": 64, "arena_id": "a",
                                "base_offset_bytes": 64}}))


def test_r3_generation_slot_or_min_tile_substitution_is_rejected():
    plan = build_ready()
    expect_code("GENERATION_MAPPING_INVALID",
                lambda: RP.generation_token(plan.loop, 2, supplied=0))  # slot 0
    expect_code("GENERATION_MAPPING_INVALID",
                lambda: RP.generation_token(plan.loop, 2, supplied=1))  # min(tile)
    bad_access = dataclasses.replace(plan.accesses[0], generation=RP.V("slot"))
    bad = dataclasses.replace(plan, accesses=(bad_access,) + plan.accesses[1:])
    expect_code("GENERATION_MAPPING_INVALID", lambda: MAT.replay_plan_addresses(bad))


def test_r3_less_than_v_plus_one_iterations_cannot_prove_reuse():
    # V=2 needs k0..2 at minimum to hit slot0 twice with different generations;
    # a 2-iteration sample must be refused, not reported as a ring proof.
    short = build_ready(rows=2)
    expect_code("PLAN_REPLAY_INCOMPLETE", lambda: MAT.replay_plan_addresses(short))


def test_r3_gt_two_versions_is_materialization_unsupported():
    plan = build_ready()
    three = dataclasses.replace(plan, buffers=(
        dataclasses.replace(plan.buffers[0], slot_count=3),))
    expect_code("PLAN_MATERIALIZATION_UNSUPPORTED",
                lambda: MAT.materialize_plan(make_tir_fixture(), three))
    dynamic = dataclasses.replace(plan, loop=dataclasses.replace(
        plan.loop, application_mode="PIPELINED"))
    expect_code("PLAN_MATERIALIZATION_UNSUPPORTED",
                lambda: MAT.materialize_plan(make_tir_fixture(), dynamic))


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
