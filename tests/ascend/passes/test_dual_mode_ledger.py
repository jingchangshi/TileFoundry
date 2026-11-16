"""Tests for Mode A 24-bit core solver, Mode B Two-Snapshot IR Diff, and ObjectivePhysicalLedger."""

import json
import pathlib

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.hir.pipeline import PipelineSpec
from tilefoundry.ir.tir.stmts import For
from tilefoundry.ir.visitor import StmtVisitor, walk_prim_function
from tilefoundry.schedule.finalize import finalize
from tilefoundry.vendor.tilelang_z3 import (
    Buffer,
    Capacity,
    Problem,
    Resource,
    Task,
)
from tilefoundry_ascend.ascend.passes import ObjectivePhysicalLedger, ScopeWaterline
from tilefoundry_ascend.ascend.schedule import (
    bitwise_conflict_pairs,
    inspect_tilelang_primfunc,
    inspect_tir_envelope,
    solve_ascend_pipeline,
    solve_loop_with_ledger,
    task_resource_mask,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget, NodeResourceMask

HEADER = (
    "from __future__ import annotations\n\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry_ascend import AscendTarget\n"
    "TOPOS = (Topology('block', 1), Topology('aic', 1), "
    "Topology('aiv', 2), Topology('lane', 64))\n"
)


def _collect_loops(fn) -> list[For]:
    loops: list[For] = []

    class _Visitor(StmtVisitor):
        def visit_For(self, stmt: For) -> None:
            if stmt.pipeline is not None:
                loops.append(stmt)
            self.generic_visit(stmt)

    walk_prim_function(_Visitor(), fn)
    return loops


def test_mode_a_24bit_parallel_aiv0_aiv1_aic_co_schedule() -> None:
    """Verify 24-bit masks allow AIV0, AIV1, and AIC to co-schedule while dual-AIV serializes."""
    m_aiv0 = NodeResourceMask.for_aiv0("VECTOR")
    m_aiv1 = NodeResourceMask.for_aiv1("VECTOR")
    m_aic = NodeResourceMask.for_aic("CUBE")
    t_aiv0 = Task(name="vec0", resources=m_aiv0.to_solver_resources(), latency=1, stage=0)
    t_aiv1 = Task(name="vec1", resources=m_aiv1.to_solver_resources(), latency=1, stage=0)
    t_aic = Task(name="cube0", resources=m_aic.to_solver_resources(), latency=1, stage=0)

    parallel_prob = Problem(
        tasks=(t_aiv0, t_aiv1, t_aic),
        resources=(
            Resource("aiv", "aiv0", "VECTOR", 1),
            Resource("aiv", "aiv1", "VECTOR", 1),
            Resource("aic", "aic", "CUBE", 1),
        ),
        max_stage=0,
        manual=True,
    )
    assert bitwise_conflict_pairs(parallel_prob.tasks) == ()
    res = solve_ascend_pipeline(parallel_prob)
    assert res.assignment.ii == 1
    assert res.assignment.phases == {"vec0": 0, "vec1": 0, "cube0": 0}

    m_dual = NodeResourceMask.for_dual_aiv("VECTOR")
    t_dual = Task(name="dual_vec", resources=m_dual.to_solver_resources(), latency=1, stage=0)
    dual_prob = Problem(
        tasks=(t_aiv0, t_aiv1, t_dual, t_aic),
        resources=(
            Resource("aiv", "aiv0", "VECTOR", 1),
            Resource("aiv", "aiv1", "VECTOR", 1),
            Resource("aic", "aic", "CUBE", 1),
        ),
        max_stage=0,
        manual=True,
    )
    conflicts = bitwise_conflict_pairs(dual_prob.tasks)
    assert conflicts == ((0, 2), (1, 2))
    dual_res = solve_ascend_pipeline(dual_prob)
    assert dual_res.assignment.ii == 2
    assert dual_res.assignment.phases["dual_vec"] != dual_res.assignment.phases["vec0"]
    assert dual_res.assignment.phases["dual_vec"] != dual_res.assignment.phases["vec1"]
    assert task_resource_mask(t_dual) == m_dual


def test_mode_a_solve_loop_with_ledger(tmp_path: pathlib.Path) -> None:
    """Verify Mode A loop solve emits an ObjectivePhysicalLedger with exact_modulo_z3."""
    body = """
@module(entry="piped_add", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def piped_add(
        a: Tensor[(4, 64), "f32"],
        b: Tensor[(4, 64), "f32"],
    ) -> Tensor[(4, 64), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            c = tf.empty(Tensor[(4, 64), "f32", "gmem"])
            for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):
                a_ub = tf.schedule((a[k, :], tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_gm_to_ub())
                b_ub = tf.schedule((b[k, :], tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    ar = tf.schedule((a_ub, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    br = tf.schedule((b_ub, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    cr = tf.schedule((ar, br, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_vadd())
                    c_ub = tf.schedule((cr, tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_reg_to_ub())
                c_row = tf.schedule((c_ub, tf.empty(Tensor[(64,), "f32", "gmem"])), op=T.ascend_ub_to_gm())
                c = tf.insert_slice(c, tf.reshape(c_row, new_shape=(1, 64)), (k, 0))
            return c
"""
    path = tmp_path / "piped_add.py"
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    mod = load_authored_ir(f"{path}:M")
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    lowered = finalize(analyzed, authored=authored)
    fn = lowered.entry_function()
    loop = _collect_loops(fn)[0]

    result, ledger = solve_loop_with_ledger(loop, fn_stmt=fn.body)
    assert isinstance(ledger, ObjectivePhysicalLedger)
    assert ledger.analysis_mode == "exact_modulo_z3"
    assert ledger.is_legal is True
    assert ledger.steady_state_ii == result.assignment.ii
    assert "UB" in ledger.waterlines
    assert ledger.waterlines["UB"].overflow_bytes == 0
    assert ledger.dominant_pipe != "NONE"
    parsed = json.loads(ledger.to_json())
    assert parsed["analysis_mode"] == "exact_modulo_z3"


def test_mode_b_two_snapshot_ir_diff_without_bisheng() -> None:
    """Verify Two-Snapshot IR Diff captures TileLang lowering passes and buffer bloat."""
    import tilelang.ascend.language as T

    @T.prim_func
    def sample_gemm(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])

    inspection = inspect_tilelang_primfunc(sample_gemm)
    ledger = inspection.ledger
    assert ledger.analysis_mode == "readonly_envelope"
    assert ledger.steady_state_ii is None
    assert ledger.is_legal is True
    assert "tl.MaterializeMultiBuffer" in inspection.lowered_passes_observed
    assert "tl.MergeUBAllocations" in inspection.lowered_passes_observed
    assert "L1" in ledger.waterlines
    assert "L0C" in ledger.waterlines
    assert ledger.waterlines["L1"].explicit_bytes == 2 * (128 * 64 * 2)
    assert ledger.waterlines["L1"].peak_allocated_bytes >= ledger.waterlines["L1"].explicit_bytes
    assert ledger.waterlines["L0C"].overflow_bytes == 0
    assert ledger.dominant_pipe in ("AIC.PIPE_FIX", "AIC.PIPE_M", "AIC.PIPE_MTE2")


def test_mode_b_detects_dynamic_while_mte2_queue_and_padding() -> None:
    """Verify Mode B detects dynamic While loops, MTE2 > 16 queue depth, and padding."""
    import tilelang.ascend.language as T

    @T.prim_func
    def dynamic_fragmented(
        X: T.Tensor((32, 256), "float32"),
        Y: T.Tensor((32, 256), "float32"),
        steps: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((32, 256), "float32")
            it = T.alloc_var("int32", 0)
            while it < steps[0]:
                for seg in T.serial(32):
                    c0 = seg * 8
                    T.copy(X[0, c0 : c0 + 5], ub[0, c0 : c0 + 5])
                T.copy(ub, Y[0:32, 0:256])
                it += 1

    res = inspect_tilelang_primfunc(
        dynamic_fragmented,
        pass_configs={"tl.enable_auto_schedule": False},
    )
    assert "while_loop" in res.ledger.opaque_dynamic_regions
    assert "data_dependent_branch" in res.ledger.opaque_dynamic_regions
    assert res.ledger.mte2_inflight_per_aiv == 32
    assert res.ledger.unaligned_dma_padding_bytes > 0
    assert "UB" in res.ledger.waterlines


def test_mode_b_detects_ub_overflow_in_two_snapshot_diff() -> None:
    """Verify Two-Snapshot IR Diff catches UB capacity overflow before bisheng."""
    import tilelang.ascend.language as T

    @T.prim_func
    def huge_ub_kernel(
        X: T.Tensor((2, 256, 128), "float32"),
        Y: T.Tensor((2, 256, 128), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((256, 128), "float32")
            b_ub = T.alloc_shared((256, 128), "float32")
            for k in T.Pipelined(2, num_stages=2):
                T.copy(X[k, 0:256, 0:128], a_ub)
                T.copy(a_ub, b_ub)
                T.copy(b_ub, Y[k, 0:256, 0:128])

    res = inspect_tilelang_primfunc(huge_ub_kernel)
    assert res.ledger.is_legal is False
    assert (
        res.ledger.max_overflow_bytes > 0
        or "lowering_rejected" in res.ledger.opaque_dynamic_regions
    )


def test_mode_b_tir_envelope_dynamic_while_and_rule_blindness(tmp_path: pathlib.Path) -> None:
    """Verify TileFoundry TIR envelope inspection and zero agent imports under ascend/."""
    body = """
@module(entry="vec_copy", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def vec_copy(a: Tensor[(4, 60), "f32"]) -> Tensor[(4, 60), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            c = tf.empty(Tensor[(4, 60), "f32", "gmem"])
            for k in range(4):
                a_ub = tf.schedule((a[k, :], tf.empty(Tensor[(60,), "f32", "smem"])), op=T.ascend_gm_to_ub())
                c_row = tf.schedule((a_ub, tf.empty(Tensor[(60,), "f32", "gmem"])), op=T.ascend_ub_to_gm())
                c = tf.insert_slice(c, tf.reshape(c_row, new_shape=(1, 60)), (k, 0))
            return c
"""
    p = tmp_path / "vec_copy.py"
    p.write_text(HEADER + "\n" + body, encoding="utf-8")
    mod = load_authored_ir(f"{p}:M")
    c_mod, c_fn = check_program(mod, mod.entry_function())
    low = finalize(analyze(c_mod, c_fn, analysis="memory"), authored=mod.entry_function())
    fn = low.entry_function()
    res = inspect_tir_envelope(fn)
    assert res.ledger.analysis_mode == "readonly_envelope"
    assert res.ledger.unaligned_dma_padding_bytes > 0
    assert "UB" in res.ledger.waterlines
    assert isinstance(res.ledger.waterlines["UB"], ScopeWaterline)

    ascend_root = pathlib.Path(__file__).resolve().parents[3] / "targets" / "ascend" / "tilefoundry_ascend" / "ascend"
    for py_file in ascend_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "tilefoundry_ascend.agent" not in text
