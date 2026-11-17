"""Tests for Mode A 24-bit core solver, Mode B Two-Snapshot IR Diff, and ObjectivePhysicalLedger."""

import json
import pathlib

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.hir.pipeline import PipelineSpec
from tilefoundry.ir.tir.stmts import For, Sequential
from tilefoundry.ir.types import TensorType
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
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    _detect_tilelang_aiv_split_factor,
    _tir_for_static_trip_count,
    _try_solve_tilelang_pipelined_ii,
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
    assert ledger.analysis_mode == "exact_modulo_z3"
    assert ledger.steady_state_ii is not None
    assert ledger.steady_state_ii > 1
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


def test_fp1_tir_for_static_trip_multiplication(tmp_path: pathlib.Path) -> None:
    """Verify _tir_for_static_trip_count and inspect_tir_envelope loop trip scaling."""
    sc = TensorType.umat_scalar()
    iv = Var("k", type=sc)
    empty_body = Sequential(())
    c0 = Constant(0, type=sc)
    c1 = Constant(1, type=sc)
    c2 = Constant(2, type=sc)
    c4 = Constant(4, type=sc)
    c7 = Constant(7, type=sc)
    assert _tir_for_static_trip_count(For(iv, c0, c4, c1, empty_body)) == 4
    assert _tir_for_static_trip_count(For(iv, c1, c7, c2, empty_body)) == 3
    assert _tir_for_static_trip_count(For(iv, c0, c0, c1, empty_body)) == 0
    assert _tir_for_static_trip_count(For(iv, c0, Var("n", type=sc), c1, empty_body)) is None
    assert _tir_for_static_trip_count(For(iv, c0, c4, c0, empty_body)) is None

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
    p = tmp_path / "vec_copy_trip.py"
    p.write_text(HEADER + "\n" + body, encoding="utf-8")
    mod = load_authored_ir(f"{p}:M")
    c_mod, c_fn = check_program(mod, mod.entry_function())
    low = finalize(analyze(c_mod, c_fn, analysis="memory"), authored=mod.entry_function())
    res = inspect_tir_envelope(low.entry_function())
    assert res.ledger.opaque_dynamic_regions == ()
    assert res.ledger.pipe_cycles["AIV0.PIPE_MTE2"] == 4
    assert res.ledger.pipe_cycles["AIV0.PIPE_MTE3"] == 4
    assert res.ledger.unaligned_dma_padding_bytes == 64

    zero_loop = For(iv, c0, c0, c1, Sequential(()))
    zero_res = inspect_tir_envelope(zero_loop)
    assert zero_res.ledger.pipe_cycles == {}
    assert zero_res.ledger.opaque_dynamic_regions == ()

    dyn_loop = For(iv, c0, Var("dyn_n", type=sc), c1, Sequential(()))
    dyn_res = inspect_tir_envelope(dyn_loop)
    assert "dynamic_for:k" in dyn_res.ledger.opaque_dynamic_regions


def test_fp1_auto_detect_dual_aiv_and_sid_branch_split() -> None:
    """Verify MixedKernel and dual_copy auto-detect 1C2V split and halve symmetric AIV work."""
    import tilelang.ascend.language as T
    from tilelang import tvm
    from tvm import tirx

    @T.prim_func
    def single_aiv_kernel(X: T.Tensor((128, 128), "float32"), Y: T.Tensor((128, 128), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((128, 128), "float32")
            T.copy(X[0:128, 0:128], x_ub)
            T.copy(x_ub, Y[0:128, 0:128])

    @T.prim_func
    def mixed_dual_kernel(X: T.Tensor((128, 128), "float32"), Y: T.Tensor((128, 128), "float32")):
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_ub = T.alloc_shared((64, 128), "float32")
            x_l1 = T.alloc_l1((128, 128), "float32")
            for _r in T.serial(4):
                T.copy(X[sid * 64 : (sid + 1) * 64, 0:128], x_ub)
            with T.SimdVF():
                m32 = T.simd.pset(32, "PAT_ALL")
                for r in T.serial(8):
                    v = T.simd.vld(x_ub[r, 0], dist="NORM_B32")
                    T.simd.vsts(x_ub[r, 0], v, mask=m32, dist="NORM_B32")
            T.dual_copy(x_ub, x_l1)
            T.copy(x_ub, Y[sid * 64 : (sid + 1) * 64, 0:128])

    @T.prim_func
    def guarded_sid_kernel(X: T.Tensor((128, 128), "float32")):
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_ub = T.alloc_shared((64, 128), "float32")
            if sid == 0:
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    for r in T.serial(4):
                        v = T.simd.vld(x_ub[r, 0], dist="NORM_B32")
                        T.simd.vsts(x_ub[r, 0], v, mask=m32, dist="NORM_B32")

    assert _detect_tilelang_aiv_split_factor(single_aiv_kernel, tirx=tirx) == 1
    assert _detect_tilelang_aiv_split_factor(mixed_dual_kernel, tirx=tirx) == 2

    cthread_iv = tirx.IterVar(tvm.ir.Range(0, 2), tirx.Var("sid", "int32"), 0, "cthread")
    synth_body = tirx.AttrStmt(cthread_iv, "thread_extent", tirx.IntImm("int32", 2), tirx.Evaluate(0))
    synth_fn = tirx.PrimFunc([], synth_body)
    assert _detect_tilelang_aiv_split_factor(synth_fn, tirx=tirx) == 2

    single_res = inspect_tilelang_primfunc(single_aiv_kernel)
    assert single_res.ledger.analysis_mode == "readonly_envelope"
    assert single_res.ledger.steady_state_ii is None
    assert single_res.ledger.pipe_cycles.get("AIV1.PIPE_V", 0) == 0

    split1 = inspect_tilelang_primfunc(mixed_dual_kernel, aiv_split_factor=1)
    auto_split = inspect_tilelang_primfunc(mixed_dual_kernel)
    assert auto_split.ledger.pipe_cycles.get("AIV1.PIPE_V", 0) > 0
    assert auto_split.ledger.pipe_cycles["AIV0.PIPE_V"] == split1.ledger.pipe_cycles["AIV0.PIPE_V"] // 2
    assert auto_split.ledger.pipe_cycles["AIV1.PIPE_MTE2"] > 0
    assert (
        auto_split.ledger.pipe_cycles["AIV0.PIPE_MTE2"]
        == split1.ledger.pipe_cycles["AIV0.PIPE_MTE2"] // 2
    )
    assert auto_split.ledger.pipe_cycles["AIV1.PIPE_MTE3"] > 0
    assert (
        auto_split.ledger.pipe_cycles["AIV0.PIPE_MTE3"]
        == split1.ledger.pipe_cycles["AIV0.PIPE_MTE3"] // 2
    )

    guarded_res = inspect_tilelang_primfunc(guarded_sid_kernel)
    assert guarded_res.ledger.pipe_cycles.get("AIV0.PIPE_V", 0) == 4
    assert guarded_res.ledger.pipe_cycles.get("AIV1.PIPE_V", 0) == 0


def test_fp1_pipelined_mode_a_steady_state_ii_and_degraded_buffers() -> None:
    """Verify Mode A Z3 II extraction on static T.Pipelined MixedKernel and dynamic fallback."""
    import tilelang.ascend.language as T
    from tilelang.utils.language import DataType as TLDataType
    from tvm import tirx

    @T.prim_func
    def mixed_pipelined_gemm(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            out_ub = T.alloc_shared((64, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.dual_copy(acc, out_ub)
            T.copy(out_ub, Y[sid * 64 : (sid + 1) * 64, 0:128])

    res = inspect_tilelang_primfunc(mixed_pipelined_gemm)
    assert res.ledger.analysis_mode == "exact_modulo_z3"
    assert res.ledger.steady_state_ii is not None
    assert res.ledger.steady_state_ii > 1

    spaces = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    degraded_ii = _try_solve_tilelang_pipelined_ii(
        mixed_pipelined_gemm,
        spaces,
        tirx=tirx,
        DataTypeCls=TLDataType,
        aiv_split_factor=2,
        has_simtvf=False,
        unreserved_ubuf=False,
        degraded_names=("x_l1", "w_l1"),
    )
    assert degraded_ii is not None
    assert degraded_ii >= res.ledger.steady_state_ii

    @T.prim_func
    def dynamic_pipelined(
        X: T.Tensor((128, 128), "float32"),
        Y: T.Tensor((128, 128), "float32"),
        steps: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((32, 128), "float32")
            for k in T.Pipelined(steps[0], num_stages=2):
                T.copy(X[0:32, 0:128], ub)
                T.copy(ub, Y[0:32, 0:128])

    dyn_res = inspect_tilelang_primfunc(
        dynamic_pipelined,
        pass_configs={"tl.enable_auto_schedule": False},
    )
    assert dyn_res.ledger.analysis_mode == "readonly_envelope"
    assert dyn_res.ledger.steady_state_ii is None


def test_fp1_physical_ledger_from_dict_roundtrip() -> None:
    """Verify ScopeWaterline and ObjectivePhysicalLedger lossless from_dict roundtrip."""
    wl_ub = ScopeWaterline(
        scope="UB",
        capacity_bytes=245760,
        explicit_bytes=32768,
        implicit_bloat_bytes=4096,
        peak_allocated_bytes=36864,
        overflow_bytes=0,
        explicit_buffers={"a_ub": 16384, "b_ub": 16384},
        implicit_buffers={"a_ub_expansion": 4096},
        degraded_single_buffers=("b_ub",),
    )
    wl_l1 = ScopeWaterline(
        scope="L1",
        capacity_bytes=1048576,
        explicit_bytes=65536,
        implicit_bloat_bytes=0,
        peak_allocated_bytes=65536,
        overflow_bytes=0,
        explicit_buffers={"x_l1": 65536},
        implicit_buffers={},
    )
    assert ScopeWaterline.from_dict(wl_ub.to_dict()) == wl_ub
    assert ScopeWaterline.from_dict(wl_l1.to_dict()) == wl_l1

    ledger = ObjectivePhysicalLedger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIC.PIPE_M": 8, "AIV0.PIPE_V": 4, "AIV1.PIPE_V": 4},
        dominant_pipe="AIC.PIPE_M",
        waterlines={"UB": wl_ub, "L1": wl_l1},
        mte2_inflight_per_aiv=2,
        unaligned_dma_padding_bytes=16,
        opaque_dynamic_regions=("simt_vf",),
    )
    assert ObjectivePhysicalLedger.from_dict(ledger.to_dict()) == ledger
    assert ObjectivePhysicalLedger.from_dict(json.loads(ledger.to_json())) == ledger

    ro_ledger = ObjectivePhysicalLedger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_MTE2": 12},
        dominant_pipe="AIV0.PIPE_MTE2",
        waterlines={"UB": wl_ub},
        mte2_inflight_per_aiv=18,
        unaligned_dma_padding_bytes=64,
        opaque_dynamic_regions=("while_loop", "dynamic_for:k"),
        grid_is_symbolic=True,
    )
    assert ObjectivePhysicalLedger.from_dict(ro_ledger.to_dict()) == ro_ledger


def test_symbolic_grid_var_bindings_and_r13_guard(tmp_path: pathlib.Path) -> None:
    """Verify symbolic grid resolution via var_bindings and R13 guard on unbound grids."""
    import importlib.util
    from tilefoundry_ascend.agent.kb import build_predicate_context, load_seed_rules

    src = """
import tilelang.ascend.language as T

def make_dyn_vec():
    m_rows = T.dynamic("m_rows")
    @T.prim_func
    def main(
        X: T.Tensor((m_rows, 64), "float32"),
        Y: T.Tensor((m_rows, 64), "float32"),
    ):
        n_blk = T.ceildiv(m_rows, 64)
        with T.Kernel(n_blk) as bx:
            ub = T.alloc_shared((64, 64), "float32")
            T.copy(X[bx * 64 : bx * 64 + 64, 0:64], ub)
            T.copy(ub, Y[bx * 64 : bx * 64 + 64, 0:64])
    return main

def make_dyn_q_head():
    q_total_rows = T.dynamic("q_total_rows")
    @T.prim_func
    def main(
        Q: T.Tensor((q_total_rows, 64), "float32"),
        Out: T.Tensor((q_total_rows, 64), "float32"),
    ):
        with T.Kernel(T.ceildiv(q_total_rows, 64)) as bx:
            ub = T.alloc_shared((64, 64), "float32")
            T.copy(Q[bx * 64 : bx * 64 + 64, 0:64], ub)
            T.copy(ub, Out[bx * 64 : bx * 64 + 64, 0:64])
    return main
"""
    mod_path = tmp_path / "dyn_grid_kernels.py"
    mod_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_tf_dyn_grid_test", str(mod_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    r13 = next(r for r in load_seed_rules() if r.rule_id == "R13_SPLIT_KV_WAVE_RECOVERY")

    vec_fn = mod.make_dyn_vec()
    bound_high = inspect_tilelang_primfunc(vec_fn, var_bindings={"m_rows": 4096})
    assert bound_high.ledger.grid_blocks == 64
    assert bound_high.ledger.grid_is_symbolic is False
    assert bound_high.ledger.wave_occupancy > 1.0
    assert r13.matches(build_predicate_context(bound_high.ledger)) is False

    unbound = inspect_tilelang_primfunc(vec_fn, var_bindings=None)
    assert unbound.ledger.grid_is_symbolic is True
    assert unbound.ledger.wave_occupancy == 1.0
    assert unbound.ledger.grid_blocks == 56
    assert r13.matches(build_predicate_context(unbound.ledger)) is False
    assert ObjectivePhysicalLedger.from_dict(unbound.ledger.to_dict()) == unbound.ledger

    q_fn = mod.make_dyn_q_head()
    bound_low = inspect_tilelang_primfunc(q_fn, var_bindings={"q_total_rows": 1 * 1 * 128})
    assert bound_low.ledger.grid_blocks == 2
    assert bound_low.ledger.grid_is_symbolic is False
    assert 0.0 < bound_low.ledger.wave_occupancy < 0.5
    assert r13.matches(build_predicate_context(bound_low.ledger)) is True


def test_tilelang_burst_efficiency_and_pipe_cycles() -> None:
    """Verify 8 KiB GM->UB DMA incurs lower burst efficiency and higher cycles/byte than 32 KiB."""
    import tilelang.ascend.language as T

    @T.prim_func
    def copy_8k(X: T.Tensor((16, 1024), "float32"), Y: T.Tensor((16, 1024), "float32")):
        with T.Kernel(1) as _:
            with T.SimdVF():
                x_ub = T.alloc_shared((2, 1024), "float32")
                T.copy(X[0:2, 0:1024], x_ub)
                T.copy(x_ub, Y[0:2, 0:1024])

    @T.prim_func
    def copy_32k(X: T.Tensor((16, 1024), "float32"), Y: T.Tensor((16, 1024), "float32")):
        with T.Kernel(1) as _:
            with T.SimdVF():
                x_ub = T.alloc_shared((8, 1024), "float32")
                T.copy(X[0:8, 0:1024], x_ub)
                T.copy(x_ub, Y[0:8, 0:1024])

    r8k = inspect_tilelang_primfunc(copy_8k)
    r32k = inspect_tilelang_primfunc(copy_32k)
    assert r8k.ledger.min_gm_dma_bytes == 8192
    assert r32k.ledger.min_gm_dma_bytes == 32768
    assert r8k.ledger.dma_burst_efficiency_ratio < r32k.ledger.dma_burst_efficiency_ratio
    mte2_8k = r8k.ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0)
    mte2_32k = r32k.ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0)
    assert mte2_8k > 0 and mte2_32k > 0
    assert (mte2_8k / 8192.0) > (mte2_32k / 32768.0)


def test_tilelang_unlimit_memory_preserves_ub_enforcement() -> None:
    """Verify T.annotate_unlimit_memory('shared') records flag without disabling UB limit."""
    import tilelang.ascend.language as T

    @T.prim_func
    def unlimit_fn(X: T.Tensor((16, 1024), "float16"), Y: T.Tensor((16, 1024), "float16")):
        with T.Kernel(1) as _:
            T.annotate_unlimit_memory("shared")
            buf = T.alloc_shared((2, 1024), "float16")
            for i in T.Pipelined(4, num_stages=2):
                T.copy(X[i * 2 : (i + 1) * 2, :], buf)
                T.copy(buf, Y[i * 2 : (i + 1) * 2, :])

    res = inspect_tilelang_primfunc(unlimit_fn)
    assert res.ledger.has_unlimit_memory_attr is True
    assert "UB" in res.ledger.waterlines
    assert res.ledger.waterlines["UB"].capacity_bytes in (248 * 1024, 256 * 1024)


def test_tilelang_dynamic_offset_pipelining_vs_static_pingpong() -> None:
    """Verify T.Pipelined marks dynamic-offset pipelining whereas static ping-pong does not."""
    import tilelang.ascend.language as T

    @T.prim_func
    def pipelined_fn(X: T.Tensor((16, 1024), "float16"), Y: T.Tensor((16, 1024), "float16")):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((2, 1024), "float16")
            for i in T.Pipelined(4, num_stages=2):
                T.copy(X[i * 2 : (i + 1) * 2, :], buf)
                T.copy(buf, Y[i * 2 : (i + 1) * 2, :])

    @T.prim_func
    def static_pingpong_fn(
        X: T.Tensor((16, 1024), "float16"), Y: T.Tensor((16, 1024), "float16")
    ):
        with T.Kernel(1) as _:
            va = T.alloc_shared((2, 1024), "float16")
            vb = T.alloc_shared((2, 1024), "float16")
            for i in T.serial(2):
                T.copy(X[i * 4 : i * 4 + 2, :], va)
                T.copy(X[i * 4 + 2 : i * 4 + 4, :], vb)
                T.copy(va, Y[i * 4 : i * 4 + 2, :])
                T.copy(vb, Y[i * 4 + 2 : i * 4 + 4, :])

    rp = inspect_tilelang_primfunc(pipelined_fn)
    rs = inspect_tilelang_primfunc(static_pingpong_fn)
    assert rp.ledger.uses_dynamic_offset_pipelining is True
    assert rs.ledger.uses_dynamic_offset_pipelining is False


def test_tilelang_coupled_reduce_sfu_vs_loop_fission() -> None:
    """Verify coupled reduce+SFU+wide-accum detection vs two-pass loop fission."""
    import tilelang.ascend.language as T

    @T.prim_func
    def coupled_fn(
        Q: T.Tensor((4, 64), "float32"),
        Out: T.Tensor((1, 64), "float32"),
        ScalarOut: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1) as _:
            q_ub = T.alloc_shared((1, 64), "float32")
            acc_ub = T.alloc_shared((4, 64), "float32")
            row_ub = T.alloc_shared((4, 1), "float32")
            for s in T.serial(4):
                T.copy(Q[s : s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    sum_v = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")
                    exp_v = T.simd.vexp(sum_v, mask=m32)
                    T.simd.vsts(row_ub[s, 0], exp_v, mask=m32, dist="ONEPT_B32")
                    acc_v = T.simd.vld(acc_ub[s, 0])
                    T.simd.vsts(acc_ub[s, 0], T.simd.vadd(acc_v, qv, mask=m32), mask=m32)
            T.copy(acc_ub[0:1, :], Out)
            T.copy(row_ub, ScalarOut)

    @T.prim_func
    def fission_fn(
        Q: T.Tensor((4, 64), "float32"),
        Out: T.Tensor((1, 64), "float32"),
        ScalarOut: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1) as _:
            q_ub = T.alloc_shared((1, 64), "float32")
            acc_ub = T.alloc_shared((4, 64), "float32")
            exb = T.alloc_shared((4, 1), "float32")
            for s in T.serial(4):
                T.copy(Q[s : s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    sum_v = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")
                    exp_v = T.simd.vexp(sum_v, mask=m32)
                    T.simd.vsts(exb[s, 0], exp_v, mask=m32, dist="ONEPT_B32")
            for s in T.serial(4):
                T.copy(Q[s : s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    acc_v = T.simd.vld(acc_ub[s, 0])
                    T.simd.vsts(acc_ub[s, 0], T.simd.vadd(acc_v, qv, mask=m32), mask=m32)
            T.copy(acc_ub[0:1, :], Out)
            T.copy(exb, ScalarOut)

    rc = inspect_tilelang_primfunc(coupled_fn)
    rf = inspect_tilelang_primfunc(fission_fn)
    assert rc.ledger.has_coupled_reduce_sfu_accum is True
    assert rf.ledger.has_coupled_reduce_sfu_accum is False
