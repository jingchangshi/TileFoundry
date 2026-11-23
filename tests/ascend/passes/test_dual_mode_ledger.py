"""Tests for Mode A 24-bit core solver, Mode B Two-Snapshot IR Diff, and ObjectivePhysicalLedger."""

import json
import pathlib

import tilelang.ascend.language as T
from tilefoundry.vendor.tilelang_z3 import (
    Buffer,
    Capacity,
    Edge,
    Problem,
    Resource,
    Reuse,
    Task,
)
from tilefoundry_ascend.ascend.passes import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule import (
    bitwise_conflict_pairs,
    inspect_tilelang_primfunc,
    pipe_cycles_from_problem,
    solve_ascend_pipeline,
    task_resource_mask,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    _detect_tilelang_aiv_split_factor,
    _try_solve_tilelang_pipelined_ii,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget, NodeResourceMask


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


def test_mode_a_solve_loop_with_ledger() -> None:
    """Verify Mode A Z3 solve emits an ObjectivePhysicalLedger with exact_modulo_z3."""
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load_a", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="load_b", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="vec_add", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
            Task(name="store_c", resources=(("aiv", "aiv0", "mte3", 1),), latency=1, stage=2),
        ),
        buffers=(
            Buffer(name="a_ub", slot_bytes=256, space="UB", owner="aiv", auto_min=1, auto_max=2),
            Buffer(name="b_ub", slot_bytes=256, space="UB", owner="aiv", auto_min=1, auto_max=2),
            Buffer(name="c_ub", slot_bytes=256, space="UB", owner="aiv", auto_min=1, auto_max=2),
        ),
        edges=(
            Edge(src="load_a", dst="vec_add", distance=0, latency=1),
            Edge(src="load_b", dst="vec_add", distance=0, latency=1),
            Edge(src="vec_add", dst="store_c", distance=0, latency=1),
        ),
        reuses=(
            Reuse(buffer="a_ub", last_reader="vec_add", next_writer="load_a", delta=0, read_latency=1),
            Reuse(buffer="b_ub", last_reader="vec_add", next_writer="load_b", delta=0, read_latency=1),
            Reuse(buffer="c_ub", last_reader="store_c", next_writer="vec_add", delta=0, read_latency=1),
        ),
        resources=(
            Resource("aiv", "aiv0", "mte2", 1),
            Resource("aiv", "aiv0", "vector", 1),
            Resource("aiv", "aiv0", "mte3", 1),
        ),
        capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes),),
        max_stage=2,
        manual=True,
    )
    result = solve_ascend_pipeline(problem)
    explicit_ub = {
        b.name: b.slot_bytes * result.buffer_versions.get(b.name, 1)
        for b in result.problem.buffers
    }
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=result.assignment.ii,
        pipe_cycles=pipe_cycles_from_problem(result.problem, result.assignment),
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=facts.ub_simdvf_usable_bytes,
                explicit_buffers=explicit_ub,
                implicit_buffers={},
            )
        },
    )
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
    import tilelang.ascend.language as T  # noqa: PLC0415

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
    import tilelang.ascend.language as T  # noqa: PLC0415

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
    import tilelang.ascend.language as T  # noqa: PLC0415

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


def test_mode_b_tilelang_envelope_unaligned_and_rule_blindness() -> None:
    """Verify TileLang envelope inspection and zero agent imports under ascend/."""
    @T.prim_func
    def vec_copy(a: T.Tensor((4, 60), "float32"), c: T.Tensor((4, 60), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, 60), "float32")
            for k in T.serial(4):
                T.copy(a[k : k + 1, 0:60], a_ub)
                T.copy(a_ub, c[k : k + 1, 0:60])

    res = inspect_tilelang_primfunc(vec_copy, pass_configs={"tl.enable_auto_schedule": False})
    assert res.ledger.analysis_mode == "readonly_envelope"
    assert res.ledger.unaligned_dma_padding_bytes > 0
    assert "UB" in res.ledger.waterlines
    assert isinstance(res.ledger.waterlines["UB"], ScopeWaterline)

    ascend_root = pathlib.Path(__file__).resolve().parents[3] / "targets" / "ascend" / "tilefoundry_ascend" / "ascend"
    for py_file in ascend_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "tilefoundry_ascend.agent" not in text


def test_fp1_tilelang_for_static_trip_multiplication() -> None:
    """Verify TileLang loop trip scaling, unaligned padding, and dynamic loop detection."""
    @T.prim_func
    def vec_copy(a: T.Tensor((4, 60), "float32"), c: T.Tensor((4, 60), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, 60), "float32")
            for k in T.serial(4):
                T.copy(a[k : k + 1, 0:60], a_ub)
                T.copy(a_ub, c[k : k + 1, 0:60])

    res = inspect_tilelang_primfunc(vec_copy, pass_configs={"tl.enable_auto_schedule": False})
    assert res.ledger.opaque_dynamic_regions == ()
    assert res.ledger.pipe_cycles["AIV0.PIPE_MTE2"] > 0
    assert res.ledger.pipe_cycles["AIV0.PIPE_MTE3"] > 0
    assert res.ledger.unaligned_dma_padding_bytes == 64

    @T.prim_func
    def dyn_loop(
        a: T.Tensor((4, 64), "float32"),
        c: T.Tensor((4, 64), "float32"),
        n_steps: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, 64), "float32")
            for k in T.serial(n_steps[0]):
                T.copy(a[k : k + 1, 0:64], a_ub)
                T.copy(a_ub, c[k : k + 1, 0:64])

    dyn_res = inspect_tilelang_primfunc(dyn_loop, pass_configs={"tl.enable_auto_schedule": False})
    assert any(tag.startswith("dynamic_for:") for tag in dyn_res.ledger.opaque_dynamic_regions)


def test_fp1_auto_detect_dual_aiv_and_sid_branch_split() -> None:
    """Verify MixedKernel and dual_copy auto-detect 1C2V split and halve symmetric AIV work."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

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
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

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
    import importlib.util  # noqa: PLC0415

    from tilefoundry_ascend.agent.kb import (  # noqa: PLC0415
        build_predicate_context,
        load_seed_rules,
    )

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
    import tilelang.ascend.language as T  # noqa: PLC0415

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
    import tilelang.ascend.language as T  # noqa: PLC0415

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
    import tilelang.ascend.language as T  # noqa: PLC0415

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
    import tilelang.ascend.language as T  # noqa: PLC0415

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


def test_dav3510_calibrated_cycles_bank_conflicts_and_dynamic_pipelined_fallback(
    tmp_path: pathlib.Path,
) -> None:
    """Verify dav-3510 Cube/Vector cycles, bank conflicts, var_bindings, and analytical II."""
    import importlib.util  # noqa: PLC0415

    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    @T.prim_func
    def gemm_bf16_small(
        A: T.Tensor((32, 64), "bfloat16"),
        B: T.Tensor((64, 32), "bfloat16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "bfloat16")
            b_l1 = T.alloc_l1((64, 32), "bfloat16")
            c_l0c = T.alloc_l0c((32, 32), "float32")
            T.copy(A[0:32, 0:64], a_l1)
            T.copy(B[0:64, 0:32], b_l1)
            T.gemm(a_l1, b_l1, c_l0c, clear_accum=True)
            T.copy(c_l0c, C[0:32, 0:32])

    @T.prim_func
    def gemm_fp8_small(
        A: T.Tensor((32, 64), "float8_e4m3fn"),
        B: T.Tensor((64, 32), "float8_e4m3fn"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "float8_e4m3fn")
            b_l1 = T.alloc_l1((64, 32), "float8_e4m3fn")
            c_l0c = T.alloc_l0c((32, 32), "float32")
            T.copy(A[0:32, 0:64], a_l1)
            T.copy(B[0:64, 0:32], b_l1)
            T.gemm(a_l1, b_l1, c_l0c, clear_accum=True)
            T.copy(c_l0c, C[0:32, 0:32])

    r_bf16 = inspect_tilelang_primfunc(gemm_bf16_small, pass_configs={"tl.enable_auto_schedule": False})
    r_fp8 = inspect_tilelang_primfunc(gemm_fp8_small, pass_configs={"tl.enable_auto_schedule": False})
    assert r_bf16.ledger.pipe_cycles["AIC.PIPE_M"] == 22
    assert r_fp8.ledger.pipe_cycles["AIC.PIPE_M"] == 14
    assert 0.0 < r_bf16.ledger.l1_bank_conflict_risk <= 0.5

    @T.prim_func
    def vec_sfu_kernel(X: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 64), "float32")
            T.copy(X[0:4, 0:64], ub)
            with T.SimdVF():
                m32 = T.simd.pset(32, "PAT_ALL")
                v = T.simd.vld(ub[0, 0], dist="NORM_B32")
                e = T.simd.vexp(v, mask=m32)
                T.simd.vsts(ub[0, 0], e, mask=m32, dist="NORM_B32")

    r_sfu = inspect_tilelang_primfunc(vec_sfu_kernel, pass_configs={"tl.enable_auto_schedule": False})
    assert r_sfu.ledger.pipe_cycles["AIV0.PIPE_V"] >= 4

    @T.prim_func
    def ub_conflict_kernel(X: T.Tensor((16, 128), "float32"), Y: T.Tensor((16, 128), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 128), "float32")
            T.copy(X[0:16, 0:128], ub)
            T.copy(ub, Y[0:16, 0:128])

    @T.prim_func
    def ub_padded_kernel(X: T.Tensor((16, 136), "float32"), Y: T.Tensor((16, 136), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 136), "float32")
            T.copy(X[0:16, 0:136], ub)
            T.copy(ub, Y[0:16, 0:136])

    r_conf = inspect_tilelang_primfunc(ub_conflict_kernel, pass_configs={"tl.enable_auto_schedule": False})
    r_pad = inspect_tilelang_primfunc(ub_padded_kernel, pass_configs={"tl.enable_auto_schedule": False})
    assert r_conf.ledger.ub_bank_conflict_risk == 1.0
    assert r_pad.ledger.ub_bank_conflict_risk == 0.0

    sym_src = """
import tilelang.ascend.language as T

def make_op():
    M = T.dynamic("M")
    H = T.dynamic("H")
    C = T.dynamic("C")
    @T.prim_func
    def main(X: T.Tensor((M, H, C), "float32"), Y: T.Tensor((M, H, C), "float32")):
        with T.Kernel(T.ceildiv(M, 32) * H) as bx:
            ub = T.alloc_shared((32, 64), "float32")
            T.copy(X[(bx // H) * 32 : (bx // H) * 32 + 32, bx % H, 0:64], ub)
            T.copy(ub, Y[(bx // H) * 32 : (bx // H) * 32 + 32, bx % H, 0:64])
    return main
"""
    sym_path = tmp_path / "sym_mhc.py"
    sym_path.write_text(sym_src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_tf_sym_mhc", str(sym_path))
    assert spec is not None and spec.loader is not None
    sym_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sym_mod)
    r_sym = inspect_tilelang_primfunc(
        sym_mod.make_op(),
        var_bindings={"M": 128, "H": 4, "C": 64},
        pass_configs={"tl.enable_auto_schedule": False},
    )
    assert r_sym.ledger.grid_is_symbolic is False
    assert r_sym.ledger.grid_blocks == 4 * 4

    @T.prim_func
    def guarded_pipelined_gemm(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Flag: T.Tensor((4,), "int32"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                if Flag[kt] > 0:
                    T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])

    spaces = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    ana_ii = _try_solve_tilelang_pipelined_ii(
        guarded_pipelined_gemm,
        spaces,
        tirx=tirx,
        DataTypeCls=TLDataType,
        aiv_split_factor=1,
        has_simtvf=False,
        unreserved_ubuf=False,
    )
    assert ana_ii is not None and ana_ii >= 1
    r_guarded = inspect_tilelang_primfunc(
        guarded_pipelined_gemm,
        pass_configs={"tl.enable_auto_schedule": False},
    )
    assert "data_dependent_branch" in r_guarded.ledger.opaque_dynamic_regions
    assert r_guarded.ledger.steady_state_ii == ana_ii
    assert r_guarded.ledger.analysis_mode == "readonly_envelope"
