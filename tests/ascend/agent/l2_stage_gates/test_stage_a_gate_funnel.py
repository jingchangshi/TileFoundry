"""Tests for Stage-A static/AST gates, entrypoint purity, routing, and TIR watermarks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
    check_npu_entrypoint_purity,
    check_repair_monotonicity,
    evaluate_ascendc_candidate,
    extract_ordered_tensor_shapes,
    resolve_trackb_compile_flags,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    compute_eq3_hap_objective,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.router import (
    CapabilityGate,
    EscalationDecision,
    PipeGapBreakdown,
    PipeGapReport,
    VALID_BACKEND_TIERS,
    VALID_HARDWARE_PIPES,
    apply_case_ratios,
    evaluate_track_escalation,
    route_execution_track,
    route_from_op_spec,
    route_initial_by_input_contract,
    scan_ascend_sdk_headers,
    scan_tilelang_pipe_gaps,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.passes import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.sil.verifier import verify_tilelang_ast
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS, AscendTarget
from tilefoundry_ascend.ascend.target.ast_semantic_gate import check_tilelang_semantic_contracts

_PRIM_FUNC_KERNEL = """
import tilelang.ascend.language as T

@T.prim_func
def main(X: T.Tensor((1, 64), "float16"), Y: T.Tensor((1, 64), "float16")):
    with T.Kernel(1):
        x_ub = T.alloc_shared((1, 64), "float16")
        T.copy(X, x_ub)
        T.copy(x_ub, Y)
"""


def _parallel_math_module(expr: str, *, simt_vf: bool = False) -> str:
    loop = f"        for i in T.Parallel(64):\n            x_ub[0, i] = {expr}\n"
    if simt_vf:
        loop = "        with T.SimtVF(threads=128):\n" + "".join(
            f"    {line}\n" for line in loop.splitlines()
        )
    return (
        "import tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((1, 64), "float16"), Y: T.Tensor((1, 64), "float16")):\n'
        "    with T.Kernel(1):\n"
        '        x_ub = T.alloc_shared((1, 64), "float16")\n'
        "        T.copy(X, x_ub)\n"
        f"{loop}"
        "        T.copy(x_ub, Y)\n"
    )


def test_stage_a_static_gates_redlines_and_1c2v_subcore_checks() -> None:
    """Verify Stage-A rejects plain Python, illegal DMA, scalar Parallel, and unguarded MTE3."""
    plain_py = """
import torch

def kernel(x: torch.Tensor) -> torch.Tensor:
    return x + 1.0
"""
    sa_py = run_stage_a_static(plain_py)
    assert sa_py.passed is False
    assert sa_py.backend_kind == "python_kernel"
    assert sa_py.fault_plane == "STAGE_A_STATIC"
    assert sa_py.failed_tier == "stage_a"
    assert "stage_a_non_kernel_callable" in sa_py.error_message

    sa_syntax = run_stage_a_static("def broken(:\n")
    assert sa_syntax.passed is False
    assert sa_syntax.fault_code == "STAGE_A_PARSE_OR_LOWER_ERROR"

    sa_empty = evaluate_ascendc_candidate("")
    assert sa_empty.passed is False
    assert sa_empty.fault_code == "ASCENDC_EMPTY_SOURCE"

    illegal_wrapper = """
ASCENDC_CODE = '''
// Illegal dav-3510 datapath: L1->GM and GM->L0A
'''
def kernel(x):
    return x
"""
    sa_bad_dma = run_stage_a_static(illegal_wrapper)
    assert sa_bad_dma.passed is False
    assert "forbidden_dav3510_dma_path" in sa_bad_dma.error_message

    bad_threads = (
        "import tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((1, 64), "float16")):\n'
        "    with T.Kernel(1, threads=128) as cid:\n"
        "        pass\n"
    )
    sa_threads = run_stage_a_static(bad_threads)
    assert sa_threads.passed is False
    assert "forbidden_kernel_threads_kwarg" in sa_threads.error_message

    for bad_expr in ("T.exp(x_ub[0, i])", 'T.Cast("float16", x_ub[0, i])'):
        sa_math = run_stage_a_static(_parallel_math_module(bad_expr))
        assert sa_math.passed is False
        assert "forbidden_tracka_parallel_scalar_math" in sa_math.error_message
    assert run_stage_a_static(_parallel_math_module("T.exp(x_ub[0, i])", simt_vf=True)).passed

    unguarded_1c2v = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def main(X: T.Tensor((64, 64), "float16"), Y: T.Tensor((64, 64), "float16")):
            with T.MixedKernel(1, sids=2) as (cid, sid):
                ub = T.alloc_shared((64, 64), "float16")
                T.copy(X, ub)
                T.copy(ub, Y)
        """
    )
    sa_1c2v = run_stage_a_static(unguarded_1c2v)
    assert sa_1c2v.passed is False
    assert sa_1c2v.fault_code == "UNCOVERED_1C2V_SUBCORE_MTE3_WRITE"

    sa_cube_req = run_stage_a_static(_PRIM_FUNC_KERNEL, require_cube_engine=True)
    assert sa_cube_req.passed is False
    assert "missing_cube_engine_primitives" in sa_cube_req.error_message


def test_evaluate_ascendc_candidate_tokens_occupancy_and_funnel_callbacks() -> None:
    """Verify AscendC forbidden tokens, occupancy/bank risk, monotonicity, and GateFunnel."""
    clean_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "  x[0] = Reg::Exp(x[0]);\n"
        "}\n"
    )
    assert evaluate_ascendc_candidate(clean_cpp).passed is True

    bad_cpp = clean_cpp.replace("Reg::Exp(x[0])", "AscendC::Exp(dst, x[0], mask)")
    bad_res = evaluate_ascendc_candidate(bad_cpp)
    assert bad_res.passed is False
    assert bad_res.error_message == "cce_forbidden_tokens: AscendC::"

    legacy_flag = clean_cpp.replace("x[0] = Reg::Exp(x[0]);", "set_flag(PIPE_V, PIPE_MTE3, 0);")
    assert evaluate_ascendc_candidate(legacy_flag).passed is False
    assert evaluate_ascendc_candidate(legacy_flag, baseline_cpp_source=legacy_flag).passed is True
    grown_flag = legacy_flag.replace("}", "  set_flag(PIPE_MTE3, PIPE_V, 0);\n}")
    assert evaluate_ascendc_candidate(grown_flag, baseline_cpp_source=legacy_flag).passed is False

    mx_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "#include <tl_templates/ascend/gemm.h>\n"
        'extern "C" __global__ __aicore__ void mm(__gm__ half* a, __gm__ half* b) {\n'
        "    asc_init();\n"
        "    asc_copy_gm2l1_nd2nz();\n"
        "    asc_mmad_mx();\n"
        "}\n"
    )
    assert (
        evaluate_ascendc_candidate(
            mx_cpp,
            freedom_mask=ConstraintRelaxationMask.top(),
            require_hardware_dma=True,
            require_cube_engine=True,
        ).passed
        is True
    )

    single_lane_simt = (
        "#include <tl_templates/ascend/common.h>\n"
        "__simt_vf__ __launch_bounds__(1) void vf(__gm__ float* x) { x[0] = 1.0f; }\n"
    )
    assert evaluate_ascendc_candidate(single_lane_simt).fault_code == "DEGENERATED_SINGLE_LANE_SIMT"

    launched_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "  __ubuf__ float* x_ub = (__ubuf__ float*)x;\n"
        "  x_ub[0] = *(x_ub + r * 256);\n"
        "}\n"
        'extern "C" int launch(void* stream) {\n'
        "  kernel_active<<<56, nullptr, stream>>>();\n"
        "  return 0;\n"
        "}\n"
    )
    launched = evaluate_ascendc_candidate(launched_cpp)
    assert launched.ledger is not None
    assert (launched.ledger.grid_blocks, launched.ledger.wave_occupancy) == (56, 1.0)
    assert launched.ledger.ub_bank_conflict_risk == 1.0
    narrow = evaluate_ascendc_candidate(launched_cpp.replace("r * 256", "r * 64"))
    assert narrow.ledger is not None and narrow.ledger.ub_bank_conflict_risk == 0.0

    simt_flags = resolve_trackb_compile_flags(
        "__simt_vf__ void vf(__gm__ float* x) {}\n", [*PURE_SIMDVF_UBUF_FLAGS, "-O2"]
    )
    assert simt_flags == ("-O2",)
    simd_flags = resolve_trackb_compile_flags(
        "// __simt_vf__ comment\n__simd_vf__ void vf(__ubuf__ float* x) {}\n", ["-O2"]
    )
    assert all(f in simd_flags for f in PURE_SIMDVF_UBUF_FLAGS)

    funnel = GateFunnel(
        skip_stage_b=False,
        compile_runner=lambda _src: StageBResult(passed=True, spill_bytes=0, vloopv2_hit=True),
        device_runner=lambda _src, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=5.0,
            residual=compute_static_dynamic_residual(
                st_a.ledger,
                {
                    "dur_us": 5.0,
                    "spill_bytes": 256,
                    "bank_conflict_ratio": 0.25,
                    "l2_hit_rate": 0.60,
                },
            ),
            telemetry={"dur_us": 5.0, "spill_bytes": 256},
        ),
    )
    ev = funnel.evaluate(
        _PRIM_FUNC_KERNEL,
        roofline={"min_gm_bytes": 16384, "compute_floor_us": 2.0},
    )
    assert ev.passed is True
    assert ev.stage_c is not None and ev.stage_c.residual is not None
    assert ev.stage_c.residual.observation is not None
    clean_score = compute_eq3_hap_objective(
        5.0, ref_perf_us=15.0, t_hw_us=2.5, spill_bytes=0, bank_conflict_ratio=0.0, l2_hit_rate=1.0
    )
    pen_score = compute_eq3_hap_objective(
        5.0, ref_perf_us=15.0, t_hw_us=2.5, residual=ev.stage_c.residual
    )
    assert 0.0 < pen_score < clean_score
    sig = extract_structural_signature(
        _PRIM_FUNC_KERNEL,
        residual=ev.stage_c.residual,
        stage_b=StageBResult(passed=True, spill_bytes=64, vloopv2_hit=False),
    )
    assert {"hw:register_spill", "hw:low_l2_hit", "hw:vloopv2_miss"} <= set(sig)

    fail_b = GateFunnel(
        skip_stage_b=False,
        compile_runner=lambda _src: StageBResult(passed=False, error_message="compile_err"),
    ).evaluate(_PRIM_FUNC_KERNEL)
    assert fail_b.passed is False and fail_b.failed_stage == "stage_b"


def test_ast_semantic_gate_and_repair_monotonicity() -> None:
    """Verify AST semantic rules S-A1..S-A4, verify_tilelang_ast, and repair monotonicity."""
    bad_nested = dedent(
        """
        def build():
            @T.prim_func
            def kern(A: T.Tensor((16,), "float32")):
                def inner(x):
                    return x + 1
                with T.Kernel(1):
                    pass
            return kern
        """
    )
    assert [v.code for v in check_tilelang_semantic_contracts(bad_nested)] == [
        "NESTED_DEF_IN_PRIM_FUNC"
    ]

    bad_rebind = "kernel = lambda x: x\n"
    assert [
        v.code
        for v in check_tilelang_semantic_contracts(
            bad_rebind, protected_exports=frozenset({"kernel"})
        )
    ] == ["PROTECTED_SYMBOL_REBIND"]

    bad_nesting = dedent(
        """
        @T.prim_func
        def kern(A: T.Tensor((16, 16), "float32")):
            with T.Kernel(1):
                for i in T.Parallel(16):
                    T.copy(A[i, :], A[i, :])
                    for k in T.Pipelined(4, num_stages=2):
                        pass
                with T.SimdVF():
                    with T.Stage(0):
                        pass
                for k in T.Pipelined(4, num_stages=2):
                    with T.Stage(1):
                        pass
        """
    )
    codes = {v.code for v in check_tilelang_semantic_contracts(bad_nesting)}
    assert {
        "TILE_OP_INSIDE_PARALLEL",
        "PIPELINED_INSIDE_PARALLEL",
        "STAGE_INSIDE_LEAF_SCOPE",
        "NONZERO_STAGE_WITHOUT_ENABLE_OFFSET",
    } <= codes

    bad_simd_off = dedent(
        """
        @T.prim_func
        def kern(buf: T.Tensor((64,), "float32")):
            with T.Kernel(1):
                with T.SimdVF():
                    r = T.simd.vld(buf[0], off=0)
                    T.simd.vsts(buf[0], r, off=0)
        """
    )
    v_sa4 = check_tilelang_semantic_contracts(bad_simd_off)
    assert len(v_sa4) == 2 and all(v.code == "SIMD_FORBIDDEN_OFF_KWARG" for v in v_sa4)

    valid_pipelined = dedent(
        """
        @T.prim_func
        def kern(A: T.Tensor((16, 16), "float16")):
            with T.Kernel(1):
                a_ub = T.alloc_shared((16, 16), "float16")
                for k in T.Pipelined(4, num_stages=2, annotations={"enable_offset": True}):
                    with T.Stage(0):
                        T.copy(A, a_ub)
        """
    )
    assert check_tilelang_semantic_contracts(valid_pipelined) == ()
    assert verify_tilelang_ast(valid_pipelined).passed is True

    parent_ta = (
        "import tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((128, 64), "float16"), Y: T.Tensor((128, 64), "float16")):\n'
        "    with T.Kernel(1):\n"
        '        a_ub = T.alloc_shared((64, 64), "float16")\n'
        "        for ko in T.Pipelined(2, num_stages=2):\n"
        "            T.copy(X[ko * 64 : (ko + 1) * 64, :], a_ub)\n"
        "            with T.SimdVF():\n"
        "                mask = T.simd.pset(32)\n"
        "                v = T.simd.vld(a_ub[0])\n"
        "                T.simd.vsts(a_ub[0], T.simd.vexp(v, mask), mask)\n"
        "            T.copy(a_ub, Y[ko * 64 : (ko + 1) * 64, :])\n"
    )
    assert (
        check_repair_monotonicity(
            parent_ta,
            parent_ta.replace("(64, 64)", "(32, 64)").replace("num_stages=2", "num_stages=1"),
        )
        == []
    )
    assert any(
        "monotonicity_regression_pipeline_dropped" in err
        for err in check_repair_monotonicity(
            parent_ta, parent_ta.replace("T.Pipelined(2, num_stages=2)", "T.serial(2)")
        )
    )
    assert any(
        "monotonicity_regression_trackb_scalar_fallback" in err
        for err in check_repair_monotonicity(
            "__simd_vf__ void vf(__ubuf__ float* a) { simd_inst::vexp(a, a, 64); }\n",
            "void vf(__ubuf__ float* a) { for (int i = 0; i < 64; ++i) a[i] += 1.0f; }\n",
            backend_kind="ascendc_trackb",
        )
    )


def test_npu_entrypoint_purity_decoy_host_fallback_and_cache_key_audit() -> None:
    """Verify check_npu_entrypoint_purity rejects decoys, host fallbacks, and polluted keys."""
    host_fallback_mod = (
        "import torch\nimport tilelang\nimport tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def _main(X: T.Tensor((1, 64), "float16")):\n    pass\n\n'
        "def op_fn(x):\n"
        "    if x.device.type != 'npu':\n        return torch.matmul(x, x)\n"
        "    return torch.rms_norm(x)\n"
    )
    errs_fb = check_npu_entrypoint_purity(host_fallback_mod, "op_fn")
    assert any("npu_host_fallback_forbidden:op_fn->rms_norm" in e for e in errs_fb)
    assert any("detached_decoy_primfunc:op_fn" in e for e in errs_fb)

    uncached_mod = (
        "import tilelang\nimport tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def _main(X: T.Tensor((1, 64), "float16")):\n    pass\n\n'
        "def op_fn(x):\n"
        '    return tilelang.compile(_main, target="ascend")(x)\n'
    )
    assert check_npu_entrypoint_purity(uncached_mod, "op_fn") == []
    assert any(
        "missing_kernel_cache" in e
        for e in check_npu_entrypoint_purity(uncached_mod, "op_fn", require_kernel_cache=True)
    )

    def _wrap_cache(key_stmt: str) -> str:
        return (
            "import ctypes\nimport tilelang\nimport tilelang.ascend.language as T\n\n"
            "_KERNEL_CACHE: dict[object, object] = {}\n\n"
            "@T.prim_func\n"
            'def _main(X: T.Tensor((1, 64), "float16")):\n    pass\n\n'
            "def op_fn(x, eps: float = 1e-5, scale: float = 0.125):\n"
            f"    {key_stmt}\n"
            "    k = _KERNEL_CACHE.get(key)\n"
            "    if k is None:\n"
            '        k = tilelang.compile(_main, target="ascend")\n'
            "        _KERNEL_CACHE[key] = k\n"
            "    return k(x)\n"
        )

    assert (
        check_npu_entrypoint_purity(
            _wrap_cache('key = (x.shape[-1], str(x.dtype), "aligned")'),
            "op_fn",
            require_kernel_cache=True,
        )
        == []
    )
    for bad_stmt in (
        "key = (x.shape[-1], eps)",
        "key = (x.shape[-1], float(scale))",
        "key = (id(x), str(x.dtype))",
        "key = (x.data_ptr(), str(x.dtype))",
        "key = (ctypes.addressof(x), str(x.dtype))",
        "key = (x.storage_offset(), str(x.dtype))",
    ):
        errs = check_npu_entrypoint_purity(
            _wrap_cache(bad_stmt), "op_fn", require_kernel_cache=True
        )
        assert any("polluted_kernel_cache_key" in e for e in errs), (bad_stmt, errs)


def test_capability_router_track_a_b_escalation_and_isa_gap_scanner() -> None:
    """Verify CapabilityGate, t=0/t>0 Track-A/B routing, case ratios, and ISA gap scanner."""
    import tilelang.ascend.language as T  # noqa: PLC0415

    @T.prim_func
    def static_gemm(
        A: T.Tensor((128, 64), "bfloat16"),
        B: T.Tensor((128, 64), "bfloat16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(56):
            a_l1 = T.alloc_l1((128, 64), "bfloat16")
            b_l1 = T.alloc_l1((128, 64), "bfloat16")
            c_l0c = T.alloc_l0c((128, 128), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, c_l0c, transpose_B=True)
            T.copy(c_l0c, C)

    dev_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 16},
        waterlines={},
    )
    d_dev = route_execution_track(static_gemm, ledger=dev_ledger)
    assert d_dev.track == "developer"
    assert d_dev.topology_axes == TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    assert d_dev.features is not None
    assert d_dev.features["cube_to_vector_ratio"] == pytest.approx(4.0)
    assert CapabilityGate.suitability_score(d_dev.topology_axes) == 1.0
    assert (
        CapabilityGate.route(
            TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt"), router_enabled=False
        ).track
        == "developer"
    )

    @T.prim_func
    def ssbuf_indirect(
        src: T.Tensor((64, 64), "float8_e4m3fn"),
        extra: T.Tensor((64, 32), "float4_e2m1fnx2"),
        idx: T.Tensor((16,), "int32"),
        out: T.Tensor((16, 64), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            ub = T.alloc_shared((16, 64), "float32")
            ss[sid] = T.Cast("uint32", sid)
            for i in T.serial(16):
                row_idx = idx[i]
                for j in T.serial(64):
                    ub[i, j] = T.Cast("float32", src[row_idx, j])
            T.copy(ub, out)

    d_ss = route_execution_track(ssbuf_indirect)
    assert d_ss.track == "manual_ssbuf"
    assert d_ss.features is not None and d_ss.features["quant_granularity"] == "mixed_fp8_fp4"

    sub_ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=2.0,
        contig_atom_ratio=0.5,
        is_atom_aligned=False,
        working_set_sram_ratio=0.2,
        working_set_vreg_ratio=0.2,
        wave_occupancy_ratio=0.5,
        oi_ridge_ratio=0.25,
        inner_contiguous_bytes=16,
    )
    d_ratio = apply_case_ratios(
        route_from_op_spec({"category": "Vector", "formula": "y = rsqrt(mean(x^2))"}),
        sub_ratios,
    )
    assert d_ratio.track == "simt_vf"
    assert d_ratio.topology_axes.iteration_regularity == "irregular_simt"

    assert (
        route_initial_by_input_contract(
            {
                "op_name": "unseen_fp8_gemm",
                "input_dtypes": ["fp8e4m3", "i8"],
                "output_dtypes": ["f16"],
            }
        ).recommended_backend_tier
        == "tilelang_dsl"
    )
    for hard_spec in (
        {"op_name": "q4_op", "input_dtypes": ["f16", "q4_0"], "output_dtypes": ["f16"]},
        {"op_name": "cas_op", "input_dtypes": ["uint64"], "requires_inplace_gm_cas": True},
        {
            "op_name": "vm_op",
            "input_dtypes": ["float32"],
            "topology_axes": {
                "compute_engine": "aiv_only",
                "memory_ring": "persistent_onchip_state",
                "iteration_regularity": "dynamic_opcode_stream",
            },
        },
    ):
        dec = route_initial_by_input_contract(hard_spec)
        assert dec.recommended_backend_tier == "ascendc_template"
        assert dec.recommended_backend_tier in VALID_BACKEND_TIERS

    assert (
        evaluate_track_escalation(
            current_tier="tilelang_dsl",
            roofline_eta=0.90,
            target_eta=0.85,
            stagnation_rounds=3,
            compiler_summary=BishengResourceSummary(),
        ).should_escalate
        is False
    )
    esc_vf = evaluate_track_escalation(
        current_tier="tilelang_dsl",
        roofline_eta=0.45,
        target_eta=0.85,
        stagnation_rounds=2,
        dominant_bound="ub_load_store",
        compiler_summary=BishengResourceSummary(vf_fusion_breaks=3),
        structural_features={"requires_dist_unpack": True},
    )
    assert isinstance(esc_vf, EscalationDecision)
    assert esc_vf.should_escalate is True and esc_vf.to_tier == "inline_ascendc_vf"

    esc_tpl = evaluate_track_escalation(
        current_tier="tilelang_dsl",
        roofline_eta=0.35,
        target_eta=0.85,
        stagnation_rounds=2,
        dominant_bound="cube",
        compiler_summary=BishengResourceSummary(alias_serialized_buffers=2),
        structural_features={
            "cube_to_vector_ratio": 1.0,
            "dynamic_cross_core_handoff": True,
            "has_dead_l1_slice": True,
        },
    )
    assert esc_tpl.should_escalate is True and esc_tpl.to_tier == "ascendc_template"

    live_report = scan_tilelang_pipe_gaps()
    assert isinstance(live_report, PipeGapReport)
    assert tuple(live_report.pipes.keys()) == VALID_HARDWARE_PIPES
    assert all(isinstance(v, PipeGapBreakdown) for v in live_report.pipes.values())
    assert 0.0 < live_report.coverage_ratio < 1.0
    assert live_report.has_gap("simt_gm_atomic_cas") is True

    mock_gap = scan_tilelang_pipe_gaps(
        tilelang_surface={"simd.vld", "simd.vst2", "shfl_sync"},
        pipes=("pipe_v_simd_simt",),
    )
    assert "simd_vld_vst" in mock_gap.supported_primitives
    assert "simt_gm_atomic_cas" in mock_gap.missing_primitives

    with pytest.raises(FileNotFoundError):
        scan_ascend_sdk_headers(header_roots=["/nonexistent/asc_sdk_path_xyz"])
    fallback_sdk = scan_ascend_sdk_headers(
        header_roots=["/nonexistent/asc_sdk_path_xyz"],
        allow_offline_catalog=True,
    )
    assert fallback_sdk.fallback_used is True and len(fallback_sdk.symbols) >= 1


def test_tir_watermarks_overflow_detection_and_symbolic_var_bindings() -> None:
    """Verify TIR watermark accounting, UB/L1 overflow rejection, and symbolic dim bindings."""
    target = AscendTarget("ascend.950pr")
    sa_legal = run_stage_a_static(_PRIM_FUNC_KERNEL, target=target, enable_track_router=True)
    assert sa_legal.passed is True and sa_legal.ledger is not None
    assert "UB" in sa_legal.ledger.waterlines
    assert sa_legal.ledger.waterlines["UB"].overflow_bytes == 0
    assert sa_legal.ledger.max_overflow_bytes == 0
    assert sa_legal.route is not None

    ub_overflow_dsl = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def main(X: T.Tensor((2048, 256), "float32"), Y: T.Tensor((2048, 256), "float32")):
            with T.Kernel(1):
                big_ub = T.alloc_shared((2048, 256), "float32")
                T.copy(X, big_ub)
                T.copy(big_ub, Y)
        """
    )
    sa_overflow = run_stage_a_static(ub_overflow_dsl, target=target)
    assert sa_overflow.passed is False
    assert sa_overflow.fault_code == "STAGE_A_STATIC_ILLEGAL"
    assert sa_overflow.ledger is not None
    assert sa_overflow.ledger.max_overflow_bytes > 0
    assert sa_overflow.ledger.waterlines["UB"].overflow_bytes > 0

    sparse_dsl = dedent(
        """
        import tilelang.ascend.language as T

        def _build_sparse_scalar_kernel(s1_dim: int, s2_dim: int, n_kv: int, top_k: int):
            total_elems = T.dynamic("total_elems", "int32")

            @T.prim_func
            def main(
                X: T.Tensor((1048576,), "float16"),
                Y: T.Tensor((total_elems,), "float16"),
                num_blocks: T.int32,
                s1_arg: T.int32,
                s2_arg: T.int32,
                n_kv_arg: T.int32,
                top_k_arg: T.int32,
            ):
                with T.Kernel(56) as cid:
                    ub = T.alloc_shared((256,), "float16")
                    for b in T.serial(num_blocks):
                        idx = (b * 56 + cid) * 256
                        T.copy(X[idx : idx + 256], ub)
                        T.copy(ub, Y[idx : idx + 256])
                    _guard = (
                        s1_arg + s2_arg + n_kv_arg + top_k_arg + s1_dim + s2_dim + n_kv + top_k
                    )

            return main
        """
    )
    dims: dict[str, int] = {"B": 2, "S1": 64, "S2": 2048, "N1": 16, "N2": 4, "topK": 128, "T": 32}
    sa_sparse = run_stage_a_static(
        sparse_dsl,
        factory_name="_build_sparse_scalar_kernel",
        dims=dims,
        target=target,
    )
    assert sa_sparse.passed is True, sa_sparse.error_message
    assert sa_sparse.ledger is not None
    assert sa_sparse.ledger.is_legal is True
    assert sa_sparse.ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0) > 0
    assert sa_sparse.ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0) < 1_000_000

    direct = inspect_tilelang_primfunc(sa_sparse.prim_func, target=target, var_bindings=dims)
    assert direct.ledger.is_legal is True

    trackb_ub_overflow = (
        "#include <tl_templates/ascend/common.h>\n"
        "constexpr int BIG_BYTES = 1048576;\n"
        "__ubuf__ float* big = (__ubuf__ float*)0;\n"
        "__ubuf__ float* tail = (__ubuf__ float*)(BIG_BYTES);\n"
        'extern "C" __global__ __vector__ void k(__gm__ float* x) { x[0] = 1.0f; }\n'
    )
    sa_tb_over = evaluate_ascendc_candidate(trackb_ub_overflow, enable_track_router=True)
    assert sa_tb_over.passed is False
    assert sa_tb_over.fault_code == "UB_OVERFLOW"
    assert sa_tb_over.ledger is not None and sa_tb_over.ledger.max_overflow_bytes > 0

    shapes = extract_ordered_tensor_shapes(
        case_entry={"input_shape": [[16, 2048], [2048]], "attrs": {"M": 16, "D": 2048}},
        proto={"outputs": [{"shape": "[M, D]"}]},
    )
    assert shapes == ((16, 2048), (2048,), (16, 2048))
    assert build_scope_waterline(
        scope="UB", capacity_bytes=262144, explicit_buffers={"a": 1024}, implicit_buffers={}
    ).overflow_bytes == 0
