"""Tests for W5 Track-B ledger, Stage-B dump order, AST-safe slicing, and wiring."""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import tilefoundry_ascend.agent.orchestrator.e2e_evolution as e2e_mod
import tilefoundry_ascend.agent.orchestrator.portfolio_runner as portfolio_mod
from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
    extract_ordered_tensor_shapes,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import PerfTargetCampaignRunner
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    _compact_trackb_repair_source,
    _extract_lowered_compiler_excerpt,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.schema import build_predicate_context
from tilefoundry_ascend.agent.orchestrator import (
    BeamCandidate,
    EvolutionTrajectory,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import build_trackb_context_view
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    _TRACKB_CORE_TOKENS,
    _build_ascendc_static_ledger,
    build_stage_b_driver_source,
    evaluate_ascendc_candidate,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_exploit_request,
    build_syntax_repair_request,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask


def _sample_dsl_source() -> str:
    return (
        "import tilelang.ascend.language as T\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(\n"
        "        X: T.Tensor((128, 128), 'bfloat16'),\n"
        "        W: T.Tensor((128, 128), 'bfloat16'),\n"
        "        Y: T.Tensor((128, 128), 'float32'),\n"
        "    ):\n"
        "        with T.Kernel(2):\n"
        "            x_l1 = T.alloc_l1((128, 64), 'bfloat16')\n"
        "            w_l1 = T.alloc_l1((128, 64), 'bfloat16')\n"
        "            acc = T.alloc_l0c((128, 128), 'float32')\n"
        "            for kt in T.Pipelined(2, num_stages=2):\n"
        "                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)\n"
        "                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)\n"
        "                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)\n"
        "            T.copy(acc, Y[0:128, 0:128])\n"
        "    return main\n"
    )


def test_trackb_static_ledger_dual_subcore_and_all_fields() -> None:
    """Verify dual-subcore AIV0/AIV1 PIPE_V split, 11 Track-B fields, and ascend_gemm_l1."""
    assert "ascend_gemm_l1" in _TRACKB_CORE_TOKENS

    cpp_source = """
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 128;
constexpr int STAGES = 2;

__aicore__ inline void helper_compute(
    __ubuf__ half* dst_ub,
    __ubuf__ half* src_ub,
    __gm__ half* gm_in,
    __gm__ half* gm_out,
    int loop_k
) {
    int sid = get_subblockid();
    __ubuf__ half* large_ub = (__ubuf__ half*)(0x38000);
    __simd_vf__ void (*vf_tag)() = nullptr;
    __cbuf__ half* l1_a = (__cbuf__ half*)(0x0000);
    __cbuf__ half* l1_b = (__cbuf__ half*)(0x20000);
    asc_copy_gm2l1_nd2nz(l1_a, gm_in, 128, static_cast<asc_load_l2_cache_mode>(0), 32, 128, 0, 0);
    asc_copy_gm2l1_nd2nz(l1_b, gm_in, 128, static_cast<asc_load_l2_cache_mode>(0), 32, 128, 0, 0);
    ascend_gemm_l1<16, 32, 32>(l1_a, l1_b);
    __ubuf__ int4b_t* q_ptr = (__ubuf__ int4b_t*)stage_ptr;
    for (int k = 0; k < loop_k; ++k) {
        int stg = k & 1;
        __ubuf__ half* stage_ptr = large_ub + stg * 4096;
        asc_copy_gm2ub(stage_ptr, gm_in + k * 64, 1, 64, 0, 0);
        asc_sync_notify(PIPE_MTE2, PIPE_V, 0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, 0);
        simd_inst::vexp(stage_ptr, stage_ptr, 64);
        simd_inst::vadd(dst_ub, stage_ptr, src_ub, 64);
        simd_inst::vmul(dst_ub, dst_ub, stage_ptr, 64);
        asc_dequant_s4(dst_ub, stage_ptr, 64);
        row_max_update(dst_ub);
        pipe_barrier(PIPE_ALL);
        asc_copy_ub2gm(gm_out + k * 64, dst_ub, 1, 64, 0, 0);
    }
    if (sid == 0) {
        simd_inst::vadd(dst_ub, dst_ub, src_ub, 64);
    }
}

extern "C" __global__ __aicore__ void test_trackb_kernel(
    __gm__ half* x, __gm__ half* y
) {
    __ubuf__ half* ub0 = (__ubuf__ half*)(0x0000);
    __ubuf__ half* ub1 = (__ubuf__ half*)(0x4000);
    helper_compute(ub0, ub1, x, y, 4);
}
"""
    ledger = _build_ascendc_static_ledger(cpp_source)
    assert ledger.pipe_cycles.get("AIV0.PIPE_V", 0) > 0
    assert ledger.pipe_cycles.get("AIV1.PIPE_V", 0) > 0
    assert abs(
        ledger.pipe_cycles["AIV0.PIPE_V"] - ledger.pipe_cycles["AIV1.PIPE_V"]
    ) <= 1
    assert ledger.uses_unreserved_ubuf is True
    assert ledger.in_loop_pipe_all_barriers >= 1
    assert ledger.min_gm_dma_bytes > 0
    assert 0.0 < ledger.dma_burst_efficiency_ratio <= 1.0
    assert ledger.dominant_contig_bytes > 0
    assert ledger.effective_cube_m >= 0
    assert ledger.has_lowbit_quant is True
    assert ledger.has_online_rescale_loop is True
    assert ledger.l1_bank_conflict_risk >= 0.0
    assert ledger.mte2_inflight_per_aiv >= 1
    assert ledger.uses_dynamic_offset_pipelining is True

    gemm_cpp = """
extern "C" __global__ __aicore__ void test_gemm_l1_kernel(
    __gm__ half* x, __gm__ half* w, __gm__ float* y
) {
    __cbuf__ half* l1_a = (__cbuf__ half*)(0x0000);
    __cbuf__ half* l1_b = (__cbuf__ half*)(0x10000);
    __cc__ float* l0c = (__cc__ float*)(0x0000);
    asc_copy_gm2l1_nd2nz(l1_a, x, 128, static_cast<asc_load_l2_cache_mode>(0), 32, 128, 0, 0);
    asc_copy_gm2l1_nd2nz(l1_b, w, 128, static_cast<asc_load_l2_cache_mode>(0), 32, 128, 0, 0);
    ascend_gemm_l1<16, 32, 32>(l1_a, l1_b);
    asc_copy_l0c2gm(
        y, l0c, (uint16_t)32, (uint16_t)16, (uint32_t)32, (uint16_t)16,
        static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0),
        static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0),
        false, true, false, false
    );
}
"""
    stage_a = evaluate_ascendc_candidate(
        gemm_cpp,
        freedom_mask=ConstraintRelaxationMask.top(),
    )
    assert stage_a.passed is True, stage_a.error_message
    assert stage_a.ledger is not None
    assert stage_a.ledger.effective_cube_m == 16
    assert stage_a.ledger.min_gm_dma_bytes == 2048


def test_stage_b_driver_dumps_before_device_compile() -> None:
    """Verify build_stage_b_driver_source compiles once and salvages dumps on failure."""
    src = build_stage_b_driver_source(
        Path("/tmp/fake_kernel.py"),
        factory_name="make_kernel",
        tir_dump_path="/tmp/fake.tir",
        kernel_dump_path="/tmp/fake.cce",
    )
    session_src = src[src.index("with _ascend_compile_session(_merged_pc):") :]
    pre_idx = session_src.index("enable_device_compile=False")
    tir_dump_idx = session_src.index("tir_dump_path")
    cce_dump_idx = session_src.index("kernel_dump_path")
    post_idx = session_src.index("enable_device_compile=True")
    assert post_idx < pre_idx
    assert post_idx < cce_dump_idx
    assert post_idx < tir_dump_idx


def test_lowered_compiler_excerpt_prioritizes_cce_and_fences() -> None:
    """Verify .cce diagnostics prioritize generated_cce and fences track dialect."""
    cce_lines = [f"// cce line {idx}" for idx in range(1, 40)]
    tir_lines = [f"# tir line {idx}" for idx in range(1, 40)]
    stage_b = StageBResult(
        passed=False,
        error_message="error at /tmp/out/kernel.cce:25:3: undeclared identifier",
        generated_cce="\n".join(cce_lines),
        lowered_tir="\n".join(tir_lines),
    )
    excerpt = _extract_lowered_compiler_excerpt(stage_b, window=2)
    assert "25: // cce line 25" in excerpt
    assert "# tir line" not in excerpt

    tir_only = StageBResult(
        passed=False,
        error_message="error at /tmp/out/tir.py:10:1: type mismatch",
        generated_cce="",
        lowered_tir="\n".join(tir_lines),
    )
    tir_excerpt = _extract_lowered_compiler_excerpt(tir_only, window=1)
    assert "10: # tir line 10" in tir_excerpt

    py_repair_req = build_syntax_repair_request(
        _sample_dsl_source(),
        "compile failure",
        lowered_excerpt="10: # tir line 10",
        lowered_excerpt_lang="python",
    )
    assert "```python\n10: # tir line 10\n```" in py_repair_req.user_prompt

    cpp_repair_req = build_syntax_repair_request(
        _sample_dsl_source(),
        "compile failure",
        lowered_excerpt="25: // cce line 25",
        lowered_excerpt_lang="cpp",
    )
    assert "```cpp\n25: // cce line 25\n```" in cpp_repair_req.user_prompt

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=10,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
        gm_roundtrip_bytes=1024,
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 5.0})
    cand = BeamCandidate(
        candidate_id="p0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_sample_dsl_source(),
        track="unconstrained",
        ledger=ledger,
        residual=res,
        latency_us=5.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(_sample_dsl_source()),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    rule = next(iter(tracker.rules.values()))
    req_py = build_exploit_request(cand, rule, escalation_tier="tilelang_dsl")
    assert "```python\n" in req_py.user_prompt
    req_cpp = build_exploit_request(cand, rule, escalation_tier="ascendc_trackb")
    assert "```cpp\n" in req_cpp.user_prompt


def test_compact_trackb_repair_source_ast_safe_and_focus_lines() -> None:
    """Verify hybrid_trackb slicing preserves Python AST, prologue offset, and focus lines."""
    prologue = "".join(f"#define PROLOGUE_MACRO_{i} {i}\n" for i in range(10))
    body_lines = [
        'extern "C" __global__ __aicore__ void big_kernel(__gm__ half* x) {',
        "    int acc = 0;",
    ]
    for idx in range(1, 520):
        if idx == 410:
            body_lines.append("    TARGET_ERROR_MARKER(x, 410);")
        else:
            body_lines.append(f"    int filler_var_{idx} = {idx};")
    body_lines.append("}")
    cpp_body = "\n".join(body_lines)

    hybrid_mod = (
        "def _render_trackb_tu(trackb_src: str) -> str:\n"
        f"    return {prologue!r} + trackb_src\n\n"
        'TRACKB_KERNEL_SOURCE = r"""\n'
        f"{cpp_body}\n"
        '"""\n\n'
        "def kernel():\n"
        "    return _render_trackb_tu(TRACKB_KERNEL_SOURCE)\n"
    )
    rendered_err_line = 10 + 1 + 412
    diag = f"/tmp/tu.cce:{rendered_err_line}:5: error: use of undeclared identifier"
    compacted = _compact_trackb_repair_source(hybrid_mod, diag, max_bytes=4500)
    parsed = ast.parse(compacted)
    fn_names = [
        node.name for node in parsed.body if isinstance(node, ast.FunctionDef)
    ]
    assert "_render_trackb_tu" in fn_names
    assert "kernel" in fn_names
    assert len(compacted.encode("utf-8")) <= 4800
    assert "TARGET_ERROR_MARKER" in compacted

    filler_block = "\n".join(f"    int line_{i} = {i};" for i in range(1, 180))
    standalone_cpp = (
        "__aicore__ inline void helper_fn(__ubuf__ half* ptr) {\n"
        f"{filler_block}\n"
        "    int FOCUS_LINE_SENTINEL = 999;\n"
        f"{filler_block}\n"
        "}\n"
        'extern "C" __global__ __aicore__ void big_kernel(__gm__ half* x) {\n'
        "    __ubuf__ half* p = (__ubuf__ half*)(0x0);\n"
        "    helper_fn(p);\n"
        "}\n"
    )
    focus_line_num = next(
        idx
        for idx, line in enumerate(standalone_cpp.splitlines(), start=1)
        if "FOCUS_LINE_SENTINEL" in line
    )
    truncated_view = build_trackb_context_view(
        standalone_cpp,
        active_kernel="big_kernel",
        max_bytes=1200,
        focus_lines=(focus_line_num,),
        focus_window=2,
    )
    assert "FOCUS_LINE_SENTINEL" in truncated_view


def test_campaign_runner_telemetry_hazards_and_pure_l0_parent_cost() -> None:
    """Verify stack_bytes, scalar_compute_loop_fallback, dcache_bypass, and pure-L0 parity."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=192 * 1024,
                explicit_buffers={"tmp": 8192},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=2048,
    )
    res = compute_static_dynamic_residual(
        ledger,
        {
            "dur_us": 12.0,
            "scalar_compute_loop_fallback": 3.0,
            "dcache_bypass_narrow_store": 2.0,
        },
    )
    proxy_res = SimpleNamespace(
        **{
            **res.to_dict(),
            "spill_bytes": res.spill_bytes,
            "stack_bytes": 128,
            "vloopv2_hit": res.vloopv2_hit,
            "vf_fusion_breaks": res.vf_fusion_breaks,
            "alias_serialized_buffers": res.alias_serialized_buffers,
            "dual_issue_ipc_collapse": res.dual_issue_ipc_collapse,
            "dcache_dcci_flushes": res.dcache_dcci_flushes,
            "has_local_membar_in_vf_call": res.has_local_membar_in_vf_call,
            "scalar_compute_loop_fallback": 3.0,
            "dcache_bypass_narrow_store": 2.0,
            "measured_aic_mac_ratio": res.measured_aic_mac_ratio,
            "measured_aiv_vec_ratio": res.measured_aiv_vec_ratio,
            "measured_mte2_wait_ratio": res.measured_mte2_wait_ratio,
            "measured_sync_stall_ratio": res.measured_sync_stall_ratio,
            "unexplained_gap_ratio": res.unexplained_gap_ratio,
            "to_dict": res.to_dict,
        }
    )
    section = PerfTargetCampaignRunner._format_msprof_section(proxy_res)
    assert "stack_bytes: 128" in section
    assert "scalar_compute_loop_fallback: 3.0" in section
    assert "dcache_bypass_narrow_store: 2.0" in section

    src_text = Path(inspect.getfile(PerfTargetCampaignRunner)).read_text(encoding="utf-8")
    assert "self._cost_advisor.predict_candidate_us(" in src_text


def test_ordered_tensor_shapes_and_prev_residual_wiring(tmp_path: Path) -> None:
    """Verify ordered_tensor_shapes extraction and forwarding across runner, distiller, e2e."""
    extracted = extract_ordered_tensor_shapes(
        case_entry={"input_shapes": [[16, 64], [16, 64]], "output_shapes": [[16, 64]]}
    )
    assert extracted == ((16, 64), (16, 64), (16, 64))

    proto_like = {
        "outputs": [{"shape": "[8, 32]"}],
    }
    assert extract_ordered_tensor_shapes(proto=proto_like) == ((8, 32),)
    assert extract_ordered_tensor_shapes(dims={"ordered_tensor_shapes": [[32, 64]]}) == (
        (32, 64),
    )

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_V": 64},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=192 * 1024,
                explicit_buffers={"ub": 2048},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=4096,
    )
    parent_res = compute_static_dynamic_residual(ledger, {"dur_us": 10.0})
    parent = BeamCandidate(
        candidate_id="p0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_sample_dsl_source(),
        track="unconstrained",
        ledger=ledger,
        residual=parent_res,
        latency_us=10.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(_sample_dsl_source()),
    )
    seen_shapes: list[object] = []

    class _SpyFunnel(GateFunnel):
        def evaluate(self, dsl_source: str, **kwargs: object):
            seen_shapes.append(kwargs.get("ordered_tensor_shapes"))
            return super().evaluate(dsl_source, **kwargs)

    funnel = _SpyFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=8.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 8.0}),
            telemetry={"dur_us": 8.0},
        ),
    )
    backend = MockLLMBackend(default_response=_sample_dsl_source())
    runner = PortfolioRunner(
        backend=backend,
        funnel=funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
    )

    seen_ctx_calls: list[dict[str, object]] = []
    orig_bpc = portfolio_mod.build_predicate_context

    def _spy_bpc(*args: object, **kwargs: object):
        seen_ctx_calls.append(dict(kwargs))
        return orig_bpc(*args, **kwargs)

    portfolio_mod.build_predicate_context = _spy_bpc
    try:
        report = asyncio.run(
            runner.run_round(
                parent,
                round_idx=1,
                factory_name="make_kernel",
                ordered_tensor_shapes=((16, 64), (16, 64)),
            )
        )
    finally:
        portfolio_mod.build_predicate_context = orig_bpc

    assert report.evaluated_candidates
    assert all(shape == ((16, 64), (16, 64)) for shape in seen_shapes)
    if "prev_residual" in inspect.signature(build_predicate_context).parameters:
        assert any("prev_residual" in call for call in seen_ctx_calls)
        assert any("prev_compiler_summary" in call for call in seen_ctx_calls)

    distill_funnel = _SpyFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=8.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 8.0}),
            telemetry={"dur_us": 8.0},
        ),
    )
    distiller = FlywheelDistiller(
        funnel=distill_funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        probe_runner=lambda _probe: {"baseline_us": 10.0, "variant_us": 8.0},
    )
    seen_shapes.clear()
    winner_dsl = _sample_dsl_source().replace("T.Kernel(2)", "T.Kernel(4)")
    traj = EvolutionTrajectory(
        problem_name="forward",
        track="unconstrained",
        total_rounds=1,
        baseline_candidate_id="base0",
        baseline_latency_us=10.0,
        baseline_estimated_cycles=100.0,
        best_candidate_id="win0",
        best_latency_us=6.0,
        best_estimated_cycles=60.0,
        speedup_vs_baseline=10.0 / 6.0,
        best_dsl_source=winner_dsl,
    )
    distiller.distill_from_trajectory(
        baseline_dsl=_sample_dsl_source(),
        trajectory=traj,
        factory_name="make_kernel",
        dims={"M": 16, "N": 64},
        ordered_tensor_shapes=((16, 64), (16, 64)),
    )
    assert ((16, 64), (16, 64)) in seen_shapes

    e2e_seen: list[object] = []
    golden_src = (
        "import torch\n"
        "def forward(x: torch.Tensor) -> torch.Tensor:\n"
        "    return x * 2.0\n"
    )

    class _E2ESpyFunnel(GateFunnel):
        def evaluate(self, dsl_source: str, **kwargs: object):
            e2e_seen.append(kwargs.get("ordered_tensor_shapes"))
            return super().evaluate(dsl_source, **kwargs)

    e2e_funnel = _E2ESpyFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=6.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 6.0}),
            telemetry={"dur_us": 6.0},
        ),
    )
    e2e_res = asyncio.run(
        e2e_mod.run_e2e_evolution(
            problem_name="forward",
            hir_source=golden_src,
            shape_contract="x: (16, 64)",
            backend=MockLLMBackend(default_response=_sample_dsl_source()),
            funnel=e2e_funnel,
            initial_dsl_source=_sample_dsl_source(),
            factory_name="make_kernel",
            num_rounds=1,
            k_rule_workers=1,
            anchor_dims={"M": 16, "N": 64},
            ordered_tensor_shapes=((16, 64), (16, 64)),
        )
    )
    assert e2e_res is not None
    assert ((16, 64), (16, 64)) in e2e_seen


def test_run_l0_l1_passes_lowered_excerpt_lang_to_repair() -> None:
    """Verify run_l0_l1_round_filter coordinates python fence for lowered_tir fallback."""
    backend = MockLLMBackend(default_response=_sample_dsl_source())
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(
            passed=False,
            error_message="error at tir.py:2:1: invalid",
            generated_cce="",
            lowered_tir="def main():\n    pass\n",
        ),
        device_runner=lambda _dsl, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=5.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 5.0}),
            telemetry={"dur_us": 5.0},
        ),
    )
    checks = asyncio.run(
        run_l0_l1_round_filter(
            [("c0", _sample_dsl_source())],
            patch_fn=lambda src: src,
            l0_checker=lambda src: run_stage_a_static(src, factory_name="make_kernel"),
            l1_compiler=lambda src: funnel._compile_runner(src)
            if funnel._compile_runner is not None
            else StageBResult(passed=False, error_message="missing"),
            repair_backend=backend,
            max_syntax_repairs=1,
            backend_kind="tilelang_dsl",
        )
    )
    assert checks
    assert backend.calls
    assert "```python\n" in backend.calls[0].user_prompt
