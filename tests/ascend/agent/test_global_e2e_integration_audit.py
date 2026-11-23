"""Comprehensive end-to-end integration audit suite for the 6 cross-module fixes."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    BayesianRuleTracker,
    build_predicate_context,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
    evaluate_ascendc_candidate,
    parse_cce_res_usage,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    PortfolioRunner,
    RoundEscalationReport,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    StaticDynamicResidual,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.router import (
    DEFAULT_ASCEND_SDK_HEADER_ROOTS,
    DEFAULT_TILELANG_CODEGEN_PATH,
    CapabilityGate,
    discover_tilelang_lowering_surface,
    resolve_canonical_topology_axes,
    route_from_op_spec,
    scan_ascend_sdk_headers,
)
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    HybridAscendCostModel,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    RegimePiecewiseSolver,
    recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil import (
    DEFAULT_OPS_ROOT,
    OpAdvisoryExtractor,
    RegimeSlice,
    SILVerifier,
    TopologyAxes,
    verify_source_against_facets,
)
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.templates import (
    VALID_FREEDOM_FACETS,
    ConstraintRelaxationMask,
    relax_facets_for_bottleneck,
)

_SIMPLE_VECTOR_DSL = """
import tilelang.ascend.language as T

def make_kernel(M: int = 64, N: int = 64):
    @T.prim_func
    def main(X: T.Tensor((64, 64), "float32"), Y: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((64, 64), "float32")
            T.copy(X, x_ub)
            T.copy(x_ub, Y)
    return main
"""


def test_audit1_stack_frame_not_spill_and_7_compiler_signals_e2e() -> None:
    """Stack Frame Size != spill, StageBResult carries summary, and signals reach KB & Beam."""
    spill_no_vloop, vloop_no_ev = parse_cce_res_usage(
        "Vector Register Usage: 32\nScalar Register Usage: 16\nStack Frame Size: 128 bytes\n"
    )
    assert spill_no_vloop == 0
    assert vloop_no_ev is False

    spill_clean, vloop_clean = parse_cce_res_usage(
        "Vector Register Usage: 32\nScalar Register Usage: 16\n"
        "Stack Frame Size: 128 bytes\nVLOOPv2 hit\n"
    )
    assert spill_clean == 0
    assert vloop_clean is True

    spill_real, vloop_miss = parse_cce_res_usage(
        "Register Spill: 64 bytes\nStack Frame Size: 128 bytes\nremark: VLOOPv2 miss: inner branch\n"
    )
    assert spill_real == 64
    assert vloop_miss is False

    comp_summary = BishengResourceSummary(
        vec_regs=64,
        scalar_regs=48,
        spill_bytes=128,
        stack_bytes=256,
        vloopv2_loops=0,
        software_loops=2,
        has_local_membar_in_vf_call=True,
        vf_fusion_breaks=3,
        alias_serialized_buffers=2,
        dual_issue_ipc_collapse=0.65,
        dcache_dcci_flushes=4,
    )
    funnel = GateFunnel(
        compile_runner=lambda _src: StageBResult(
            passed=True,
            spill_bytes=comp_summary.spill_bytes,
            vloopv2_hit=comp_summary.vloopv2_hit,
            compiler_summary=comp_summary,
        ),
        device_runner=lambda _src, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=42.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 42.0}),
            telemetry={"dur_us": 42.0},
        ),
    )
    ev = funnel.evaluate(_SIMPLE_VECTOR_DSL, factory_name="make_kernel")
    assert ev.passed is True
    assert ev.stage_b is not None and ev.stage_b.compiler_summary == comp_summary
    assert ev.stage_c is not None and ev.stage_c.residual is not None

    res = ev.stage_c.residual
    assert res.spill_bytes == 128
    assert res.vloopv2_hit is False
    assert res.vf_fusion_breaks == 3
    assert res.alias_serialized_buffers == 2
    assert res.dual_issue_ipc_collapse == pytest.approx(0.65)
    assert res.dcache_dcci_flushes == 4
    assert res.has_local_membar_in_vf_call is True

    assert StaticDynamicResidual.from_dict(res.to_dict()) == res

    sig = extract_structural_signature(_SIMPLE_VECTOR_DSL, residual=res, stage_b=ev.stage_b)
    assert "hw:register_spill" in sig
    assert "hw:vloopv2_miss" in sig
    assert "hw:vf_fusion_break" in sig
    assert "hw:alias_serialized" in sig
    assert "hw:ipc_collapse" in sig
    assert "hw:dcache_dcci" in sig
    assert "hw:local_membar_vf" in sig

    ctx = build_predicate_context(ev.stage_a.ledger, residual=res)
    for key in (
        "spill_bytes",
        "vloopv2_hit",
        "vf_fusion_breaks",
        "alias_serialized_buffers",
        "dual_issue_ipc_collapse",
        "dcache_dcci_flushes",
        "has_local_membar_in_vf_call",
    ):
        assert key in ctx
        assert key in ctx["prev"]

    compiler_rule = AscendCausalRule(
        rule_id="R_TEST_COMPILER_SIGNALS",
        regime_guard="True",
        precondition=(
            "vf_fusion_breaks >= 2 and alias_serialized_buffers >= 1 "
            "and dual_issue_ipc_collapse > 0.5 and dcache_dcci_flushes >= 1 "
            "and has_local_membar_in_vf_call and spill_bytes > 0 and (not vloopv2_hit)"
        ),
        transform_template="Fuse VF and eliminate alias/dcci/membar.",
        expected_postcondition="vf_fusion_breaks == 0 and spill_bytes == 0",
        source_provenance="audit",
    )
    assert compiler_rule.matches(ctx) is True


def test_audit2_reconciled_truth_tables_and_bayesian_axis_transfer_in_runners(tmp_path: Path) -> None:
    """Reconcile op_analyzer vs capability_gate, bridge via canonical axes, and wire runners."""
    if DEFAULT_OPS_ROOT.is_dir():
        extractor = OpAdvisoryExtractor(DEFAULT_OPS_ROOT)
        for op_name in (
            "attention_residual",
            "depthwise_conv_2d",
            "dilation_2d",
            "gcd",
            "top_k",
            "gqa",
            "grouped_matmul",
            "kimi_delta_attention",
        ):
            adv = extractor.extract(op_name)
            dec = route_from_op_spec(adv.to_dict())
            assert dec.topology_axes == adv.topology_axes == resolve_canonical_topology_axes(adv)

    unseen_axes = resolve_canonical_topology_axes(
        "unseen_sparse_attention_v2",
        features={
            "static_affine_bounds": False,
            "cube_to_vector_ratio": 1.0,
            "quant_granularity": "none",
            "indirect_sparse_gather": True,
            "cross_iter_recurrence": "associative_online_rescale",
            "dynamic_cross_core_handoff": True,
            "inner_contiguous_bytes": 128,
            "persistent_wave_occupancy": 1.0,
        },
    )
    assert unseen_axes.compute_engine == "coupled_1c2v"
    assert unseen_axes.iteration_regularity == "irregular_simt"
    assert CapabilityGate.suitability_score(unseen_axes) < 1.0

    tracker = BayesianRuleTracker.from_seed_yaml()
    funnel = GateFunnel(
        compile_runner=lambda _s: StageBResult(passed=True),
        device_runner=lambda _s, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=20.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 20.0}),
            telemetry={"dur_us": 20.0},
        ),
    )
    sa = funnel.evaluate(_SIMPLE_VECTOR_DSL, factory_name="make_kernel").stage_a
    assert sa.ledger is not None
    parent = BeamCandidate(
        candidate_id="seed_vec",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SIMPLE_VECTOR_DSL,
        track="developer",
        ledger=sa.ledger,
        residual=compute_static_dynamic_residual(sa.ledger, {"dur_us": 35.0}),
        latency_us=35.0,
        accuracy_passed=True,
    )
    creative_json = json.dumps(
        {
            "physical_hypothesis": "vector test",
            "target_local_metric": "dur_us",
            "dsl_source": _SIMPLE_VECTOR_DSL,
        }
    )
    runner = PortfolioRunner(
        backend=MockLLMBackend(
            responses={
                "exploit_rule": [_SIMPLE_VECTOR_DSL, _SIMPLE_VECTOR_DSL],
                "explore_creative": [creative_json, creative_json],
            }
        ),
        funnel=funnel,
        tracker=tracker,
        k_rule_workers=2,
    )
    custom_axes = TopologyAxes("aiv_only", "persistent_onchip_state", "irregular_simt")
    report = asyncio.run(
        runner.run_round(parent, round_idx=1, factory_name="make_kernel", topology_axes=custom_axes)
    )
    assert len(report.evaluated_candidates) == 3
    assert "ce:aiv_only" in tracker.axis_posteriors
    assert "mr:persistent_onchip_state" in tracker.axis_posteriors
    assert "ir:irregular_simt" in tracker.axis_posteriors

    mod_file = tmp_path / "dummy_op.py"
    mod_file.write_text(
        "def _select_dummy_config(**kw):\n"
        "    return {'M': 64, 'N': 64}\n\n"
        "def _build_dummy_prim_func(M: int = 64, N: int = 64):\n"
        "    return make_kernel(M, N)\n\n"
        + _SIMPLE_VECTOR_DSL,
        encoding="utf-8",
    )
    r_spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_dummy_prim_func",),
        sample_builder_kwargs={"M": 64, "N": 64},
        config_selector_fn="_select_dummy_config",
        editable_symbols=("_select_dummy_config", "_build_dummy_prim_func"),
        representative_hir_dims={"M": 64, "N": 64},
    )
    anchor = AnchorCaseSpec("level1/dummy", 1, "main", 50.0, 10.0)
    camp_tracker = BayesianRuleTracker.from_seed_yaml()
    camp_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "ragged_group")
    camp = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="dummy_op", max_rounds=1, k_rule_workers=1),
        target_module_path=mod_file,
        perf_target=PerfTargetSpec("dummy_op", 1.5, 1.2, 0.8),
        anchors=[anchor],
        regime_specs=[r_spec],
        llm_backend=MockLLMBackend(
            responses={
                "exploit_rule": ["def _select_dummy_config(**kw):\n    return {'M': 64, 'N': 64}\n"],
                "explore_creative": [
                    json.dumps(
                        {
                            "physical_hypothesis": "creative",
                            "target_local_metric": "dur_us",
                            "dsl_source": "def _select_dummy_config(**kw):\n    return {'M': 64, 'N': 64}\n",
                        }
                    ),
                    json.dumps(
                        {
                            "physical_hypothesis": "creative2",
                            "target_local_metric": "dur_us",
                            "dsl_source": "def _select_dummy_config(**kw):\n    return {'M': 64, 'N': 64}\n",
                        }
                    ),
                ],
            }
        ),
        tracker=camp_tracker,
        l0_checker=lambda _src, _reg: StageAResult(passed=True, prim_func=None, ledger=sa.ledger),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, ancs: {
            cid: aggregate_surrogate_result(
                cid,
                "dummy_op",
                [build_anchor_metrics(a, accuracy_passed=True, cand_us=20.0) for a in ancs],
            )
            for cid, _p in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        source_root=tmp_path,
        topology_axes=camp_axes,
    )
    camp_report = asyncio.run(camp.run())
    assert camp_report.total_rounds == 1
    assert "ce:coupled_1c2v" in camp_tracker.axis_posteriors
    assert "mr:bidirectional_cv_ring" in camp_tracker.axis_posteriors
    assert "ir:ragged_group" in camp_tracker.axis_posteriors


def test_audit3_live_sdk_header_and_tilelang_codegen_scanner() -> None:
    """Scan real Ascend SDK headers on disk (fail closed unless fallback allowed)."""
    with pytest.raises(FileNotFoundError):
        scan_ascend_sdk_headers(header_roots=["/nonexistent/asc_sdk_path_xyz"])
    fallback = scan_ascend_sdk_headers(
        header_roots=["/nonexistent/asc_sdk_path_xyz"],
        allow_offline_catalog=True,
    )
    assert fallback.fallback_used is True
    assert fallback.source == "catalog_fallback"
    assert "CrossCoreSetFlag" in fallback.symbols

    has_cann_sdk = any(
        ("tl_templates" not in str(p)) and Path(p).is_dir()
        for p in DEFAULT_ASCEND_SDK_HEADER_ROOTS
    )
    if has_cann_sdk:
        live_sdk = scan_ascend_sdk_headers()
        assert live_sdk.fallback_used is False
        assert live_sdk.source == "sdk_headers"
        assert live_sdk.header_files_scanned > 100
        assert "CrossCoreSetFlag" in live_sdk.symbols
    elif any(Path(p).is_dir() for p in DEFAULT_ASCEND_SDK_HEADER_ROOTS):
        live_sdk = scan_ascend_sdk_headers()
        assert live_sdk.fallback_used is False
        assert live_sdk.source == "sdk_headers"
        assert live_sdk.header_files_scanned >= 1
        assert "SetFlag" in live_sdk.symbols
        assert "WaitFlag" in live_sdk.symbols

    src, syms = discover_tilelang_lowering_surface()
    if Path(DEFAULT_TILELANG_CODEGEN_PATH).is_file():
        assert "codegen" in src
        assert "ASCEND_CODEGEN" in syms
        assert "target.build.tilelang_ascend" in syms


def test_audit4_mece_7facet_relaxation_lattice_and_verifier() -> None:
    """Minimal causal facet unlocking + per-facet verification + CFG SetFlag/WaitFlag symmetry."""
    assert VALID_FREEDOM_FACETS == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "exec_mode_and_spr",
        "sram_layout_and_aliasing",
        "dma_and_cache_hierarchy",
        "pipeline_and_sync_graph",
        "grid_and_workspace_topology",
    )
    default_mask = ConstraintRelaxationMask.bottom()
    assert default_mask.unlocked_facets() == ("alg_compute_and_monoid", "tile_and_stage_knobs")
    assert default_mask.locked_facets() == (
        "exec_mode_and_spr",
        "sram_layout_and_aliasing",
        "dma_and_cache_hierarchy",
        "pipeline_and_sync_graph",
        "grid_and_workspace_topology",
    )
    top_mask = ConstraintRelaxationMask.top()
    assert top_mask.is_unrestricted_top is True
    assert top_mask.locked_facets() == ()

    m_ovf = relax_facets_for_bottleneck(default_mask, l1_overflow=True)
    assert m_ovf.sram_layout_and_aliasing is True
    assert m_ovf.pipeline_and_sync_graph is False
    assert m_ovf.dma_and_cache_hierarchy is False
    assert m_ovf.exec_mode_and_spr is False
    assert m_ovf.grid_and_workspace_topology is False

    m_exec = relax_facets_for_bottleneck(
        default_mask,
        compiler_summary=BishengResourceSummary(spill_bytes=64),
    )
    assert m_exec.exec_mode_and_spr is True
    assert m_exec.sram_layout_and_aliasing is False
    assert m_exec.pipeline_and_sync_graph is False

    m_sync = relax_facets_for_bottleneck(
        default_mask,
        residual=StaticDynamicResidual(30.0, 0.8, 0.2, 0.2, 0.1, 0.35, 0.2),
    )
    assert m_sync.pipeline_and_sync_graph is True
    assert m_sync.sram_layout_and_aliasing is False
    assert m_sync.dma_and_cache_hierarchy is False
    assert m_sync.grid_and_workspace_topology is False

    m_dma = relax_facets_for_bottleneck(
        default_mask,
        dominant_bound="hbm_mte2_stall",
    )
    assert m_dma.dma_and_cache_hierarchy is True
    assert m_dma.grid_and_workspace_topology is False

    m_grid = relax_facets_for_bottleneck(
        default_mask,
        dominant_bound="wave_quantization_swat",
    )
    assert m_grid.grid_and_workspace_topology is True
    assert m_grid.pipeline_and_sync_graph is False

    m_all = m_ovf.join(m_exec).join(m_sync).join(m_dma).join(m_grid)
    assert m_all == top_mask
    assert m_all.is_unrestricted_top is True

    assert verify_source_against_facets("Reg::Exp(a, b);", default_mask).passed is True
    assert verify_source_against_facets("l1_slot_aliasing(p, k);", default_mask).passed is False
    assert verify_source_against_facets("l1_slot_aliasing(p, k);", m_ovf).passed is True
    assert (
        verify_source_against_facets(
            "CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);",
            m_ovf,
        ).passed
        is False
    )
    assert (
        verify_source_against_facets(
            "CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);",
            m_sync,
        ).passed
        is True
    )
    assert (
        verify_source_against_facets(
            "__simt_vf__ void k() { l1_slot_aliasing(p, k); Nd2NzParams x; STREAMK_SCHEDULE; CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0); }",
            top_mask,
        ).passed
        is True
    )
    assert (
        verify_source_against_facets(
            "void k(bool c) { if (c) { CrossCoreSetFlag<0x2, PIPE_FIX>(0); } }",
            top_mask,
        ).passed
        is False
    )
    assert (
        verify_source_against_facets("L1->GM DataCopy;", top_mask).passed is False
    )


def test_audit5_bidirectional_live_buffer_fixed_point_resolver() -> None:
    """Re-solver shrinks tiles on live buffer overflow and expands when buffers are removed."""
    solver = RegimePiecewiseSolver(AscendTarget("ascend.950pr"))
    gemm_axes = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    vec_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    regime = RegimeSlice(
        regime_tag="aligned_main",
        case_nums=(1,),
        representative_dims={"M": 128, "N": 128, "K": 64},
        has_unaligned_tail=False,
    )
    base_plan = solver.solve_regime(
        regime,
        op_name="custom_gemm",
        topology_axes=gemm_axes,
        uses_cube=True,
    )
    assert base_plan.policy["tile_mnk"] == [128, 128, 64]

    overflow_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 128},
        waterlines={
            "L1": build_scope_waterline(
                scope="L1",
                capacity_bytes=524288,
                explicit_buffers={"A": 262144, "B": 262144},
                implicit_buffers={"extra_tmp": 262144},
            ),
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=253952,
                explicit_buffers={"ub_out": 200000},
                implicit_buffers={"ub_extra": 200000},
            ),
        },
    )
    shrunk = recompute_regime_for_live_buffers(
        solver,
        regime,
        overflow_ledger,
        op_name="custom_gemm",
        topology_axes=gemm_axes,
        uses_cube=True,
    )
    assert shrunk.policy["tile_mnk"] != base_plan.policy["tile_mnk"] or shrunk.policy["l1_stages"] < base_plan.policy["l1_stages"]
    assert shrunk.policy["ub_estimated_bytes"] <= 253952
    assert shrunk.policy["l1_estimated_bytes"] <= 524288

    vec_reg = RegimeSlice(
        regime_tag="vec_main",
        case_nums=(1,),
        representative_dims={"M": 128, "N": 1024, "rows": 128, "cols": 1024},
        has_unaligned_tail=False,
    )
    vec_base = solver.solve_regime(
        vec_reg,
        op_name="custom_vec",
        topology_axes=vec_axes,
        uses_cube=False,
    )
    reduced_bufs = {"UB": {"lean_tile": max(1024, int(vec_base.policy["ub_estimated_bytes"]) // 4)}}
    expanded = recompute_regime_for_live_buffers(
        solver,
        vec_reg,
        reduced_bufs,
        op_name="custom_vec",
        topology_axes=vec_axes,
        uses_cube=False,
    )
    assert expanded.policy["row_pack_factor"] >= vec_base.policy["row_pack_factor"]
    assert expanded.policy["ub_estimated_bytes"] <= 262144


def test_audit6_ascendc_candidate_eval_and_closed_loop_escalation_hook() -> None:
    """Track-B C++ passes Stage A without SyntaxError, and PortfolioRunner escalates."""
    cpp_ok = """
    // Valid bottom() compute body (no raw CrossCore or L1 aliasing)
    constexpr int CUBE_TILE_M = 64;
    constexpr int CUBE_TILE_N = 128;
    constexpr int CUBE_TILE_K = 64;
    Reg::Exp(dst, src, mask);
    """
    m_bot = ConstraintRelaxationMask.bottom()
    res_ok = evaluate_ascendc_candidate(
        cpp_ok,
        freedom_mask=m_bot,
        verifier=SILVerifier(),
        enable_track_router=True,
    )
    assert res_ok.passed is True
    assert res_ok.backend_kind == "ascendc"
    assert res_ok.ledger is not None and res_ok.ledger.is_legal is True

    funnel = GateFunnel(skip_stage_b=True)
    ev = funnel.evaluate(cpp_ok, backend_kind="ascendc", freedom_mask=m_bot)
    assert ev.passed is True
    assert ev.stage_a.backend_kind == "ascendc"

    cpp_bad_sync = "CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);"
    res_bad = evaluate_ascendc_candidate(cpp_bad_sync, freedom_mask=m_bot)
    assert res_bad.passed is False
    assert "pipeline_and_sync_graph=locked" in res_bad.error_message

    runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
    )
    cand = BeamCandidate(
        candidate_id="stagnant_tl",
        round_idx=2,
        worker_kind="rule_exploit",
        rule_id="R01_DUAL_COPY_1C2V_SPLIT",
        dsl_source=_SIMPLE_VECTOR_DSL,
        track="developer",
        ledger=res_ok.ledger,
        residual=StaticDynamicResidual(
            measured_latency_us=80.0,
            predicted_bottleneck_util=0.85,
            measured_aic_mac_ratio=0.0,
            measured_aiv_vec_ratio=0.30,
            measured_mte2_wait_ratio=0.20,
            measured_sync_stall_ratio=0.35,
            unexplained_gap_ratio=0.40,
            vf_fusion_breaks=3,
        ),
        latency_us=80.0,
        accuracy_passed=True,
    )
    esc_rep = runner.check_round_escalation(
        cand,
        current_tier="tilelang_dsl",
        target_eta=0.85,
        stagnation_rounds=2,
        ideal_roofline_us=25.0,
        dominant_bound="ub_load_store",
        cost_model=HybridAscendCostModel(AscendTarget("ascend.950pr")),
    )
    assert isinstance(esc_rep, RoundEscalationReport)
    assert esc_rep.escalation.should_escalate is True
    assert esc_rep.escalation.to_tier == "inline_ascendc_vf"
    assert esc_rep.relaxed_mask.pipeline_and_sync_graph is True
    assert esc_rep.relaxed_mask.sram_layout_and_aliasing is False
    assert esc_rep.cost_prediction.pred_us > 0.0
    assert esc_rep.roofline_eta == pytest.approx(25.0 / 80.0)


def test_audit7_zero_legacy_interfaces_lint_gate() -> None:
    """Assert ZERO superseded legacy interfaces under targets/ascend/tilefoundry_ascend/."""
    root = Path(__file__).resolve().parents[3] / "targets" / "ascend" / "tilefoundry_ascend"
    assert root.is_dir(), f"Missing target root: {root}"

    forbidden_tokens = (
        "VALID_PARADIGMS",
        "b1_functor_slot",
        "b2_composable_atoms",
        "b3_guarded_full_rewrite",
        "VALID_TRACK_B_FREEDOM_LEVELS",
        "VALID_ISA_DOMAINS",
        "paradigm_posteriors",
    )
    hits: list[str] = []
    for py_file in sorted(root.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        ast.parse(source, filename=str(py_file))
        for tok in forbidden_tokens:
            if tok in source:
                hits.append(f"{py_file.relative_to(root)} contains {tok}")
    assert hits == [], f"Superseded legacy interfaces still present: {hits}"
