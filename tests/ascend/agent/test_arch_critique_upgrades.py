"""Offline verification suite for the 5 architectural critique refinements."""

from __future__ import annotations

import inspect

import pytest

from tilefoundry_ascend.agent.campaign.campaign_runner import CampaignConfig
from tilefoundry_ascend.agent.campaign.marathon import MarathonConfig
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import PortfolioRunner
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    TileConfigCandidate,
    evaluate_tile_candidate,
)
from tilefoundry_ascend.agent.router import (
    VALID_HARDWARE_PIPES,
    CapabilityGate,
    EscalationDecision,
    PipeGapBreakdown,
    PipeGapReport,
    evaluate_track_escalation,
    route_execution_track,
    route_from_op_spec,
    route_initial_by_input_contract,
    scan_tilelang_pipe_gaps,
)
from tilefoundry_ascend.ascend.cost_model import (
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
    BishengResourceSummary,
    CompilerBehaviorDetector,
    CompilerBehaviorRegistry,
    HybridAscendCostModel,
    parse_bisheng_res_usage,
    register_compiler_detector,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil import (
    VALID_COMPUTE_ENGINES,
    VALID_FREEDOM_FACETS,
    VALID_ITERATION_REGULARITIES,
    VALID_MEMORY_RINGS,
    AdvisoryEnvelope,
    ConstraintRelaxationMask,
    RegimeSlice,
    SchedulePolicy,
    SILVerifier,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.templates import (
    CANONICAL_TOPOLOGY_PRESETS,
    SHELL_CV_RING_PIPELINE,
    ComposableComputeSlotSpec,
    get_template_registry,
    relax_facets_for_bottleneck,
)


def test_tier1_compiler_behavior_registry_detects_all_7_phenomena_and_custom_plugin() -> None:
    """Verify all 9 built-in compiler phenomena + extensible register_compiler_detector plugin."""
    detectors = DEFAULT_COMPILER_BEHAVIOR_REGISTRY.list_detectors()
    assert len(detectors) == 9
    assert "scalar_compute_loop_fallback" in detectors
    assert "dcache_bypass_narrow_store" in detectors

    clean_res = "Vector Register Usage: 32\nScalar Register Usage: 16\nStack Frame Size: 128 bytes\n"
    clean_asm = "VLOOPv2 r0, 32\nVADD v0, v1, v2\n"
    clean_sum = parse_bisheng_res_usage(clean_res, clean_asm)
    assert clean_sum.vloopv2_hit is True
    assert clean_sum.vf_fusion_breaks == 0
    assert clean_sum.alias_serialized_buffers == 0
    assert clean_sum.dual_issue_ipc_collapse == 0.0
    assert clean_sum.dcache_dcci_flushes == 0
    assert clean_sum.extra_signals == {}

    hazard_res = (
        "Vector Register Usage: 64\n"
        "Scalar Register Usage: 48\n"
        "Register Spill: 256 bytes\n"
        "Stack Frame Size: 2048 bytes\n"
    )
    hazard_dump = """
    asc_vf_call(vf_part0);
    LocalMemBar();
    asc_vf_call(vf_part1);
    asc_vf_call(vf_part2);
    // enable_offset=true, may_alias on ping-pong UB stage
    .LBB0_1:
        vmula acc0, v1, v2
        vmula acc0, v3, v4
        vmula acc0, v5, v6
        vexp v7, acc0
        vsqrt v8, v7
        vdiv v9, v8, v1
        dcci
        dcci
        cbnz x0, .LBB0_1
    """
    bad_sum = parse_bisheng_res_usage(hazard_res, hazard_dump)
    assert bad_sum.spill_bytes == 256
    assert bad_sum.stack_bytes == 2048
    assert bad_sum.software_loops >= 1
    assert bad_sum.has_local_membar_in_vf_call is True
    assert bad_sum.vf_fusion_breaks >= 2
    assert bad_sum.alias_serialized_buffers >= 2
    assert bad_sum.dual_issue_ipc_collapse > 0.0
    assert bad_sum.dcache_dcci_flushes >= 2

    rt = BishengResourceSummary.from_dict(bad_sum.to_dict())
    assert rt == bad_sum

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    base_wl = {
        "vec_flops": 5e9,
        "gm_bytes": 16 * 1024 * 1024,
        "contig_burst_bytes": 32768,
    }
    p_clean = model.predict(base_wl, bisheng_res=clean_sum)
    p_vf = model.predict(base_wl, bisheng_res=BishengResourceSummary(vf_fusion_breaks=3))
    p_alias = model.predict(base_wl, bisheng_res=BishengResourceSummary(alias_serialized_buffers=2))
    p_ipc = model.predict(base_wl, bisheng_res=BishengResourceSummary(dual_issue_ipc_collapse=0.8))
    p_dcci = model.predict(base_wl, bisheng_res=BishengResourceSummary(dcache_dcci_flushes=4))

    assert p_vf.tier1_penalty_us > p_clean.tier1_penalty_us
    assert p_alias.tier1_penalty_us > p_clean.tier1_penalty_us
    assert p_ipc.tier1_penalty_us > p_clean.tier1_penalty_us
    assert p_dcci.tier1_penalty_us > p_clean.tier1_penalty_us

    custom_reg: CompilerBehaviorRegistry = DEFAULT_COMPILER_BEHAVIOR_REGISTRY.clone()
    det = register_compiler_detector(
        "mte2_queue_stall",
        lambda _res, asm: asm.count("MTE2_STALL_CYCLE"),
        lambda summary, ctx: float(summary.extra_signals.get("mte2_queue_stall", 0.0)) * 2.5,
        registry=custom_reg,
        description="Custom MTE2 queue stall detector",
    )
    assert isinstance(det, CompilerBehaviorDetector)
    custom_sum = parse_bisheng_res_usage(
        clean_res,
        "VLOOPv2 r0, 32\nMTE2_STALL_CYCLE\nMTE2_STALL_CYCLE\n",
        registry=custom_reg,
    )
    assert custom_sum.extra_signals["mte2_queue_stall"] == pytest.approx(2.0)

    custom_model = HybridAscendCostModel(AscendTarget("ascend.950pr"), behavior_registry=custom_reg)
    p_custom = custom_model.predict(base_wl, bisheng_res=custom_sum)
    assert p_custom.tier1_penalty_us >= p_clean.tier1_penalty_us + 5.0


def test_topology_axes_roundtrip_and_structural_regime_dispatch() -> None:
    """Verify 3-axis TopologyAxes schema and op_name-free solver dispatch."""
    for ce in VALID_COMPUTE_ENGINES:
        assert ce in ("aiv_only", "aic_only", "coupled_1c2v")
    for mr in VALID_MEMORY_RINGS:
        assert mr in (
            "gm_ub_stream",
            "ub_random_atomic",
            "gm_l1_l0_fixpipe",
            "l0c_ub_epilogue",
            "bidirectional_cv_ring",
            "persistent_onchip_state",
        )
    for ir in VALID_ITERATION_REGULARITIES:
        assert ir in (
            "affine_simd",
            "irregular_simt",
            "ragged_group",
            "dynamic_opcode_stream",
        )

    for axes in CANONICAL_TOPOLOGY_PRESETS:
        assert TopologyAxes.from_dict(axes.to_dict()) == axes

    reg = RegimeSlice(
        regime_tag="hybrid_main",
        case_nums=(1,),
        representative_dims={"M": 192, "N": 128, "K": 128, "D_k": 128, "D_v": 128},
        has_unaligned_tail=True,
        has_empty_group=True,
    )
    default_axes = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
    adv_plain = AdvisoryEnvelope(
        op_name="unseen_op_alpha",
        cannbot_task="custom/unseen_op_alpha",
        level=4,
        topology_axes=default_axes,
        input_dtypes=("f16",),
        output_dtypes=("f16",),
        roofline={"dominant_bound": "cube"},
        structural_features={},
        regimes=(reg,),
    )
    assert adv_plain.to_dict()["topology_axes"] == default_axes.to_dict()
    assert AdvisoryEnvelope.from_dict(adv_plain.to_dict()) == adv_plain

    hybrid_axes = TopologyAxes(
        compute_engine="coupled_1c2v",
        memory_ring="persistent_onchip_state",
        iteration_regularity="ragged_group",
    )
    adv_axes = AdvisoryEnvelope(
        op_name="unseen_op_alpha",
        cannbot_task="custom/unseen_op_alpha",
        level=4,
        topology_axes=hybrid_axes,
        input_dtypes=("f16",),
        output_dtypes=("f16",),
        roofline={"dominant_bound": "cube"},
        structural_features={},
        regimes=(reg,),
    )
    assert adv_axes.to_dict()["topology_axes"] == hybrid_axes.to_dict()
    assert AdvisoryEnvelope.from_dict(adv_axes.to_dict()) == adv_axes

    solver = RegimePiecewiseSolver(AscendTarget("ascend.950pr"))
    plan_unknown_name = solver.solve_regime(
        reg,
        op_name="completely_unseen_operator_xyz",
        topology_axes=hybrid_axes,
        roofline={"dominant_bound": "cube"},
        uses_cube=True,
        has_persistent_state=True,
        has_dead_l1_slice=True,
        quant_mode="none",
    )
    plan_misleading_name = solver.solve_regime(
        reg,
        op_name="elementwise_add_rms_norm",
        topology_axes=hybrid_axes,
        roofline={"dominant_bound": "cube"},
        uses_cube=True,
        has_persistent_state=True,
        has_dead_l1_slice=True,
        quant_mode="none",
    )
    assert plan_unknown_name.policy == plan_misleading_name.policy
    assert plan_unknown_name.persistent_state_space in ("L1", "UB")
    assert plan_unknown_name.persistent_state_bytes == 128 * 128 * 4
    assert plan_unknown_name.policy["tile_mnk"][0] >= 82
    assert plan_unknown_name.policy["l1_half_bank_split"] is True
    assert plan_unknown_name.policy["ensure_empty_subblock_sync"] is True
    assert plan_unknown_name.policy["l1_slot_aliasing"]
    assert plan_unknown_name.policy["topology_axes"] == hybrid_axes.to_dict()

    sil_pol = SchedulePolicy.from_dict(plan_unknown_name.to_dict())
    assert sil_pol.topology_axes == hybrid_axes
    report = SILVerifier().verify_policy(sil_pol, adv_axes, regime=reg)
    assert report.passed is True, report.errors


def test_causal_t0_router_t_gt_0_escalation_and_isa_gap_scanner() -> None:
    """Verify causal t=0 contract routing, t>0 escalation, and 6-domain ISA gap scanner."""
    for op_name in ("gqa", "grouped_matmul", "gather", "kimi_delta_attention", "brand_new_op"):
        dec = route_initial_by_input_contract(
            {
                "op_name": op_name,
                "input_dtypes": ["float16", "bfloat16"],
                "output_dtypes": ["float32"],
            }
        )
        assert dec.recommended_backend_tier == "tilelang_dsl"

    fp8_dec = route_initial_by_input_contract(
        {"op_name": "unseen_fp8_gemm", "input_dtypes": ["fp8e4m3", "i8"], "output_dtypes": ["f16"]}
    )
    assert fp8_dec.recommended_backend_tier == "tilelang_dsl"

    for sb_dtype in ("f4e2m1", "f4e1m2", "i4", "u4", "q4_0", "mxfp4"):
        d_sub = route_initial_by_input_contract(
            {"op_name": "unseen_quant", "input_dtypes": ["f16", sb_dtype], "output_dtypes": ["f16"]}
        )
        assert d_sub.recommended_backend_tier == "ascendc_template", sb_dtype

    d_cas = route_initial_by_input_contract(
        {"op_name": "hash_probe", "input_dtypes": ["uint64"], "requires_inplace_gm_cas": True}
    )
    assert d_cas.recommended_backend_tier == "ascendc_template"

    d_vm = route_initial_by_input_contract(
        {
            "op_name": "unseen_interpreter",
            "input_dtypes": ["float32"],
            "topology_axes": {
                "compute_engine": "aiv_only",
                "memory_ring": "persistent_onchip_state",
                "iteration_regularity": "dynamic_opcode_stream",
            },
        }
    )
    assert d_vm.recommended_backend_tier == "ascendc_template"

    assert (
        route_from_op_spec(
            {
                "category": "IndexGather",
                "inputs": [{"dtype": ["float16"]}, {"dtype": ["int32"]}],
            }
        ).recommended_backend_tier
        == "ascendc_template"
    )
    assert (
        route_execution_track(
            TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
        ).track
        == "developer"
    )
    assert CapabilityGate.suitability_score(
        TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    ) == 1.0
    assert (
        CapabilityGate.route(
            TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt"),
            router_enabled=False,
        ).track
        == "developer"
    )

    no_esc = evaluate_track_escalation(
        current_tier="tilelang_dsl",
        roofline_eta=0.90,
        target_eta=0.85,
        stagnation_rounds=3,
        dominant_bound="vector",
        compiler_summary=BishengResourceSummary(),
        structural_features={},
    )
    assert isinstance(no_esc, EscalationDecision)
    assert no_esc.should_escalate is False

    early_no_esc = evaluate_track_escalation(
        current_tier="tilelang_dsl",
        roofline_eta=0.50,
        target_eta=0.85,
        stagnation_rounds=1,
        dominant_bound="ub_load_store",
        compiler_summary=BishengResourceSummary(vf_fusion_breaks=2),
        structural_features={},
    )
    assert early_no_esc.should_escalate is False

    esc_vf = evaluate_track_escalation(
        current_tier="tilelang_dsl",
        roofline_eta=0.45,
        target_eta=0.85,
        stagnation_rounds=2,
        dominant_bound="ub_load_store",
        compiler_summary=BishengResourceSummary(vf_fusion_breaks=3),
        structural_features={"requires_dist_unpack": True},
    )
    assert esc_vf.should_escalate is True
    assert esc_vf.to_tier == "inline_ascendc_vf"

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
    assert esc_tpl.should_escalate is True
    assert esc_tpl.to_tier == "ascendc_template"

    live_report = scan_tilelang_pipe_gaps()
    assert isinstance(live_report, PipeGapReport)
    assert tuple(live_report.pipes.keys()) == VALID_HARDWARE_PIPES
    assert all(isinstance(v, PipeGapBreakdown) for v in live_report.pipes.values())
    assert 0.0 < live_report.coverage_ratio < 1.0
    assert live_report.has_gap("simt_gm_atomic_cas") is True
    assert live_report.has_gap("hardware_bitonic_sort32") is True

    mock_gap = scan_tilelang_pipe_gaps(
        tilelang_surface={"simd.vld", "simd.vst2", "shfl_sync"},
        pipes=("pipe_v_simd_simt",),
    )
    assert "simd_vld_vst" in mock_gap.supported_primitives
    assert "simt_gm_atomic_cas" in mock_gap.missing_primitives


def test_mece_7_facet_freedom_lattice_and_composable_compute_slot() -> None:
    """Verify 7-facet ConstraintRelaxationMask lattice and ComposableComputeSlotSpec."""
    assert VALID_FREEDOM_FACETS == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "exec_mode_and_spr",
        "sram_layout_and_aliasing",
        "dma_and_cache_hierarchy",
        "pipeline_and_sync_graph",
        "grid_and_workspace_topology",
    )

    m_bot = ConstraintRelaxationMask.bottom()
    m_top = ConstraintRelaxationMask.top()
    assert m_bot.unlocked_facets() == ("alg_compute_and_monoid", "tile_and_stage_knobs")
    assert m_top.is_unrestricted_top is True
    assert m_top.locked_facets() == ()

    m_sram = relax_facets_for_bottleneck(m_bot, l1_overflow=True)
    assert m_sram.sram_layout_and_aliasing is True
    assert m_sram.pipeline_and_sync_graph is False

    m_pipe = relax_facets_for_bottleneck(m_sram, dominant_bound="crosscore_stall")
    assert m_pipe.pipeline_and_sync_graph is True
    assert m_pipe.sram_layout_and_aliasing is True

    simd_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    pol_m = SchedulePolicy(
        regime_tag="main",
        topology_axes=simd_axes,
        backend_tier="ascendc_template",
        freedom_mask=m_pipe,
    )
    assert pol_m.to_dict()["freedom_mask"] == m_pipe.to_dict()
    assert SchedulePolicy.from_dict(pol_m.to_dict()) == pol_m

    with pytest.raises(ValueError, match="sram_layout_and_aliasing"):
        ComposableComputeSlotSpec(
            op_name="bad_sram",
            freedom_mask=m_bot,
            l1_alias_map={"P": "K_rope"},
        )
    with pytest.raises(ValueError, match="pipeline_and_sync_graph"):
        ComposableComputeSlotSpec(
            op_name="bad_sync",
            freedom_mask=m_bot,
            custom_ring_sync="CrossCoreSetFlag<mode=4>",
        )

    slot_unlocked = ComposableComputeSlotSpec(
        op_name="unseen_hybrid_attn",
        freedom_mask=m_bot.unlock(
            "exec_mode_and_spr",
            "sram_layout_and_aliasing",
            "dma_and_cache_hierarchy",
            "pipeline_and_sync_graph",
        ),
        coordination_shell=SHELL_CV_RING_PIPELINE,
        slot_name="online_softmax_epilogue",
        atom_sequence=("Reg::Load", "Reg::Exp", "Reg::ReduceDataBlock", "Reg::Store"),
        l1_alias_map={"P_stage": "K_rope_dead_slice"},
        custom_ring_sync="CrossCoreSetFlag<mode=4, reverseAck=14>",
        compute_body_cpp="Reg::Exp(scores, scores, maskFull);",
        invariants=("Half-L1 <= 262144B", "symmetric empty subblock sync"),
    )
    assert ComposableComputeSlotSpec.from_dict(slot_unlocked.to_dict()) == slot_unlocked

    rendered = get_template_registry().render_with_compute_slot(
        slot_unlocked,
        {"tile_mnk": (96, 112, 64), "l1_stages": 2},
    )
    assert "Algorithmic Synthesis Contract" in rendered.ascendc_cpp
    assert "P_stage->K_rope_dead_slice" in rendered.ascendc_cpp
    assert "Reg::Exp(scores, scores, maskFull);" in rendered.ascendc_cpp
    assert "CUBE_TILE_M = 96U" in rendered.ascendc_cpp
    assert rendered.policy["freedom_mask"] == slot_unlocked.freedom_mask.to_dict()

    slot_top = ComposableComputeSlotSpec(
        op_name="free_ast_kernel",
        freedom_mask=ConstraintRelaxationMask.top(),
        coordination_shell=SHELL_CV_RING_PIPELINE,
        l1_alias_map={"P_stage": "K_rope_dead_slice"},
        custom_ring_sync="CrossCoreSetFlag<mode=4, reverseAck=14>",
        compute_body_cpp="Mmad(l0C, l1A, l1B, mmad); PipeBarrier<PIPE_ALL>();",
    )
    rendered_top = get_template_registry().render_with_compute_slot(slot_top)
    assert "unrestricted_top=True" in rendered_top.ascendc_cpp
    assert "CUBE_TILE_M" not in rendered_top.ascendc_cpp


def test_tilesize_prescreen_default_off_per_dtype_half_l1_and_ablation_gate() -> None:
    """Verify default-on enforcement, per-dtype Cube MACs/Half-L1 scoring, and ablation gate."""
    assert CampaignConfig(operator_name="rms_norm").enable_tilesize_prescreen is True
    assert (
        inspect.signature(MarathonConfig).parameters["enable_tilesize_prescreen"].default
        is True
    )
    assert (
        inspect.signature(PortfolioRunner.__init__)
        .parameters["enable_tilesize_prescreen"]
        .default
        is True
    )

    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    fp16_score = evaluate_tile_candidate(
        TileConfigCandidate(bm=128, bn=128, bk=64, stages=2, dtype_bytes=2, dtype="f16"),
        facts,
    )
    fp8_score = evaluate_tile_candidate(
        TileConfigCandidate(bm=128, bn=128, bk=64, stages=2, dtype_bytes=1, dtype="fp8e4m3"),
        facts,
    )
    fp32_score = evaluate_tile_candidate(
        TileConfigCandidate(bm=128, bn=128, bk=64, stages=2, dtype_bytes=4, dtype="f32"),
        facts,
    )
    assert fp8_score.cube_macs_per_cycle == 8192
    assert fp16_score.cube_macs_per_cycle == 4096
    assert fp32_score.cube_macs_per_cycle == 256
    assert fp8_score.fixpipe_to_cube_ratio > fp16_score.fixpipe_to_cube_ratio > fp32_score.fixpipe_to_cube_ratio

    no_split_score = evaluate_tile_candidate(
        TileConfigCandidate(
            bm=128, bn=128, bk=64, stages=2, dtype_bytes=2, dtype="f16", l1_half_bank_split=False
        ),
        facts,
    )
    assert fp16_score.half_l1_penalty == 0.0
    assert no_split_score.half_l1_penalty == 0.0
    assert no_split_score.score <= fp16_score.score
