"""Offline verification suite for the second-round architectural critique alignments."""

from __future__ import annotations

from tilefoundry_ascend.agent.kb import AscendCausalRule, BayesianRuleTracker
from tilefoundry_ascend.agent.router import (
    ASCEND_DEVKIT_ISA_CATALOG,
    VALID_HARDWARE_PIPES,
    PipeGapReport,
    scan_tilelang_pipe_gaps,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    RegimePiecewiseSolver,
    compute_critical_cube_tile_m,
)
from tilefoundry_ascend.ascend.sil import (
    DEFAULT_OPS_ROOT,
    AdvisoryEnvelope,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    OpAdvisoryExtractor,
    RegimeSlice,
    SchedulePolicy,
    SILVerifier,
    TopologyAxes,
    synthesize_default_policy,
    verify_policy_envelope,
    verify_source_against_facets,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.templates import (
    ComposableComputeSlotSpec,
)


def test_compute_critical_cube_tile_m_and_fp32_regime_sizing() -> None:
    """Verify the exact Roofline waterline formula across dtypes.

    Covers FP16/BF16/FP8/INT8/FP4/FP32 and solver integration.
    """
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)

    for dt in ("f16", "float16", "bf16", "bfloat16"):
        assert compute_critical_cube_tile_m(facts, dt) == 82
    for dt in ("fp8e4m3", "float8_e4m3fn", "fp8e5m2", "i8", "int8", "u8"):
        assert compute_critical_cube_tile_m(facts, dt) == 82
    for dt in ("f4e2m1", "f4e1m2", "float4_e2m1"):
        assert compute_critical_cube_tile_m(facts, dt) == 82
    for dt in ("f32", "float32"):
        assert compute_critical_cube_tile_m(facts, dt) == 11

    assert compute_critical_cube_tile_m(facts, "f16", dual_operand_miss=True) == 164
    assert compute_critical_cube_tile_m(facts, "f32", dual_operand_miss=True) == 21

    solver = RegimePiecewiseSolver(AscendTarget("ascend.950pr"))

    gemm_axes = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    cv_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    reg_small = RegimeSlice(
        regime_tag="fp32_small_m",
        case_nums=(1,),
        representative_dims={"M": 32, "N": 128, "K": 64},
        has_unaligned_tail=False,
    )
    fp32_plan = solver.solve_regime(
        reg_small,
        op_name="custom_fp32_gemm",
        topology_axes=gemm_axes,
        roofline={"dominant_bound": "cube"},
        input_dtypes=("f32",),
        uses_cube=True,
    )
    assert fp32_plan.policy["tile_mnk"][0] == 32

    reg_16 = RegimeSlice(
        regime_tag="fp32_m16",
        case_nums=(2,),
        representative_dims={"M": 16, "N": 128, "K": 64},
        has_unaligned_tail=False,
    )
    fp32_m16 = solver.solve_regime(
        reg_16,
        op_name="custom_fp32_cv",
        topology_axes=cv_axes,
        roofline={"dominant_bound": "cube"},
        input_dtypes=("f32",),
        uses_cube=True,
    )
    assert fp32_m16.policy["tile_mnk"][0] == 16

    for dt in ("f16", "bf16", "fp8e4m3", "i8"):
        reg_m64 = RegimeSlice(
            regime_tag=f"{dt}_m64",
            case_nums=(3,),
            representative_dims={"M": 64, "N": 128, "K": 64},
            has_unaligned_tail=False,
        )
        plan = solver.solve_regime(
            reg_m64,
            op_name=f"custom_{dt}_gemm",
            topology_axes=gemm_axes,
            roofline={"dominant_bound": "cube"},
            input_dtypes=(dt,),
            uses_cube=True,
        )
        assert plan.policy["tile_mnk"][0] == 96


def test_end_to_end_topology_axes_extractor_verifier_and_bayesian_transfer() -> None:
    """Verify topology axes across extractor, policy, verifier, and tracker.

    Covers SILVerifier cross-axis rules and zero-shot Bayesian transfer.
    """
    if DEFAULT_OPS_ROOT.exists():
        extractor = OpAdvisoryExtractor(DEFAULT_OPS_ROOT)
        for op_name in ("gqa", "grouped_matmul", "kimi_delta_attention", "mhc_sinkhorn", "sparse_flash_attention"):
            adv = extractor.extract(op_name)
            assert isinstance(adv.topology_axes, TopologyAxes)
            pol_env = synthesize_default_policy(adv)
            for pol in pol_env.policies.values():
                assert pol.topology_axes == adv.topology_axes
            report = verify_policy_envelope(pol_env, adv)
            assert report.passed is True, f"{op_name} failed: {report.errors}"

        sfa_adv = extractor.extract("sparse_flash_attention")
        assert sfa_adv.topology_axes == TopologyAxes(
            compute_engine="coupled_1c2v",
            memory_ring="bidirectional_cv_ring",
            iteration_regularity="irregular_simt",
        )
        sink_adv = extractor.extract("mhc_sinkhorn")
        assert sink_adv.topology_axes == TopologyAxes(
            compute_engine="aiv_only",
            memory_ring="persistent_onchip_state",
            iteration_regularity="affine_simd",
        )

    verifier = SILVerifier()
    reg = RegimeSlice(
        regime_tag="main",
        case_nums=(1,),
        representative_dims={"M": 128, "N": 128, "K": 64},
        has_unaligned_tail=True,
    )

    coupled_axes = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
    adv_coupled = AdvisoryEnvelope(
        op_name="coupled_test",
        cannbot_task="level4/coupled_test",
        level=4,
        topology_axes=coupled_axes,
        input_dtypes=("f16",),
        output_dtypes=("f16",),
        roofline={},
        structural_features={"uses_cube": True},
        regimes=(reg,),
    )
    rep_depth = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="main",
            topology_axes=coupled_axes,
            sync_mode="custom_ring_flag",
            reverse_ack_period=16,
        ),
        adv_coupled,
        regime=reg,
    )
    assert rep_depth.passed is False
    assert any("max_crosscore_depth=15" in e for e in rep_depth.errors)

    ragged_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
    rep_ragged = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="main",
            topology_axes=ragged_axes,
            cv_datapath="none",
            ensure_empty_subblock_sync=False,
        ),
        adv_coupled,
        regime=reg,
    )
    assert rep_ragged.passed is False
    assert any("ragged_group" in e for e in rep_ragged.errors)

    coupled_simt_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "irregular_simt")
    adv_cs = AdvisoryEnvelope(
        op_name="coupled_simt_test",
        cannbot_task="level4/coupled_simt_test",
        level=4,
        topology_axes=coupled_simt_axes,
        input_dtypes=("f16",),
        output_dtypes=("f16",),
        roofline={},
        structural_features={"uses_cube": True, "uses_simt": True},
        regimes=(reg,),
    )
    rep_ub_over = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="main",
            topology_axes=coupled_simt_axes,
            vector_mode="simdvf",
            ub_256kb_unlock=False,
            ub_estimated_bytes=240000,
        ),
        adv_cs,
        regime=reg,
    )
    assert rep_ub_over.passed is False
    assert any("32KB SIMT D-cache carveout" in e for e in rep_ub_over.errors)

    rep_ub_unlock = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="main",
            topology_axes=coupled_simt_axes,
            vector_mode="simdvf",
            ub_256kb_unlock=True,
            ub_estimated_bytes=200000,
        ),
        adv_cs,
        regime=reg,
    )
    assert rep_ub_unlock.passed is False
    assert rep_ub_unlock.suggested_fixes.get("ub_256kb_unlock") is False

    rep_ub_ok = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="main",
            topology_axes=coupled_simt_axes,
            vector_mode="simdvf",
            ub_256kb_unlock=False,
            ub_estimated_bytes=HardwareLegalityBounds().ub_simtvf_bytes,
        ),
        adv_cs,
        regime=reg,
    )
    assert rep_ub_ok.passed is True, rep_ub_ok.errors

    r_ragged = AscendCausalRule(
        rule_id="R_TEST_RAGGED",
        regime_guard="True",
        precondition="True",
        transform_template="ensure_empty_subblock_sync=True",
        expected_postcondition="True",
        source_provenance="test",
        priority=10,
        alpha_prior=1.0,
        beta_prior=1.0,
    )
    r_persist = AscendCausalRule(
        rule_id="R_TEST_PERSIST",
        regime_guard="True",
        precondition="True",
        transform_template="persistent_state=True",
        expected_postcondition="True",
        source_provenance="test",
        priority=10,
        alpha_prior=1.0,
        beta_prior=1.0,
    )
    r_neutral = AscendCausalRule(
        rule_id="R_TEST_NEUTRAL",
        regime_guard="True",
        precondition="True",
        transform_template="noop=True",
        expected_postcondition="True",
        source_provenance="test",
        priority=10,
        alpha_prior=1.0,
        beta_prior=1.0,
    )
    r_bad = AscendCausalRule(
        rule_id="R_TEST_BAD",
        regime_guard="True",
        precondition="True",
        transform_template="bad=True",
        expected_postcondition="True",
        source_provenance="test",
        priority=10,
        alpha_prior=1.0,
        beta_prior=1.0,
    )

    tracker = BayesianRuleTracker.from_rules([r_ragged, r_persist, r_neutral, r_bad])
    grouped_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
    persist_axes = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")

    for _ in range(4):
        tracker.record_outcome(
            "R_TEST_RAGGED",
            postcondition_realized=True,
            speed_gain=0.25,
            topology_axes=grouped_axes,
        )
        tracker.record_outcome(
            "R_TEST_PERSIST",
            postcondition_realized=True,
            speed_gain=0.30,
            topology_axes=persist_axes,
        )
        tracker.record_outcome(
            "R_TEST_BAD",
            postcondition_realized=False,
            speed_gain=0.0,
            topology_axes=grouped_axes,
        )

    assert "ce:coupled_1c2v" in tracker.axis_posteriors
    assert "ir:ragged_group" in tracker.axis_posteriors
    assert "mr:persistent_onchip_state" in tracker.axis_posteriors

    hybrid_axes = TopologyAxes(
        compute_engine="coupled_1c2v",
        memory_ring="persistent_onchip_state",
        iteration_regularity="ragged_group",
    )
    ranked_hybrid = tracker.rank_applicable_rules(
        {},
        topology_axes=hybrid_axes,
    )
    ranked_ids = [r.rule_id for r in ranked_hybrid]
    assert set(ranked_ids[:2]) == {"R_TEST_RAGGED", "R_TEST_PERSIST"}
    assert ranked_ids.index("R_TEST_NEUTRAL") > 1
    assert ranked_ids[-1] == "R_TEST_BAD"

    restored = BayesianRuleTracker.from_rules([r_ragged, r_persist, r_neutral, r_bad])
    restored.load_posteriors_dict(tracker.to_dict())
    assert [r.rule_id for r in restored.rank_applicable_rules({}, topology_axes=hybrid_axes)] == ranked_ids

    ranked_no_axes = tracker.rank_applicable_rules({})
    assert len(ranked_no_axes) == 4


def test_orthogonal_7_hardware_pipes_taxonomy_and_gap_scanner() -> None:
    """Verify pure 7-pipe physical hardware execution taxonomy and gap scanner."""
    assert VALID_HARDWARE_PIPES == (
        "pipe_s_scalar_spr",
        "pipe_mte2_mte3_dma",
        "pipe_mte1_l1_to_l0",
        "pipe_m_cube",
        "pipe_fix_fixpipe",
        "pipe_v_simd_simt",
        "pipe_sync_barrier_mutex",
    )

    for spec in ASCEND_DEVKIT_ISA_CATALOG:
        assert spec.hardware_pipe in VALID_HARDWARE_PIPES

    pipe_report = scan_tilelang_pipe_gaps()
    assert isinstance(pipe_report, PipeGapReport)
    assert tuple(pipe_report.pipes.keys()) == VALID_HARDWARE_PIPES
    for pipe in VALID_HARDWARE_PIPES:
        assert pipe_report.pipes[pipe].total_capabilities >= 1

    assert "mte1_load_2d_3d" in (
        pipe_report.pipes["pipe_mte1_l1_to_l0"].supported_primitives
        + pipe_report.pipes["pipe_mte1_l1_to_l0"].missing_primitives
    )
    assert "cube_mmad_sparse" in (
        pipe_report.pipes["pipe_m_cube"].supported_primitives
        + pipe_report.pipes["pipe_m_cube"].missing_primitives
    )
    fix_all = (
        pipe_report.pipes["pipe_fix_fixpipe"].supported_primitives
        + pipe_report.pipes["pipe_fix_fixpipe"].missing_primitives
    )
    assert "fixpipe_deq_scale" in fix_all
    assert "fixpipe_l0c_to_ub_split_m" in fix_all
    scalar_all = (
        pipe_report.pipes["pipe_s_scalar_spr"].supported_primitives
        + pipe_report.pipes["pipe_s_scalar_spr"].missing_primitives
    )
    assert "scalar_dcci_cache_invalidate" in scalar_all
    assert "scalar_set_ctrl_spr" in scalar_all

    mock_pipe = scan_tilelang_pipe_gaps(
        tilelang_surface={"load_2d", "dcci", "fixpipe_deq_scale"},
        pipes=("pipe_mte1_l1_to_l0", "pipe_s_scalar_spr", "pipe_fix_fixpipe"),
    )
    assert "mte1_load_2d_3d" in mock_pipe.supported_primitives
    assert "scalar_dcci_cache_invalidate" in mock_pipe.supported_primitives
    assert "fixpipe_deq_scale" in mock_pipe.supported_primitives
    assert "fixpipe_l0c_to_ub_split_m" in mock_pipe.missing_primitives
    assert mock_pipe.has_gap("pipe_fix_fixpipe") is True
    assert mock_pipe.has_gap("pipe_mte1_l1_to_l0") is False


def test_mece_7_facet_source_verification_and_cfg_flag_symmetry_gate() -> None:
    """Verify MECE 7-facet lattice source safety and CFG flag symmetry.

    The lattice is checked from bottom to top.
    """
    m_bot = ConstraintRelaxationMask.bottom()
    m_dma = m_bot.unlock("dma_and_cache_hierarchy")
    m_sync = m_bot.unlock("pipeline_and_sync_graph")
    m_top = ConstraintRelaxationMask.top()

    b1 = ComposableComputeSlotSpec(
        op_name="hybrid_cv_attn",
        freedom_mask=m_bot,
        coordination_shell=TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
        slot_name="online_softmax",
        compute_body_cpp="Reg::Exp(scores, scores, maskFull);",
        invariants=("Half-L1 <= 262144B",),
    )

    assert verify_source_against_facets(b1.compute_body_cpp, m_bot).passed is True
    assert (
        verify_source_against_facets(
            "Nd2NzParams p; DataCopyPad(ub, gm, pad);",
            m_bot,
        ).passed
        is False
    )
    assert (
        verify_source_against_facets(
            "CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);",
            m_bot,
        ).passed
        is False
    )
    assert (
        verify_source_against_facets(
            "Nd2NzParams p; DataCopyPad(ub, gm, pad); Reg::Exp(scores, scores, maskFull);",
            m_dma,
        ).passed
        is True
    )
    assert (
        verify_source_against_facets(
            "CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);",
            m_dma,
        ).passed
        is False
    )
    assert (
        verify_source_against_facets(
            "STAGE_DMA_LOAD(a, b); CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);",
            m_sync,
        ).passed
        is True
    )
    assert (
        verify_source_against_facets(
            "STAGE_DMA_LOAD(a, b); l1_slot_aliasing(p, k); Nd2NzParams p; CrossCoreSetFlag<0x2, PIPE_FIX>(0); CrossCoreWaitFlag(0);",
            m_top,
        ).passed
        is True
    )

    balanced_early_skip = """
    void KernelFn(int m) {
        for (int g = 0; g < 4; ++g) {
            if (m <= 0) {
                CrossCoreSetFlag<0x2, PIPE_FIX>(0);
                CrossCoreWaitFlag(0);
                continue;
            }
            CrossCoreSetFlag<0x2, PIPE_FIX>(1);
            CrossCoreWaitFlag(1);
        }
    }
    """
    assert verify_source_against_facets(balanced_early_skip, m_top).passed is True

    balanced_if_else = """
    void KernelFn(bool cond) {
        if (cond) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }
        CrossCoreWaitFlag(0);
    }
    """
    assert verify_source_against_facets(balanced_if_else, m_top).passed is True

    unbalanced_separate_ifs = """
    void KernelFn(bool a, bool b) {
        if (a) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }
        if (b) {
            CrossCoreWaitFlag(0);
        }
    }
    """
    rep_sep = verify_source_against_facets(unbalanced_separate_ifs, m_top)
    assert rep_sep.passed is False

    unbalanced_single = """
    void KernelFn(bool cond) {
        if (cond) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }
        CrossCoreWaitFlag(0);
    }
    """
    assert verify_source_against_facets(unbalanced_single, m_top).passed is False

    mismatched_ids = """
    CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    CrossCoreWaitFlag(1);
    """
    assert verify_source_against_facets(mismatched_ids, m_top).passed is False

    over_id = """
    CrossCoreSetFlag<0x2, PIPE_FIX>(16);
    CrossCoreWaitFlag(16);
    """
    assert verify_source_against_facets(over_id, m_top, max_flag_depth=15).passed is False

    over_inflight = "\n".join(["CrossCoreSetFlag<0x2, PIPE_FIX>(0);"] * 16 + ["CrossCoreWaitFlag(0);"] * 16)
    assert verify_source_against_facets(over_inflight, m_top, max_flag_depth=15).passed is False

    unbalanced_loop = """
    for (int i = 0; i < 4; ++i) {
        CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    }
    """
    assert verify_source_against_facets(unbalanced_loop, m_top).passed is False

    bad_mutex = "HwMutexLock(28); HwMutexUnlock(28);"
    assert verify_source_against_facets(bad_mutex, m_top).passed is False
