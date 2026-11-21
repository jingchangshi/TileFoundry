"""WP-2 end-to-end tests for the 5-axis observation space and 1:1 facet unlocking."""

from __future__ import annotations

import pytest

from tilefoundry_ascend.agent.kb.schema import build_predicate_context
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    evaluate_ascendc_candidate,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compiler_summary_from_residual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.router.capability_gate import (
    EscalationDecision,
    RouteDecision,
    StructuralFeatures,
    normalize_router_tier,
    route_from_op_spec,
    route_initial_by_input_contract,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.observation_space import (
    AxisResidual5D,
    ObservationVector5D,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.templates.registry import relax_facets_for_bottleneck


def _rich_ledger() -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=3,
        pipe_cycles={
            "AIC.PIPE_M": 100,
            "AIC.CUBE": 20,
            "AIV0.PIPE_V": 300,
            "AIV1.PIPE_V": 100,
        },
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=4096,
                explicit_buffers={"ub_out": 4096},
                implicit_buffers={},
            ),
            "L1": build_scope_waterline(
                scope="L1",
                capacity_bytes=8192,
                explicit_buffers={"l1_in": 9000},
                implicit_buffers={},
            ),
        },
        mte2_inflight_per_aiv=2,
        unaligned_dma_padding_bytes=64,
        grid_blocks=8,
        wave_occupancy=0.8,
        gm_roundtrip_bytes=4096,
        dominant_contig_bytes=64,
        has_indirect_gather=True,
        has_online_rescale_loop=True,
        has_lowbit_quant=True,
        in_loop_pipe_all_barriers=1,
        dma_burst_efficiency_ratio=0.75,
        uses_dynamic_offset_pipelining=True,
        ub_bank_conflict_risk=0.1,
        l1_bank_conflict_risk=0.05,
        microarch_profile={"subkernel_count": 3, "mte2_bytes_hint": 2048},
    )


def _rich_compiler_summary() -> BishengResourceSummary:
    return BishengResourceSummary(
        vec_regs=32,
        scalar_regs=16,
        spill_bytes=128,
        stack_bytes=64,
        vloopv2_loops=0,
        software_loops=2,
        has_local_membar_in_vf_call=True,
        vf_fusion_breaks=2,
        alias_serialized_buffers=1,
        dual_issue_ipc_collapse=0.1,
        dcache_dcci_flushes=1,
    )


def _rich_telemetry() -> dict[str, object]:
    return {
        "measured_latency_us": 42.0,
        "aic_mac_ratio": 0.6,
        "aiv_vec_ratio": 0.5,
        "roofline_eta": 0.7,
        "mte2_wait_ratio": 0.3,
        "aic_fixpipe": 0.2,
        "l2_hit_rate": 0.6,
        "bank_conflict_ratio": 0.25,
        "sync_stall_ratio": 0.22,
        "pseudo_double_buffer_serialized": True,
    }


def _quiet_ledger() -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 10},
        waterlines={},
    )


def test_from_tiers_aligns_all_three_telemetry_tiers() -> None:
    ledger = _rich_ledger()
    observation = ledger.to_observation_5d(
        compiler_summary=_rich_compiler_summary(),
        telemetry=_rich_telemetry(),
    )
    assert observation.compute_issue.l0_cube_cycles == 120
    assert observation.compute_issue.l0_vec_cycles == 400
    assert observation.compute_issue.l0_cube_to_vec_ratio == 120 / 400
    assert observation.compute_issue.l1_vloopv2_loops == 0
    assert observation.compute_issue.l1_software_loops == 2
    assert observation.compute_issue.l1_vf_fusion_breaks == 2
    assert observation.compute_issue.l1_dual_issue_collapse == 0.1
    assert observation.compute_issue.l2_aic_mac_ratio == 0.6
    assert observation.compute_issue.l2_aiv_vec_ratio == 0.5
    assert observation.compute_issue.l2_roofline_eta == 0.7
    assert observation.offchip_dma.l0_gm_roundtrip_bytes == 4096
    assert observation.offchip_dma.l0_dominant_contig_bytes == 64
    assert observation.offchip_dma.l0_dma_burst_efficiency == 0.75
    assert observation.offchip_dma.l0_mte2_inflight_per_aiv == 2
    assert observation.offchip_dma.l0_unaligned_padding_bytes == 64
    assert observation.offchip_dma.l1_dcache_dcci_flushes == 1
    assert observation.offchip_dma.l1_objdump_mte2_bytes == 2048
    assert observation.offchip_dma.l2_mte2_wait_ratio == 0.3
    assert observation.offchip_dma.l2_fixpipe_ratio == 0.2
    assert observation.offchip_dma.l2_cache_hit_rate == 0.6
    assert observation.onchip_sram_reg.l0_max_overflow_bytes == 808
    assert observation.onchip_sram_reg.l0_ub_peak_bytes == 4096
    assert observation.onchip_sram_reg.l0_l1_peak_bytes == 9000
    assert observation.onchip_sram_reg.l0_ub_bank_conflict_risk == 0.1
    assert observation.onchip_sram_reg.l0_l1_bank_conflict_risk == 0.05
    assert observation.onchip_sram_reg.l0_degraded_single_buffers == 0
    assert observation.onchip_sram_reg.l1_vec_regs == 32
    assert observation.onchip_sram_reg.l1_scalar_regs == 16
    assert observation.onchip_sram_reg.l1_spill_bytes == 128
    assert observation.onchip_sram_reg.l1_stack_bytes == 64
    assert observation.onchip_sram_reg.l2_bank_conflict_ratio == 0.25
    assert observation.pipeline_sync.l0_steady_state_ii == 3
    assert observation.pipeline_sync.l0_in_loop_pipe_all_barriers == 1
    assert observation.pipeline_sync.l0_uses_dynamic_offset_pipe is True
    assert observation.pipeline_sync.l1_local_membar_in_vf is True
    assert observation.pipeline_sync.l1_alias_serialized_buffers == 1
    assert observation.pipeline_sync.l2_sync_stall_ratio == 0.22
    assert observation.pipeline_sync.l2_pseudo_double_buffer is True
    assert observation.grid_irregularity.l0_grid_blocks == 8
    assert observation.grid_irregularity.l0_wave_occupancy == 0.8
    assert observation.grid_irregularity.l0_has_indirect_gather is True
    assert observation.grid_irregularity.l0_has_online_rescale is True
    assert observation.grid_irregularity.l0_has_lowbit_quant is True
    assert observation.grid_irregularity.l1_subkernel_count == 3
    assert observation.grid_irregularity.l2_measured_latency_us == 42.0
    assert observation.has_l2_telemetry is True


def test_observation_provenance_separates_l1_summary_from_l2_measurement() -> None:
    ledger = _rich_ledger()
    full = ledger.to_observation_5d(
        compiler_summary=_rich_compiler_summary(),
        telemetry=_rich_telemetry(),
    )
    assert full.has_l1_compiler_summary is True
    assert full.measured_l2_fields == (
        "l2_measured_latency_us",
        "l2_aic_mac_ratio",
        "l2_aiv_vec_ratio",
        "l2_roofline_eta",
        "l2_mte2_wait_ratio",
        "l2_fixpipe_ratio",
        "l2_cache_hit_rate",
        "l2_bank_conflict_ratio",
        "l2_sync_stall_ratio",
        "l2_pseudo_double_buffer",
    )

    no_compiler = ledger.to_observation_5d(telemetry=_rich_telemetry())
    assert no_compiler.has_l1_compiler_summary is False
    assert no_compiler.measured_l2_fields == full.measured_l2_fields

    no_hit_rate = ledger.to_observation_5d(
        compiler_summary=_rich_compiler_summary(),
        telemetry={"measured_latency_us": 42.0},
    )
    assert no_hit_rate.offchip_dma.l2_cache_hit_rate == 1.0
    assert no_hit_rate.measured_l2_fields == ("l2_measured_latency_us",)

    restored = ObservationVector5D.from_dict(no_hit_rate.to_dict())
    assert restored == no_hit_rate
    assert restored.has_l1_compiler_summary is True
    assert restored.measured_l2_fields == ("l2_measured_latency_us",)

    proxy = ObservationVector5D.from_tiers(
        ledger,
        compiler_summary=_rich_compiler_summary(),
        telemetry={"l2_hit_rate": 0.3, "is_analytical_proxy": True},
    )
    assert proxy.has_l2_telemetry is False
    assert proxy.measured_l2_fields == ()


def test_compiler_summary_from_residual_round_trips_real_l1_penalties() -> None:
    ledger = _rich_ledger()
    summary = _rich_compiler_summary()
    residual = compute_static_dynamic_residual(
        ledger,
        _rich_telemetry(),
        compiler_summary=summary,
    )
    assert residual.has_l1_compiler_summary is True
    assert compiler_summary_from_residual(residual) == summary

    l2_only = compute_static_dynamic_residual(ledger, _rich_telemetry())
    assert l2_only.has_l1_compiler_summary is False
    assert compiler_summary_from_residual(l2_only) is None
    assert "l2_cache_hit_rate" in l2_only.measured_l2_fields

    legacy = StaticDynamicResidual(30.0, 0.8, 0.2, 0.2, 0.1, 0.35, 0.2)
    assert legacy.has_l1_compiler_summary is False
    assert legacy.measured_l2_fields == ()
    assert compiler_summary_from_residual(legacy) is None


def test_compute_axis_residuals_uses_same_coaxial_thresholds() -> None:
    observation = _rich_ledger().to_observation_5d(
        compiler_summary=_rich_compiler_summary(),
        telemetry=_rich_telemetry(),
    )
    residuals = observation.compute_axis_residuals()
    assert residuals.compute_issue_gap == 0.5
    assert residuals.offchip_dma_gap == 0.4
    assert residuals.onchip_sram_reg_gap == 1.0
    assert residuals.pipeline_sync_gap == 0.6
    assert residuals.grid_wave_gap == 0.2
    assert residuals.dominant_axis == "onchip_sram_reg"
    assert observation.to_dict()["residuals"] == {
        "compute_issue_gap": 0.5,
        "offchip_dma_gap": 0.4,
        "onchip_sram_reg_gap": 1.0,
        "pipeline_sync_gap": 0.6,
        "grid_wave_gap": 0.2,
    }


def test_grid_wave_gap_charges_tail_wave_of_multi_wave_grids() -> None:
    """A partially filled final wave wastes occupancy even when the grid spans whole waves."""

    def gap(wave_occupancy: float) -> float:
        ledger = build_objective_physical_ledger(
            analysis_mode="exact_modulo_z3",
            is_legal=True,
            steady_state_ii=2,
            pipe_cycles={"AIC.PIPE_M": 10},
            waterlines={},
            wave_occupancy=wave_occupancy,
        )
        return ledger.to_observation_5d().compute_axis_residuals().grid_wave_gap

    assert gap(1.07) == pytest.approx(0.465)
    assert gap(2.0) == 0.0
    assert gap(1.5) == pytest.approx(0.25)
    assert gap(0.8) == pytest.approx(0.2)
    assert gap(0.0) == 0.0


def test_compute_issue_gap_activates_for_compute_limited_low_roofline_eta() -> None:
    """Compute-bound kernels with poor Roofline efficiency expose the compute gap."""
    ledger = _quiet_ledger()

    def gap(telemetry: dict[str, object]) -> float:
        observation = ledger.to_observation_5d(telemetry=telemetry)
        return observation.compute_axis_residuals().compute_issue_gap

    assert gap({"aic_mac_ratio": 0.8, "aiv_vec_ratio": 0.1, "roofline_eta": 0.4}) == (
        pytest.approx(0.6)
    )
    assert gap({"aic_mac_ratio": 0.1, "aiv_vec_ratio": 0.1, "roofline_eta": 0.4}) == 0.0
    assert gap({"aic_mac_ratio": 0.8, "aiv_vec_ratio": 0.1, "roofline_eta": 0.7}) == 0.0
    assert gap({"aic_mac_ratio": 0.8, "aiv_vec_ratio": 0.1, "roofline_eta": 0.0}) == 0.0


def test_registry_merges_dominant_bound_and_overflow_fallback_when_5d_is_quiet() -> None:
    """A quiet 5D residual still merges dominant-bound and overflow fallback facets."""
    quiet = compute_static_dynamic_residual(_quiet_ledger(), {})
    assert quiet.observation is not None
    assert quiet.observation.compute_axis_residuals() == AxisResidual5D(0.0, 0.0, 0.0, 0.0, 0.0)
    bottom = ConstraintRelaxationMask.bottom()
    assert relax_facets_for_bottleneck(
        bottom,
        dominant_bound="hbm",
        residual=quiet,
    ).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "dma_and_cache_hierarchy",
    )
    assert relax_facets_for_bottleneck(
        bottom,
        residual=quiet,
        l1_overflow=True,
    ).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
    )
    assert relax_facets_for_bottleneck(
        bottom,
        residual=quiet,
        ub_overflow=True,
    ).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
    )
    assert relax_facets_for_bottleneck(bottom, residual=quiet).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
    )


def test_predicate_context_exports_axis_gaps_at_current_and_prev_levels() -> None:
    """Rules read all five axis gaps and the dominant axis on both context levels."""
    ledger = _rich_ledger()
    residual = compute_static_dynamic_residual(
        ledger,
        _rich_telemetry(),
        compiler_summary=_rich_compiler_summary(),
    )
    ctx = build_predicate_context(ledger, residual=residual)
    expected: dict[str, float | str] = {
        "compute_issue_gap": 0.5,
        "offchip_dma_gap": 0.4,
        "onchip_sram_reg_gap": 1.0,
        "pipeline_sync_gap": 0.6,
        "grid_wave_gap": 0.2,
        "dominant_axis": "onchip_sram_reg",
    }
    prev_ctx = ctx["prev"]
    assert isinstance(prev_ctx, dict)
    for key, value in expected.items():
        assert ctx[key] == value
        assert prev_ctx[key] == value
    with_prev = build_predicate_context(
        ledger,
        residual=residual,
        prev_ledger=_rich_ledger(),
    )
    parent_ctx = with_prev["prev"]
    assert isinstance(parent_ctx, dict)
    assert parent_ctx["onchip_sram_reg_gap"] == 1.0
    assert parent_ctx["grid_wave_gap"] == 0.2
    assert parent_ctx["dominant_axis"] == "onchip_sram_reg"
    static_ctx = build_predicate_context(_quiet_ledger())
    assert static_ctx["dominant_axis"] == "compute_issue"
    static_prev = static_ctx["prev"]
    assert isinstance(static_prev, dict)
    assert static_prev["compute_issue_gap"] == 0.0


def test_axis_residual_thresholds_unlock_exactly_one_facet_each() -> None:
    bottom = ConstraintRelaxationMask.bottom()
    assert AxisResidual5D(0.24, 0.24, 0.19, 0.19, 0.24).unlock_causal_facets(bottom) == bottom
    assert AxisResidual5D(0.25, 0.0, 0.0, 0.0, 0.0).unlock_causal_facets(bottom).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "exec_mode_and_spr",
    )
    assert AxisResidual5D(0.0, 0.25, 0.0, 0.0, 0.0).unlock_causal_facets(bottom).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "dma_and_cache_hierarchy",
    )
    assert AxisResidual5D(0.0, 0.0, 0.20, 0.0, 0.0).unlock_causal_facets(bottom).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
    )
    assert AxisResidual5D(0.0, 0.0, 0.0, 0.20, 0.0).unlock_causal_facets(bottom).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "pipeline_and_sync_graph",
    )
    assert AxisResidual5D(0.0, 0.0, 0.0, 0.0, 0.25).unlock_causal_facets(bottom).unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "grid_and_workspace_topology",
    )
    saturated = AxisResidual5D(1.0, 1.0, 1.0, 1.0, 1.0).unlock_causal_facets(bottom)
    assert saturated == ConstraintRelaxationMask.top()


def test_missing_telemetry_is_flagged_and_never_fabricated() -> None:
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 10},
        waterlines={},
    )
    observation = ledger.to_observation_5d()
    assert observation.has_l2_telemetry is False
    assert observation.has_l1_compiler_summary is False
    assert observation.measured_l2_fields == ()
    assert observation.compute_issue.l2_aic_mac_ratio == 0.0
    assert observation.compute_issue.l2_aiv_vec_ratio == 0.0
    assert observation.compute_issue.l2_roofline_eta == 0.0
    assert observation.offchip_dma.l2_mte2_wait_ratio == 0.0
    assert observation.offchip_dma.l2_fixpipe_ratio == 0.0
    assert observation.offchip_dma.l2_cache_hit_rate == 1.0
    assert observation.onchip_sram_reg.l2_bank_conflict_ratio == 0.0
    assert observation.pipeline_sync.l2_sync_stall_ratio == 0.0
    assert observation.pipeline_sync.l2_pseudo_double_buffer is False
    assert observation.grid_irregularity.l2_measured_latency_us == 0.0
    assert observation.compute_axis_residuals() == AxisResidual5D(0.0, 0.0, 0.0, 0.0, 0.0)

    proxy = ObservationVector5D.from_tiers(
        ledger,
        telemetry={"dur_us": 10.0, "is_analytical_proxy": True},
    )
    assert proxy.has_l2_telemetry is False
    assert proxy.grid_irregularity.l2_measured_latency_us == 10.0


def test_registry_unlocks_only_the_axis_residual_facet_from_observation() -> None:
    ledger = _rich_ledger()
    residual = compute_static_dynamic_residual(ledger, _rich_telemetry())
    assert residual.observation is not None
    relaxed = relax_facets_for_bottleneck(
        ConstraintRelaxationMask.bottom(),
        residual=residual,
    )
    assert relaxed.unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
        "dma_and_cache_hierarchy",
        "pipeline_and_sync_graph",
    )

    overflow_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 10},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=1024,
                explicit_buffers={"ub_stage": 2048},
                implicit_buffers={},
            )
        },
    )
    overflow_observation = overflow_ledger.to_observation_5d()
    assert overflow_observation.compute_axis_residuals().onchip_sram_reg_gap == 1.0
    overflow_residual = compute_static_dynamic_residual(overflow_ledger, {})
    overflow_relaxed = relax_facets_for_bottleneck(
        ConstraintRelaxationMask.bottom(),
        residual=overflow_residual,
    )
    assert overflow_relaxed.unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
    )


def test_registry_legacy_residual_without_observation_keeps_old_causal_semantics() -> None:
    legacy = StaticDynamicResidual(30.0, 0.8, 0.2, 0.2, 0.1, 0.35, 0.2)
    assert legacy.observation is None
    relaxed = relax_facets_for_bottleneck(
        ConstraintRelaxationMask.bottom(),
        residual=legacy,
    )
    assert relaxed.unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "pipeline_and_sync_graph",
    )


def test_registry_observation_residuals_are_the_exclusive_unlock_driver() -> None:
    overflow_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 10},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=1024,
                explicit_buffers={"ub_stage": 2048},
                implicit_buffers={},
            )
        },
    )
    observed = compute_static_dynamic_residual(overflow_ledger, {})
    assert observed.observation is not None
    relaxed = relax_facets_for_bottleneck(
        ConstraintRelaxationMask.bottom().unlock("grid_and_workspace_topology"),
        dominant_bound="dma",
        residual=observed,
    )
    assert relaxed.unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
        "grid_and_workspace_topology",
    )

    legacy = StaticDynamicResidual(30.0, 0.8, 0.2, 0.2, 0.1, 0.1, 0.2)
    assert legacy.observation is None
    legacy_relaxed = relax_facets_for_bottleneck(
        ConstraintRelaxationMask.bottom(),
        dominant_bound="dma",
        residual=legacy,
    )
    assert legacy_relaxed.unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
        "dma_and_cache_hierarchy",
    )


def test_observation_and_residual_serialization_round_trip() -> None:
    ledger = _rich_ledger()
    observation = ledger.to_observation_5d(
        compiler_summary=_rich_compiler_summary(),
        telemetry=_rich_telemetry(),
    )
    assert ObservationVector5D.from_dict(observation.to_dict()) == observation
    assert ObservationVector5D.from_dict(observation.to_dict()).has_l2_telemetry is True
    residual = compute_static_dynamic_residual(
        ledger,
        _rich_telemetry(),
        compiler_summary=_rich_compiler_summary(),
    )
    restored = StaticDynamicResidual.from_dict(residual.to_dict())
    assert restored == residual
    assert restored.observation == observation
    legacy_payload = residual.to_dict()
    legacy_payload.pop("observation")
    restored_legacy = StaticDynamicResidual.from_dict(legacy_payload)
    assert restored_legacy.observation is None
    assert restored_legacy.measured_sync_stall_ratio == residual.measured_sync_stall_ratio


_PROTO: dict[str, object] = {
    "op_name": "demo_reduce",
    "input_dtypes": ["float16"],
    "output_dtypes": ["float16"],
    "topology_axes": {
        "compute_engine": "aiv_only",
        "memory_ring": "gm_ub_stream",
        "iteration_regularity": "affine_simd",
    },
    "structural_features": {
        "phi_1": True,
        "phi_2": 0.0,
        "phi_3": "none",
        "phi_4": False,
        "phi_5": "none",
        "phi_6": False,
    },
}


def test_structural_features_project_real_case_ratios() -> None:
    ratios = CasePhysicsRatios(
        case_id=7,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.1,
        working_set_vreg_ratio=0.1,
        wave_occupancy_ratio=0.375,
        oi_ridge_ratio=1.0,
        inner_contiguous_bytes=48,
    )
    features = StructuralFeatures.from_case_ratios(
        ratios,
        static_affine_bounds=False,
        cube_to_vector_ratio=2.0,
        quant_granularity="per_tensor",
        indirect_sparse_gather=True,
        cross_iter_recurrence="none",
        dynamic_cross_core_handoff=False,
    )
    assert features.phi_7_inner_contiguous_bytes == 48
    assert features.phi_8_persistent_wave_occupancy == 0.375
    assert features.phi_2_cube_to_vector_ratio == 2.0
    assert features.phi_3_quant_granularity == "per_tensor"
    assert features.phi_4_indirect_sparse_gather is True


def test_router_without_case_ratios_never_fabricates_alignment_defaults() -> None:
    decision = route_from_op_spec(_PROTO)
    assert decision.features.phi_7_inner_contiguous_bytes == 0
    assert decision.features.phi_8_persistent_wave_occupancy == 0.0
    initial = route_initial_by_input_contract(_PROTO)
    assert initial.features.phi_7_inner_contiguous_bytes == 0
    assert initial.features.phi_8_persistent_wave_occupancy == 0.0


def test_router_consumes_case_ratios_and_case_specialized_axes() -> None:
    ratios = CasePhysicsRatios(
        case_id=9,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.1,
        working_set_vreg_ratio=0.1,
        wave_occupancy_ratio=0.2,
        oi_ridge_ratio=1.0,
        has_empty_group=True,
        inner_contiguous_bytes=256,
    )
    expected_axes = TopologyAxes("aiv_only", "gm_ub_stream", "ragged_group")
    decision = route_from_op_spec(_PROTO, case_ratios=ratios)
    assert decision.features.phi_7_inner_contiguous_bytes == 256
    assert decision.features.phi_8_persistent_wave_occupancy == 0.2
    assert decision.topology_axes == expected_axes
    assert any("case ratios" in reason for reason in decision.rationale)
    initial = route_initial_by_input_contract(_PROTO, case_ratios=ratios)
    assert initial.features.phi_7_inner_contiguous_bytes == 256
    assert initial.features.phi_8_persistent_wave_occupancy == 0.2
    assert initial.topology_axes == expected_axes


def test_case_ratios_never_downgrade_input_contract_hard_walls() -> None:
    subbyte_proto: dict[str, object] = {
        "op_name": "dequant_gemv",
        "input_dtypes": ["float4_e2m1"],
        "output_dtypes": ["float16"],
    }
    neutral_ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.1,
        working_set_vreg_ratio=0.1,
        wave_occupancy_ratio=1.0,
        oi_ridge_ratio=1.0,
        inner_contiguous_bytes=32,
    )
    decision = route_initial_by_input_contract(
        subbyte_proto,
        case_ratios=neutral_ratios,
    )
    assert decision.recommended_backend_tier == "ascendc_template"
    assert decision.rationale == ("input_contract: sub-byte packed dtypes ('float4_e2m1',)",)


def test_router_tier_aliases_normalize_to_internal_ids_at_the_boundary() -> None:
    assert normalize_router_tier("tilelang_dsl") == "tilelang_dsl"
    for alias in ("cce_cpp20", "cce_trackb", "hybrid_trackb", "tl_templates_cpp20", "ascendc_trackb"):
        assert normalize_router_tier(alias) == "ascendc_template"
    assert normalize_router_tier("CCE_CPP20") == "ascendc_template"
    assert normalize_router_tier("inline_cce_vf") == "inline_ascendc_vf"

    with pytest.raises(ValueError, match="unsupported router tier"):
        normalize_router_tier("unknown_tier")

    features = StructuralFeatures(
        phi_1_static_affine_bounds=False,
        phi_2_cube_to_vector_ratio=0.0,
        phi_3_quant_granularity="none",
        phi_4_indirect_sparse_gather=False,
        phi_5_cross_iter_recurrence="none",
        phi_6_dynamic_cross_core_handoff=False,
        phi_7_inner_contiguous_bytes=0,
        phi_8_persistent_wave_occupancy=0.0,
    )
    decision = RouteDecision(
        track="developer",
        features=features,
        rationale=(),
        recommended_primitives=(),
        recommended_backend_tier="cce_cpp20",
    )
    assert decision.recommended_backend_tier == "ascendc_template"
    escalation = EscalationDecision(
        should_escalate=True,
        from_tier="cce_trackb",
        to_tier="inline_cce_vf",
    )
    assert escalation.from_tier == "ascendc_template"
    assert escalation.to_tier == "inline_ascendc_vf"


_TRACKB_VECTOR_SNIPPET = """
constexpr int TILE_ELEMENTS = 64;
Exp(dst, src, mask);
"""

_STATIC_COPY_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(X: T.Tensor((64, 128), "float32"), Y: T.Tensor((64, 128), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64, 128), "float32")
            T.copy(X[0:64, 0:128], a_ub)
            T.copy(a_ub, Y[0:64, 0:128])
    return main
"""


def _sub32_case_ratios() -> CasePhysicsRatios:
    return CasePhysicsRatios(
        case_id=11,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.1,
        working_set_vreg_ratio=0.1,
        wave_occupancy_ratio=0.25,
        oi_ridge_ratio=1.0,
        inner_contiguous_bytes=16,
    )


def test_trackb_stage_a_without_case_ratios_never_fabricates_physical_case() -> None:
    bottom = ConstraintRelaxationMask.bottom()
    unprojected = evaluate_ascendc_candidate(
        _TRACKB_VECTOR_SNIPPET, freedom_mask=bottom, enable_track_router=True
    )
    assert unprojected.passed is True
    assert unprojected.features is not None
    assert unprojected.features.phi_7_inner_contiguous_bytes == 0
    assert unprojected.features.phi_8_persistent_wave_occupancy == 0.0
    assert unprojected.route is not None
    assert unprojected.route.track == "developer"

    projected = evaluate_ascendc_candidate(
        _TRACKB_VECTOR_SNIPPET,
        freedom_mask=bottom,
        enable_track_router=True,
        case_ratios=_sub32_case_ratios(),
    )
    assert projected.passed is True
    assert projected.features is not None
    assert projected.features.phi_7_inner_contiguous_bytes == 16
    assert projected.features.phi_8_persistent_wave_occupancy == 0.25
    assert projected.route is not None
    assert projected.route.track == "simt_vf"


def test_stage_a_static_projects_case_ratios_into_router_features() -> None:
    base = run_stage_a_static(
        _STATIC_COPY_DSL, factory_name="make_kernel", enable_track_router=True
    )
    assert base.passed is True
    assert base.features is not None
    ratios = _sub32_case_ratios()
    projected = run_stage_a_static(
        _STATIC_COPY_DSL,
        factory_name="make_kernel",
        enable_track_router=True,
        case_ratios=ratios,
    )
    assert projected.passed is True
    assert projected.features is not None
    assert projected.features.phi_7_inner_contiguous_bytes == ratios.inner_contiguous_bytes
    assert projected.features.phi_8_persistent_wave_occupancy == ratios.wave_occupancy_ratio
    assert projected.features.phi_1_static_affine_bounds == base.features.phi_1_static_affine_bounds
    assert projected.route is not None
    assert projected.route.track == "simt_vf"
