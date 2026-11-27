"""Unit tests for 13-symptom simplex conservation, signal extraction, and priority ordering."""

from __future__ import annotations

import math

import pytest

from tilefoundry_ascend.agent.kb.schema import (
    DETERMINISTIC_SYMPTOM_FAMILIES,
    SYMPTOM_FAMILY_TO_AXIS,
    SYMPTOM_FAMILY_TO_FACETS,
    SymptomFamily,
    compute_symptom_simplex,
    normalize_symptom_simplex,
    rank_symptom_families,
    should_escalate_diagnostic_or_creative,
)
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)


def _assert_simplex_valid(simplex: list[tuple[SymptomFamily, float]]) -> None:
    assert len(simplex) == 14
    assert {fam for fam, _ in simplex} == set(SymptomFamily)
    for fam, prob in simplex:
        assert 0.0 <= prob <= 1.0, f"out of range share for {fam.value}: {prob}"
    assert math.isclose(sum(prob for _, prob in simplex), 1.0, abs_tol=1e-9)


def test_symptom_taxonomy_facet_masks_and_normalization_conservation() -> None:
    """Verify 13 deterministic families, facet mappings, and decoupled simplex normalization."""
    assert len(SymptomFamily) == 14
    assert len(DETERMINISTIC_SYMPTOM_FAMILIES) == 13
    assert SymptomFamily.UNMODELED_SILICON_RESIDUAL not in DETERMINISTIC_SYMPTOM_FAMILIES
    assert SYMPTOM_FAMILY_TO_AXIS[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == "unmodeled_residual"
    assert (
        SYMPTOM_FAMILY_TO_FACETS[SymptomFamily.UNMODELED_SILICON_RESIDUAL]
        == VALID_FREEDOM_FACETS
    )
    assert set(SYMPTOM_FAMILY_TO_AXIS) == set(SymptomFamily)
    assert set(SYMPTOM_FAMILY_TO_FACETS) == set(SymptomFamily)

    base_mask = ConstraintRelaxationMask.bottom()
    for facets in SYMPTOM_FAMILY_TO_FACETS.values():
        assert facets
        assert all(f in VALID_FREEDOM_FACETS for f in facets)
        unlocked = base_mask.unlock(*facets)
        assert all(unlocked.is_unlocked(f) for f in facets)

    raw = {
        SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP: 0.6,
        SymptomFamily.SRAM_REGISTER_SPILL_STACK: 0.2,
        SymptomFamily.SCALAR_LOOP_TRAP: -1.0,
        SymptomFamily.DMA_L2_THRASH_REUSE_MISS: float("nan"),
        SymptomFamily.UNMODELED_SILICON_RESIDUAL: 0.9,
    }
    shares_at_roof = normalize_symptom_simplex(raw, roofline_eta=1.0, has_l2_pmu=False)
    assert shares_at_roof[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.75)
    assert shares_at_roof[SymptomFamily.SRAM_REGISTER_SPILL_STACK] == pytest.approx(0.25)
    assert shares_at_roof[SymptomFamily.SCALAR_LOOP_TRAP] == pytest.approx(0.0)
    assert shares_at_roof[SymptomFamily.DMA_L2_THRASH_REUSE_MISS] == pytest.approx(0.0)
    assert shares_at_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)
    assert math.isclose(sum(shares_at_roof.values()), 1.0, rel_tol=1e-9)

    zero_raw = {fam: 0.0 for fam in DETERMINISTIC_SYMPTOM_FAMILIES}
    under_roof = normalize_symptom_simplex(zero_raw, roofline_eta=0.5, has_l2_pmu=True)
    assert under_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(1.0)
    assert sum(under_roof[f] for f in DETERMINISTIC_SYMPTOM_FAMILIES) == pytest.approx(0.0)

    exact_roof = normalize_symptom_simplex(zero_raw, roofline_eta=1.0, has_l2_pmu=True)
    assert exact_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)


def test_observation_to_signal_extraction_and_conservation_profiles() -> None:
    """Verify ObservationVector5D signal extraction and conservation across waste profiles."""
    static_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120},
        waterlines={},
        gm_roundtrip_bytes=4096,
        microarch_profile={"pipeline_depth": 2},
    )
    static_obs = ObservationVector5D.from_tiers(static_ledger, telemetry={"dur_us": 25.0})
    assert static_obs.has_l2_pmu is False
    assert static_obs.num_stages == 2
    assert static_obs.compute_engine == "aiv_only"
    static_simplex = compute_symptom_simplex(static_obs)
    _assert_simplex_valid(static_simplex)
    assert dict(static_simplex)[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(1.0)
    assert rank_symptom_families(static_obs) == static_simplex

    pmu_simplex = compute_symptom_simplex(
        {
            "roofline_eta": 0.40,
            "measured_aiv_vec_ratio": 0.45,
            "l2_hit_rate": 0.40,
            "measured_mte2_wait_ratio": 0.35,
            "has_l2_pmu": True,
        }
    )
    _assert_simplex_valid(pmu_simplex)
    assert dict(pmu_simplex)[SymptomFamily.DMA_L2_THRASH_REUSE_MISS] == pytest.approx(1.0)

    decoupled_at_roof = compute_symptom_simplex(
        {
            "roofline_eta": 1.0,
            "transfer_inflation_ratio": 3.0,
            "spill_bytes": 512,
            "measured_aiv_vec_ratio": 0.90,
        }
    )
    _assert_simplex_valid(decoupled_at_roof)
    dec_map = dict(decoupled_at_roof)
    assert dec_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.5)
    assert dec_map[SymptomFamily.SRAM_REGISTER_SPILL_STACK] == pytest.approx(0.5)

    unmodeled_only = compute_symptom_simplex(
        {
            "roofline_eta": 0.50,
            "l0_vec_cycles": 200,
            "measured_aiv_vec_ratio": 0.50,
            "has_l2_pmu": True,
        }
    )
    _assert_simplex_valid(unmodeled_only)
    assert unmodeled_only[0] == (SymptomFamily.UNMODELED_SILICON_RESIDUAL, pytest.approx(1.0))

    ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=0.5,
        is_atom_aligned=False,
        working_set_sram_ratio=0.25,
        working_set_vreg_ratio=0.25,
        wave_occupancy_ratio=0.5,
        oi_ridge_ratio=0.2,
        inner_contiguous_bytes=16,
    )
    burst_simplex = compute_symptom_simplex(
        {
            "l0_vec_cycles": 100,
            "dma_burst_efficiency_ratio": 0.4,
            "unaligned_dma_padding_bytes": 16,
            "dcache_bypass_narrow_store": 4.0,
        },
        case_ratios=ratios,
    )
    _assert_simplex_valid(burst_simplex)
    assert dict(burst_simplex)[SymptomFamily.DMA_UNALIGNED_SHORT_BURST] == pytest.approx(1.0)


def test_topology_and_pipeline_hard_gates() -> None:
    """Verify aiv_only, aic_only, coupled_1c2v, and num_stages <= 1 hard gates."""
    aiv_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 160, "AIC.PIPE_FIX": 300},
        waterlines={},
        microarch_profile={"compute_engine": "aiv_only", "pipeline_depth": 2},
    )
    aiv_obs = ObservationVector5D.from_tiers(
        aiv_ledger,
        telemetry={"aic_fixpipe": 0.80, "aiv_vec_ratio": 0.50},
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
    )
    aiv_map = dict(compute_symptom_simplex(aiv_obs))
    assert aiv_map[SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE] == 0.0
    assert aiv_map[SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE] == 0.0

    aic_map = dict(
        compute_symptom_simplex(
            {
                "compute_engine": "aic_only",
                "vloopv2_hit": False,
                "vf_fusion_breaks": 3,
                "AIV0.PIPE_V": 100,
                "AIV1.PIPE_V": 0,
                "AIC.PIPE_M": 200,
                "AIC.PIPE_FIX": 400,
                "aic_mac_ratio": 0.50,
            }
        )
    )
    assert aic_map[SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE] == 0.0
    assert aic_map[SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED] == 0.0
    assert aic_map[SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE] > 0.0

    imbal_obs = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={},
    ).to_observation_5d()
    assert (
        dict(compute_symptom_simplex(imbal_obs))[SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE]
        == 0.0
    )
    coupled_map = dict(
        compute_symptom_simplex(
            imbal_obs,
            topology_axes=TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
        )
    )
    assert coupled_map[SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE] > 0.5

    single_stage_obs = ObservationVector5D.from_tiers(
        build_objective_physical_ledger(
            analysis_mode="exact_modulo_z3",
            is_legal=True,
            steady_state_ii=1,
            pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100},
            waterlines={},
            uses_dynamic_offset_pipelining=False,
            microarch_profile={"pipeline_depth": 1},
        ),
        telemetry={
            "pseudo_double_buffer_serialized": True,
            "sync_stall_ratio": 0.45,
            "aiv_vec_ratio": 0.40,
        },
    )
    single_map = dict(compute_symptom_simplex(single_stage_obs))
    assert single_map[SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER] == 0.0
    assert single_map[SymptomFamily.SYNC_BARRIER_SERIALIZATION] > 0.0
    assert ObservationVector5D.from_dict(single_stage_obs.to_dict()) == single_stage_obs


def test_priority_ordering_all_signals_and_escalation_policy() -> None:
    """Verify physical priority tie-breaking, remaining waste signals, and escalation."""
    tied = compute_symptom_simplex(
        {
            "l0_cube_cycles": 0,
            "l0_vec_cycles": 0,
            "transfer_inflation_ratio": 3.5,
            "has_l2_pmu": True,
            "roofline_eta": 0.5,
        }
    )
    _assert_simplex_valid(tied)
    assert dict(tied)[SymptomFamily.SCALAR_LOOP_TRAP] == pytest.approx(0.5)
    assert dict(tied)[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.5)
    assert tied[0][0] == SymptomFamily.SCALAR_LOOP_TRAP
    assert tied[1][0] == SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP

    zero_all = compute_symptom_simplex({"roofline_eta": 1.0})
    _assert_simplex_valid(zero_all)
    zero_tail = [fam for fam, prob in zero_all if prob == 0.0]
    assert zero_tail.index(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP) < zero_tail.index(
        SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT
    )
    assert zero_tail.index(SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT) < zero_tail.index(
        SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE
    )

    ineff = dict(
        compute_symptom_simplex(
            {
                "l0_vec_cycles": 100,
                "measured_aiv_vec_ratio": 0.80,
                "effective_compute_eta": 0.15,
                "has_l2_pmu": True,
            }
        )
    )
    assert ineff[SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT] == pytest.approx(1.0)

    bank = dict(
        compute_symptom_simplex(
            {
                "l0_vec_cycles": 100,
                "measured_aiv_vec_ratio": 0.40,
                "spill_bytes": 0,
                "bank_conflict_ratio": 0.30,
                "has_l2_pmu": True,
            }
        )
    )
    assert bank[SymptomFamily.SRAM_UB_L1_BANK_CONFLICT] == pytest.approx(1.0)

    wave = dict(
        compute_symptom_simplex(
            {"l0_vec_cycles": 100, "wave_occupancy": 0.25, "grid_blocks": 8, "num_aiv_cores": 56}
        )
    )
    assert wave[SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL] == pytest.approx(1.0)

    unmodeled_dom = [
        (SymptomFamily.UNMODELED_SILICON_RESIDUAL, 0.70),
        (SymptomFamily.DMA_L2_THRASH_REUSE_MISS, 0.30),
    ]
    assert should_escalate_diagnostic_or_creative(unmodeled_dom, stagnation_rounds=0) is False
    assert should_escalate_diagnostic_or_creative(unmodeled_dom, stagnation_rounds=1) is True

    concentrated = [
        (SymptomFamily.DMA_L2_THRASH_REUSE_MISS, 0.90),
        (SymptomFamily.UNMODELED_SILICON_RESIDUAL, 0.10),
    ]
    assert should_escalate_diagnostic_or_creative(concentrated, stagnation_rounds=2) is False

    flat_entropy = [(fam, 1.0 / float(len(SymptomFamily))) for fam in SymptomFamily]
    assert should_escalate_diagnostic_or_creative(flat_entropy, stagnation_rounds=1) is True
    assert should_escalate_diagnostic_or_creative([], stagnation_rounds=2) is False
