"""Tests for 14-family simplex conservation, Stage-1 topology gates, and escalation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.kb.schema import (
    DETERMINISTIC_SYMPTOM_FAMILIES,
    SYMPTOM_FAMILY_TO_AXIS,
    SYMPTOM_FAMILY_TO_FACETS,
    AscendCausalRule,
    SymptomFamily,
    compute_symptom_simplex,
    load_seed_rules,
    rank_symptom_families,
    save_rules_yaml,
    should_escalate_diagnostic_or_creative,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    compute_eq3_hap_objective,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import VALID_FREEDOM_FACETS, TopologyAxes


def _assert_simplex_valid(simplex: list[tuple[SymptomFamily, float]]) -> None:
    assert len(simplex) == 14
    families = [fam for fam, _ in simplex]
    assert set(families) == set(SymptomFamily)
    for fam, prob in simplex:
        assert 0.0 <= prob <= 1.0, f"out of range share for {fam.value}: {prob}"
    total = sum(prob for _, prob in simplex)
    assert abs(total - 1.0) < 1e-9, f"simplex sum drifted from 1.0: {total}"


def test_fourteenth_symptom_family_contract_and_seed_rule_guard(tmp_path: Path) -> None:
    """Verify 14-family taxonomy, facet/axis mappings, and rule validation."""
    assert len(SymptomFamily) == 14
    assert len(DETERMINISTIC_SYMPTOM_FAMILIES) == 13
    assert SymptomFamily.UNMODELED_SILICON_RESIDUAL not in DETERMINISTIC_SYMPTOM_FAMILIES
    assert (
        SYMPTOM_FAMILY_TO_AXIS[SymptomFamily.UNMODELED_SILICON_RESIDUAL]
        == "unmodeled_residual"
    )
    assert (
        SYMPTOM_FAMILY_TO_FACETS[SymptomFamily.UNMODELED_SILICON_RESIDUAL]
        == VALID_FREEDOM_FACETS
    )

    seed_rules = load_seed_rules()
    assert len(seed_rules) >= 41
    for rule in seed_rules:
        assert SymptomFamily(rule.symptom_family) in DETERMINISTIC_SYMPTOM_FAMILIES

    with pytest.raises(ValueError, match="invalid symptom_family"):
        AscendCausalRule(
            rule_id="R_BAD_UNMODELED",
            regime_guard="is_legal",
            precondition="True",
            transform_template="Do nothing.",
            expected_postcondition="True",
            source_provenance="test",
            symptom_family=SymptomFamily.UNMODELED_SILICON_RESIDUAL.value,
        )

    bad_yaml = tmp_path / "bad_rules.yaml"
    valid_rule = AscendCausalRule(
        rule_id="R_OK",
        regime_guard="is_legal",
        precondition="True",
        transform_template="Optimize.",
        expected_postcondition="True",
        source_provenance="test",
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
    )
    save_rules_yaml([valid_rule], bad_yaml)
    assert len(load_seed_rules(bad_yaml)) == 1
    bad_yaml.write_text(
        bad_yaml.read_text(encoding="utf-8").replace(
            SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
            SymptomFamily.UNMODELED_SILICON_RESIDUAL.value,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid symptom_family"):
        load_seed_rules(bad_yaml)


def test_simplex_conservation_across_static_pmu_zero_excess_and_overloaded_profiles() -> None:
    """Verify exact sum-to-one conservation across static-only, PMU, saturated, and overloaded."""
    static_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120},
        waterlines={},
        gm_roundtrip_bytes=4096,
        microarch_profile={"pipeline_depth": 2},
    )
    static_obs = ObservationVector5D.from_tiers(
        static_ledger,
        telemetry={"dur_us": 25.0},
    )
    assert static_obs.has_l2_pmu is False
    assert static_obs.num_stages == 2
    assert static_obs.compute_engine == "aiv_only"
    static_simplex = compute_symptom_simplex(static_obs)
    _assert_simplex_valid(static_simplex)
    static_map = dict(static_simplex)
    assert static_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(1.0)
    assert static_map[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)

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
    pmu_map = dict(pmu_simplex)
    assert pmu_map[SymptomFamily.DMA_L2_THRASH_REUSE_MISS] == pytest.approx(1.0)
    assert pmu_map[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)

    decoupled_at_roof = compute_symptom_simplex(
        {
            "roofline_eta": 1.0,
            "transfer_inflation_ratio": 3.0,
            "spill_bytes": 512,
            "measured_aiv_vec_ratio": 0.90,
        }
    )
    _assert_simplex_valid(decoupled_at_roof)
    decoupled_map = dict(decoupled_at_roof)
    assert decoupled_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.5)
    assert decoupled_map[SymptomFamily.SRAM_REGISTER_SPILL_STACK] == pytest.approx(0.5)
    assert decoupled_map[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)

    unmodeled_only = compute_symptom_simplex(
        {
            "roofline_eta": 0.50,
            "l0_vec_cycles": 200,
            "measured_aiv_vec_ratio": 0.50,
            "has_l2_pmu": True,
        }
    )
    _assert_simplex_valid(unmodeled_only)
    assert unmodeled_only[0][0] == SymptomFamily.UNMODELED_SILICON_RESIDUAL
    assert unmodeled_only[0][1] == pytest.approx(1.0)

    overloaded_no_pmu = compute_symptom_simplex(
        {
            "roofline_eta": 0.50,
            "transfer_inflation_ratio": 3.0,
            "spill_bytes": 512,
            "bank_conflict_ratio": 0.0,
            "l0_vec_cycles": 200,
            "has_l2_pmu": False,
        }
    )
    _assert_simplex_valid(overloaded_no_pmu)
    no_pmu_map = dict(overloaded_no_pmu)
    assert no_pmu_map[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)
    assert no_pmu_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.5)
    assert no_pmu_map[SymptomFamily.SRAM_REGISTER_SPILL_STACK] == pytest.approx(0.5)

    overloaded_with_pmu = compute_symptom_simplex(
        {
            "roofline_eta": 0.50,
            "transfer_inflation_ratio": 3.0,
            "spill_bytes": 512,
            "l0_vec_cycles": 200,
            "measured_aiv_vec_ratio": 0.50,
            "has_l2_pmu": True,
        }
    )
    _assert_simplex_valid(overloaded_with_pmu)
    with_pmu_map = dict(overloaded_with_pmu)
    assert with_pmu_map[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)
    assert with_pmu_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.5)
    assert with_pmu_map[SymptomFamily.SRAM_REGISTER_SPILL_STACK] == pytest.approx(0.5)
    assert rank_symptom_families(static_obs) == static_simplex


def test_stage1_topology_and_pipeline_hard_gates() -> None:
    """Verify aiv_only, aic_only, and num_stages <= 1 hard-gate inapplicable families."""
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
    assert aiv_obs.compute_engine == "aiv_only"
    aiv_map = dict(compute_symptom_simplex(aiv_obs))
    assert aiv_map[SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE] == 0.0
    assert aiv_map[SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE] == 0.0

    aic_obs = ObservationVector5D.from_tiers(
        build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=None,
            pipe_cycles={"AIC.PIPE_M": 200, "AIV0.PIPE_V": 40},
            waterlines={},
        ),
        telemetry={"aic_mac_ratio": 0.50},
        topology_axes=TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd"),
    )
    assert aic_obs.compute_engine == "aic_only"
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
    assert single_stage_obs.num_stages == 1
    single_stage_map = dict(compute_symptom_simplex(single_stage_obs))
    assert single_stage_map[SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER] == 0.0
    assert single_stage_map[SymptomFamily.SYNC_BARRIER_SERIALIZATION] > 0.0

    restored_obs = ObservationVector5D.from_dict(single_stage_obs.to_dict())
    assert restored_obs == single_stage_obs
    assert restored_obs.compute_engine == "aiv_only"
    assert restored_obs.num_stages == 1
    assert restored_obs.has_l2_pmu is True


def test_tie_break_follows_physical_priority_not_ascii() -> None:
    """Verify tied simplex entries follow canonical physical priority, not ASCII order."""
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
    tied_map = dict(tied)
    assert tied_map[SymptomFamily.SCALAR_LOOP_TRAP] == pytest.approx(0.5)
    assert tied_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.5)
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


def test_escalation_trigger_and_residual_beam_integration() -> None:
    """Verify escalation logic, residual unexplained_gap_ratio, and eq3 objective."""
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
    assert math.isclose(sum(v for _, v in flat_entropy), 1.0, rel_tol=1e-9)
    assert should_escalate_diagnostic_or_creative(flat_entropy, stagnation_rounds=1) is True
    assert should_escalate_diagnostic_or_creative(flat_entropy, stagnation_rounds=0) is False

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100},
        waterlines={},
        gm_roundtrip_bytes=4096,
    )
    residual = compute_static_dynamic_residual(
        ledger,
        {"dur_us": 20.0, "sync_stall_ratio": 0.30},
    )
    assert residual.observation is not None
    expected_unmodeled = dict(compute_symptom_simplex(residual.observation))[
        SymptomFamily.UNMODELED_SILICON_RESIDUAL
    ]
    assert residual.unexplained_gap_ratio == pytest.approx(expected_unmodeled, abs=1e-6)

    clean = compute_eq3_hap_objective(
        5.0,
        ref_perf_us=15.0,
        t_hw_us=2.5,
        spill_bytes=0,
        bank_conflict_ratio=0.0,
        l2_hit_rate=1.0,
    )
    spill_only = compute_eq3_hap_objective(
        5.0,
        ref_perf_us=15.0,
        t_hw_us=2.5,
        spill_bytes=1024,
        bank_conflict_ratio=0.0,
        l2_hit_rate=1.0,
    )
    assert spill_only == pytest.approx(clean)
    bank_penalized = compute_eq3_hap_objective(
        5.0,
        ref_perf_us=15.0,
        t_hw_us=2.5,
        spill_bytes=0,
        bank_conflict_ratio=0.25,
        l2_hit_rate=0.70,
    )
    assert 0.0 < bank_penalized < clean
