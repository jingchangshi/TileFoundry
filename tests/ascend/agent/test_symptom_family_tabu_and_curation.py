"""Unit tests for SymptomFamily taxonomy, anti-preconditions, scoped tabu, and offline curation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec, PerfTargetSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    _should_trigger_p2_deep_dive,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    select_fallback_track_from_semantics,
    select_initial_track_from_semantics,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb import (
    SYMPTOM_FAMILY_TO_AXIS,
    SYMPTOM_FAMILY_TO_FACETS,
    AscendCausalRule,
    BayesianRuleTracker,
    FamilyTabuEntry,
    SymptomFamily,
    TabuEntry,
    append_kb_event_shard,
    append_rule_misapplication,
    append_winning_edge,
    build_predicate_context,
    load_seed_rules,
    rank_symptom_families,
    save_distilled_rule_file,
    save_staging_candidate_rule,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel, StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend
from tilefoundry_ascend.agent.orchestrator.residual_beam import StaticDynamicResidual
from tilefoundry_ascend.ascend.passes import (
    HirBlueprintPacket,
    HirRooflineSummary,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)

_BASELINE_MODULE = """
import tilelang.ascend.language as T

def _select_reg_a_config(tile_m=1):
    return {"tile_m": tile_m}

def _build_reg_a(tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main
"""

_IMPROVED_EXPLORE_SLICE = """
def _build_reg_a(tile_m=4):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(8):
            a_ub = T.alloc_shared((64, 128), "bfloat16")
            T.copy(X[0:64, 0:128], a_ub)
            T.copy(a_ub, Y[0:64, 0:128])
    return main
"""

_FLAT_RULE_SLICE = """
def _build_reg_a(tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main
"""


def _dummy_blueprint(regime_tag: str = "reg_a") -> dict[str, HirBlueprintPacket]:
    roofline = HirRooflineSummary(
        anchor_dims={"M": 128, "D": 128},
        compute_flops=16384,
        min_gm_bytes=65536,
        arithmetic_intensity=0.25,
        compute_floor_us=0.5,
        memory_floor_us=2.0,
        ideal_roofline_us=2.0,
        dominant_bound="memory",
        requested_anchor_dims={"M": 128, "D": 128},
        resolved_hir_dims={"M": 128, "D": 128},
    )
    return {
        regime_tag: HirBlueprintPacket(
            entry_name="reg_a",
            roofline=roofline,
            lowering_diagnostics=(),
            requested_anchor_dims={"M": 128, "D": 128},
            resolved_hir_dims={"M": 128, "D": 128},
        )
    }


def _surrogate(
    cid: str,
    anchors: tuple[AnchorCaseSpec, ...],
    lat_us: float,
    *,
    all_ok: bool = True,
) -> CandidateSurrogateResult:
    metrics = [
        build_anchor_metrics(spec, accuracy_passed=all_ok, cand_us=lat_us) for spec in anchors
    ]
    return aggregate_surrogate_result(cid, "synthetic_op", metrics)


def test_all_seed_rules_have_valid_symptom_family_and_anti_precondition(tmp_path: Path) -> None:
    """Validate all 41 seed rules declare SymptomFamily, mechanism_tag, and anti_precondition."""
    rules = load_seed_rules()
    assert len(rules) == 41
    assert len(SymptomFamily) == 13
    assert set(SYMPTOM_FAMILY_TO_AXIS) == set(SymptomFamily)
    assert set(SYMPTOM_FAMILY_TO_FACETS) == set(SymptomFamily)
    assert len({r.mechanism_tag for r in rules}) == 41

    base_mask = ConstraintRelaxationMask.bottom()
    for fam, facets in SYMPTOM_FAMILY_TO_FACETS.items():
        assert facets, f"{fam} must unlock at least one facet"
        assert all(facet in VALID_FREEDOM_FACETS for facet in facets)
        unlocked = base_mask.unlock(*facets)
        assert all(unlocked.is_unlocked(facet) for facet in facets)

    covered_families: set[SymptomFamily] = set()
    for rule in rules:
        assert rule.symptom_family, f"{rule.rule_id} missing symptom_family"
        assert rule.mechanism_tag, f"{rule.rule_id} missing mechanism_tag"
        assert rule.concept_description, f"{rule.rule_id} missing concept_description"
        fam = SymptomFamily(rule.symptom_family)
        covered_families.add(fam)
        validate_rule_predicate(rule.anti_precondition)
        roundtrip = AscendCausalRule.from_dict(rule.to_dict())
        assert roundtrip.symptom_family == rule.symptom_family
        assert roundtrip.anti_precondition == rule.anti_precondition
        assert roundtrip.mechanism_tag == rule.mechanism_tag
        assert roundtrip.concept_description == rule.concept_description
    assert covered_families == set(SymptomFamily)

    with pytest.raises(ValueError, match="invalid symptom_family"):
        AscendCausalRule(
            rule_id="R_BAD_FAM",
            regime_guard="is_legal",
            precondition="PIPE_V > 0",
            transform_template="Do something",
            expected_postcondition="PIPE_V <= prev.PIPE_V",
            source_provenance="ref:test",
            symptom_family="NON_EXISTENT_FAMILY",
        )

    bad_yaml = tmp_path / "missing_family.yaml"
    bad_yaml.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "rule_id": "R_MISSING_FAM",
                        "regime_guard": "is_legal",
                        "precondition": "PIPE_V > 0",
                        "transform_template": "Do something",
                        "expected_postcondition": "PIPE_V <= prev.PIPE_V",
                        "source_provenance": "ref:test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty symptom_family"):
        load_seed_rules(bad_yaml)


def test_anti_precondition_blocks_matching_with_orthogonal_ratios() -> None:
    """Verify anti_precondition gates confusable rules using orthogonal physical ratios."""
    rules = {r.rule_id: r for r in load_seed_rules()}
    ub_half = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"a": 131072},
        implicit_buffers={},
    )
    ub_tight = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"a": 225280},
        implicit_buffers={},
    )

    burst_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 64},
        waterlines={"UB": ub_half},
        wave_occupancy=0.45,
        dma_burst_efficiency_ratio=0.55,
    )
    ctx_ok = build_predicate_context(burst_ledger)
    assert ctx_ok["transfer_inflation_ratio"] == pytest.approx(1.0)
    assert ctx_ok["effective_compute_eta"] == pytest.approx(0.0)
    assert ctx_ok["ub_occupancy_ratio"] == pytest.approx(0.5)
    assert ctx_ok["prev"]["transfer_inflation_ratio"] == pytest.approx(1.0)
    assert ctx_ok["prev"]["effective_compute_eta"] == pytest.approx(0.0)
    assert ctx_ok["prev"]["ub_occupancy_ratio"] == pytest.approx(0.5)
    assert rules["R18_WAVE_QUANTIZED_BURST_TILING"].matches(ctx_ok) is True

    tight_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 64},
        waterlines={"UB": ub_tight},
        wave_occupancy=0.45,
        dma_burst_efficiency_ratio=0.55,
    )
    ctx_tight = build_predicate_context(tight_ledger)
    assert ctx_tight["ub_occupancy_ratio"] > 0.82
    assert rules["R18_WAVE_QUANTIZED_BURST_TILING"].matches(ctx_tight) is False

    ctx_spill = build_predicate_context(
        burst_ledger,
        residual=StaticDynamicResidual(
            measured_latency_us=30.0,
            predicted_bottleneck_util=0.8,
            measured_aic_mac_ratio=0.0,
            measured_aiv_vec_ratio=0.3,
            measured_mte2_wait_ratio=0.5,
            measured_sync_stall_ratio=0.0,
            unexplained_gap_ratio=0.1,
            spill_bytes=64,
        ),
    )
    assert rules["R18_WAVE_QUANTIZED_BURST_TILING"].matches(ctx_spill) is False

    gather_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_MTE2": 64},
        waterlines={},
        mte2_inflight_per_aiv=24,
        gm_roundtrip_bytes=65536,
    )
    ctx_gather_normal = build_predicate_context(
        gather_ledger,
        extra={"transfer_inflation_ratio": 1.05},
    )
    ctx_gather_inflated = build_predicate_context(
        gather_ledger,
        extra={"transfer_inflation_ratio": 1.80},
    )
    assert rules["R04_MTE2_SPARSE_GATHER_COALESCE"].matches(ctx_gather_normal) is True
    assert rules["R04_MTE2_SPARSE_GATHER_COALESCE"].matches(ctx_gather_inflated) is False

    fission_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 96, "AIV0.PIPE_MTE2": 32},
        waterlines={"UB": ub_half},
        has_coupled_reduce_sfu_accum=True,
    )
    ctx_low_eta = build_predicate_context(
        fission_ledger,
        extra={"effective_compute_eta": 0.20},
    )
    ctx_peak_eta = build_predicate_context(
        fission_ledger,
        extra={"effective_compute_eta": 0.72},
    )
    assert rules["R20_LOOP_FISSION_REDUCE_SFU_HIDING"].matches(ctx_low_eta) is True
    assert rules["R20_LOOP_FISSION_REDUCE_SFU_HIDING"].matches(ctx_peak_eta) is False

    barrier_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64},
        waterlines={},
    )
    ctx_bar_ok = build_predicate_context(barrier_ledger)
    ctx_bar_pseudo = build_predicate_context(
        barrier_ledger,
        residual=StaticDynamicResidual(
            measured_latency_us=25.0,
            predicted_bottleneck_util=0.9,
            measured_aic_mac_ratio=0.3,
            measured_aiv_vec_ratio=0.1,
            measured_mte2_wait_ratio=0.4,
            measured_sync_stall_ratio=0.3,
            unexplained_gap_ratio=0.3,
            pseudo_double_buffer_serialized=True,
        ),
    )
    assert rules["R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION"].matches(ctx_bar_ok) is True
    assert rules["R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION"].matches(ctx_bar_pseudo) is False


def test_rank_symptom_families_and_p2_deep_dive_trigger_policy() -> None:
    """Verify 12-family symptom ranking and P2 deep-dive tie threshold at stagnation >= 2."""
    roundtrip_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_MTE2": 96},
        waterlines={},
        gm_roundtrip_bytes=131072,
    )
    ctx_rt = build_predicate_context(
        roundtrip_ledger,
        extra={"transfer_inflation_ratio": 2.4},
    )
    ranked_rt = rank_symptom_families(ctx_rt)
    assert len(ranked_rt) == 13
    assert ranked_rt[0][0] == SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP
    assert ranked_rt[0][1] > 0.7

    obs_spill = roundtrip_ledger.to_observation_5d(
        telemetry={"dur_us": 20.0, "sync_stall_ratio": 0.45, "bank_conflict_ratio": 0.30},
    )
    ranked_obs = rank_symptom_families(obs_spill)
    assert len(ranked_obs) == 13
    assert ranked_obs[0][0] == SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP

    tied = [
        (SymptomFamily.DMA_L2_THRASH_REUSE_MISS, 0.54),
        (SymptomFamily.SYNC_BARRIER_SERIALIZATION, 0.49),
    ]
    separated = [
        (SymptomFamily.DMA_L2_THRASH_REUSE_MISS, 0.72),
        (SymptomFamily.SYNC_BARRIER_SERIALIZATION, 0.45),
    ]
    assert _should_trigger_p2_deep_dive(0, tied) is False
    assert _should_trigger_p2_deep_dive(1, tied) is False
    assert _should_trigger_p2_deep_dive(2, tied) is True
    assert _should_trigger_p2_deep_dive(3, tied) is True
    assert _should_trigger_p2_deep_dive(2, separated) is False
    assert _should_trigger_p2_deep_dive(2, [(SymptomFamily.DMA_L2_THRASH_REUSE_MISS, 0.5)]) is False


def test_online_rulebook_frozen_by_default_and_opt_in_distilled_sync(tmp_path: Path) -> None:
    """Verify sync_from_experience_lake ignores unreviewed distilled rules unless opted in."""
    lake_dir = tmp_path / "lake"
    unreviewed = AscendCausalRule(
        rule_id="R_UNREVIEWED_DISTILLED_01",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Speculative loop rewrite.",
        expected_postcondition="PIPE_V < prev.PIPE_V",
        source_provenance="flywheel:ablation",
        scope="cross_op",
    )
    save_distilled_rule_file(lake_dir, unreviewed)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    append_kb_event_shard(
        lake_dir,
        run_id="peer_run",
        rule_id="R01_DUAL_COPY_1C2V_SPLIT",
        postcondition_realized=True,
        speed_gain=0.22,
        topology_axes=axes,
        operator_name="softmax",
        round_idx=1,
    )
    append_kb_event_shard(
        lake_dir,
        run_id="peer_run",
        rule_id="R_UNREVIEWED_DISTILLED_01",
        postcondition_realized=True,
        speed_gain=0.35,
        topology_axes=axes,
        operator_name="softmax",
        round_idx=1,
    )

    frozen_tracker = BayesianRuleTracker.from_experience_lake(lake_dir)
    assert len(frozen_tracker.rules) == 41
    assert "R_UNREVIEWED_DISTILLED_01" not in frozen_tracker.rules
    assert frozen_tracker.posteriors["R01_DUAL_COPY_1C2V_SPLIT"].successes == 1

    opt_in_tracker = BayesianRuleTracker.from_seed_yaml()
    opt_in_tracker.sync_from_experience_lake(lake_dir, allow_unreviewed_distilled_rules=True)
    assert "R_UNREVIEWED_DISTILLED_01" in opt_in_tracker.rules
    assert opt_in_tracker.posteriors["R_UNREVIEWED_DISTILLED_01"].successes == 1


def test_scoped_tabu_rule_cooldown_family_escalation_parent_clear_and_roundtrip(
    tmp_path: Path,
) -> None:
    """Verify scoped rule/family cooldowns, parent-change reset, and checkpoint round-trip."""
    tracker = BayesianRuleTracker.from_seed_yaml()
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE2": 64},
        waterlines={},
        mte2_inflight_per_aiv=24,
        unaligned_dma_padding_bytes=16,
        wave_occupancy=0.45,
        dma_burst_efficiency_ratio=0.55,
    )
    ctx = build_predicate_context(ledger)
    base_matches = {r.rule_id for r in tracker.rank_applicable_rules(ctx)}
    assert "R04_MTE2_SPARSE_GATHER_COALESCE" in base_matches
    assert "R16_PADDED_PITCH_TAIL_MASKING" in base_matches
    assert "R18_WAVE_QUANTIZED_BURST_TILING" in base_matches

    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_parent_1",
        current_round=1,
        reason="l1_compile_failed",
    )
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_parent_1",
        current_round=2,
    ) is True
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_parent_1",
        current_round=3,
    ) is True
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_parent_1",
        current_round=4,
    ) is False
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R16_PADDED_PITCH_TAIL_MASKING",
        parent_sha="sha_parent_1",
        current_round=2,
    ) is False

    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R16_PADDED_PITCH_TAIL_MASKING",
        parent_sha="sha_parent_1",
        current_round=2,
        reason="l2_no_speedup",
    )
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        parent_sha="sha_parent_1",
        current_round=3,
    ) is True
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        parent_sha="sha_parent_1",
        current_round=4,
    ) is False

    filtered_r3 = {
        r.rule_id
        for r in tracker.select_top_rules(
            ctx,
            regime_tag="reg_a",
            parent_sha="sha_parent_1",
            current_round=3,
        )
    }
    assert "R04_MTE2_SPARSE_GATHER_COALESCE" not in filtered_r3
    assert "R16_PADDED_PITCH_TAIL_MASKING" not in filtered_r3
    assert "R18_WAVE_QUANTIZED_BURST_TILING" not in filtered_r3

    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_parent_2",
        current_round=2,
    ) is False
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        parent_sha="sha_parent_2",
        current_round=3,
    ) is False
    unblocked_new_parent = {
        r.rule_id
        for r in tracker.rank_matching_rules(
            ctx,
            regime_tag="reg_a",
            parent_sha="sha_parent_2",
            current_round=3,
        )
    }
    assert "R04_MTE2_SPARSE_GATHER_COALESCE" in unblocked_new_parent
    assert "R18_WAVE_QUANTIZED_BURST_TILING" in unblocked_new_parent

    payload = tracker.to_dict()
    assert len(tracker.tabu_entries) == 2
    assert isinstance(tracker.tabu_entries[0], TabuEntry)
    assert len(tracker.family_tabu_entries) == 1
    assert isinstance(tracker.family_tabu_entries[0], FamilyTabuEntry)

    restored = BayesianRuleTracker.from_dict(payload)
    assert restored.tabu_entries == tracker.tabu_entries
    assert restored.family_tabu_entries == tracker.family_tabu_entries
    assert restored.family_failure_streaks == tracker.family_failure_streaks
    assert restored.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        parent_sha="sha_parent_1",
        current_round=3,
    ) is True

    json_file = tmp_path / "tracker_tabu.json"
    tracker.save_json(json_file)
    from_disk = BayesianRuleTracker.from_seed_yaml()
    from_disk.load_posteriors_json(json_file)
    assert from_disk.tabu_entries == tracker.tabu_entries
    assert from_disk.family_tabu_entries == tracker.family_tabu_entries


def test_flywheel_stages_offline_evidence_without_mutating_online_rulebook(
    tmp_path: Path,
) -> None:
    """Verify frozen online rulebook, offline staging/evidence writes, tabu, and P1 gating."""
    lake_dir = tmp_path / "experience_lake"
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASELINE_MODULE, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt.json"

    rule = AscendCausalRule(
        rule_id="R_LOCAL_WAVE",
        symptom_family=SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL.value,
        regime_guard="PIPE_V > 0",
        precondition="wave_occupancy < 0.5",
        transform_template="Increase grid core blocks cleanly.",
        expected_postcondition="wave_occupancy > prev.wave_occupancy",
        source_provenance="ref:test",
        alpha_prior=10.0,
        beta_prior=1.0,
    )
    tracker = BayesianRuleTracker.from_rules([rule])
    initial_rule_ids = set(tracker.rules)

    anchors = (AnchorCaseSpec("level1/synthetic_op", 1, "reg_a", 20.0, 2.0),)
    regimes = (
        RegimeTargetSpec(
            regime_tag="reg_a",
            builder_candidates=("_build_reg_a",),
            sample_builder_kwargs={"tile_m": 1},
            description="Synthetic test regime",
            representative_hir_dims={"M": 128, "D": 128},
            representative_anchor_dims={"M": 128, "D": 128},
            config_selector_fn="_select_reg_a_config",
            editable_symbols=("_select_reg_a_config", "_build_reg_a"),
            allowed_helper_prefix="_reg_a_",
        ),
    )

    creative_reply = json.dumps(
        {
            "physical_hypothesis": "Scale kernel blocks to 8 to lift wave occupancy",
            "target_local_metric": "wave_occupancy",
            "dsl_source": _IMPROVED_EXPLORE_SLICE,
        }
    )
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_FLAT_RULE_SLICE],
            "explore_creative": [creative_reply],
        }
    )

    def _l0_check(mod_src: str, _reg: RegimeTargetSpec) -> StageAResult:
        is_fast = "T.Kernel(8)" in mod_src
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=40 if is_fast else 120,
            pipe_cycles={"AIV0.PIPE_V": 40 if is_fast else 120},
            waterlines={},
            grid_blocks=8 if is_fast else 1,
            wave_occupancy=0.80 if is_fast else 0.05,
        )
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    def _l2_eval(cands, aa):
        out = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            lat = 8.0 if "T.Kernel(8)" in src else 21.0
            out[cid] = _surrogate(cid, tuple(aa), lat)
        return out

    msprof_calls: list[str] = []

    def _msprof_collect(_op, cand_path, _anc_seq):
        src = Path(cand_path).read_text(encoding="utf-8")
        msprof_calls.append("fast" if "T.Kernel(8)" in src else "base")
        return {
            "reg_a": {
                "dur_us": 8.0 if "T.Kernel(8)" in src else 20.0,
                "aiv_vec_ratio": 0.45,
                "mte2_wait_ratio": 0.20,
            }
        }

    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        probe_runner=lambda _spec: {
            "baseline_dur_us": 40.0,
            "optimized_dur_us": 18.0,
            "numerical_ok": 1.0,
        },
        min_edit_gain=0.05,
        min_probe_speedup=1.08,
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic_op",
            max_rounds=1,
            k_rule_workers=1,
            min_surrogate_gain=0.25,
            min_flywheel_gain=0.03,
            allow_unreviewed_distilled_rules=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_op", 5.0, 4.0, 0.99),
        anchors=anchors,
        regime_specs=regimes,
        regime_blueprints=_dummy_blueprint("reg_a"),
        llm_backend=backend,
        tracker=tracker,
        flywheel_distiller=flywheel,
        l0_checker=_l0_check,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=_msprof_collect,
        checkpoint_path=ckpt_path,
        experience_lake_dir=lake_dir,
    )
    report = asyncio.run(runner.run())
    assert report.total_rounds == 1
    assert report.promoted_rounds == (1,)
    assert set(tracker.rules) == initial_rule_ids
    assert not (lake_dir / "kb" / "distilled_rules").exists()

    staging_files = sorted((lake_dir / "kb" / "staging_candidates").glob("*.json"))
    assert staging_files
    staged_payload = json.loads(staging_files[0].read_text(encoding="utf-8"))
    assert staged_payload["status"] == "pending_review"
    assert "ablation_report" in staged_payload

    winning_file = lake_dir / "kb" / "evidence" / "winning_edges.jsonl"
    assert winning_file.is_file()
    winning_rows = [json.loads(line) for line in winning_file.read_text(encoding="utf-8").splitlines()]
    assert len(winning_rows) == 1
    assert winning_rows[0]["worker_kind"] == "creative_explore"
    assert winning_rows[0]["relative_gain"] >= 0.05

    misapp_file = lake_dir / "kb" / "evidence" / "rule_misapplications.jsonl"
    assert misapp_file.is_file()
    misapp_rows = [json.loads(line) for line in misapp_file.read_text(encoding="utf-8").splitlines()]
    assert len(misapp_rows) == 1
    assert misapp_rows[0]["rule_id"] == "R_LOCAL_WAVE"
    assert misapp_rows[0]["symptom_family"] == SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL.value
    assert misapp_rows[0]["parent_observation"] is not None

    assert len(tracker.tabu_entries) == 1
    assert tracker.tabu_entries[0].rule_id == "R_LOCAL_WAVE"
    assert tracker.tabu_entries[0].expire_round == 3
    ckpt_payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert ckpt_payload["tracker_state"]["tabu_entries"]

    assert msprof_calls == ["base", "fast"]


def test_curate_kb_from_lake_cli_markdown_and_json(tmp_path: Path) -> None:
    """Smoke-test scripts/curate_kb_from_lake.py for both markdown and json outputs."""
    lake_dir = tmp_path / "lake"
    candidate = AscendCausalRule(
        rule_id="R_STAGED_BURST_01",
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
        regime_guard="is_legal",
        precondition="dma_burst_efficiency_ratio < 0.75",
        anti_precondition="ub_occupancy_ratio > 0.82",
        transform_template="Coalesce vector row bursts to 32 KiB.",
        expected_postcondition="dma_burst_efficiency_ratio > prev.dma_burst_efficiency_ratio",
        source_provenance="flywheel:ablation",
        scope="operator_local",
        origin_operator="level2/softmax",
    )
    save_staging_candidate_rule(
        lake_dir,
        candidate,
        ablation_report={
            "baseline_latency_us": 40.0,
            "winner_latency_us": 28.0,
            "kept_positive_chunks": [{"chunk": {"edit_kind": "coalesced_copy"}}],
        },
    )
    append_winning_edge(
        lake_dir,
        {
            "operator": "level2/softmax",
            "regime_tag": "vector",
            "worker_kind": "creative_explore",
            "parent_sha": "aaaa1111bbbb2222",
            "winner_sha": "cccc3333dddd4444",
            "baseline_latency_us": 40.0,
            "winner_latency_us": 28.0,
            "relative_gain": 0.30,
            "loo_chunks": [{"chunk": {"edit_kind": "coalesced_copy"}}],
            "staged_rule_ids": ["R_STAGED_BURST_01"],
        },
    )
    append_rule_misapplication(
        lake_dir,
        {
            "operator": "level2/softmax",
            "regime_tag": "vector",
            "rule_id": "R18_WAVE_QUANTIZED_BURST_TILING",
            "symptom_family": SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
            "parent_latency_us": 20.0,
            "cand_latency_us": 22.0,
            "latency_ratio": 1.10,
            "transfer_inflation_ratio": 1.85,
            "effective_compute_eta": 0.15,
            "ub_occupancy_ratio": 0.88,
            "reason": "latency_regression_gt_3pct",
            "parent_observation": {
                "residuals": {
                    "compute_issue_gap": 0.1,
                    "offchip_dma_gap": 0.7,
                    "onchip_sram_reg_gap": 0.2,
                    "pipeline_sync_gap": 0.0,
                    "grid_wave_gap": 0.1,
                }
            },
        },
    )

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "curate_kb_from_lake.py"
    json_proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--lake-dir",
            str(lake_dir),
            "--format",
            "json",
            "--top-k",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(json_proc.stdout)
    assert parsed["summary"]["staging_candidates_count"] == 1
    assert parsed["summary"]["winning_edges_count"] == 1
    assert parsed["summary"]["rule_misapplications_count"] == 1
    assert parsed["staging_candidates"][0]["rule_id"] == "R_STAGED_BURST_01"
    assert parsed["top_misapplied_rules"][0]["rule_id"] == "R18_WAVE_QUANTIZED_BURST_TILING"
    assert parsed["top_misapplied_rules"][0]["dominant_axes"] == {"offchip_dma": 1}

    md_proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--lake-dir",
            str(lake_dir),
            "--format",
            "markdown",
            "--top-k",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "# Ascend KB Offline Curation Report" in md_proc.stdout
    assert "R_STAGED_BURST_01" in md_proc.stdout
    assert "R18_WAVE_QUANTIZED_BURST_TILING" in md_proc.stdout
    assert "## Top Winning Edges" in md_proc.stdout


def test_stage0_track_selection_uses_blueprint_roofline_and_schedule_delta() -> None:
    """Verify blueprint Roofline physics routes coupled 1C2V and unaligned vector reductions."""
    analyzer = PyTorchSemanticAnalyzer()

    coupled_src = """
import torch

def fused_attn(q, k, v):
    s = torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)
"""
    coupled_sem = analyzer.analyze_source(coupled_src, entry_name="fused_attn")
    assert coupled_sem.base_axes.compute_engine == "coupled_1c2v"
    aligned_ratio = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=4.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.4,
        working_set_vreg_ratio=0.5,
        wave_occupancy_ratio=1.0,
        oi_ridge_ratio=0.8,
        inner_contiguous_bytes=128,
    )
    coupled_bp = HirBlueprintPacket(
        entry_name="fused_attn",
        roofline=HirRooflineSummary(
            anchor_dims={"B": 1, "S": 128, "D": 64},
            compute_flops=2_097_152,
            min_gm_bytes=65536,
            arithmetic_intensity=32.0,
            compute_floor_us=4.0,
            memory_floor_us=3.5,
            ideal_roofline_us=4.0,
            dominant_bound="compute",
            cube_flops=1_572_864,
            vector_flops=524_288,
            cube_floor_us=4.0,
            vec_floor_us=2.0,
        ),
        lowering_diagnostics=(),
    )
    assert (
        select_initial_track_from_semantics(
            coupled_sem,
            (aligned_ratio,),
            blueprints={"coupled": coupled_bp},
        )
        == "tilelang_dsl"
    )
    assert (
        select_fallback_track_from_semantics(
            coupled_sem,
            (aligned_ratio,),
            blueprints={"coupled": coupled_bp},
        )
        == "hybrid_trackb"
    )

    reduce_src = """
import torch

def row_sum(x):
    return torch.sum(x, dim=-1)
"""
    reduce_sem = analyzer.analyze_source(reduce_src, entry_name="row_sum")
    unaligned_ratio = CasePhysicsRatios(
        case_id=2,
        stride_ratio=1.0,
        contig_atom_ratio=0.75,
        is_atom_aligned=False,
        working_set_sram_ratio=0.3,
        working_set_vreg_ratio=0.4,
        wave_occupancy_ratio=1.0,
        oi_ridge_ratio=0.1,
        inner_contiguous_bytes=202,
    )
    reduce_bp = HirBlueprintPacket(
        entry_name="row_sum",
        roofline=HirRooflineSummary(
            anchor_dims={"M": 1024, "D": 101},
            compute_flops=103_424,
            min_gm_bytes=413_696,
            arithmetic_intensity=0.25,
            compute_floor_us=0.5,
            memory_floor_us=2.0,
            ideal_roofline_us=2.0,
            dominant_bound="memory",
            vector_flops=103_424,
            vec_floor_us=0.5,
        ),
        lowering_diagnostics=(),
    )
    assert (
        select_initial_track_from_semantics(
            reduce_sem,
            (unaligned_ratio,),
            blueprints={"vector": reduce_bp},
        )
        == "tilelang_dsl"
    )
    assert (
        select_fallback_track_from_semantics(
            reduce_sem,
            (unaligned_ratio,),
            blueprints={"vector": reduce_bp},
        )
        == "cce_trackb"
    )
