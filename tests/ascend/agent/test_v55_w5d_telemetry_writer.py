"""Unit tests for RoundReceipt and TelemetryLedgerWriter."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignRoundRecord,
    PendingL4,
)
from tilefoundry_ascend.agent.campaign.fault_types import FaultPlane, L3Resolution, TypedFault
from tilefoundry_ascend.agent.campaign.promotion_coordinator import PendingPromotion
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L4CannbenchResult,
)
from tilefoundry_ascend.agent.campaign.telemetry_ledger_writer import (
    RoundReceipt,
    TelemetryLedgerWriter,
    source_sha256,
)
from tilefoundry_ascend.agent.cost_advisor import create_cost_advisor
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.passes import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes


def _make_ledger(*, gm_roundtrip_bytes: int = 32768, is_legal: bool = True) -> object:
    waterlines = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"a": 65536},
            implicit_buffers={},
        )
    }
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=is_legal,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120},
        waterlines=waterlines,
        gm_roundtrip_bytes=gm_roundtrip_bytes,
        grid_blocks=28,
    )


def _metric(
    case_num: int,
    regime_tag: str,
    cand_us: float,
    *,
    ref_us: float = 40.0,
    t_hw_us: float = 4.0,
    accuracy_passed: bool = True,
    burst_batch_cv: float = 0.02,
) -> AnchorEvalMetrics:
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=accuracy_passed,
        cand_us=cand_us,
        ref_perf_us=ref_us,
        t_hw_us=t_hw_us,
        speedup=(ref_us / cand_us) if cand_us > 0.0 else 0.0,
        roofline_eta=(t_hw_us / cand_us) if cand_us > 0.0 else 0.0,
        eq3_score=(ref_us / cand_us) if cand_us > 0.0 else 0.0,
        burst_batch_cv=burst_batch_cv,
    )


def _surrogate_result(
    candidate_id: str,
    anchors: tuple[AnchorEvalMetrics, ...],
    *,
    operator_name: str = "synthetic_op",
) -> CandidateSurrogateResult:
    valid = [a for a in anchors if a.cand_us > 0.0]
    avg_sp = sum(a.speedup for a in valid) / max(1, len(valid))
    min_sp = min((a.speedup for a in valid), default=0.0)
    avg_eta = sum(a.roofline_eta for a in valid) / max(1, len(valid))
    all_ok = all(a.accuracy_passed and a.cand_us > 0.0 for a in anchors)
    return CandidateSurrogateResult(
        candidate_id=candidate_id,
        operator_name=operator_name,
        all_passed=all_ok,
        avg_speedup=avg_sp,
        min_case_speedup=min_sp,
        avg_roofline_eta=avg_eta,
        surrogate_score=avg_sp,
        anchors=anchors,
    )


def test_round_receipt_frozen_and_serializes() -> None:
    """RoundReceipt is immutable and serializes round timings, prewarm stats, and count."""
    receipt = RoundReceipt(
        round_idx=3,
        regime_tag="main",
        promoted=True,
        l3_resolution=L3Resolution.PROMOTED.value,
        stagnation_after=0,
        admitted_rule_ids=("R_FLYWHEEL_001",),
        stage_timings_sec={"l0_l1_sec": 0.1234567},
        prewarm_stats={"total_tasks": 4, "cache_hits": 2},
        cost_advisor_predictions=5,
        emitted_event_count=3,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(receipt, "promoted", False)
    payload = receipt.to_dict()
    assert payload["round_idx"] == 3
    assert payload["regime_tag"] == "main"
    assert payload["promoted"] is True
    assert payload["admitted_rule_ids"] == ["R_FLYWHEEL_001"]
    assert payload["stage_timings_sec"] == {"l0_l1_sec": 0.123457}
    assert payload["prewarm_stats"] == {"total_tasks": 4, "cache_hits": 2}
    assert payload["cost_advisor_predictions"] == 5
    assert payload["emitted_event_count"] == 3


def test_emit_round_funnel_events_fault_planes_and_occupancy(tmp_path: Path) -> None:
    """Typed stage, compiler, numeric, and occupancy faults emit exact fault planes."""
    ckpt_path = tmp_path / "ckpt.json"
    axes = TopologyAxes("coupled_1c2v", "gm_ub_stream", "affine_simd")
    writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        run_id="run_funnel_01",
        checkpoint_path=ckpt_path,
        topology_axes=axes,
    )
    regime = RegimeTargetSpec("main", ("_build_main",), {})

    chk_occ = L0L1CandidateCheck(
        candidate_id="cand_occ",
        passed=True,
        failed_tier=None,
        patched_module_source="def _build_main():\n    return 1\n",
        snippet_source="def _build_main():\n    return 1\n",
    )
    chk_static = L0L1CandidateCheck(
        candidate_id="cand_static",
        passed=False,
        failed_tier="stage_a",
        patched_module_source="def _build_main():\n    return 2\n",
        snippet_source="def _build_main():\n    return 2\n",
        stage_a=StageAResult(
            passed=False,
            prim_func=None,
            ledger=None,
            fault_plane="STAGE_A_STATIC",
            fault_code="UB_CONTRACT_ERROR",
            failed_tier="stage_a",
        ),
        error_message="UB contract violation",
    )
    chk_prescreen = L0L1CandidateCheck(
        candidate_id="cand_prescreen",
        passed=False,
        failed_tier="cost_prescreen",
        patched_module_source="def _build_main():\n    return 3\n",
        snippet_source="def _build_main():\n    return 3\n",
        error_message="pruned by cost model",
    )
    chk_compile = L0L1CandidateCheck(
        candidate_id="cand_compile",
        passed=False,
        failed_tier="stage_b",
        patched_module_source="def _build_main():\n    return 4\n",
        snippet_source="def _build_main():\n    return 4\n",
        stage_b=StageBResult(
            passed=False,
            error_message="bisheng stack spill",
            elapsed_sec=0.45,
            spill_bytes=512,
            compiler_summary=BishengResourceSummary(spill_bytes=512, stack_bytes=2048),
            fault_plane="STAGE_B_COMPILE",
            fault_code="VF_STACK_OVERFLOW",
            failed_tier="stage_b",
        ),
    )

    occupancy_fault = TypedFault(
        plane=FaultPlane.INFRA_OCCUPANCY,
        code="L2_INFRA_OCCUPANCY",
        failed_tier="l2_infra_occupancy",
        detail="device occupied",
        retriable=True,
    )
    receipt = writer.emit_round_funnel_events(
        round_idx=1,
        regime_spec=regime,
        l0_l1_checks=[chk_occ, chk_static, chk_prescreen, chk_compile],
        failed_entries=[
            ("cand_occ", "l2_infra_occupancy", "device occupied"),
            ("cand_extra_occ", "l2_infra_occupancy", "card busy"),
            ("cand_extra_l0", "stage_a", "invalid tile"),
            ("cand_extra_l1", "stage_b", "bisheng failed"),
            ("cand_extra_noop", "noop_identical", "unchanged"),
        ],
        l2_infra_fault=occupancy_fault,
        stage_timings_sec={"l0_l1_sec": 0.25},
        prewarm_stats={"total_tasks": 2, "cache_hits": 1},
        cost_advisor_predictions=4,
    )

    assert receipt.emitted_event_count == 8
    assert receipt.round_idx == 1
    assert receipt.regime_tag == "main"
    assert receipt.cost_advisor_predictions == 4

    jsonl_path = tmp_path / "telemetry" / "round_funnel_events.jsonl"
    assert jsonl_path.is_file()
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 8
    by_cid = {row["candidate_id"]: row for row in rows}
    assert by_cid["cand_occ"]["fault_plane"] == FaultPlane.INFRA_OCCUPANCY.value
    assert by_cid["cand_extra_occ"]["fault_plane"] == FaultPlane.INFRA_OCCUPANCY.value
    assert by_cid["cand_static"]["fault_plane"] == FaultPlane.STATIC_CONTRACT.value
    assert by_cid["cand_prescreen"]["fault_plane"] == FaultPlane.STATIC_CONTRACT.value
    assert by_cid["cand_extra_l0"]["fault_plane"] == FaultPlane.STATIC_CONTRACT.value
    assert by_cid["cand_compile"]["fault_plane"] == FaultPlane.COMPILER_LOWERING.value
    assert by_cid["cand_compile"]["spill_bytes"] == 512
    assert by_cid["cand_compile"]["stack_bytes"] == 2048
    assert by_cid["cand_extra_l1"]["fault_plane"] == FaultPlane.COMPILER_LOWERING.value
    assert by_cid["cand_extra_noop"]["fault_plane"] == FaultPlane.NOOP_IDENTICAL.value
    assert all(row["fault_plane"] != FaultPlane.UNMODELED_FAULT.value for row in rows)


def test_save_and_restore_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Checkpoint save/restore preserves all state, diagnostics, pending records, and source."""
    target_path = tmp_path / "kernel.py"
    incumbent_src = "def _build_main(tile=16):\n    return tile\n"
    cand_src = "def _build_main(tile=32):\n    return tile\n"
    target_path.write_text(incumbent_src, encoding="utf-8")
    ckpt_path = tmp_path / "checkpoint.json"

    tracker = BayesianRuleTracker.from_seed_yaml()
    first_rule_id = next(iter(tracker.posteriors))
    tracker.record_outcome(first_rule_id, postcondition_realized=True, speed_gain=0.15)

    beam = TwoLaneBeamSearch(exploit_width=2, stepping_stone_width=2)
    writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        run_id="run_ckpt_01",
        target_module_path=target_path,
        checkpoint_path=ckpt_path,
        tracker=tracker,
        beams={"main": beam},
        regime_specs=(RegimeTargetSpec("main", ("_build_main",), {}),),
        scoreboard_anchors=(AnchorCaseSpec("op/synthetic", 1, "main", 40.0, 4.0),),
        guardrail_cases=(
            GuardrailCaseSpec(
                task="op/synthetic",
                case_num=2,
                regime_tag="main",
                incumbent_ref_us=18.0,
            ),
        ),
        calibrated_dispatch_floor_us=1.25,
    )
    diag_entry = writer.ensure_regime_diag_entry("main")
    diag_entry["l0_reject_reasons"] = {"UB_OVERFLOW": 2}
    diag_entry["l1_compiler_penalty_rejects"] = 1
    writer.msprof_cache[(source_sha256(incumbent_src), "main")] = {"aic_mac_ratio": 0.42}
    writer.boundary_ledger.record_compile_wall(
        regime_tag="main",
        cand_sha="abc123",
        knob_signature="tile=64",
        error_code_norm="UB_SRAM_OVERFLOW",
        ub_bytes=300000,
    )
    writer.last_round_promoted_by_regime["main"] = True
    writer.compile_wall_streak_by_regime["main"] = 2

    baseline_res = _surrogate_result("baseline", (_metric(1, "main", 20.0),))
    best_res = _surrogate_result("cand_1", (_metric(1, "main", 10.0),))
    round_rec = CampaignRoundRecord(
        round_idx=1,
        regime_tag="main",
        builder_name="_build_main",
        parent_lane="exploit",
        creative_effort="medium",
        l0_l1_passed_ids=("cand_1",),
        l0_l1_failed=(("cand_0", "stage_b", "spill"),),
        best_round_speedup=4.0,
        best_so_far_speedup=4.0,
        promoted=True,
        l3_resolution=L3Resolution.PROMOTED.value,
        stagnation_rounds_after=0,
        admitted_rule_ids=("R_FLYWHEEL_001",),
        local_rel_gain=0.5,
        stage_timings_sec={"l0_l1_sec": 0.1},
        prewarm_stats={"total_tasks": 1, "cache_hits": 1},
        cost_advisor_predictions=2,
    )
    pending_promo = PendingPromotion(
        candidate_source=cand_src,
        candidate_sha256=source_sha256(cand_src),
        target_path=str(target_path),
        round_idx=2,
        regime_tag="main",
        candidate_result=best_res,
        candidate_id="cand_2",
        execution_id="exec_p",
        execution_terminal=False,
    )
    pending_l4 = PendingL4(
        target_path=str(target_path),
        candidate_relative="kernel.py",
        candidate_sha256=source_sha256(incumbent_src),
        execution_id="exec_l4",
        execution_terminal=True,
        prior_stop_reason="goal_met",
    )

    writer.save_checkpoint(
        incumbent_source=incumbent_src,
        baseline_result=baseline_res,
        best_result=best_res,
        active_anchors=writer.scoreboard_anchors,
        frozen_regimes={"tail"},
        current_regime="main",
        stagnation_rounds=1,
        rounds=[round_rec],
        promoted_rounds=[1],
        rolled_back_rounds=[],
        escalated_rounds=[2],
        admitted_rule_ids=["R_FLYWHEEL_001"],
        stop_reason="in_progress",
        l4_result=None,
        stagnation_by_regime={"main": 1},
        last_unpromoted_regime="tail",
        pending_promotion=pending_promo,
        pending_l4=pending_l4,
    )
    writer.persist_runtime_state("custom_flag", {"ready": True})

    restored_tracker = BayesianRuleTracker.from_seed_yaml()
    restored_writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        target_module_path=target_path,
        checkpoint_path=ckpt_path,
        tracker=restored_tracker,
        beams={"main": TwoLaneBeamSearch(exploit_width=2, stepping_stone_width=2)},
        regime_specs=(RegimeTargetSpec("main", ("_build_main",), {}),),
    )
    (
        r_inc,
        r_base,
        r_best,
        r_anchors,
        r_frozen,
        r_cur_reg,
        r_stag,
        r_rounds,
        r_prom,
        r_rb,
        r_esc,
        r_admitted,
        r_stag_map,
        r_last_unprom,
        r_pending_promo,
        r_pending_l4,
    ) = restored_writer.restore_from_checkpoint()

    assert restored_writer.run_id == "run_ckpt_01"
    assert r_inc == incumbent_src
    assert r_base.avg_speedup == pytest.approx(baseline_res.avg_speedup)
    assert r_best.avg_speedup == pytest.approx(best_res.avg_speedup)
    assert len(r_anchors) == 1
    assert r_frozen == {"tail"}
    assert r_cur_reg == "main"
    assert r_stag == 1
    assert len(r_rounds) == 1
    assert r_rounds[0].round_idx == 1
    assert r_prom == [1]
    assert r_rb == []
    assert r_esc == [2]
    assert r_admitted == ["R_FLYWHEEL_001"]
    assert r_stag_map == {"main": 1}
    assert r_last_unprom == "tail"
    assert r_pending_promo is not None
    assert r_pending_promo.candidate_id == "cand_2"
    assert r_pending_l4 is not None
    assert r_pending_l4.execution_id == "exec_l4"
    assert target_path.read_text(encoding="utf-8") == cand_src
    assert restored_writer.msprof_cache[(source_sha256(incumbent_src), "main")] == {
        "aic_mac_ratio": 0.42
    }
    assert restored_writer.boundary_ledger.min_overflow_ub_bytes == {"main": 300000}
    assert restored_writer.regime_diag_stats["main"]["l1_compiler_penalty_rejects"] == 1
    assert restored_writer.compile_wall_streak_by_regime == {"main": 2}
    assert restored_tracker.posteriors[first_rule_id].trials == tracker.posteriors[first_rule_id].trials

    writer.persist_round_candidate_artifact(
        L0L1CandidateCheck(
            candidate_id="r1_c1",
            passed=False,
            failed_tier="stage_b",
            patched_module_source=cand_src,
            snippet_source=cand_src,
            error_message="compile error",
            stage_b=StageBResult(passed=False, error_message="err", full_diagnostic="full trace"),
        ),
        round_idx=1,
    )
    assert (tmp_path / "candidates" / "r1_c1.py").read_text(encoding="utf-8") == cand_src
    assert "full trace" in (tmp_path / "candidates" / "r1_c1.diag.txt").read_text(encoding="utf-8")


def test_build_stagnation_diagnosis_and_cause_classification() -> None:
    """Stagnation diagnosis aggregates per-regime stats and classifies primary causes."""
    writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        perf_target=PerfTargetSpec("synthetic_op", 2.0, 1.5, 0.85),
        regime_specs=(
            RegimeTargetSpec("compile_stuck", ("_build_a",), {}),
            RegimeTargetSpec("starved", ("_build_b",), {}),
        ),
    )
    e1 = writer.ensure_regime_diag_entry("compile_stuck")
    e1["l0_reject_reasons"] = {"UB_SRAM_OVERFLOW": 3}
    e1["l1_compiler_penalty_rejects"] = 2
    e1["top_symptom_families"] = [{"symptom_family": "sram_pressure", "severity": 0.9}]

    e2 = writer.ensure_regime_diag_entry("starved")
    e2["rule_starvation_rounds"] = 4

    serialized = writer.serialize_diag_stats()
    writer_restored = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        perf_target=PerfTargetSpec("synthetic_op", 2.0, 1.5, 0.85),
        regime_specs=(
            RegimeTargetSpec("compile_stuck", ("_build_a",), {}),
            RegimeTargetSpec("starved", ("_build_b",), {}),
        ),
    )
    writer_restored.restore_diag_stats(serialized)

    best_res = _surrogate_result(
        "best",
        (
            _metric(1, "compile_stuck", 10.0, t_hw_us=4.0),
            _metric(2, "starved", 10.0, t_hw_us=4.0),
        ),
    )
    diag = writer_restored.build_stagnation_diagnosis(
        best_result=best_res,
        stop_reason="all_regimes_converged",
    )
    assert diag["primary_stagnation_cause"] == "l0_l1_compile_wall"
    regimes = diag["regimes"]
    assert isinstance(regimes, dict)
    assert regimes["compile_stuck"]["primary_stagnation_cause"] == "l0_l1_compile_wall"
    assert regimes["starved"]["primary_stagnation_cause"] == "rule_starvation"

    assert (
        writer.classify_primary_stagnation_cause(
            rule_starvation_rounds=0,
            cost_prescreen_filtered_count=0,
            l0_rejects=0,
            l1_rejects=0,
            l2_no_gain_rounds=0,
            l3_rollback_count=0,
            final_roofline_eta=0.90,
            target_roofline_eta=0.85,
            stop_reason="roofline_saturated",
        )
        == "roofline_saturated"
    )
    assert (
        writer.classify_primary_stagnation_cause(
            rule_starvation_rounds=0,
            cost_prescreen_filtered_count=0,
            l0_rejects=0,
            l1_rejects=0,
            l2_no_gain_rounds=2,
            l3_rollback_count=1,
            final_roofline_eta=0.40,
            target_roofline_eta=0.85,
        )
        == "l3_accuracy_instability"
    )
    assert (
        writer.classify_primary_stagnation_cause(
            rule_starvation_rounds=0,
            cost_prescreen_filtered_count=3,
            l0_rejects=1,
            l1_rejects=0,
            l2_no_gain_rounds=1,
            l3_rollback_count=0,
            final_roofline_eta=0.40,
            target_roofline_eta=0.85,
        )
        == "cost_prescreen_mismatch"
    )


def test_record_kb_event_and_observations_and_rule_eval(tmp_path: Path) -> None:
    """KB event shards, surrogate/L4 observations, rule evaluation, and Track-B evidence work."""
    lake_dir = tmp_path / "lake"
    ckpt_path = tmp_path / "ckpt.json"
    advisor = create_cost_advisor("shadow", operator_name="synthetic_op")
    roofline = HirRooflineSummary(
        anchor_dims={"M": 128},
        compute_flops=1024,
        min_gm_bytes=16384,
        arithmetic_intensity=1.0,
        compute_floor_us=2.0,
        memory_floor_us=3.0,
        ideal_roofline_us=3.0,
        dominant_bound="memory",
    )
    blueprint = HirBlueprintPacket(entry_name="kernel", roofline=roofline)
    regime = RegimeTargetSpec(
        "main",
        ("_build_main",),
        {},
        representative_anchor_dims={"M": 128},
        runtime_backend="ascendc_trackb",
    )
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        run_id="run_kb_01",
        checkpoint_path=ckpt_path,
        experience_lake_dir=lake_dir,
        cost_advisor=advisor,
        regime_specs=(regime,),
        regime_blueprints={"main": blueprint},
        topology_axes=axes,
    )

    shard = writer.record_kb_event(
        rule_id="R_TEST_001",
        postcondition_realized=True,
        speed_gain=0.25,
        topology_axes=axes,
        round_idx=2,
    )
    assert shard is not None and shard.is_file()
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["rule_id"] == "R_TEST_001"
    assert rows[0]["postcondition_realized"] is True
    assert rows[0]["speed_gain"] == pytest.approx(0.25)
    assert rows[0]["round_idx"] == 2

    no_lake_writer = TelemetryLedgerWriter(operator_name="synthetic_op")
    assert (
        no_lake_writer.record_kb_event(rule_id="R_TEST_001", postcondition_realized=False)
        is None
    )

    parent_src = "def _build_main(tile=16):\n    return tile\n"
    cand_src = "def _build_main(tile=32):\n    return tile\n"
    parent_ledger = _make_ledger(gm_roundtrip_bytes=65536)
    cand_ledger = _make_ledger(gm_roundtrip_bytes=32768)
    parent_res = _surrogate_result("parent", (_metric(1, "main", 20.0),))
    cand_res = _surrogate_result("cand", (_metric(1, "main", 10.0),))
    anchors = (AnchorCaseSpec("op/synthetic", 1, "main", 40.0, 4.0),)

    writer.record_surrogate_observations(
        source=cand_src,
        result=cand_res,
        active_anchors=anchors,
        parent_source=parent_src,
        parent_result=parent_res,
        parent_ledger=parent_ledger,
        ledger=cand_ledger,
        axes=axes,
        worker_kind="exploit_rule",
        rule_id="R_TEST_001",
        regime_tag="main",
    )
    writer.record_l4_observations(
        source=cand_src,
        l4_result=L4CannbenchResult(
            passed=True,
            avg_speedup=4.2,
            score=4.2,
            per_case_elapsed_us={1: 9.5},
            per_case_speedups={1: 4.21},
        ),
        active_anchors=anchors,
    )

    lake = ExperienceLake(lake_dir)
    loaded_cands, loaded_obs = lake.load_records()
    assert source_sha256(cand_src) in loaded_cands
    assert any(obs.eval_tier == "l2_pass" for obs in loaded_obs)
    assert any(obs.eval_tier == "l4_official" for obs in loaded_obs)

    rule = AscendCausalRule(
        rule_id="R_TEST_001",
        regime_guard="True",
        precondition="True",
        transform_template="halve gm roundtrip",
        expected_postcondition="gm_roundtrip_bytes < prev.gm_roundtrip_bytes",
        source_provenance="unit_test",
    )
    parent_residual = compute_static_dynamic_residual(parent_ledger, {"dur_us": 20.0})
    cand_residual = compute_static_dynamic_residual(cand_ledger, {"dur_us": 10.0})
    ok, gain = writer.evaluate_causal_rule_outcome(
        rule=rule,
        res_all_passed=True,
        rolled_back=False,
        ledger=cand_ledger,
        features={},
        cand_residual=cand_residual,
        cand_lat_us=10.0,
        cand_obj_score=4.0,
        parent_ledger=parent_ledger,
        parent_residual=parent_residual,
        parent_lat_us=20.0,
        parent_obj_score=2.0,
        has_cand_msprof=False,
    )
    assert ok is True
    assert gain == pytest.approx(0.5)

    ok_rb, gain_rb = writer.evaluate_causal_rule_outcome(
        rule=rule,
        res_all_passed=True,
        rolled_back=True,
        ledger=cand_ledger,
        features={},
        cand_residual=cand_residual,
        cand_lat_us=10.0,
        cand_obj_score=4.0,
        parent_ledger=parent_ledger,
        parent_residual=parent_residual,
        parent_lat_us=20.0,
        parent_obj_score=2.0,
        has_cand_msprof=False,
    )
    assert (ok_rb, gain_rb) == (False, 0.0)

    chk = L0L1CandidateCheck(
        candidate_id="cand_tb",
        passed=True,
        failed_tier=None,
        patched_module_source=cand_src,
        snippet_source=cand_src,
        stage_a=StageAResult(passed=True, prim_func=None, ledger=cand_ledger),
        stage_b=StageBResult(passed=True),
    )
    evidence = writer.collect_trackb_round_evidence(
        round_idx=1,
        regime_spec=regime,
        blueprint=blueprint,
        live_plan=None,
        active_mask=ConstraintRelaxationMask(),
        checks=[chk],
        l2_results={"cand_tb": cand_res},
        active_kernel="kernel",
        l2_run_ids={"cand_tb": "l2_001"},
        parent_sha256=source_sha256(parent_src),
        parent_l1_summary=None,
        parent_l1_summary_source="absent",
    )
    assert len(evidence) == 1
    assert evidence[0]["candidate_id"] == "cand_tb"
    assert evidence[0]["l2_run_id"] == "l2_001"
    assert writer.run_flywheel_distillation(
        baseline_source=parent_src,
        winner_source=cand_src,
        baseline_latency_us=20.0,
        winner_latency_us=10.0,
        func_name="_build_main",
        regime_spec=regime,
        active_anchors=anchors,
        topology_axes=axes,
    ) == []
