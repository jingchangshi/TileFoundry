"""Consolidated tests for TelemetryLedgerWriter, FunnelTelemetry, and CalibrationService."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.calibration_service import (
    CalibrationService,
    CalibrationSnapshot,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    MAX_CLEAN_CALIBRATION_BURST_CV,
    SPEARMAN_QUARANTINE_FLOOR,
    DeterministicSearchBoundaryLedger,
    RoundFunnelEvent,
    append_round_funnel_event,
    assert_unmodeled_fault_share_bounded,
    compute_spearman_rho,
    is_clean_calibration_sample,
)
from tilefoundry_ascend.agent.campaign.l3_l4_recovery import PendingL4
from tilefoundry_ascend.agent.campaign.promotion_coordinator import PendingPromotion
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L2BatchResults,
    L3AcceptanceResult,
    L4CannbenchResult,
)
from tilefoundry_ascend.agent.campaign.telemetry_ledger_writer import (
    RoundReceipt,
    TelemetryLedgerWriter,
    source_sha256,
)
from tilefoundry_ascend.agent.contracts import CampaignRoundRecord
from tilefoundry_ascend.agent.cost_advisor import create_cost_advisor
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, L3Resolution, TypedFault
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger, build_scope_waterline
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket, HirRooflineSummary
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes


def _ledger(*, gm_bytes: int = 32768) -> object:
    wl = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"a": 32768},
            implicit_buffers={},
        )
    }
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120},
        waterlines=wl,
        gm_roundtrip_bytes=gm_bytes,
        grid_blocks=28,
    )


def _metric(
    case_num: int,
    regime_tag: str,
    cand_us: float,
    *,
    ref_us: float = 40.0,
    t_hw_us: float = 4.0,
    burst_batch_cv: float = 0.02,
    eq3_score: float | None = None,
) -> AnchorEvalMetrics:
    sp = ref_us / cand_us if cand_us > 0.0 else 0.0
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=True,
        cand_us=cand_us,
        ref_perf_us=ref_us,
        t_hw_us=t_hw_us,
        speedup=sp,
        roofline_eta=(t_hw_us / cand_us) if cand_us > 0.0 else 0.0,
        eq3_score=sp if eq3_score is None else eq3_score,
        burst_batch_cv=burst_batch_cv,
    )


def _surrogate(cid: str, anchors: tuple[AnchorEvalMetrics, ...]) -> CandidateSurrogateResult:
    valid = [a for a in anchors if a.cand_us > 0.0]
    avg_sp = sum(a.speedup for a in valid) / max(1, len(valid))
    return CandidateSurrogateResult(
        candidate_id=cid,
        operator_name="synthetic_op",
        all_passed=all(a.accuracy_passed and a.cand_us > 0.0 for a in anchors),
        avg_speedup=avg_sp,
        min_case_speedup=min((a.speedup for a in valid), default=0.0),
        avg_roofline_eta=sum(a.roofline_eta for a in valid) / max(1, len(valid)),
        surrogate_score=avg_sp,
        anchors=anchors,
    )


def test_funnel_telemetry_spearman_and_boundary_ledger(tmp_path: Path) -> None:
    """Verify RoundFunnelEvent fault mapping, Spearman quarantine, and boundary ledger."""
    evt_ok = RoundFunnelEvent.from_typed_fault(
        run_id="r1",
        operator="synthetic_op",
        round_idx=1,
        regime_tag="main",
        topology_axes={"compute_engine": "aiv_only", "memory_ring": "gm_ub_stream"},
        candidate_id="c_ok",
        cand_sha="sha1",
        worker_kind="rule_exploit",
        fault=None,
    )
    evt_noop = evt_ok.apply_typed_fault(
        TypedFault(plane=FaultPlane.NOOP_IDENTICAL, code="NOOP_IDENTICAL", failed_tier="noop_identical", detail="dup")
    )
    assert evt_ok.fault_plane == FaultPlane.NONE.value
    assert evt_noop.noop_identical is True
    assert assert_unmodeled_fault_share_bounded([evt_ok, evt_noop]) == 0.0
    out_path = append_round_funnel_event(tmp_path / "tel", evt_ok)
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 1

    assert is_clean_calibration_sample(MAX_CLEAN_CALIBRATION_BURST_CV) is True
    assert is_clean_calibration_sample(MAX_CLEAN_CALIBRATION_BURST_CV + 0.01) is False
    assert is_clean_calibration_sample(float("nan")) is False
    assert compute_spearman_rho({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}) is None
    assert compute_spearman_rho({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 10.0, "b": 20.0, "c": 30.0}) == pytest.approx(1.0)

    ledger = DeterministicSearchBoundaryLedger()
    assert ledger.record_compile_wall(
        regime_tag="prefill",
        cand_sha="sha_ub",
        knob_signature="M=256",
        error_code_norm="UB_SRAM_OVERFLOW",
        ub_bytes=200000,
    )
    assert ledger.record_compile_wall(
        regime_tag="prefill",
        cand_sha="sha_stk",
        knob_signature="M=128",
        error_code_norm="VF_STACK_OVERFLOW",
        stack_bytes=16384,
    )
    assert ledger.precheck_known_infeasible(regime_tag="prefill", cand_sha="sha_ub") == "deterministic_boundary_sha:UB_SRAM_OVERFLOW"
    assert ledger.precheck_known_infeasible(regime_tag="prefill", cand_sha="other", knob_signature="M=256") == "deterministic_boundary_knobs:UB_SRAM_OVERFLOW"
    assert "ub_bytes >= 200000" in ledger.format_regime_boundary_prompt("prefill")
    assert "stack_bytes=16384" in ledger.format_regime_boundary_prompt("prefill")

    ledger.record_spearman_rho("prefill", SPEARMAN_QUARANTINE_FLOOR - 0.1)
    assert ledger.is_cost_prescreen_quarantined("prefill") is False
    ledger.record_spearman_rho("prefill", SPEARMAN_QUARANTINE_FLOOR - 0.05)
    assert ledger.is_cost_prescreen_quarantined("prefill") is True
    assert DeterministicSearchBoundaryLedger.from_dict(ledger.to_dict()) == ledger


def test_telemetry_writer_receipt_funnel_events_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Verify RoundReceipt, JSONL funnel events, and checkpoint save/restore roundtrip."""
    receipt = RoundReceipt(
        round_idx=1,
        regime_tag="main",
        promoted=True,
        l3_resolution=L3Resolution.PROMOTED.value,
        stagnation_after=0,
        admitted_rule_ids=("R1",),
        stage_timings_sec={"l0_l1_sec": 0.1234567},
        prewarm_stats={"total_tasks": 2, "cache_hits": 1},
        cost_advisor_predictions=3,
        emitted_event_count=2,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(receipt, "promoted", False)
    assert receipt.to_dict()["stage_timings_sec"] == {"l0_l1_sec": 0.123457}

    target_path = tmp_path / "kernel.py"
    inc_src = "def _build_main(tile=16):\n    return tile\n"
    cand_src = "def _build_main(tile=32):\n    return tile\n"
    target_path.write_text(inc_src, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt.json"

    tracker = BayesianRuleTracker.from_seed_yaml()
    first_rule = next(iter(tracker.posteriors))
    tracker.record_outcome(first_rule, postcondition_realized=True, speed_gain=0.15)
    writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        run_id="run_w",
        target_module_path=target_path,
        checkpoint_path=ckpt_path,
        tracker=tracker,
        beams={"main": TwoLaneBeamSearch()},
        regime_specs=(RegimeTargetSpec("main", ("_build_main",), {}),),
        scoreboard_anchors=(AnchorCaseSpec("op/synthetic", 1, "main", 40.0, 4.0),),
        guardrail_cases=(GuardrailCaseSpec("op/synthetic", 2, "main", 18.0),),
        calibrated_dispatch_floor_us=1.25,
        topology_axes=TopologyAxes("coupled_1c2v", "gm_ub_stream", "affine_simd"),
    )
    chk_occ = L0L1CandidateCheck("cand_occ", True, None, cand_src, cand_src)
    chk_sb = L0L1CandidateCheck(
        "cand_sb",
        False,
        "stage_b",
        cand_src,
        cand_src,
        stage_b=StageBResult(
            passed=False,
            error_message="spill",
            spill_bytes=512,
            compiler_summary=BishengResourceSummary(spill_bytes=512, stack_bytes=2048),
            fault_plane="STAGE_B_COMPILE",
            fault_code="VF_STACK_OVERFLOW",
            failed_tier="stage_b",
        ),
    )
    occ_fault = TypedFault(FaultPlane.INFRA_OCCUPANCY, "L2_INFRA_OCCUPANCY", "l2_infra_occupancy", "busy", True)
    emitted = writer.emit_round_funnel_events(
        round_idx=1,
        regime_spec=RegimeTargetSpec("main", ("_build_main",), {}),
        l0_l1_checks=[chk_occ, chk_sb],
        failed_entries=[("cand_occ", "l2_infra_occupancy", "busy"), ("cand_noop", "noop_identical", "dup")],
        l2_infra_fault=occ_fault,
    )
    assert emitted.emitted_event_count == 3
    rows = [json.loads(line) for line in (tmp_path / "telemetry" / "round_funnel_events.jsonl").read_text().splitlines()]
    by_cid = {r["candidate_id"]: r for r in rows}
    assert by_cid["cand_occ"]["fault_plane"] == FaultPlane.INFRA_OCCUPANCY.value
    assert by_cid["cand_sb"]["fault_plane"] == FaultPlane.COMPILER_LOWERING.value
    assert by_cid["cand_noop"]["fault_plane"] == FaultPlane.NOOP_IDENTICAL.value

    writer.ensure_regime_diag_entry("main")["l1_compiler_penalty_rejects"] = 1
    writer.msprof_cache[(source_sha256(inc_src), "main")] = {"aic_mac_ratio": 0.4}
    base_res = _surrogate("base", (_metric(1, "main", 20.0),))
    best_res = _surrogate("cand", (_metric(1, "main", 10.0),))
    round_rec = CampaignRoundRecord(
        round_idx=1,
        regime_tag="main",
        builder_name="_build_main",
        parent_lane="exploit",
        creative_effort="medium",
        l0_l1_passed_ids=("cand",),
        l0_l1_failed=(),
        best_round_speedup=4.0,
        best_so_far_speedup=4.0,
        promoted=True,
        l3_resolution=L3Resolution.PROMOTED.value,
    )
    writer.save_checkpoint(
        incumbent_source=inc_src,
        baseline_result=base_res,
        best_result=best_res,
        active_anchors=writer.scoreboard_anchors,
        frozen_regimes={"tail"},
        current_regime="main",
        stagnation_rounds=0,
        rounds=[round_rec],
        promoted_rounds=[1],
        rolled_back_rounds=[],
        escalated_rounds=[],
        admitted_rule_ids=[],
        stop_reason="in_progress",
        l4_result=None,
        pending_promotion=PendingPromotion(
            candidate_source=cand_src,
            candidate_sha256=source_sha256(cand_src),
            target_path=str(target_path),
            round_idx=2,
            regime_tag="main",
            candidate_result=best_res,
            candidate_id="cand_2",
            execution_id="ex_p",
            execution_terminal=False,
        ),
        pending_l4=PendingL4(target_path=str(target_path), candidate_sha256=source_sha256(inc_src), execution_id="ex_l4"),
    )
    restored_writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        target_module_path=target_path,
        checkpoint_path=ckpt_path,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        beams={"main": TwoLaneBeamSearch()},
        regime_specs=(RegimeTargetSpec("main", ("_build_main",), {}),),
    )
    r_inc, _, _, _, r_frozen, r_reg, _, r_rounds, *_ = restored_writer.restore_from_checkpoint()
    assert r_inc == inc_src and r_frozen == {"tail"} and r_reg == "main" and len(r_rounds) == 1
    assert target_path.read_text(encoding="utf-8") == cand_src
    assert restored_writer.msprof_cache[(source_sha256(inc_src), "main")] == {"aic_mac_ratio": 0.4}


def test_telemetry_writer_lake_kb_evidence_and_stagnation_diagnosis(tmp_path: Path) -> None:
    """Verify ExperienceLake writes, KB events, rule evaluation, Track-B evidence, and diagnosis."""
    lake_dir = tmp_path / "lake"
    roofline = HirRooflineSummary({"M": 128}, 1024, 16384, 1.0, 2.0, 3.0, 3.0, "memory")
    bp = HirBlueprintPacket(entry_name="kernel", roofline=roofline)
    regime = RegimeTargetSpec("main", ("_build_main",), {}, representative_anchor_dims={"M": 128}, runtime_backend="ascendc_trackb")
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    writer = TelemetryLedgerWriter(
        operator_name="synthetic_op",
        run_id="run_kb",
        checkpoint_path=tmp_path / "ckpt.json",
        experience_lake_dir=lake_dir,
        cost_advisor=create_cost_advisor("shadow", operator_name="synthetic_op"),
        perf_target=PerfTargetSpec("synthetic_op", 2.0, 1.5, 0.85),
        regime_specs=(regime,),
        regime_blueprints={"main": bp},
        topology_axes=axes,
    )
    shard = writer.record_kb_event(rule_id="R_TEST", postcondition_realized=True, speed_gain=0.2, topology_axes=axes, round_idx=1)
    assert shard is not None and shard.is_file()

    p_src, c_src = "def _build_main(t=16):\n    return t\n", "def _build_main(t=32):\n    return t\n"
    p_ledger, c_ledger = _ledger(gm_bytes=65536), _ledger(gm_bytes=32768)
    p_res = _surrogate("p", (_metric(1, "main", 20.0),))
    c_res = _surrogate("c", (_metric(1, "main", 10.0),))
    anchors = (AnchorCaseSpec("op/synthetic", 1, "main", 40.0, 4.0),)
    writer.record_surrogate_observations(
        source=c_src,
        result=c_res,
        active_anchors=anchors,
        parent_source=p_src,
        parent_result=p_res,
        parent_ledger=p_ledger,
        ledger=c_ledger,
        axes=axes,
        worker_kind="exploit_rule",
        rule_id="R_TEST",
        regime_tag="main",
    )
    writer.record_l4_observations(
        source=c_src,
        l4_result=L4CannbenchResult(passed=True, per_case_elapsed_us={1: 9.5}, per_case_speedups={1: 4.2}),
        active_anchors=anchors,
    )
    loaded_cands, loaded_obs = ExperienceLake(lake_dir).load_records()
    assert source_sha256(c_src) in loaded_cands
    assert {o.eval_tier for o in loaded_obs} >= {"l2_pass", "l4_official"}

    rule = AscendCausalRule("R_TEST", "True", "True", "halve gm", "gm_roundtrip_bytes < prev.gm_roundtrip_bytes", "test")
    ok, gain = writer.evaluate_causal_rule_outcome(
        rule=rule,
        res_all_passed=True,
        rolled_back=False,
        ledger=c_ledger,
        features={},
        cand_residual=compute_static_dynamic_residual(c_ledger, {"dur_us": 10.0}),
        cand_lat_us=10.0,
        cand_obj_score=4.0,
        parent_ledger=p_ledger,
        parent_residual=compute_static_dynamic_residual(p_ledger, {"dur_us": 20.0}),
        parent_lat_us=20.0,
        parent_obj_score=2.0,
        has_cand_msprof=False,
    )
    assert ok is True and gain == pytest.approx(0.5)

    chk = L0L1CandidateCheck("c", True, None, c_src, c_src, stage_a=StageAResult(True, None, c_ledger), stage_b=StageBResult(True))
    ev = writer.collect_trackb_round_evidence(
        round_idx=1,
        regime_spec=regime,
        blueprint=bp,
        live_plan=None,
        active_mask=ConstraintRelaxationMask(),
        checks=[chk],
        l2_results={"c": c_res},
        active_kernel="kernel",
        l2_run_ids={"c": "l2_1"},
        parent_sha256=source_sha256(p_src),
        parent_l1_summary=None,
        parent_l1_summary_source="absent",
    )
    assert len(ev) == 1 and ev[0]["l2_run_id"] == "l2_1"

    writer.ensure_regime_diag_entry("main")["rule_starvation_rounds"] = 3
    diag = writer.build_stagnation_diagnosis(best_result=c_res, stop_reason="budget_exhausted")
    assert diag["primary_stagnation_cause"] == "rule_starvation"
    assert writer.classify_primary_stagnation_cause(
        rule_starvation_rounds=0, cost_prescreen_filtered_count=0, l0_rejects=0, l1_rejects=0,
        l2_no_gain_rounds=0, l3_rollback_count=0, final_roofline_eta=0.9, target_roofline_eta=0.85, stop_reason="roofline_saturated",
    ) == "roofline_saturated"


def test_calibration_service_snapshot_floor_msprof_and_warmstart(tmp_path: Path) -> None:
    """Verify CalibrationService snapshot, dispatch floor, msprof keying, and phase-0 warmstart."""
    inc_src = "def _select_cfg():\n    return {'bm': 16}\n\ndef _build_decode():\n    return 16\n"
    lat_src = "def _select_cfg():\n    return {'bm': 64}\n\ndef _build_decode():\n    return 64\n"
    target_path = tmp_path / "kernel.py"
    target_path.write_text(inc_src, encoding="utf-8")
    anchors = [
        AnchorCaseSpec("op/synthetic", 101, "decode", 20.0, 4.0),
        AnchorCaseSpec("op/synthetic", 102, "decode", 30.0, 6.0),
    ]
    regimes = [
        RegimeTargetSpec(
            "decode",
            ("_build_decode",),
            {"tile_m": 16},
            config_selector_fn="_select_cfg",
            representative_anchor_dims={"M": 1, "N": 1024},
        )
    ]
    base_res = _surrogate(
        "incumbent_baseline",
        (_metric(101, "decode", 25.0, ref_us=20.0, t_hw_us=4.0, eq3_score=0.5), _metric(102, "decode", 12.0, ref_us=30.0, t_hw_us=6.0, eq3_score=0.9)),
    )
    svc = CalibrationService(
        operator_name="synthetic_op",
        run_l2_batch_fn=lambda _c, _a: L2BatchResults({"incumbent_baseline": base_res}, {"incumbent_baseline": "r1"}, null_launch_us=1.75),
        msprof_collector=lambda _op, _p, seq: {s.regime_tag: {"pipe_v": float(s.case_num)} for s in seq},
        p2_deep_dive_collector=lambda _op, _p, _seq: {"decode": {"hotspot": "loop_0"}},
    )
    snap = svc.build_snapshot(inc_src, anchors, regimes)
    assert isinstance(snap, CalibrationSnapshot)
    assert snap.calibrated_dispatch_floor_us == pytest.approx(1.75)
    sha = hashlib.sha256(inc_src.encode("utf-8")).hexdigest()
    assert snap.msprof_by_case == {(sha, 101): {"pipe_v": 101.0}}
    assert svc.calibrate_dispatch_floor(base_res, null_launch_us=None) == pytest.approx(6.0)
    assert svc.collect_p2_deep_dive_for_regime(inc_src, base_res, anchors, regimes[0]) == {"hotspot": "loop_0"}

    @dataclass(frozen=True)
    class _LatCand:
        patched_source: str
        pred_us: float

    win_res = _surrogate("phase0_lat_0", (_metric(101, "decode", 8.0), _metric(102, "decode", 8.0)))
    ws_svc = CalibrationService(
        operator_name="synthetic_op",
        target_module_path=target_path,
        run_l0_fn=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_ledger()),
        predict_candidate_us_fn=lambda _wl: 20.0,
        sweep_lattice_fn=lambda *_a, **_kw: [_LatCand(lat_src, 10.0)],
        run_l2_batch_fn=lambda _c, _a: {"phase0_lat_0": win_res},
        evaluate_tail_guard_fn=lambda **kw: (True, kw["top_res"]),
        run_l3_contract_fn=lambda _p: L3AcceptanceResult(passed=True),
        enable_phase0_lattice_sweep=True,
        regime_specs=regimes,
    )
    out_src, out_res = ws_svc.run_phase0_lattice_warmstart(inc_src, base_res, anchors)
    assert out_src == lat_src and out_res == win_res
    assert target_path.read_text(encoding="utf-8") == lat_src
