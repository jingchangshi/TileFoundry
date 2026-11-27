"""Tests for split-track KB rules and Top-2 L2 survivor promotion fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import tilefoundry_ascend.agent.campaign.promotion_coordinator as pc_mod
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.agent.campaign.promotion_coordinator import (
    PromotionCoordinator,
    RankedL2Candidate,
    evaluate_promotion_gate,
    select_top_two_l2_survivors,
    to_conservation_point,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE,
    FAILURE_KIND_INFRASTRUCTURE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
)
from tilefoundry_ascend.agent.kb import (
    BayesianRuleTracker,
    SymptomFamily,
    build_predicate_context,
    load_seed_rules,
)
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger


def _anchor_metric(
    case_num: int,
    cand_us: float,
    *,
    ref_us: float = 100.0,
    t_hw_us: float = 10.0,
    accuracy_passed: bool = True,
    paired_rel_mad: float = 0.02,
    burst_batch_cv: float = 0.0,
    ring_burst_k: int = 1,
) -> AnchorEvalMetrics:
    speedup = ref_us / cand_us if cand_us > 0.0 else 0.0
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag="main",
        accuracy_passed=accuracy_passed,
        cand_us=cand_us,
        ref_perf_us=ref_us,
        t_hw_us=t_hw_us,
        speedup=speedup,
        roofline_eta=t_hw_us / cand_us if cand_us > 0.0 else 0.0,
        eq3_score=speedup,
        ring_burst_k=ring_burst_k,
        burst_batch_cv=burst_batch_cv,
        paired_rel_mad=paired_rel_mad,
    )


def _surrogate(cid: str, cand_us: float, *, all_passed: bool = True) -> CandidateSurrogateResult:
    metrics = (
        _anchor_metric(1, cand_us, accuracy_passed=all_passed),
        _anchor_metric(2, cand_us * 1.1, accuracy_passed=all_passed),
    )
    return aggregate_surrogate_result(cid, "op_test", metrics)


def test_seed_rules_split_r19_and_add_r19a_and_r35_z3() -> None:
    """Verify R19 Track-B isolation, R19A TileLang stage pipeline, and R35 Z3 latency rule."""
    by_id = {rule.rule_id: rule for rule in load_seed_rules()}
    r19 = by_id["R19_STATIC_PINGPONG_UNROLL_NO_ALIAS"]
    r19a = by_id["R19A_CONSTRAINED_STAGE_OFFSET_PIPELINE"]
    r35_split = by_id["R35_PURE_VECTOR_DUAL_SUBCORE_SPLIT"]
    r35_z3 = by_id["R35_Z3_TASK_CYCLE_LATENCY_OVERRIDE"]

    assert r19.compatible_backends == ("ascendc_trackb", "hybrid_trackb")
    assert "ub_a" in r19.transform_template and "ub_b" in r19.transform_template
    assert "asc_sync_notify" in r19.transform_template
    assert "asc_sync_wait" in r19.transform_template
    assert "T.Pipelined" not in r19.transform_template
    assert "T.serial" not in r19.transform_template

    assert r19a.symptom_family == SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value
    assert r19a.compatible_backends == ("tilelang_dsl",)
    assert "enable_offset" in r19a.transform_template
    assert "T.Stage" in r19a.transform_template
    assert "T.annotate_buffer_versions" in r19a.transform_template

    assert r35_split.symptom_family == SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE.value
    assert r35_z3.symptom_family == SymptomFamily.SYNC_BARRIER_SERIALIZATION.value
    assert r35_z3.compatible_backends == ("tilelang_dsl",)
    assert "T.Task(" in r35_z3.transform_template
    assert "T.SimdVF(" in r35_z3.transform_template
    assert "T.SimtVF(" in r35_z3.transform_template

    tracker = BayesianRuleTracker.from_seed_yaml()
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 80, "AIV0.PIPE_MTE2": 80},
        waterlines={},
        uses_dynamic_offset_pipelining=True,
        in_loop_pipe_all_barriers=2,
    )
    ctx = build_predicate_context(ledger)
    tl_ids = {r.rule_id for r in tracker.rank_applicable_rules(ctx, runtime_backend="tilelang_dsl")}
    tb_ids = {r.rule_id for r in tracker.rank_applicable_rules(ctx, runtime_backend="cce_trackb")}
    assert "R19A_CONSTRAINED_STAGE_OFFSET_PIPELINE" in tl_ids
    assert "R35_Z3_TASK_CYCLE_LATENCY_OVERRIDE" in tl_ids
    assert "R19_STATIC_PINGPONG_UNROLL_NO_ALIAS" not in tl_ids
    assert "R19_STATIC_PINGPONG_UNROLL_NO_ALIAS" in tb_ids
    assert "R19A_CONSTRAINED_STAGE_OFFSET_PIPELINE" not in tb_ids
    assert "R35_Z3_TASK_CYCLE_LATENCY_OVERRIDE" not in tb_ids


def test_to_conservation_point_forwards_burst_cv_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify to_conservation_point preserves burst CV and burst_count forwarding."""
    cand = _anchor_metric(1, 20.0, burst_batch_cv=0.468, ring_burst_k=16)
    inc = _anchor_metric(1, 30.0, burst_batch_cv=0.125, ring_burst_k=8)
    pt = to_conservation_point(cand, inc)
    assert pt.case_num == 1
    assert pt.cand_us == pytest.approx(20.0)
    assert pt.inc_us == pytest.approx(30.0)

    @dataclass(frozen=True)
    class _ExtendedPoint:
        case_num: int
        regime_tag: str
        ref_us: float
        inc_us: float
        cand_us: float
        t_roof_us: float
        paired_rel_mad: float
        accuracy_passed: bool
        cand_burst_cv: float = 0.0
        ref_burst_cv: float = 0.0
        burst_count: int = 1

    monkeypatch.setattr(pc_mod, "AnchorConservationPoint", _ExtendedPoint)
    ext_pt = to_conservation_point(cand, inc)
    assert ext_pt.cand_burst_cv == pytest.approx(0.468)
    assert ext_pt.ref_burst_cv == pytest.approx(0.125)
    assert ext_pt.burst_count == 16

    ext_no_inc = to_conservation_point(cand, None)
    assert ext_no_inc.ref_burst_cv == pytest.approx(0.0)
    assert ext_no_inc.burst_count == 16


def test_select_top_two_l2_survivors_and_l3_fallback_matrix(tmp_path: Path) -> None:
    """Verify Top-2 ranking, rank-1 promotion, rank-2 fallback, dual rejection, and deferral."""
    target_path = tmp_path / "kernels.py"
    incumbent_src = "INCUMBENT = 1\n"
    target_path.write_text(incumbent_src, encoding="utf-8")

    def _write_and_verify(path: Path, text: str) -> str:
        path.write_text(text, encoding="utf-8")
        return ""

    coordinator = PromotionCoordinator(
        target_module_path=target_path,
        write_and_verify_fn=_write_and_verify,
    )
    regime = RegimeTargetSpec("main", ("_build",), {})
    anchors = (
        AnchorCaseSpec("synthetic/op", 1, "main", 100.0, 10.0),
        AnchorCaseSpec("synthetic/op", 2, "main", 100.0, 10.0),
    )
    best_res = _surrogate("inc", 60.0)

    candidates = (
        RankedL2Candidate("c_fail", "FAIL = 1\n", _surrogate("c_fail", 20.0, all_passed=False)),
        RankedL2Candidate("c_slow", "SLOW = 1\n", _surrogate("c_slow", 59.5)),
        RankedL2Candidate("c_mid", "MID = 1\n", _surrogate("c_mid", 45.0)),
        RankedL2Candidate("c_fast", "FAST = 1\n", _surrogate("c_fast", 30.0)),
        RankedL2Candidate("c_good", "GOOD = 1\n", _surrogate("c_good", 40.0)),
    )
    ranked = select_top_two_l2_survivors(
        candidates,
        best_result=best_res,
        regime_spec=regime,
        active_anchors=anchors,
    )
    assert ranked.rank_1 is not None and ranked.rank_1.candidate_id == "c_fast"
    assert ranked.rank_2 is not None and ranked.rank_2.candidate_id == "c_good"
    assert len(ranked.survivors) == 2
    rejected_map = dict(ranked.rejected)
    assert rejected_map["c_fail"] == "l2_not_passed"
    assert rejected_map["c_slow"] == "insufficient_gain"

    empty_decision = coordinator.coordinate_top_two_l3_promotion(
        select_top_two_l2_survivors((), best_result=best_res, regime_spec=regime),
        incumbent_source=incumbent_src,
        round_idx=1,
        regime_tag="main",
        run_l3_fn=lambda _c: L3AcceptanceResult(passed=True),
    )
    assert empty_decision.resolution is L3Resolution.NOT_RUN

    seen_rank1_win: list[str] = []

    def _l3_rank1_passes(cand: RankedL2Candidate) -> L3AcceptanceResult:
        seen_rank1_win.append(cand.candidate_id)
        return L3AcceptanceResult(passed=True, run_id="r1")

    win1 = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=1,
        regime_tag="main",
        run_l3_fn=_l3_rank1_passes,
    )
    assert seen_rank1_win == ["c_fast"]
    assert win1.resolution is L3Resolution.PROMOTED
    assert win1.promoted_candidate is not None
    assert win1.promoted_candidate.candidate_id == "c_fast"
    assert win1.fallback_used is False
    assert target_path.read_text(encoding="utf-8") == "FAST = 1\n"

    target_path.write_text(incumbent_src, encoding="utf-8")
    seen_fallback: list[str] = []

    def _l3_rank1_fails_rank2_passes(cand: RankedL2Candidate) -> L3AcceptanceResult:
        seen_fallback.append(cand.candidate_id)
        if cand.candidate_id == "c_fast":
            return L3AcceptanceResult(
                passed=False,
                failure_kind=FAILURE_KIND_CANDIDATE,
                failed_case_nums=(10, 15),
                error_message="vf-stack overflow on D=512",
            )
        return L3AcceptanceResult(passed=True, run_id="r2")

    win2 = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=2,
        regime_tag="main",
        run_l3_fn=_l3_rank1_fails_rank2_passes,
    )
    assert seen_fallback == ["c_fast", "c_good"]
    assert win2.resolution is L3Resolution.PROMOTED
    assert win2.promoted_candidate is not None
    assert win2.promoted_candidate.candidate_id == "c_good"
    assert win2.fallback_used is True
    assert win2.new_guardrail_cases == (10, 15)
    assert target_path.read_text(encoding="utf-8") == "GOOD = 1\n"

    target_path.write_text(incumbent_src, encoding="utf-8")

    def _l3_both_reject(cand: RankedL2Candidate) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            passed=False,
            failure_kind=FAILURE_KIND_CANDIDATE,
            failed_case_nums=(10,) if cand.candidate_id == "c_fast" else (10, 22),
            error_message="accuracy mismatch",
        )

    both_rej = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=3,
        regime_tag="main",
        run_l3_fn=_l3_both_reject,
    )
    assert both_rej.resolution is L3Resolution.CANDIDATE_REJECTED
    assert both_rej.fallback_used is True
    assert both_rej.new_guardrail_cases == (10, 22)
    assert target_path.read_text(encoding="utf-8") == incumbent_src

    seen_infra: list[str] = []

    def _l3_infra_defer(cand: RankedL2Candidate) -> L3AcceptanceResult:
        seen_infra.append(cand.candidate_id)
        return L3AcceptanceResult(
            passed=False,
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_terminal=False,
            run_id="infra_1",
            error_message="rc=75 device busy",
        )

    deferred = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=4,
        regime_tag="main",
        run_l3_fn=_l3_infra_defer,
    )
    assert seen_infra == ["c_fast"]
    assert deferred.resolution is L3Resolution.INFRA_DEFERRED
    assert deferred.fallback_used is False
    assert deferred.pending_promotion is not None
    assert deferred.pending_promotion.candidate_id == "c_fast"
    assert target_path.read_text(encoding="utf-8") == incumbent_src


def test_paired_guardrail_uses_live_incumbent_over_vendor_ref_fallback() -> None:
    """Verify paired guardrails compare against live incumbent rather than vendor ref."""
    regime_spec = RegimeTargetSpec("main", ("_build",), {})
    best_result = aggregate_surrogate_result(
        "inc",
        "mla",
        (
            _anchor_metric(1, 50.0, ref_us=35.0, t_hw_us=10.0),
            _anchor_metric(2, 60.0, ref_us=40.0, t_hw_us=12.0),
        ),
    )
    cand_result = aggregate_surrogate_result(
        "cand_fast",
        "mla",
        (
            AnchorEvalMetrics(
                case_num=1,
                regime_tag="main",
                accuracy_passed=True,
                cand_us=45.0,
                ref_perf_us=35.0,
                t_hw_us=10.0,
                speedup=35.0 / 45.0,
                roofline_eta=10.0 / 45.0,
                eq3_score=35.0 / 45.0,
                ring_burst_k=24,
                burst_batch_cv=0.01,
                paired_speedup_vs_incumbent=50.0 / 45.0,
                paired_rel_mad=0.01,
            ),
            AnchorEvalMetrics(
                case_num=2,
                regime_tag="main",
                accuracy_passed=True,
                cand_us=54.0,
                ref_perf_us=40.0,
                t_hw_us=12.0,
                speedup=40.0 / 54.0,
                roofline_eta=12.0 / 54.0,
                eq3_score=40.0 / 54.0,
                ring_burst_k=24,
                burst_batch_cv=0.01,
                paired_speedup_vs_incumbent=60.0 / 54.0,
                paired_rel_mad=0.01,
            ),
        ),
    )
    guardrail = GuardrailCaseSpec(
        task="level4/mla",
        case_num=3,
        regime_tag="main",
        incumbent_ref_us=42.39,
        t_hw_us=10.3,
    )
    anchors = (
        AnchorCaseSpec(task="level4/mla", case_num=1, regime_tag="main", ref_perf_us=35.0, t_hw_us=10.0),
        AnchorCaseSpec(task="level4/mla", case_num=2, regime_tag="main", ref_perf_us=40.0, t_hw_us=12.0),
    )

    faster_paired_guardrail = AnchorEvalMetrics(
        case_num=3,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=52.0,
        ref_perf_us=42.39,
        t_hw_us=10.3,
        speedup=42.39 / 52.0,
        roofline_eta=10.3 / 52.0,
        eq3_score=42.39 / 52.0,
        ring_burst_k=24,
        burst_batch_cv=0.01,
        paired_speedup_vs_incumbent=1.05,
        paired_rel_mad=0.01,
    )
    ok, _res, reason = evaluate_promotion_gate(
        candidate_id="cand_fast",
        cur_res=cand_result,
        best_result=best_result,
        regime_spec=regime_spec,
        guardrail_cases=(guardrail,),
        cand_guardrail_metrics={3: faster_paired_guardrail},
        unchanged_siblings=set(),
        active_anchors=anchors,
    )
    assert ok is True
    assert reason == ""

    regressed_paired_guardrail = AnchorEvalMetrics(
        case_num=3,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=65.0,
        ref_perf_us=42.39,
        t_hw_us=10.3,
        speedup=42.39 / 65.0,
        roofline_eta=10.3 / 65.0,
        eq3_score=42.39 / 65.0,
        ring_burst_k=24,
        burst_batch_cv=0.01,
        paired_speedup_vs_incumbent=0.80,
        paired_rel_mad=0.01,
    )
    ok_reg, _res_reg, reason_reg = evaluate_promotion_gate(
        candidate_id="cand_fast",
        cur_res=cand_result,
        best_result=best_result,
        regime_spec=regime_spec,
        guardrail_cases=(guardrail,),
        cand_guardrail_metrics={3: regressed_paired_guardrail},
        unchanged_siblings=set(),
        active_anchors=anchors,
    )
    assert ok_reg is False
    assert reason_reg == "guardrail_latency_regression"

    unpaired_guardrail = GuardrailCaseSpec(
        task="level4/mla",
        case_num=3,
        regime_tag="main",
        incumbent_ref_us=20.0,
        t_hw_us=10.3,
    )
    unpaired_regressed = AnchorEvalMetrics(
        case_num=3,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=25.0,
        ref_perf_us=20.0,
        t_hw_us=10.3,
        speedup=20.0 / 25.0,
        roofline_eta=10.3 / 25.0,
        eq3_score=20.0 / 25.0,
        ring_burst_k=1,
        burst_batch_cv=0.0,
        paired_speedup_vs_incumbent=0.0,
        paired_rel_mad=0.02,
    )
    ok_unpaired, _res_unp, reason_unp = evaluate_promotion_gate(
        candidate_id="cand_fast",
        cur_res=cand_result,
        best_result=best_result,
        regime_spec=regime_spec,
        guardrail_cases=(unpaired_guardrail,),
        cand_guardrail_metrics={3: unpaired_regressed},
        unchanged_siblings=set(),
        active_anchors=anchors,
    )
    assert ok_unpaired is False
    assert reason_unp == "guardrail_latency_regression"
