"""Integration tests for infra isolation, promotion gates, flywheel, and NPU provenance."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import (
    AnchorCaseSpec,
    CampaignConfig,
    CampaignReport,
    DEFAULT_REGIME_SPECS,
    PerfTargetCampaignRunner,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.agent.campaign.flywheel_coordinator import FlywheelCoordinator
from tilefoundry_ascend.agent.campaign.promotion_coordinator import (
    PromotionCoordinator,
    evaluate_promotion_gate,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_INFRASTRUCTURE,
    AgentGatesDeviceFaultError,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
)
from tilefoundry_ascend.agent.flywheel import FlywheelDistiller
from tilefoundry_ascend.agent.kb import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator import GateFunnel, MockLLMBackend, StageBResult
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult
from tilefoundry_ascend.ascend.cost_model.experience_lake import CaseObservationRecord
from tilefoundry_ascend.ascend.passes import (
    build_objective_physical_ledger,
    build_scope_waterline,
)

_BASELINE_SRC = (
    "def _build_main(tile_m=16):\n"
    "    return int(tile_m)\n\n"
    "def kernel(x):\n"
    "    return _build_main(), x\n"
)

_CANDIDATE_SRC = (
    "def _build_main(tile_m=32):\n"
    "    return int(tile_m)\n"
)


def _stage_a() -> StageAResult:
    waterlines = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"a": 65536},
            implicit_buffers={},
        )
    }
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120},
        waterlines=waterlines,
        gm_roundtrip_bytes=32768,
        grid_blocks=28,
    )
    return StageAResult(passed=True, prim_func=None, ledger=ledger)


def _metric(
    case_num: int,
    regime_tag: str,
    cand_us: float,
    *,
    ref_us: float = 40.0,
    t_hw_us: float = 4.0,
    paired_rel_mad: float = 0.02,
    accuracy_passed: bool = True,
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
        paired_rel_mad=paired_rel_mad,
    )


def _surrogate_result(
    candidate_id: str,
    anchors: tuple[AnchorEvalMetrics, ...],
) -> CandidateSurrogateResult:
    valid = [a for a in anchors if a.cand_us > 0.0]
    avg_sp = sum(a.speedup for a in valid) / max(1, len(valid))
    min_sp = min((a.speedup for a in valid), default=0.0)
    avg_eta = sum(a.roofline_eta for a in valid) / max(1, len(valid))
    all_ok = all(a.accuracy_passed and a.cand_us > 0.0 for a in anchors)
    return CandidateSurrogateResult(
        candidate_id=candidate_id,
        operator_name="synthetic_v54",
        all_passed=all_ok,
        avg_speedup=avg_sp,
        min_case_speedup=min_sp,
        avg_roofline_eta=avg_eta,
        surrogate_score=avg_sp,
        anchors=anchors,
    )


@pytest.mark.parametrize(
    ("fault_plane", "fault_code", "err_msg"),
    [
        ("INFRA_DEVICE_FAULT", "NPU_DEVICE_TSD_FAULT", "BLOCKED_ENV: E39006 TsdOpen failed"),
        ("INFRA_OCCUPANCY", "NPU_OCCUPANCY_BUSY", "BLOCKED_ENV: device occupied"),
    ],
)
def test_l2_infra_fault_isolates_bandit_stagnation_and_logs_funnel(
    tmp_path: Path,
    fault_plane: str,
    fault_code: str,
    err_msg: str,
) -> None:
    """L2 device/occupancy faults keep bandit trials and stagnation untouched."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASELINE_SRC, encoding="utf-8")
    lake_dir = tmp_path / "lake"
    tracker = BayesianRuleTracker.from_seed_yaml()
    initial_trials = {rid: st.trials for rid, st in tracker.posteriors.items()}
    anchors = (AnchorCaseSpec("synthetic/v54", 1, "main", 40.0, 4.0),)
    regimes = (RegimeTargetSpec("main", ("_build_main",), {}),)
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_CANDIDATE_SRC] * 4,
            "explore_creative": [
                "# target_bottleneck: aiv_vec\n"
                "# structural_hypothesis: Double tile_m\n"
                f"```python\n{_CANDIDATE_SRC}```"
            ]
            * 4,
        }
    )

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        if all(cid in {"baseline", "incumbent_baseline"} for cid, _p in cands):
            return {
                cands[0][0]: _surrogate_result(cands[0][0], (_metric(1, "main", 20.0),)),
            }
        raise AgentGatesDeviceFaultError(
            err_msg,
            fault_plane=fault_plane,
            fault_code=fault_code,
            failed_tier="l2_npu",
            device_id="0",
        )

    ckpt_path = tmp_path / "ckpt.json"
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="synthetic_v54", max_rounds=1),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_v54", 2.0, 1.5, 0.85),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda _s, _r: _stage_a(),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        checkpoint_path=ckpt_path,
        experience_lake_dir=lake_dir,
    )
    report = asyncio.run(runner.run())

    assert isinstance(report, CampaignReport)
    assert report.total_rounds == 1
    assert report.promoted_rounds == ()
    assert report.rolled_back_rounds == ()
    assert report.rounds[0].stagnation_rounds_after == 0
    assert target_path.read_text(encoding="utf-8") == _BASELINE_SRC
    for rid, state in tracker.posteriors.items():
        assert state.trials == initial_trials[rid]

    funnel_files = list((tmp_path / "telemetry").rglob("round_funnel_events.jsonl"))
    assert funnel_files
    rows = [
        json.loads(line)
        for path in funnel_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("fault_plane") == fault_plane for row in rows)


def test_l3_infra_deferred_isolates_bandit_and_stagnation(tmp_path: Path) -> None:
    """L3 infrastructure failure resolves to INFRA_DEFERRED without bandit or stall penalty."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASELINE_SRC, encoding="utf-8")
    lake_dir = tmp_path / "lake"
    tracker = BayesianRuleTracker.from_seed_yaml()
    initial_trials = {rid: st.trials for rid, st in tracker.posteriors.items()}
    anchors = (AnchorCaseSpec("synthetic/v54", 1, "main", 40.0, 4.0),)
    regimes = (RegimeTargetSpec("main", ("_build_main",), {}),)
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_CANDIDATE_SRC] * 4,
            "explore_creative": [
                "# target_bottleneck: aiv_vec\n"
                "# structural_hypothesis: Double tile_m\n"
                f"```python\n{_CANDIDATE_SRC}```"
            ]
            * 4,
        }
    )

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        _active: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat = 10.0 if "tile_m=32" in src else 20.0
            out[cid] = _surrogate_result(cid, (_metric(1, "main", lat),))
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="synthetic_v54", max_rounds=1),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_v54", 3.0, 2.0, 0.85),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda _s, _r: _stage_a(),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(
            passed=False,
            run_id="run-l3-def",
            error_message="BLOCKED_ENV: occupancy contention",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            failure_category="occupancy",
            execution_terminal=True,
        ),
        experience_lake_dir=lake_dir,
    )
    report = asyncio.run(runner.run())

    assert report.promotion_blocked is True
    assert report.stop_reason == "promotion_blocked"
    assert report.promoted_rounds == ()
    assert report.rolled_back_rounds == ()
    assert report.rounds[0].l3_resolution == L3Resolution.INFRA_DEFERRED.value
    assert report.rounds[0].stagnation_rounds_after == 0
    assert target_path.read_text(encoding="utf-8") == _BASELINE_SRC
    for rid, state in tracker.posteriors.items():
        assert state.trials == initial_trials[rid]


def test_promotion_coordinator_and_flywheel_coordinator_and_report_fields(
    tmp_path: Path,
) -> None:
    """Verify PromotionCoordinator, FlywheelCoordinator 3-sigma gate, and provenance fields."""
    target_path = tmp_path / "target_kernels.py"
    target_path.write_text(_BASELINE_SRC, encoding="utf-8")
    written: list[tuple[Path, str]] = []

    def _write_and_verify(path: Path, text: str) -> str:
        written.append((path, text))
        path.write_text(text, encoding="utf-8")
        return ""

    coordinator = PromotionCoordinator(
        target_module_path=target_path,
        write_and_verify_fn=_write_and_verify,
    )
    target_path.write_text(_CANDIDATE_SRC, encoding="utf-8")
    coordinator.rollback_to_incumbent(_BASELINE_SRC)
    assert target_path.read_text(encoding="utf-8") == _BASELINE_SRC
    assert len(written) == 1

    regime = RegimeTargetSpec("main", ("_build_main",), {})
    anchor_spec = (AnchorCaseSpec("synthetic/v54", 1, "main", 40.0, 4.0),)
    best_res = _surrogate_result("inc", (_metric(1, "main", 20.0, paired_rel_mad=0.05),))
    fast_res = _surrogate_result("fast", (_metric(1, "main", 15.0, paired_rel_mad=0.05),))
    noise_res = _surrogate_result("noise", (_metric(1, "main", 19.8, paired_rel_mad=0.05),))

    ok_fast, _, reason_fast = evaluate_promotion_gate(
        candidate_id="fast",
        cur_res=fast_res,
        best_result=best_res,
        regime_spec=regime,
        guardrail_cases=(),
        cand_guardrail_metrics={},
        unchanged_siblings=set(),
        active_anchors=anchor_spec,
    )
    assert ok_fast is True
    assert reason_fast == ""

    ok_noise, _, reason_noise = evaluate_promotion_gate(
        candidate_id="noise",
        cur_res=noise_res,
        best_result=best_res,
        regime_spec=regime,
        guardrail_cases=(),
        cand_guardrail_metrics={},
        unchanged_siblings=set(),
        active_anchors=anchor_spec,
    )
    assert ok_noise is False
    assert reason_noise == "insufficient_gain"

    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
    )
    fw_coord = FlywheelCoordinator(flywheel=flywheel)
    fw_coord.seed_baseline("main", _BASELINE_SRC, 20.0, 1.0)
    assert fw_coord.should_trigger("main", 18.5, 1.0) is False
    assert fw_coord.should_trigger("main", 15.0, 1.0) is True

    rep = CampaignReport(
        operator_name="synthetic_v54",
        stop_reason="goal_met",
        total_rounds=1,
        baseline_result=best_res,
        best_result=fast_res,
        rounds=(),
        real_npu_path=True,
        calibrated_dispatch_floor_us=1.25,
        geometric_speedup=2.67,
    )
    payload = rep.to_dict()
    assert payload["real_npu_path"] is True
    assert payload["calibrated_dispatch_floor_us"] == pytest.approx(1.25)
    assert payload["geometric_speedup"] == pytest.approx(2.67)

    obs = CaseObservationRecord(
        run_id="run-v54",
        operator="synthetic_v54",
        regime_tag="main",
        cand_sha="abc123",
        eval_tier="l2_pass",
        case_num=1,
        measured_us=15.0,
        accuracy_passed=True,
        is_real_npu=True,
    )
    assert obs.to_dict()["is_real_npu"] is True
    assert "add_rms_norm_dynamic_quant" in DEFAULT_REGIME_SPECS or len(DEFAULT_REGIME_SPECS) > 0
