"""Consolidated tests for RoundContextBuilder, L3L4RecoveryCoordinator, and Flywheel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.agent.campaign.campaign_runner import CampaignConfig, CampaignRoundRecord
from tilefoundry_ascend.agent.campaign.flywheel_coordinator import (
    DistilledCheckpoint,
    FlywheelCoordinator,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import DeterministicSearchBoundaryLedger
from tilefoundry_ascend.agent.campaign.helpers import select_active_regime
from tilefoundry_ascend.agent.campaign.l3_l4_recovery import (
    L3L4RecoveryCoordinator,
    PendingL4,
    canonical_device_id,
    case_dim_alignment_flags,
    extract_l3_failed_case_nums,
    parse_device_ids,
    resolve_campaign_device_plan,
    resolve_l4_device_pool,
    source_sha256,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import (
    PendingPromotion,
    classify_l3_resolution,
)
from tilefoundry_ascend.agent.campaign.round_context_builder import (
    ChampionState,
    RoundContextBuilder,
    RoundSynthesisBundle,
)
from tilefoundry_ascend.agent.cost_advisor import create_cost_advisor
from tilefoundry_ascend.agent.gates.gate_funnel import StageAResult
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, SymptomFamily
from tilefoundry_ascend.agent.npu import (
    FAILURE_KIND_CANDIDATE,
    FAILURE_KIND_INFRASTRUCTURE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
    aggregate_surrogate_result,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket, HirRooflineSummary
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


@dataclass(frozen=True)
class _BuilderCfg:
    operator_name: str = "softmax"
    pure_golden: bool = False
    stagnation_escalate_rounds: int = 1
    k_rule_workers: int = 1
    enable_tilesize_prescreen: bool = True
    exploit_effort: str = "medium"
    allow_unreviewed_distilled_rules: bool = False


def _metric(
    case_num: int,
    regime_tag: str,
    cand_us: float,
    *,
    ref_us: float = 30.0,
    t_hw_us: float = 5.0,
) -> AnchorEvalMetrics:
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=True,
        cand_us=cand_us,
        ref_perf_us=ref_us,
        t_hw_us=t_hw_us,
        speedup=ref_us / cand_us,
        roofline_eta=min(1.0, t_hw_us / cand_us),
        eq3_score=ref_us / cand_us,
        paired_rel_mad=0.01,
    )


def _surrogate(cid: str, metrics: tuple[AnchorEvalMetrics, ...]) -> CandidateSurrogateResult:
    return aggregate_surrogate_result(cid, "softmax", metrics)


def _ledger(ii: int = 100) -> object:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=ii,
        pipe_cycles={"AIV0.PIPE_V": 1000, "AIV0.PIPE_MTE2": 800},
        waterlines={},
        gm_roundtrip_bytes=4096,
        grid_blocks=16,
    )


def test_round_context_builder_regime_switch_eta_and_prompt_assembly(tmp_path: Path) -> None:
    """Verify RoundContextBuilder selects regimes, computes same-case eta, and builds prompts."""
    module_src = (
        "def _select_softmax_config(d_dim: int = 256) -> dict[str, object]:\n"
        "    return {'d_dim': d_dim}\n\n"
        "def _build_softmax_func(d_dim: int = 256) -> object:\n"
        "    return d_dim\n\n"
        "def _get_kernel(d_dim: int) -> object:\n"
        "    return _build_softmax_func(d_dim)\n\n"
        "def kernel(x: object) -> object:\n"
        "    return _get_kernel(256)\n"
    )
    spec_a = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("_build_softmax_func",),
        sample_builder_kwargs={"d_dim": 256},
        config_selector_fn="_select_softmax_config",
        editable_symbols=("_select_softmax_config", "_build_softmax_func"),
        representative_anchor_dims={"d_dim": 256},
    )
    spec_b = RegimeTargetSpec(
        regime_tag="large",
        builder_candidates=("_build_softmax_func",),
        sample_builder_kwargs={"d_dim": 4096},
        config_selector_fn="_select_softmax_config",
        editable_symbols=("_select_softmax_config", "_build_softmax_func"),
        representative_anchor_dims={"d_dim": 4096},
    )
    best_res = _surrogate(
        "inc",
        (
            _metric(7, "small", 250.0, ref_us=300.0, t_hw_us=50.0),
            _metric(8, "large", 20.0, ref_us=25.0, t_hw_us=19.0),
        ),
    )
    anchors = (
        AnchorCaseSpec("op/softmax", 7, "small", 300.0, 50.0),
        AnchorCaseSpec("op/softmax", 8, "large", 25.0, 19.0),
    )
    picked = select_active_regime(
        (spec_a, spec_b),
        PerfTargetSpec("softmax", 5.0, 5.0, 0.99),
        None,
        set(),
        best_res,
        anchors,
    )
    assert picked is not None and picked.regime_tag == "small"

    target = AscendTarget("ascend.950pr")
    rule = AscendCausalRule(
        rule_id="R_VEC_01",
        regime_guard="True",
        precondition="True",
        transform_template="Vectorize loop tail.",
        expected_postcondition="True",
        source_provenance="unit_test",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
    )
    boundary = DeterministicSearchBoundaryLedger()
    boundary.record_compile_wall(
        regime_tag="small",
        cand_sha="deadbeef",
        knob_signature="tile=128",
        error_code_norm="UB_SRAM_OVERFLOW",
        ub_bytes=280000,
    )
    boundary.record_spearman_rho("small:aiv_only|gm_ub_stream|affine_simd", 0.05)
    boundary.record_spearman_rho("small:aiv_only|gm_ub_stream|affine_simd", 0.05)

    blueprint = HirBlueprintPacket(
        entry_name="kernel",
        roofline=HirRooflineSummary(
            anchor_dims={"d_dim": 256},
            compute_flops=1_000_000,
            min_gm_bytes=9_999_999,
            arithmetic_intensity=1.0,
            compute_floor_us=1000.0,
            memory_floor_us=500.0,
            ideal_roofline_us=1000.0,
            dominant_bound="compute_bound",
        ),
    )
    ledger = _ledger()
    parent_sha = hashlib.sha256(module_src.encode("utf-8")).hexdigest()
    builder = RoundContextBuilder(
        config=_BuilderCfg(),
        regime_specs=(spec_a, spec_b),
        target=target,
        tracker=BayesianRuleTracker.from_rules([rule]),
        cost_advisor=create_cost_advisor("advisory", operator_name="softmax"),
        regime_solver=RegimePiecewiseSolver(target),
        source_root=tmp_path,
        perf_target=PerfTargetSpec("softmax", 5.0, 5.0, 0.99),
        boundary_ledger=boundary,
        regime_blueprints={"small": blueprint, "large": blueprint},
        run_l0_fn=lambda _s, _r, **_kw: StageAResult(passed=True, prim_func=None, ledger=ledger),
        l1_summary_fn=lambda _s, _r: None,
        collect_msprof_fn=lambda src, *_a: {(source_sha256(src), 7): {"dur_us": 200.0, "aiv_vec_ratio": 0.45}},
        collect_p2_fn=lambda *_a: {"p2_hotspots": [{"source_location": "line 10", "instruction_count": 42}]},
        resolve_op_semantics_fn=lambda: None,
        resolve_cannbench_tasks_root_fn=lambda: None,
        lookup_case_dims_fn=lambda _c, _t: {"d_dim": 256},
    )
    stone_src = "def _select_softmax_config(d_dim: int = 256) -> dict[str, object]:\n    return {'d_dim': d_dim, 'u': 2}\n"
    beam = TwoLaneBeamSearch()
    beam.stepping_stone_beam = [
        BeamCandidate(
            candidate_id="stone_0",
            round_idx=0,
            worker_kind="creative_explore",
            rule_id=None,
            dsl_source=stone_src,
            track="unconstrained",
            ledger=ledger,
            residual=compute_static_dynamic_residual(ledger, {"dur_us": 180.0}),
            latency_us=180.0,
            accuracy_passed=True,
            structural_signature="sig0",
            hypothesis="Unroll factor 2",
        )
    ]
    state = ChampionState(
        incumbent_source=module_src,
        best_result=best_res,
        active_anchors=anchors,
        regime_specs=(spec_a, spec_b),
        stagnation_by_regime={"small": 1, "large": 0},
        beams={"small": beam},
        msprof_by_case={(parent_sha, 7): {"dur_us": 200.0, "aiv_vec_ratio": 0.45}},
        regime_blueprints={"small": blueprint, "large": blueprint},
    )
    bundle = builder.build_round_bundle(state, round_idx=2, regime_spec=spec_a)
    assert isinstance(bundle, RoundSynthesisBundle)
    assert bundle.escalated is True
    assert bundle.parent_lane == "stepping_stone"
    assert bundle.selected_stone_id == "stone_0"
    assert bundle.gate_decision.allow_candidate_pruning is False
    assert bundle.parent_p2_diag is None or isinstance(bundle.parent_p2_diag, dict)
    assert len(bundle.worker_specs) >= 2
    exploit_req = bundle.worker_specs[0][3]
    assert "<read_only_host_tail>" in exploit_req.user_prompt
    assert "VERIFIED PHYSICAL SEARCH BOUNDARIES" in exploit_req.user_prompt
    assert "HARDWARE MSPROF TELEMETRY" in exploit_req.user_prompt
    assert "# target_bottleneck:" in exploit_req.system_prompt


def test_l3_l4_recovery_classification_device_pool_and_pending_promotion(tmp_path: Path) -> None:
    """Verify L3 resolution classification, card pool resolution, and PendingPromotion recovery."""
    assert classify_l3_resolution(L3AcceptanceResult(passed=True)) is L3Resolution.PROMOTED
    assert classify_l3_resolution(
        L3AcceptanceResult(passed=False, failure_kind=FAILURE_KIND_CANDIDATE, execution_terminal=True)
    ) is L3Resolution.CANDIDATE_REJECTED
    assert classify_l3_resolution(
        L3AcceptanceResult(passed=False, failure_kind=FAILURE_KIND_INFRASTRUCTURE, execution_terminal=False)
    ) is L3Resolution.INFRA_DEFERRED

    assert parse_device_ids("0, 01, 2 ") == ("0", "01", "2")
    with pytest.raises(ValueError, match="empty entry"):
        parse_device_ids("0,,2")
    assert canonical_device_id(" 007 ", field="device_id") == "7"
    assert resolve_campaign_device_plan(device_id="", device_ids=(), env_device_ids=("02", "0"), visible_device_ids=("0",)) == ("0", ("0",))
    assert resolve_l4_device_pool(CampaignConfig("op", device_id="07", device_ids=("2", "07", "10"), retry_device_id="2")) == ("7", "2", "10")

    flags = case_dim_alignment_flags({"M": 64, "K": 77, "dtype": "float16"})
    assert flags["has_odd_dim"] is True and flags["odd_or_prime_tail"] is True
    assert extract_l3_failed_case_nums(L3AcceptanceResult(passed=False, error_message="fail case=14 case=15 case=14")) == [14, 15]

    target_path = tmp_path / "kernels.py"
    ckpt_path = tmp_path / "ckpt.json"
    inc_src = "def _build():\n    return 1\n"
    cand_src = "def _build():\n    return 2\n"
    target_path.write_text(cand_src, encoding="utf-8")

    coord = L3L4RecoveryCoordinator(
        operator_name="synthetic_op",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="3",
        device_ids=("3", "5"),
        checkpoint_path=ckpt_path,
        l3_contract_runner=lambda _p, **kw: L3AcceptanceResult(
            passed=(kw.get("device_id") == "5"),
            execution_id="ex_alt",
            run_id="run_alt",
            device_id=str(kw.get("device_id") or "3"),
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_terminal=True,
        ),
        guardrail_cases=[GuardrailCaseSpec("op/synthetic", 14, "r0", 12.0, 2.0)],
        last_guardrail_metrics_by_cid={"c1": {14: _metric(14, "r0", 7.5)}},
        sleep_fn=lambda _s: None,
    )
    cand_res = _surrogate("c1", (_metric(1, "r0", 4.0),))
    base_res = _surrogate("base", (_metric(1, "r0", 9.0),))
    pending = PendingPromotion(
        candidate_source=cand_src,
        candidate_sha256=source_sha256(cand_src),
        target_path=str(target_path),
        round_idx=1,
        regime_tag="r0",
        candidate_result=cand_res,
        candidate_id="c1",
        execution_id="exec_1",
        run_id="run_1",
        device_id="3",
        execution_terminal=False,
    )
    rounds = [
        CampaignRoundRecord(
            round_idx=1,
            regime_tag="r0",
            builder_name="_build",
            parent_lane="exploit",
            creative_effort="medium",
            l0_l1_passed_ids=("c1",),
            l0_l1_failed=(),
            best_round_speedup=2.0,
            best_so_far_speedup=2.0,
            promoted=False,
            l3_resolution=L3Resolution.INFRA_DEFERRED.value,
        )
    ]
    prom_r: list[int] = []
    rb_r: list[int] = [1]
    stag = {"r0": 2}
    new_inc, new_best, new_stag, next_p, blocked = coord.resolve_pending_promotion(
        pending,
        incumbent_source=inc_src,
        best_result=base_res,
        stagnation_by_regime=stag,
        rounds=rounds,
        promoted_rounds=prom_r,
        rolled_back_rounds=rb_r,
        active_anchors=(),
    )
    assert blocked is False and next_p is None
    assert new_inc == cand_src and new_best == cand_res
    assert new_stag["r0"] == 0 and prom_r == [1] and rb_r == []
    assert coord.guardrail_cases[0].incumbent_ref_us == 7.5
    assert rounds[0].l3_resolution == L3Resolution.PROMOTED.value


def test_l3_l4_recovery_pending_l4_card_rotation_and_digest_guard(tmp_path: Path) -> None:
    """Verify PendingL4 card-pool rotation across occupancy faults and digest guard."""
    target_path = tmp_path / "kernels.py"
    ckpt_path = tmp_path / "ckpt.json"
    inc_src = "def _build():\n    return 1\n"
    target_path.write_text(inc_src, encoding="utf-8")

    seen_devs: list[str] = []

    def _l4_run(_p: Path, *, device_id: str = "", **_kw: object) -> L4CannbenchResult:
        seen_devs.append(device_id)
        if device_id == "1":
            return L4CannbenchResult(
                passed=False,
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                blocked_env_category="occupancy",
                device_id="1",
                execution_terminal=True,
                error_message="BLOCKED_ENV occupancy",
            )
        return L4CannbenchResult(passed=True, device_id=device_id, case_count=5, passed_cases=5)

    coord = L3L4RecoveryCoordinator(
        operator_name="synthetic_op",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="1",
        device_ids=("1", "2"),
        checkpoint_path=ckpt_path,
        l4_cannbench_runner=_l4_run,
        sleep_fn=lambda _s: None,
    )
    l4_res = coord.run_final_l4_tracked(incumbent_source=inc_src, prior_stop_reason="goal_met")
    assert l4_res.passed is True
    assert seen_devs == ["1", "2"]
    assert coord.pending_l4 is None

    drifted_pending = PendingL4(
        target_path=str(target_path),
        candidate_relative="kernels.py",
        candidate_sha256="0" * 64,
        execution_id="exec_old",
        device_id="1",
        execution_terminal=False,
    )
    drift_res, still_pending = coord.resolve_pending_l4(drifted_pending, incumbent_source=inc_src)
    assert drift_res.passed is False
    assert "does not match" in drift_res.error_message
    assert still_pending is not None


def test_flywheel_coordinator_trigger_checkpoint_and_distillation(tmp_path: Path) -> None:
    """Verify FlywheelCoordinator 3-sigma gating, checkpoint restore, and distillation."""
    lake_dir = tmp_path / "lake"
    distill_calls: list[dict[str, object]] = []

    class _FakeFlywheel:
        def __init__(self, tracker: BayesianRuleTracker) -> None:
            self._tracker = tracker

        def distill_from_pair(self, **kwargs: object) -> object:
            distill_calls.append(dict(kwargs))
            rule = AscendCausalRule(
                rule_id="R_FLYWHEEL_01",
                regime_guard="True",
                precondition="True",
                transform_template="Optimize tile.",
                expected_postcondition="True",
                source_provenance="flywheel",
            )
            outcome = SimpleNamespace(rule=rule, admitted=True, verified_on_candidate=True, hardware_verified=True)
            return SimpleNamespace(
                admitted_rule_ids=("R_FLYWHEEL_01",),
                distilled_outcomes=(outcome,),
                kept_positive_chunks=(),
                to_dict=lambda: {"admitted_rule_ids": ["R_FLYWHEEL_01"]},
            )

    tracker = BayesianRuleTracker(rules={})
    fw = FlywheelCoordinator(flywheel=_FakeFlywheel(tracker))  # type: ignore[arg-type]
    fw.seed_baseline("main", "BASE = 1\n", 100.0, 1.0)
    assert fw.should_trigger("main", 99.0, 1.0) is False
    assert fw.should_trigger("main", 85.0, 1.0) is True

    spec = RegimeTargetSpec("main", ("_build",), {})
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    def _write_text(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    admitted = fw.run_distillation(
        operator_name="synthetic_op",
        run_id="run_fw",
        tracker=tracker,
        baseline_source="BASE = 1\n",
        winner_source="WIN = 2\n",
        baseline_latency_us=100.0,
        winner_latency_us=80.0,
        sigma_paired_us=1.0,
        func_name="_build",
        regime_spec=spec,
        topology_axes=axes,
        batch_latency_evaluator=lambda _p: {},
        campaign_kb_path=tmp_path / "kb.yaml",
        experience_lake_dir=lake_dir,
        allow_unreviewed=False,
        has_l2_executor=True,
        real_npu_path=True,
        source_sha256_fn=source_sha256,
        write_text_fn=_write_text,
    )
    assert admitted == ["R_FLYWHEEL_01"]
    assert len(distill_calls) == 1
    assert fw.last_distilled_ckpt_by_regime["main"].latency_us == pytest.approx(80.0)

    serialized = fw.serialize_checkpoints()
    fw2 = FlywheelCoordinator(flywheel=None)
    fw2.restore_checkpoints(serialized)
    assert fw2.last_distilled_ckpt_by_regime["main"] == DistilledCheckpoint("WIN = 2\n", 80.0, 1.0)
    fw2.restore_checkpoints(None, raw_latencies={"main": 50.0}, fallback_source="FB")
    assert fw2.last_distilled_ckpt_by_regime["main"].source == "FB"
    edges = (lake_dir / "kb" / "evidence" / "winning_edges.jsonl").read_text(encoding="utf-8")
    assert "R_FLYWHEEL_01" in edges
    assert json.loads(edges.splitlines()[0])["winner_latency_us"] == pytest.approx(80.0)
