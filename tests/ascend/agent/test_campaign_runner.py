"""Offline unit tests for WP8 surrogate anchors, AST isolation, and campaign state machine."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from tilefoundry_ascend.agent import (
    AnchorCaseSpec,
    BayesianRuleTracker,
    CampaignConfig,
    CampaignRoundRecord,
    CandidateSurrogateResult,
    FlywheelDistiller,
    GateFunnel,
    L3AcceptanceResult,
    L4CannbenchResult,
    MockLLMBackend,
    PerfTargetCampaignRunner,
    PerfTargetSpec,
    RegimeTargetSpec,
    append_dynamic_anchor,
    get_default_anchors,
    get_default_perf_target,
)
from tilefoundry_ascend.agent.campaign import (
    aggregate_surrogate_result,
    build_anchor_metrics,
    build_arg_parser,
    compute_case_eq3_score,
    extract_regime_snippet,
    patch_regime_into_module,
    surrogate_cannbench,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import DEFAULT_REGIME_SPECS
from tilefoundry_ascend.agent.campaign.campaign_runner import _extract_regime_abi_context
from tilefoundry_ascend.agent.orchestrator import StageAResult, StageBResult, TwoLaneBeamSearch
from tilefoundry_ascend.agent.router import extract_structural_features, route_execution_track
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger

_MULTI_REGIME_MODULE = """
import tilelang.ascend.language as T

_KERNEL_CACHE = {"kept": True}

def _build_regime_a(tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main

def _build_regime_b(tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((64, 64), "bfloat16"), Y: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((64, 64), "bfloat16")
            T.copy(X[0:64, 0:64], b_ub)
            T.copy(b_ub, Y[0:64, 0:64])
    return main

def entrypoint(x):
    return _KERNEL_CACHE, _build_regime_a(), _build_regime_b(), x
"""

_IMPROVED_REGIME_A = """
def _build_regime_a(tile_m=4):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(2):
            a_ub = T.alloc_shared((64, 128), "bfloat16")
            T.copy(X[0:64, 0:128], a_ub)
            T.copy(a_ub, Y[0:64, 0:128])
    return main
"""

_SSBUF_REGIME_A = """
def _build_regime_a(tile_m=2):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.MixedKernel(1) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            ss[sid] = T.Cast("uint32", 0)
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main
"""

_IMPROVED_REGIME_B = """
def _build_regime_b(tile_m=4):
    @T.prim_func
    def main(X: T.Tensor((64, 64), "bfloat16"), Y: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(2):
            b_ub = T.alloc_shared((32, 64), "bfloat16")
            T.copy(X[0:32, 0:64], b_ub)
            T.copy(b_ub, Y[0:32, 0:64])
    return main
"""


def _make_stage_a(source: str) -> StageAResult:
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=80 if "T.Kernel(2)" in source else 120,
        pipe_cycles={"AIV0.PIPE_V": 80 if "T.Kernel(2)" in source else 120},
        waterlines={},
    )
    feats = extract_structural_features(None, ledger=ledger)
    route = route_execution_track(feats)
    return StageAResult(
        passed=True,
        prim_func=None,
        ledger=ledger,
        features=feats,
        route=route,
    )


def _make_surrogate_result(
    cid: str,
    op_name: str,
    anchors: tuple[AnchorCaseSpec, ...],
    latencies_by_case: dict[int, float],
    *,
    all_ok: bool = True,
) -> CandidateSurrogateResult:
    metrics = [
        build_anchor_metrics(
            spec,
            accuracy_passed=all_ok,
            cand_us=latencies_by_case[spec.case_num],
        )
        for spec in anchors
    ]
    return aggregate_surrogate_result(cid, op_name, metrics)


def _setup_mock_cann_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "mock_cannbench"
    meta_dir = repo / "tasks" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "level4": {
            "attention_residual": {
                "5": {"baseline_perf_us": 17.166, "t_hw_us": 5.722},
                "9": {"baseline_perf_us": 190.669, "t_hw_us": 63.5563},
            },
            "mla": {
                "14": {"baseline_perf_us": 61.485, "t_hw_us": 20.4949},
            },
        }
    }
    (meta_dir / "950pr.json").write_text(json.dumps(meta_payload), encoding="utf-8")
    return repo


def test_anchor_registry_dynamic_append_and_eq3_scoring(tmp_path: Path) -> None:
    """Verify anchor registry defaults, Eq.3 scoring, dynamic append, and AST patching."""
    ar_anchors = get_default_anchors("attention_residual")
    mla_anchors = get_default_anchors("mla")
    assert [a.case_num for a in ar_anchors] == [2, 4, 14, 16, 17]
    assert [a.case_num for a in mla_anchors] == [1, 6, 7, 11, 13]

    ar_target = get_default_perf_target("attention_residual")
    mla_target = get_default_perf_target("mla")
    assert ar_target.target_avg_speedup == 2.85
    assert ar_target.target_min_case_speedup == 1.60
    assert ar_target.target_roofline_eta == 0.90
    assert ar_target.target_surrogate_score == 97.5
    assert ar_target.min_optimization_rounds == 3
    assert mla_target.target_avg_speedup == 2.50
    assert mla_target.target_min_case_speedup == 1.35
    assert mla_target.target_roofline_eta == 0.85
    assert mla_target.target_surrogate_score == 92.0
    assert mla_target.min_optimization_rounds == 3
    assert mla_target.per_anchor_max_us == {7: 8.0, 13: 95.0}
    assert PerfTargetSpec.from_dict(ar_target.to_dict()) == ar_target
    assert PerfTargetSpec.from_dict(mla_target.to_dict()) == mla_target

    eq3_ar2 = compute_case_eq3_score(1.5, 2.87, 0.5)
    assert abs(eq3_ar2 - 0.2967359) < 1e-5

    mock_repo = _setup_mock_cann_repo(tmp_path)
    appended = append_dynamic_anchor(
        ar_anchors,
        operator_name="attention_residual",
        case_num=5,
        cannbench_repo=mock_repo,
    )
    assert [a.case_num for a in appended] == [2, 4, 14, 16, 17, 5]
    assert appended[-1].ref_perf_us == 17.166
    assert appended[-1].t_hw_us == 5.722

    same = append_dynamic_anchor(
        appended,
        operator_name="attention_residual",
        case_num=5,
        cannbench_repo=mock_repo,
    )
    assert len(same) == 6

    spec = RegimeTargetSpec("regime_a", ("_build_regime_a",), {})
    fname, snippet = extract_regime_snippet(_MULTI_REGIME_MODULE, spec)
    assert fname == "_build_regime_a"
    assert "def _build_regime_a" in snippet
    assert "_build_regime_b" not in snippet

    patched = patch_regime_into_module(_MULTI_REGIME_MODULE, fname, _IMPROVED_REGIME_A)
    assert "_KERNEL_CACHE" in patched
    assert "def _build_regime_b" in patched
    assert "with T.Kernel(2):" in patched


def test_campaign_early_stop_on_goal_met_and_roofline_saturated(tmp_path: Path) -> None:
    """Verify early termination on goal_met (with flywheel admission) and roofline_saturated."""
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 4.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_b", 20.0, 8.0),
    )
    regimes = (
        RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),
        RegimeTargetSpec("regime_b", ("_build_regime_b",), {}),
    )
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A, _IMPROVED_REGIME_A],
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Double core concurrency",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ],
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            if "T.Kernel(2)" in src:
                out[cid] = _make_surrogate_result(
                    cid, "attention_residual", tuple(active_anchors), {2: 4.5, 4: 9.0}
                )
            else:
                out[cid] = _make_surrogate_result(
                    cid, "attention_residual", tuple(active_anchors), {2: 12.0, 4: 24.0}
                )
        return out

    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        min_edit_gain=0.0,
        min_probe_speedup=1.0,
    )
    kb_out = tmp_path / "campaign_rules.yaml"
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=6),
        target_module_path=target_path,
        perf_target=PerfTargetSpec(
            "attention_residual", 2.0, 1.5, 0.88, min_optimization_rounds=1
        ),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        tracker=tracker,
        flywheel_distiller=flywheel,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_goal.json",
        campaign_kb_path=kb_out,
    )
    report_goal = asyncio.run(runner.run())
    assert report_goal.stop_reason == "goal_met"
    assert report_goal.total_rounds == 1
    assert report_goal.promoted_rounds == (1,)
    assert "with T.Kernel(2):" in target_path.read_text(encoding="utf-8")

    target_sat = tmp_path / "kernels_sat.py"
    target_sat.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    backend_sat = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A, _IMPROVED_REGIME_A],
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Saturate Roofline eta",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ],
        }
    )

    def _l2_sat(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            if "T.Kernel(2)" in src:
                out[cid] = _make_surrogate_result(
                    cid, "attention_residual", tuple(active_anchors), {2: 5.0, 4: 10.0}
                )
            else:
                out[cid] = _make_surrogate_result(
                    cid, "attention_residual", tuple(active_anchors), {2: 12.0, 4: 24.0}
                )
        return out

    runner_sat = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=6),
        target_module_path=target_sat,
        perf_target=PerfTargetSpec(
            "attention_residual", 5.0, 4.0, 0.78, min_optimization_rounds=1
        ),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend_sat,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_sat,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_sat.json",
    )
    report_sat = asyncio.run(runner_sat.run())
    assert report_sat.stop_reason == "roofline_saturated"
    assert report_sat.total_rounds == 1


def test_campaign_patience_escalation_xhigh_stepping_stone_and_regime_rotation(
    tmp_path: Path,
) -> None:
    """Verify escalation to stepping_stone + xhigh at stagnation 2 and freeze at stagnation 4."""
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_b", 20.0, 6.0),
    )
    regimes = (
        RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),
        RegimeTargetSpec("regime_b", ("_build_regime_b",), {}),
    )
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    creative_a_payload = json.dumps(
        {
            "physical_hypothesis": "Novel SSBUF path",
            "target_local_metric": "sync_stall",
            "dsl_source": _SSBUF_REGIME_A,
        }
    )
    creative_b_payload = json.dumps(
        {
            "physical_hypothesis": "Regime B probe",
            "target_local_metric": "aiv_vec",
            "dsl_source": _IMPROVED_REGIME_B,
        }
    )

    class _RegimeDispatchBackend:
        def __init__(self) -> None:
            self.calls = []

        async def generate(self, req, *, effort_override=None):
            _ = effort_override
            self.calls.append(req)
            is_a = "regime_a:" in req.user_prompt
            if req.role == "explore_creative":
                return creative_a_payload if is_a else creative_b_payload
            return _IMPROVED_REGIME_A if is_a else _IMPROVED_REGIME_B

    backend = _RegimeDispatchBackend()

    def _l2_stagnant(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            if "T.alloc_ss" in src:
                out[cid] = _make_surrogate_result(
                    cid, "attention_residual", tuple(active_anchors), {2: 14.0, 4: 28.0}
                )
            else:
                out[cid] = _make_surrogate_result(
                    cid, "attention_residual", tuple(active_anchors), {2: 12.0, 4: 24.0}
                )
        return out

    tracker = BayesianRuleTracker.from_seed_yaml()
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=12,
            stagnation_escalate_rounds=2,
            stagnation_freeze_rounds=4,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 2.0, 1.5, 0.85),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_stagnant,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_rot.json",
    )
    report = asyncio.run(runner.run())
    assert report.stop_reason == "all_regimes_converged"
    assert report.total_rounds == 8
    assert report.frozen_regimes == ("regime_a", "regime_b")
    assert report.escalated_rounds == (5, 6, 7, 8)
    assert [r.regime_tag for r in report.rounds] == ["regime_a", "regime_b"] * 4
    assert [r.creative_effort for r in report.rounds] == ["high"] * 8
    assert report.rounds[4].parent_lane == "stepping_stone"
    assert report.rounds[4].creative_effort == "high"

    creative_calls = [c for c in backend.calls if c.role == "explore_creative"]
    assert len(creative_calls) >= 8
    assert all(c.effort == "high" for c in creative_calls)
    assert any(
        "regime_a:" in c.user_prompt and "T.alloc_ss" in c.user_prompt
        for c in creative_calls[4:]
    )
    for call in creative_calls:
        joined = call.system_prompt + "\n" + call.user_prompt
        assert "### NEGATIVE CONSTRAINTS (ALREADY EXPLORED BY PARALLEL WORKERS)" in joined
        for rule in tracker.active_rules():
            assert rule.rule_id not in joined


def test_campaign_l3_contract_failure_atomic_rollback_and_budget_exhausted(
    tmp_path: Path,
) -> None:
    """Verify failed L3 atomically rolls back target module and stops on max_rounds cap."""
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 3.0),)
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),)
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 4,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Fast but breaks non-anchor case",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 2,
        }
    )

    def _l2_fast(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat = 5.0 if "T.Kernel(2)" in src else 12.0
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), {2: lat}
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=2),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_fast,
        l3_contract_runner=lambda _p: L3AcceptanceResult(
            passed=False, error_message="case 20 precision mismatch"
        ),
        checkpoint_path=tmp_path / "ckpt_rollback.json",
    )
    report = asyncio.run(runner.run())
    assert report.stop_reason == "budget_exhausted"
    assert report.rolled_back_rounds == (1, 2)
    assert report.promoted_rounds == ()
    assert target_path.read_text(encoding="utf-8") == _MULTI_REGIME_MODULE


def test_campaign_atomic_checkpoint_and_exact_resume_plus_l4_catch(
    tmp_path: Path,
) -> None:
    """Verify exact checkpoint resume across restarts and L4 non-anchor catch round."""
    mock_repo = _setup_mock_cann_repo(tmp_path)
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 3.0),)
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),)
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    ckpt_path = tmp_path / "campaign_ckpt.json"

    backend1 = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 4,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Round 1 speedup",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 2,
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat_map = {}
            for spec in active_anchors:
                lat_map[spec.case_num] = 7.0 if "T.Kernel(2)" in src else 12.0
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), lat_map
            )
        return out

    runner1 = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=2),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend1,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
        cannbench_repo=mock_repo,
    )
    rep1 = asyncio.run(runner1.run(resume=False))
    assert rep1.total_rounds == 2
    assert rep1.promoted_rounds == (1,)
    saved_incumbent = target_path.read_text(encoding="utf-8")
    assert "with T.Kernel(2):" in saved_incumbent

    target_path.write_text("corrupted_before_crash_recovery", encoding="utf-8")
    backend2 = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 6,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Resume + L4 tail repair",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 6,
        }
    )
    l4_calls = [0]

    def _l4_runner(_path: Path) -> L4CannbenchResult:
        l4_calls[0] += 1
        if l4_calls[0] == 1:
            return L4CannbenchResult(
                passed=True,
                avg_speedup=1.9,
                score=85.0,
                per_case_speedups={2: 1.5, 5: 0.80},
                per_case_elapsed_us={2: 6.6, 5: 21.4},
            )
        return L4CannbenchResult(
            passed=True,
            avg_speedup=2.05,
            score=88.0,
            per_case_speedups={2: 1.5, 5: 1.15},
            per_case_elapsed_us={2: 6.6, 5: 14.9},
        )

    runner2 = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=4,
            run_final_l4=True,
            l4_regression_ratio=0.95,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend2,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        l4_cannbench_runner=_l4_runner,
        checkpoint_path=ckpt_path,
        cannbench_repo=mock_repo,
    )
    rep2 = asyncio.run(runner2.run(resume=True))
    assert target_path.read_text(encoding="utf-8") == saved_incumbent
    assert [r.round_idx for r in rep2.rounds] == [1, 2, 3, 4, 5]
    assert len(backend2.calls) == 9
    assert l4_calls[0] == 2
    assert [a.case_num for a in rep2.active_anchors] == [2, 5]
    assert rep2.l4_result is not None and rep2.l4_result.per_case_speedups[5] == 1.15


def test_fp3_campaign_beam_persistence_roofline_eta_and_transport_exemption(
    tmp_path: Path,
) -> None:
    """Verify --resume restores stepping_stone_beam, passes roofline_eta, and exempts 503."""
    from tilefoundry_ascend.agent.orchestrator import TwoLaneBeamSearch  # noqa: PLC0415

    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 4.0),)
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {"bm": 64, "bn": 64, "bk": 64}),)
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt_beam.json"

    backend1 = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 2,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Novel SSBUF handoff",
                        "target_local_metric": "sync_stall",
                        "dsl_source": _SSBUF_REGIME_A,
                    }
                )
            ]
            * 2,
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat = 15.0 if "T.alloc_ss" in src else 12.0
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), {2: lat}
            )
        return out

    beam1 = TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1)
    runner1 = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            stagnation_escalate_rounds=1,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend1,
        beam=beam1,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
    )
    asyncio.run(runner1.run(resume=False))
    assert any("T.alloc_ss" in c.structural_signature for c in beam1.stepping_stone_beam)
    expected_gap = round(
        beam1.exploit_beam[0].residual.predicted_bottleneck_util - 0.3333333333333333,
        6,
    )
    assert abs(beam1.exploit_beam[0].residual.unexplained_gap_ratio - max(0.0, expected_gap)) < 1e-4

    ckpt_payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert "beam_state" in ckpt_payload
    prior_trials = {rid: st.trials for rid, st in runner1._tracker.posteriors.items()}

    class _FailingBackend:
        def __init__(self) -> None:
            self.calls = []

        async def generate(self, req, *, effort_override=None):
            self.calls.append(req)
            raise RuntimeError("503 stream idle timeout")

    fail_tracker = BayesianRuleTracker.from_seed_yaml()
    fail_backend = _FailingBackend()
    beam2 = TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1)
    runner2 = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=2,
            stagnation_escalate_rounds=1,
            enable_tilesize_prescreen=True,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=fail_backend,
        tracker=fail_tracker,
        beam=beam2,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
    )
    asyncio.run(runner2.run(resume=True))
    assert any("T.alloc_ss" in c.structural_signature for c in runner2._beam.stepping_stone_beam)
    assert (
        ("T.MixedKernel", "T.alloc_ss") in runner2._beam.exploited_signatures
        or ("T.MixedKernel", "T.alloc_ss", "track:developer") in runner2._beam.exploited_signatures
    )
    assert any("T.alloc_ss" in c.user_prompt for c in fail_backend.calls)
    assert any("### TILE PRESCREEN GUIDANCE" in c.user_prompt for c in fail_backend.calls)
    for rid, stats in fail_tracker.posteriors.items():
        assert stats.trials == prior_trials[rid]


def test_np4_default_regime_specs_hir_dims_and_navigation_packets() -> None:
    """Verify default regime representative_hir_dims compile into valid HIR/TIR packets."""
    from tilefoundry_ascend.agent.campaign import get_default_regime_specs  # noqa: PLC0415
    from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (  # noqa: PLC0415
        compile_hir_blueprint,
    )
    from tilefoundry_ascend.ascend.target import AscendTarget  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[3]
    target = AscendTarget("ascend.950pr")

    for op_name in ("attention_residual", "mla"):
        op_dir = repo_root / "examples" / "ascend950pr_ops" / op_name
        model_src = (op_dir / "model.py").read_text(encoding="utf-8")
        sched_src = (op_dir / "scheduled.py").read_text(encoding="utf-8")
        specs = get_default_regime_specs(op_name)
        assert len(specs) == 3
        for spec in specs:
            assert spec.representative_hir_dims
            roundtrip = RegimeTargetSpec.from_dict(spec.to_dict())
            assert roundtrip.representative_hir_dims == spec.representative_hir_dims
            pkt = compile_hir_blueprint(
                model_src,
                scheduled_hir_source=sched_src,
                anchor_dims=spec.representative_hir_dims,
                target=target,
            )
            assert pkt.roofline.ideal_roofline_us > 0.0
            assert pkt.tir_script is not None
            assert pkt.tir_ledger is not None
            assert pkt.tir_ledger.is_legal is True
            assert pkt.lowering_diagnostics == ()


def test_np4_campaign_autodiscovery_nav_injection_router_off_and_17_rules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify sibling HIR/TIR autodiscovery, zero capability_gate calls, and R01-R17."""
    import tilefoundry_ascend.agent.router.capability_gate as cap_gate  # noqa: PLC0415
    from tilefoundry_ascend.agent.kb import build_predicate_context  # noqa: PLC0415
    from tilefoundry_ascend.agent.orchestrator import (  # noqa: PLC0415
        compute_static_dynamic_residual,
    )
    from tilefoundry_ascend.ascend.passes import build_scope_waterline  # noqa: PLC0415

    router_calls: list[str] = []

    def _forbid_extract(*_args, **_kwargs):
        router_calls.append("extract")
        raise AssertionError("extract_structural_features must not be called")

    def _forbid_route(*_args, **_kwargs):
        router_calls.append("route")
        raise AssertionError("route_execution_track must not be called")

    monkeypatch.setattr(cap_gate, "extract_structural_features", _forbid_extract)
    monkeypatch.setattr(cap_gate, "route_execution_track", _forbid_route)

    repo_root = Path(__file__).resolve().parents[3]
    ar_dir = repo_root / "examples" / "ascend950pr_ops" / "attention_residual"
    (tmp_path / "model.py").write_text(
        (ar_dir / "model.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "scheduled.py").write_text(
        (ar_dir / "scheduled.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),)
    regimes = (
        RegimeTargetSpec(
            "small_decode",
            ("_build_regime_a",),
            {},
            description="Compact decode regime.",
            representative_hir_dims={"N": 2, "B": 1, "S": 1, "D": 64},
        ),
    )

    disjoint_ssbuf_dsl = """
def _build_regime_a(tile_m=2):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.MixedKernel(1, sids=2) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            a_ub = T.alloc_shared((64, 128), "bfloat16")
            ss[sid] = T.Cast("uint32", 0)
            T.copy(X[sid * 64 : (sid + 1) * 64, 0:128], a_ub)
            T.copy(a_ub, Y[sid * 64 : (sid + 1) * 64, 0:128])
    return main
"""
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 4,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Novel SSBUF path",
                        "target_local_metric": "sync_stall",
                        "dsl_source": disjoint_ssbuf_dsl,
                    }
                )
            ]
            * 4,
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat = 15.0 if "T.alloc_ss" in src else 12.0
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), {2: lat}
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=2),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_np4.json",
    )

    assert runner._authored_hir_path == (tmp_path / "model.py").resolve()
    assert runner._scheduled_hir_path == (tmp_path / "scheduled.py").resolve()
    assert "small_decode" in runner._regime_blueprints

    active_by_id = {rule.rule_id: rule for rule in runner._tracker.active_rules()}
    assert len(active_by_id) >= 33
    expected_new_rules = (
        "R13_SPLIT_KV_WAVE_RECOVERY",
        "R14_GQA_HEAD_M_FOLDING",
        "R15_ASYMMETRIC_1C2V_STAGE_FISSION",
        "R16_PADDED_PITCH_TAIL_MASKING",
        "R17_RMEM_ACCUM_AND_256KB_UB_UNLOCK",
        "R18_WAVE_QUANTIZED_BURST_TILING",
        "R19_STATIC_PINGPONG_UNROLL_NO_ALIAS",
        "R20_LOOP_FISSION_REDUCE_SFU_HIDING",
        "R21_SUBROW_DEPTH_OR_WIDTH_CORE_SHARDING",
    )
    for rid in expected_new_rules:
        assert rid in active_by_id

    led_r13 = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=64,
        pipe_cycles={"AIV0.PIPE_V": 64},
        waterlines={},
        grid_blocks=4,
        wave_occupancy=4.0 / 56.0,
    )
    assert active_by_id["R13_SPLIT_KV_WAVE_RECOVERY"].matches(
        build_predicate_context(
            led_r13,
            features=None,
            residual=compute_static_dynamic_residual(led_r13, {"dur_us": 10.0}),
        )
    )

    led_r14 = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=64,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
        waterlines={},
        effective_cube_m=4,
    )
    assert active_by_id["R14_GQA_HEAD_M_FOLDING"].matches(
        build_predicate_context(
            led_r14,
            features=None,
            residual=compute_static_dynamic_residual(led_r14, {"dur_us": 10.0}),
        )
    )

    ub_wl = build_scope_waterline(
        scope="UB",
        capacity_bytes=196608,
        explicit_buffers={"buf": 180224},
        implicit_buffers={},
    )
    led_r15_r17 = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=64,
        pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 64, "AIV1.PIPE_V": 64},
        waterlines={"UB": ub_wl},
        unaligned_dma_padding_bytes=32,
        in_loop_ub_reload_stores=2,
        uses_unreserved_ubuf=False,
    )
    ctx_r15_r17 = build_predicate_context(
        led_r15_r17,
        features=None,
        residual=compute_static_dynamic_residual(led_r15_r17, {"dur_us": 10.0}),
    )
    assert active_by_id["R15_ASYMMETRIC_1C2V_STAGE_FISSION"].matches(ctx_r15_r17)
    assert active_by_id["R16_PADDED_PITCH_TAIL_MASKING"].matches(ctx_r15_r17)
    assert active_by_id["R17_RMEM_ACCUM_AND_256KB_UB_UNLOCK"].matches(ctx_r15_r17)

    report = asyncio.run(runner.run(resume=False))
    assert report.total_rounds == 2
    assert router_calls == []
    assert backend.calls

    roles_seen = {req.role for req in backend.calls}
    assert "exploit_rule" in roles_seen
    assert "explore_creative" in roles_seen
    worker_reqs = [
        req for req in backend.calls if req.role in ("exploit_rule", "explore_creative")
    ]
    assert worker_reqs
    for req in worker_reqs:
        assert "HIR ANALYTICAL ROOFLINE" in req.user_prompt
        assert "LOWERED ASCEND TIR BLUEPRINT" in req.user_prompt
        assert "TIR BLUEPRINT VS CURRENT TILELANG LEDGER DELTA" in req.user_prompt
        assert "### TILELANG ASCEND PRIMITIVE REFERENCE" in req.user_prompt
        assert "### EXECUTION TRACK" not in req.user_prompt
        assert "### TARGET TRACK" not in req.user_prompt

    assert runner._beam.exploit_beam
    assert all(c.track == "unconstrained" for c in runner._beam.exploit_beam)
    assert all(c.features is None for c in runner._beam.exploit_beam)


def test_no_round0_premature_stop_and_disable_early_goal_stop(tmp_path: Path) -> None:
    """Verify baseline meeting target never stops at Round 0 and disable flag runs max_rounds."""
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 4.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_b", 20.0, 8.0),
    )
    regimes = (
        RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),
        RegimeTargetSpec("regime_b", ("_build_regime_b",), {}),
    )
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    def _l2_already_fast(cands, active_anchors):
        return {
            cid: _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), {2: 4.1, 4: 8.2}
            )
            for cid, _mod_path in cands
        }

    strong_target = PerfTargetSpec(
        "attention_residual",
        2.0,
        1.5,
        0.90,
        target_surrogate_score=95.0,
        min_optimization_rounds=3,
    )
    baseline_res = _l2_already_fast([("base", target_path)], anchors)["base"]
    assert baseline_res.meets_perf_target(strong_target) is True
    assert baseline_res.is_roofline_saturated(strong_target) is True

    backend_min_rounds = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A, _IMPROVED_REGIME_B] * 4,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Keep optimizing",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 6,
        }
    )
    runner_min_rounds = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=5),
        target_module_path=target_path,
        perf_target=strong_target,
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend_min_rounds,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_already_fast,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_min_rounds.json",
    )
    rep_min = asyncio.run(runner_min_rounds.run())
    assert rep_min.total_rounds == 3
    assert rep_min.stop_reason == "goal_met"

    target_full = tmp_path / "kernels_full.py"
    target_full.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    backend_full = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A, _IMPROVED_REGIME_B] * 6,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Run full max_rounds",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 8,
        }
    )
    runner_full = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=4,
            disable_early_goal_stop=True,
        ),
        target_module_path=target_full,
        perf_target=strong_target,
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend_full,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_already_fast,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_full.json",
    )
    rep_full = asyncio.run(runner_full.run())
    assert rep_full.total_rounds == 4
    assert rep_full.stop_reason == "budget_exhausted"


def test_surrogate_dual_criterion_tail_guard_and_meets_perf_target(tmp_path: Path) -> None:
    """Verify 4-metric target check, tail guard rejection, and dual-criterion L3 promotion."""
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_b", 1000.0, 200.0),
    )
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),)

    strict_spec = PerfTargetSpec(
        "attention_residual",
        target_avg_speedup=1.5,
        target_min_case_speedup=1.2,
        target_roofline_eta=0.5,
        target_surrogate_score=80.0,
        per_anchor_max_us={4: 700.0},
    )
    low_score = _make_surrogate_result("low_score", "attention_residual", anchors, {2: 4.0, 4: 650.0})
    assert low_score.avg_speedup > 1.5
    assert low_score.min_case_speedup > 1.2
    assert low_score.avg_roofline_eta < 0.5 or low_score.surrogate_score < 80.0
    assert low_score.meets_perf_target(strict_spec) is False

    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    small_inflate_dsl = _IMPROVED_REGIME_A
    bottleneck_fix_dsl = _SSBUF_REGIME_A

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [
                small_inflate_dsl,
                small_inflate_dsl,
                bottleneck_fix_dsl,
                bottleneck_fix_dsl,
            ],
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Inflate small anchor only",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": small_inflate_dsl,
                    }
                ),
                json.dumps(
                    {
                        "physical_hypothesis": "Fix large bottleneck anchor",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": bottleneck_fix_dsl,
                    }
                ),
            ],
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            if "T.alloc_ss" in src:
                lats = {2: 10.0, 4: 500.0}
            elif "T.Kernel(2)" in src:
                lats = {2: 2.0, 4: 1050.0}
            else:
                lats = {2: 10.0, 4: 1000.0}
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), lats
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=2,
            k_rule_workers=0,
            min_surrogate_gain=0.25,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec(
            "attention_residual", 5.0, 5.0, 0.95, target_surrogate_score=99.0
        ),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_dual.json",
    )
    report = asyncio.run(runner.run())
    assert report.promoted_rounds == (2,)
    assert report.rounds[0].promoted is False
    assert report.rounds[1].promoted is True
    assert report.rounds[1].best_so_far_surrogate_score > report.baseline_result.surrogate_score
    assert "T.alloc_ss" in target_path.read_text(encoding="utf-8")

    restored_rec = CampaignRoundRecord.from_dict(
        {
            "round_idx": 1,
            "regime_tag": "regime_a",
            "builder_name": "_build_regime_a",
            "parent_lane": "exploit",
            "creative_effort": "high",
            "l0_l1_passed_ids": ["c0"],
            "l0_l1_failed": [],
            "best_round_speedup": 1.1,
            "best_so_far_speedup": 1.1,
            "promoted": False,
            "rolled_back_l3": False,
            "stagnation_rounds_after": 1,
        }
    )
    assert restored_rec.best_round_surrogate_score == 0.0
    assert restored_rec.best_so_far_surrogate_score == 0.0


def test_campaign_runner_cli_arg_parser() -> None:
    """Verify campaign_runner CLI parser supports disable flag and campaign options."""
    parser = build_arg_parser()
    parsed = parser.parse_args(
        [
            "--operator",
            "mla",
            "--max-rounds",
            "6",
            "--max-wall-clock-min",
            "45",
            "--device-id",
            "2",
            "--disable-early-goal-stop",
            "--min-surrogate-gain",
            "0.5",
            "--checkpoint",
            "ckpt.json",
            "--report-json",
            "report.json",
            "--run-final-l4",
            "--resume",
            "--k-rule-workers",
            "3",
            "--target-module",
            "custom_kernels.py",
            "--source-root",
            ".",
            "--env-file",
            "env.sh",
        ]
    )
    assert parsed.operator == "mla"
    assert parsed.max_rounds == 6
    assert parsed.max_wall_clock_min == 45.0
    assert parsed.device_id == "2"
    assert parsed.disable_early_goal_stop is True
    assert parsed.min_surrogate_gain == 0.5
    assert parsed.checkpoint == "ckpt.json"
    assert parsed.report_json == "report.json"
    assert parsed.run_final_l4 is True
    assert parsed.resume is True
    assert parsed.k_rule_workers == 3
    assert parsed.target_module == "custom_kernels.py"
    assert parsed.source_root == "."
    assert parsed.env_file == "env.sh"

    defaults = parser.parse_args(["--operator", "attention_residual"])
    assert defaults.disable_early_goal_stop is False
    assert defaults.min_surrogate_gain == 0.25
    assert defaults.target_module is None


def test_scoped_multi_symbol_regime_slice_extract_and_patch(tmp_path: Path) -> None:
    """Verify multi-symbol extraction, patching, helper lifecycle, and guardrails."""
    import pytest  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import (  # noqa: PLC0415
        RegimeSpec,
        get_default_regime_specs,
        patch_regime_into_module,
    )

    assert RegimeSpec is RegimeTargetSpec
    repo_root = Path(__file__).resolve().parents[3]
    mla_src = (repo_root / "examples" / "ascend950pr_ops" / "mla" / "kernels.py").read_text(
        encoding="utf-8"
    )
    mla_specs = {spec.regime_tag: spec for spec in get_default_regime_specs("mla")}
    ar_specs = {
        spec.regime_tag: spec for spec in get_default_regime_specs("attention_residual")
    }
    assert len(mla_specs) == 3 and len(ar_specs) == 3
    for spec in (*ar_specs.values(), *mla_specs.values()):
        assert spec.config_selector_fn
        assert spec.editable_symbols
        assert spec.allowed_helper_prefix
        assert RegimeTargetSpec.from_dict(spec.to_dict()) == spec

    split_spec = mla_specs["split_kv_decode"]
    fname, snippet = extract_regime_snippet(mla_src, split_spec)
    assert fname == "_build_mla_split_kv_decode_func"
    assert "def _select_mla_decode_config(" in snippet
    assert "def _build_mla_split_kv_decode_func(" in snippet
    assert "def _build_mla_split_kv_combine_func(" in snippet
    assert "_select_mla_bnsd_config" not in snippet
    assert "_build_mla_prefill_func" not in snippet
    assert "def mla(" not in snippet

    helper_src = "def _mla_decode_helper(x: int) -> int:\n    return x + 1\n\n"
    assert '"num_kv_splits": splits,' in snippet
    updated_snippet = (
        helper_src
        + snippet.replace('"num_kv_splits": splits,', '"num_kv_splits": 4,', 1)
    )
    patched = patch_regime_into_module(mla_src, split_spec, updated_snippet)
    assert "def _mla_decode_helper(" in patched
    assert '"num_kv_splits": 4,' in patched
    _, re_extracted = extract_regime_snippet(patched, split_spec)
    assert "def _mla_decode_helper(" in re_extracted

    cleaned = patch_regime_into_module(patched, split_spec, snippet)
    assert "def _mla_decode_helper(" not in cleaned

    partial_without_combine = patch_regime_into_module(
        mla_src,
        split_spec,
        snippet.split("def _build_mla_split_kv_combine_func")[0],
    )
    assert "def _build_mla_split_kv_combine_func(" in partial_without_combine
    assert "def _build_mla_split_kv_decode_func(" in partial_without_combine

    with pytest.raises(ValueError, match="at least one authorized regime symbol"):
        patch_regime_into_module(mla_src, split_spec, '"""empty candidate"""\n')

    with pytest.raises(ValueError, match="Unauthorized top-level symbol"):
        patch_regime_into_module(
            mla_src,
            split_spec,
            snippet + "\ndef _bad_helper():\n    return 1\n",
        )

    with pytest.raises(ValueError, match="protected symbol"):
        patch_regime_into_module(
            mla_src,
            split_spec,
            snippet + "\ndef mla(*args, **kwargs):\n    return None\n",
        )

    with pytest.raises(ValueError, match="protected symbol"):
        patch_regime_into_module(
            mla_src,
            split_spec,
            snippet + "\n_KERNEL_CACHE = {}\n",
        )

    with pytest.raises(ValueError, match="protected symbol"):
        patch_regime_into_module(
            mla_src,
            split_spec,
            snippet + "\ndef _select_mla_bnsd_config():\n    return {}\n",
        )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="mla", max_rounds=1),
        target_module_path=tmp_path / "mla_kernels.py",
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, anc: {
            cid: _make_surrogate_result(cid, "mla", tuple(anc), {1: 10.0, 6: 20.0, 7: 4.0, 11: 90.0, 13: 20.0})
            for cid, _ in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
    )
    baseline = _make_surrogate_result(
        "base",
        "mla",
        get_default_anchors("mla"),
        {1: 60.0, 6: 200.0, 7: 4.5, 11: 150.0, 13: 60.0},
    )
    rule = runner._tracker.active_rules()[0]
    exp_req = runner._build_exploit_regime_request(
        snippet_source=snippet,
        func_name=fname,
        regime_spec=split_spec,
        rule=rule,
        best_result=baseline,
    )
    cre_req = runner._build_creative_regime_request(
        snippet_source=snippet,
        func_name=fname,
        regime_spec=split_spec,
        best_result=baseline,
        effort="high",
    )
    for req in (exp_req, cre_req):
        assert "_select_mla_decode_config" in req.system_prompt
        assert "_build_mla_split_kv_combine_func" in req.system_prompt
        assert "_mla_decode_*" in req.system_prompt


def test_config_selector_pass_configs_and_kernel_cache_plumbing(monkeypatch) -> None:
    """Verify _select_*_config, PrimFunc.attrs, Stage A 256 KiB UB, and _KERNEL_CACHE."""
    import tilelang  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import resolve_regime_config  # noqa: PLC0415
    from tilefoundry_ascend.agent.orchestrator import run_stage_a_static  # noqa: PLC0415
    from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS  # noqa: PLC0415

    sample_mod = '''
import tilelang.ascend.language as T
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

def _select_ar_small_decode_config(d_dim=64, tl_dtype="float16", schedule_mode="tuned"):
    return {
        "tile_m": 1,
        "pass_configs": {},
        "compile_flags": [],
    }

def _build_ar_small_decode_func(d_dim=64, tl_dtype="float16", tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((16, d_dim), tl_dtype), Y: T.Tensor((16, d_dim), tl_dtype)):
        with T.Kernel(1) as cid:
            ub = T.alloc_shared((tile_m, d_dim), tl_dtype)
            T.copy(X[0:tile_m, 0:d_dim], ub)
            with T.SimdVF():
                m32 = T.simd.pset(32, "PAT_ALL")
                v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                _ = v_zero
            T.copy(ub, Y[0:tile_m, 0:d_dim])
    return main
'''
    spec = RegimeTargetSpec(
        regime_tag="small_decode",
        builder_candidates=("_build_ar_small_decode_func",),
        sample_builder_kwargs={"d_dim": 64, "tl_dtype": "float16", "tile_m": 99},
        config_selector_fn="_select_ar_small_decode_config",
        editable_symbols=("_select_ar_small_decode_config", "_build_ar_small_decode_func"),
        allowed_helper_prefix="_ar_small_decode_",
    )
    res_no_flags = resolve_regime_config(sample_mod, spec)
    assert res_no_flags.builder_kwargs["tile_m"] == 1
    st_no_flags = run_stage_a_static(
        sample_mod,
        factory_name=res_no_flags.builder_name,
        factory_kwargs=res_no_flags.builder_kwargs,
        pass_configs=res_no_flags.pass_configs or None,
        compile_flags=res_no_flags.compile_flags or None,
    )
    assert st_no_flags.passed and st_no_flags.ledger is not None
    assert st_no_flags.ledger.uses_unreserved_ubuf is False
    assert st_no_flags.ledger.waterlines["UB"].capacity_bytes == 248 * 1024

    mod_unlocked = sample_mod.replace(
        '"tile_m": 1,\n        "pass_configs": {},\n        "compile_flags": [],',
        '"tile_m": 4,\n        "pass_configs": {"tl.enable_auto_schedule": False},\n'
        '        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS),',
    )
    res_unlocked = resolve_regime_config(mod_unlocked, spec)
    assert res_unlocked.builder_kwargs["tile_m"] == 4
    assert res_unlocked.compile_flags == PURE_SIMDVF_UBUF_FLAGS
    st_unlocked = run_stage_a_static(
        mod_unlocked,
        factory_name=res_unlocked.builder_name,
        factory_kwargs=res_unlocked.builder_kwargs,
        pass_configs=res_unlocked.pass_configs or None,
        compile_flags=res_unlocked.compile_flags or None,
    )
    assert st_unlocked.passed and st_unlocked.ledger is not None
    assert st_unlocked.ledger.uses_unreserved_ubuf is True
    assert st_unlocked.ledger.waterlines["UB"].capacity_bytes == 262144
    assert (
        st_unlocked.ledger.waterlines["UB"].peak_allocated_bytes
        > st_no_flags.ledger.waterlines["UB"].peak_allocated_bytes
    )

    mod_attr = sample_mod.replace(
        "    return main",
        '    return main.with_attr("tilelang_compile_flags", list(PURE_SIMDVF_UBUF_FLAGS))',
    )
    st_attr = run_stage_a_static(
        mod_attr,
        factory_name="_build_ar_small_decode_func",
        factory_kwargs={"d_dim": 64, "tl_dtype": "float16"},
        config_selector_name="_select_ar_small_decode_config",
    )
    assert st_attr.passed and st_attr.ledger is not None
    assert st_attr.ledger.uses_unreserved_ubuf is True
    assert st_attr.ledger.waterlines["UB"].capacity_bytes == 262144

    from examples.ascend950pr_ops.attention_residual import kernels as ar_kernels  # noqa: PLC0415
    from examples.ascend950pr_ops.mla import kernels as mla_kernels  # noqa: PLC0415

    compile_calls: list[tuple[object, object]] = []

    def _fake_compile(prog, **kwargs):
        compile_calls.append((kwargs.get("pass_configs"), kwargs.get("compile_flags")))
        return object()

    monkeypatch.setattr(tilelang, "compile", _fake_compile)
    ar_kernels._KERNEL_CACHE.clear()
    k1 = ar_kernels._get_kernel(
        2, 256, "bfloat16", 1, "tuned", pass_configs={"k": 1}, compile_flags=["--flag-a"]
    )
    k2 = ar_kernels._get_kernel(
        2, 256, "bfloat16", 1, "tuned", pass_configs={"k": 1}, compile_flags=["--flag-a"]
    )
    k3 = ar_kernels._get_kernel(
        2, 256, "bfloat16", 1, "tuned", pass_configs={"k": 2}, compile_flags=["--flag-a"]
    )
    k4 = ar_kernels._get_kernel(
        2, 256, "bfloat16", 1, "tuned", pass_configs={"k": 1}, compile_flags=["--flag-b"]
    )
    assert k1 is k2
    assert k1 is not k3 and k1 is not k4
    assert len(ar_kernels._KERNEL_CACHE) == 3

    mla_kernels._KERNEL_CACHE.clear()
    mk1 = mla_kernels._get_kernel(
        128, 1, 512, 64, "float16", "BSND", False, 64, True, "tuned",
        pass_configs={"a": 1}, compile_flags=PURE_SIMDVF_UBUF_FLAGS,
    )
    mk2 = mla_kernels._get_kernel(
        128, 1, 512, 64, "float16", "BSND", False, 64, True, "tuned",
        pass_configs={"a": 2}, compile_flags=PURE_SIMDVF_UBUF_FLAGS,
    )
    assert mk1 is not mk2
    assert len(mla_kernels._KERNEL_CACHE) == 2
    assert len(compile_calls) == 5


def test_all_six_production_regimes_extract_resolve_and_pass_stage_a() -> None:
    """Verify all 6 DEFAULT_REGIME_SPECS extract, resolve, and pass Stage A cleanly."""
    from examples.ascend950pr_ops.mla import kernels as mla_kernels  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import (  # noqa: PLC0415
        get_default_regime_specs,
        resolve_regime_config,
    )
    from tilefoundry_ascend.agent.orchestrator import run_stage_a_static  # noqa: PLC0415
    from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc  # noqa: PLC0415
    from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[3]
    module_paths = {
        "attention_residual": (
            repo_root / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
        ),
        "mla": repo_root / "examples" / "ascend950pr_ops" / "mla" / "kernels.py",
    }
    for op_name, path in module_paths.items():
        source = path.read_text(encoding="utf-8")
        for spec in get_default_regime_specs(op_name):
            fname, snippet = extract_regime_snippet(source, spec)
            assert fname in spec.builder_candidates
            for sym in spec.editable_symbols:
                assert sym in snippet
            resolved = resolve_regime_config(source, spec)
            stage_a = run_stage_a_static(
                source,
                factory_name=resolved.builder_name,
                factory_kwargs=resolved.builder_kwargs,
                pass_configs=resolved.pass_configs or None,
                compile_flags=resolved.compile_flags or None,
            )
            assert stage_a.passed, f"{op_name}/{spec.regime_tag}: {stage_a.error_message}"
            assert stage_a.ledger is not None and stage_a.ledger.is_legal
            assert stage_a.ledger.uses_unreserved_ubuf is True
            assert stage_a.ledger.waterlines["UB"].capacity_bytes == 262144

    decode_fn = mla_kernels._build_mla_split_kv_decode_func(num_kv_splits=4)
    combine_fn = mla_kernels._build_mla_split_kv_combine_func(num_kv_splits=4)
    for fn in (decode_fn, combine_fn):
        res = inspect_tilelang_primfunc(fn, compile_flags=list(PURE_SIMDVF_UBUF_FLAGS))
        assert res.ledger.is_legal, res.lowering_error
        assert res.ledger.uses_unreserved_ubuf is True


def test_per_regime_beam_isolation_objective_ranking_and_checkpoint(tmp_path: Path) -> None:
    """Verify isolated per-regime beams, objective_score ranking, and beam_states roundtrip."""
    from tilefoundry_ascend.agent.orchestrator import (  # noqa: PLC0415
        BeamCandidate,
        TwoLaneBeamSearch,
        compute_static_dynamic_residual,
    )

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 10.0, "roofline_eta": 0.5})

    beam = TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1)
    c_low_obj = BeamCandidate(
        candidate_id="low_obj_fast",
        round_idx=1,
        worker_kind="rule_exploit",
        rule_id=None,
        dsl_source=_IMPROVED_REGIME_A,
        track="unconstrained",
        ledger=ledger,
        residual=res,
        latency_us=5.0,
        accuracy_passed=True,
        objective_score=70.0,
    )
    c_high_obj = BeamCandidate(
        candidate_id="high_obj_best",
        round_idx=1,
        worker_kind="rule_exploit",
        rule_id=None,
        dsl_source=_SSBUF_REGIME_A,
        track="unconstrained",
        ledger=ledger,
        residual=res,
        latency_us=8.0,
        accuracy_passed=True,
        objective_score=95.0,
    )
    beam.update([c_low_obj, c_high_obj])
    assert [c.candidate_id for c in beam.exploit_beam] == ["high_obj_best"]
    beam_rt = TwoLaneBeamSearch.from_dict(beam.to_dict())
    assert beam_rt.exploit_beam[0].objective_score == 95.0

    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_b", 20.0, 6.0),
    )
    regimes = (
        RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),
        RegimeTargetSpec("regime_b", ("_build_regime_b",), {}),
    )
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt_multi_beam.json"

    creative_a = json.dumps(
        {
            "physical_hypothesis": "SSBUF on regime A",
            "target_local_metric": "sync_stall",
            "dsl_source": _SSBUF_REGIME_A,
        }
    )
    creative_b = json.dumps(
        {
            "physical_hypothesis": "Kernel(2) on regime B",
            "target_local_metric": "aiv_vec",
            "dsl_source": _IMPROVED_REGIME_B,
        }
    )
    class _IsolationDispatchBackend:
        def __init__(self) -> None:
            self.calls = []

        async def generate(self, req, *, effort_override=None):
            _ = effort_override
            self.calls.append(req)
            is_a = "regime_a:" in req.user_prompt
            if req.role == "explore_creative":
                return creative_a if is_a else creative_b
            return _IMPROVED_REGIME_A if is_a else _IMPROVED_REGIME_B

    backend = _IsolationDispatchBackend()

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat_a = 15.0 if "T.alloc_ss" in src else 12.0
            lat_b = 24.0
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), {2: lat_a, 4: lat_b}
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=8,
            stagnation_escalate_rounds=2,
            stagnation_freeze_rounds=4,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        beam=TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1),
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
    )
    asyncio.run(runner.run(resume=False))

    beam_a = runner.beam_for_regime("regime_a")
    beam_b = runner.beam_for_regime("regime_b")
    assert beam_a is not beam_b
    assert any("T.alloc_ss" in c.dsl_source for c in beam_a.stepping_stone_beam)
    assert all("T.alloc_ss" not in c.dsl_source for c in beam_b.exploit_beam)
    assert all("T.alloc_ss" not in c.dsl_source for c in beam_b.stepping_stone_beam)

    ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert "beam_states" in ckpt_data
    assert set(ckpt_data["beam_states"]) == {"regime_a", "regime_b"}

    resumed = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=8),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
    )
    resumed._restore_from_checkpoint()
    assert any("T.alloc_ss" in c.dsl_source for c in resumed.beam_for_regime("regime_a").stepping_stone_beam)
    assert all(
        "T.alloc_ss" not in c.dsl_source
        for c in resumed.beam_for_regime("regime_b").stepping_stone_beam
    )


def test_campaign_axis_pruning_and_cross_topology_isolation(tmp_path: Path) -> None:
    """Verify round-end 3-axis TopologyAxes pruning never blocks rules in unrelated topologies."""
    from tilefoundry_ascend.ascend.sil.schema import TopologyAxes  # noqa: PLC0415
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),)
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {"bm": 64, "bn": 64, "bk": 64}),)
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    tracker = BayesianRuleTracker.from_seed_yaml()
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 24,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "No gain probe",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 12,
        }
    )

    def _l2_flat(cands, active_anchors):
        return {
            cid: _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), {2: 12.0}
            )
            for cid, _p in cands
        }

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=10,
            k_rule_workers=2,
            stagnation_freeze_rounds=12,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_flat,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_prune.json",
    )
    report = asyncio.run(runner.run(resume=False))
    pruned_any = [rid for r in report.rounds for rid in r.pruned_rule_ids]
    assert pruned_any
    vec_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    cv_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "irregular_simt")
    assert "mr:gm_ub_stream" in tracker.axis_posteriors
    for rid in pruned_any:
        assert tracker.is_pruned(rid, topology_axes=vec_axes) is True
        assert tracker.is_pruned(rid, topology_axes=cv_axes) is False
        assert tracker.is_pruned(rid) is False


def test_resolve_regime_for_case_and_dynamic_l4_regression_routing(tmp_path: Path) -> None:
    """Verify lookup_case_dims, resolve_regime_for_case, and L4 routing to non-zero regime."""
    import yaml  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import (  # noqa: PLC0415
        get_default_regime_specs,
        lookup_case_dims,
        resolve_regime_for_case,
    )

    ar_specs = get_default_regime_specs("attention_residual")
    mla_specs = get_default_regime_specs("mla")
    assert resolve_regime_for_case("attention_residual", {"B": 1, "S": 1, "D": 4096}, ar_specs).regime_tag == "small_decode"
    assert resolve_regime_for_case("attention_residual", {"B": 1, "S": 333, "D": 2047}, ar_specs).regime_tag == "unaligned_d"
    assert resolve_regime_for_case("attention_residual", {"B": 2, "S": 64, "D": 4096}, ar_specs).regime_tag == "aligned_prefill"
    assert resolve_regime_for_case("mla", {"inputLayout": "BNSD", "S": 128}, mla_specs).regime_tag == "bnsd_gather"
    assert resolve_regime_for_case("mla", {"inputLayout": "BSND", "S": 2}, mla_specs).regime_tag == "split_kv_decode"
    assert resolve_regime_for_case("mla", {"inputLayout": "BSND", "S": 128}, mla_specs).regime_tag == "prefill_compute"

    repo = _setup_mock_cann_repo(tmp_path)
    ar_dir = repo / "tasks" / "level4" / "attention_residual"
    ar_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "cases": [
            {
                "case_id": 5,
                "input_shape": [[8, 1, 512, 1024], [1, 512, 1024], [1024], [1024]],
                "attrs": {"alpha": 0.95, "epsilon": 1e-6},
            },
            {
                "case_id": 9,
                "input_shape": [[4, 1, 4096, 2560], [1, 4096, 2560], [2560], [2560]],
                "attrs": {"alpha": 0.5, "epsilon": 1e-6},
            },
        ]
    }
    (ar_dir / "cases.yaml").write_text(yaml.safe_dump(cases_payload), encoding="utf-8")
    dims_5 = lookup_case_dims("attention_residual", 5, cannbench_repo=repo)
    assert dims_5["N"] == 8 and dims_5["B"] == 1 and dims_5["S"] == 512 and dims_5["D"] == 1024

    mla_dir = repo / "tasks" / "level4" / "mla"
    mla_dir.mkdir(parents=True, exist_ok=True)
    mla_cases = {
        "cases": [
            {
                "case_id": 14,
                "input_shape": [
                    [2, 128, 128, 448],
                    [2, 128, 128, 64],
                    [2, 1, 512, 448],
                    [2, 1, 512, 64],
                    [2, 1, 512, 448],
                ],
                "attrs": {"numKVHeads": 1, "inputLayout": "BNSD", "is_causal": True},
            }
        ]
    }
    (mla_dir / "cases.yaml").write_text(yaml.safe_dump(mla_cases), encoding="utf-8")
    dims_14 = lookup_case_dims("mla", 14, cannbench_repo=repo)
    assert dims_14["B"] == 2 and dims_14["N_q"] == 128 and dims_14["S"] == 128
    assert dims_14["D_nope"] == 448 and dims_14["D_rope"] == 64
    assert dims_14["S_kv"] == 512 and dims_14["N_kv"] == 1 and dims_14["inputLayout"] == "BNSD"

    regimes = (
        RegimeTargetSpec("small_decode", ("_build_regime_a",), {}),
        RegimeTargetSpec("aligned_prefill", ("_build_regime_b",), {}),
    )
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),)
    target_path = tmp_path / "kernels_l4.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_B] * 4,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Fix aligned prefill tail",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _IMPROVED_REGIME_B,
                    }
                )
            ]
            * 2,
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat_map = {}
            for a in active_anchors:
                if a.case_num == 5:
                    lat_map[5] = 8.0 if "def _build_regime_b(tile_m=4)" in src else 22.0
                else:
                    lat_map[a.case_num] = 5.0
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), lat_map
            )
        return out

    l4_calls = [0]

    def _l4_runner(_p: Path) -> L4CannbenchResult:
        l4_calls[0] += 1
        if l4_calls[0] == 1:
            return L4CannbenchResult(
                passed=True,
                avg_speedup=1.4,
                score=82.0,
                per_case_speedups={2: 2.0, 5: 0.75},
                per_case_elapsed_us={2: 5.0, 5: 22.8},
            )
        return L4CannbenchResult(
            passed=True,
            avg_speedup=2.1,
            score=91.0,
            per_case_speedups={2: 2.0, 5: 2.1},
            per_case_elapsed_us={2: 5.0, 5: 8.1},
        )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=0,
            run_final_l4=True,
            l4_regression_ratio=0.95,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        l4_cannbench_runner=_l4_runner,
        checkpoint_path=tmp_path / "ckpt_l4_route.json",
        cannbench_repo=repo,
    )
    rep = asyncio.run(runner.run(resume=False))
    assert rep.total_rounds == 1
    assert rep.rounds[0].regime_tag == "aligned_prefill"
    assert rep.rounds[0].builder_name == "_build_regime_b"
    assert rep.active_anchors[-1].case_num == 5
    assert rep.active_anchors[-1].regime_tag == "aligned_prefill"
    assert "def _build_regime_b(tile_m=4)" in target_path.read_text(encoding="utf-8")


def test_for_regime_and_msprof_collector_caching_wakes_r05_and_r11(tmp_path: Path) -> None:
    """Verify strict for_regime filtering and msprof caching feeding R05/R11 and prompts."""
    from tilefoundry_ascend.agent.campaign import bottleneck_anchor_for_regime  # noqa: PLC0415
    from tilefoundry_ascend.ascend.passes import build_scope_waterline  # noqa: PLC0415

    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_a", 20.0, 4.0),
        AnchorCaseSpec("level4/attention_residual", 16, "regime_b", 100.0, 20.0),
    )
    mixed = _make_surrogate_result(
        "mixed", "attention_residual", anchors, {2: 4.0, 4: 18.0, 16: 25.0}
    )
    reg_a = mixed.for_regime("regime_a")
    assert [a.case_num for a in reg_a.anchors] == [2, 4]
    assert reg_a.avg_speedup != mixed.avg_speedup
    bn_a = bottleneck_anchor_for_regime(mixed, "regime_a")
    assert bn_a is not None and bn_a.case_num == 4
    empty_res = mixed.for_regime("nonexistent")
    assert empty_res.anchors == () and empty_res.surrogate_score == 0.0

    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),)
    target_path = tmp_path / "kernels_msprof.py"
    target_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt_msprof.json"

    ub_half = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"a": 131072},
        implicit_buffers={},
    )
    r05_r11_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 64, "AIV1.PIPE_V": 64},
        waterlines={"UB": ub_half},
        dominant_contig_bytes=512,
        has_online_rescale_loop=True,
    )

    def _l0_r05_r11(_src: str, _reg: RegimeTargetSpec) -> StageAResult:
        return StageAResult(passed=True, prim_func=None, ledger=r05_r11_ledger)

    from tilefoundry_ascend.agent.kb import build_predicate_context  # noqa: PLC0415
    from tilefoundry_ascend.agent.orchestrator import (  # noqa: PLC0415
        compute_static_dynamic_residual,
    )

    no_msprof_ctx = build_predicate_context(
        r05_r11_ledger,
        residual=compute_static_dynamic_residual(
            r05_r11_ledger, {"dur_us": 18.0, "roofline_eta": 0.2}
        ),
    )
    probe_tracker = BayesianRuleTracker.from_seed_yaml()
    no_msprof_ids = {r.rule_id for r in probe_tracker.rank_applicable_rules(no_msprof_ctx)}
    assert "R05_SSBUF_RING_METADATA_HANDOFF" not in no_msprof_ids
    assert "R11_L2_NZ_PAD_BANK_CONFLICT_AVOID" not in no_msprof_ids

    msprof_calls: list[list[int]] = []

    def _mock_msprof_collector(op_name, cand_path, target_anchors):
        assert op_name == "attention_residual"
        assert Path(cand_path).is_file()
        msprof_calls.append([int(a.case_num) for a in target_anchors])
        return {
            "regime_a": {
                "aic_mac_ratio": 0.0,
                "aiv_vec_ratio": 0.20,
                "mte2_wait_ratio": 0.30,
                "sync_stall_ratio": 0.60,
                "dur_us": 18.0,
            }
        }

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_IMPROVED_REGIME_A] * 6,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Fix sync stall and bank conflict",
                        "target_local_metric": "sync_stall_ratio",
                        "dsl_source": _IMPROVED_REGIME_A,
                    }
                )
            ]
            * 4,
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat2 = 2.5 if "T.Kernel(2)" in src else 4.0
            lat4 = 6.0 if "T.Kernel(2)" in src else 18.0
            out[cid] = _make_surrogate_result(
                cid,
                "attention_residual",
                tuple(active_anchors),
                {2: lat2, 4: lat4, 16: 25.0},
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=2,
            k_rule_workers=3,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 3.0, 0.95),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l0_checker=_l0_r05_r11,
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=_mock_msprof_collector,
        checkpoint_path=ckpt_path,
    )
    asyncio.run(runner.run(resume=False))

    assert len(msprof_calls) == 2
    assert msprof_calls[0] == [4]
    assert msprof_calls[1] == [4]

    rule_ids_prompted = [
        req.user_prompt for req in backend.calls if req.role == "exploit_rule"
    ]
    assert any("R05_SSBUF_RING_METADATA_HANDOFF" in p for p in rule_ids_prompted)
    assert any("R11_L2_NZ_PAD_BANK_CONFLICT_AVOID" in p for p in rule_ids_prompted)
    assert any("### HARDWARE MSPROF TELEMETRY" in req.user_prompt for req in backend.calls)

    ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert "msprof_cache" in ckpt_data and len(ckpt_data["msprof_cache"]) == 2
    runner_resumed = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=2),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 3.0, 0.95),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        l0_checker=_l0_r05_r11,
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=_mock_msprof_collector,
        checkpoint_path=ckpt_path,
    )
    runner_resumed._restore_from_checkpoint()
    assert len(runner_resumed._msprof_cache) == 2


def test_prewarm_surrogate_tilelang_cache_pure_cpu_without_harness_prepare(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify prewarm compiles via cases.yaml without calling Harness.prepare."""
    import yaml  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import prewarm_surrogate_tilelang_cache  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[3]
    cann_repo = _setup_mock_cann_repo(tmp_path)
    ar_dir = cann_repo / "tasks" / "level4" / "attention_residual"
    ar_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "cases": [
            {
                "case_id": 2,
                "input_shape": [[2, 1, 1, 4096], [1, 1, 4096], [4096], [4096]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            },
            {
                "case_id": 3,
                "input_shape": [[2, 1, 2, 4096], [1, 2, 4096], [4096], [4096]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            },
            {
                "case_id": 4,
                "input_shape": [[8, 1, 16, 1000], [1, 16, 1000], [1000], [1000]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            },
        ]
    }
    (ar_dir / "cases.yaml").write_text(yaml.safe_dump(cases_payload), encoding="utf-8")

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")
    bench_dir = fake_root / "tests" / "ascend" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "cannbench_contract.py").write_text(
        "class Harness:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise RuntimeError('NPU prepare forbidden')\n"
        "    def prepare(self, case):\n"
        "        raise RuntimeError('NPU prepare forbidden')\n",
        encoding="utf-8",
    )

    tl_stub_dir = tmp_path / "tl_stub"
    tl_stub_dir.mkdir(parents=True, exist_ok=True)
    (tl_stub_dir / "tilelang.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    log_path = cache_dir / 'compiled_entries.jsonl'\n"
        "    entry = {\n"
        "        'prog': prog,\n"
        "        'target': target,\n"
        "        'out_idx': out_idx,\n"
        "        'pass_configs': pass_configs,\n"
        "        'compile_flags': compile_flags,\n"
        "    }\n"
        "    with log_path.open('a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps(entry) + '\\n')\n"
        "    return {'cached': prog}\n",
        encoding="utf-8",
    )

    cand_get_kernel = tmp_path / "cand_get_kernel.py"
    cand_get_kernel.write_text(
        "import tilelang\n"
        "def _select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=1):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': ['--flag-a']}\n"
        "def _build_ar_small_decode_func(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        "    return {'kind': 'small', 'n': n_depth, 'd': d_dim, 'dtype': tl_dtype}\n"
        "def _select_ar_unaligned_config(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=4):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': ['--flag-b']}\n"
        "def _build_ar_unaligned_func(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        "    return {'kind': 'unaligned', 'n': n_depth, 'd': d_dim, 'dtype': tl_dtype}\n"
        "def _get_kernel(n_depth, d_dim, tl_dtype, tile_m, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(n_depth, d_dim, tl_dtype, tile_m, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[4], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n"
        "def attention_residual(*args, **kwargs):\n"
        "    raise RuntimeError('Entrypoint execution forbidden on CPU')\n",
        encoding="utf-8",
    )

    cand_generic = tmp_path / "cand_generic.py"
    cand_generic.write_text(
        "def _build_ar_small_decode_func(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        "    return {'kind': 'generic_small', 'n': n_depth, 'd': d_dim}\n"
        "def _build_ar_unaligned_func(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        "    return {'kind': 'generic_unaligned', 'n': n_depth, 'd': d_dim}\n",
        encoding="utf-8",
    )

    tl_cache_dir = tmp_path / "tl_cache"
    ref_cache_dir = tmp_path / "ref_cache"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TILEFOUNDRY_NPU_TILELANG_SOURCE", str(tl_stub_dir))
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SETUP", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_PYTHON", raising=False)

    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 3, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.9, 3.3),
        AnchorCaseSpec("level4/attention_residual", 999, "small_decode", 1.0, 0.5),
    )
    out_dir = prewarm_surrogate_tilelang_cache(
        "attention_residual",
        [("c_gk", cand_get_kernel), ("c_gen", cand_generic)],
        anchors,
        cannbench_repo=cann_repo,
        cache_dir=ref_cache_dir,
        tl_cache_dir=tl_cache_dir,
        source_root=fake_root,
    )
    assert out_dir == tl_cache_dir
    entries = [
        json.loads(line)
        for line in (tl_cache_dir / "compiled_entries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 4
    kinds = [e["prog"]["kind"] for e in entries]
    assert kinds == ["small", "unaligned", "generic_small", "generic_unaligned"]
    assert entries[0]["out_idx"] == [4]
    assert entries[0]["compile_flags"] == ["--flag-a"]

    mla_dir = cann_repo / "tasks" / "level4" / "mla"
    mla_dir.mkdir(parents=True, exist_ok=True)
    mla_cases = {
        "cases": [
            {
                "case_id": 7,
                "input_shape": [
                    [1, 1, 128, 512],
                    [1, 1, 128, 64],
                    [1, 2048, 1, 512],
                    [1, 2048, 1, 64],
                    [1, 2048, 1, 512],
                ],
                "dtype": "float16",
                "attrs": {"numKVHeads": 1, "inputLayout": "BSND", "is_causal": False},
            }
        ]
    }
    (mla_dir / "cases.yaml").write_text(yaml.safe_dump(mla_cases), encoding="utf-8")

    mla_cand = tmp_path / "cand_mla.py"
    mla_cand.write_text(
        "import tilelang\n"
        "def _select_mla_decode_config(b_dim=1, s_dim=1, s_kv=2048, n_q=128, n_kv=1, "
        "d_nope=512, d_rope=64, tl_dtype='float16', input_layout='BSND', "
        "is_causal=False, schedule_mode='tuned'):\n"
        "    return {'bm': 64, 'aligned_group_q': True, 'num_kv_splits': 4, "
        "'pass_configs': {}, 'compile_flags': []}\n"
        "def _build_mla_split_kv_decode_func(n_q=128, n_kv=1, d_nope=512, d_rope=64, "
        "tl_dtype='float16', input_layout='BSND', is_causal=False, bm=64, "
        "aligned_group_q=True, num_kv_splits=1, schedule_mode='tuned'):\n"
        "    return {'kind': 'mla_stage1', 'splits': num_kv_splits}\n"
        "def _get_kernel(n_q, n_kv, d_nope, d_rope, tl_dtype, input_layout, is_causal, "
        "bm, aligned_group_q, schedule_mode, *, builder_fn=None, num_kv_splits=1, "
        "pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(n_q=n_q, n_kv=n_kv, d_nope=d_nope, d_rope=d_rope, "
        "tl_dtype=tl_dtype, input_layout=input_layout, is_causal=is_causal, bm=bm, "
        "aligned_group_q=aligned_group_q, num_kv_splits=num_kv_splits, "
        "schedule_mode=schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=None, "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n"
        "def _get_combine_kernel(d_nope, tl_dtype, num_kv_splits, *, "
        "pass_configs=None, compile_flags=None):\n"
        "    return tilelang.compile({'kind': 'mla_combine', 'splits': num_kv_splits}, "
        "target='ascend', out_idx=[2], pass_configs=pass_configs, compile_flags=compile_flags)\n",
        encoding="utf-8",
    )
    mla_cache_dir = tmp_path / "mla_tl_cache"
    prewarm_surrogate_tilelang_cache(
        "mla",
        [("c_mla", mla_cand)],
        (AnchorCaseSpec("level4/mla", 7, "split_kv_decode", 4.5, 1.5),),
        cannbench_repo=cann_repo,
        cache_dir=ref_cache_dir,
        tl_cache_dir=mla_cache_dir,
        source_root=fake_root,
    )
    mla_entries = [
        json.loads(line)
        for line in (mla_cache_dir / "compiled_entries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [e["prog"] for e in mla_entries] == [
        {"kind": "mla_stage1", "splits": 4},
        {"kind": "mla_combine", "splits": 4},
    ]
    assert mla_entries[1]["out_idx"] == [2]


def test_run_agent_gates_cmd_retries_transient_blocked_env(monkeypatch) -> None:
    responses = [
        subprocess.CompletedProcess(
            args=["agent-gates"],
            returncode=1,
            stdout=json.dumps({"status": "BLOCKED_ENV", "runId": "r1"}),
            stderr="npu-smi info -m timed out",
        ),
        subprocess.CompletedProcess(
            args=["agent-gates"],
            returncode=1,
            stdout=json.dumps({"status": "BLOCKED_ENV", "runId": "r2"}),
            stderr="",
        ),
        subprocess.CompletedProcess(
            args=["agent-gates"],
            returncode=0,
            stdout=json.dumps({"status": "PASS", "runId": "r3", "receiptPath": "/tmp/r3.json"}),
            stderr="",
        ),
    ]
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(cmd, **kwargs):
        _ = kwargs
        calls.append(list(cmd))
        return responses.pop(0)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", fake_run)
    proc, parsed = surrogate_cannbench._run_agent_gates_cmd(
        ["agent-gates", "run"],
        cwd=".",
        env={"PATH": "/usr/bin"},
        timeout_sec=60,
        max_attempts=4,
        backoff_sec=2.0,
        sleep_fn=sleeps.append,
    )
    assert proc.returncode == 0
    assert parsed["status"] == "PASS"
    assert parsed["runId"] == "r3"
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_extract_regime_abi_context_covers_ar_and_mla_signatures() -> None:
    root = Path(__file__).resolve().parents[3]
    ar_src = (
        root / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
    ).read_text(encoding="utf-8")
    ar_spec = next(
        s for s in DEFAULT_REGIME_SPECS["attention_residual"] if s.regime_tag == "small_decode"
    )
    ar_abi = _extract_regime_abi_context(ar_src, ar_spec)
    assert "IMMUTABLE HOST / PRIMFUNC ABI CONTRACT" in ar_abi
    assert "def _build_ar_small_decode_func(" in ar_abi
    assert "@T.prim_func\ndef main(" in ar_abi
    assert "V: T.Tensor((v_rows, d_dim), tl_dtype)" in ar_abi
    assert "alpha_val: T.float32" in ar_abi

    mla_src = (root / "examples" / "ascend950pr_ops" / "mla" / "kernels.py").read_text(
        encoding="utf-8"
    )
    mla_spec = next(s for s in DEFAULT_REGIME_SPECS["mla"] if s.regime_tag == "split_kv_decode")
    mla_abi = _extract_regime_abi_context(mla_src, mla_spec)
    assert "def _build_mla_split_kv_decode_func(" in mla_abi
    assert "def _build_mla_split_kv_combine_func(" in mla_abi
    assert "Q_nope: T.Tensor((q_total_rows, d_nope), tl_dtype)" in mla_abi
    assert "scale_val: T.float32" in mla_abi


def test_is_transient_blocked_env_retries_dependency_flake_not_device_fail():
    """Verify DEPENDENCY_NOT_PASSED triggers retry while real device FAIL does not."""
    import subprocess  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: PLC0415
        _is_transient_blocked_env,
    )

    proc_fail = subprocess.CompletedProcess(args=["agent-gates"], returncode=1, stdout="", stderr="")
    dep_flake = {
        "status": "FAIL",
        "gates": [
            {"gateId": "tilefoundry-npu-cpu", "status": "FAIL", "failures": ["NONZERO_EXIT"]},
            {
                "gateId": "tilefoundry-npu-device",
                "status": "NOT_RUN",
                "failures": ["DEPENDENCY_NOT_PASSED"],
            },
        ],
    }
    assert _is_transient_blocked_env(proc_fail, dep_flake) is True

    real_device_fail = {
        "status": "FAIL",
        "gates": [
            {"gateId": "tilefoundry-npu-cpu", "status": "PASS", "failures": []},
            {"gateId": "tilefoundry-npu-device", "status": "FAIL", "failures": ["NONZERO_EXIT"]},
        ],
    }
    assert _is_transient_blocked_env(proc_fail, real_device_fail) is False


def test_up3_domain_envelope_prompt_and_host_dispatch_cache(tmp_path: Path, monkeypatch) -> None:
    """Verify InputDomainEnvelope prompts, builder_kwargs unpacking, and _DISPATCH_CACHE."""
    import torch  # noqa: PLC0415
    from examples.ascend950pr_ops.attention_residual import kernels as ar_kernels  # noqa: PLC0415
    from examples.ascend950pr_ops.mla import kernels as mla_kernels  # noqa: PLC0415
    from tilefoundry_ascend.agent import (  # noqa: PLC0415
        BayesianRuleTracker,
        CandidateSurrogateResult,
        PerfTargetCampaignRunner,
    )
    from tilefoundry_ascend.agent.campaign import (  # noqa: PLC0415
        DEFAULT_REGIME_SPECS,
        get_input_domain_envelope,
        resolve_regime_config,
    )
    from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
        _FORBIDDEN_SLICE_SYMBOLS,
    )

    ar_env = get_input_domain_envelope("attention_residual")
    mla_env = get_input_domain_envelope("mla")
    assert ar_env is not None and mla_env is not None
    assert "builder_kwargs" in ar_env.format_prompt_block()
    assert "builder_kwargs" in mla_env.format_prompt_block()
    assert "_DISPATCH_CACHE" in _FORBIDDEN_SLICE_SYMBOLS

    dummy_dsl = (
        "def _select_demo_config(n_depth=8):\n"
        "    return {'tile_m': 16, 'builder_kwargs': {'omit_max_shift': True}}\n\n"
        "def _build_demo(tile_m=8, omit_max_shift=False):\n"
        "    return (int(tile_m), bool(omit_max_shift))\n"
    )
    spec = DEFAULT_REGIME_SPECS["attention_residual"][0]
    from dataclasses import replace  # noqa: PLC0415

    demo_spec = replace(
        spec,
        config_selector_fn="_select_demo_config",
        builder_candidates=("_build_demo",),
        editable_symbols=("_select_demo_config", "_build_demo"),
    )
    resolved = resolve_regime_config(dummy_dsl, demo_spec, dims={"N": 8})
    assert resolved.builder_kwargs.get("omit_max_shift") is True

    tracker = BayesianRuleTracker.from_seed_yaml()
    sample_rule = tracker.active_rules()[0]
    root = Path(__file__).resolve().parents[3]
    for op_name in ("attention_residual", "mla"):
        real_runner = PerfTargetCampaignRunner(
            config=CampaignConfig(operator_name=op_name, max_rounds=1),
            target_module_path=root / "examples" / "ascend950pr_ops" / op_name / "kernels.py",
            llm_backend=MockLLMBackend({}),
        )
        op_spec = DEFAULT_REGIME_SPECS[op_name][0]
        dummy_res = CandidateSurrogateResult(
            candidate_id="cand_0",
            operator_name=op_name,
            all_passed=True,
            avg_speedup=1.0,
            min_case_speedup=1.0,
            avg_roofline_eta=0.5,
            surrogate_score=0.5,
        )
        exp_req = real_runner._build_exploit_regime_request(
            snippet_source="def _select_demo():\n    return {}\n",
            func_name=op_spec.builder_candidates[0],
            regime_spec=op_spec,
            rule=sample_rule,
            best_result=dummy_res,
        )
        crt_req = real_runner._build_creative_regime_request(
            snippet_source="def _select_demo():\n    return {}\n",
            func_name=op_spec.builder_candidates[0],
            regime_spec=op_spec,
            best_result=dummy_res,
            effort="high",
        )
        for req in (exp_req, crt_req):
            assert f"### INPUT DOMAIN & FAST-MATH ENVELOPE ({op_name})" in req.user_prompt
            assert "builder_kwargs" in req.user_prompt
        for rule in tracker.active_rules():
            assert rule.rule_id not in crt_req.system_prompt + "\n" + crt_req.user_prompt

    ar_calls = {"sel": 0, "bk": None}

    def _fake_ar_sel(n_depth, d_dim, tl_dtype, schedule_mode, m_rows):
        ar_calls["sel"] += 1
        return {
            "tile_m": 16,
            "pass_configs": {},
            "compile_flags": (),
            "builder_kwargs": {"omit_max_shift": True},
        }

    def _fake_ar_get_kernel(*_args, extra_builder_kwargs=None, **_kwargs):
        ar_calls["bk"] = extra_builder_kwargs
        return lambda v_f, x_f, _w, _g, _a, _e: torch.zeros_like(x_f)

    monkeypatch.setattr(ar_kernels, "_select_ar_aligned_config", _fake_ar_sel)
    monkeypatch.setattr(ar_kernels, "_get_kernel", _fake_ar_get_kernel)
    ar_kernels._DISPATCH_CACHE.clear()
    v_stack = torch.zeros((4, 2, 16, 64), dtype=torch.float16)
    x_in = torch.zeros((2, 16, 64), dtype=torch.float16)
    w_in = torch.zeros((64,), dtype=torch.float16)
    g_in = torch.ones((64,), dtype=torch.float16)
    ar_kernels.attention_residual(v_stack, x_in, w_in, g_in, 1.0)
    ar_kernels.attention_residual(v_stack, x_in, w_in, g_in, 1.0)
    assert ar_calls["sel"] == 1
    assert ar_calls["bk"] == {"omit_max_shift": True}

    mla_calls = {"sel": 0, "bk": None}

    def _fake_mla_sel(**_kwargs):
        mla_calls["sel"] += 1
        return {
            "bm": 64,
            "aligned_group_q": True,
            "num_kv_splits": 1,
            "pass_configs": {},
            "compile_flags": (),
            "builder_kwargs": {"fast_exp2": True},
        }

    def _fake_mla_get_kernel(*_args, extra_builder_kwargs=None, **_kwargs):
        mla_calls["bk"] = extra_builder_kwargs
        return lambda qn, _qr, _kn, _kr, _v, _s: torch.zeros_like(qn)

    monkeypatch.setattr(mla_kernels, "_select_mla_prefill_config", _fake_mla_sel)
    monkeypatch.setattr(mla_kernels, "_get_kernel", _fake_mla_get_kernel)
    mla_kernels._DISPATCH_CACHE.clear()
    qn = torch.zeros((1, 8, 4, 128), dtype=torch.float16)
    qr = torch.zeros((1, 8, 4, 64), dtype=torch.float16)
    kn = torch.zeros((1, 8, 1, 128), dtype=torch.float16)
    kr = torch.zeros((1, 8, 1, 64), dtype=torch.float16)
    vv = torch.zeros((1, 8, 1, 128), dtype=torch.float16)
    mla_kernels.mla(qn, qr, kn, kr, vv)
    mla_kernels.mla(qn, qr, kn, kr, vv)
    assert mla_calls["sel"] == 1
    assert mla_calls["bk"] == {"fast_exp2": True}


def test_wp_final1_pruned_inline_regimes_partial_override_and_mla_combine_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify inline pruned kernels, partial override + orphan GC, cache, and combine gate."""
    import pytest  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from examples.ascend950pr_ops.mla import kernels as mla_kernels  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import campaign_runner as cr_mod  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import resolve_regime_config  # noqa: PLC0415
    from tilefoundry_ascend.agent.orchestrator import run_stage_a_static  # noqa: PLC0415

    root = Path(__file__).resolve().parents[3]
    ar_path = root / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
    mla_path = root / "examples" / "ascend950pr_ops" / "mla" / "kernels.py"
    ar_src = ar_path.read_text(encoding="utf-8")
    mla_src = mla_path.read_text(encoding="utf-8")

    for spec in DEFAULT_REGIME_SPECS["attention_residual"]:
        _, snip = extract_regime_snippet(ar_src, spec)
        assert "@T.prim_func" in snip
        assert "_build_attention_residual_prim_func(" not in snip
        if spec.regime_tag in ("small_decode", "aligned_prefill"):
            assert "rem > 0" not in snip
        else:
            assert "rem > 0" in snip
        assert f"def {spec.allowed_helper_prefix}init_row" in snip

    for spec in DEFAULT_REGIME_SPECS["mla"]:
        _, snip = extract_regime_snippet(mla_src, spec)
        assert "@T.prim_func" in snip
        assert "_build_mla_prim_func(" not in snip
        if spec.regime_tag == "bnsd_gather":
            assert '"BSND"' not in snip
        elif spec.regime_tag == "prefill_compute":
            assert '"BNSD"' not in snip

    ar_aligned_spec = next(
        s for s in DEFAULT_REGIME_SPECS["attention_residual"] if s.regime_tag == "aligned_prefill"
    )
    sel_only_patch = (
        "def _select_ar_aligned_config(\n"
        "    n_depth: int = 8,\n"
        "    d_dim: int = 4096,\n"
        '    tl_dtype: str = "bfloat16",\n'
        '    schedule_mode: str = "tuned",\n'
        "    m_rows: int = 2048,\n"
        ") -> dict[str, object]:\n"
        "    _ = (n_depth, d_dim, tl_dtype, m_rows)\n"
        "    return {\n"
        '        "tile_m": 4,\n'
        '        "pass_configs": {},\n'
        '        "compile_flags": _ar_default_flags(schedule_mode),\n'
        "    }\n"
    )
    patched_sel_only = patch_regime_into_module(ar_src, ar_aligned_spec, sel_only_patch)
    assert "def _build_ar_aligned_func(" in patched_sel_only
    assert "def _ar_aligned_step_row(" in patched_sel_only
    res_sel_only = resolve_regime_config(patched_sel_only, ar_aligned_spec)
    assert res_sel_only.builder_kwargs["tile_m"] == 4
    stage_a_partial = run_stage_a_static(
        patched_sel_only,
        factory_name=res_sel_only.builder_name,
        factory_kwargs=res_sel_only.builder_kwargs,
        pass_configs=res_sel_only.pass_configs or None,
        compile_flags=res_sel_only.compile_flags or None,
    )
    assert stage_a_partial.passed, stage_a_partial.error_message

    _, aligned_snip = extract_regime_snippet(ar_src, ar_aligned_spec)
    init_start = aligned_snip.index("@T.macro\ndef _ar_aligned_init_row(")
    init_end = aligned_snip.index("@T.macro\ndef _ar_aligned_step_row(")
    init_macro = aligned_snip[init_start:init_end]
    patched_macro_only = patch_regime_into_module(
        ar_src,
        ar_aligned_spec,
        init_macro.replace("T.float32(-1.0e30)", "T.float32(-1.0e35)"),
    )
    assert "T.float32(-1.0e35)" in patched_macro_only
    assert "def _ar_aligned_step_row(" in patched_macro_only
    assert "def _build_ar_aligned_func(" in patched_macro_only

    builder_start = aligned_snip.index("def _build_ar_aligned_func(")
    new_init_macro = init_macro.replace("_ar_aligned_init_row", "_ar_aligned_init_v2")
    replaced_builder = aligned_snip[builder_start:].replace(
        "_ar_aligned_init_row(", "_ar_aligned_init_v2("
    )
    patched_gc = patch_regime_into_module(
        ar_src,
        ar_aligned_spec,
        new_init_macro + "\n\n" + replaced_builder,
    )
    assert "def _ar_aligned_init_v2(" in patched_gc
    assert "def _ar_aligned_init_row(" not in patched_gc
    assert "def _ar_aligned_step_row(" in patched_gc

    with pytest.raises(ValueError, match="protected symbol"):
        patch_regime_into_module(
            mla_src,
            next(s for s in DEFAULT_REGIME_SPECS["mla"] if s.regime_tag == "split_kv_decode"),
            "_WORKSPACE_CACHE = {}\n",
        )

    mla_kernels._DISPATCH_CACHE.clear()
    mla_kernels._WORKSPACE_CACHE.clear()
    captured_ptrs: list[tuple[int, int, bool, bool]] = []
    captured_ck: list[object] = []

    def _fake_stage1(qn_f, qr_f, kn_c, kr_c, v_c, o_p, lse_p, sc):
        captured_ptrs.append(
            (o_p.data_ptr(), lse_p.data_ptr(), o_p.is_contiguous(), lse_p.is_contiguous())
        )

    def _fake_combine(o_p, lse_p):
        return o_p.sum(dim=0).to(torch.float16)

    def _fake_sel_decode(**_kw):
        return {
            "bm": 64,
            "aligned_group_q": True,
            "num_kv_splits": 4,
            "pass_configs": {},
            "compile_flags": (),
            "combine_kwargs": {"vec_w": 64},
        }

    def _fake_get_kernel(*_args, **_kwargs):
        return _fake_stage1

    def _fake_get_combine(*_args, extra_combine_kwargs=None, **_kwargs):
        captured_ck.append(extra_combine_kwargs)
        return _fake_combine

    orig_empty = torch.empty
    empty_calls = [0]

    def _counting_empty(*args, **kwargs):
        empty_calls[0] += 1
        return orig_empty(*args, **kwargs)

    monkeypatch.setattr(mla_kernels, "_select_mla_decode_config", _fake_sel_decode)
    monkeypatch.setattr(mla_kernels, "_get_kernel", _fake_get_kernel)
    monkeypatch.setattr(mla_kernels, "_get_combine_kernel", _fake_get_combine)
    monkeypatch.setattr(mla_kernels.torch, "empty", _counting_empty)

    qn_d = torch.zeros((1, 1, 128, 512), dtype=torch.float16)
    qr_d = torch.zeros((1, 1, 128, 64), dtype=torch.float16)
    kn_d = torch.zeros((1, 1024, 1, 512), dtype=torch.float16)
    kr_d = torch.zeros((1, 1024, 1, 64), dtype=torch.float16)
    vv_d = torch.zeros((1, 1024, 1, 512), dtype=torch.float16)

    mla_kernels.mla(qn_d, qr_d, kn_d, kr_d, vv_d, num_kv_splits_override=4)
    mla_kernels.mla(qn_d, qr_d, kn_d, kr_d, vv_d, num_kv_splits_override=4)
    mla_kernels.mla(qn_d, qr_d, kn_d, kr_d, vv_d, num_kv_splits_override=2)
    assert empty_calls[0] == 2
    assert captured_ptrs[0] == captured_ptrs[1]
    assert captured_ptrs[0][:2] == captured_ptrs[2][:2]
    assert all(p[2] and p[3] for p in captured_ptrs)
    assert captured_ck == [{"vec_w": 64}] * 3

    split_spec = next(s for s in DEFAULT_REGIME_SPECS["mla"] if s.regime_tag == "split_kv_decode")
    broken_combine_mod = mla_src.replace(
        "lse_ub = T.alloc_shared((splits, 8), \"float32\")",
        "lse_ub = T.alloc_shared((splits * 4096, 256), \"float32\")",
        1,
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="mla", max_rounds=1),
        target_module_path=mla_path,
        perf_target=PerfTargetSpec("mla", 2.5, 1.35, 0.85),
        regime_specs=(split_spec,),
        llm_backend=MockLLMBackend({}),
    )
    l0_bad = runner._run_l0(broken_combine_mod, split_spec)
    assert not l0_bad.passed
    assert "combine_kernel(_build_mla_split_kv_combine_func)" in l0_bad.error_message

    l1_builders: list[str] = []

    def _fake_l1(mod_src, *, builder_name, **_kw):
        l1_builders.append(builder_name)
        if builder_name == "_build_mla_split_kv_combine_func":
            return StageBResult(passed=False, error_message="broken combine ir")
        return StageBResult(passed=True)

    monkeypatch.setattr(cr_mod, "run_l1_compile_check", _fake_l1)
    l1_bad = runner._run_l1(broken_combine_mod, split_spec)
    assert not l1_bad.passed
    assert l1_builders == ["_build_mla_split_kv_decode_func", "_build_mla_split_kv_combine_func"]
    assert "combine_kernel(_build_mla_split_kv_combine_func): broken combine ir" == l1_bad.error_message


def test_wp_final2_prewarm_builder_kwargs_and_local_regime_l3_promotion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify prewarm extra builder/combine kwargs forwarding and local regime L3 promotion."""
    import yaml  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: PLC0415
        prewarm_surrogate_tilelang_cache,
    )

    fake_root = tmp_path / "repo"
    (fake_root / "src").mkdir(parents=True)
    ascend_agent_dir = fake_root / "targets" / "ascend"
    ascend_agent_dir.mkdir(parents=True)
    (ascend_agent_dir / "tilefoundry_ascend").symlink_to(
        Path(__file__).resolve().parents[3] / "targets" / "ascend" / "tilefoundry_ascend"
    )

    cann_repo = tmp_path / "cann"
    mla_dir = cann_repo / "tasks" / "level4" / "mla"
    mla_dir.mkdir(parents=True, exist_ok=True)
    (mla_dir / "cases.yaml").write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "case_id": 7,
                        "input_shape": [
                            [1, 1, 128, 512],
                            [1, 1, 128, 64],
                            [1, 2048, 1, 512],
                            [1, 2048, 1, 64],
                            [1, 2048, 1, 512],
                        ],
                        "dtype": "float16",
                        "attrs": {"numKVHeads": 1, "inputLayout": "BSND", "is_causal": False},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tl_stub_dir = tmp_path / "tl_stub"
    tl_stub_dir.mkdir()
    (tl_stub_dir / "tilelang.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "def compile(prog, *, target='ascend', out_idx=None, pass_configs=None, "
        "compile_flags=None):\n"
        "    cdir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cdir.mkdir(parents=True, exist_ok=True)\n"
        "    with (cdir / 'compiled_entries.jsonl').open('a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps({'prog': prog, 'out_idx': out_idx}) + '\\n')\n"
        "    return {'cached': prog}\n",
        encoding="utf-8",
    )

    mla_cand = tmp_path / "cand_mla_kw.py"
    mla_cand.write_text(
        "import tilelang\n"
        "def _select_mla_decode_config(b_dim=1, s_dim=1, s_kv=2048, n_q=128, n_kv=1, "
        "d_nope=512, d_rope=64, tl_dtype='float16', input_layout='BSND', "
        "is_causal=False, schedule_mode='tuned'):\n"
        "    return {'bm': 64, 'aligned_group_q': True, 'num_kv_splits': 4, "
        "'pass_configs': {}, 'compile_flags': [], "
        "'builder_kwargs': {'unroll_k': 2}, 'combine_kwargs': {'vec_w': 64}}\n"
        "def _build_mla_split_kv_decode_func(n_q=128, n_kv=1, d_nope=512, d_rope=64, "
        "tl_dtype='float16', input_layout='BSND', is_causal=False, bm=64, "
        "aligned_group_q=True, num_kv_splits=1, schedule_mode='tuned', unroll_k=1):\n"
        "    return {'kind': 'stage1', 'splits': num_kv_splits, 'unroll_k': unroll_k}\n"
        "def _get_kernel(n_q, n_kv, d_nope, d_rope, tl_dtype, input_layout, is_causal, "
        "bm, aligned_group_q, schedule_mode, *, builder_fn=None, num_kv_splits=1, "
        "pass_configs=None, compile_flags=None, extra_builder_kwargs=None):\n"
        "    extra = dict(extra_builder_kwargs or {})\n"
        "    prog = builder_fn(n_q=n_q, n_kv=n_kv, d_nope=d_nope, d_rope=d_rope, "
        "tl_dtype=tl_dtype, input_layout=input_layout, is_causal=is_causal, bm=bm, "
        "aligned_group_q=aligned_group_q, num_kv_splits=num_kv_splits, "
        "schedule_mode=schedule_mode, **extra)\n"
        "    prog['extra_builder_kwargs'] = extra\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=None)\n"
        "def _get_combine_kernel(d_nope, tl_dtype, num_kv_splits, *, "
        "pass_configs=None, compile_flags=None, extra_combine_kwargs=None):\n"
        "    return tilelang.compile({'kind': 'combine', 'splits': num_kv_splits, "
        "'extra_combine_kwargs': dict(extra_combine_kwargs or {})}, target='ascend', "
        "out_idx=[2])\n",
        encoding="utf-8",
    )

    tl_cache = tmp_path / "tl_cache_wp2"
    ref_cache = tmp_path / "ref_cache_wp2"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TILEFOUNDRY_NPU_TILELANG_SOURCE", str(tl_stub_dir))
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SETUP", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_PYTHON", raising=False)

    prewarm_surrogate_tilelang_cache(
        "mla",
        [("c_mla_kw", mla_cand)],
        (AnchorCaseSpec("level4/mla", 7, "split_kv_decode", 4.5, 1.5),),
        cannbench_repo=cann_repo,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    entries = [
        json.loads(line)
        for line in (tl_cache / "compiled_entries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 2
    assert entries[0]["prog"]["unroll_k"] == 2
    assert entries[0]["prog"]["extra_builder_kwargs"] == {"unroll_k": 2}
    assert entries[1]["prog"]["extra_combine_kwargs"] == {"vec_w": 64}

    six_anchors = (
        AnchorCaseSpec("level4/attention_residual", 1, "regime_a", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 2, "regime_b", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 3, "regime_b", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 4, "regime_b", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 5, "regime_b", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 6, "regime_b", 10.0, 2.0),
    )
    regimes = (
        RegimeTargetSpec("regime_a", ("_build_regime_a",), {}),
        RegimeTargetSpec("regime_b", ("_build_regime_b",), {}),
    )
    target_mod = tmp_path / "kernels_local_gain.py"
    target_mod.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")

    local_win_src = _IMPROVED_REGIME_A
    regress_tail_src = _SSBUF_REGIME_A

    def _l2_local_win(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lats = {i: 10.0 for i in range(1, 7)}
            if "T.alloc_ss" in src:
                lats[1] = 9.40
                lats[6] = 10.55
            elif "with T.Kernel(2):" in src:
                lats[1] = 9.40
            out[cid] = _make_surrogate_result(
                cid, "attention_residual", tuple(active_anchors), lats
            )
        return out

    base_res = _l2_local_win([("base", target_mod)], six_anchors)["base"]
    win_mod = tmp_path / "win_preview.py"
    win_mod.write_text(
        patch_regime_into_module(_MULTI_REGIME_MODULE, "_build_regime_a", local_win_src),
        encoding="utf-8",
    )
    win_res = _l2_local_win([("win", win_mod)], six_anchors)["win"]
    global_gain = win_res.surrogate_score - base_res.surrogate_score
    assert 0.0 < global_gain < 0.25
    assert win_res.for_regime("regime_a").avg_speedup >= 1.02 * base_res.for_regime("regime_a").avg_speedup

    l3_calls = [0]

    def _l3_pass(_p: Path) -> L3AcceptanceResult:
        l3_calls[0] += 1
        return L3AcceptanceResult(passed=True)

    backend_win = MockLLMBackend(
        responses={
            "exploit_rule": [local_win_src] * 2,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Regime A local 6pct gain",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": local_win_src,
                    }
                )
            ],
        }
    )
    runner_win = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=1,
            min_surrogate_gain=0.25,
        ),
        target_module_path=target_mod,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 5.0, 0.99, target_surrogate_score=99.0),
        anchors=six_anchors,
        regime_specs=regimes,
        llm_backend=backend_win,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_local_win,
        l3_contract_runner=_l3_pass,
        checkpoint_path=tmp_path / "ckpt_local_win.json",
    )
    rep_win = asyncio.run(runner_win.run())
    assert rep_win.promoted_rounds == (1,)
    assert l3_calls[0] == 1

    target_mod_reg = tmp_path / "kernels_local_regress.py"
    target_mod_reg.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    backend_reg = MockLLMBackend(
        responses={
            "exploit_rule": [regress_tail_src] * 2,
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Regime A gain with tail regression",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": regress_tail_src,
                    }
                )
            ],
        }
    )
    runner_reg = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=1,
            min_surrogate_gain=0.25,
        ),
        target_module_path=target_mod_reg,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 5.0, 0.99, target_surrogate_score=99.0),
        anchors=six_anchors,
        regime_specs=regimes,
        llm_backend=backend_reg,
        l0_checker=lambda src, _reg: _make_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_local_win,
        l3_contract_runner=_l3_pass,
        checkpoint_path=tmp_path / "ckpt_local_reg.json",
    )
    rep_reg = asyncio.run(runner_reg.run())
    assert rep_reg.promoted_rounds == ()
    assert l3_calls[0] == 1


def test_wp_final3_persistent_wave_occupancy_weighted_burst_pipe_all_and_r21() -> None:
    """Verify persistent active wave occupancy, byte-weighted burst, PIPE_ALL, and R21."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import build_anchor_var_bindings  # noqa: PLC0415
    from tilefoundry_ascend.agent.kb import (  # noqa: PLC0415
        build_predicate_context,
        load_seed_rules,
    )
    from tilefoundry_ascend.agent.orchestrator import (  # noqa: PLC0415
        compute_static_dynamic_residual,
        run_stage_a_static,
    )
    from tilefoundry_ascend.ascend.passes import ObjectivePhysicalLedger  # noqa: PLC0415
    from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc  # noqa: PLC0415

    rules = {r.rule_id: r for r in load_seed_rules()}
    assert len(rules) >= 33
    root = Path(__file__).resolve().parents[3]

    ar_src = (
        root / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
    ).read_text(encoding="utf-8")
    ar_dims = {"N": 2, "B": 1, "S": 1, "D": 256}
    ar_vb = build_anchor_var_bindings(ar_dims, kwargs={"n_depth": 2, "d_dim": 256})
    ar_res = run_stage_a_static(
        ar_src,
        factory_name="_build_ar_small_decode_func",
        factory_kwargs={"n_depth": 2, "d_dim": 256, "tl_dtype": "bfloat16", "tile_m": 1},
        config_selector_name="_select_ar_small_decode_config",
        dims=ar_dims,
        var_bindings=ar_vb,
    )
    assert ar_res.passed and ar_res.ledger is not None
    assert ar_res.ledger.grid_blocks == 1
    assert ar_res.ledger.active_grid_blocks == 1
    assert ar_res.ledger.wave_occupancy < 0.05
    assert ar_res.ledger.in_loop_pipe_all_barriers >= 1

    ar_rt = ObjectivePhysicalLedger.from_dict(ar_res.ledger.to_dict())
    assert ar_rt.active_grid_blocks == 1
    assert ar_rt.in_loop_pipe_all_barriers == ar_res.ledger.in_loop_pipe_all_barriers

    ctx_ar = build_predicate_context(
        ar_res.ledger,
        residual=compute_static_dynamic_residual(ar_res.ledger, {"dur_us": 10.0}),
    )
    assert ctx_ar["in_loop_pipe_all_barriers"] >= 1
    assert rules["R21_SUBROW_DEPTH_OR_WIDTH_CORE_SHARDING"].matches(ctx_ar) is True
    assert rules["R18_WAVE_QUANTIZED_BURST_TILING"].matches(ctx_ar) is False
    assert rules["R19_STATIC_PINGPONG_UNROLL_NO_ALIAS"].matches(ctx_ar) is True

    mla_src = (root / "examples" / "ascend950pr_ops" / "mla" / "kernels.py").read_text(
        encoding="utf-8"
    )
    mla_dims = {"B": 1, "S": 1, "S_kv": 4096, "N_q": 128, "N_kv": 1, "D_nope": 512, "D_rope": 64}
    mla_vb = build_anchor_var_bindings(mla_dims, kwargs={"n_q": 128, "n_kv": 1})
    mla_s1_unsplit = run_stage_a_static(
        mla_src,
        factory_name="_build_mla_split_kv_decode_func",
        factory_kwargs={"n_q": 128, "n_kv": 1, "d_nope": 512, "d_rope": 64, "tl_dtype": "float16", "input_layout": "BSND", "is_causal": False, "bm": 64, "aligned_group_q": True, "num_kv_splits": 1},
        config_selector_name=None,
        dims=mla_dims,
        var_bindings=mla_vb,
    )
    assert mla_s1_unsplit.passed and mla_s1_unsplit.ledger is not None
    assert mla_s1_unsplit.ledger.wave_occupancy < 0.15
    ctx_s1 = build_predicate_context(
        mla_s1_unsplit.ledger,
        residual=compute_static_dynamic_residual(mla_s1_unsplit.ledger, {"dur_us": 20.0}),
    )
    assert rules["R13_SPLIT_KV_WAVE_RECOVERY"].matches(ctx_s1) is True

    mla_s1_tuned = run_stage_a_static(
        mla_src,
        factory_name="_build_mla_split_kv_decode_func",
        factory_kwargs={"n_q": 128, "n_kv": 1, "d_nope": 512, "d_rope": 64, "tl_dtype": "float16", "input_layout": "BSND", "is_causal": False, "bm": 64, "aligned_group_q": True, "num_kv_splits": 1},
        config_selector_name="_select_mla_decode_config",
        dims=mla_dims,
        var_bindings=mla_vb,
    )
    assert mla_s1_tuned.passed and mla_s1_tuned.ledger is not None
    assert mla_s1_tuned.ledger.active_grid_blocks == 28
    assert mla_s1_tuned.ledger.wave_occupancy >= 0.9

    mla_s8 = run_stage_a_static(
        mla_src.replace('"num_kv_splits": splits,', '"num_kv_splits": 8,', 1),
        factory_name="_build_mla_split_kv_decode_func",
        factory_kwargs={"n_q": 128, "n_kv": 1, "d_nope": 512, "d_rope": 64, "tl_dtype": "float16", "input_layout": "BSND", "is_causal": False, "bm": 64, "aligned_group_q": True, "num_kv_splits": 8},
        config_selector_name="_select_mla_decode_config",
        dims=mla_dims,
        var_bindings=mla_vb,
    )
    assert mla_s8.passed and mla_s8.ledger is not None
    assert mla_s8.ledger.wave_occupancy >= 0.50

    @T.prim_func
    def weighted_dma_pipe_all(
        K_main: T.Tensor((4, 64, 512), "float16"),
        K_rope: T.Tensor((4, 64, 64), "float16"),
        Out: T.Tensor((4, 64, 512), "float16"),
    ):
        with T.Kernel(4) as cid:
            ub_main = T.alloc_shared((64, 512), "float16")
            ub_rope = T.alloc_shared((64, 64), "float16")
            for k in T.serial(4):
                T.copy(K_main[k, :, :], ub_main)
                T.copy(K_rope[k, :, :], ub_rope)
                T.ascend_pipe_barrier("PIPE_ALL")
            T.copy(ub_main, Out[cid, :, :])

    @T.prim_func
    def weighted_dma_no_barrier(
        K_main: T.Tensor((4, 64, 512), "float16"),
        K_rope: T.Tensor((4, 64, 64), "float16"),
        Out: T.Tensor((4, 64, 512), "float16"),
    ):
        with T.Kernel(4) as cid:
            ub_main = T.alloc_shared((64, 512), "float16")
            ub_rope = T.alloc_shared((64, 64), "float16")
            for k in T.serial(4):
                T.copy(K_main[k, :, :], ub_main)
                T.copy(K_rope[k, :, :], ub_rope)
            T.copy(ub_main, Out[cid, :, :])

    res_with_bar = inspect_tilelang_primfunc(weighted_dma_pipe_all)
    res_no_bar = inspect_tilelang_primfunc(weighted_dma_no_barrier)
    assert res_with_bar.ledger.dma_burst_efficiency_ratio > 0.92
    assert res_with_bar.ledger.in_loop_pipe_all_barriers == 1
    assert res_no_bar.ledger.in_loop_pipe_all_barriers == 0

    ctx_bar = build_predicate_context(res_with_bar.ledger)
    assert rules["R19_STATIC_PINGPONG_UNROLL_NO_ALIAS"].matches(ctx_bar) is True
    ctx_post_bar = build_predicate_context(res_no_bar.ledger, prev_ledger=res_with_bar.ledger)
    assert rules["R19_STATIC_PINGPONG_UNROLL_NO_ALIAS"].postcondition_realized(ctx_post_bar) is True


def test_wp_final4_pluggable_prescreen_cube_norm_stepping_stone_and_msprof_sync(
    tmp_path: Path,
) -> None:
    """Verify WP-FINAL-4 default-OFF prescreen, Cube norm, tags, lineage, and msprof sync."""
    from types import SimpleNamespace  # noqa: PLC0415

    from tests.ascend.agent.test_surrogate_batch_npu import _extract_msprof_metrics  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign.campaign_runner import build_arg_parser  # noqa: PLC0415
    from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (  # noqa: PLC0415
        summarize_excluded_mechanisms,
    )
    from tilefoundry_ascend.agent.orchestrator.residual_beam import (  # noqa: PLC0415
        TwoLaneBeamSearch,
        extract_structural_signature,
    )
    from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (  # noqa: PLC0415
        TileConfigCandidate,
        prescreen_live_ir_neighborhood,
    )

    default_cfg = CampaignConfig(operator_name="mla")
    assert default_cfg.enable_tilesize_prescreen is False
    parsed_default = build_arg_parser().parse_args(["--operator", "mla"])
    parsed_enabled = build_arg_parser().parse_args(
        ["--operator", "mla", "--enable-tilesize-prescreen"]
    )
    assert parsed_default.enable_tilesize_prescreen is False
    assert parsed_enabled.enable_tilesize_prescreen is True

    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),)
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {"bm": 64, "bn": 64, "bk": 64}),)
    creative_reply = json.dumps(
        {
            "physical_hypothesis": "Static pingpong buffers eliminate aliasing",
            "target_local_metric": "mte2_wait_ratio",
            "dsl_source": _SSBUF_REGIME_A,
        }
    )

    def _l2_eval(cands, active_anchors):
        out = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            lat = 15.0 if "T.alloc_ss" in src else 12.0
            out[cid] = _make_surrogate_result(
                cid,
                "attention_residual",
                tuple(active_anchors),
                {2: lat},
            )
        return out

    for flag, expected_present in ((False, False), (True, True)):
        t_path = tmp_path / f"kernels_{int(flag)}.py"
        t_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
        backend = MockLLMBackend(
            responses={
                "exploit_rule": [_IMPROVED_REGIME_A] * 8,
                "explore_creative": [creative_reply] * 8,
            }
        )
        beam = TwoLaneBeamSearch(exploit_width=2, stepping_stone_width=1)
        runner = PerfTargetCampaignRunner(
            config=CampaignConfig(
                operator_name="attention_residual",
                max_rounds=2,
                stagnation_escalate_rounds=1,
                enable_tilesize_prescreen=flag,
            ),
            target_module_path=t_path,
            perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
            anchors=anchors,
            regime_specs=regimes,
            llm_backend=backend,
            beam=beam,
            l0_checker=lambda src, _reg: _make_stage_a(src),
            l1_compiler=lambda _src, _reg: StageBResult(passed=True),
            l2_batch_evaluator=_l2_eval,
            l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
            checkpoint_path=tmp_path / f"ckpt_{int(flag)}.json",
        )
        asyncio.run(runner.run(resume=False))
        creative_calls = [c for c in backend.calls if c.role == "explore_creative"]
        assert creative_calls
        assert any("### TILE PRESCREEN GUIDANCE" in c.user_prompt for c in creative_calls) is expected_present
        if not flag:
            stone = beam.stepping_stone_beam[0]
            assert stone.worker_kind == "creative_explore"
            assert "Static pingpong buffers eliminate aliasing" in stone.hypothesis
            assert "mte2_wait_ratio" in stone.hypothesis
            assert stone.parent_id == "incumbent_r0"
            assert any(
                "### STEPPING-STONE PARENT HYPOTHESIS" in c.user_prompt and stone.hypothesis in c.user_prompt
                for c in creative_calls
            )
            r2_rule = next((c for c in beam.exploit_beam if c.round_idx == 2), None)
            assert r2_rule is not None and r2_rule.parent_id == stone.candidate_id

    cube_dsl = (
        "import tilelang.ascend.language as T\n\n"
        "def make_cube_kernel(bm: int = 64, bn: int = 64):\n"
        "    bk = 64\n"
        "    @T.prim_func\n"
        "    def main(\n"
        "        X: T.Tensor((512, 512), 'bfloat16'),\n"
        "        W: T.Tensor((512, 512), 'bfloat16'),\n"
        "        Y: T.Tensor((512, 512), 'float32'),\n"
        "    ):\n"
        "        with T.Kernel(4) as bx:\n"
        "            x_l1 = T.alloc_l1((bm, bk), 'bfloat16')\n"
        "            w_l1 = T.alloc_l1((bn, bk), 'bfloat16')\n"
        "            acc = T.alloc_l0c((bm, bn), 'float32')\n"
        "            for kt in T.Pipelined(4, num_stages=2):\n"
        "                T.copy(X[0:bm, kt * bk : (kt + 1) * bk], x_l1)\n"
        "                T.copy(W[0:bn, kt * bk : (kt + 1) * bk], w_l1)\n"
        "                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)\n"
        "            T.copy(acc, Y[0:bm, 0:bn])\n"
        "    return main\n"
    )
    cube_summary = prescreen_live_ir_neighborhood(
        cube_dsl,
        factory_name="make_cube_kernel",
        center=TileConfigCandidate(bm=64, bn=64, bk=64, stages=2),
        max_variants=8,
    )
    cube_rank_1 = next(line for line in cube_summary.splitlines() if line.strip().startswith("1."))
    assert "OK" in cube_rank_1
    import re  # noqa: PLC0415

    m_tile = re.search(r"bm=(\d+),\s*bn=(\d+)", cube_rank_1)
    assert m_tile is not None
    assert int(m_tile.group(1)) * int(m_tile.group(2)) > 64 * 64

    pingpong_split_dsl = (
        "import tilelang.ascend.language as T\n\n"
        "def _select_cfg():\n"
        "    return {'num_kv_splits': 8}\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((4, 64), 'float16'), Out: T.Tensor((4, 64), 'float32')):\n"
        "        with T.Kernel(4) as bx:\n"
        "            va = T.alloc_shared((1, 64), 'float16')\n"
        "            vb = T.alloc_shared((1, 64), 'float16')\n"
        "            acc = T.alloc_fragment((64,), 'float32')\n"
        "            for i in range(2):\n"
        "                T.copy(X[i, :], va)\n"
        "                T.ascend_pipe_barrier('PIPE_ALL')\n"
        "            for j in range(64):\n"
        "                acc[j] = acc[j] + vb[0, j] * 1.0\n"
        "            T.copy(acc, Out[bx, :])\n"
        "    return main\n"
    )
    sig = extract_structural_signature(pingpong_split_dsl)
    for tag in (
        "static_pingpong_2buf",
        "loop_fission_multi_pass",
        "pipe_all_barrier",
        "vfma_unshifted",
        "split_kv_multi",
    ):
        assert tag in sig
    excluded_desc = summarize_excluded_mechanisms((), [sig])
    joined_desc = " ".join(excluded_desc)
    assert "Static double-buffered" in joined_desc
    assert "Multi-pass loop fission" in joined_desc
    assert "Split-KV" in joined_desc

    pct_prof = SimpleNamespace(
        aic_mad=90.0,
        aiv_vec=20.0,
        aic_mte2=10.0,
        aiv_mte2=5.0,
        aic_fixpipe=15.0,
        aic_total_cycles=1000,
        aiv_total_cycles=1000,
        dur_us=12.5,
    )
    pct_metrics = _extract_msprof_metrics(pct_prof)
    assert abs(float(pct_metrics["aic_mac_ratio"]) - 0.90) < 1e-6
    assert abs(float(pct_metrics["aiv_vec_ratio"]) - 0.20) < 1e-6
    assert abs(float(pct_metrics["sync_stall_ratio"]) - 0.0) < 1e-6

    stall_prof = SimpleNamespace(
        aic_mad=0.20,
        aiv_vec=0.20,
        aic_mte2=0.10,
        aiv_mte2=0.10,
        aic_total_cycles=1000,
        aiv_total_cycles=1000,
        dur_us=25.0,
    )
    stall_metrics = _extract_msprof_metrics(stall_prof)
    assert abs(float(stall_metrics["sync_stall_ratio"]) - 0.65) < 1e-6


def test_l3_full_contract_via_agent_gates_timeout_and_target_module_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify L3 AgentGates helper passes timeout_sec and candidate module env vars cleanly."""
    import subprocess  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import surrogate_cannbench  # noqa: PLC0415

    fake_env = tmp_path / "env.sh"
    fake_env.write_text("TILEFOUNDRY_NPU_CANNBENCH_REPO=/tmp/cb\n", encoding="utf-8")
    cand_mod = tmp_path / "cand_kernels.py"
    cand_mod.write_text("x = 1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_subprocess_run(cmd, *, text, capture_output, cwd, env, timeout):
        _ = (text, capture_output, cwd)
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps({"status": "PASS", "runId": "r_l3", "receiptPath": "/tmp/r_l3.json"}),
            stderr="",
        )

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(surrogate_cannbench, "_resolve_env_file_path", lambda _p: fake_env)
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "/tmp/ag")
    monkeypatch.setattr(surrogate_cannbench, "resolve_cannbench_repo", lambda **_kw: tmp_path)
    monkeypatch.setattr(
        surrogate_cannbench,
        "ensure_npu_compare_shim_dir",
        lambda path, **_kw: Path(path),
    )

    l3_res = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand_mod,
        source_root=tmp_path,
        env_file=fake_env,
        device_id="2",
        timeout_sec=123,
    )
    assert l3_res.passed is True
    assert l3_res.run_id == "r_l3"
    assert captured["timeout"] == 123
    env_sent = captured["env"]
    assert env_sent["TF_L3_TARGET_OPERATOR"] == "attention_residual"
    assert env_sent["TF_L3_TARGET_MODULE_PATH"] == str(cand_mod.resolve())
    assert "--device-id" in captured["cmd"] and "2" in captured["cmd"]


def test_run_l0_l1_round_filter_isolates_syntax_repair_timeout(tmp_path: Path) -> None:
    """Verify syntax_repair timeouts fail only the affected candidate without penalizing rules."""
    from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: PLC0415
        run_l0_l1_round_filter,
    )

    class _RepairTimeoutBackend:
        def __init__(self) -> None:
            self.calls = []

        async def generate(self, req, *, effort_override=None):
            _ = effort_override
            self.calls.append(req)
            if req.role == "syntax_repair":
                raise TimeoutError("agy active_ceiling_timeout (240.0s) exceeded")
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Timeout test",
                        "target_local_metric": "sync_stall",
                        "dsl_source": _SSBUF_REGIME_A,
                    }
                )
            return _IMPROVED_REGIME_A

    backend = _RepairTimeoutBackend()
    checks = asyncio.run(
        run_l0_l1_round_filter(
            [("cand_ok", _IMPROVED_REGIME_A), ("cand_timeout", _SSBUF_REGIME_A)],
            patch_fn=lambda snip: snip,
            l0_checker=lambda snip: (
                StageAResult(passed=True, prim_func=None, ledger=_make_stage_a(snip).ledger)
                if "T.alloc_ss" not in snip
                else StageAResult(passed=False, prim_func=None, ledger=None, error_message="bad_ssbuf")
            ),
            l1_compiler=lambda _src: StageBResult(passed=True),
            repair_backend=backend,
            max_syntax_repairs=1,
            slice_hint="authorized slice symbols (_build_ar_small_decode_func)",
        )
    )
    by_id = {c.candidate_id: c for c in checks}
    assert by_id["cand_ok"].passed is True
    assert by_id["cand_timeout"].passed is False
    assert by_id["cand_timeout"].failed_tier == "llm_transport"
    assert "TimeoutError" in by_id["cand_timeout"].error_message
    assert any("Authorized target: authorized slice symbols" in c.system_prompt for c in backend.calls)

    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "regime_a", 10.0, 2.0),)
    regimes = (RegimeTargetSpec("regime_a", ("_build_regime_a",), {"bm": 64, "bn": 64, "bk": 64}),)
    t_path = tmp_path / "kernels_repair_timeout.py"
    t_path.write_text(_MULTI_REGIME_MODULE, encoding="utf-8")
    tracker = BayesianRuleTracker.from_seed_yaml()
    before_trials = {rid: st.trials for rid, st in tracker.posteriors.items()}
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=1),
        target_module_path=t_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=_RepairTimeoutBackend(),
        tracker=tracker,
        l0_checker=lambda src, _reg: (
            _make_stage_a(src)
            if "def _build_regime_a(tile_m=1)" in src
            else StageAResult(passed=False, prim_func=None, ledger=None, error_message="bad")
        ),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, aa: {
            cid: _make_surrogate_result(cid, "attention_residual", tuple(aa), {2: 10.0})
            for cid, _p in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
    )
    report = asyncio.run(runner.run())
    assert report.total_rounds == 1
    assert any(tier == "llm_transport" for _wid, tier, _msg in report.rounds[0].l0_l1_failed)
    for rid, st in tracker.posteriors.items():
        assert st.trials == before_trials[rid]


def test_stepping_stone_splicing_causal_credit_and_live_flywheel_wiring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify stepping-stone parent splicing, causal PMU credit, and AgentGates probe wiring."""
    import ast  # noqa: PLC0415
    from tilefoundry_ascend.agent.flywheel.ablation_distiller import MicroProbeSpec  # noqa: PLC0415
    from tilefoundry_ascend.agent.kb.schema import AscendCausalRule  # noqa: PLC0415

    multi_sym_module = """
import tilelang.ascend.language as T

_KERNEL_CACHE = {}

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

def _select_reg_b_config(tile_m=1):
    return {"tile_m": tile_m}

def _build_reg_b(tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((64, 64), "bfloat16"), Y: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((64, 64), "bfloat16")
            T.copy(X[0:64, 0:64], b_ub)
            T.copy(b_ub, Y[0:64, 0:64])
    return main
"""
    regimes = (
        RegimeTargetSpec(
            regime_tag="reg_a",
            builder_candidates=("_build_reg_a",),
            sample_builder_kwargs={"tile_m": 1},
            description="Regime A",
            config_selector_fn="_select_reg_a_config",
            editable_symbols=("_select_reg_a_config", "_build_reg_a"),
            allowed_helper_prefix="_reg_a_",
        ),
        RegimeTargetSpec(
            regime_tag="reg_b",
            builder_candidates=("_build_reg_b",),
            sample_builder_kwargs={"tile_m": 1},
            description="Regime B",
            config_selector_fn="_select_reg_b_config",
            editable_symbols=("_select_reg_b_config", "_build_reg_b"),
            allowed_helper_prefix="_reg_b_",
        ),
    )
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "reg_a", 20.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 16, "reg_b", 1000.0, 240.0),
    )

    stone_partial_dsl = """
def _reg_a_helper():
    return 2

def _build_reg_a(tile_m=2):
    _n = _reg_a_helper()
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.MixedKernel(_n) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            ss[sid] = T.Cast("uint32", 0)
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main
"""
    child_partial_dsl = """
def _build_reg_a(tile_m=4):
    _n = _reg_a_helper()
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(8) as bx:
            a_ub = T.alloc_shared((64, 128), "bfloat16")
            T.copy(X[0:64, 0:128], a_ub)
            T.copy(a_ub, Y[0:64, 0:128])
    return main
"""
    flat_dsl = """
def _build_reg_a(tile_m=1):
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main
"""
    causal_rule = AscendCausalRule(
        rule_id="R_TEST_WAVE_ONLY",
        regime_guard="PIPE_V > 0",
        precondition="wave_occupancy < 0.5",
        transform_template="Scale core blocks with helper macro.",
        expected_postcondition="wave_occupancy > prev.wave_occupancy and measured_mte2_wait_ratio < prev.measured_mte2_wait_ratio",
        source_provenance="test",
        alpha_prior=10.0,
        beta_prior=1.0,
    )
    tracker = BayesianRuleTracker.from_rules([causal_rule])

    creative_r1 = json.dumps(
        {
            "physical_hypothesis": "Introduce helper macro and SSBUF stone",
            "target_local_metric": "sync_stall_ratio",
            "dsl_source": stone_partial_dsl,
        }
    )
    creative_r2 = json.dumps(
        {
            "physical_hypothesis": "Keep inherited helper macro",
            "target_local_metric": "wave_occupancy",
            "dsl_source": child_partial_dsl,
        }
    )
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [flat_dsl, child_partial_dsl, child_partial_dsl],
            "explore_creative": [creative_r1, creative_r2],
        }
    )

    def _l0_check(mod_src: str, reg: RegimeTargetSpec) -> StageAResult:
        if "_reg_a_helper()" in mod_src and "def _reg_a_helper" not in mod_src:
            return StageAResult(passed=False, prim_func=None, ledger=None, error_message="NameError: _reg_a_helper")
        ns: dict[str, object] = {"T": type("_DummyT", (), {"prim_func": staticmethod(lambda f: f)})}
        try:
            ast.parse(mod_src)
        except SyntaxError as exc:
            return StageAResult(passed=False, prim_func=None, ledger=None, error_message=str(exc))
        _ = ns
        high_wave = "T.Kernel(8)" in mod_src
        wave_val = 0.50 if high_wave else (0.10 if "T.MixedKernel" in mod_src else 0.05)
        cyc = 50 if high_wave else (110 if "T.MixedKernel" in mod_src else 120)
        blocks = 8 if high_wave else (2 if "T.MixedKernel" in mod_src else 1)
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=cyc,
            pipe_cycles={"AIV0.PIPE_V": cyc},
            waterlines={},
            grid_blocks=blocks,
            wave_occupancy=wave_val,
        )
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    def _l2_eval(cands, aa):
        out = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            lat_a = 8.0 if "T.Kernel(8)" in src else (22.0 if "T.MixedKernel" in src else 20.0)
            out[cid] = _make_surrogate_result(cid, "attention_residual", tuple(aa), {2: lat_a, 16: 250.0})
        return out

    msprof_calls: list[list[int]] = []

    def _msprof_collect(op_name, cand_path, anc_seq):
        src = Path(cand_path).read_text(encoding="utf-8")
        msprof_calls.append([int(a.case_num) for a in anc_seq])
        wait = 0.12 if "T.Kernel(8)" in src else 0.45
        return {anc_seq[0].regime_tag: {"dur_us": 8.0 if "T.Kernel(8)" in src else 20.0, "aiv_vec": 0.45, "aiv_mte2": wait, "sync_stall_ratio": 0.1}}

    t_path = tmp_path / "kernels_splice.py"
    t_path.write_text(multi_sym_module, encoding="utf-8")
    kb_out = tmp_path / "kb_out.yaml"
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        probe_runner=lambda _spec: {"baseline_dur_us": 40.0, "optimized_dur_us": 20.0, "numerical_ok": 1.0},
        min_edit_gain=0.05,
        min_probe_speedup=1.08,
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=2,
            stagnation_escalate_rounds=1,
            stagnation_freeze_rounds=4,
            k_rule_workers=1,
            min_surrogate_gain=0.25,
            min_flywheel_gain=0.03,
        ),
        target_module_path=t_path,
        perf_target=PerfTargetSpec("attention_residual", 2.8, 1.5, 0.95),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=backend,
        tracker=tracker,
        beam=TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1),
        flywheel_distiller=flywheel,
        l0_checker=_l0_check,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=_msprof_collect,
        checkpoint_path=tmp_path / "ckpt_splice.json",
        campaign_kb_path=kb_out,
    )
    report = asyncio.run(runner.run())
    assert report.total_rounds == 2
    assert report.rounds[1].parent_lane == "stepping_stone"
    assert report.rounds[1].promoted is True
    assert "def _reg_a_helper" in t_path.read_text(encoding="utf-8")
    assert "with T.Kernel(8)" in t_path.read_text(encoding="utf-8")
    st = tracker.posteriors["R_TEST_WAVE_ONLY"]
    assert st.failures == 1 and st.successes == 1
    assert report.admitted_rule_ids
    assert kb_out.is_file()

    env_f = tmp_path / "env.sh"
    env_f.write_text("TILEFOUNDRY_NPU_CANNBENCH_REPO=" + str(tmp_path) + "\n", encoding="utf-8")
    fake_ag = tmp_path / "agent-gates"
    fake_ag.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", str(fake_ag))
    seen_req_files: list[str] = []

    def _fake_ag_cmd(cmd, *, cwd, env, timeout_sec, **kwargs):
        req_file = Path(env["TF_MICRO_PROBE_REQUEST"])
        seen_req_files.append(str(req_file))
        req_json = json.loads(req_file.read_text(encoding="utf-8"))
        Path(req_json["response_path"]).write_text(
            json.dumps({"passed": True, "metrics": {"baseline_dur_us": 50.0, "optimized_dur_us": 20.0, "speedup": 2.5, "numerical_ok": 1.0}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "{}", ""), {"status": "PASS"}

    monkeypatch.setattr(surrogate_cannbench, "_run_agent_gates_cmd", _fake_ag_cmd)
    probe_metrics = surrogate_cannbench.run_micro_probe_via_agent_gates(
        MicroProbeSpec(
            probe_id="p_test",
            probe_kind="core_parallelism",
            factory_name="make_kernel",
            baseline_probe_source="def make_kernel(): pass",
            optimized_probe_source="def make_kernel(): pass",
            line_count_baseline=30,
            line_count_optimized=30,
            tensor_shape=(16, 128),
        ),
        source_root=tmp_path,
        env_file=env_f,
        device_id="2",
        prewarm_cache=False,
    )
    assert seen_req_files
    assert probe_metrics["speedup"] == 2.5


def test_default_anchors_hydrate_live_metadata_from_cannbench_repo(tmp_path: Path) -> None:
    """Verify get_default_anchors hydrates live 950pr.json values for DEFAULT_ANCHORS ops."""
    repo = tmp_path / "live_cannbench"
    mla_dir = repo / "tasks" / "level4" / "mla"
    meta_dir = repo / "tasks" / "metadata"
    mla_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    (mla_dir / "proto.yaml").write_text("operator:\n  name: MLA\n", encoding="utf-8")
    (mla_dir / "cases.yaml").write_text(
        "cases:\n"
        "- case_id: 1\n  input_shape: [[2, 64, 128, 512]]\n  dtype: [float16]\n  attrs: {inputLayout: BSND}\n"
        "- case_id: 6\n  input_shape: [[1, 128, 128, 448]]\n  dtype: [float16]\n  attrs: {inputLayout: BSND}\n"
        "- case_id: 7\n  input_shape: [[1, 1, 128, 512]]\n  dtype: [float16]\n  attrs: {inputLayout: BSND}\n"
        "- case_id: 11\n  input_shape: [[128, 1, 128, 512]]\n  dtype: [float16]\n  attrs: {inputLayout: BSND}\n"
        "- case_id: 13\n  input_shape: [[2, 128, 64, 512]]\n  dtype: [float16]\n  attrs: {inputLayout: BNSD}\n",
        encoding="utf-8",
    )
    meta_payload = {
        "level4": {
            "mla": {
                "1": {"baseline_perf_us": 41.25, "t_hw_us": 13.75},
                "6": {"baseline_perf_us": 30.7423, "t_hw_us": 10.2474},
                "7": {"baseline_perf_us": 5.25, "t_hw_us": 1.75},
                "11": {"baseline_perf_us": 68.6341, "t_hw_us": 22.878},
                "13": {"baseline_perf_us": 39.0, "t_hw_us": 12.5},
            }
        }
    }
    (meta_dir / "950pr.json").write_text(json.dumps(meta_payload), encoding="utf-8")

    anchors = get_default_anchors("mla", cannbench_repo=repo)
    assert [a.case_num for a in anchors] == [1, 6, 7, 11, 13]
    assert [a.regime_tag for a in anchors] == [
        "prefill_compute",
        "prefill_compute",
        "split_kv_decode",
        "split_kv_decode",
        "bnsd_gather",
    ]
    by_case = {a.case_num: a for a in anchors}
    assert abs(by_case[1].ref_perf_us - 41.25) < 1e-6
    assert abs(by_case[1].t_hw_us - 13.75) < 1e-6
    assert abs(by_case[6].ref_perf_us - 30.7423) < 1e-6
    assert abs(by_case[6].t_hw_us - 10.2474) < 1e-6
    assert abs(by_case[11].ref_perf_us - 68.6341) < 1e-6
    assert abs(by_case[11].t_hw_us - 22.878) < 1e-6
    assert abs(by_case[13].ref_perf_us - 39.0) < 1e-6
    assert abs(by_case[13].t_hw_us - 12.5) < 1e-6







