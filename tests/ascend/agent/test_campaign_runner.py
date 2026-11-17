"""Offline unit tests for WP8 surrogate anchors, AST isolation, and campaign state machine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tilefoundry_ascend.agent import (
    AnchorCaseSpec,
    BayesianRuleTracker,
    CampaignConfig,
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
    compute_case_eq3_score,
    extract_regime_snippet,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.orchestrator import StageAResult, StageBResult
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
    meta_dir = repo / "operators" / "metadata"
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
    assert ar_target.target_avg_speedup == 1.95
    assert ar_target.target_roofline_eta == 0.78
    assert mla_target.target_avg_speedup == 0.95
    assert mla_target.per_anchor_max_us == {7: 8.0, 13: 95.0}

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
        perf_target=PerfTargetSpec("attention_residual", 2.0, 1.5, 0.95),
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
        perf_target=PerfTargetSpec("attention_residual", 5.0, 4.0, 0.78),
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

    exploit_replies = [_IMPROVED_REGIME_A] * 8 + [_IMPROVED_REGIME_B] * 8
    creative_replies = [
        json.dumps(
            {
                "physical_hypothesis": "Novel SSBUF path",
                "target_local_metric": "sync_stall",
                "dsl_source": _SSBUF_REGIME_A,
            }
        )
        for _ in range(4)
    ] + [
        json.dumps(
            {
                "physical_hypothesis": "Regime B probe",
                "target_local_metric": "aiv_vec",
                "dsl_source": _IMPROVED_REGIME_B,
            }
        )
        for _ in range(4)
    ]
    backend = MockLLMBackend(
        responses={
            "exploit_rule": exploit_replies,
            "explore_creative": creative_replies,
        }
    )

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
    assert report.escalated_rounds == (3, 4, 7, 8)
    assert [r.regime_tag for r in report.rounds[:4]] == ["regime_a"] * 4
    assert [r.regime_tag for r in report.rounds[4:]] == ["regime_b"] * 4
    assert report.rounds[2].parent_lane == "stepping_stone"
    assert report.rounds[2].creative_effort == "xhigh"

    creative_calls = [c for c in backend.calls if c.role == "explore_creative"]
    assert [c.effort for c in creative_calls[:4]] == ["high", "high", "xhigh", "xhigh"]
    assert "T.alloc_ss" in creative_calls[2].user_prompt
    for call in creative_calls:
        joined = call.system_prompt + "\n" + call.user_prompt
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
            * 3,
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
