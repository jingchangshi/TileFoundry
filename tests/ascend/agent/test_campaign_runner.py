"""Offline unit tests for WP8 surrogate anchors, AST isolation, and campaign state machine."""

from __future__ import annotations

import asyncio
import json
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

    exploit_replies = [_IMPROVED_REGIME_A] * 8 + [_IMPROVED_REGIME_B] * 8
    creative_replies = [
        json.dumps(
            {
                "physical_hypothesis": "Novel SSBUF path",
                "target_local_metric": "sync_stall",
                "dsl_source": _SSBUF_REGIME_A,
            }
        )
        for _ in range(8)
    ] + [
        json.dumps(
            {
                "physical_hypothesis": "Regime B probe",
                "target_local_metric": "aiv_vec",
                "dsl_source": _IMPROVED_REGIME_B,
            }
        )
        for _ in range(8)
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
    assert [c.effort for c in creative_calls[:8]] == ["high"] * 4 + ["xhigh"] * 4
    assert "T.alloc_ss" in creative_calls[4].user_prompt
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
    from tilefoundry_ascend.agent.orchestrator import TwoLaneBeamSearch

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
    from tilefoundry_ascend.agent.campaign import get_default_regime_specs
    from tilefoundry_ascend.ascend.passes.hir_tir_bridge import compile_hir_navigation_packet
    from tilefoundry_ascend.ascend.target import AscendTarget

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
            pkt = compile_hir_navigation_packet(
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
    import tilefoundry_ascend.agent.router.capability_gate as cap_gate
    from tilefoundry_ascend.agent.kb import build_predicate_context
    from tilefoundry_ascend.agent.orchestrator import compute_static_dynamic_residual
    from tilefoundry_ascend.ascend.passes import build_scope_waterline

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
    assert "small_decode" in runner._regime_nav_packets

    active_by_id = {rule.rule_id: rule for rule in runner._tracker.active_rules()}
    assert len(active_by_id) == 17
    expected_new_rules = (
        "R13_SPLIT_KV_WAVE_RECOVERY",
        "R14_GQA_HEAD_M_FOLDING",
        "R15_ASYMMETRIC_1C2V_STAGE_FISSION",
        "R16_PADDED_PITCH_TAIL_MASKING",
        "R17_RMEM_ACCUM_AND_256KB_UB_UNLOCK",
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
    assert parsed.source_root == "."
    assert parsed.env_file == "env.sh"

    defaults = parser.parse_args(["--operator", "attention_residual"])
    assert defaults.disable_early_goal_stop is False
    assert defaults.min_surrogate_gain == 0.25


def test_scoped_multi_symbol_regime_slice_extract_and_patch(tmp_path: Path) -> None:
    """Verify multi-symbol extraction, patching, helper lifecycle, and guardrails."""
    import pytest
    from tilefoundry_ascend.agent.campaign import (
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
    updated_snippet = (
        helper_src
        + snippet.replace('"num_kv_splits": 1,', '"num_kv_splits": 4,', 1)
    )
    patched = patch_regime_into_module(mla_src, split_spec, updated_snippet)
    assert "def _mla_decode_helper(" in patched
    assert '"num_kv_splits": 4,' in patched
    _, re_extracted = extract_regime_snippet(patched, split_spec)
    assert "def _mla_decode_helper(" in re_extracted

    cleaned = patch_regime_into_module(patched, split_spec, snippet)
    assert "def _mla_decode_helper(" not in cleaned

    with pytest.raises(ValueError, match="missing required regime symbol"):
        patch_regime_into_module(
            mla_src,
            split_spec,
            snippet.split("def _build_mla_split_kv_combine_func")[0],
        )

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
    import tilelang
    from tilefoundry_ascend.agent.campaign import resolve_regime_config
    from tilefoundry_ascend.agent.orchestrator import run_stage_a_static
    from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

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

    from examples.ascend950pr_ops.attention_residual import kernels as ar_kernels
    from examples.ascend950pr_ops.mla import kernels as mla_kernels

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
    from examples.ascend950pr_ops.mla import kernels as mla_kernels
    from tilefoundry_ascend.agent.campaign import get_default_regime_specs, resolve_regime_config
    from tilefoundry_ascend.agent.orchestrator import run_stage_a_static
    from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc
    from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

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
