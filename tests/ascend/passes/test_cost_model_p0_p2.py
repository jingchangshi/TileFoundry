"""Unit and integration tests covering CostModel P0, P1, and P2 upgrades."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec, PerfTargetSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _parse_cce_res_usage_with_summary,
    build_stage_b_driver_source,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.ascend.cost_model import (
    DEFAULT_LAUNCH_OVERHEAD_US,
    DEFAULT_RESIDUAL_BASIS_SPECS,
    CostFeatures,
    HybridAscendCostModel,
    build_cost_features_from_ledger,
    evaluate_cost_model_ranking,
    load_official_reports,
    run_cost_model_backtest,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_p0_chip_scaling_floors_persistence_and_observe_batch(tmp_path: Path) -> None:
    """Verify P0 chip-wide scaling, launch/FixPipe/CV floors, persistence, and observe_batch."""
    target = AscendTarget("ascend.950pr")
    axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=200,
        pipe_cycles={"AIC.PIPE_M": 1200, "AIV0.PIPE_V": 400},
        gm_roundtrip_bytes=65536,
        grid_blocks=28,
        waterlines={},
    )
    blueprint = SimpleNamespace(
        roofline={"cube_flops": 2.0e12, "vector_flops": 1.0e10, "min_gm_bytes": 4_000_000}
    )
    cf = build_cost_features_from_ledger(
        ledger,
        live_plan={"policy": {"tile_mnk": [128, 128, 64], "l1_stages": 2, "cube_operand_dtype": "f16"}},
        axes=axes,
        blueprint=blueprint,
        target=target,
    )
    assert cf.total_tiles == 28
    assert cf.gm_bytes >= 4_000_000.0
    assert cf.cube_flops >= 2.0e12
    assert cf.fixpipe_bytes > 0.0
    assert cf.cv_sync_events >= 4
    assert cf.topology_axes == axes

    model = HybridAscendCostModel(target)
    tiny_pred = model.predict(
        CostFeatures(gm_bytes=4096.0, contig_burst_bytes=4096, kernel_launches=1, topology_axes=axes)
    )
    assert tiny_pred.launch_overhead_us == pytest.approx(DEFAULT_LAUNCH_OVERHEAD_US)
    assert tiny_pred.pred_us >= DEFAULT_LAUNCH_OVERHEAD_US

    pred_coupled = model.predict(cf)
    assert pred_coupled.fixpipe_floor_us > 0.0
    assert pred_coupled.cv_sync_us > 0.0
    assert pred_coupled.topology_bucket == "coupled_1c2v:bidirectional_cv_ring"

    seq_path = tmp_path / "seq.jsonl"
    batch_path = tmp_path / "batch.jsonl"
    m_seq = HybridAscendCostModel(target, calibration_path=seq_path)
    m_batch = HybridAscendCostModel(target, calibration_path=batch_path)
    observations = [
        (cf, pred_coupled.pred_us * 1.4),
        (CostFeatures(gm_bytes=8192.0, vec_flops=1e9, topology_axes=axes), 9.5),
        (CostFeatures(gm_bytes=65536.0, cube_flops=1e11, topology_axes=axes), 80.0),
    ]
    for feat, lat in observations:
        m_seq.observe(feat, lat, persist=True)
    m_batch.observe_batch(observations, persist=True)

    assert len(m_seq.weights) == len(DEFAULT_RESIDUAL_BASIS_SPECS)
    assert len(m_seq.weights) == 12
    assert math.isclose(m_seq.predict(cf).pred_us, m_batch.predict(cf).pred_us, rel_tol=1e-9)
    reloaded = HybridAscendCostModel(target, calibration_path=batch_path)
    assert reloaded.sample_count == 3
    assert math.isclose(reloaded.predict(cf).pred_us, m_batch.predict(cf).pred_us, rel_tol=1e-9)


def test_p1_stage_b_cce_dump_7_detectors_and_hierarchical_ridge(tmp_path: Path) -> None:
    """Verify Stage B dump hook, all 7 Tier-1 CCE detectors, hierarchical buckets, and eval."""
    cand_path = tmp_path / "cand.py"
    cand_path.write_text("# dummy", encoding="utf-8")
    dump_path = tmp_path / "lowered_kernel.cce"
    driver = build_stage_b_driver_source(cand_path, kernel_dump_path=dump_path)
    assert "kernel_dump_path" in driver
    assert "kernel_source" in driver

    cce_src = """
    // comment __simd_vf__ should be ignored
    extern __simd_vf__ void vf_proto();
    __simd_vf__ void vf_a() {
        pipe_barrier(PIPE_ALL);
        LocalMemBar();
        dcache_invalidate();
    }
    __simd_vf__ void vf_b() {}
    void host_entry() {
        int offset = dynamic_offset(idx);
        asc_vf_call<1>(vf_a);
        asc_vf_call<1>(vf_b);
        vmad v0, v1, v2
        vmad v0, v3, v4
        vexp v5, v0
        vexp v6, v5
    }
    .LBB0_1:
        sub x0, x0, #1
        cbnz x0, .LBB0_1
    """
    res_out = "Vector Register Usage: 96\nScalar Register Usage: 48\nRegister Spill: 512 bytes\ncannot use vloop_v2"
    spill_b, vloop_hit, summary = _parse_cce_res_usage_with_summary(res_out, asm_or_dump_text=cce_src)
    assert spill_b == 512
    assert vloop_hit is False
    assert summary.software_loops >= 1
    assert summary.has_local_membar_in_vf_call is True
    assert summary.vf_fusion_breaks == 1
    assert summary.alias_serialized_buffers >= 1
    assert summary.dual_issue_ipc_collapse > 0.0
    assert summary.dcache_dcci_flushes >= 1

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    aiv_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    cv_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    aiv_feat = CostFeatures(vec_flops=2e10, gm_bytes=8_000_000.0, topology_axes=aiv_axes)
    cv_feat = CostFeatures(cube_flops=1e11, vec_flops=1e10, gm_bytes=8_000_000.0, topology_axes=cv_axes)
    unseen_axes = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "ragged_group")
    unseen_feat = CostFeatures(cube_flops=1e11, gm_bytes=8_000_000.0, topology_axes=unseen_axes)

    base_aiv = model.predict(aiv_feat).pred_us
    base_cv = model.predict(cv_feat).pred_us
    batch = [
        (aiv_feat, base_aiv * 2.2),
        (aiv_feat, base_aiv * 2.3),
        (aiv_feat, base_aiv * 2.1),
        (cv_feat, base_cv * 0.65),
        (cv_feat, base_cv * 0.70),
        (cv_feat, base_cv * 0.68),
    ]
    model.observe_batch(batch, persist=False)
    assert "aiv_only:gm_ub_stream" in model.bucket_weights
    assert "coupled_1c2v:bidirectional_cv_ring" in model.bucket_weights
    assert model.predict(aiv_feat).pred_us > base_aiv * 1.4
    assert model.predict(cv_feat).pred_us < base_cv * 0.9
    assert model.predict(unseen_feat).topology_bucket == "aic_only:gm_l1_l0_fixpipe"

    report = evaluate_cost_model_ranking(
        [10.0, 20.0, 30.0, 5.0, 15.0],
        [11.0, 22.0, 33.0, 5.5, 16.5],
        groups=["opA", "opA", "opA", "opB", "opB"],
    )
    assert report.mean_intra_group_spearman_rho == pytest.approx(1.0)
    assert report.pairwise_accuracy == pytest.approx(1.0)
    assert report.top1_regret_ratio == pytest.approx(0.0)


def test_p2_stage_a_topk_pruning_penalty_exemption_and_backtest(tmp_path: Path) -> None:
    """Verify Stage A -> CostModel Top-K -> Stage B pruning, tracker exemption, and backtest."""
    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    ledger_fast = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIC.PIPE_M": 500},
        gm_roundtrip_bytes=16384,
        grid_blocks=28,
        waterlines={},
    )
    ledger_mid = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=250,
        pipe_cycles={"AIC.PIPE_M": 2000, "AIV0.PIPE_V": 2000, "AIV0.PIPE_MTE2": 2000},
        gm_roundtrip_bytes=131072,
        grid_blocks=28,
        waterlines={},
    )
    ledger_slow = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4000,
        pipe_cycles={"AIC.PIPE_M": 40000},
        gm_roundtrip_bytes=8 * 1024 * 1024,
        grid_blocks=28,
        waterlines={},
    )
    passers = [
        ("cand_slow", "", StageAResult(passed=True, prim_func=None, ledger=ledger_slow)),
        ("cand_fast", "", StageAResult(passed=True, prim_func=None, ledger=ledger_fast)),
        ("cand_mid", "", StageAResult(passed=True, prim_func=None, ledger=ledger_mid)),
    ]
    ranked = rank_stage_a_candidates_with_cost_model(passers, cost_model=model)
    assert [r.candidate_id for r in ranked] == ["cand_fast", "cand_mid", "cand_slow"]

    l1_calls: list[str] = []

    def _l0(mod_src: str) -> StageAResult:
        if "FAIL_A" in mod_src:
            return StageAResult(passed=False, prim_func=None, ledger=None, error_message="bad_ir")
        mapping = {
            "FAST": ledger_fast,
            "MID": ledger_mid,
            "SLOW": ledger_slow,
        }
        for key, ldg in mapping.items():
            if key in mod_src:
                return StageAResult(passed=True, prim_func=None, ledger=ldg)
        return StageAResult(passed=True, prim_func=None, ledger=ledger_slow)

    def _l1(mod_src: str) -> StageBResult:
        l1_calls.append(mod_src)
        return StageBResult(passed=True)

    checks = asyncio.run(
        run_l0_l1_round_filter(
            [
                ("w0", "SLOW"),
                ("w1", "FAIL_A"),
                ("w2", "FAST"),
                ("w3", "MID"),
            ],
            patch_fn=lambda s: s,
            l0_checker=_l0,
            l1_compiler=_l1,
            max_syntax_repairs=0,
            stage_a_cost_ranker=lambda items: [
                (item.candidate_id, item.pred_us)
                for item in rank_stage_a_candidates_with_cost_model(items, cost_model=model)
            ],
            cost_prescreen_top_k=2,
        )
    )
    assert [c.candidate_id for c in checks] == ["w0", "w1", "w2", "w3"]
    assert set(l1_calls) == {"FAST", "MID"}
    assert len(l1_calls) == 2
    assert checks[0].passed is False and checks[0].failed_tier == "cost_prescreen"
    assert checks[1].passed is False and checks[1].failed_tier == "stage_a"
    assert checks[2].passed is True and checks[3].passed is True

    assert CampaignConfig(operator_name="softmax").enable_tilesize_prescreen is True
    mod_file = tmp_path / "mod.py"
    base_src = "def kernel(m_dim=64):\n    # BASE\n    return m_dim\n"
    mod_file.write_text(base_src, encoding="utf-8")
    ckpt_path = tmp_path / "state" / "checkpoint.json"
    tracker = BayesianRuleTracker.from_seed_yaml()
    prior_trials = {rid: st.trials for rid, st in tracker.posteriors.items()}

    rule_calls = 0

    class _StubBackend:
        async def generate(self, req, *, effort_override=None):
            nonlocal rule_calls
            _ = effort_override
            if req.role == "exploit_rule":
                rule_calls += 1
                if rule_calls == 1:
                    return "```python\ndef kernel(m_dim=65):\n    # MID\n    return m_dim\n```"
                return "```python\ndef kernel(m_dim=66):\n    # SLOW\n    return m_dim\n```"
            return json.dumps(
                {
                    "physical_hypothesis": "try fast candidate",
                    "target_local_metric": "latency_us",
                    "dsl_source": "def kernel(m_dim=67):\n    # FAST\n    return m_dim\n",
                }
            )

    def _l0_campaign(mod_src: str, _spec: RegimeTargetSpec) -> StageAResult:
        ldg = ledger_slow if "# SLOW" in mod_src else (ledger_fast if "# FAST" in mod_src else ledger_mid)
        return StageAResult(passed=True, prim_func=None, ledger=ldg)

    def _l2_campaign(cands, anchors):
        out = {}
        for cid, cpath in cands:
            text = Path(cpath).read_text(encoding="utf-8")
            lat = 5.0 if "# FAST" in text else 12.0
            anc_res = [
                AnchorEvalMetrics(
                    case_num=a.case_num,
                    regime_tag=a.regime_tag,
                    accuracy_passed=True,
                    cand_us=lat,
                    ref_perf_us=a.ref_perf_us,
                    t_hw_us=a.t_hw_us,
                    speedup=a.ref_perf_us / lat,
                    roofline_eta=a.t_hw_us / lat,
                    eq3_score=90.0 if lat <= 6.0 else 60.0,
                )
                for a in anchors
            ]
            out[cid] = CandidateSurrogateResult(
                candidate_id=cid,
                operator_name="softmax",
                all_passed=True,
                avg_speedup=anc_res[0].speedup,
                min_case_speedup=anc_res[0].speedup,
                avg_roofline_eta=anc_res[0].roofline_eta,
                surrogate_score=anc_res[0].eq3_score,
                anchors=tuple(anc_res),
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="softmax",
            max_rounds=1,
            k_rule_workers=2,
            max_syntax_repairs=0,
            enable_tilesize_prescreen=True,
            cost_prescreen_top_k=1,
            cost_model_mode="advisory",
        ),
        target_module_path=mod_file,
        perf_target=PerfTargetSpec(
            operator_name="softmax",
            target_avg_speedup=5.0,
            target_min_case_speedup=5.0,
            target_roofline_eta=0.99,
            target_surrogate_score=99.0,
        ),
        anchors=(
            AnchorCaseSpec(
                task="level2/softmax",
                case_num=1,
                regime_tag="default",
                ref_perf_us=10.0,
                t_hw_us=5.0,
            ),
        ),
        regime_specs=(
            RegimeTargetSpec(
                regime_tag="default",
                builder_candidates=("kernel",),
                sample_builder_kwargs={"m_dim": 64},
                editable_symbols=("kernel",),
                representative_hir_dims={"total_elements": 32768, "M": 64, "D": 512},
                representative_anchor_dims={"M": 64, "D": 512},
            ),
        ),
        llm_backend=_StubBackend(),
        tracker=tracker,
        l0_checker=_l0_campaign,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_campaign,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=lambda _op, _p, _a: {},
        checkpoint_path=ckpt_path,
        sil_advisory=SimpleNamespace(
            topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
            roofline={"median_gm_bytes": 65536},
            input_dtypes=("float16",),
        ),
    )
    report = asyncio.run(runner.run(resume=False))
    expected_cal_path = (ckpt_path.parent / "cost_model_calibration.jsonl").resolve()
    assert runner._cost_calibration_path == expected_cal_path
    assert expected_cal_path.is_file()
    assert runner._cost_advisor.underlying_cost_model is not None
    assert runner._cost_advisor.underlying_cost_model.sample_count >= 1
    changed_rids = [
        rid for rid in prior_trials if tracker.posteriors[rid].trials != prior_trials[rid]
    ]
    assert set(changed_rids).issubset(set(prior_trials))
    assert "r1_rule_0" in report.rounds[0].l0_l1_passed_ids
    assert "r1_creative" in report.rounds[0].l0_l1_passed_ids

    rep_dir1 = tmp_path / "reports_a"
    rep_dir2 = tmp_path / "reports_b"
    payload = {
        "operators": [
            {
                "rel_path": "level2/fake_op_a",
                "operator": "FakeOpA",
                "score": 80.0,
                "cases": [
                    {
                        "case_id": "level2/fake_op_a_1",
                        "elapsed_us": 6.0,
                        "t_hw_us": 2.0,
                        "baseline_perf_us": 10.0,
                        "speedup": 1.67,
                        "perf_score": 80.0,
                        "accuracy": {"passed": True},
                    },
                    {
                        "case_id": "level2/fake_op_a_2",
                        "elapsed_us": 18.0,
                        "t_hw_us": 10.0,
                        "baseline_perf_us": 30.0,
                        "speedup": 1.67,
                        "perf_score": 80.0,
                        "accuracy": {"passed": True},
                    },
                ],
            },
            {
                "rel_path": "level3/fake_op_b",
                "operator": "FakeOpB",
                "score": 70.0,
                "cases": [
                    {
                        "case_id": "level3/fake_op_b_1",
                        "elapsed_us": 9.0,
                        "t_hw_us": 3.0,
                        "baseline_perf_us": 15.0,
                        "speedup": 1.67,
                        "perf_score": 70.0,
                        "accuracy": {"passed": True},
                    },
                    {
                        "case_id": "level3/fake_op_b_2",
                        "elapsed_us": 24.0,
                        "t_hw_us": 12.0,
                        "baseline_perf_us": 36.0,
                        "speedup": 1.50,
                        "perf_score": 70.0,
                        "accuracy": {"passed": True},
                    },
                ],
            },
        ]
    }
    rep_dir1.mkdir(parents=True)
    rep_dir2.mkdir(parents=True)
    (rep_dir1 / "official_final.json").write_text(json.dumps(payload), encoding="utf-8")
    (rep_dir2 / "official_final.json").write_text(json.dumps(payload), encoding="utf-8")
    missing_ops = tmp_path / "no_tasks_here"
    with pytest.raises(ValueError, match="Canonical ops_root does not exist"):
        load_official_reports([rep_dir1, rep_dir2], ops_root=missing_ops)

    tmp_ops = tmp_path / "ops_tasks"
    op_a_dir = tmp_ops / "level2" / "fake_op_a"
    op_b_dir = tmp_ops / "level3" / "fake_op_b"
    op_a_dir.mkdir(parents=True)
    op_b_dir.mkdir(parents=True)
    (tmp_ops / "metadata").mkdir(parents=True)
    (tmp_ops / "metadata" / "950pr.json").write_text(
        json.dumps({"level2": {}, "level3": {}}),
        encoding="utf-8",
    )
    (tmp_ops / "suite_manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"id": "fake_op_a", "level": "L2", "path": "tasks/level2/fake_op_a"},
                    {"id": "fake_op_b", "level": "L3", "path": "tasks/level3/fake_op_b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (op_a_dir / "proto.yaml").write_text(
        "operator:\n"
        "  name: fake_op_a\n"
        "  category: ElementWise\n"
        "  inputs:\n"
        "    - name: x\n"
        "      dtype: [float16]\n"
        "  outputs:\n"
        "    - name: y\n"
        "      dtype: [float16]\n"
        "  formula: y = exp(x)\n",
        encoding="utf-8",
    )
    (op_a_dir / "cases.yaml").write_text(
        "cases:\n"
        "  - case_id: 1\n"
        "    input_shape: [[16, 1024]]\n"
        "    dtype: [float16]\n"
        "  - case_id: 2\n"
        "    input_shape: [[32, 2048]]\n"
        "    dtype: [float16]\n",
        encoding="utf-8",
    )
    (op_b_dir / "proto.yaml").write_text(
        "operator:\n"
        "  name: fake_op_b\n"
        "  category: Contraction\n"
        "  inputs:\n"
        "    - name: x\n"
        "      dtype: [float16]\n"
        "    - name: w\n"
        "      dtype: [float16]\n"
        "  outputs:\n"
        "    - name: y\n"
        "      dtype: [float16]\n"
        "  formula: y = x @ w\n",
        encoding="utf-8",
    )
    (op_b_dir / "cases.yaml").write_text(
        "cases:\n"
        "  - case_id: 1\n"
        "    input_shape: [[64, 128], [128, 256]]\n"
        "    dtype: [float16, float16]\n"
        "  - case_id: 2\n"
        "    input_shape: [[128, 256], [256, 512]]\n"
        "    dtype: [float16, float16]\n",
        encoding="utf-8",
    )

    recs, scanned, dedup_reps = load_official_reports([rep_dir1, rep_dir2], ops_root=tmp_ops)
    assert scanned == 2
    assert dedup_reps == 2
    assert len(recs) == 4

    summary = run_cost_model_backtest([rep_dir1, rep_dir2], ops_root=tmp_ops)
    assert summary.unique_operator_cases == 4
    assert summary.p0_p1_loco_cv.mean_intra_group_spearman_rho == pytest.approx(1.0)
    assert summary.enable_prescreen_recommended is False
