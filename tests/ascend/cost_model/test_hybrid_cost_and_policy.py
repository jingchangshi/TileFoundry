"""Consolidated L3 tests for HybridAscendCostModel, cost gate policy, P0-P2, and Stage-6."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.ascend950pr_ops.rms_norm.kernels import (
    _build_rms_norm_prim_func,
    _select_rms_norm_config,
)
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import run_l0_l1_round_filter
from tilefoundry_ascend.agent.kb import (
    SymptomFamily,
    build_predicate_context,
    load_seed_rules,
    rank_symptom_families,
)
from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator import tilesize_prescreen as tp
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _parse_cce_res_usage_with_summary,
    build_stage_b_driver_source,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import compute_static_dynamic_residual
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.ascend.cost_model import (
    DEFAULT_LAUNCH_OVERHEAD_US,
    DEFAULT_RESIDUAL_BASIS_SPECS,
    Ascend950PipeScheduleEvaluator,
    BishengResourceSummary,
    CaseObservationRecord,
    CostFeatures,
    CostGateDecision,
    CostModelMode,
    ExperienceLake,
    HybridAscendCostModel,
    MicroarchProfile,
    build_cost_features_from_ledger,
    evaluate_cost_gate_policy,
    evaluate_cost_model_ranking,
    extract_microarch_profile_from_tirx,
    is_paradigm_shift_exempt,
    parse_bisheng_res_usage,
    project_case_workload,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    _penalty_scalar_compute_loop_fallback,
)
from tilefoundry_ascend.ascend.passes.observation_space import (
    ObservationVector5D,
    compute_transfer_inflation_ratio,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def _make_vector_ledger(
    *,
    grid_blocks: int = 56,
    active_grid_blocks: int | None = None,
    min_gm_dma_bytes: int = 4096,
    ub_scalar_compute_loops: int = 0,
    ub_bank_conflict_risk: float = 0.0,
    unaligned_dma_padding_bytes: int = 0,
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 4000, "AIV0.PIPE_MTE2": 800},
        waterlines={},
        grid_blocks=grid_blocks,
        active_grid_blocks=active_grid_blocks,
        gm_roundtrip_bytes=65536,
        dominant_contig_bytes=32768,
        min_gm_dma_bytes=min_gm_dma_bytes,
        ub_scalar_compute_loops=ub_scalar_compute_loops,
        ub_bank_conflict_risk=ub_bank_conflict_risk,
        unaligned_dma_padding_bytes=unaligned_dma_padding_bytes,
    )


def _make_cube_ledger(
    *,
    grid_blocks: int = 28,
    active_grid_blocks: int | None = None,
    l1_bank_conflict_risk: float = 0.0,
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIC.PIPE_M": 4000, "AIV0.PIPE_V": 1000, "AIV0.PIPE_MTE2": 800},
        waterlines={},
        grid_blocks=grid_blocks,
        active_grid_blocks=active_grid_blocks,
        gm_roundtrip_bytes=65536,
        dominant_contig_bytes=32768,
        min_gm_dma_bytes=4096,
        l1_bank_conflict_risk=l1_bank_conflict_risk,
    )


@pytest.mark.parametrize(
    ("kwargs", "exp_mode", "exp_engaged", "exp_guidance", "exp_prune", "exp_reason"),
    [
        ({}, CostModelMode.OFF, False, False, False, "disabled_by_config"),
        (
            {
                "mode": CostModelMode.SHADOW,
                "bucket_clean_samples": 30,
                "out_of_sample_spearman_rho": 0.65,
            },
            CostModelMode.SHADOW,
            False,
            False,
            False,
            "shadow_observation_only",
        ),
        (
            {
                "mode": CostModelMode.ADVISORY,
                "bucket_clean_samples": 2,
                "out_of_sample_spearman_rho": 0.8,
            },
            CostModelMode.ADVISORY,
            False,
            False,
            False,
            "insufficient_bucket_samples",
        ),
        (
            {
                "mode": CostModelMode.ADVISORY,
                "bucket_clean_samples": 30,
                "out_of_sample_spearman_rho": 0.65,
            },
            CostModelMode.ADVISORY,
            True,
            True,
            False,
            "advisory_t_stat_validated",
        ),
        (
            {"enable_prescreen_flag": False, "mode": "advisory"},
            CostModelMode.OFF,
            False,
            False,
            False,
            "disabled_by_config",
        ),
    ],
)
def test_cost_gate_policy_transitions_and_paradigm_exemption(
    kwargs: dict[str, object],
    exp_mode: CostModelMode,
    exp_engaged: bool,
    exp_guidance: bool,
    exp_prune: bool,
    exp_reason: str,
) -> None:
    """Verify CostGateDecision transitions and creative/cross-topology pruning exemptions."""
    dec = evaluate_cost_gate_policy(**kwargs)
    assert isinstance(dec, CostGateDecision)
    assert (dec.mode, dec.engaged, dec.allow_prompt_guidance, dec.allow_candidate_pruning) == (
        exp_mode,
        exp_engaged,
        exp_guidance,
        exp_prune,
    )
    assert dec.reason == exp_reason

    axes_a = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    axes_b = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    assert is_paradigm_shift_exempt(axes_a, axes_a, is_creative=False) is False
    assert is_paradigm_shift_exempt(axes_a, axes_a, is_creative=True) is True
    assert (
        is_paradigm_shift_exempt(
            axes_a, axes_a, cand_workspace_stages=2, parent_workspace_stages=0
        )
        is True
    )
    assert is_paradigm_shift_exempt(axes_a, axes_b) is True

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    parent_l = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIC.PIPE_M": 400},
        gm_roundtrip_bytes=16384,
        grid_blocks=28,
        waterlines={},
    )
    slow_l = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=2000,
        pipe_cycles={"AIC.PIPE_M": 20000},
        gm_roundtrip_bytes=1_048_576,
        grid_blocks=28,
        waterlines={},
    )
    ranked = rank_stage_a_candidates_with_cost_model(
        [
            ("exploit_slow", "", StageAResult(passed=True, prim_func=None, ledger=slow_l)),
            ("creative_slow", "", StageAResult(passed=True, prim_func=None, ledger=slow_l)),
        ],
        cost_model=model,
        axes=axes_b,
        parent_ledger=parent_l,
        creative_candidate_ids={"creative_slow"},
    )
    by_id = {item.candidate_id: item for item in ranked}
    assert by_id["exploit_slow"].coarse_reject_exempt is False
    assert by_id["creative_slow"].coarse_reject_exempt is True


def test_ledger_to_cost_features_bridge_and_prescreen_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 4 ledger signals raise Stage-A cost, coupled active cores, and prescreen shapes."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    vec_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    clean_cf = build_cost_features_from_ledger(_make_vector_ledger(), target=target, axes=vec_axes)
    clean_pred = model.predict(clean_cf, bisheng_res=None)
    assert clean_pred.pred_us == model.predict(clean_cf, bisheng_res=None).pred_us
    assert "dcache_bypass_narrow_store" not in clean_cf.extra_signals
    assert (clean_cf.ub_bank_conflict_ratio, clean_cf.has_unaligned_tail, clean_cf.active_aiv_cores) == (
        0.0,
        False,
        56,
    )

    for ledger_kw, check in (
        (
            {"min_gm_dma_bytes": 8},
            lambda cf, p: cf.extra_signals.get("dcache_bypass_narrow_store") == pytest.approx(4.0)
            and p.pred_us > clean_pred.pred_us,
        ),
        (
            {"ub_scalar_compute_loops": 1},
            lambda cf, p: cf.extra_signals.get("scalar_compute_loop_fallback") == pytest.approx(1.0)
            and p.pred_us > clean_pred.pred_us,
        ),
        (
            {"ub_bank_conflict_risk": 0.5},
            lambda cf, p: cf.ub_bank_conflict_ratio == pytest.approx(0.5)
            and p.bank_conflict_us > 0.0
            and p.pred_us > clean_pred.pred_us,
        ),
        (
            {"unaligned_dma_padding_bytes": 16},
            lambda cf, p: cf.has_unaligned_tail is True
            and p.tier1_penalty_us > clean_pred.tier1_penalty_us
            and p.pred_us > clean_pred.pred_us,
        ),
        (
            {"grid_blocks": 56, "active_grid_blocks": 4},
            lambda cf, p: (cf.active_aic_cores, cf.active_aiv_cores) == (4, 4)
            and p.pred_us > clean_pred.pred_us,
        ),
    ):
        cf = build_cost_features_from_ledger(
            _make_vector_ledger(**ledger_kw), target=target, axes=vec_axes
        )
        assert check(cf, model.predict(cf, bisheng_res=None))

    coupled_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    clean_cube_cf = build_cost_features_from_ledger(
        _make_cube_ledger(grid_blocks=28, active_grid_blocks=28), target=target, axes=coupled_axes
    )
    clean_cube_pred = model.predict(clean_cube_cf, bisheng_res=None)
    assert (clean_cube_cf.active_aic_cores, clean_cube_cf.active_aiv_cores) == (28, 56)

    l1_conflict_cf = build_cost_features_from_ledger(
        _make_cube_ledger(l1_bank_conflict_risk=0.6), target=target, axes=coupled_axes
    )
    l1_conflict_pred = model.predict(l1_conflict_cf, bisheng_res=None)
    assert l1_conflict_cf.l1_half_bank_split is False and l1_conflict_cf.l1_stages >= 2
    assert l1_conflict_pred.bank_conflict_us > clean_cube_pred.bank_conflict_us

    explicit_split_cf = build_cost_features_from_ledger(
        _make_cube_ledger(l1_bank_conflict_risk=0.6),
        target=target,
        axes=coupled_axes,
        live_plan={"policy": {"l1_half_bank_split": True, "l1_stages": 2}},
    )
    assert model.predict(explicit_split_cf, bisheng_res=None).bank_conflict_us == pytest.approx(
        clean_cube_pred.bank_conflict_us
    )
    override_cf = build_cost_features_from_ledger(
        _make_cube_ledger(grid_blocks=28, active_grid_blocks=4),
        target=target,
        axes=coupled_axes,
        live_plan={"policy": {"active_aic_cores": 14, "active_aiv_cores": 28}},
    )
    assert (override_cf.active_aic_cores, override_cf.active_aiv_cores) == (14, 28)

    seen_shapes: list[object] = []

    def fake_stage_a(*_args: object, **kwargs: object) -> gf.StageAResult:
        seen_shapes.append(kwargs.get("ordered_tensor_shapes"))
        return gf.StageAResult(passed=True, prim_func=None, ledger=_make_vector_ledger())

    monkeypatch.setattr(gf, "run_stage_a_static", fake_stage_a)
    shapes = [[16, 128], [16, 128]]
    summary = tp.prescreen_live_ir_neighborhood(
        "def build_kernel(bm=64, bn=64, bk=64, stages=2):\n    return None\n",
        factory_name="build_kernel",
        center=tp.TileConfigCandidate(bm=64, bn=64, bk=64, stages=2),
        max_variants=2,
        ordered_tensor_shapes=shapes,
    )
    assert "Live-IR Tile Prescreen Summary:" in summary and seen_shapes == [shapes, shapes]


def test_p0_p1_p2_scaling_detectors_hierarchical_ridge_and_pruning(tmp_path: Path) -> None:
    """Verify P0 chip scaling, P1 CCE dump & 7 detectors, hierarchical ridge, and P2 Top-K."""
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
    assert cf.total_tiles == 28 and cf.gm_bytes >= 4_000_000.0
    assert cf.fixpipe_bytes > 0.0 and cf.cv_sync_events >= 4

    model = HybridAscendCostModel(target)
    tiny = model.predict(
        CostFeatures(gm_bytes=4096.0, contig_burst_bytes=4096, kernel_launches=1, topology_axes=axes)
    )
    assert tiny.launch_overhead_us == pytest.approx(DEFAULT_LAUNCH_OVERHEAD_US)
    pred_coupled = model.predict(cf)
    assert pred_coupled.fixpipe_floor_us > 0.0 and pred_coupled.cv_sync_us > 0.0

    seq_path, batch_path = tmp_path / "seq.jsonl", tmp_path / "batch.jsonl"
    m_seq = HybridAscendCostModel(target, calibration_path=seq_path)
    m_batch = HybridAscendCostModel(target, calibration_path=batch_path)
    obs = [
        (cf, pred_coupled.pred_us * 1.4),
        (CostFeatures(gm_bytes=8192.0, vec_flops=1e9, topology_axes=axes), 9.5),
        (CostFeatures(gm_bytes=65536.0, cube_flops=1e11, topology_axes=axes), 80.0),
    ]
    for feat, lat in obs:
        m_seq.observe(feat, lat, persist=True)
    m_batch.observe_batch(obs, persist=True)
    assert len(m_seq.weights) == len(DEFAULT_RESIDUAL_BASIS_SPECS) == 12
    assert math.isclose(m_seq.predict(cf).pred_us, m_batch.predict(cf).pred_us, rel_tol=1e-9)
    reloaded = HybridAscendCostModel(target, calibration_path=batch_path)
    assert reloaded.sample_count == 3

    cand_path = tmp_path / "cand.py"
    cand_path.write_text("# dummy", encoding="utf-8")
    driver = build_stage_b_driver_source(cand_path, kernel_dump_path=tmp_path / "lowered.cce")
    assert "kernel_dump_path" in driver and "kernel_source" in driver

    cce_src = """
    extern __simd_vf__ void vf_proto();
    __simd_vf__ void vf_a() { pipe_barrier(PIPE_ALL); LocalMemBar(); dcache_invalidate(); }
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
    spill_b, vloop_hit, summary = _parse_cce_res_usage_with_summary(
        "Vector Register Usage: 96\nRegister Spill: 512 bytes\ncannot use vloop_v2",
        asm_or_dump_text=cce_src,
    )
    assert spill_b == 512 and vloop_hit is False
    assert summary.software_loops >= 1 and summary.has_local_membar_in_vf_call is True
    assert summary.vf_fusion_breaks == 1 and summary.dual_issue_ipc_collapse > 0.0
    assert summary.dcache_dcci_flushes >= 1

    aiv_feat = CostFeatures(
        vec_flops=2e10,
        gm_bytes=8_000_000.0,
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
    )
    cv_feat = CostFeatures(cube_flops=1e11, vec_flops=1e10, gm_bytes=8_000_000.0, topology_axes=axes)
    base_aiv, base_cv = model.predict(aiv_feat).pred_us, model.predict(cv_feat).pred_us
    model.observe_batch(
        [(aiv_feat, base_aiv * 2.2)] * 3 + [(cv_feat, base_cv * 0.68)] * 3, persist=False
    )
    assert model.predict(aiv_feat).pred_us > base_aiv * 1.4
    assert model.predict(cv_feat).pred_us < base_cv * 0.9

    rep = evaluate_cost_model_ranking(
        [10.0, 20.0, 30.0, 5.0, 15.0],
        [11.0, 22.0, 33.0, 5.5, 16.5],
        groups=["opA", "opA", "opA", "opB", "opB"],
    )
    assert rep.mean_intra_group_spearman_rho == pytest.approx(1.0) and rep.top1_regret_ratio == 0.0

    ledger_fast = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIC.PIPE_M": 500},
        gm_roundtrip_bytes=16384,
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
    l1_calls: list[str] = []
    checks = asyncio.run(
        run_l0_l1_round_filter(
            [("w0", "SLOW"), ("w1", "FAIL_A"), ("w2", "FAST")],
            patch_fn=lambda s: s,
            l0_checker=lambda s: StageAResult(
                passed=("FAIL_A" not in s),
                prim_func=None,
                ledger=(None if "FAIL_A" in s else (ledger_fast if "FAST" in s else ledger_slow)),
            ),
            l1_compiler=lambda s: l1_calls.append(s) or StageBResult(passed=True),
            max_syntax_repairs=0,
            stage_a_cost_ranker=lambda items: [
                (item.candidate_id, item.pred_us)
                for item in rank_stage_a_candidates_with_cost_model(items, cost_model=model)
            ],
            cost_prescreen_top_k=1,
        )
    )
    assert l1_calls == ["FAST"]
    assert (checks[0].failed_tier, checks[1].failed_tier, checks[2].passed) == (
        "cost_prescreen",
        "stage_a",
        True,
    )


def test_stage6_scalar_and_narrow_penalties_anomaly_guards_and_inflation(
    tmp_path: Path,
) -> None:
    """Verify Stage-6 Tier-0 penalties, >2.5 anomaly outlier guards, and transfer inflation."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    clean = CostFeatures(
        vec_flops=4.0e10, gm_bytes=12.0e6, total_tiles=56, contig_burst_bytes=32768, topology_axes=axes
    )
    scalar_bad = CostFeatures(
        vec_flops=4.0e10,
        gm_bytes=12.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        extra_signals={"scalar_compute_loop_fallback": 1.0},
    )
    pred_clean, pred_scalar = model.predict(clean), model.predict(scalar_bad)
    assert pred_scalar.vec_floor_us == pytest.approx(pred_clean.vec_floor_us * 128.0)
    assert pred_scalar.pred_us > pred_clean.pred_us * 10.0

    narrow_clean = CostFeatures(
        vec_flops=1.0e7, gm_bytes=32.0e6, total_tiles=56, contig_burst_bytes=32768, topology_axes=axes
    )
    narrow_bad = CostFeatures(
        vec_flops=1.0e7,
        gm_bytes=32.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        extra_signals={"dcache_bypass_narrow_store": 32.0},
    )
    assert model.predict(narrow_bad).hbm_floor_us == pytest.approx(
        model.predict(narrow_clean).hbm_floor_us * 16.5
    )

    bounded = _penalty_scalar_compute_loop_fallback(
        BishengResourceSummary(extra_signals={"scalar_compute_loop_fallback": 4.0}),
        {"base_ref": 20.0, "vec_floor_us": 5.0},
    )
    assert bounded == pytest.approx(7.0)

    lake = ExperienceLake(tmp_path / "lake")
    guarded = HybridAscendCostModel(target, anomaly_lake=lake)
    wl = CostFeatures(
        vec_flops=2.0e10,
        gm_bytes=16.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        extra_signals={"custom_diag": 2.0},
    )
    expected_analytical = guarded.predict(wl).pred_us
    for _ in range(4):
        assert guarded.observe(wl, expected_analytical * 1.5) == pytest.approx(expected_analytical)
    assert guarded.sample_count == 4
    w_before = dict(guarded.weights_by_name)
    guarded.observe(wl, expected_analytical * math.exp(2.6))
    guarded.observe(wl, expected_analytical * math.exp(-2.6))
    assert guarded.sample_count == 4 and guarded.weights_by_name == w_before
    rows = [
        json.loads(ln)
        for ln in (lake.root_dir / "unmodeled_anomalies.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    assert len(rows) == 2 and rows[0]["log_residual"] == pytest.approx(2.6)

    floor_bytes = 150 * 1024 * 1024
    spill_bytes = 32 * 1024 * 1024
    roofline = {"min_gm_bytes": floor_bytes, "compute_floor_us": 12.0}
    clean_l = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 200},
        waterlines={},
        grid_blocks=16,
        gm_roundtrip_bytes=0,
        gm_traffic_scope="full_grid",
    )
    spill_l = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 200},
        waterlines={},
        grid_blocks=16,
        gm_roundtrip_bytes=spill_bytes,
        gm_traffic_scope="full_grid",
    )
    assert compute_transfer_inflation_ratio(clean_l, roofline=roofline) == pytest.approx(1.0)
    assert ObservationVector5D.from_tiers(
        spill_l, telemetry={"dur_us": 25.0}, roofline=roofline
    ).transfer_inflation_ratio == pytest.approx((floor_bytes + spill_bytes) / float(floor_bytes))
    measured_res = compute_static_dynamic_residual(
        clean_l,
        {"dur_us": 30.0, "hbm_read_bytes": 120 * 1024 * 1024, "hbm_write_bytes": 60 * 1024 * 1024},
        roofline=roofline,
    )
    assert measured_res.transfer_inflation_ratio == pytest.approx(1.2)
    rec = CaseObservationRecord(
        run_id="r",
        operator="op",
        regime_tag="s",
        cand_sha="abc",
        eval_tier="l2_pass",
        case_num=1,
        measured_us=25.0,
        accuracy_passed=True,
        min_gm_bytes=65536.0,
        compute_floor_us=10.5,
        transfer_inflation_ratio=1.35,
    )
    rec_d = rec.to_dict()
    assert CaseObservationRecord.from_dict(rec_d).transfer_inflation_ratio == pytest.approx(1.35)
    for k in ("min_gm_bytes", "compute_floor_us", "transfer_inflation_ratio"):
        rec_d.pop(k)
    assert CaseObservationRecord.from_dict(rec_d).min_gm_bytes == 0.0


def test_symptom_ranking_seed_rules_and_anchor_specialization() -> None:
    """Verify L2>L1>L0 symptom ranking, R34/R19/R36/R37 predicates, and rms_norm replay."""
    high_vec = dict(
        rank_symptom_families(
            {
                "measured_aiv_vec_ratio": 0.85,
                "measured_aic_mac_ratio": 0.0,
                "l0_cube_cycles": 0,
                "l0_vec_cycles": 0,
                "has_l1_compiler_summary": True,
                "l1_vec_regs": 0,
            }
        )
    )
    assert high_vec[SymptomFamily.SCALAR_LOOP_TRAP] == 0.0

    capped = dict(
        rank_symptom_families(
            {
                "measured_aiv_vec_ratio": 0.25,
                "scalar_compute_loop_fallback": 2.0,
                "measured_mte2_wait_ratio": 0.70,
                "l2_hit_rate": 0.20,
            }
        )
    )
    assert capped[SymptomFamily.SCALAR_LOOP_TRAP] == pytest.approx(0.30 / 1.10, rel=1e-4)

    rules = {r.rule_id: r for r in load_seed_rules()}
    assert len(rules) >= 41
    simtvf_l = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 0, "AIV0.PIPE_MTE2": 100},
        waterlines={},
        opaque_dynamic_regions=("simt_vf",),
        dma_burst_efficiency_ratio=0.72,
    )
    ctx_r34 = build_predicate_context(
        simtvf_l,
        residual={
            "l2_aiv_vec_ratio": 0.25,
            "l2_aic_mac_ratio": 0.0,
            "effective_compute_eta": 0.18,
            "l2_hit_rate": 0.40,
            "transfer_inflation_ratio": 1.15,
        },
        extra={"contig_atom_ratio": 0.25},
    )
    assert rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"].matches(ctx_r34) is False
    assert rules["R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE"].matches(ctx_r34) is True

    target = AscendTarget("ascend.950pr")
    cm = HybridAscendCostModel(target)
    cfg = _select_rms_norm_config(1024, "bfloat16", m_rows=4096, schedule_mode="tuned")
    fn = _build_rms_norm_prim_func(
        d_dim=1024,
        tl_dtype="bfloat16",
        tile_m=int(cfg["tile_m"]),
        num_stages=int(cfg["num_stages"]),
        schedule_mode="tuned",
    )
    res = inspect_tilelang_primfunc(
        fn,
        target=target,
        pass_configs=cfg["pass_configs"],
        compile_flags=cfg["compile_flags"],
        var_bindings={"m_rows": 4096},
    )
    prof = extract_microarch_profile_from_tirx(fn, var_bindings={"m_rows": 4096})
    wl = project_case_workload(
        res.ledger,
        microarch_profile=prof,
        case_dims={"B": 32, "S": 128, "D": 1024, "m_rows": 4096},
        policy=cfg,
        axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        target=target,
    )
    assert wl["profile_is_per_core_full_loop"] is True
    assert abs(math.log(49.37 / max(1e-6, cm.predict(wl).pred_us))) < 1.5

    fn_simt = _build_rms_norm_prim_func(
        d_dim=16, tl_dtype="bfloat16", tile_m=256, num_stages=2, schedule_mode="tuned"
    )
    prof_simt = extract_microarch_profile_from_tirx(fn_simt, var_bindings={"m_rows": 4096})
    assert prof_simt.scalar_ops > 0 and prof_simt.int_alu_ops == 0
    assert Ascend950PipeScheduleEvaluator(target).evaluate_profile(prof_simt, total_tiles=56).ave_compute_ns < 5000.0
