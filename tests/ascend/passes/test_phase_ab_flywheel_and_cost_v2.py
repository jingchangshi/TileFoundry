"""Comprehensive tests for Phase A' (Experience Lake & Flywheel) and Phase B' (Ascend 950 Cost v2)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import tilelang.ascend.language as T

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.pipeline import PipelineSpec
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, Sequential
from tilefoundry.ir.types import DType, StorageKind, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec, PerfTargetSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.campaign.marathon import MarathonConfig
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    AstEditChunk,
    FlywheelDistiller,
    distill_causal_rule,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker,
    append_kb_event_shard,
    save_distilled_rule_file,
)
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel, StageAResult, StageBResult
from tilefoundry_ascend.ascend.cost_model import (
    DIRECTION_SLOPE_MULT,
    Ascend950PipeScheduleEvaluator,
    CandidateArtifactRecord,
    CaseObservationRecord,
    CostFeatures,
    ExperienceLake,
    HybridAscendCostModel,
    MicroarchProfile,
    build_cost_features_from_ledger,
    compute_source_sha256,
    extract_microarch_profile_from_cce,
    extract_microarch_profile_from_tirx,
    merge_microarch_profiles,
    normalize_ns_to_z3_ticks,
    project_case_workload,
)
from tilefoundry_ascend.ascend.ir.tir.instructions import AscendGmToUb, AscendUbToGm
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    extract_tir_loop_problem,
    solve_ascend_pipeline,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_microarch_profile_tir_cce_extraction_and_merge() -> None:
    """Verify TIR + CCE MicroarchProfile extractors capture AVE, DMA, loops, and Track-B merge."""

    @T.prim_func
    def _vec_f16_kernel(X: T.Tensor((128, 128), "float16"), Y: T.Tensor((128, 128), "float16")):
        with T.Kernel(4) as _:
            a_ub = T.alloc_shared((32, 128), "float16")
            b_ub = T.alloc_shared((32, 128), "float16")
            for k in T.Pipelined(4, num_stages=2):
                T.copy(X[k * 32 : (k + 1) * 32, 0:128], a_ub)
                for i, j in T.Parallel(32, 128):
                    b_ub[i, j] = T.exp(a_ub[i, j])
                T.copy(b_ub, Y[k * 32 : (k + 1) * 32, 0:128])

    tir_prof = extract_microarch_profile_from_tirx(_vec_f16_kernel)
    assert tir_prof.source_kind == "tirx"
    assert not tir_prof.is_empty
    assert tir_prof.gm_to_ub_bytes >= 4 * 32 * 128 * 2
    assert tir_prof.ub_to_gm_bytes >= 4 * 32 * 128 * 2
    assert tir_prof.ub_load_bytes > 0
    assert tir_prof.ub_store_bytes > 0
    assert tir_prof.sfu_f16_ops > 0
    assert tir_prof.pipeline_depth >= 2
    assert tir_prof.inner_loop_trips >= 4

    cce_src = """
    extern "C" __global__ __aicore__ void k_part1() {
        Mmad(c, a, b);
        copy_gm_to_cbuf(l1, gm);
        copy_cbuf_to_ca(l0a, l1);
        copy_matrix_cc_to_ubuf(ub, l0c);
        CrossCoreSetFlag<2>(0);
    }
    extern "C" __global__ __aicore__ void k_part2() {
        CrossCoreWaitFlag(0);
        half v = vexp(x);
        float z = vexp(y);
        __shared__ half tile[32][64];
        __simt_vf__();
        copy_ubuf_to_cbuf(l1, ub);
        pipe_barrier(PIPE_ALL);
    }
    """
    cce_prof = extract_microarch_profile_from_cce(cce_src)
    assert cce_prof.source_kind == "cce"
    assert cce_prof.subkernel_count == 2
    assert cce_prof.cube_mmad_count == 1
    assert cce_prof.cv_sync_barriers == 2
    assert cce_prof.pipe_all_barriers == 1
    assert cce_prof.sfu_f16_ops >= 1
    assert cce_prof.sfu_f32_ops >= 1
    assert cce_prof.ub_to_l1_bytes > 0
    assert cce_prof.l0c_to_ub_bytes > 0
    assert cce_prof.simt_shm_conflict_factor >= 1.5

    merged = merge_microarch_profiles(tir_prof, cce_prof)
    assert merged.source_kind == "merged"
    # Preserves exact TIR byte volumes over CCE approximations for populated edges
    assert merged.gm_to_ub_bytes == tir_prof.gm_to_ub_bytes
    assert merged.ub_to_gm_bytes == tir_prof.ub_to_gm_bytes
    # Fills missing Track-B / post-codegen signals from CCE
    assert merged.subkernel_count == 2
    assert merged.cv_sync_barriers == 2
    assert merged.pipe_all_barriers == 1
    assert merged.ub_to_l1_bytes == cce_prof.ub_to_l1_bytes


def test_pipe_schedule_evaluator_ascend950_physics_and_z3_ticks() -> None:
    """Verify SFU F16>F32, 2L/1S UB ports, UB->L1/L0C->UB slopes, drain bubble, and Z3 solve."""
    ev = Ascend950PipeScheduleEvaluator(AscendTarget("ascend.950pr"))

    # 1. SFU F16 vs F32 penalty (SFU F16 weight 4.0 vs SFU F32 weight 2.0)
    p_f16 = MicroarchProfile(sfu_f16_ops=512, ub_load_bytes=65536, ub_store_bytes=65536, source_kind="tirx")
    p_f32 = MicroarchProfile(sfu_f32_ops=512, ub_load_bytes=65536, ub_store_bytes=65536, source_kind="tirx")
    r_f16 = ev.evaluate_profile(p_f16)
    r_f32 = ev.evaluate_profile(p_f32)
    assert r_f16.ave_compute_ns > r_f32.ave_compute_ns
    assert r_f16.total_ns > r_f32.total_ns

    # 2. 2-Load / 1-Store UB asymmetry
    p_store_heavy = MicroarchProfile(ub_load_bytes=32768, ub_store_bytes=32768, source_kind="tirx")
    p_load_only = MicroarchProfile(ub_load_bytes=65536, ub_store_bytes=0, source_kind="tirx")
    assert ev.evaluate_profile(p_store_heavy).ave_io_ns > ev.evaluate_profile(p_load_only).ave_io_ns

    # 3. Directional MTE3 (UB->L1 0.5x) and FixPipe (L0C->UB 0.877x)
    assert DIRECTION_SLOPE_MULT["ub_to_l1"] == pytest.approx(0.50)
    assert DIRECTION_SLOPE_MULT["l0c_to_ub"] == pytest.approx(0.877)
    r_ub_l1 = ev.evaluate_profile(MicroarchProfile(ub_to_l1_bytes=131072, source_kind="tirx"))
    r_ub_gm = ev.evaluate_profile(MicroarchProfile(ub_to_gm_bytes=131072, source_kind="tirx"))
    assert r_ub_l1.pipe_ns["MTE3"] < r_ub_gm.pipe_ns["MTE3"]

    r_l0c_ub = ev.evaluate_profile(MicroarchProfile(l0c_to_ub_bytes=131072, source_kind="tirx"))
    r_l0c_gm = ev.evaluate_profile(MicroarchProfile(l0c_to_gm_bytes=131072, source_kind="tirx"))
    assert r_l0c_ub.pipe_ns["FIXPIPE"] < r_l0c_gm.pipe_ns["FIXPIPE"]

    # 4. Pipeline depth=2 reduces drain bubble / loop_total_ns compared to depth=1
    base_pipe_prof = dict(
        gm_to_ub_bytes=262144,
        ub_load_bytes=65536,
        ub_store_bytes=65536,
        fp_alu_ops=4096,
        inner_loop_trips=8,
        source_kind="tirx",
    )
    r_d1 = ev.evaluate_profile(MicroarchProfile(pipeline_depth=1, **base_pipe_prof))
    r_d2 = ev.evaluate_profile(MicroarchProfile(pipeline_depth=2, **base_pipe_prof))
    assert r_d2.loop_total_ns < r_d1.loop_total_ns

    # 5. Wire through HybridAscendCostModel.predict
    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    assert model.predict(CostFeatures(microarch_profile=p_f16.to_dict())).pred_us > model.predict(
        CostFeatures(microarch_profile=p_f32.to_dict())
    ).pred_us
    assert model.predict(CostFeatures(microarch_profile=MicroarchProfile(pipeline_depth=2, **base_pipe_prof).to_dict())).pred_us < model.predict(
        CostFeatures(microarch_profile=MicroarchProfile(pipeline_depth=1, **base_pipe_prof).to_dict())
    ).pred_us

    # 6. Continuous ns estimation + Z3 normalization + solver execution
    ticks = normalize_ns_to_z3_ticks([12.5, 450.0, 2400.0], max_ticks=12)
    assert len(ticks) == 3
    assert all(1 <= t <= 12 for t in ticks)
    assert ticks[0] <= ticks[1] <= ticks[2]

    d_dyn = DimVar("D", 32, 1024)
    gm_ty = TensorType((4, d_dyn), DType.f16, None, StorageKind.GMEM)
    ub_ty = TensorType((d_dyn,), DType.f16, None, StorageKind.SMEM)
    sc = TensorType.umat_scalar()
    k = Var("k", type=sc)
    gm_in = Var("gm_in", type=gm_ty)
    gm_out = Var("gm_out", type=gm_ty)
    ub_var = Var("ub_buf", type=ub_ty)
    loop = For(
        k,
        Constant(0, type=sc),
        Constant(4, type=sc),
        Constant(1, type=sc),
        Sequential(
            (
                LetStmt(ub_var, Call(AllocTensor(tensor_type=ub_ty), (), type=ub_ty)),
                Evaluate(AscendGmToUb(), (gm_in, ub_var)),
                Evaluate(AscendUbToGm(), (ub_var, gm_out)),
            )
        ),
        pipeline=PipelineSpec(max_stage=1, max_buffer_versions=2, enable_offset=True),
    )
    problem, _ = extract_tir_loop_problem(loop, dim_hints={"D": 512})
    assert all(1 <= int(t.latency) <= 12 for t in problem.tasks)
    sched_res = solve_ascend_pipeline(problem)
    assert sched_res.assignment.ii >= 1
    reg_res = RegimePiecewiseSolver().solve_regime(
        {"regime_tag": "vec", "representative_dims": {"M": 4, "D": 512}},
        op_name="rms_norm",
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        loop=loop,
    )
    assert reg_res.ii is not None and reg_res.ii >= 1


def test_experience_lake_dedup_replay_and_two_head_calibrator(tmp_path: Path) -> None:
    """Verify sharded ExperienceLake dedup, idempotent replay, Head-1 MSProf bounds, Head-2 delta ranking."""
    lake_dir = tmp_path / "exp_lake"
    lake = ExperienceLake(lake_dir)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    parent_src = "def kernel_v0():\n    pass\n"
    fast_src = "def kernel_v1():\n    # double buffer\n    pass\n"
    parent_sha = compute_source_sha256(parent_src)
    fast_sha = compute_source_sha256(fast_src)

    parent_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=180,
        pipe_cycles={"AIV0.PIPE_V": 900, "AIV0.PIPE_MTE2": 600},
        gm_roundtrip_bytes=65536,
        grid_blocks=28,
        waterlines={},
        microarch_profile=MicroarchProfile(
            fp_alu_ops=1024,
            gm_to_ub_bytes=65536,
            ub_to_gm_bytes=65536,
            pipeline_depth=1,
            inner_loop_trips=4,
            source_kind="tirx",
        ).to_dict(),
    )
    fast_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=95,
        pipe_cycles={"AIV0.PIPE_V": 800, "AIV0.PIPE_MTE2": 450},
        gm_roundtrip_bytes=65536,
        grid_blocks=28,
        waterlines={},
        microarch_profile=MicroarchProfile(
            fp_alu_ops=1024,
            gm_to_ub_bytes=65536,
            ub_to_gm_bytes=65536,
            pipeline_depth=2,
            inner_loop_trips=4,
            source_kind="tirx",
        ).to_dict(),
    )

    p1 = lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=parent_sha,
            operator="level2/softmax",
            regime_tag="large_row",
            topology_axes=axes.to_dict(),
            base_ledger=parent_ledger.to_dict(),
            microarch_profile=dict(parent_ledger.microarch_profile),
        )
    )
    # Second write is deduplicated (same sha)
    p2 = lake.register_candidate(
        CandidateArtifactRecord(cand_sha=parent_sha, operator="level2/softmax")
    )
    assert p1 == p2
    assert (lake.candidates_dir / f"{parent_sha}.json").is_file()

    lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=fast_sha,
            operator="level2/softmax",
            regime_tag="large_row",
            parent_sha=parent_sha,
            topology_axes=axes.to_dict(),
            base_ledger=fast_ledger.to_dict(),
            microarch_profile=dict(fast_ledger.microarch_profile),
        )
    )
    # Two run shards append without collision
    lake.append_observation(
        CaseObservationRecord(
            run_id="run_a",
            operator="level2/softmax",
            regime_tag="large_row",
            cand_sha=parent_sha,
            eval_tier="baseline",
            case_num=1,
            measured_us=120.0,
            accuracy_passed=True,
            case_dims={"M": 1024, "N": 2048},
            t_hw_us=45.0,
            msprof_pmu={"dur_us": 120.0, "aiv_vec": 0.70, "aiv_mte2": 0.35},
        )
    )
    lake.append_observation(
        CaseObservationRecord(
            run_id="run_b",
            operator="level2/softmax",
            regime_tag="large_row",
            cand_sha=fast_sha,
            eval_tier="l2_pass",
            case_num=1,
            measured_us=68.0,
            accuracy_passed=True,
            parent_sha=parent_sha,
            case_dims={"M": 1024, "N": 2048},
            t_hw_us=45.0,
            msprof_pmu={"dur_us": 68.0, "aiv_vec": 0.65, "aiv_mte2": 0.25},
        )
    )
    assert len(list(lake.runs_dir.glob("*.jsonl"))) == 2
    pairs = lake.extract_pairwise_deltas()
    assert len(pairs) == 1

    # Replay into CostModel idempotently
    cm = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    stats1 = cm.replay_experience_lake(lake_dir)
    assert stats1 == {"observations": 2, "msprof_events": 2, "pair_deltas": 1}
    stats2 = cm.replay_experience_lake(lake_dir)
    assert stats2 == {"observations": 0, "msprof_events": 0, "pair_deltas": 0}
    assert cm.sample_count == 2
    assert cm.pair_delta_count == 1
    assert cm.msprof_event_count == 2

    # Check parametric projection handles varying case_dims cleanly
    w_small = project_case_workload(fast_ledger, case_dims={"M": 128, "N": 128}, axes=axes)
    w_large = project_case_workload(fast_ledger, case_dims={"M": 4096, "N": 4096}, axes=axes)
    assert float(w_large["gm_bytes"]) > float(w_small["gm_bytes"])
    assert int(w_large["total_tiles"]) >= int(w_small["total_tiles"])

    # Head-1 msprof pipe scale multipliers remain Bayes-shrunk and strictly bounded in [0.5, 2.0]
    cm_ms = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    # 2e9 vec_flops -> ~169us analytical VECTOR floor vs measured 5000us * 0.95 -> upward scaling clamped <= 2.0
    vec_wl_lo_floor = CostFeatures(vec_flops=2e9, gm_bytes=1e6, total_tiles=56, topology_axes=axes)
    for _ in range(40):
        mults_hi = cm_ms.observe_msprof(vec_wl_lo_floor, {"aiv_vec": 0.95}, measured_us=5000.0)
        assert all(0.5 <= v <= 2.0 for v in mults_hi.values())
    assert mults_hi["VECTOR"] <= 2.0 and mults_hi["VECTOR"] > 1.2

    cm_lo = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    vec_wl_hi_floor = CostFeatures(vec_flops=2e11, gm_bytes=1e6, total_tiles=56, topology_axes=axes)
    for _ in range(40):
        mults_lo = cm_lo.observe_msprof(vec_wl_hi_floor, {"aiv_vec": 0.05}, measured_us=1.0)
        assert all(0.5 <= v <= 2.0 for v in mults_lo.values())
    assert mults_lo["VECTOR"] >= 0.5 and mults_lo["VECTOR"] < 0.9

    # Head-2 delta head learns faster vs slower candidate ratios on compiler/spill signals
    cm_delta = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    wl_p = build_cost_features_from_ledger(parent_ledger, axes=axes)
    wl_fast = CostFeatures.coerce(
        {**wl_p.__dict__, "spill_bytes": 0, "vloopv2_loops": 2, "software_loops": 0}
    )
    wl_slow = CostFeatures.coerce(
        {**wl_p.__dict__, "spill_bytes": 4096, "vloopv2_loops": 0, "software_loops": 3}
    )
    for _ in range(4):
        cm_delta.observe_pair_delta(wl_fast, 50.0, wl_slow, 120.0, refit=False)
        cm_delta.observe_pair_delta(wl_slow, 120.0, wl_fast, 50.0, refit=False)
    cm_delta._refit_delta_ridge()
    assert cm_delta.predict_delta_ratio(wl_fast, wl_slow) < 1.0
    assert cm_delta.predict_delta_ratio(wl_slow, wl_fast) > 1.0


def test_kb_local_isolation_graduation_revival_and_loo_cap(tmp_path: Path) -> None:
    """Verify operator_local isolation, cross_op promotion, cooldown revival, and LOO max_loo_chunks cap."""
    lake_dir = tmp_path / "kb_lake"
    tracker = BayesianRuleTracker.from_seed_yaml()
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    local_rule = AscendCausalRule(
        rule_id="R_TEST_LOCAL_01",
        regime_guard="is_legal",
        precondition="PIPE_V >= 0",
        transform_template="Unroll innermost loop by 4 and align UB buffer",
        expected_postcondition="any_overflow_bytes == 0",
        source_provenance="flywheel:ablation",
        alpha_prior=2.0,
        beta_prior=1.0,
        scope="operator_local",
        origin_operator="level2/op_a",
    )
    tracker.admit_rule(local_rule)
    save_distilled_rule_file(lake_dir, local_rule)

    # Visible to op_a, isolated from op_b while operator_local
    vis_a = {r.rule_id for r in tracker.active_rules(topology_axes=axes, operator_name="level2/op_a")}
    vis_b = {r.rule_id for r in tracker.active_rules(topology_axes=axes, operator_name="level2/op_b")}
    assert "R_TEST_LOCAL_01" in vis_a
    assert "R_TEST_LOCAL_01" not in vis_b

    # Record >=2 successes with high expected_rate and graduate -> cross_op
    for ridx in (1, 2):
        tracker.record_outcome(
            "R_TEST_LOCAL_01",
            postcondition_realized=True,
            speed_gain=0.15,
            topology_axes=axes,
        )
        append_kb_event_shard(
            lake_dir,
            run_id="run_op_a",
            rule_id="R_TEST_LOCAL_01",
            postcondition_realized=True,
            speed_gain=0.15,
            topology_axes=axes,
            operator_name="level2/op_a",
            round_idx=ridx,
        )
    promoted = tracker.promote_verified_local_rules()
    assert "R_TEST_LOCAL_01" in promoted
    assert tracker.rules["R_TEST_LOCAL_01"].scope == "cross_op"
    save_distilled_rule_file(lake_dir, tracker.rules["R_TEST_LOCAL_01"])

    vis_b_after = {r.rule_id for r in tracker.active_rules(topology_axes=axes, operator_name="level2/op_b")}
    assert "R_TEST_LOCAL_01" in vis_b_after

    # Verify cross-process reload from ExperienceLake replays rules + posteriors + graduation
    reloaded = BayesianRuleTracker.from_experience_lake(lake_dir)
    assert "R_TEST_LOCAL_01" in reloaded.rules
    assert reloaded.rules["R_TEST_LOCAL_01"].scope == "cross_op"
    assert reloaded.posteriors["R_TEST_LOCAL_01"].successes == 2

    # Prune at round 2 -> revived after 10 rounds cooldown (at round 12)
    bad_rule = AscendCausalRule(
        rule_id="R_TEST_REVIVE_02",
        regime_guard="is_legal",
        precondition="PIPE_V >= 0",
        transform_template="Speculative tile tweak",
        expected_postcondition="any_overflow_bytes == 0",
        source_provenance="flywheel:ablation",
        alpha_prior=1.0,
        beta_prior=1.0,
    )
    tracker.admit_rule(bad_rule)
    for _ in range(6):
        tracker.record_outcome("R_TEST_REVIVE_02", postcondition_realized=False)
    pruned = tracker.prune_underperforming(min_trials=5, prune_threshold=0.25, round_idx=2)
    assert "R_TEST_REVIVE_02" in pruned
    assert tracker.posteriors["R_TEST_REVIVE_02"].pruned is True
    assert tracker.posteriors["R_TEST_REVIVE_02"].pruned_at_round == 2

    assert tracker.decay_and_revive(round_idx=8, min_cooldown_rounds=10) == ()
    revived = tracker.decay_and_revive(round_idx=12, min_cooldown_rounds=10)
    assert "R_TEST_REVIVE_02" in revived
    assert tracker.posteriors["R_TEST_REVIVE_02"].pruned is False

    # Also verify LOO max_loo_chunks cap on real source diff
    chunk_test = distill_causal_rule(
        AstEditChunk(
            chunk_id="c1",
            edit_kind="barrier_prune",
            target_path="body",
            before_src="T.pipe_barrier()",
            after_src="# removed",
            ablated_dsl="pass",
        ),
        rule_id="R_CAP_CHECK",
        scope="operator_local",
        origin_operator="level2/op_a",
    )
    assert chunk_test.scope == "operator_local"

    base_dsl = (
        "def k(tile_m=1, tile_n=1, stages=1, flag=0):\n"
        "    a = tile_m + tile_n\n"
        "    b = stages + flag\n"
        "    return a + b\n"
    )
    win_dsl = (
        "def k(tile_m=4, tile_n=8, stages=2, flag=1):\n"
        "    a = tile_m + tile_n\n"
        "    b = stages + flag\n"
        "    return a + b\n"
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 400},
        gm_roundtrip_bytes=32768,
        grid_blocks=16,
        waterlines={},
    )
    funnel = GateFunnel(skip_stage_b=True)
    # Use custom stage_a ledger stub via monkeypatching run_stage_a_static or distill_leave_one_out directly
    distiller = FlywheelDistiller(funnel=funnel, tracker=tracker)
    _ = ledger
    rep = distiller.distill_leave_one_out(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        winner_latency_us=10.0,
        baseline_latency_us=20.0,
        max_loo_chunks=3,
        operator_name="level2/op_a",
        rule_scope="operator_local",
    )
    assert rep.total_chunks <= 3


def test_campaign_runner_logs_all_tiers_and_keeps_prescreen_off(tmp_path: Path) -> None:
    """End-to-end offline runner check: Baseline + L2 pass/fail + L4 logged; prescreen strictly OFF."""
    # 1. Verify hard defaults stay OFF
    assert MarathonConfig("level2/softmax").enable_tilesize_prescreen is False
    assert MarathonConfig("level2/softmax").cost_prescreen_top_k is None
    assert CampaignConfig(operator_name="level2/softmax").enable_tilesize_prescreen is False
    assert CampaignConfig(operator_name="level2/softmax").cost_prescreen_top_k is None

    mod_src = '''
def build_vec(tile_m=1):
    """baseline"""
    return tile_m
'''
    mod_path = tmp_path / "op_mod.py"
    mod_path.write_text(mod_src, encoding="utf-8")
    ckpt_path = tmp_path / "state" / "checkpoint.json"
    lake_dir = tmp_path / "state" / "experience_lake"

    reg = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("build_vec",),
        sample_builder_kwargs={},
        description="small test regime",
        representative_anchor_dims={"M": 128, "N": 512},
    )
    anchors = [
        AnchorCaseSpec("level2/softmax", 1, "small", ref_perf_us=100.0, t_hw_us=20.0),
    ]
    perf_target = PerfTargetSpec(
        operator_name="level2/softmax",
        target_avg_speedup=10.0,
        target_min_case_speedup=10.0,
        target_roofline_eta=0.99,
        target_surrogate_score=99.9,
        min_optimization_rounds=1,
    )

    cand_good_snip = 'def build_vec(tile_m=4):\n    """fast"""\n    return tile_m\n'
    cand_bad_acc_snip = 'def build_vec(tile_m=8):\n    """bad accuracy"""\n    return tile_m\n'

    class _RoleMockBackend:
        async def generate(self, req):
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "test hypothesis",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": cand_bad_acc_snip,
                    }
                )
            return cand_good_snip

    backend = _RoleMockBackend()
    tracker = BayesianRuleTracker.from_rules(
        [
            AscendCausalRule(
                rule_id="R_TEST_ALWAYS",
                regime_guard="is_legal",
                precondition="True",
                transform_template="Increase tile_m to 4",
                expected_postcondition="measured_latency_us <= prev.measured_latency_us",
                source_provenance="test",
            )
        ]
    )

    def _l0(_src: str, _spec: RegimeTargetSpec) -> StageAResult:
        return StageAResult(
            passed=True,
            prim_func=None,
            ledger=build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=120,
                pipe_cycles={"AIV0.PIPE_V": 500, "AIV0.PIPE_MTE2": 200},
                gm_roundtrip_bytes=32768,
                grid_blocks=28,
                waterlines={},
                microarch_profile=MicroarchProfile(
                    fp_alu_ops=256, gm_to_ub_bytes=32768, ub_to_gm_bytes=32768, source_kind="tirx"
                ).to_dict(),
            ),
        )

    def _l1(_src: str, _spec: RegimeTargetSpec) -> StageBResult:
        return StageBResult(
            passed=True,
            microarch_profile=MicroarchProfile(sfu_f16_ops=8, source_kind="cce").to_dict(),
        )

    def _l2_eval(cands, anc_seq):
        out = {}
        for cid, cpath in cands:
            txt = cpath.read_text(encoding="utf-8")
            if cid == "incumbent_baseline":
                metrics = [
                    AnchorEvalMetrics(
                        case_num=anc_seq[0].case_num,
                        regime_tag="small",
                        accuracy_passed=True,
                        cand_us=100.0,
                        ref_perf_us=100.0,
                        t_hw_us=20.0,
                        speedup=1.0,
                        roofline_eta=0.20,
                        eq3_score=10.0,
                    )
                ]
                out[cid] = CandidateSurrogateResult(
                    candidate_id=cid,
                    operator_name="level2/softmax",
                    all_passed=True,
                    avg_speedup=1.0,
                    min_case_speedup=1.0,
                    avg_roofline_eta=0.20,
                    surrogate_score=10.0,
                    anchors=tuple(metrics),
                )
            elif "fast" in txt:
                metrics = [
                    AnchorEvalMetrics(
                        case_num=anc_seq[0].case_num,
                        regime_tag="small",
                        accuracy_passed=True,
                        cand_us=40.0,
                        ref_perf_us=100.0,
                        t_hw_us=20.0,
                        speedup=2.5,
                        roofline_eta=0.50,
                        eq3_score=55.0,
                    )
                ]
                out[cid] = CandidateSurrogateResult(
                    candidate_id=cid,
                    operator_name="level2/softmax",
                    all_passed=True,
                    avg_speedup=2.5,
                    min_case_speedup=2.5,
                    avg_roofline_eta=0.50,
                    surrogate_score=55.0,
                    anchors=tuple(metrics),
                )
            else:
                # L2 compiled & timed, but accuracy failed -> still valuable for CostModel!
                metrics = [
                    AnchorEvalMetrics(
                        case_num=anc_seq[0].case_num,
                        regime_tag="small",
                        accuracy_passed=False,
                        cand_us=35.0,
                        ref_perf_us=100.0,
                        t_hw_us=20.0,
                        speedup=0.0,
                        roofline_eta=0.0,
                        eq3_score=0.0,
                        error_message="acc mismatch",
                    )
                ]
                out[cid] = CandidateSurrogateResult(
                    candidate_id=cid,
                    operator_name="level2/softmax",
                    all_passed=False,
                    avg_speedup=0.0,
                    min_case_speedup=0.0,
                    avg_roofline_eta=0.0,
                    surrogate_score=0.0,
                    anchors=tuple(metrics),
                    error_message="acc mismatch",
                )
        return out

    def _msprof(op_name: str, cand_path: str, anc_seq):
        src = Path(cand_path).read_text(encoding="utf-8")
        if "baseline" in src:
            return {"small": {"dur_us": 100.0, "aiv_vec": 0.45, "aiv_mte2": 0.35}}
        if "fast" in src:
            return {"small": {"dur_us": 40.0, "aiv_vec": 0.72, "aiv_mte2": 0.20}}
        return {}

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="level2/softmax",
            max_rounds=1,
            k_rule_workers=1,
            max_syntax_repairs=0,
            run_final_l4=True,
            l4_regression_ratio=0.0,
        ),
        target_module_path=mod_path,
        perf_target=perf_target,
        anchors=anchors,
        regime_specs=[reg],
        llm_backend=backend,
        tracker=tracker,
        l0_checker=_l0,
        l1_compiler=_l1,
        l2_batch_evaluator=_l2_eval,
        msprof_collector=_msprof,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True, run_id="l3_ok"),
        l4_cannbench_runner=lambda _p: L4CannbenchResult(
            passed=True,
            avg_speedup=2.4,
            score=75.0,
            per_case_speedups={1: 2.4, 2: 2.1},
            per_case_elapsed_us={1: 41.0, 2: 52.0},
            run_id="l4_ok",
        ),
        checkpoint_path=ckpt_path,
        experience_lake_dir=lake_dir,
    )
    report = asyncio.run(runner.run(resume=False))
    assert list(report.promoted_rounds) == [1]

    lake = ExperienceLake(lake_dir)
    cands, obs = lake.load_records()
    tiers = {o.eval_tier for o in obs}
    assert {"baseline", "l2_pass", "l2_acc_fail", "l4_official"}.issubset(tiers)
    # Candidate that failed accuracy without msprof collection must NOT inherit parent msprof
    acc_fail_obs = [o for o in obs if o.eval_tier == "l2_acc_fail"]
    assert acc_fail_obs and acc_fail_obs[0].msprof_pmu == {}
    # Promoted candidate has merged Stage-B sfu_f16_ops in candidate artifact
    assert any(int(c.microarch_profile.get("sfu_f16_ops", 0) or 0) >= 8 for c in cands.values())
    # Experience lake has pairwise delta for same-case fast vs baseline
    assert lake.extract_pairwise_deltas()


def test_record_l4_observations_hydrates_non_anchor_metadata_from_cannbench(
    tmp_path: Path,
) -> None:
    """Verify _record_l4_observations populates ref_perf_us, t_hw_us, and eq3 for non-anchors."""
    cann_repo = tmp_path / "cann_repo"
    op_dir = cann_repo / "tasks" / "level2" / "softmax"
    meta_dir = cann_repo / "tasks" / "metadata"
    op_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (op_dir / "proto.yaml").write_text("operator:\n  name: Softmax\n", encoding="utf-8")
    (op_dir / "cases.yaml").write_text(
        "cases:\n"
        "- case_id: 1\n  input_shape: [[32, 128]]\n  dtype: [float16]\n"
        "- case_id: 2\n  input_shape: [[64, 256]]\n  dtype: [float16]\n"
        "- case_id: 3\n  input_shape: [[16, 512]]\n  dtype: [float16]\n",
        encoding="utf-8",
    )
    (meta_dir / "950pr.json").write_text(
        json.dumps(
            {
                "level2": {
                    "softmax": {
                        "1": {"baseline_perf_us": 100.0, "t_hw_us": 20.0},
                        "2": {"baseline_perf_us": 200.0, "t_hw_us": 50.0},
                        "3": {"baseline_perf_us": 60.0, "t_hw_us": 20.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    mod_path = tmp_path / "softmax_mod.py"
    mod_src = "def build_vec(tile_m=1):\n    return tile_m\n"
    mod_path.write_text(mod_src, encoding="utf-8")
    lake_dir = tmp_path / "experience_lake"
    reg = RegimeTargetSpec("small", ("build_vec",), {}, representative_anchor_dims={"M": 32, "D": 128})
    anchors = (AnchorCaseSpec("level2/softmax", 1, "small", ref_perf_us=100.0, t_hw_us=20.0),)

    class _DummyBackend:
        async def generate(self, req):
            _ = req
            return ""

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/softmax", max_rounds=1),
        target_module_path=mod_path,
        cannbench_repo=cann_repo,
        anchors=anchors,
        regime_specs=[reg],
        llm_backend=_DummyBackend(),
        l0_checker=lambda _s, _r: StageAResult(
            passed=True,
            prim_func=None,
            ledger=build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=100,
                pipe_cycles={"AIV0.PIPE_V": 200},
                gm_roundtrip_bytes=16384,
                grid_blocks=16,
                waterlines={},
            ),
        ),
        checkpoint_path=tmp_path / "ckpt.json",
        experience_lake_dir=lake_dir,
    )
    l4_res = L4CannbenchResult(
        passed=True,
        avg_speedup=2.25,
        score=82.0,
        per_case_speedups={1: 2.5, 3: 2.0},
        per_case_elapsed_us={1: 40.0, 2: 100.0, 3: 30.0},
        run_id="l4_non_anchor",
    )
    runner._record_l4_observations(source=mod_src, l4_result=l4_res, active_anchors=anchors)

    _cands, obs = ExperienceLake(lake_dir).load_records()
    by_case = {int(o.case_num): o for o in obs if o.eval_tier == "l4_official"}
    assert set(by_case.keys()) == {1, 2, 3}
    assert by_case[1].ref_perf_us == pytest.approx(100.0)
    assert by_case[1].t_hw_us == pytest.approx(20.0)
    assert by_case[1].eq3_score == pytest.approx(0.8)
    assert by_case[2].ref_perf_us == pytest.approx(200.0)
    assert by_case[2].t_hw_us == pytest.approx(50.0)
    assert by_case[2].speedup == pytest.approx(2.0)
    assert by_case[2].eq3_score == pytest.approx(0.75)
    assert by_case[3].ref_perf_us == pytest.approx(60.0)
    assert by_case[3].t_hw_us == pytest.approx(20.0)
    assert by_case[3].speedup == pytest.approx(2.0)
    assert by_case[3].eq3_score == pytest.approx(0.8)

