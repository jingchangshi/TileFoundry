"""Consolidated L3 tests for Bisheng resource parser, residual basis, and ExperienceLake."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model import (
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
    DEFAULT_RESIDUAL_BASIS_SPECS,
    BishengResourceSummary,
    CalibrationSample,
    CandidateArtifactRecord,
    CaseObservationRecord,
    CostFeatures,
    ExperienceLake,
    HybridAscendCostModel,
    MicroarchProfile,
    ResidualBasisContext,
    ResidualBasisRegistry,
    ResidualBasisSpec,
    build_cost_features_from_ledger,
    compute_source_sha256,
    count_reachable_ub_scalar_compute_loops,
    extract_microarch_profile_from_cce,
    parse_bisheng_res_usage,
    project_case_workload,
    topology_bucket_for_axes,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def _sample_ledger(
    *,
    gm_roundtrip_bytes: int = 0,
    grid_blocks: int = 16,
    gm_traffic_scope: str = "per_tile",
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 200, "AIV1.PIPE_V": 200},
        waterlines={},
        grid_blocks=grid_blocks,
        gm_roundtrip_bytes=gm_roundtrip_bytes,
        gm_traffic_scope=gm_traffic_scope,
        microarch_profile=MicroarchProfile(
            fp_alu_ops=1024,
            gm_to_ub_bytes=65536,
            ub_to_gm_bytes=65536,
            pipeline_depth=2,
            inner_loop_trips=4,
            source_kind="tirx",
        ).to_dict(),
    )


def test_bisheng_res_parser_outer_gm_decoupling_and_trackb_scalar_loops(tmp_path: Path) -> None:
    """Decouple outer GM metadata loops while catching reachable Track-B scalar loops."""
    mixed_src = """
    #include <tl_templates/ascend/common.h>
    extern "C" __global__ __aicore__ void grouped_kernel(
        __gm__ half* a, __gm__ half* b, __gm__ float* out, __gm__ int32_t* group_list, int n
    ) {
        __ubuf__ float* ub_zero = (__ubuf__ float*)0;
        for (int z = 0; z < 8; ++z) { ub_zero[z] = 0.0f; }
        int m_off = 0;
        for (int g = 0; g < n; ++g) {
            int m_g = group_list[g];
            for (int kb = 0; kb < (m_g + 15) / 16; ++kb) {
                asc_mmad(a + m_off, b, out + m_off, 16, 16, 16,
                         asc_unit_flag_mode::DISABLE, true, false, true);
            }
            simd_inst::vadd(ub_zero, ub_zero, ub_zero);
            m_off += m_g;
        }
    }
    extern "C" int launch(void* s, void* a, void* b, void* out, void* gl, int n) {
        grouped_kernel<<<28, nullptr, s>>>((__gm__ half*)a, (__gm__ half*)b, (__gm__ float*)out, (__gm__ int32_t*)gl, n);
        return 0;
    }
    """
    assert count_reachable_ub_scalar_compute_loops(mixed_src) == 0
    assert (
        parse_bisheng_res_usage(asm_or_dump_text=mixed_src).extra_signals.get(
            "scalar_compute_loop_fallback", 0.0
        )
        == 0.0
    )

    ub_scalar_src = """
    #include <tl_templates/ascend/common.h>
    extern "C" __global__ __vector__ void ub_k(__gm__ float* x, __gm__ float* y) {
        __ubuf__ float* ub_x = (__ubuf__ float*)0;
        asc_copy_gm2ub_align(ub_x, x, 256);
        for (int i = 0; i < 64; ++i) { ub_x[i] = ub_x[i] * 0.5f + 1.0f; }
        asc_copy_ub2gm_align(y, ub_x, 256);
    }
    extern "C" int launch(void* s, void* x, void* y) {
        ub_k<<<28, nullptr, s>>>((__gm__ float*)x, (__gm__ float*)y);
        return 0;
    }
    """
    assert count_reachable_ub_scalar_compute_loops(ub_scalar_src) == 1

    dead_cube_live_scalar = """
    #include <tl_templates/ascend/common.h>
    extern "C" __global__ __aicore__ void gqa_cube_mmad(__gm__ float* a) {
        asc_mmad(a, a, a, 16, 16, 16, asc_unit_flag_mode::DISABLE, true, false, true);
    }
    extern "C" __global__ __vector__ void gqa_kernel(__gm__ float* q, __gm__ float* out) {
        for (int i = 0; i < 64; ++i) { out[i] += q[i] * 0.5f; }
    }
    extern "C" int launch(void* stream, void* q, void* out) {
        gqa_kernel<<<28, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)out);
        return 0;
    }
    """
    assert (
        parse_bisheng_res_usage(asm_or_dump_text=dead_cube_live_scalar).extra_signals.get(
            "scalar_compute_loop_fallback", 0.0
        )
        >= 1.0
    )

    single_simt = """
    #include <tl_templates/ascend/common.h>
    __simt_vf__ __launch_bounds__(1) void vf_one(__gm__ float* q, __gm__ float* out) {
        for (int i = 0; i < 64; ++i) { out[i] = q[i] + 1.0f; }
    }
    extern "C" __global__ __vector__ void k_one(__gm__ float* q, __gm__ float* out) {
        asc_vf_call<vf_one>(dim3(1, 1, 1), q, out);
    }
    extern "C" int launch(void* stream, void* q, void* out) {
        k_one<<<28, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)out);
        return 0;
    }
    """
    assert (
        parse_bisheng_res_usage(asm_or_dump_text=single_simt).extra_signals.get(
            "scalar_compute_loop_fallback", 0.0
        )
        >= 1.0
    )
    wide_simt = single_simt.replace("__launch_bounds__(1)", "__launch_bounds__(128)").replace(
        "dim3(1, 1, 1)", "dim3(128, 1, 1)"
    )
    assert (
        parse_bisheng_res_usage(asm_or_dump_text=wide_simt).extra_signals.get(
            "scalar_compute_loop_fallback", 0.0
        )
        == 0.0
    )

    cce_src = """
    extern "C" __global__ __aicore__ void cube_k() {
        asc_copy_gm2l1_nd2nz(a_l1, a_gm, 128, 0, 128, 128, 128, false);
        ascend_gemm_l1<128, 64, 128, 32, false, half, float>(l0c, a_l1, b_l1, 1);
        asc_copy_l0c2ub(ub_out, l0c);
        asc_sync_notify(0);
    }
    __simd_vf__ inline void vec_k(__ubuf__ float* ub_out, __gm__ float* out_gm) {
        asc_sync_wait(0);
        asc_reduce_max(ub_out, ub_out);
        asc_sync_pipe(PIPE_ALL);
        asc_copy_ub2gm(out_gm, ub_out);
    }
    """
    prof = extract_microarch_profile_from_cce(cce_src)
    assert (prof.subkernel_count, prof.cube_mmad_count) == (2, 1)
    assert prof.gm_to_l1_bytes > 0 and prof.l0c_to_ub_bytes > 0 and prof.ub_to_gm_bytes > 0
    assert (prof.cv_sync_barriers, prof.pipe_all_barriers) == (2, 1)

    lake = ExperienceLake(tmp_path / "lake")
    out_path = lake.append_unmodeled_anomaly({"measured_us": 1000.0, "topology_bucket": "aiv"})
    assert out_path == lake.root_dir / "unmodeled_anomalies.jsonl" and out_path.is_file()
    assert ": " not in out_path.read_text(encoding="utf-8").splitlines()[0]
    assert set(DEFAULT_COMPILER_BEHAVIOR_REGISTRY.list_detectors())


def test_residual_basis_full_rank_head2_gating_and_legacy_upgrade(tmp_path: Path) -> None:
    """Verify 12-dim full-rank basis, Head-2 within-case specs, sample gating, and 13D upgrade."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    assert len(DEFAULT_RESIDUAL_BASIS_SPECS) == 12
    assert model.basis_registry.spec_names(head2_only=False) == (
        "intercept", "log_tier0_scale", "cube_bound_share", "vec_bound_share",
        "wave_tail_waste", "l2_miss_ratio", "short_dma_burst_ind", "tier1_penalty_ratio",
        "launch_overhead_ratio", "is_coupled_1c2v", "is_aic_only", "is_irregular_simt",
    )

    axes_aiv = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    axes_aic = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    axes_cv = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    axes_irreg = TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")
    assert topology_bucket_for_axes(None) == "global"
    assert topology_bucket_for_axes(axes_cv) == "coupled_1c2v:bidirectional_cv_ring"
    gates = {spec.name: spec.min_bucket_samples for spec in DEFAULT_RESIDUAL_BASIS_SPECS}
    assert gates["intercept"] == 2 and gates["is_coupled_1c2v"] == 8

    workloads = [
        CostFeatures(cube_flops=5e12, gm_bytes=8e6, total_tiles=28, contig_burst_bytes=32768, topology_axes=axes_aic),
        CostFeatures(cube_flops=2e12, gm_bytes=32e6, total_tiles=29, contig_burst_bytes=4096, swizzle_window=4, topology_axes=axes_aic),
        CostFeatures(vec_flops=4e10, gm_bytes=12e6, total_tiles=56, contig_burst_bytes=32768, topology_axes=axes_aiv),
        CostFeatures(vec_flops=1e10, gm_bytes=64e6, total_tiles=60, contig_burst_bytes=8192, swizzle_window=4, spill_bytes=512, topology_axes=axes_aiv),
        CostFeatures(cube_flops=3e12, vec_flops=2e10, gm_bytes=20e6, total_tiles=30, contig_burst_bytes=32768, kernel_launches=2, topology_axes=axes_cv),
        CostFeatures(cube_flops=8e11, vec_flops=5e10, gm_bytes=10e6, total_tiles=28, contig_burst_bytes=4096, software_loops=2, topology_axes=axes_cv),
        CostFeatures(vec_flops=6e9, gm_bytes=2e6, total_tiles=57, contig_burst_bytes=4096, kernel_launches=3, topology_axes=axes_irreg),
        CostFeatures(vec_flops=2e9, gm_bytes=128e6, total_tiles=56, contig_burst_bytes=32768, l2_reuse_fraction=0.6, topology_axes=axes_irreg),
        CostFeatures(cube_flops=1e11, vec_flops=1e9, gm_bytes=4e6, total_tiles=40, contig_burst_bytes=16384, kernel_launches=4, topology_axes=axes_aic),
        CostFeatures(cube_flops=4e12, vec_flops=8e9, gm_bytes=80e6, total_tiles=56, contig_burst_bytes=32768, l2_reuse_fraction=0.45, topology_axes=axes_cv),
        CostFeatures(vec_flops=8e10, gm_bytes=16e6, total_tiles=70, contig_burst_bytes=2048, vf_fusion_breaks=3, topology_axes=axes_aiv),
        CostFeatures(cube_flops=6e11, vec_flops=3e10, gm_bytes=48e6, total_tiles=35, contig_burst_bytes=32768, l2_reuse_fraction=0.2, topology_axes=axes_cv),
        CostFeatures(vec_flops=3e9, gm_bytes=24e6, total_tiles=80, contig_burst_bytes=8192, spill_bytes=1024, topology_axes=axes_irreg),
        CostFeatures(cube_flops=9e12, gm_bytes=120e6, total_tiles=28, contig_burst_bytes=32768, kernel_launches=1, topology_axes=axes_aic),
        CostFeatures(vec_flops=1e7, gm_bytes=8192.0, total_tiles=16, contig_burst_bytes=8192, kernel_launches=4, topology_axes=axes_aiv),
        CostFeatures(cube_flops=2e10, vec_flops=5e8, gm_bytes=65536.0, total_tiles=28, contig_burst_bytes=32768, kernel_launches=3, topology_axes=axes_cv),
    ]
    new_rows, legacy_rows = [], []
    for wl in workloads:
        pred = model.predict(wl)
        assert pred.feature_vector == ResidualBasisRegistry.vectorize(pred.named_features, DEFAULT_RESIDUAL_BASIS_SPECS)
        new_rows.append(pred.feature_vector)
        denom = max(1e-6, pred.cube_floor_us + pred.vec_floor_us + pred.hbm_floor_us)
        legacy_rows.append((*pred.feature_vector[:4], pred.hbm_floor_us / denom, *pred.feature_vector[4:]))
    sv_new = np.linalg.svd(np.asarray(new_rows, dtype=np.float64), compute_uv=False)
    sv_leg = np.linalg.svd(np.asarray(legacy_rows, dtype=np.float64), compute_uv=False)
    assert sv_leg[-1] < 1e-10 and sv_new[-1] > 1e-3 and float(sv_new[0] / sv_new[-1]) < 500.0

    head2_specs = model.basis_registry.head2_specs()
    assert len(head2_specs) == 8 and all(s.varies_within_case for s in head2_specs)
    wl_fast = CostFeatures(vec_flops=2e10, gm_bytes=16e6, total_tiles=56, contig_burst_bytes=32768, vloopv2_loops=2, topology_axes=axes_aiv)
    wl_slow = CostFeatures(vec_flops=2e10, gm_bytes=16e6, total_tiles=60, contig_burst_bytes=4096, software_loops=3, spill_bytes=2048, topology_axes=axes_aiv)
    f_us, s_us = model.predict(wl_fast).pred_us * 0.9, model.predict(wl_slow).pred_us * 1.4
    for _ in range(3):
        model.observe_pair_delta(wl_fast, f_us, wl_slow, s_us)
    assert model.active_head2_spec_names() == ()
    model.observe_pair_delta(wl_fast, f_us, wl_slow, s_us)
    assert set(model.active_head2_spec_names()) == {"wave_tail_waste", "short_dma_burst_ind", "tier1_penalty_ratio"}
    for _ in range(2):
        model.observe_pair_delta(wl_slow, s_us, wl_fast, f_us)
    assert len(model.active_head2_spec_names()) == 8
    ckpt = tmp_path / "two_head.json"
    model.save_calibration(ckpt)
    reloaded = HybridAscendCostModel(target, calibration_path=ckpt)
    assert reloaded.pair_delta_count == 6 and reloaded.predict_delta_ratio(wl_fast, wl_slow) == pytest.approx(model.predict_delta_ratio(wl_fast, wl_slow), rel=1e-9)

    gated = HybridAscendCostModel(target, ridge_alpha=0.1)
    cands = [
        CostFeatures(vec_flops=1e10, gm_bytes=8e6, total_tiles=56, contig_burst_bytes=32768, topology_axes=axes_aiv),
        CostFeatures(vec_flops=1e10, gm_bytes=16e6, total_tiles=60, contig_burst_bytes=4096, topology_axes=axes_aiv),
        CostFeatures(vec_flops=1e10, gm_bytes=32e6, total_tiles=64, contig_burst_bytes=4096, spill_bytes=1024, topology_axes=axes_aiv),
    ]
    bps = [gated.predict(c).pred_us for c in cands]
    gated.observe(cands[0], bps[0] * 1.8)
    assert gated.active_head0_spec_names("global") == ()
    gated.observe(cands[2], bps[2] * 0.5)
    assert gated.active_head0_spec_names("aiv_only:gm_ub_stream") == ("intercept",)
    for idx, mult in ((1, 1.5), (0, 1.2), (1, 1.1), (2, 1.3), (0, 1.25), (2, 1.35)):
        gated.observe(cands[idx], bps[idx] * mult)
    assert len(gated.active_head0_spec_names("aiv_only:gm_ub_stream")) == 12

    legacy_vec = [1.0, 0.42, 0.30, 0.20, 0.50, 0.25, 0.60, 1.0, 0.35, 0.12, 1.0, 0.0, 0.0]
    sample = CalibrationSample.from_dict({"feature_vector": legacy_vec, "measured_us": 88.0, "analytical_us": 50.0, "topology_bucket": "coupled_1c2v:bidirectional_cv_ring"})
    assert sample.basis_schema_version == 2 and len(sample.named_features) == 12
    assert CalibrationSample.from_dict(sample.to_dict()) == sample

    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text(
        json.dumps({
            "samples": [{"feature_vector": legacy_vec, "measured_us": m, "analytical_us": 50.0, "topology_bucket": "coupled_1c2v:bidirectional_cv_ring"} for m in (80.0, 84.0)],
            "pair_deltas": [{"delta_x": [0.0, 0.1, 0.0, 0.0, 0.0, -0.2, 0.0, -1.0, -0.3, 0.0, 0.0, 0.0, 0.0], "delta_y": -0.4} for _ in range(4)],
        }),
        encoding="utf-8",
    )
    m_leg = HybridAscendCostModel(target, calibration_path=legacy_json)
    assert (m_leg.sample_count, m_leg.pair_delta_count, len(m_leg.weights), len(m_leg.delta_weights)) == (2, 4, 12, 8)

    with pytest.raises(ValueError, match="non-empty"):
        ResidualBasisSpec(name="  ", varies_within_case=True, min_bucket_samples=2, extractor=lambda _c: 0.0)
    with pytest.raises(ValueError, match="min_bucket_samples"):
        ResidualBasisSpec(name="x", varies_within_case=True, min_bucket_samples=0, extractor=lambda _c: 0.0)
    reg = ResidualBasisRegistry()
    cloned = reg.clone()
    custom = ResidualBasisSpec(name="burst_kb", varies_within_case=True, min_bucket_samples=4, extractor=lambda ctx: ctx.contig_burst_bytes / 1024.0)
    cloned.register(custom)
    assert len(reg.head0_specs()) == 12 and len(cloned.head0_specs()) == 13
    with pytest.raises(ValueError, match="already registered"):
        cloned.register(custom, overwrite=False)
    ctx = ResidualBasisContext(tier0_us=100.0, cube_floor_us=40.0, vec_floor_us=20.0, hbm_floor_us=40.0, tier1_penalty_us=15.0, launch_us=3.2, wave_ratio=1.25, l2_hit_rate=0.30, contig_burst_bytes=8192, axes=axes_cv)
    assert cloned.extract_named(ctx)["burst_kb"] == pytest.approx(8.0)


def test_experience_lake_dedup_replay_snapshots_and_gm_scope(tmp_path: Path) -> None:
    """Verify lake dedup/replay, per-regime snapshots, custom basis, and GM scope accounting."""
    target = AscendTarget("ascend.950pr")
    lake = ExperienceLake(tmp_path / "lake")
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    p_sha, c_sha = compute_source_sha256("def base(): return 0"), compute_source_sha256("def opt(): return 1")
    ub_hi = build_scope_waterline(scope="UB", capacity_bytes=262144, explicit_buffers={"x": 200000}, implicit_buffers={})
    ub_lo = build_scope_waterline(scope="UB", capacity_bytes=262144, explicit_buffers={"x": 65536}, implicit_buffers={})
    p_ldg = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=200,
        pipe_cycles={"AIV0.PIPE_V": 900, "AIV0.PIPE_MTE2": 600},
        gm_roundtrip_bytes=65536, grid_blocks=60, waterlines={"UB": ub_hi},
        microarch_profile=MicroarchProfile(fp_alu_ops=1024, gm_to_ub_bytes=65536, ub_to_gm_bytes=65536, pipeline_depth=1, inner_loop_trips=4, source_kind="tirx").to_dict(),
    )
    c_ldg = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 700, "AIV0.PIPE_MTE2": 450},
        gm_roundtrip_bytes=65536, grid_blocks=56, waterlines={"UB": ub_lo},
        microarch_profile=MicroarchProfile(fp_alu_ops=1024, gm_to_ub_bytes=65536, ub_to_gm_bytes=65536, pipeline_depth=2, inner_loop_trips=4, source_kind="tirx").to_dict(),
    )
    lake.register_candidate(
        CandidateArtifactRecord(cand_sha=p_sha, operator="op/n", regime_tag="wide", topology_axes=axes.to_dict(), base_ledger=p_ldg.to_dict(), microarch_profile=dict(p_ldg.microarch_profile))
    )
    lake.register_candidate(
        CandidateArtifactRecord(cand_sha=c_sha, operator="op/n", regime_tag="wide", parent_sha=p_sha, topology_axes=axes.to_dict(), base_ledger=c_ldg.to_dict(), microarch_profile=dict(c_ldg.microarch_profile))
    )
    for cn, p_us, c_us in ((1, 110.0, 58.0), (2, 195.0, 96.0), (3, 360.0, 175.0), (4, 690.0, 330.0)):
        lake.append_observation(
            CaseObservationRecord(run_id="r1", operator="op/n", regime_tag="wide", cand_sha=p_sha, eval_tier="baseline", case_num=cn, measured_us=p_us, accuracy_passed=True, case_dims={"M": 512 * cn, "N": 2048}, t_hw_us=p_us * 0.35, msprof_pmu={"dur_us": p_us, "aiv_vec": 0.65})
        )
        lake.append_observation(
            CaseObservationRecord(run_id="r1", operator="op/n", regime_tag="wide", cand_sha=c_sha, eval_tier="l2_pass", case_num=cn, measured_us=c_us, accuracy_passed=True, parent_sha=p_sha, case_dims={"M": 512 * cn, "N": 2048}, t_hw_us=p_us * 0.35, msprof_pmu={"dur_us": c_us, "aiv_vec": 0.72})
        )
    lake.append_observation(
        CaseObservationRecord(run_id="ext", operator="op/n", regime_tag="narrow", cand_sha=c_sha, eval_tier="l2_pass", case_num=1, measured_us=45.0, accuracy_passed=True, parent_sha=p_sha, case_dims={"M": 128, "N": 256}, t_hw_us=15.0)
    )

    m_rep = HybridAscendCostModel(target)
    assert m_rep.rebuild_from_experience_lake(lake) == 9
    assert (m_rep.sample_count, m_rep.pair_delta_count, m_rep.msprof_event_count) == (9, 4, 8)
    w_first = m_rep.weights
    assert m_rep.rebuild_from_experience_lake(lake) == 9 and m_rep.weights == pytest.approx(w_first, rel=1e-12)

    m_rep.basis_registry.register(
        ResidualBasisSpec(
            name="wave_x_l2_miss",
            varies_within_case=True,
            min_bucket_samples=4,
            extractor=lambda ctx: max(0.0, ctx.wave_ratio - 1.0) * (1.0 - ctx.l2_hit_rate),
        )
    )
    assert m_rep.rebuild_from_experience_lake(lake) == 9
    assert len(m_rep.weights) == 13 and "wave_x_l2_miss" in m_rep.active_head2_spec_names()

    m_dedup = HybridAscendCostModel(target)
    wl_p = project_case_workload(p_ldg, case_dims={"M": 512, "N": 2048}, axes=axes)
    m_dedup.observe(wl_p, 110.0, persist=False, refit=False)
    m_dedup.mark_online_observation_key(cand_sha=p_sha, operator="op/n", regime_tag="wide", case_num=1, eval_tier="baseline", measured_us=110.0)
    stats = m_dedup.replay_experience_lake(lake, exclude_run_id="r1")
    assert stats["observations"] == 1

    p_sm = lake.register_candidate(
        CandidateArtifactRecord(cand_sha="dead01", operator="op/a", regime_tag="small", parent_sha="cafe00", base_ledger={}, bisheng_summary={"spill_bytes": 256})
    )
    assert p_sm.name == "dead01_small.json"
    by_reg, _ = lake.load_records_by_regime()
    assert by_reg[("dead01", "small")].bisheng_summary["spill_bytes"] == 256

    cand = BeamCandidate(
        candidate_id="c1", round_idx=2, worker_kind="dsl_mutate", rule_id="rx", dsl_source="def k(): pass",
        track="tilelang_dsl", ledger=p_ldg, residual=compute_static_dynamic_residual(p_ldg, {"dur_us": 45.0}),
        latency_us=45.0, accuracy_passed=True, case_latencies_us={1: 42.0}, policy={"bm": 64},
    )
    assert BeamCandidate.from_dict(json.loads(json.dumps(cand.to_dict()))).policy == {"bm": 64}

    bp = SimpleNamespace(roofline={"min_gm_bytes": 100_000})
    per_tile = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100}, gm_roundtrip_bytes=500_000, gm_traffic_scope="per_tile",
        grid_blocks=56, waterlines={},
    )
    full_grid = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100}, gm_roundtrip_bytes=500_000, gm_traffic_scope="full_grid",
        grid_blocks=56, waterlines={},
    )
    assert build_cost_features_from_ledger(per_tile, blueprint=bp, target=target).gm_bytes == pytest.approx(500_000.0 * 56.0)
    assert build_cost_features_from_ledger(full_grid, blueprint=bp, target=target).gm_bytes == pytest.approx(500_000.0)
    assert float(project_case_workload({"grid_blocks": 56, "gm_roundtrip_bytes": 1_000_000, "gm_traffic_scope": "full_grid", "pipe_cycles": {"AIV0.PIPE_V": 100}}, target=target)["gm_bytes"]) == pytest.approx(1_000_000.0)
