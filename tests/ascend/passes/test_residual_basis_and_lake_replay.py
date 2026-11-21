"""Tests for ResidualBasisRegistry, sample-gated Ridge, GM scope, and ExperienceLake replay."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tilefoundry_ascend.ascend.cost_model import (
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
    project_case_workload,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_residual_basis_full_rank_eliminates_simplex_collinearity() -> None:
    """Verify 12-dim default basis is full-rank without exact simplex collinearity."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    reg = model.basis_registry

    head0_names = reg.spec_names(head2_only=False)
    assert len(DEFAULT_RESIDUAL_BASIS_SPECS) == 12
    assert len(head0_names) == 12
    assert "hbm_bound_share" not in head0_names
    assert head0_names == (
        "intercept",
        "log_tier0_scale",
        "cube_bound_share",
        "vec_bound_share",
        "wave_tail_waste",
        "l2_miss_ratio",
        "short_dma_burst_ind",
        "tier1_penalty_ratio",
        "launch_overhead_ratio",
        "is_coupled_1c2v",
        "is_aic_only",
        "is_irregular_simt",
    )

    axes_aiv = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    axes_aic = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    axes_cv = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    axes_irreg = TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")

    workloads = [
        CostFeatures(cube_flops=5e12, gm_bytes=8e6, total_tiles=28, contig_burst_bytes=32768, topology_axes=axes_aic),
        CostFeatures(cube_flops=2e12, gm_bytes=32e6, total_tiles=29, contig_burst_bytes=4096, swizzle_window=4, topology_axes=axes_aic),
        CostFeatures(vec_flops=4e10, gm_bytes=12e6, total_tiles=56, contig_burst_bytes=32768, topology_axes=axes_aiv),
        CostFeatures(vec_flops=1e10, gm_bytes=64e6, total_tiles=60, contig_burst_bytes=8192, swizzle_window=4, spill_bytes=512, topology_axes=axes_aiv),
        CostFeatures(cube_flops=3e12, vec_flops=2e10, gm_bytes=20e6, total_tiles=30, contig_burst_bytes=32768, kernel_launches=2, topology_axes=axes_cv),
        CostFeatures(cube_flops=8e11, vec_flops=5e10, gm_bytes=10e6, total_tiles=28, contig_burst_bytes=4096, software_loops=2, vloopv2_loops=0, topology_axes=axes_cv),
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

    new_rows: list[tuple[float, ...]] = []
    legacy_rows: list[tuple[float, ...]] = []
    for wl in workloads:
        pred = model.predict(wl)
        assert len(pred.feature_vector) == 12
        assert pred.feature_vector == ResidualBasisRegistry.vectorize(
            pred.named_features,
            DEFAULT_RESIDUAL_BASIS_SPECS,
        )
        new_rows.append(pred.feature_vector)
        denom = max(1e-6, pred.cube_floor_us + pred.vec_floor_us + pred.hbm_floor_us)
        hbm_share = pred.hbm_floor_us / denom
        legacy_row = (*pred.feature_vector[:4], hbm_share, *pred.feature_vector[4:])
        assert len(legacy_row) == 13
        legacy_rows.append(legacy_row)

    x_new = np.asarray(new_rows, dtype=np.float64)
    x_legacy = np.asarray(legacy_rows, dtype=np.float64)
    sv_new = np.linalg.svd(x_new, compute_uv=False)
    sv_legacy = np.linalg.svd(x_legacy, compute_uv=False)

    assert sv_legacy[-1] < 1e-10
    assert sv_new[-1] > 1e-3
    cond_new = float(sv_new[0] / sv_new[-1])
    cond_legacy = float(sv_legacy[0] / max(1e-18, sv_legacy[-1]))
    assert cond_new < 500.0
    assert cond_new * 1e6 < cond_legacy


def test_head2_uses_only_within_case_varying_specs_and_roundtrips(tmp_path: Path) -> None:
    """Verify Head-2 projects strictly varies_within_case=True specs with zero dead dims."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    head2_specs = model.basis_registry.head2_specs()
    head2_names = tuple(spec.name for spec in head2_specs)

    assert len(head2_specs) == 8
    assert all(spec.varies_within_case for spec in head2_specs)
    assert head2_names == (
        "log_tier0_scale",
        "cube_bound_share",
        "vec_bound_share",
        "wave_tail_waste",
        "l2_miss_ratio",
        "short_dma_burst_ind",
        "tier1_penalty_ratio",
        "launch_overhead_ratio",
    )
    for dead_name in ("intercept", "is_coupled_1c2v", "is_aic_only", "is_irregular_simt"):
        assert dead_name not in head2_names
        assert dead_name not in model.delta_weights_by_name
    assert len(model.delta_weights) == 8

    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    wl_fast = CostFeatures(
        vec_flops=2e10,
        gm_bytes=16e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        vloopv2_loops=2,
        software_loops=0,
        spill_bytes=0,
        topology_axes=axes,
    )
    wl_slow = CostFeatures(
        vec_flops=2e10,
        gm_bytes=16e6,
        total_tiles=60,
        contig_burst_bytes=4096,
        vloopv2_loops=0,
        software_loops=3,
        spill_bytes=2048,
        topology_axes=axes,
    )

    for _ in range(3):
        model.observe_pair_delta(wl_fast, 40.0, wl_slow, 160.0)
    assert model.active_head2_spec_names() == ()
    assert all(w == 0.0 for w in model.delta_weights)

    model.observe_pair_delta(wl_fast, 40.0, wl_slow, 160.0)
    assert set(model.active_head2_spec_names()) == {
        "wave_tail_waste",
        "short_dma_burst_ind",
        "tier1_penalty_ratio",
    }
    for name, val in model.delta_weights_by_name.items():
        if name not in model.active_head2_spec_names():
            assert val == 0.0

    for _ in range(2):
        model.observe_pair_delta(wl_slow, 160.0, wl_fast, 40.0)
    assert len(model.active_head2_spec_names()) == 8
    ratio_fast = model.predict_delta_ratio(wl_fast, wl_slow)
    ratio_slow = model.predict_delta_ratio(wl_slow, wl_fast)
    assert ratio_fast < 1.0
    assert ratio_slow > 1.0

    ckpt = tmp_path / "two_head.json"
    model.save_calibration(ckpt)
    reloaded = HybridAscendCostModel(target, calibration_path=ckpt)
    assert reloaded.pair_delta_count == 6
    assert len(reloaded.delta_weights) == 8
    assert reloaded.predict_delta_ratio(wl_fast, wl_slow) == pytest.approx(ratio_fast, rel=1e-9)


def test_sample_gated_ridge_prevents_small_sample_overfitting_and_unlocks_progressively() -> None:
    """Verify sample gating activates intercept at 2-3, mid specs at 4-5, and full at 8+."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target, ridge_alpha=0.1)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    cands = [
        CostFeatures(vec_flops=1e10, gm_bytes=8e6, total_tiles=56, contig_burst_bytes=32768, topology_axes=axes),
        CostFeatures(vec_flops=1e10, gm_bytes=16e6, total_tiles=60, contig_burst_bytes=4096, topology_axes=axes),
        CostFeatures(vec_flops=1e10, gm_bytes=32e6, total_tiles=64, contig_burst_bytes=4096, spill_bytes=1024, topology_axes=axes),
    ]
    base_preds = [model.predict(c).pred_us for c in cands]
    assert base_preds[0] < base_preds[1] < base_preds[2]

    model.observe(cands[0], base_preds[0] * 1.8)
    assert model.active_head0_spec_names("global") == ()
    assert all(w == 0.0 for w in model.weights)

    model.observe(cands[2], base_preds[2] * 0.5)
    bucket = "aiv_only:gm_ub_stream"
    assert model.active_head0_spec_names("global") == ("intercept",)
    assert model.active_head0_spec_names(bucket) == ("intercept",)
    for name, val in model.weights_by_name.items():
        if name != "intercept":
            assert val == 0.0
    for name, val in model.bucket_weights_by_name[bucket].items():
        if name != "intercept":
            assert val == 0.0

    preds_after_2 = [model.predict(c).pred_us for c in cands]
    assert preds_after_2[0] < preds_after_2[1] < preds_after_2[2]

    model.observe(cands[1], base_preds[1] * 1.5)
    assert model.active_head0_spec_names(bucket) == ("intercept",)
    preds_after_3 = [model.predict(c).pred_us for c in cands]
    assert preds_after_3[0] < preds_after_3[1] < preds_after_3[2]

    model.observe(cands[0], base_preds[0] * 1.2)
    assert set(model.active_head0_spec_names(bucket)) == {
        "intercept",
        "wave_tail_waste",
        "short_dma_burst_ind",
        "tier1_penalty_ratio",
    }
    for name, val in model.bucket_weights_by_name[bucket].items():
        if name not in model.active_head0_spec_names(bucket):
            assert val == 0.0

    model.observe(cands[1], base_preds[1] * 1.1)
    model.observe(cands[2], base_preds[2] * 1.3)
    assert len(model.active_head0_spec_names(bucket)) == 9
    for topo_flag in ("is_coupled_1c2v", "is_aic_only", "is_irregular_simt"):
        assert topo_flag not in model.active_head0_spec_names(bucket)
        assert model.bucket_weights_by_name[bucket][topo_flag] == 0.0

    model.observe(cands[0], base_preds[0] * 1.25)
    model.observe(cands[2], base_preds[2] * 1.35)
    assert len(model.active_head0_spec_names("global")) == 12
    assert len(model.active_head0_spec_names(bucket)) == 12


def test_legacy_13d_calibration_sample_and_checkpoint_upgrade(tmp_path: Path) -> None:
    """Verify legacy 13-element samples and checkpoints upgrade losslessly to 12 named keys."""
    legacy_vec = [
        1.0,
        0.42,
        0.30,
        0.20,
        0.50,
        0.25,
        0.60,
        1.0,
        0.35,
        0.12,
        1.0,
        0.0,
        0.0,
    ]
    sample = CalibrationSample.from_dict(
        {
            "feature_vector": legacy_vec,
            "measured_us": 88.0,
            "analytical_us": 50.0,
            "topology_bucket": "coupled_1c2v:bidirectional_cv_ring",
        }
    )
    assert sample.basis_schema_version == 2
    assert len(sample.named_features) == 12
    assert sample.named_features["intercept"] == pytest.approx(1.0)
    assert sample.named_features["log_tier0_scale"] == pytest.approx(0.42)
    assert sample.named_features["cube_bound_share"] == pytest.approx(0.30)
    assert sample.named_features["vec_bound_share"] == pytest.approx(0.20)
    assert sample.named_features["wave_tail_waste"] == pytest.approx(0.25)
    assert sample.named_features["l2_miss_ratio"] == pytest.approx(0.60)
    assert sample.named_features["short_dma_burst_ind"] == pytest.approx(1.0)
    assert sample.named_features["tier1_penalty_ratio"] == pytest.approx(0.35)
    assert sample.named_features["launch_overhead_ratio"] == pytest.approx(0.12)
    assert sample.named_features["is_coupled_1c2v"] == pytest.approx(1.0)
    assert sample.named_features["is_aic_only"] == pytest.approx(0.0)
    assert sample.named_features["is_irregular_simt"] == pytest.approx(0.0)
    assert len(sample.feature_vector) == 12

    serialized = sample.to_dict()
    assert serialized["basis_schema_version"] == 2
    assert isinstance(serialized["named_features"], dict)
    assert CalibrationSample.from_dict(serialized) == sample

    legacy_json = tmp_path / "legacy_ckpt.json"
    legacy_payload = {
        "samples": [
            {
                "feature_vector": legacy_vec,
                "measured_us": 80.0,
                "analytical_us": 50.0,
                "topology_bucket": "coupled_1c2v:bidirectional_cv_ring",
            },
            {
                "feature_vector": legacy_vec,
                "measured_us": 84.0,
                "analytical_us": 50.0,
                "topology_bucket": "coupled_1c2v:bidirectional_cv_ring",
            },
        ],
        "pair_deltas": [
            {"delta_x": [0.0, 0.1, 0.0, 0.0, 0.0, -0.2, 0.0, -1.0, -0.3, 0.0, 0.0, 0.0, 0.0], "delta_y": -0.4}
            for _ in range(4)
        ],
    }
    legacy_json.write_text(json.dumps(legacy_payload), encoding="utf-8")

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"), calibration_path=legacy_json)
    assert model.sample_count == 2
    assert model.pair_delta_count == 4
    assert len(model.weights) == 12
    assert len(model.delta_weights) == 8
    assert model.weights_by_name["intercept"] > 0.0
    assert model.delta_weights_by_name["tier1_penalty_ratio"] != 0.0

    legacy_jsonl = tmp_path / "legacy_cal.jsonl"
    legacy_jsonl.write_text(
        "\n".join(json.dumps(item) for item in legacy_payload["samples"]) + "\n",
        encoding="utf-8",
    )
    model_jsonl = HybridAscendCostModel(AscendTarget("ascend.950pr"), calibration_path=legacy_jsonl)
    assert model_jsonl.sample_count == 2
    assert model_jsonl.weights_by_name["intercept"] == pytest.approx(
        model.weights_by_name["intercept"],
        rel=1e-9,
    )


def test_rebuild_from_experience_lake_and_custom_basis_registration(tmp_path: Path) -> None:
    """Verify full ExperienceLake rebuild reproduces state and populates custom basis specs."""
    lake = ExperienceLake(tmp_path / "lake")
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    parent_sha = compute_source_sha256("def kernel_base(): return 0")
    child_sha = compute_source_sha256("def kernel_opt(): return 1")
    other_regime_sha = compute_source_sha256("def kernel_other(): return 2")

    ub_high = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"x": 200000},
        implicit_buffers={},
    )
    ub_low = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"x": 65536},
        implicit_buffers={},
    )

    parent_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=200,
        pipe_cycles={"AIV0.PIPE_V": 900, "AIV0.PIPE_MTE2": 600},
        gm_roundtrip_bytes=65536,
        grid_blocks=60,
        waterlines={"UB": ub_high},
        microarch_profile=MicroarchProfile(
            fp_alu_ops=1024,
            gm_to_ub_bytes=65536,
            ub_to_gm_bytes=65536,
            pipeline_depth=1,
            inner_loop_trips=4,
            source_kind="tirx",
        ).to_dict(),
    )
    child_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 700, "AIV0.PIPE_MTE2": 450},
        gm_roundtrip_bytes=65536,
        grid_blocks=56,
        waterlines={"UB": ub_low},
        microarch_profile=MicroarchProfile(
            fp_alu_ops=1024,
            gm_to_ub_bytes=65536,
            ub_to_gm_bytes=65536,
            pipeline_depth=2,
            inner_loop_trips=4,
            source_kind="tirx",
        ).to_dict(),
    )

    lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=parent_sha,
            operator="level2/norm_op",
            regime_tag="wide",
            topology_axes=axes.to_dict(),
            base_ledger=parent_ledger.to_dict(),
            microarch_profile=dict(parent_ledger.microarch_profile),
            bisheng_summary=BishengResourceSummary(spill_bytes=512, software_loops=1).to_dict(),
        )
    )
    lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=child_sha,
            operator="level2/norm_op",
            regime_tag="wide",
            parent_sha=parent_sha,
            topology_axes=axes.to_dict(),
            base_ledger=child_ledger.to_dict(),
            microarch_profile=dict(child_ledger.microarch_profile),
            bisheng_summary=BishengResourceSummary(spill_bytes=0, vloopv2_loops=2).to_dict(),
        )
    )
    lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=other_regime_sha,
            operator="level2/norm_op",
            regime_tag="narrow",
            parent_sha=parent_sha,
            topology_axes=axes.to_dict(),
            base_ledger=child_ledger.to_dict(),
            microarch_profile=dict(child_ledger.microarch_profile),
        )
    )

    for case_num, m_dim, p_us, c_us in (
        (1, 512, 110.0, 58.0),
        (2, 1024, 195.0, 96.0),
        (3, 2048, 360.0, 175.0),
        (4, 4096, 690.0, 330.0),
    ):
        lake.append_observation(
            CaseObservationRecord(
                run_id="r1",
                operator="level2/norm_op",
                regime_tag="wide",
                cand_sha=parent_sha,
                eval_tier="baseline",
                case_num=case_num,
                measured_us=p_us,
                accuracy_passed=True,
                case_dims={"M": m_dim, "N": 2048},
                t_hw_us=p_us * 0.35,
                msprof_pmu={"dur_us": p_us, "aiv_vec": 0.65, "aiv_mte2": 0.30},
            )
        )
        lake.append_observation(
            CaseObservationRecord(
                run_id="r1",
                operator="level2/norm_op",
                regime_tag="wide",
                cand_sha=child_sha,
                eval_tier="l2_pass",
                case_num=case_num,
                measured_us=c_us,
                accuracy_passed=True,
                parent_sha=parent_sha,
                case_dims={"M": m_dim, "N": 2048},
                t_hw_us=p_us * 0.35,
                msprof_pmu={"dur_us": c_us, "aiv_vec": 0.72, "aiv_mte2": 0.25},
            )
        )

    lake.append_observation(
        CaseObservationRecord(
            run_id="r2",
            operator="level2/norm_op",
            regime_tag="narrow",
            cand_sha=other_regime_sha,
            eval_tier="l2_pass",
            case_num=1,
            measured_us=45.0,
            accuracy_passed=True,
            parent_sha=parent_sha,
            case_dims={"M": 128, "N": 256},
            t_hw_us=15.0,
        )
    )

    pairs = lake.extract_pairwise_deltas()
    assert len(pairs) == 4
    assert all(cand_obs.regime_tag == "wide" for _ca, cand_obs, _pa, _po in pairs)

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    replayed_1 = model.rebuild_from_experience_lake(lake)
    assert replayed_1 == 9
    assert model.sample_count == 9
    assert model.pair_delta_count == 4
    assert model.msprof_event_count == 8
    weights_first = model.weights

    replayed_2 = model.rebuild_from_experience_lake(lake)
    assert replayed_2 == 9
    assert model.sample_count == 9
    assert model.pair_delta_count == 4
    assert model.weights == pytest.approx(weights_first, rel=1e-12)

    custom_spec = ResidualBasisSpec(
        name="wave_x_high_ub",
        varies_within_case=True,
        min_bucket_samples=4,
        extractor=lambda ctx: max(0.0, ctx.wave_ratio - 1.0) * ctx.ub_occupancy_ratio,
    )
    model.basis_registry.register(custom_spec)
    replayed_custom = model.rebuild_from_experience_lake(lake)
    assert replayed_custom == 9
    assert len(model.weights) == 13
    assert len(model.delta_weights) == 9
    assert "wave_x_high_ub" in model.weights_by_name
    assert "wave_x_high_ub" in model.delta_weights_by_name
    assert "wave_x_high_ub" in model.active_head0_spec_names("global")
    assert "wave_x_high_ub" in model.active_head2_spec_names()

    probe_cf = build_cost_features_from_ledger(parent_ledger, axes=axes)
    assert probe_cf.ub_occupancy_ratio == pytest.approx(200000.0 / 262144.0)
    probe_pred = model.predict(probe_cf)
    assert "wave_x_high_ub" in probe_pred.named_features
    expected_custom = max(0.0, probe_pred.wave_quantization_ratio - 1.0) * probe_cf.ub_occupancy_ratio
    assert probe_pred.named_features["wave_x_high_ub"] == pytest.approx(expected_custom)
    assert expected_custom > 0.0


def test_build_cost_features_respects_explicit_gm_traffic_scope() -> None:
    """Verify exact per_tile vs full_grid GM traffic accounting without 0.5*nav heuristic."""
    target = AscendTarget("ascend.950pr")
    blueprint = SimpleNamespace(roofline={"min_gm_bytes": 800_000})

    per_tile_ledger = SimpleNamespace(
        grid_blocks=56,
        gm_roundtrip_bytes=500_000,
        gm_traffic_scope="per_tile",
        pipe_cycles={"AIV0.PIPE_V": 100},
        dominant_contig_bytes=32768,
        steady_state_ii=100,
        waterlines={},
        microarch_profile={},
    )
    cf_per_tile = build_cost_features_from_ledger(
        per_tile_ledger,
        blueprint=blueprint,
        target=target,
    )
    assert cf_per_tile.gm_bytes == pytest.approx(500_000.0 * 56.0)

    full_grid_ledger = SimpleNamespace(
        grid_blocks=56,
        gm_roundtrip_bytes=200_000,
        gm_traffic_scope="full_grid",
        pipe_cycles={"AIV0.PIPE_V": 100},
        dominant_contig_bytes=32768,
        steady_state_ii=100,
        waterlines={},
        microarch_profile={},
    )
    cf_full_grid = build_cost_features_from_ledger(
        full_grid_ledger,
        blueprint=SimpleNamespace(roofline={"min_gm_bytes": 100_000}),
        target=target,
    )
    assert cf_full_grid.gm_bytes == pytest.approx(200_000.0)

    cf_full_grid_map = build_cost_features_from_ledger(
        {
            "grid_blocks": 56,
            "gm_roundtrip_bytes": 1_500_000,
            "gm_traffic_scope": "full_grid",
            "pipe_cycles": {"AIV0.PIPE_V": 100},
        },
        blueprint=blueprint,
        target=target,
    )
    assert cf_full_grid_map.gm_bytes == pytest.approx(1_500_000.0)

    class _MethodLedger:
        grid_blocks = 56
        gm_roundtrip_bytes = 500_000
        gm_traffic_scope = "full_grid"
        pipe_cycles = {"AIV0.PIPE_V": 100}
        dominant_contig_bytes = 32768
        steady_state_ii = 100
        waterlines: dict[str, object] = {}
        microarch_profile: dict[str, object] = {}

        def total_gm_bytes(self, grid_blocks: int | None = None) -> int:
            _ = grid_blocks
            return 12_345_678

    cf_method = build_cost_features_from_ledger(_MethodLedger(), blueprint=blueprint, target=target)
    assert cf_method.gm_bytes == pytest.approx(12_345_678.0)

    real_per_tile = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100},
        gm_roundtrip_bytes=500_000,
        gm_traffic_scope="per_tile",
        grid_blocks=56,
        waterlines={},
    )
    real_full_grid = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100},
        gm_roundtrip_bytes=500_000,
        gm_traffic_scope="full_grid",
        grid_blocks=56,
        waterlines={},
    )
    assert build_cost_features_from_ledger(
        real_per_tile,
        blueprint=blueprint,
        target=target,
    ).gm_bytes == pytest.approx(500_000.0 * 56.0)
    assert build_cost_features_from_ledger(
        real_full_grid,
        blueprint=SimpleNamespace(roofline={"min_gm_bytes": 100_000}),
        target=target,
    ).gm_bytes == pytest.approx(500_000.0)

    proj_full = project_case_workload(
        {
            "grid_blocks": 56,
            "gm_roundtrip_bytes": 1_000_000,
            "gm_traffic_scope": "full_grid",
            "pipe_cycles": {"AIV0.PIPE_V": 100},
        },
        target=target,
    )
    assert float(proj_full["gm_bytes"]) == pytest.approx(1_000_000.0)

    proj_tile = project_case_workload(
        {
            "grid_blocks": 56,
            "gm_roundtrip_bytes": 1_000_000,
            "gm_traffic_scope": "per_tile",
            "pipe_cycles": {"AIV0.PIPE_V": 100},
        },
        target=target,
    )
    assert float(proj_tile["gm_bytes"]) == pytest.approx(56_000_000.0)


def test_residual_basis_context_and_registry_validation() -> None:
    """Verify ResidualBasisSpec validation, duplicate guard, and clone isolation."""
    with pytest.raises(ValueError, match="non-empty"):
        ResidualBasisSpec(name="  ", varies_within_case=True, min_bucket_samples=2, extractor=lambda _c: 0.0)
    with pytest.raises(ValueError, match="min_bucket_samples"):
        ResidualBasisSpec(name="bad_min", varies_within_case=True, min_bucket_samples=0, extractor=lambda _c: 0.0)

    reg = ResidualBasisRegistry()
    cloned = reg.clone()
    custom = ResidualBasisSpec(
        name="ub_pressure",
        varies_within_case=True,
        min_bucket_samples=4,
        extractor=lambda ctx: ctx.ub_occupancy_ratio,
    )
    cloned.register(custom)
    assert len(reg.head0_specs()) == 12
    assert len(cloned.head0_specs()) == 13
    with pytest.raises(ValueError, match="already registered"):
        cloned.register(custom, overwrite=False)

    ctx = ResidualBasisContext(
        tier0_us=100.0,
        cube_floor_us=40.0,
        vec_floor_us=20.0,
        hbm_floor_us=40.0,
        tier1_penalty_us=15.0,
        launch_us=3.2,
        wave_ratio=1.25,
        l2_hit_rate=0.30,
        contig_burst_bytes=8192,
        ub_occupancy_ratio=0.75,
        axes=TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
    )
    named = cloned.extract_named(ctx)
    assert named["ub_pressure"] == pytest.approx(0.75)
    assert named["cube_bound_share"] == pytest.approx(0.40)
    assert named["vec_bound_share"] == pytest.approx(0.20)
    assert named["wave_tail_waste"] == pytest.approx(0.25)
    assert named["l2_miss_ratio"] == pytest.approx(0.70)
    assert named["short_dma_burst_ind"] == pytest.approx(1.0)
    assert named["tier1_penalty_ratio"] == pytest.approx(0.15)
    assert named["launch_overhead_ratio"] == pytest.approx(0.032)
    assert named["is_coupled_1c2v"] == pytest.approx(1.0)
    assert named["is_aic_only"] == pytest.approx(0.0)
    assert named["is_irregular_simt"] == pytest.approx(0.0)
    assert math.isclose(named["log_tier0_scale"], math.log1p(100.0) / 10.0)
