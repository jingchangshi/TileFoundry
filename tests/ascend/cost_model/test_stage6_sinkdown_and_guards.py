"""Tests for Stage-6 mechanism sink-down, anomaly guards, and bucket invariants."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tilefoundry_ascend.ascend.cost_model import (
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
    DEFAULT_RESIDUAL_BASIS_SPECS,
    CostFeatures,
    ExperienceLake,
    HybridAscendCostModel,
    MicroarchProfile,
    parse_bisheng_res_usage,
    topology_bucket_for_axes,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_scalar_compute_loop_fallback_inflates_zero_sample_prediction_and_ranking() -> None:
    """Scalar loop fallback inflates Tier-0 compute floor and ranks behind vector."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    clean = CostFeatures(
        vec_flops=4.0e10,
        gm_bytes=12.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
    )
    degraded = CostFeatures(
        vec_flops=4.0e10,
        gm_bytes=12.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        extra_signals={"scalar_compute_loop_fallback": 1.0},
    )

    pred_clean = model.predict(clean)
    pred_degraded = model.predict(degraded)

    assert pred_clean.tier2_residual_factor == pytest.approx(1.0)
    assert pred_degraded.tier2_residual_factor == pytest.approx(1.0)
    assert pred_degraded.vec_floor_us == pytest.approx(pred_clean.vec_floor_us * 128.0)
    assert pred_degraded.tier0_physics_us > pred_clean.tier0_physics_us * 10.0
    assert pred_degraded.pred_us > pred_clean.pred_us * 10.0

    ranked = sorted(
        [("degraded", pred_degraded.pred_us), ("clean", pred_clean.pred_us)],
        key=lambda item: item[1],
    )
    assert [name for name, _ in ranked] == ["clean", "degraded"]

    prof = MicroarchProfile(
        fp_alu_ops=2048,
        gm_to_ub_bytes=32768,
        ub_to_gm_bytes=32768,
        pipeline_depth=2,
        inner_loop_trips=8,
        source_kind="tirx",
    ).to_dict()
    clean_prof = CostFeatures(
        vec_flops=0.0,
        gm_bytes=12.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        microarch_profile=prof,
    )
    degraded_prof = CostFeatures(
        vec_flops=0.0,
        gm_bytes=12.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        microarch_profile=prof,
        extra_signals={"scalar_compute_loop_fallback": 1.0},
    )
    assert (
        model.predict(degraded_prof).vec_floor_us
        > model.predict(clean_prof).vec_floor_us * 100.0
    )


def test_dcache_bypass_narrow_store_inflates_hbm_floor_and_ranking() -> None:
    """Narrow GM bypass stores inflate Tier-0 HBM floor and rank behind coalesced DMA."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    clean = CostFeatures(
        vec_flops=1.0e7,
        gm_bytes=32.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
    )
    degraded = CostFeatures(
        vec_flops=1.0e7,
        gm_bytes=32.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        extra_signals={"dcache_bypass_narrow_store": 32.0},
    )

    pred_clean = model.predict(clean)
    pred_degraded = model.predict(degraded)

    assert pred_degraded.hbm_floor_us == pytest.approx(pred_clean.hbm_floor_us * 16.5)
    assert pred_degraded.tier0_physics_us > pred_clean.tier0_physics_us * 5.0
    assert pred_degraded.pred_us > pred_clean.pred_us * 5.0

    ranked = sorted(
        [("degraded", pred_degraded.pred_us), ("clean", pred_clean.pred_us)],
        key=lambda item: item[1],
    )
    assert [name for name, _ in ranked] == ["clean", "degraded"]


def test_optional_bisheng_stage6_detectors_integrate_with_cost_model() -> None:
    """Verify bisheng parser detectors flow into HybridAscendCostModel when registered."""
    registered = set(DEFAULT_COMPILER_BEHAVIOR_REGISTRY.list_detectors())
    if not {"scalar_compute_loop_fallback", "dcache_bypass_narrow_store"}.issubset(registered):
        pytest.skip("Stage-5/6 bisheng detectors not yet registered in this snapshot")

    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    clean_wl = CostFeatures(
        vec_flops=2.0e10,
        gm_bytes=16.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
    )
    scalar_asm = (
        "for (int i = 0; i < 1024; ++i) {\n"
        "  out_gm[i] = in_gm[i] * 2.0f + 1.0f;\n"
        "}\n"
    )
    bypass_asm = (
        "for (int i = 0; i < 1024; ++i) {\n"
        "  tl::write_gm_bypass_dcache<half>(out_gm + i, val);\n"
        "}\n"
    )
    scalar_sum = parse_bisheng_res_usage(asm_or_dump_text=scalar_asm)
    bypass_sum = parse_bisheng_res_usage(asm_or_dump_text=bypass_asm)
    pred_clean = model.predict(clean_wl)
    pred_scalar = model.predict(clean_wl, bisheng_res=scalar_sum)
    pred_bypass = model.predict(clean_wl, bisheng_res=bypass_sum)

    assert pred_scalar.pred_us > pred_clean.pred_us * 2.0
    assert pred_bypass.pred_us > pred_clean.pred_us * 2.0


def test_experience_lake_append_unmodeled_anomaly_writes_compact_jsonl(tmp_path: Path) -> None:
    """ExperienceLake.append_unmodeled_anomaly creates parent dirs and writes compact JSONL."""
    lake = ExperienceLake(tmp_path / "nested" / "lake")
    record = {
        "measured_us": 1000.0,
        "analytical_us": 10.0,
        "log_residual": math.log(100.0),
        "topology_bucket": "aiv_only:gm_ub_stream",
        "dominant_bound": "vector",
        "extra_signals": {"scalar_compute_loop_fallback": 1.0},
    }
    out_path = lake.append_unmodeled_anomaly(record)
    assert out_path == lake.root_dir / "unmodeled_anomalies.jsonl"
    assert out_path.is_file()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert ": " not in lines[0]
    parsed = json.loads(lines[0])
    assert parsed["measured_us"] == pytest.approx(1000.0)
    assert parsed["topology_bucket"] == "aiv_only:gm_ub_stream"


def test_observe_residual_anomaly_guard_skips_ridge_and_logs_outliers(tmp_path: Path) -> None:
    """Outliers with |ln(measured/analytical)| > 2.5 log to JSONL without polluting Ridge."""
    target = AscendTarget("ascend.950pr")
    lake = ExperienceLake(tmp_path / "lake")
    model = HybridAscendCostModel(target, anomaly_lake=lake)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    wl = CostFeatures(
        vec_flops=2.0e10,
        gm_bytes=16.0e6,
        total_tiles=56,
        contig_burst_bytes=32768,
        topology_axes=axes,
        extra_signals={"custom_diag": 2.0},
    )

    t01 = model._compute_tier0_and_tier1(wl)
    expected_analytical = max(1e-6, t01.tier0_us + t01.tier1_penalty_us)
    assert model.predict(wl).pred_us == pytest.approx(expected_analytical)

    for _ in range(4):
        ret = model.observe(wl, expected_analytical * 1.5)
        assert ret == pytest.approx(expected_analytical)

    assert len(model._samples) == 4
    weights_before = dict(model.weights_by_name)
    bucket_weights_before = {
        k: dict(v) for k, v in model.bucket_weights_by_name.items()
    }
    assert bucket_weights_before["aiv_only:gm_ub_stream"]["intercept"] > 0.0

    ret_high = model.observe(wl, expected_analytical * math.exp(2.6))
    ret_low = model.observe(wl, expected_analytical * math.exp(-2.6))
    assert ret_high == pytest.approx(expected_analytical)
    assert ret_low == pytest.approx(expected_analytical)

    assert len(model._samples) == 4
    assert model.weights_by_name == weights_before
    assert model.bucket_weights_by_name == bucket_weights_before

    anomaly_file = lake.root_dir / "unmodeled_anomalies.jsonl"
    assert anomaly_file.is_file()
    rows = [json.loads(ln) for ln in anomaly_file.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rows) == 2
    assert rows[0]["log_residual"] == pytest.approx(2.6)
    assert rows[1]["log_residual"] == pytest.approx(-2.6)
    for row in rows:
        assert row["analytical_us"] == pytest.approx(expected_analytical)
        assert row["topology_bucket"] == "aiv_only:gm_ub_stream"
        assert row["dominant_bound"] == t01.dominant
        assert row["extra_signals"] == {"custom_diag": 2.0}

    cal_path = tmp_path / "cal_dir" / "calibration.json"
    cal_model = HybridAscendCostModel(target, calibration_path=cal_path)
    cal_model.observe(wl, expected_analytical * math.exp(2.49))
    assert len(cal_model._samples) == 1
    default_anomaly = cal_path.parent / "unmodeled_anomalies.jsonl"
    assert not default_anomaly.exists()

    cal_model.observe(wl, expected_analytical * math.exp(2.51))
    assert len(cal_model._samples) == 1
    assert default_anomaly.is_file()
    assert len(default_anomaly.read_text(encoding="utf-8").splitlines()) == 1


def test_topology_bucket_and_basis_gates_remain_intact() -> None:
    """TopologyAxes buckets and ResidualBasisRegistry sample gates remain unmodified."""
    assert topology_bucket_for_axes(None) == "global"
    combos = (
        ("aiv_only", "gm_ub_stream", "affine_simd"),
        ("aic_only", "gm_l1_l0_fixpipe", "affine_simd"),
        ("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
        ("aiv_only", "ub_random_atomic", "irregular_simt"),
    )
    for ce, mr, ir in combos:
        axes = TopologyAxes(ce, mr, ir)
        assert topology_bucket_for_axes(axes) == f"{ce}:{mr}"

    gates = {spec.name: spec.min_bucket_samples for spec in DEFAULT_RESIDUAL_BASIS_SPECS}
    assert gates == {
        "intercept": 2,
        "log_tier0_scale": 6,
        "cube_bound_share": 6,
        "vec_bound_share": 6,
        "wave_tail_waste": 4,
        "l2_miss_ratio": 6,
        "short_dma_burst_ind": 4,
        "tier1_penalty_ratio": 4,
        "launch_overhead_ratio": 6,
        "is_coupled_1c2v": 8,
        "is_aic_only": 8,
        "is_irregular_simt": 8,
    }
