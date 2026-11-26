"""Tests for conservative dispatch floor, paired SEM, case-aligned eta, and simplex."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tests.ascend.agent import test_surrogate_batch_npu as npu_batch
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
    collect_distinct_static_width_dims,
    compute_case_aligned_eta,
)
from tilefoundry_ascend.agent.campaign.latency_algebra import (
    MAD_TO_SIGMA,
    AnchorConservationPoint,
    calibrate_dispatch_floor_from_series,
    calibrate_dispatch_floor_us,
    check_regime_net_improvement,
    compute_paired_sigma_us,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import to_conservation_point
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import AnchorEvalMetrics
from tilefoundry_ascend.agent.kb.schema import (
    DETERMINISTIC_SYMPTOM_FAMILIES,
    AscendCausalRule,
    SymptomFamily,
    compute_symptom_simplex,
    normalize_runtime_backend,
    normalize_symptom_simplex,
)
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger


def test_paired_sigma_eliminates_magic_floor_and_uses_burst_sem() -> None:
    """Verify compute_paired_sigma_us and AnchorConservationPoint drop the 1e-3 magic floor."""
    assert compute_paired_sigma_us(100.0, 0.0, 0.0, 0.0, 1) == pytest.approx(0.0)
    assert compute_paired_sigma_us(100.0, 0.0, 0.468, 0.10, 16) == pytest.approx(
        100.0 * 0.468 / 4.0
    )
    assert compute_paired_sigma_us(100.0, 0.25, 0.40, 0.10, 4) == pytest.approx(25.0)
    assert compute_paired_sigma_us(float("nan"), 0.5, 0.5, 0.5, 0) == pytest.approx(0.0)
    assert compute_paired_sigma_us(100.0, -0.2, -0.3, float("inf"), -3) == pytest.approx(0.0)

    pt_zero = AnchorConservationPoint(
        case_num=1,
        regime_tag="prefill_compute",
        ref_us=34.85,
        inc_us=34.85,
        cand_us=34.85,
        t_roof_us=11.62,
        paired_rel_mad=0.0,
        accuracy_passed=True,
    )
    assert pt_zero.sigma_paired_us() == pytest.approx(0.0)

    pt_sem = AnchorConservationPoint(
        case_num=1,
        regime_tag="prefill_compute",
        ref_us=34.85,
        inc_us=34.85,
        cand_us=34.85,
        t_roof_us=11.62,
        paired_rel_mad=0.0,
        accuracy_passed=True,
        cand_burst_cv=0.468,
        ref_burst_cv=0.20,
        burst_count=9,
    )
    assert pt_sem.sigma_paired_us() == pytest.approx(34.85 * 0.468 / 3.0)

    pt_mad = AnchorConservationPoint(
        case_num=1,
        regime_tag="prefill_compute",
        ref_us=100.0,
        inc_us=100.0,
        cand_us=90.0,
        t_roof_us=10.0,
        paired_rel_mad=0.20,
        accuracy_passed=True,
        cand_burst_cv=0.20,
        burst_count=4,
    )
    assert pt_mad.sigma_paired_us() == pytest.approx(MAD_TO_SIGMA * 0.20 * 100.0)


def test_dispatch_floor_bounded_by_ref_and_null_launch_releases_mla_case7() -> None:
    """Verify ref_us and null_launch_us upper-bound dispatch_floor_us on mla Case 7."""
    case7 = AnchorConservationPoint(
        case_num=7,
        regime_tag="split_kv_decode",
        ref_us=4.56,
        inc_us=23.68,
        cand_us=23.68,
        t_roof_us=1.52,
        paired_rel_mad=0.0,
        accuracy_passed=True,
    )
    cap_no_null = calibrate_dispatch_floor_us([case7])
    assert cap_no_null == pytest.approx(4.56 - 1.52)
    assert case7.recoverable_sched_us(cap_no_null) > 18.0

    floor_with_null = calibrate_dispatch_floor_us([case7], null_launch_us=2.0)
    assert floor_with_null == pytest.approx(2.0)
    assert case7.recoverable_sched_us(floor_with_null) > 18.0

    tighter_ref_cap = calibrate_dispatch_floor_us([case7], null_launch_us=10.0)
    assert tighter_ref_cap == pytest.approx(4.56 - 1.52)

    assert calibrate_dispatch_floor_us([], null_launch_us=1.75) == pytest.approx(1.75)
    assert calibrate_dispatch_floor_us([case7], null_launch_us=0.0) == pytest.approx(4.56 - 1.52)
    assert calibrate_dispatch_floor_us([case7], null_launch_us=float("nan")) == pytest.approx(
        4.56 - 1.52
    )

    series_floor = calibrate_dispatch_floor_from_series(
        [23.68, 30.74],
        [4.56, 30.74],
        [1.52, 10.25],
        null_launch_us=2.1,
    )
    assert series_floor == pytest.approx(2.1)
    series_cap = calibrate_dispatch_floor_from_series(
        [23.68],
        [4.56],
        [1.52],
        null_launch_us=9.5,
    )
    assert series_cap == pytest.approx(4.56 - 1.52)
    assert calibrate_dispatch_floor_from_series([], [], [], null_launch_us=1.5) == pytest.approx(
        1.5
    )


def test_null_launch_helper_and_fast_response_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _measure_null_launch_us fallback and _fast_response_payload batch field."""
    monkeypatch.setattr(torch, "npu", None, raising=False)
    assert npu_batch._measure_null_launch_us() is None

    batch_with_null = npu_batch._fast_response_payload(
        request_id="req-1",
        request_sha256="a" * 64,
        mode="batch",
        status="ok",
        null_launch_us=1.85,
    )
    assert batch_with_null["null_launch_us"] == pytest.approx(1.85)
    assert batch_with_null["schema"] == npu_batch.FAST_RESPONSE_SCHEMA
    assert batch_with_null["mode"] == "batch"

    for bad_val in (None, 0.0, -1.0, float("nan"), float("inf")):
        batch_without_null = npu_batch._fast_response_payload(
            request_id="req-2",
            request_sha256="b" * 64,
            mode="batch",
            status="ok",
            null_launch_us=bad_val,
        )
        assert "null_launch_us" not in batch_without_null

    profile_payload = npu_batch._fast_response_payload(
        request_id="req-3",
        request_sha256="c" * 64,
        mode="profile",
        status="ok",
        null_launch_us=1.85,
    )
    assert "null_launch_us" not in profile_payload


def test_case_aligned_eta_and_static_width_collection(tmp_path: Path) -> None:
    """Verify compute_case_aligned_eta clamping and static width deduplication."""
    anchor = AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5)
    assert compute_case_aligned_eta(anchor, 2.0) == pytest.approx(0.25)
    assert compute_case_aligned_eta(anchor, 0.25) == pytest.approx(1.0)
    assert compute_case_aligned_eta(anchor, 0.0) == pytest.approx(0.0)
    assert compute_case_aligned_eta(anchor, -5.0) == pytest.approx(0.0)
    assert compute_case_aligned_eta(anchor, float("nan")) == pytest.approx(0.0)

    task_dir = tmp_path / "tasks" / "level4" / "attention_residual"
    task_dir.mkdir(parents=True)
    (task_dir / "cases.yaml").write_text(
        "\n".join(
            [
                "cases:",
                "  - case_id: 2",
                "    input_shape: [[2, 1, 1, 4096], [1, 1, 4096]]",
                "    dtype: bfloat16",
                "  - case_id: 4",
                "    input_shape: [[8, 1, 16, 1000], [1, 16, 1000]]",
                "    dtype: bfloat16",
                "  - case_id: 5",
                "    input_shape: [[8, 1, 32, 1000], [1, 32, 1000]]",
                "    dtype: bfloat16",
                "  - case_id: 14",
                "    input_shape: [[8, 1, 8, 777], [1, 8, 777]]",
                "    dtype: float16",
                "  - case_id: 16",
                "    input_shape: [[8, 1, 4096, 4096], [1, 4096, 4096]]",
                "    dtype: bfloat16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    unaligned_spec = RegimeTargetSpec(
        regime_tag="unaligned_d",
        builder_candidates=("_build_ar_unaligned_func",),
        sample_builder_kwargs={"tl_dtype": "bfloat16"},
        representative_anchor_dims={"N": 8, "B": 1, "S": 16, "D": 1000},
    )
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.93, 3.31),
        AnchorCaseSpec("level4/attention_residual", 16, "aligned_prefill", 1098.0, 366.0),
    )
    guardrails = (
        GuardrailCaseSpec("level4/attention_residual", 14, "unaligned_d", 67.0, 22.0),
    )
    widths = collect_distinct_static_width_dims(
        "attention_residual",
        unaligned_spec,
        anchors,
        guardrails,
        cannbench_repo=tmp_path,
    )
    extracted = sorted((int(w["D"]), str(w["tl_dtype"])) for w in widths)
    assert extracted == [(777, "float16"), (1000, "bfloat16")]


def test_static_width_collection_collapses_same_bytewidth_dtypes(tmp_path: Path) -> None:
    """Collapse equal-bytewidth float16/bfloat16 widths while keeping float32 distinct."""
    task_dir = tmp_path / "tasks" / "level4" / "attention_residual"
    task_dir.mkdir(parents=True)
    (task_dir / "cases.yaml").write_text(
        "\n".join(
            [
                "cases:",
                "  - case_id: 1",
                "    input_shape: [[8, 1, 4096, 4096], [1, 4096, 4096]]",
                "    dtype: float16",
                "  - case_id: 2",
                "    input_shape: [[8, 1, 4096, 4096], [1, 4096, 4096]]",
                "    dtype: bfloat16",
                "  - case_id: 3",
                "    input_shape: [[8, 1, 4096, 4096], [1, 4096, 4096]]",
                "    dtype: float32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    aligned_spec = RegimeTargetSpec(
        regime_tag="aligned_prefill",
        builder_candidates=("_build_ar_aligned_func",),
        sample_builder_kwargs={"tl_dtype": "bfloat16"},
        representative_anchor_dims={"N": 8, "B": 1, "S": 4096, "D": 4096},
    )
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 1, "aligned_prefill", 1098.0, 366.0),
        AnchorCaseSpec("level4/attention_residual", 2, "aligned_prefill", 1098.0, 366.0),
        AnchorCaseSpec("level4/attention_residual", 3, "aligned_prefill", 2196.0, 732.0),
    )
    widths = collect_distinct_static_width_dims(
        "attention_residual",
        aligned_spec,
        anchors,
        cannbench_repo=tmp_path,
    )
    assert len(widths) == 2
    assert [(int(w["D"]), str(w["tl_dtype"])) for w in widths] == [
        (4096, "float16"),
        (4096, "float32"),
    ]


def test_observation_5d_prefers_case_aligned_eta_and_clamps() -> None:
    """Verify ObservationVector5D.from_tiers avoids cross-case roofline division."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={},
    )
    static_roofline = {"min_gm_bytes": 16384, "compute_floor_us": 345.0}

    case_aligned_obs = ObservationVector5D.from_tiers(
        ledger,
        telemetry={
            "anchor_case_idx": 2,
            "case_t_hw_us": 0.5,
            "dur_us": 2.0,
        },
        roofline=static_roofline,
    )
    assert case_aligned_obs.effective_compute_eta == pytest.approx(0.25)

    eta_aligned_obs = ObservationVector5D.from_tiers(
        ledger,
        telemetry={
            "anchor_case_idx": 2,
            "roofline_eta": 0.30,
            "dur_us": 2.0,
        },
        roofline=static_roofline,
    )
    assert eta_aligned_obs.effective_compute_eta == pytest.approx(0.30)

    clamped_obs = ObservationVector5D.from_tiers(
        ledger,
        telemetry={"dur_us": 2.0},
        roofline=static_roofline,
    )
    assert clamped_obs.effective_compute_eta == pytest.approx(1.0)


def test_normalize_symptom_simplex_and_backend_isolation() -> None:
    """Verify decoupled simplex normalization and strict dialect filtering."""
    raw = {
        SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP: 0.6,
        SymptomFamily.SRAM_REGISTER_SPILL_STACK: 0.2,
        SymptomFamily.UNMODELED_SILICON_RESIDUAL: 0.9,
    }
    shares_at_roof = normalize_symptom_simplex(raw, roofline_eta=1.0, has_l2_pmu=False)
    assert shares_at_roof[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.75)
    assert shares_at_roof[SymptomFamily.SRAM_REGISTER_SPILL_STACK] == pytest.approx(0.25)
    assert shares_at_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)
    assert math.isclose(sum(shares_at_roof.values()), 1.0, rel_tol=1e-9)

    zero_raw = {fam: 0.0 for fam in DETERMINISTIC_SYMPTOM_FAMILIES}
    shares_under_roof = normalize_symptom_simplex(zero_raw, roofline_eta=0.5, has_l2_pmu=True)
    assert shares_under_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(1.0)
    assert sum(shares_under_roof[f] for f in DETERMINISTIC_SYMPTOM_FAMILIES) == pytest.approx(0.0)

    shares_at_exact_roof = normalize_symptom_simplex(zero_raw, roofline_eta=1.0, has_l2_pmu=True)
    assert shares_at_exact_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)

    simplex_at_roof = dict(
        compute_symptom_simplex(
            {
                "roofline_eta": 1.0,
                "transfer_inflation_ratio": 3.0,
                "l0_vec_cycles": 100,
                "has_l2_pmu": False,
            }
        )
    )
    assert simplex_at_roof[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(1.0)
    assert simplex_at_roof[SymptomFamily.UNMODELED_SILICON_RESIDUAL] == pytest.approx(0.0)

    assert normalize_runtime_backend("CCE_CPP20") == "ascendc_trackb"
    assert normalize_runtime_backend("ascendc_template") == "ascendc_trackb"
    assert normalize_runtime_backend("cce_trackb") == "ascendc_trackb"
    assert normalize_runtime_backend("hybrid_trackb") == "hybrid_trackb"
    assert normalize_runtime_backend("TileLang_DSL") == "tilelang_dsl"

    tl_only = AscendCausalRule(
        rule_id="R_TL_ONLY",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Use TileLang DSL.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
        compatible_backends=("tilelang_dsl",),
    )
    assert tl_only.supports_backend("tilelang_dsl") is True
    for other in ("cce_cpp20", "ascendc_template", "cce_trackb", "hybrid_trackb"):
        assert tl_only.supports_backend(other) is False

    cpp_only = AscendCausalRule(
        rule_id="R_CPP_ONLY",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Use CCE C++20.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
        compatible_backends=("cce_cpp20",),
    )
    assert cpp_only.supports_backend("cce_cpp20") is True
    assert cpp_only.supports_backend("ascendc_template") is True
    assert cpp_only.supports_backend("cce_trackb") is True
    assert cpp_only.supports_backend("tilelang_dsl") is False
    assert cpp_only.supports_backend("hybrid_trackb") is False


def test_real_mla_case7_dispatch_floor_and_recoverable_sched() -> None:
    """Verify real MLA sibling latencies keep Case 7 dispatch floor <= 3.04us."""
    mla_specs = (
        (1, "prefill_compute", 34.8532, 34.8532, 11.6177),
        (6, "prefill_compute", 30.7423, 30.7423, 10.2474),
        (7, "split_kv_decode", 23.68, 4.56, 1.52),
        (11, "split_kv_decode", 68.6341, 68.6341, 22.8780),
        (13, "bnsd_gather", 34.8532, 34.8532, 11.6177),
    )
    metrics = [
        AnchorEvalMetrics(
            case_num=cnum,
            regime_tag=rtag,
            accuracy_passed=True,
            cand_us=cand_us,
            ref_perf_us=ref_us,
            t_hw_us=roof_us,
            speedup=ref_us / cand_us,
            roofline_eta=roof_us / cand_us,
            eq3_score=ref_us / cand_us,
        )
        for cnum, rtag, cand_us, ref_us, roof_us in mla_specs
    ]
    points = [to_conservation_point(m, m) for m in metrics]
    case7_pt = next(p for p in points if p.case_num == 7)

    floor_no_null = calibrate_dispatch_floor_us(points, null_launch_us=None)
    assert floor_no_null <= 3.04 + 1e-9
    assert floor_no_null == pytest.approx(3.04)
    assert case7_pt.recoverable_sched_us(floor_no_null) > 18.0

    floor_with_null = calibrate_dispatch_floor_us(points, null_launch_us=1.8)
    assert floor_with_null <= 3.04
    assert floor_with_null == pytest.approx(1.8)
    assert case7_pt.recoverable_sched_us(floor_with_null) > 18.0


def test_to_conservation_point_isolates_paired_from_stale_unpaired_baseline() -> None:
    """Ignore stale unpaired inc burst CV when paired_speedup_vs_incumbent > 0."""
    inc = AnchorEvalMetrics(
        case_num=1,
        regime_tag="prefill_compute",
        accuracy_passed=True,
        cand_us=100.0,
        ref_perf_us=120.0,
        t_hw_us=20.0,
        speedup=1.2,
        roofline_eta=0.2,
        eq3_score=1.2,
        ring_burst_k=7,
        burst_batch_cv=0.092,
        paired_speedup_vs_incumbent=0.0,
        paired_rel_mad=0.045,
    )
    paired_cand = AnchorEvalMetrics(
        case_num=1,
        regime_tag="prefill_compute",
        accuracy_passed=True,
        cand_us=100.0 / 1.046,
        ref_perf_us=120.0,
        t_hw_us=20.0,
        speedup=1.2 * 1.046,
        roofline_eta=0.2 * 1.046,
        eq3_score=1.2 * 1.046,
        ring_burst_k=7,
        burst_batch_cv=0.008,
        paired_speedup_vs_incumbent=1.046,
        paired_rel_mad=0.008,
    )
    paired_pt = to_conservation_point(paired_cand, inc)
    assert paired_pt.ref_burst_cv == pytest.approx(0.0)
    assert paired_pt.cand_burst_cv == pytest.approx(0.008)
    assert paired_pt.paired_rel_mad == pytest.approx(0.008)
    assert paired_pt.burst_count == 7
    ok, gain_us, thresh_us = check_regime_net_improvement([paired_pt], sigma_mult=2.0)
    assert ok is True
    assert gain_us > thresh_us

    unpaired_cand = AnchorEvalMetrics(
        case_num=1,
        regime_tag="prefill_compute",
        accuracy_passed=True,
        cand_us=100.0 / 1.046,
        ref_perf_us=120.0,
        t_hw_us=20.0,
        speedup=1.2 * 1.046,
        roofline_eta=0.2 * 1.046,
        eq3_score=1.2 * 1.046,
        ring_burst_k=7,
        burst_batch_cv=0.008,
        paired_speedup_vs_incumbent=0.0,
        paired_rel_mad=0.008,
    )
    unpaired_pt = to_conservation_point(unpaired_cand, inc)
    assert unpaired_pt.ref_burst_cv == pytest.approx(0.092)
    assert unpaired_pt.paired_rel_mad == pytest.approx(0.045)
