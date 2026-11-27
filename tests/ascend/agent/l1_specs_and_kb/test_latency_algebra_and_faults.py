"""Unit tests for conservative latency algebra, same-case eta, residuals, and fault types."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from tilefoundry_ascend.agent.kb.schema import SymptomFamily, compute_symptom_simplex
from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec, compute_case_aligned_eta
from tilefoundry_ascend.agent.specs.fault_types import (
    FaultPlane,
    L3Resolution,
    TypedFault,
    TypedFaultError,
    coerce_fault_plane,
)
from tilefoundry_ascend.agent.specs.latency_algebra import (
    MAD_TO_SIGMA,
    AnchorConservationPoint,
    StaticDynamicResidual,
    calibrate_dispatch_floor_from_series,
    calibrate_dispatch_floor_us,
    check_family_and_sibling_non_regression,
    check_guardrail_non_regression,
    check_regime_net_improvement,
    compute_adaptive_poison_pill_timeout_sec,
    compute_paired_sigma_us,
    compute_static_dynamic_residual,
    geometric_speedup,
    is_regime_saturated,
    regime_bandit_ucb_us,
    total_recoverable_sched_us,
)
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


@dataclass(frozen=True)
class _CompilerSummaryStub:
    spill_bytes: int = 0
    stack_bytes: int = 0
    vec_regs: int = 16
    scalar_regs: int = 12
    vf_fusion_breaks: int = 0
    alias_serialized_buffers: int = 0
    dual_issue_ipc_collapse: float = 0.0
    dcache_dcci_flushes: int = 0
    has_local_membar_in_vf_call: bool = False
    software_loops: int = 0
    vloopv2_loops: int = 1
    extra_signals: dict[str, float] = field(default_factory=dict)


def _pt(
    case_num: int,
    *,
    regime_tag: str = "decode",
    ref_us: float = 100.0,
    inc_us: float = 50.0,
    cand_us: float = 40.0,
    t_roof_us: float = 10.0,
    paired_rel_mad: float = 0.02,
    accuracy_passed: bool = True,
    cand_burst_cv: float = 0.0,
    ref_burst_cv: float = 0.0,
    burst_count: int = 1,
) -> AnchorConservationPoint:
    return AnchorConservationPoint(
        case_num=case_num,
        regime_tag=regime_tag,
        ref_us=ref_us,
        inc_us=inc_us,
        cand_us=cand_us,
        t_roof_us=t_roof_us,
        paired_rel_mad=paired_rel_mad,
        accuracy_passed=accuracy_passed,
        cand_burst_cv=cand_burst_cv,
        ref_burst_cv=ref_burst_cv,
        burst_count=burst_count,
    )


def test_paired_sigma_batch_stderr_and_recoverable_sched() -> None:
    """Verify paired MAD scaling, burst CV batch standard error, and recoverable latency."""
    assert MAD_TO_SIGMA == pytest.approx(1.482602218505602)
    assert compute_paired_sigma_us(100.0, 0.0, 0.0, 0.0, 1) == pytest.approx(0.0)
    assert compute_paired_sigma_us(100.0, 0.0, 0.468, 0.10, 16) == pytest.approx(
        100.0 * 0.468 / 4.0
    )
    assert compute_paired_sigma_us(100.0, 0.25, 0.40, 0.10, 4) == pytest.approx(25.0)
    assert compute_paired_sigma_us(float("nan"), 0.5, 0.5, 0.5, 0) == pytest.approx(0.0)
    assert compute_paired_sigma_us(100.0, -0.2, -0.3, float("inf"), -3) == pytest.approx(0.0)

    p_mad = _pt(1, inc_us=100.0, paired_rel_mad=0.05, cand_us=35.0, t_roof_us=12.0)
    assert p_mad.sigma_paired_us() == pytest.approx(MAD_TO_SIGMA * 0.05 * 100.0)
    assert p_mad.recoverable_sched_us(dispatch_floor_us=3.0) == pytest.approx(20.0)
    assert p_mad.recoverable_sched_us(dispatch_floor_us=30.0) == pytest.approx(0.0)

    p_burst = _pt(
        2,
        inc_us=100.0,
        cand_us=100.0,
        paired_rel_mad=0.0,
        cand_burst_cv=0.40,
        ref_burst_cv=0.10,
        burst_count=4,
    )
    assert p_burst.sigma_paired_us() == pytest.approx(20.0)


def test_conservative_two_sided_dispatch_floor_and_bandit_saturation() -> None:
    """Verify dispatch floor upper-bounding, saturation, UCB decay, and telescoping."""
    case7 = _pt(
        7,
        regime_tag="split_kv_decode",
        ref_us=4.56,
        inc_us=23.68,
        cand_us=23.68,
        t_roof_us=1.52,
    )
    cap_no_null = calibrate_dispatch_floor_us([case7])
    assert cap_no_null == pytest.approx(4.56 - 1.52)
    assert case7.recoverable_sched_us(cap_no_null) > 18.0
    assert calibrate_dispatch_floor_us([case7], null_launch_us=2.0) == pytest.approx(2.0)
    assert calibrate_dispatch_floor_us([case7], null_launch_us=10.0) == pytest.approx(4.56 - 1.52)
    assert calibrate_dispatch_floor_us([], null_launch_us=1.75) == pytest.approx(1.75)
    assert calibrate_dispatch_floor_us([case7], null_launch_us=float("nan")) == pytest.approx(
        4.56 - 1.52
    )

    assert calibrate_dispatch_floor_from_series(
        [23.68, 30.74], [4.56, 30.74], [1.52, 10.25], null_launch_us=2.1
    ) == pytest.approx(2.1)
    assert calibrate_dispatch_floor_from_series(
        [23.68], [4.56], [1.52], null_launch_us=9.5
    ) == pytest.approx(4.56 - 1.52)
    assert calibrate_dispatch_floor_from_series([], [], [], null_launch_us=1.5) == pytest.approx(
        1.5
    )

    pts = [
        _pt(1, inc_us=20.0, cand_us=15.0, t_roof_us=11.0, paired_rel_mad=0.05),
        _pt(2, inc_us=80.0, cand_us=70.0, t_roof_us=50.0, paired_rel_mad=0.02),
        _pt(3, inc_us=10.0, cand_us=1.0, t_roof_us=0.5, accuracy_passed=False),
    ]
    assert calibrate_dispatch_floor_us(pts) == pytest.approx(4.0)
    assert calibrate_dispatch_floor_us([pts[2]]) == pytest.approx(0.0)
    assert total_recoverable_sched_us(pts[:2], 4.0) == pytest.approx(16.0)
    assert regime_bandit_ucb_us(pts[:2], 4.0, 0) == pytest.approx(16.0)
    assert regime_bandit_ucb_us(pts[:2], 4.0, 3) == pytest.approx(8.0)
    assert regime_bandit_ucb_us(pts[:2], 4.0, -2) == pytest.approx(16.0)
    assert is_regime_saturated(pts[:2], 4.0, sigma_mult=2.0) is False
    sat_pts = [
        _pt(1, inc_us=20.0, cand_us=15.0, t_roof_us=10.0, paired_rel_mad=0.05),
        _pt(2, inc_us=50.0, cand_us=55.0, t_roof_us=50.0, paired_rel_mad=0.05),
    ]
    assert is_regime_saturated(sat_pts, 4.0, sigma_mult=2.0) is True
    assert is_regime_saturated([], 4.0) is False

    t0, t1, t2 = (120.0, 48.0, 200.0), (60.0, 24.0, 100.0), (30.0, 16.0, 50.0)
    step_logs = sum(
        math.log(
            geometric_speedup(
                [_pt(i, ref_us=a, cand_us=b) for i, (a, b) in enumerate(zip(pa, pb))]
            )
        )
        for pa, pb in ((t0, t1), (t1, t2))
    )
    overall = geometric_speedup(
        [_pt(i, ref_us=a, cand_us=b) for i, (a, b) in enumerate(zip(t0, t2))]
    )
    assert math.log(overall) == pytest.approx(step_logs, rel=1e-12, abs=1e-12)
    assert geometric_speedup([]) == pytest.approx(0.0)
    assert geometric_speedup([_pt(1, ref_us=10.0, cand_us=0.0)]) == pytest.approx(0.0)


def test_guardrails_net_improvement_timeout_and_same_case_eta() -> None:
    """Verify two-sided guardrail checks, net improvement thresholds, timeout, and case eta."""
    ok_pt = _pt(1, inc_us=100.0, cand_us=104.0, paired_rel_mad=0.01)
    reg_pt = _pt(2, inc_us=100.0, cand_us=105.0, paired_rel_mad=0.01)
    bad_acc = _pt(3, inc_us=100.0, cand_us=50.0, accuracy_passed=False)
    assert check_guardrail_non_regression(ok_pt, sigma_mult=3.0) is True
    assert check_guardrail_non_regression(reg_pt, sigma_mult=3.0) is False
    assert check_guardrail_non_regression(bad_acc) is False

    passed_all, failed_empty = check_family_and_sibling_non_regression([ok_pt], [ok_pt])
    assert passed_all is True and failed_empty == []
    passed_mix, failed_ids = check_family_and_sibling_non_regression(
        [ok_pt, reg_pt], [bad_acc, reg_pt]
    )
    assert passed_mix is False and failed_ids == [2, 3]

    improve_pts = [
        _pt(10, inc_us=100.0, cand_us=95.0, paired_rel_mad=0.01),
        _pt(11, inc_us=100.0, cand_us=97.0, paired_rel_mad=0.01),
    ]
    ok, gain, thresh = check_regime_net_improvement(improve_pts, sigma_mult=2.0)
    assert ok is True and gain == pytest.approx(8.0) and gain > thresh

    noise_pts = [
        _pt(10, inc_us=100.0, cand_us=99.0, paired_rel_mad=0.01),
        _pt(11, inc_us=100.0, cand_us=99.0, paired_rel_mad=0.01),
    ]
    ok_noise, gain_noise, thresh_noise = check_regime_net_improvement(noise_pts, sigma_mult=2.0)
    assert ok_noise is False and thresh_noise > gain_noise
    assert check_regime_net_improvement([]) == (False, 0.0, 0.0)
    assert check_regime_net_improvement([bad_acc])[0] is False

    assert compute_adaptive_poison_pill_timeout_sec([10.0, 20.0]) == pytest.approx(10.0)
    assert compute_adaptive_poison_pill_timeout_sec([1_000_000.0]) == pytest.approx(25.0)
    assert compute_adaptive_poison_pill_timeout_sec([5_000_000.0]) == pytest.approx(60.0)

    anchor = AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5)
    assert compute_case_aligned_eta(anchor, 2.0) == pytest.approx(0.25)
    assert compute_case_aligned_eta(anchor, 0.25) == pytest.approx(1.0)
    assert compute_case_aligned_eta(anchor, 0.0) == pytest.approx(0.0)
    assert compute_case_aligned_eta(anchor, float("nan")) == pytest.approx(0.0)

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={},
    )
    static_roof = {"min_gm_bytes": 16384, "compute_floor_us": 345.0}
    obs_case = ObservationVector5D.from_tiers(
        ledger,
        telemetry={"anchor_case_idx": 2, "case_t_hw_us": 0.5, "dur_us": 2.0},
        roofline=static_roof,
    )
    assert obs_case.effective_compute_eta == pytest.approx(0.25)
    obs_eta = ObservationVector5D.from_tiers(
        ledger,
        telemetry={"anchor_case_idx": 2, "roofline_eta": 0.30, "dur_us": 2.0},
        roofline=static_roof,
    )
    assert obs_eta.effective_compute_eta == pytest.approx(0.30)


def test_static_dynamic_residual_compute_and_roundtrip() -> None:
    """Verify StaticDynamicResidual computation via CompilerSummaryLike and serialization."""
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100},
        waterlines={},
        gm_roundtrip_bytes=4096,
        uses_dynamic_offset_pipelining=True,
        ub_scalar_compute_loops=2,
    )
    comp = _CompilerSummaryStub(
        spill_bytes=32,
        stack_bytes=64,
        vf_fusion_breaks=3,
        alias_serialized_buffers=2,
        dual_issue_ipc_collapse=0.45,
        dcache_dcci_flushes=1,
        has_local_membar_in_vf_call=True,
        software_loops=2,
        vloopv2_loops=0,
        extra_signals={"dcache_bypass_narrow_store": 4.0},
    )
    residual = compute_static_dynamic_residual(
        ledger,
        {
            "dur_us": 20.0,
            "aiv_vec_ratio": 0.40,
            "mte2_wait_ratio": 0.35,
            "sync_stall_ratio": 0.25,
            "bank_conflict_ratio": 0.15,
            "l2_hit_rate": 0.72,
        },
        compiler_summary=comp,
        roofline={"min_gm_bytes": 4096, "compute_floor_us": 4.0},
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
    )
    assert residual.has_l1_compiler_summary is True
    assert residual.predicted_bottleneck_util == pytest.approx(1.0)
    assert residual.pseudo_double_buffer_serialized is True
    assert residual.spill_bytes == 32
    assert residual.stack_bytes == 64
    assert residual.vf_fusion_breaks == 3
    assert residual.alias_serialized_buffers == 2
    assert residual.dual_issue_ipc_collapse == pytest.approx(0.45)
    assert residual.dcache_dcci_flushes == 1
    assert residual.has_local_membar_in_vf_call is True
    assert residual.vloopv2_hit is False
    assert residual.scalar_compute_loop_fallback == pytest.approx(2.0)
    assert residual.dcache_bypass_narrow_store == pytest.approx(4.0)
    assert residual.measured_l2_fields
    assert residual.observation is not None
    expected_unmodeled = dict(compute_symptom_simplex(residual.observation))[
        SymptomFamily.UNMODELED_SILICON_RESIDUAL
    ]
    assert residual.unexplained_gap_ratio == pytest.approx(expected_unmodeled, abs=1e-6)

    restored = StaticDynamicResidual.from_dict(residual.to_dict())
    assert restored == residual
    with pytest.raises(ValueError, match="residual field 'observation' must be a mapping"):
        StaticDynamicResidual.from_dict({"observation": 123})


def test_fault_planes_l3_resolution_and_typed_fault() -> None:
    """Verify FaultPlane taxonomy, alias coercion, bandit policy, and TypedFault."""
    non_penalizing = {
        FaultPlane.NONE,
        FaultPlane.INFRA_OCCUPANCY,
        FaultPlane.INFRA_DEVICE_FAULT,
        FaultPlane.LLM_TRANSPORT,
    }
    for plane in FaultPlane:
        assert plane.penalizes_bandit() is (plane not in non_penalizing)

    assert [r.value for r in L3Resolution] == [
        "NOT_RUN",
        "PROMOTED",
        "CANDIDATE_REJECTED",
        "INFRA_DEFERRED",
    ]
    assert coerce_fault_plane("STAGE_A_STATIC") is FaultPlane.STATIC_CONTRACT
    assert coerce_fault_plane("STAGE_B_COMPILE") is FaultPlane.COMPILER_LOWERING
    assert coerce_fault_plane("INFRA_TOOLCHAIN") is FaultPlane.INFRA_DEVICE_FAULT
    assert coerce_fault_plane("COST_PRESCREEN") is FaultPlane.STATIC_CONTRACT
    assert coerce_fault_plane("") is FaultPlane.NONE
    assert coerce_fault_plane("UNKNOWN_TAG") is FaultPlane.UNMODELED_FAULT

    fault = TypedFault(
        plane=FaultPlane.CANDIDATE_NUMERIC,
        code="L3_ATOL_FAIL",
        failed_tier="l3_accuracy",
        detail="max diff 0.25 on case 14",
        failed_cases=(14, 17),
        retriable=False,
    )
    assert TypedFault.from_dict(fault.to_dict()) == fault
    coerced = TypedFault.from_dict(
        {"plane": "NOT_REAL", "code": "X", "failed_cases": [True, 4, "9", "bad"]}
    )
    assert coerced.plane is FaultPlane.UNMODELED_FAULT
    assert coerced.failed_cases == (4, 9)

    err = TypedFaultError(fault)
    assert err.fault == fault
    assert "case 14" in str(err)
