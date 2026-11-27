"""Unit tests for latency conservation algebra, typed fault planes, and telemetry."""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import pytest


def _bootstrap_isolated_worktree() -> None:
    """Resolve worktree modules even when editable install or sibling edits differ."""
    root = Path(__file__).resolve().parents[3]
    ascend_root = str(root / "targets" / "ascend")
    if ascend_root not in sys.path:
        sys.path.insert(0, ascend_root)
    pkg_dir = root / "targets" / "ascend" / "tilefoundry_ascend"
    try:
        mod = importlib.import_module("tilefoundry_ascend.agent.specs.fault_types")
        mod_file = getattr(mod, "__file__", "") or ""
        if Path(mod_file).resolve().is_relative_to(root):
            return
    except ImportError:
        pass
    stale = [
        k
        for k in sys.modules
        if k == "tilefoundry_ascend" or k.startswith("tilefoundry_ascend.")
    ]
    for key in stale:
        del sys.modules[key]
    for mod_name, sub_path in (
        ("tilefoundry_ascend", pkg_dir),
        ("tilefoundry_ascend.agent", pkg_dir / "agent"),
        ("tilefoundry_ascend.agent.campaign", pkg_dir / "agent" / "campaign"),
        ("tilefoundry_ascend.agent.specs", pkg_dir / "agent" / "specs"),
    ):
        mod = types.ModuleType(mod_name)
        mod.__path__ = [str(sub_path)]
        sys.modules[mod_name] = mod


_bootstrap_isolated_worktree()

import tilefoundry_ascend.agent.campaign.funnel_telemetry as funnel_telemetry  # noqa: E402
from tilefoundry_ascend.agent.specs.anchor_specs import (  # noqa: E402
    AnchorCaseSpec,
    RegimeTargetSpec,
    inject_lake_counterexample_anchors,
)
from tilefoundry_ascend.agent.specs.fault_types import (  # noqa: E402
    FaultPlane,
    L3Resolution,
    TypedFault,
    TypedFaultError,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (  # noqa: E402
    RoundFunnelEvent,
    append_round_funnel_event,
    assert_unmodeled_fault_share_bounded,
)
from tilefoundry_ascend.agent.specs.latency_algebra import (  # noqa: E402
    MAD_TO_SIGMA,
    AnchorConservationPoint,
    calibrate_dispatch_floor_us,
    check_family_and_sibling_non_regression,
    check_guardrail_non_regression,
    check_regime_net_improvement,
    compute_adaptive_poison_pill_timeout_sec,
    geometric_speedup,
    is_regime_saturated,
    regime_bandit_ucb_us,
    total_recoverable_sched_us,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (  # noqa: E402
    append_counterexample_boundary,
)


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
    )


def test_anchor_conservation_point_sigma_and_recoverable_sched() -> None:
    """Verify paired MAD sigma scaling, burst CV SEM floor, and recoverable schedule."""
    assert MAD_TO_SIGMA == pytest.approx(1.482602218505602)

    p_mad = _pt(1, inc_us=100.0, paired_rel_mad=0.05, cand_us=35.0, t_roof_us=12.0)
    assert p_mad.sigma_paired_us() == pytest.approx(MAD_TO_SIGMA * 0.05 * 100.0)

    p_zero = _pt(2, inc_us=100.0, paired_rel_mad=0.0)
    assert p_zero.sigma_paired_us() == pytest.approx(0.0)

    p_burst = AnchorConservationPoint(
        case_num=3,
        regime_tag="decode",
        ref_us=100.0,
        inc_us=100.0,
        cand_us=100.0,
        t_roof_us=10.0,
        paired_rel_mad=0.0,
        accuracy_passed=True,
        cand_burst_cv=0.40,
        ref_burst_cv=0.10,
        burst_count=4,
    )
    assert p_burst.sigma_paired_us() == pytest.approx(20.0)

    p_tiny = _pt(4, inc_us=0.0, paired_rel_mad=0.0)
    assert p_tiny.sigma_paired_us() == pytest.approx(0.0)

    assert p_mad.recoverable_sched_us(dispatch_floor_us=3.0) == pytest.approx(20.0)
    assert p_mad.recoverable_sched_us(dispatch_floor_us=30.0) == pytest.approx(0.0)


def test_dispatch_floor_calibration_and_bandit_saturation() -> None:
    """Verify null-kernel and empirical floor calibration, saturation, and UCB decay."""
    pts = [
        _pt(1, inc_us=20.0, cand_us=15.0, t_roof_us=11.0, paired_rel_mad=0.05),
        _pt(2, inc_us=80.0, cand_us=70.0, t_roof_us=50.0, paired_rel_mad=0.02),
        _pt(3, inc_us=10.0, cand_us=1.0, t_roof_us=0.5, accuracy_passed=False),
    ]
    assert calibrate_dispatch_floor_us(pts, null_launch_us=2.5) == pytest.approx(2.5)
    assert calibrate_dispatch_floor_us(pts) == pytest.approx(4.0)
    assert calibrate_dispatch_floor_us([pts[2]]) == pytest.approx(0.0)
    assert calibrate_dispatch_floor_us([]) == pytest.approx(0.0)

    floor = 4.0
    assert total_recoverable_sched_us(pts[:2], floor) == pytest.approx(16.0)
    assert regime_bandit_ucb_us(pts[:2], floor, 0) == pytest.approx(16.0)
    assert regime_bandit_ucb_us(pts[:2], floor, 3) == pytest.approx(8.0)
    assert regime_bandit_ucb_us(pts[:2], floor, -2) == pytest.approx(16.0)
    assert is_regime_saturated(pts[:2], floor, sigma_mult=2.0) is False

    sat_pts = [
        _pt(1, inc_us=20.0, cand_us=15.0, t_roof_us=10.0, paired_rel_mad=0.05),
        _pt(2, inc_us=50.0, cand_us=55.0, t_roof_us=50.0, paired_rel_mad=0.05),
    ]
    assert is_regime_saturated(sat_pts, dispatch_floor_us=4.0, sigma_mult=2.0) is True
    assert is_regime_saturated([], dispatch_floor_us=4.0) is False


def test_geometric_speedup_telescoping_identity() -> None:
    """Verify log-space geometric speedup telescoping ln(T0/Tk) = sum ln(T_{j-1}/Tj)."""
    t0 = (120.0, 48.0, 200.0)
    t1 = (80.0, 32.0, 100.0)
    t2 = (60.0, 24.0, 80.0)
    t3 = (30.0, 16.0, 50.0)
    trajectory = (t0, t1, t2, t3)

    step_log_sum = 0.0
    for prev_t, next_t in zip(trajectory[:-1], trajectory[1:], strict=True):
        step_pts = [
            _pt(idx, ref_us=a, inc_us=a, cand_us=b)
            for idx, (a, b) in enumerate(zip(prev_t, next_t, strict=True))
        ]
        step_log_sum += math.log(geometric_speedup(step_pts))

    overall_pts = [
        _pt(idx, ref_us=a, inc_us=t2[idx], cand_us=b)
        for idx, (a, b) in enumerate(zip(t0, t3, strict=True))
    ]
    overall_speedup = geometric_speedup(overall_pts)
    assert math.log(overall_speedup) == pytest.approx(step_log_sum, rel=1e-12, abs=1e-12)
    assert overall_speedup == pytest.approx((4.0 * 3.0 * 4.0) ** (1.0 / 3.0))
    assert geometric_speedup([]) == pytest.approx(0.0)
    assert geometric_speedup([_pt(1, ref_us=10.0, cand_us=0.0)]) == pytest.approx(0.0)


def test_guardrail_family_sibling_and_net_improvement_checks() -> None:
    """Verify 3-sigma guardrail, family+sibling checks, 2-sigma net gain, and timeout."""
    base = _pt(1, inc_us=100.0, cand_us=104.0, paired_rel_mad=0.01)
    assert check_guardrail_non_regression(base, sigma_mult=3.0) is True

    regressed = _pt(2, inc_us=100.0, cand_us=105.0, paired_rel_mad=0.01)
    assert check_guardrail_non_regression(regressed, sigma_mult=3.0) is False

    bad_acc = _pt(3, inc_us=100.0, cand_us=50.0, accuracy_passed=False)
    assert check_guardrail_non_regression(bad_acc) is False

    passed_all, failed_empty = check_family_and_sibling_non_regression([base], [base])
    assert passed_all is True
    assert failed_empty == []

    passed_mix, failed_ids = check_family_and_sibling_non_regression(
        [base, regressed],
        [bad_acc, regressed],
    )
    assert passed_mix is False
    assert failed_ids == [2, 3]

    improve_pts = [
        _pt(10, inc_us=100.0, cand_us=95.0, paired_rel_mad=0.01),
        _pt(11, inc_us=100.0, cand_us=97.0, paired_rel_mad=0.01),
    ]
    ok, gain, thresh = check_regime_net_improvement(improve_pts, sigma_mult=2.0)
    expected_sigma = MAD_TO_SIGMA * 0.01 * 100.0
    assert gain == pytest.approx(8.0)
    assert thresh == pytest.approx(2.0 * math.sqrt(2.0) * expected_sigma)
    assert ok is True

    noise_pts = [
        _pt(10, inc_us=100.0, cand_us=99.0, paired_rel_mad=0.01),
        _pt(11, inc_us=100.0, cand_us=99.0, paired_rel_mad=0.01),
    ]
    ok_noise, gain_noise, thresh_noise = check_regime_net_improvement(noise_pts, sigma_mult=2.0)
    assert gain_noise == pytest.approx(2.0)
    assert ok_noise is False
    assert thresh_noise > gain_noise

    ok_acc, _, _ = check_regime_net_improvement(
        [_pt(12, inc_us=100.0, cand_us=10.0, accuracy_passed=False)]
    )
    assert ok_acc is False

    assert compute_adaptive_poison_pill_timeout_sec([10.0, 20.0]) == pytest.approx(10.0)
    assert compute_adaptive_poison_pill_timeout_sec([1_000_000.0]) == pytest.approx(25.0)
    assert compute_adaptive_poison_pill_timeout_sec([5_000_000.0]) == pytest.approx(60.0)


def test_fault_planes_l3_resolution_and_typed_fault() -> None:
    """Verify FaultPlane bandit policy, L3Resolution states, TypedFault, and errors."""
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

    fault = TypedFault(
        plane=FaultPlane.CANDIDATE_NUMERIC,
        code="L3_ATOL_FAIL",
        failed_tier="l3_accuracy",
        detail="max diff 0.25 on case 14",
        failed_cases=(14, 17),
        retriable=False,
    )
    restored = TypedFault.from_dict(fault.to_dict())
    assert restored == fault

    unknown = TypedFault.from_dict({"plane": "NOT_A_REAL_PLANE", "code": "X"})
    assert unknown.plane == FaultPlane.UNMODELED_FAULT

    err = TypedFaultError(fault)
    assert err.fault == fault
    assert "case 14" in str(err)


def test_funnel_telemetry_typed_fault_and_unmodeled_share(tmp_path: Path) -> None:
    """Verify regex scraping removal, RoundFunnelEvent typed faults, and share bound."""
    assert not hasattr(funnel_telemetry, "_ERROR_CODE_RULES")
    assert not hasattr(funnel_telemetry, "normalize_compile_error_code")

    base_evt = RoundFunnelEvent(
        run_id="r1",
        operator="mla",
        round_idx=1,
        regime_tag="split_kv_decode",
        topology_axes={"compute_engine": "cube"},
        candidate_id="c1",
        cand_sha="sha1",
        worker_kind="creative_explore",
    )
    assert base_evt.fault_plane == FaultPlane.NONE.value
    assert base_evt.l3_resolution == L3Resolution.NOT_RUN.value
    assert base_evt.is_real_npu is False

    fault = TypedFault(
        plane=FaultPlane.COMPILER_LOWERING,
        code="UB_SRAM_OVERFLOW",
        failed_tier="stage_b",
        detail="ub_bytes exceeded",
    )
    applied = base_evt.apply_typed_fault(fault)
    assert applied.fault_plane == FaultPlane.COMPILER_LOWERING.value
    assert applied.error_code_norm == "UB_SRAM_OVERFLOW"
    assert applied.failed_tier == "stage_b"

    noop_evt = RoundFunnelEvent.from_typed_fault(
        run_id="r1",
        operator="mla",
        round_idx=2,
        regime_tag="split_kv_decode",
        topology_axes={},
        candidate_id="c2",
        cand_sha="sha2",
        worker_kind="rule_codegen",
        fault=TypedFault(
            plane=FaultPlane.NOOP_IDENTICAL,
            code="NOOP_IDENTICAL",
            failed_tier="stage_a",
            detail="identical",
        ),
        l3_resolution=L3Resolution.CANDIDATE_REJECTED.value,
        is_real_npu=True,
    )
    assert noop_evt.noop_identical is True
    assert noop_evt.is_real_npu is True
    assert noop_evt.l3_resolution == L3Resolution.CANDIDATE_REJECTED.value

    out_path = append_round_funnel_event(tmp_path / "tel", noop_evt)
    assert out_path.is_file()

    assert assert_unmodeled_fault_share_bounded([base_evt]) == pytest.approx(0.0)

    unmodeled_evt = base_evt.apply_typed_fault(
        TypedFault(
            plane=FaultPlane.UNMODELED_FAULT,
            code="UNKNOWN",
            failed_tier="stage_b",
            detail="unclassified",
        )
    )
    events_ok = [base_evt] * 5 + [applied] * 9 + [noop_evt] + [unmodeled_evt]
    assert assert_unmodeled_fault_share_bounded(events_ok, max_share=0.10) == pytest.approx(
        1.0 / 11.0
    )

    events_at_limit = [applied] * 9 + [unmodeled_evt]
    with pytest.raises(AssertionError):
        assert_unmodeled_fault_share_bounded(events_at_limit, max_share=0.10)


def test_inject_lake_counterexample_anchors_uncapped_and_immutable(tmp_path: Path) -> None:
    """Verify lake counterexample injection grows without fixed 2-case cap."""
    lake_dir = tmp_path / "lake"
    for cid in (10, 11, 12, 13):
        append_counterexample_boundary(
            lake_dir,
            {
                "operator": "attention_residual",
                "regime_tag": "unaligned_d",
                "case_num": cid,
                "ref_perf_us": 50.0 + cid,
                "t_hw_us": 10.0 + cid,
            },
        )

    initial = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
    )
    regimes = (
        RegimeTargetSpec("small_decode", ("_b0",), {}),
        RegimeTargetSpec("unaligned_d", ("_b1",), {}),
    )
    injected = inject_lake_counterexample_anchors(
        initial,
        operator_name="attention_residual",
        regime_specs=regimes,
        experience_lake_dir=lake_dir,
    )
    assert len(initial) == 1
    assert [a.case_num for a in injected] == [2, 10, 11, 12, 13]
