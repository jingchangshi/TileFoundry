"""Conservation-law latency algebra, paired-MAD noise floors, and bandit UCB helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

MAD_TO_SIGMA: float = 1.482602218505602


@dataclass(frozen=True)
class AnchorConservationPoint:
    """Single anchor measurement decomposing latency into floor, roofline, and schedule."""

    case_num: int
    regime_tag: str
    ref_us: float
    inc_us: float
    cand_us: float
    t_roof_us: float
    paired_rel_mad: float
    accuracy_passed: bool

    def sigma_paired_us(self) -> float:
        """Convert paired relative MAD into a 1-sigma uncertainty floor in microseconds."""
        inc_val = float(self.inc_us) if math.isfinite(self.inc_us) else 0.0
        mad_val = float(self.paired_rel_mad) if math.isfinite(self.paired_rel_mad) else 0.0
        base = max(inc_val, 1e-6)
        return max(1e-3 * base, MAD_TO_SIGMA * max(0.0, mad_val) * base)

    def recoverable_sched_us(self, dispatch_floor_us: float) -> float:
        """Return non-negative recoverable scheduling latency above floor and roofline."""
        cand_val = float(self.cand_us) if math.isfinite(self.cand_us) else 0.0
        floor_val = float(dispatch_floor_us) if math.isfinite(dispatch_floor_us) else 0.0
        roof_val = float(self.t_roof_us) if math.isfinite(self.t_roof_us) else 0.0
        return max(0.0, cand_val - max(0.0, floor_val) - max(0.0, roof_val))


def calibrate_dispatch_floor_us(
    points: Sequence[AnchorConservationPoint],
    *,
    null_launch_us: float | None = None,
) -> float:
    """Calibrate dispatch floor from null-kernel launch or minimum anchor excess."""
    if null_launch_us is not None:
        val = float(null_launch_us)
        if math.isfinite(val) and val >= 0.0:
            return val
    residuals = [
        max(0.0, float(p.cand_us) - max(0.0, float(p.t_roof_us)))
        for p in points
        if (
            p.accuracy_passed
            and math.isfinite(p.cand_us)
            and p.cand_us > 0.0
            and math.isfinite(p.t_roof_us)
        )
    ]
    return min(residuals) if residuals else 0.0


def geometric_speedup(points: Sequence[AnchorConservationPoint]) -> float:
    """Compute geometric mean speedup exp(mean(ln(ref_us / cand_us))) across valid points."""
    log_ratios = [
        math.log(float(p.ref_us) / float(p.cand_us))
        for p in points
        if (
            math.isfinite(p.ref_us)
            and math.isfinite(p.cand_us)
            and p.ref_us > 0.0
            and p.cand_us > 0.0
        )
    ]
    if not log_ratios:
        return 0.0
    return math.exp(sum(log_ratios) / len(log_ratios))


def total_recoverable_sched_us(
    points: Sequence[AnchorConservationPoint],
    dispatch_floor_us: float,
) -> float:
    """Sum recoverable scheduling latency across all conservation points."""
    return sum(p.recoverable_sched_us(dispatch_floor_us) for p in points)


def check_guardrail_non_regression(
    point: AnchorConservationPoint,
    *,
    sigma_mult: float = 3.0,
) -> bool:
    """Return True if point passes accuracy and stays within sigma_mult * sigma of incumbent."""
    return bool(
        point.accuracy_passed
        and math.isfinite(point.cand_us)
        and math.isfinite(point.inc_us)
        and point.cand_us > 0.0
        and point.cand_us <= point.inc_us + float(sigma_mult) * point.sigma_paired_us()
    )


def check_family_and_sibling_non_regression(
    active_points: Sequence[AnchorConservationPoint],
    sibling_points: Sequence[AnchorConservationPoint],
    *,
    sigma_mult: float = 3.0,
) -> tuple[bool, list[int]]:
    """Verify 3-sigma non-regression across active and sibling anchor points."""
    failed: list[int] = []
    seen: set[int] = set()
    for point in (*active_points, *sibling_points):
        if not check_guardrail_non_regression(point, sigma_mult=sigma_mult):
            cnum = int(point.case_num)
            if cnum not in seen:
                seen.add(cnum)
                failed.append(cnum)
    return (len(failed) == 0, failed)


def check_regime_net_improvement(
    points: Sequence[AnchorConservationPoint],
    *,
    sigma_mult: float = 2.0,
) -> tuple[bool, float, float]:
    """Check whether aggregate net latency gain exceeds sigma_mult * sqrt(sum(sigma^2))."""
    if not points:
        return (False, 0.0, 0.0)
    all_valid = all(
        p.accuracy_passed
        and math.isfinite(p.cand_us)
        and p.cand_us > 0.0
        and math.isfinite(p.inc_us)
        for p in points
    )
    net_gain_us = sum(
        float(p.inc_us) - float(p.cand_us)
        for p in points
        if math.isfinite(p.inc_us) and math.isfinite(p.cand_us)
    )
    threshold_us = max(0.0, float(sigma_mult)) * math.sqrt(
        sum(p.sigma_paired_us() ** 2 for p in points)
    )
    return (all_valid and net_gain_us > threshold_us, net_gain_us, threshold_us)


def is_regime_saturated(
    points: Sequence[AnchorConservationPoint],
    dispatch_floor_us: float,
    *,
    sigma_mult: float = 2.0,
) -> bool:
    """Return True when all regime points lie within sigma_mult * sigma of floor + roofline."""
    if not points:
        return False
    return all(
        p.recoverable_sched_us(dispatch_floor_us) <= float(sigma_mult) * p.sigma_paired_us()
        for p in points
    )


def regime_bandit_ucb_us(
    points: Sequence[AnchorConservationPoint],
    dispatch_floor_us: float,
    n_trials: int,
) -> float:
    """Score regime priority from recoverable scheduling headroom decayed by trial count."""
    total_sched = total_recoverable_sched_us(points, dispatch_floor_us)
    return total_sched / math.sqrt(1.0 + max(0, int(n_trials)))


def compute_adaptive_poison_pill_timeout_sec(inc_latencies_us: Sequence[float]) -> float:
    """Compute bounded poison-pill worker timeout scaled from incumbent anchor latencies."""
    total_sec = 0.0
    for raw_u in inc_latencies_us:
        val = float(raw_u)
        safe_u = val if math.isfinite(val) and val > 0.0 else 0.0
        total_sec += max(safe_u * 1e-6, 1e-3)
    return min(60.0, max(10.0, 25.0 * total_sec))


__all__ = [
    "AnchorConservationPoint",
    "MAD_TO_SIGMA",
    "calibrate_dispatch_floor_us",
    "check_family_and_sibling_non_regression",
    "check_guardrail_non_regression",
    "check_regime_net_improvement",
    "compute_adaptive_poison_pill_timeout_sec",
    "geometric_speedup",
    "is_regime_saturated",
    "regime_bandit_ucb_us",
    "total_recoverable_sched_us",
]
