"""Conservation-law latency algebra, paired-MAD noise floors, and static-dynamic residuals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tilefoundry_ascend.agent.kb.schema import SymptomFamily, compute_symptom_simplex
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.observation_space import (
    CompilerSummaryLike,
    ObservationVector5D,
    RooflineSummaryLike,
    compute_transfer_inflation_ratio,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

MAD_TO_SIGMA: float = 1.482602218505602


def _read_float(payload: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    """Read the first present numeric telemetry key, otherwise return the neutral default."""
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _read_int(payload: Mapping[str, object], *keys: str, default: int = 0) -> int:
    """Read the first present integral telemetry key, otherwise return the neutral default."""
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return default


def _read_bool(payload: Mapping[str, object], key: str, default: bool = False) -> bool:
    """Read a boolean telemetry flag with Python truthiness, otherwise return the default."""
    if key not in payload:
        return default
    return bool(payload[key])


def compute_paired_sigma_us(
    base_us: float,
    paired_rel_mad: float,
    cand_burst_cv: float,
    ref_burst_cv: float,
    burst_count: int,
) -> float:
    """Compute paired latency uncertainty from paired MAD and batch standard error."""
    safe_base = max(0.0, float(base_us)) if math.isfinite(base_us) else 0.0
    safe_mad = max(0.0, float(paired_rel_mad)) if math.isfinite(paired_rel_mad) else 0.0
    safe_cand_cv = max(0.0, float(cand_burst_cv)) if math.isfinite(cand_burst_cv) else 0.0
    safe_ref_cv = max(0.0, float(ref_burst_cv)) if math.isfinite(ref_burst_cv) else 0.0
    n_eff = max(1, int(burst_count))
    sigma_from_diff = safe_base * safe_mad
    sigma_from_batch_se = safe_base * max(safe_cand_cv, safe_ref_cv) / math.sqrt(n_eff)
    return max(sigma_from_diff, sigma_from_batch_se)


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
    cand_burst_cv: float = 0.0
    ref_burst_cv: float = 0.0
    burst_count: int = 1

    def sigma_paired_us(self) -> float:
        """Convert paired relative MAD and burst CVs into 1-sigma uncertainty in microseconds."""
        inc_val = float(self.inc_us) if math.isfinite(self.inc_us) else 0.0
        mad_val = float(self.paired_rel_mad) if math.isfinite(self.paired_rel_mad) else 0.0
        scaled_mad = MAD_TO_SIGMA * max(0.0, mad_val)
        return compute_paired_sigma_us(
            base_us=max(0.0, inc_val),
            paired_rel_mad=scaled_mad,
            cand_burst_cv=self.cand_burst_cv,
            ref_burst_cv=self.ref_burst_cv,
            burst_count=self.burst_count,
        )

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
    """Bound dispatch floor using fastest observed real kernel and null-launch overhead."""
    upper_bounds: list[float] = []
    for p in points:
        if not (
            p.accuracy_passed
            and math.isfinite(p.cand_us)
            and p.cand_us > 0.0
            and math.isfinite(p.t_roof_us)
        ):
            continue
        cand_val = float(p.cand_us)
        ref_val = float(p.ref_us) if math.isfinite(p.ref_us) else 0.0
        fastest_real_us = min(cand_val, ref_val) if ref_val > 0.0 else cand_val
        upper_bounds.append(max(0.0, fastest_real_us - max(0.0, float(p.t_roof_us))))
    observed_cap = min(upper_bounds) if upper_bounds else 0.0
    if null_launch_us is not None:
        val = float(null_launch_us)
        if math.isfinite(val) and val > 0.0:
            return min(val, observed_cap) if upper_bounds else val
    return observed_cap


def calibrate_dispatch_floor_from_series(
    cand_us_by_case: Sequence[float],
    ref_us_by_case: Sequence[float],
    roofline_us_by_case: Sequence[float],
    null_launch_us: float | None = None,
) -> float:
    """Bound fixed launch overhead from aligned candidate, reference, and roofline series."""
    upper_bounds: list[float] = []
    for cand_us, ref_us, roof_us in zip(
        cand_us_by_case, ref_us_by_case, roofline_us_by_case
    ):
        cand_val = float(cand_us)
        ref_val = float(ref_us)
        roof_val = float(roof_us)
        if not (math.isfinite(cand_val) and math.isfinite(roof_val)):
            continue
        if math.isfinite(ref_val) and ref_val > 0.0 and cand_val > 0.0:
            fastest_real_us = min(cand_val, ref_val)
        elif cand_val > 0.0:
            fastest_real_us = cand_val
        elif math.isfinite(ref_val) and ref_val > 0.0:
            fastest_real_us = ref_val
        else:
            continue
        upper_bounds.append(max(0.0, fastest_real_us - max(0.0, roof_val)))
    observed_cap = min(upper_bounds) if upper_bounds else 0.0
    if null_launch_us is not None:
        val = float(null_launch_us)
        if math.isfinite(val) and val > 0.0:
            return min(val, observed_cap) if upper_bounds else val
    return observed_cap


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


@dataclass(frozen=True)
class StaticDynamicResidual(JsonRecordMixin):
    """Gap between predicted ledger utilization and measured NPU msprof counters."""

    measured_latency_us: float
    predicted_bottleneck_util: float
    measured_aic_mac_ratio: float
    measured_aiv_vec_ratio: float
    measured_mte2_wait_ratio: float
    measured_sync_stall_ratio: float
    unexplained_gap_ratio: float
    pseudo_double_buffer_serialized: bool = False
    spill_bytes: int = 0
    bank_conflict_ratio: float = 0.0
    l2_hit_rate: float = 1.0
    vf_fusion_breaks: int = 0
    alias_serialized_buffers: int = 0
    dual_issue_ipc_collapse: float = 0.0
    dcache_dcci_flushes: int = 0
    has_local_membar_in_vf_call: bool = False
    vloopv2_hit: bool = True
    scalar_compute_loop_fallback: float = 0.0
    dcache_bypass_narrow_store: float = 0.0
    observation: ObservationVector5D | None = None
    transfer_inflation_ratio: float = 0.0
    effective_compute_eta: float = 0.0
    stack_bytes: int = 0

    @property
    def has_l1_compiler_summary(self) -> bool:
        """Report whether L1 penalties came from a real bisheng compiler summary."""
        return self.observation is not None and self.observation.has_l1_compiler_summary

    @property
    def measured_l2_fields(self) -> tuple[str, ...]:
        """Report which L2 fields were actually observed in msprof telemetry."""
        return self.observation.measured_l2_fields if self.observation is not None else ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StaticDynamicResidual:
        """Reconstruct StaticDynamicResidual from a serialized dictionary."""
        raw_observation = payload.get("observation")
        if raw_observation is None:
            observation = None
        elif isinstance(raw_observation, ObservationVector5D):
            observation = raw_observation
        elif isinstance(raw_observation, Mapping):
            observation = ObservationVector5D.from_dict(raw_observation)
        else:
            raise ValueError(
                f"residual field 'observation' must be a mapping, got "
                f"{type(raw_observation).__name__}"
            )
        default_inflation = (
            observation.transfer_inflation_ratio if observation is not None else 0.0
        )
        default_eff_eta = observation.effective_compute_eta if observation is not None else 0.0
        default_scalar_fallback = (
            float(getattr(observation.compute_issue, "l1_scalar_compute_loop_fallback", 0.0) or 0.0)
            if observation is not None
            else 0.0
        )
        default_narrow_store = (
            float(getattr(observation.offchip_dma, "l1_dcache_bypass_narrow_store", 0.0) or 0.0)
            if observation is not None
            else 0.0
        )
        default_stack = (
            int(getattr(observation.onchip_sram_reg, "l1_stack_bytes", 0) or 0)
            if observation is not None
            else 0
        )
        return cls(
            measured_latency_us=float(payload.get("measured_latency_us", 0.0)),
            predicted_bottleneck_util=float(payload.get("predicted_bottleneck_util", 0.0)),
            measured_aic_mac_ratio=float(payload.get("measured_aic_mac_ratio", 0.0)),
            measured_aiv_vec_ratio=float(payload.get("measured_aiv_vec_ratio", 0.0)),
            measured_mte2_wait_ratio=float(payload.get("measured_mte2_wait_ratio", 0.0)),
            measured_sync_stall_ratio=float(payload.get("measured_sync_stall_ratio", 0.0)),
            unexplained_gap_ratio=float(payload.get("unexplained_gap_ratio", 0.0)),
            pseudo_double_buffer_serialized=bool(
                payload.get("pseudo_double_buffer_serialized", False)
            ),
            spill_bytes=int(payload.get("spill_bytes", 0)),
            stack_bytes=max(0, _read_int(payload, "stack_bytes", default=default_stack)),
            bank_conflict_ratio=float(payload.get("bank_conflict_ratio", 0.0)),
            l2_hit_rate=float(payload.get("l2_hit_rate", 1.0)),
            vf_fusion_breaks=max(0, int(payload.get("vf_fusion_breaks", 0) or 0)),
            alias_serialized_buffers=max(0, int(payload.get("alias_serialized_buffers", 0) or 0)),
            dual_issue_ipc_collapse=max(
                0.0, min(1.0, float(payload.get("dual_issue_ipc_collapse", 0.0) or 0.0))
            ),
            dcache_dcci_flushes=max(0, int(payload.get("dcache_dcci_flushes", 0) or 0)),
            has_local_membar_in_vf_call=bool(payload.get("has_local_membar_in_vf_call", False)),
            vloopv2_hit=bool(payload.get("vloopv2_hit", True)),
            scalar_compute_loop_fallback=max(
                0.0,
                _read_float(
                    payload,
                    "scalar_compute_loop_fallback",
                    default=default_scalar_fallback,
                ),
            ),
            dcache_bypass_narrow_store=max(
                0.0,
                _read_float(
                    payload,
                    "dcache_bypass_narrow_store",
                    default=default_narrow_store,
                ),
            ),
            observation=observation,
            transfer_inflation_ratio=max(
                0.0, _read_float(payload, "transfer_inflation_ratio", default=default_inflation)
            ),
            effective_compute_eta=max(
                0.0, _read_float(payload, "effective_compute_eta", default=default_eff_eta)
            ),
        )


def compute_static_dynamic_residual(
    ledger: ObjectivePhysicalLedger,
    telemetry: Mapping[str, object],
    *,
    compiler_summary: CompilerSummaryLike | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
    topology_axes: object | None = None,
) -> StaticDynamicResidual:
    """Compute StaticDynamicResidual from ledger, msprof telemetry, compiler, and Roofline."""
    measured_us = _read_float(telemetry, "measured_latency_us", "dur_us")
    aic_mac = _read_float(telemetry, "aic_mac_ratio", "aic_mad")
    aiv_vec = _read_float(telemetry, "aiv_vec_ratio", "aiv_vec")
    mte2_wait = _read_float(telemetry, "mte2_wait_ratio", "aiv_mte2", "aic_mte2")
    sync_stall = _read_float(telemetry, "sync_stall_ratio")

    positive_cycles = [v for v in ledger.pipe_cycles.values() if v > 0]
    if positive_cycles and ledger.steady_state_ii is not None and ledger.steady_state_ii > 0:
        dom = max(positive_cycles)
        predicted_util = min(1.0, float(dom) / float(max(dom, ledger.steady_state_ii)))
    elif positive_cycles:
        dom = max(positive_cycles)
        non_dom = sum(positive_cycles) - dom
        predicted_util = min(1.0, float(dom) / float(max(1.0, dom + 0.25 * non_dom)))
    else:
        predicted_util = 0.0

    observation = ObservationVector5D.from_tiers(
        ledger,
        compiler_summary=compiler_summary,
        telemetry=telemetry,
        roofline=roofline,
        topology_axes=topology_axes,
    )
    simplex = dict(compute_symptom_simplex(observation, topology_axes=topology_axes))
    unexplained = float(simplex.get(SymptomFamily.UNMODELED_SILICON_RESIDUAL, 0.0))

    steady_ii = ledger.steady_state_ii
    ii_looks_pipelined = steady_ii is None or steady_ii <= 2
    pseudo_serialized = bool(
        ledger.uses_dynamic_offset_pipelining
        and ii_looks_pipelined
        and (mte2_wait >= 0.32 or unexplained >= 0.25 or sync_stall >= 0.20)
    )
    comp_spill = compiler_summary.spill_bytes if compiler_summary is not None else 0
    spill_bytes = max(0, comp_spill, _read_int(telemetry, "spill_bytes"))
    comp_stack = compiler_summary.stack_bytes if compiler_summary is not None else 0
    stack_bytes = max(0, comp_stack, _read_int(telemetry, "stack_bytes"))
    bank_conflict_ratio = max(
        0.0,
        min(
            1.0,
            _read_float(
                telemetry,
                "bank_conflict_ratio",
                "ub_bank_conflict_ratio",
                "resource_conflict_ratio",
            ),
        ),
    )
    l2_hit_rate = max(
        0.0,
        min(
            1.0,
            _read_float(telemetry, "l2_hit_rate", "l2_cache_hit_rate", default=1.0),
        ),
    )

    vf_fusion_breaks = max(
        0,
        _read_int(telemetry, "vf_fusion_breaks"),
        compiler_summary.vf_fusion_breaks if compiler_summary is not None else 0,
    )
    alias_serialized_buffers = max(
        0,
        _read_int(telemetry, "alias_serialized_buffers"),
        compiler_summary.alias_serialized_buffers if compiler_summary is not None else 0,
    )
    dcache_dcci_flushes = max(
        0,
        _read_int(telemetry, "dcache_dcci_flushes"),
        compiler_summary.dcache_dcci_flushes if compiler_summary is not None else 0,
    )
    dual_issue_ipc_collapse = max(
        0.0,
        min(
            1.0,
            max(
                _read_float(telemetry, "dual_issue_ipc_collapse"),
                (
                    compiler_summary.dual_issue_ipc_collapse
                    if compiler_summary is not None
                    else 0.0
                ),
            ),
        ),
    )
    has_local_membar_in_vf_call = _read_bool(
        telemetry, "has_local_membar_in_vf_call"
    ) or (
        compiler_summary.has_local_membar_in_vf_call
        if compiler_summary is not None
        else False
    )
    if "vloopv2_hit" in telemetry:
        vloopv2_hit = _read_bool(telemetry, "vloopv2_hit")
    elif compiler_summary is not None:
        vloopv2_hit = not (
            compiler_summary.software_loops > 0 and compiler_summary.vloopv2_loops == 0
        )
    else:
        vloopv2_hit = True

    raw_extra = getattr(compiler_summary, "extra_signals", None)
    extra_map = raw_extra if isinstance(raw_extra, Mapping) else {}
    scalar_fallback = max(
        0.0,
        _read_float(telemetry, "scalar_compute_loop_fallback"),
        _read_float(extra_map, "scalar_compute_loop_fallback"),
        float(ledger.ub_scalar_compute_loops),
    )
    narrow_store = max(
        0.0,
        _read_float(telemetry, "dcache_bypass_narrow_store"),
        _read_float(extra_map, "dcache_bypass_narrow_store"),
    )

    transfer_inflation_ratio = compute_transfer_inflation_ratio(
        ledger,
        telemetry=telemetry,
        roofline=roofline,
    )
    return StaticDynamicResidual(
        measured_latency_us=measured_us,
        predicted_bottleneck_util=round(predicted_util, 6),
        measured_aic_mac_ratio=aic_mac,
        measured_aiv_vec_ratio=aiv_vec,
        measured_mte2_wait_ratio=mte2_wait,
        measured_sync_stall_ratio=sync_stall,
        unexplained_gap_ratio=round(unexplained, 6),
        pseudo_double_buffer_serialized=pseudo_serialized,
        spill_bytes=spill_bytes,
        bank_conflict_ratio=round(bank_conflict_ratio, 6),
        l2_hit_rate=round(l2_hit_rate, 6),
        vf_fusion_breaks=vf_fusion_breaks,
        alias_serialized_buffers=alias_serialized_buffers,
        dual_issue_ipc_collapse=round(dual_issue_ipc_collapse, 6),
        dcache_dcci_flushes=dcache_dcci_flushes,
        has_local_membar_in_vf_call=has_local_membar_in_vf_call,
        vloopv2_hit=vloopv2_hit,
        scalar_compute_loop_fallback=round(scalar_fallback, 6),
        dcache_bypass_narrow_store=round(narrow_store, 6),
        observation=observation,
        transfer_inflation_ratio=transfer_inflation_ratio,
        effective_compute_eta=observation.effective_compute_eta,
        stack_bytes=stack_bytes,
    )


__all__ = [
    "AnchorConservationPoint",
    "MAD_TO_SIGMA",
    "StaticDynamicResidual",
    "calibrate_dispatch_floor_from_series",
    "calibrate_dispatch_floor_us",
    "check_family_and_sibling_non_regression",
    "check_guardrail_non_regression",
    "check_regime_net_improvement",
    "compute_adaptive_poison_pill_timeout_sec",
    "compute_paired_sigma_us",
    "compute_static_dynamic_residual",
    "geometric_speedup",
    "is_regime_saturated",
    "regime_bandit_ucb_us",
    "total_recoverable_sched_us",
]
