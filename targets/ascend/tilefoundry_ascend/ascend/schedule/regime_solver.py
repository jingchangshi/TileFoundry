"""Hybrid hierarchical piecewise regime solver for Ascend 950 (dav-3510)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tilefoundry.vendor.tilelang_z3 import Problem
from tilefoundry_ascend.ascend.json_record import json_clone
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    dtype_bytes,
    solve_pipeline_with_monotone_backoff,
)
from tilefoundry_ascend.ascend.sil import PolicyEnvelope, SchedulePolicy
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    TopologyAxes,
    WaveBand,
    compute_critical_cube_tile_m,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

SOLVER_MODE_ALGEBRAIC = "algebraic_piecewise"
SOLVER_MODE_Z3 = "z3_smt"
_LEGACY_ALGEBRAIC_MODES = frozenset({"analytical_piecewise"})
_VALID_CALIPER_ACTIONS = frozenset({"none", "ub_halving", "l1_stage_backoff"})
_MAX_CALIPER_STEPS = 3
_CV_DMA_PATHS = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->L1", "UB->GM"]
_L0C_UB_DMA_PATHS = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->GM"]
_FIXPIPE_DMA_PATHS = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->GM"]
_CUBE_DTYPE_ALIASES = {
    "f4e2m1": "f4e2m1", "float4_e2m1": "f4e2m1", "f4e1m2": "f4e1m2",
    "float4_e1m2": "f4e1m2", "float8_e4m3fn": "fp8e4m3", "float8_e5m2": "fp8e5m2", "uint8": "u8",
}


def normalize_solver_mode(mode: str) -> str:
    """Normalize legacy solver mode names onto canonical solver identifiers."""
    n = str(mode).strip()
    return SOLVER_MODE_ALGEBRAIC if n in _LEGACY_ALGEBRAIC_MODES else n


@runtime_checkable
class RooflineSummaryLike(Protocol):
    """Structural protocol satisfied by HirRooflineSummary without import cycles."""

    anchor_dims: dict[str, int]
    min_gm_bytes: int
    dominant_bound: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the Roofline summary into a JSON-compatible dictionary."""
        ...


def _coerce_roofline_mapping(raw: RooflineSummaryLike | Mapping[str, object] | None) -> dict[str, object]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, RooflineSummaryLike):
        return raw.to_dict()
    raise TypeError(f"roofline must be HirRooflineSummary, Mapping, or None, got {type(raw).__name__}")


def _coerce_topology_axes(raw: object) -> TopologyAxes | None:
    if raw is None or isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping):
        return TopologyAxes.from_dict({str(k): str(v) for k, v in raw.items()})
    raise TypeError(f"topology_axes must be TopologyAxes or Mapping, got {type(raw).__name__}")


def _synthesize_axes_from_flags(
    *, uses_cube: bool, has_persistent_state: bool, has_dead_l1_slice: bool,
    has_empty_group: bool, has_indirect_gather: bool, has_recurrence: bool, quant_mode: str,
) -> TopologyAxes:
    """Construct orthogonal TopologyAxes purely from structural flags (zero op_name reliance)."""
    has_quant = bool(quant_mode and quant_mode not in ("none", "false"))
    if not uses_cube:
        engine = "aiv_only"
    elif any((has_persistent_state, has_empty_group, has_indirect_gather, has_recurrence, has_dead_l1_slice, has_quant)):
        engine = "coupled_1c2v"
    else:
        engine = "aic_only"

    if has_persistent_state:
        ring = "persistent_onchip_state"
    elif uses_cube and (has_recurrence or has_dead_l1_slice):
        ring = "bidirectional_cv_ring"
    elif uses_cube and has_empty_group:
        ring = "l0c_ub_epilogue"
    elif uses_cube:
        ring = "gm_l1_l0_fixpipe"
    else:
        ring = "ub_random_atomic" if has_indirect_gather else "gm_ub_stream"

    reg = "ragged_group" if has_empty_group else ("irregular_simt" if has_indirect_gather else "affine_simd")
    return TopologyAxes(compute_engine=engine, memory_ring=ring, iteration_regularity=reg)


def _as_int_dims(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    dims: dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, str)):
            raise ValueError(f"regime dim {k!r} must be an int or decimal string")
        dims[str(k)] = int(v)
    return dims


def _as_case_ratios(raw: object) -> tuple[CasePhysicsRatios, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("case_ratios must be a sequence of CasePhysicsRatios or mappings")
    out: list[CasePhysicsRatios] = []
    for item in raw:
        if isinstance(item, CasePhysicsRatios):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(CasePhysicsRatios.from_dict({str(k): v for k, v in item.items()}))
        else:
            raise ValueError("case_ratios entries must be CasePhysicsRatios or mappings")
    return tuple(out)


def _select_representative_ratio(ratios: tuple[CasePhysicsRatios, ...], rep_id: int | None) -> CasePhysicsRatios | None:
    if not ratios:
        return None
    return ratios[len(ratios) // 2] if rep_id is None else next((r for r in ratios if r.case_id == rep_id), None)


def _regime_attr(regime: Mapping[str, object] | object, key: str, default: object = None) -> object:
    return regime.get(key, default) if isinstance(regime, Mapping) else getattr(regime, key, default)


def _extract_regime_fields(
    regime: Mapping[str, object] | object,
) -> tuple[str, dict[str, int], bool, bool, dict[str, object], TopologyAxes | None, CasePhysicsRatios | None]:
    tag = str(_regime_attr(regime, "regime_tag", "default"))
    raw_dims = _regime_attr(regime, "representative_dims")
    if raw_dims is None and isinstance(regime, Mapping):
        raw_dims = regime.get("dims")
    raw_rf = _regime_attr(regime, "roofline")
    rf = _coerce_roofline_mapping(raw_rf) if isinstance(raw_rf, (Mapping, RooflineSummaryLike)) else {}
    raw_cid = _regime_attr(regime, "representative_case_id")
    if raw_cid is not None and (isinstance(raw_cid, bool) or not isinstance(raw_cid, int)):
        if not isinstance(regime, Mapping):
            raise ValueError("representative_case_id must be an int")
        cid = None
    else:
        cid = int(raw_cid) if raw_cid is not None else None
    return (
        tag, _as_int_dims(raw_dims), bool(_regime_attr(regime, "has_unaligned_tail", False)),
        bool(_regime_attr(regime, "has_empty_group", False)), rf,
        _coerce_topology_axes(_regime_attr(regime, "topology_axes")),
        _select_representative_ratio(_as_case_ratios(_regime_attr(regime, "case_ratios")), cid),
    )


def _mapping_float(payload: Mapping[str, object], key: str, default: float = 0.0) -> float:
    raw = payload.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"roofline field {key!r} must be numeric, got {type(raw).__name__}")
    return float(raw)


def _is_cube_bound(
    roofline: Mapping[str, object] | None, dims: Mapping[str, int],
    *, uses_cube: bool | None = None, axes: TopologyAxes | None = None,
) -> bool:
    if roofline:
        if any(k in str(roofline.get("dominant_bound", "")).lower() for k in ("cube", "compute", "aic")):
            return True
        cube_us = _mapping_float(roofline, "cube_floor_us")
        if cube_us > 0.0 and cube_us >= max(_mapping_float(roofline, "memory_floor_us"), _mapping_float(roofline, "vec_floor_us")):
            return True
        if _mapping_float(roofline, "arithmetic_intensity") >= 128.0:
            return True
    m_val = dims.get("M", dims.get("S_q", dims.get("S", dims.get("m", 0))))
    cube_ring = axes is not None and axes.memory_ring in ("gm_l1_l0_fixpipe", "l0c_ub_epilogue", "bidirectional_cv_ring")
    if (uses_cube is True or (axes is not None and axes.uses_cube)) and (uses_cube is True or cube_ring):
        return m_val == 0 or m_val >= 64
    return False


def _int_or_none(raw: object) -> int | None:
    return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None


@dataclass(frozen=True)
class RegimePlanResult:
    """Solved schedule policy and solver telemetry for one concrete shape regime."""

    regime_tag: str
    op_name: str
    topology_axes: TopologyAxes
    solver_mode: str
    policy: dict[str, object]
    representative_dims: dict[str, int] = field(default_factory=dict)
    ii: int | None = None
    buffer_versions: dict[str, int] = field(default_factory=dict)
    task_stages: dict[str, int] = field(default_factory=dict)
    fallback_used: bool = False
    fallback_reason: str = ""
    persistent_state_space: str = "none"
    persistent_state_bytes: int = 0
    notes: tuple[str, ...] = ()
    caliper_steps: int = 0
    caliper_action: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "solver_mode", normalize_solver_mode(self.solver_mode))
        steps = int(self.caliper_steps)
        if steps < 0:
            raise ValueError(f"caliper_steps must be non-negative, got {steps}")
        object.__setattr__(self, "caliper_steps", steps)
        action = str(self.caliper_action)
        if action not in _VALID_CALIPER_ACTIONS:
            raise ValueError(f"caliper_action must be one of {sorted(_VALID_CALIPER_ACTIONS)}, got {action!r}")
        object.__setattr__(self, "caliper_action", action)

    def to_dict(self) -> dict[str, object]:
        """Return a SIL SchedulePolicy-compatible dictionary."""
        return dict(self.policy)

    def to_summary_dict(self) -> dict[str, object]:
        """Return full regime solve metadata alongside the SIL policy dict."""
        return {
            "regime_tag": self.regime_tag, "op_name": self.op_name,
            "topology_axes": self.topology_axes.to_dict(), "solver_mode": self.solver_mode,
            "policy": dict(self.policy), "representative_dims": dict(self.representative_dims),
            "ii": self.ii, "buffer_versions": dict(self.buffer_versions),
            "task_stages": dict(self.task_stages), "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason, "persistent_state_space": self.persistent_state_space,
            "persistent_state_bytes": self.persistent_state_bytes, "notes": list(self.notes),
            "caliper_steps": self.caliper_steps, "caliper_action": self.caliper_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimePlanResult:
        """Reconstruct a RegimePlanResult from serialized summary metadata."""
        axes = _coerce_topology_axes(payload.get("topology_axes")) or resolve_canonical_topology_axes({})
        raw_bv, raw_ts = payload.get("buffer_versions"), payload.get("task_stages")
        raw_pol, raw_notes = payload.get("policy"), payload.get("notes")
        return cls(
            regime_tag=str(payload.get("regime_tag", "default")), op_name=str(payload.get("op_name", "")),
            topology_axes=axes,
            solver_mode=normalize_solver_mode(str(payload.get("solver_mode", SOLVER_MODE_ALGEBRAIC))),
            policy={str(k): v for k, v in raw_pol.items()} if isinstance(raw_pol, Mapping) else {},
            representative_dims=_as_int_dims(payload.get("representative_dims")),
            ii=_int_or_none(payload.get("ii")),
            buffer_versions={str(k): int(v) for k, v in raw_bv.items()} if isinstance(raw_bv, Mapping) else {},
            task_stages={str(k): int(v) for k, v in raw_ts.items()} if isinstance(raw_ts, Mapping) else {},
            fallback_used=bool(payload.get("fallback_used", False)),
            fallback_reason=str(payload.get("fallback_reason", "")),
            persistent_state_space=str(payload.get("persistent_state_space", "none")),
            persistent_state_bytes=_int_or_none(payload.get("persistent_state_bytes")) or 0,
            notes=tuple(str(i) for i in raw_notes) if isinstance(raw_notes, (list, tuple)) else (),
            caliper_steps=_int_or_none(payload.get("caliper_steps")) or 0,
            caliper_action=str(payload.get("caliper_action", "none")),
        )

    def to_sil_policy(self) -> SchedulePolicy:
        """Convert to ``ascend.sil.SchedulePolicy``."""
        return SchedulePolicy.from_dict(self.policy)


@dataclass(frozen=True)
class PiecewiseSchedulePlan:
    """Complete piecewise SIL policy envelope across all regimes of an operator."""

    op_name: str
    topology_axes: TopologyAxes
    policies: dict[str, dict[str, object]]
    results: dict[str, RegimePlanResult]
    selected_rules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a SIL PolicyEnvelope-compatible dictionary."""
        return {
            "op_name": self.op_name, "topology_axes": json_clone(self.topology_axes),
            "policies": {tag: dict(pol) for tag, pol in self.policies.items()},
            "selected_rules": list(self.selected_rules),
        }

    def to_sil_policy_envelope(self) -> PolicyEnvelope:
        """Convert to ``ascend.sil.PolicyEnvelope``."""
        return PolicyEnvelope.from_dict(self.to_dict())


def _align16(val: int, lo: int = 16, hi: int | None = None) -> int:
    aligned = max(lo, ((max(1, val) + 15) // 16) * 16)
    return min(hi, aligned) if hi is not None else aligned


def _resolve_solver_axes(
    *, explicit_axes: TopologyAxes | None, uses_cube: bool | None, has_persistent_state: bool | None,
    has_dead_l1_slice: bool | None, has_empty_group: bool, has_indirect_gather: bool,
    has_recurrence: bool, quant_mode: str | None, fallback_spec: object,
) -> TopologyAxes:
    if explicit_axes is not None:
        return explicit_axes
    if any(v is not None for v in (uses_cube, has_persistent_state, has_dead_l1_slice, quant_mode)) or has_indirect_gather:
        return _synthesize_axes_from_flags(
            uses_cube=bool(uses_cube) if uses_cube is not None else False,
            has_persistent_state=bool(has_persistent_state), has_dead_l1_slice=bool(has_dead_l1_slice),
            has_empty_group=has_empty_group, has_indirect_gather=has_indirect_gather,
            has_recurrence=has_recurrence, quant_mode=quant_mode or "none",
        )
    if isinstance(fallback_spec, Mapping):
        return resolve_canonical_topology_axes(fallback_spec)
    return _coerce_topology_axes(getattr(fallback_spec, "topology_axes", None)) or resolve_canonical_topology_axes({})


class RegimePiecewiseSolver:
    """Hierarchical solver dispatching static loops to Z3 and structural topologies analytically."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self.target = target or AscendTarget("ascend.950pr")
        facts = self.target.get_facts(AscendSpaceFacts)
        if not isinstance(facts, AscendSpaceFacts):
            raise RuntimeError(f"target {self.target.identity!r} did not provide AscendSpaceFacts")
        self.facts = facts

    def _is_ragged_or_recurrent(
        self, *, axes: TopologyAxes, has_empty_group: bool, has_indirect_gather: bool,
        has_recurrence: bool, has_persistent_state: bool | None, problem: Problem | None,
    ) -> bool:
        if problem is None or has_empty_group or has_indirect_gather or has_recurrence or has_persistent_state:
            return True
        return axes.iteration_regularity in ("irregular_simt", "ragged_group", "dynamic_opcode_stream") or axes.memory_ring in (
            "ub_random_atomic", "l0c_ub_epilogue", "bidirectional_cv_ring", "persistent_onchip_state",
        )

    def _base_policy(self, *, regime_tag: str, topology_axes: TopologyAxes, cube_dtype: str) -> dict[str, object]:
        return {
            "regime_tag": regime_tag, "topology_axes": topology_axes.to_dict(), "backend_tier": "tilelang_dsl",
            "tile_mnk": [128, 128, 64], "row_pack_factor": 1, "num_kv_splits": 1,
            "l1_stages": 2, "l1_half_bank_split": True, "l1_slot_aliasing": {}, "l0c_buffers": 1,
            "ub_256kb_unlock": False, "ub_estimated_bytes": 0, "l1_estimated_bytes": 0,
            "ub_bank_pad_datablocks": 0, "ub_rw_shift_datablocks": 0, "ub_row_stride_bytes": 0,
            "cv_datapath": "none", "sync_mode": "mode4_flag", "preload_depth": 2, "reverse_ack_period": 14,
            "ensure_empty_subblock_sync": True, "swizzle_window": 4, "tail_wave_split": True,
            "streamk_mode": "none", "l2_bypass_operands": [], "vector_mode": "simdvf",
            "direct_no_fusion_vf": False, "multi_row_unroll": 1, "reduce_accumulators": 1,
            "simt_threads": 512, "hf32_allowed": False, "fast_unshifted_exp": False,
            "w4_mode": "none", "dma_paths_used": [], "cube_operand_dtype": cube_dtype,
        }

    def _plan_analytical_regime(
        self, *, regime_tag: str, op_name: str, axes: TopologyAxes, explicit_axes: bool,
        dims: dict[str, int], roofline: RooflineSummaryLike | Mapping[str, object] | None,
        has_unaligned_tail: bool, has_empty_group: bool, input_dtypes: tuple[str, ...],
        has_nan_inf_cases: bool, uses_cube: bool | None = None, has_persistent_state: bool | None = None,
        has_dead_l1_slice: bool | None = None, has_indirect_gather: bool = False, quant_mode: str | None = None,
        representative_ratios: CasePhysicsRatios | None = None, solver_mode: str = SOLVER_MODE_ALGEBRAIC,
        fallback_used: bool = False, fallback_reason: str = "",
    ) -> RegimePlanResult:
        del explicit_axes
        roofline_map = _coerce_roofline_mapping(roofline)
        merged_dims = {**_as_int_dims(roofline_map.get("anchor_dims")), **dims}
        elem_dtype = input_dtypes[0] if input_dtypes else "f16"
        elem_bytes = max(1, math.ceil(dtype_bytes(elem_dtype)))
        eff_uses_cube = bool(uses_cube) if uses_cube is not None else axes.uses_cube
        eff_persistent = bool(has_persistent_state) if has_persistent_state is not None else (axes.memory_ring == "persistent_onchip_state")
        eff_dead_l1 = bool(has_dead_l1_slice)
        eff_indirect = bool(has_indirect_gather) or axes.iteration_regularity == "irregular_simt" or axes.memory_ring == "ub_random_atomic"
        eff_ragged = axes.iteration_regularity == "ragged_group" or bool(has_empty_group)
        raw_qmode = (quant_mode or "").strip().lower() or "none"

        if not eff_uses_cube:
            cube_dtype = ""
        elif raw_qmode in ("q4_0", "w4a16", "int4", "i4", "u4", "aiv_unpack_f16"):
            cube_dtype = "q4_0"
        else:
            dt_low = elem_dtype.strip().lower()
            cube_dtype = _CUBE_DTYPE_ALIASES.get(dt_low, dt_low or "f16")

        policy = self._base_policy(regime_tag=regime_tag, topology_axes=axes, cube_dtype=cube_dtype)
        cube_bound = _is_cube_bound(roofline_map, merged_dims, uses_cube=eff_uses_cube, axes=axes)
        notes: list[str] = []
        state_space, state_bytes = "none", 0
        physics_wave: WaveBand | None = None
        if representative_ratios is not None:
            sig = representative_ratios.phase_signature(axes)
            physics_wave = sig[4]
            notes.append(f"Physics phase {'/'.join(str(s) for s in sig)}")

        l1_half = self.facts.l1_bytes // 2
        ub_simdvf, ub_nom, ub_simtvf = self.facts.ub_simdvf_usable_bytes, self.facts.ub_nominal_bytes, self.facts.ub_simtvf_usable_bytes
        ub_coupled_cap = ub_simtvf if (axes.compute_engine == "coupled_1c2v" and axes.iteration_regularity == "irregular_simt") else ub_simdvf

        m_dim = dims.get("M", dims.get("S_q", dims.get("S", dims.get("B_total", 128))))
        n_dim = dims.get("N", dims.get("D_v", dims.get("D", 128)))
        k_dim = dims.get("K", dims.get("D_k", dims.get("D", 128)))
        crit_m = compute_critical_cube_tile_m(self.facts, elem_dtype)
        aligned_crit_m = _align16(crit_m)

        def _set_ub_unlock(est: int, *, pad: int = 1, shift: int = 0, stride: int = 0) -> None:
            unlock = ub_simdvf < est <= ub_nom
            policy.update({
                "ub_bank_pad_datablocks": pad, "ub_rw_shift_datablocks": shift,
                "ub_row_stride_bytes": stride, "ub_256kb_unlock": unlock,
                "ub_estimated_bytes": min(ub_nom if unlock else ub_simdvf, est),
            })

        if eff_uses_cube and (
            axes.memory_ring in ("bidirectional_cv_ring", "persistent_onchip_state", "l0c_ub_epilogue")
            or eff_persistent or eff_ragged
        ):
            if axes.memory_ring == "bidirectional_cv_ring":
                if cube_bound or m_dim >= crit_m:
                    if crit_m >= 82:
                        tm = aligned_crit_m if (m_dim < 128 or eff_indirect or eff_dead_l1) else 128
                    else:
                        tm = 128 if (m_dim >= 128 and not eff_indirect and not eff_dead_l1) else _align16(m_dim, aligned_crit_m, 96)
                    tm = max(aligned_crit_m, tm)
                else:
                    tm = _align16(m_dim, 16, 64)
                    policy["num_kv_splits"] = max(1, min(self.facts.num_aic_cores, dims.get("S_kv", 1024) // 128))
                    policy["l2_bypass_operands"] = ["K", "V"]
                tn, tk = (112 if tm == 96 else (112 if not (cube_bound or m_dim >= crit_m) else 128)), min(64, max(32, k_dim))
                stage_l1 = (tm * tk + tk * tn) * elem_bytes
                if stage_l1 > l1_half:
                    tk = max(32, (l1_half // max(1, (tm + tn) * elem_bytes)) // 16 * 16)
                    stage_l1 = (tm * tk + tk * tn) * elem_bytes
                l1_est, cv_dp, shift_db, dma_paths = stage_l1 * 2, "l0c_to_ub_split_m+ub_to_l1_nd2nz", 8, _CV_DMA_PATHS
                if eff_persistent:
                    state_bytes = max(32, min(128, k_dim)) * max(32, min(128, n_dim)) * 4
                    state_space = "L1" if state_bytes <= l1_half else "UB"
                    notes.append("Persistent on-chip FP32 state paired with bidirectional CV ring")
            elif eff_persistent:
                tm = (aligned_crit_m if crit_m >= 82 else _align16(m_dim, aligned_crit_m, 64)) if cube_bound else (64 if crit_m >= 82 else _align16(m_dim, aligned_crit_m, 64))
                if eff_ragged and cube_bound and tm < crit_m:
                    tm = aligned_crit_m
                tk = max(32, min(128, dims.get("D_k", dims.get("K", dims.get("D", 128)))))
                tn = max(32, min(128, dims.get("D_v", dims.get("N", dims.get("D", 128)))))
                state_bytes = tk * tn * 4
                l1_est = min(self.facts.l1_bytes, state_bytes + (tm * tk) * elem_bytes * 2)
                cv_dp, shift_db, dma_paths = "l0c_to_ub_split_m+ub_to_l1_nd2nz", 8, _CV_DMA_PATHS
                state_space = "L1" if state_bytes <= l1_half else "UB"
                notes.append("Persistent core ownership + FP32 state residency without GM roundtrips")
            else:
                tm = max(aligned_crit_m, 128 if m_dim >= 128 else (aligned_crit_m if crit_m >= 82 else _align16(m_dim, aligned_crit_m, 96))) if cube_bound else (64 if crit_m >= 82 else _align16(m_dim, aligned_crit_m, 64))
                tn, tk = (112 if tm == 96 else 128), 64
                l1_est = min(self.facts.l1_bytes, (tm * tk + tk * tn) * elem_bytes * 2)
                cv_dp, shift_db, dma_paths = "l0c_to_ub_split_m", 0, _L0C_UB_DMA_PATHS

            policy.update({
                "tile_mnk": [tm, tn, tk], "l1_stages": 2, "l1_half_bank_split": True,
                "l1_estimated_bytes": l1_est, "l0c_buffers": 2 if tm * tn * 4 * 2 <= self.facts.l0c_bytes else 1,
                "cv_datapath": cv_dp, "ub_bank_pad_datablocks": 1, "ub_rw_shift_datablocks": shift_db,
                "ub_row_stride_bytes": tn * 4 + 32,
                "ub_estimated_bytes": min(ub_coupled_cap, (tm // 2) * (tn * 4 + 32) * 2),
                "dma_paths_used": dma_paths,
            })
            if eff_indirect:
                policy["simt_threads"] = 512
                notes.append(f"AIV-ahead indirect/sparse index resolution + Cube M>={crit_m} ring")
            if eff_ragged:
                if has_empty_group:
                    notes.append("Empty group fast skip (m==0) with symmetric subblock sync parity")
                notes.append("Rolling cross-group core assignment + per-group BaseMBalance")
        elif eff_uses_cube:
            tm = max(aligned_crit_m, 128) if (m_dim >= 128 or m_dim == 0) else ((aligned_crit_m if crit_m >= 82 else _align16(m_dim, aligned_crit_m, 96)) if m_dim >= crit_m else (64 if crit_m >= 82 else aligned_crit_m))
            if cube_bound and tm < crit_m:
                tm = aligned_crit_m
            tn, tk = (112 if tm == 96 else 128), 64
            is_coupled = raw_qmode not in ("none", "false") or axes.compute_engine == "coupled_1c2v"
            policy.update({
                "tile_mnk": [tm, tn, tk], "l1_estimated_bytes": min(self.facts.l1_bytes, (tm * tk + tk * tn) * elem_bytes * 2),
                "l0c_buffers": 2 if tm * tn * 4 * 2 <= self.facts.l0c_bytes else 1,
                "cv_datapath": "l0c_to_ub_split_m" if is_coupled else "fixpipe_to_gm",
                "dma_paths_used": _L0C_UB_DMA_PATHS if is_coupled else _FIXPIPE_DMA_PATHS,
            })
            if raw_qmode in ("q4_0", "w4a16", "int4", "i4", "u4", "aiv_unpack_f16"):
                policy["w4_mode"] = "aiv_unpack_f16"
            elif raw_qmode in ("mxa8w4", "mxa8w4_bitshift_x64"):
                policy["w4_mode"] = "mxa8w4_bitshift_x64"
        if eff_uses_cube and eff_dead_l1:
            policy["l1_slot_aliasing"] = {"reused_stage": "dead_l1_slice"}
            notes.append("In-place L1 slot aliasing on dead L1 slice")

        if not eff_uses_cube and axes.iteration_regularity == "dynamic_opcode_stream":
            num_regs = max(4, min(64, dims.get("R", dims.get("num_regs", 16))))
            w_slice = _align16(math.ceil(max(64, dims.get("W", dims.get("N", dims.get("cols", 1024)))) / max(1, self.facts.num_aiv_cores)), 32, 2048)
            while w_slice > 32 and num_regs * (w_slice * 4 + 32) > ub_nom:
                w_slice = max(32, (w_slice // 2 // 16) * 16)
            row_bytes = w_slice * 4 + 32
            state_bytes, state_space = num_regs * row_bytes, "UB"
            policy.update({"tile_mnk": [num_regs, w_slice, 1], "l1_stages": 1, "l1_half_bank_split": False, "sync_mode": "none", "direct_no_fusion_vf": True, "dma_paths_used": ["GM->UB", "UB->GM"]})
            _set_ub_unlock(state_bytes, stride=row_bytes)
            notes.append("Horizontal W-slice striping across AIV cores; 100% UB-resident VM registers")
        elif not eff_uses_cube and eff_persistent:
            rows, cols = min(max(1, m_dim), 64), max(16, min(2048, dims.get("N", dims.get("D", dims.get("cols", 64)))))
            row_bytes = cols * 4 + 32
            while rows > 1 and rows * row_bytes * 2 > ub_nom:
                rows //= 2
            state_bytes, state_space = rows * row_bytes, "UB"
            policy.update({"tile_mnk": [rows, cols, 1], "l1_stages": 1, "l1_half_bank_split": False, "fast_unshifted_exp": not has_nan_inf_cases, "dma_paths_used": ["GM->UB", "UB->GM"]})
            _set_ub_unlock(state_bytes * 2, shift=8, stride=row_bytes)
            policy["ub_256kb_unlock"] = ub_simdvf < state_bytes <= ub_nom
            policy["ub_estimated_bytes"] = min(ub_nom if policy["ub_256kb_unlock"] else ub_simdvf, state_bytes * 2)
            notes.append("Persistent FP32 UB state residency across iterative loop")
        elif not eff_uses_cube and eff_indirect:
            tm, tn = min(64, max(16, m_dim)), min(128, max(32, n_dim))
            row_bytes = tn * elem_bytes + 32
            policy.update({
                "tile_mnk": [tm, tn, 1], "l1_stages": 1, "l1_half_bank_split": False, "vector_mode": "simtvf",
                "ub_bank_pad_datablocks": 1, "ub_rw_shift_datablocks": 8, "ub_row_stride_bytes": row_bytes,
                "ub_estimated_bytes": min(ub_simtvf, tm * row_bytes * 2), "dma_paths_used": ["GM->UB", "UB->GM"],
            })
            notes.append("SIMT __launch_bounds__(512) + warp-privatized +8-padded UB accumulators")
        elif not eff_uses_cube:
            raw_tail_n = merged_dims.get("N", merged_dims.get("D", merged_dims.get("cols", 1024)))
            rf_bound, rf_min_gm = str(roofline_map.get("dominant_bound", "")).strip().lower(), int(_mapping_float(roofline_map, "min_gm_bytes"))
            ub_half, work_elem_b = ub_simdvf // 2, max(4, elem_bytes)
            twopass_elems = max(16, ((ub_half // work_elem_b) // 16) * 16)
            raw_ub_tile = min((ub_half - 32) // max(1, 2 * work_elem_b), (ub_simdvf // 4 - 32) // max(1, elem_bytes), self.facts.max_mmad_mnk)
            ub_tile_elems = max(16, min(1 << (raw_ub_tile.bit_length() - 1), raw_ub_tile)) if raw_ub_tile >= 16 else 16
            is_mem_bound = rf_bound in ("memory", "memory_bandwidth")
            exceeds = raw_tail_n > twopass_elems or max(1, raw_tail_n) * elem_bytes > ub_half
            if is_mem_bound and (exceeds or (rf_min_gm > ub_simdvf and max(1, raw_tail_n) * work_elem_b > ub_half)):
                tn, pack = max(16, min(raw_tail_n, ub_tile_elems)), 1
                row_bytes = max(32, tn * elem_bytes)
                policy.update({"ub_tile_elems": tn, "twopass_switch_elems": twopass_elems, "requires_twopass_reduction": True})
                notes.append(f"Algebraic UB twopass threshold={twopass_elems} elems, ub_tile_elems={tn}")
            else:
                tn = max(1, min(2048, raw_tail_n))
                row_bytes = max(32, tn * elem_bytes)
                pack = min(16, max(1, (16384 + row_bytes - 1) // row_bytes))
                while pack > 1 and pack * (row_bytes + 32) * 4 > ub_simdvf:
                    pack //= 2
                if is_mem_bound:
                    policy.update({"ub_tile_elems": tn, "twopass_switch_elems": twopass_elems, "requires_twopass_reduction": False})
            policy.update({
                "tile_mnk": [pack, tn, 1], "row_pack_factor": pack, "multi_row_unroll": 2 if row_bytes <= 16384 else 1,
                "l1_stages": 1, "l1_half_bank_split": False, "fast_unshifted_exp": not has_nan_inf_cases, "dma_paths_used": ["GM->UB", "UB->GM"],
            })
            _set_ub_unlock(pack * (row_bytes + 32) * 4, stride=row_bytes + 32)

        if has_unaligned_tail:
            notes.append("Predicated / DataCopyPad unaligned tail handling enabled")
        if physics_wave is not None:
            policy["tail_wave_split"] = physics_wave != "balanced_waves" or has_unaligned_tail

        return RegimePlanResult(
            regime_tag=regime_tag, op_name=op_name, topology_axes=axes, solver_mode=solver_mode,
            policy=policy, representative_dims=dict(dims), fallback_used=fallback_used,
            fallback_reason=fallback_reason, persistent_state_space=state_space,
            persistent_state_bytes=state_bytes, notes=tuple(notes),
        )

    def solve_regime(
        self, regime: Mapping[str, object] | object, *, op_name: str = "",
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        problem: Problem | None = None, input_dtypes: tuple[str, ...] = ("f16",),
        has_nan_inf_cases: bool = False, has_indirect_gather: bool = False,
        has_recurrence: bool = False, uses_cube: bool | None = None,
        has_persistent_state: bool | None = None, has_dead_l1_slice: bool | None = None,
        quant_mode: str | None = None, unreserved_ubuf: bool = False,
    ) -> RegimePlanResult:
        """Solve one shape regime via Z3 projection or algebraic piecewise planning."""
        del unreserved_ubuf
        tag, dims, has_tail, has_empty, reg_rf, reg_axes, rep_ratios = _extract_regime_fields(regime)
        explicit_axes = _coerce_topology_axes(topology_axes) or reg_axes
        resolved_axes = _resolve_solver_axes(
            explicit_axes=explicit_axes, uses_cube=uses_cube, has_persistent_state=has_persistent_state,
            has_dead_l1_slice=has_dead_l1_slice, has_empty_group=has_empty,
            has_indirect_gather=has_indirect_gather, has_recurrence=has_recurrence,
            quant_mode=quant_mode, fallback_spec=regime,
        )
        resolved_rf = _coerce_roofline_mapping(roofline) if roofline is not None else reg_rf
        is_ragged = self._is_ragged_or_recurrent(
            axes=resolved_axes, has_empty_group=has_empty, has_indirect_gather=has_indirect_gather,
            has_recurrence=has_recurrence, has_persistent_state=has_persistent_state, problem=problem,
        )
        base_plan = self._plan_analytical_regime(
            regime_tag=tag, op_name=op_name, axes=resolved_axes, explicit_axes=(explicit_axes is not None),
            dims=dims, roofline=resolved_rf, has_unaligned_tail=has_tail, has_empty_group=has_empty,
            input_dtypes=input_dtypes, has_nan_inf_cases=has_nan_inf_cases, uses_cube=uses_cube,
            has_persistent_state=has_persistent_state, has_dead_l1_slice=has_dead_l1_slice,
            has_indirect_gather=has_indirect_gather, quant_mode=quant_mode,
            representative_ratios=rep_ratios, solver_mode=SOLVER_MODE_ALGEBRAIC if is_ragged else "z3_projected",
        )
        if is_ragged:
            return base_plan

        assert problem is not None
        z3_res = solve_pipeline_with_monotone_backoff(problem)
        policy = dict(base_plan.policy)
        bufs, vers = z3_res.problem.buffers, z3_res.buffer_versions
        ub_bytes = sum(b.slot_bytes * vers.get(b.name, 1) for b in bufs if b.space == "UB")
        l1_bytes = sum(b.slot_bytes * vers.get(b.name, 1) for b in bufs if b.space == "L1")
        l1_vers = [vers.get(b.name, 1) for b in bufs if b.space == "L1"]
        l0c_vers = [vers.get(b.name, 1) for b in bufs if b.space == "L0C"]
        if l1_vers:
            policy["l1_stages"], policy["l1_half_bank_split"] = max(l1_vers), max(l1_vers) >= 2
        if l0c_vers:
            policy["l0c_buffers"] = max(l0c_vers)
        if ub_bytes > 0:
            policy["ub_estimated_bytes"], policy["ub_256kb_unlock"] = ub_bytes, ub_bytes > self.facts.ub_simdvf_usable_bytes
        if l1_bytes > 0:
            policy["l1_estimated_bytes"] = l1_bytes

        return RegimePlanResult(
            regime_tag=tag, op_name=op_name, topology_axes=resolved_axes, solver_mode="z3_projected",
            policy=policy, representative_dims=dims, ii=z3_res.assignment.ii,
            buffer_versions=dict(vers), task_stages=dict(z3_res.task_stages),
            persistent_state_space=base_plan.persistent_state_space,
            persistent_state_bytes=base_plan.persistent_state_bytes,
            notes=("Z3 modulo schedule projected at regime representative dims",),
        )

    def solve_all_regimes(
        self, regimes: Sequence[Mapping[str, object] | object], *, op_name: str = "",
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        problem: Problem | None = None, input_dtypes: tuple[str, ...] = ("f16",),
        has_nan_inf_cases: bool = False, has_indirect_gather: bool = False,
        has_recurrence: bool = False, uses_cube: bool | None = None,
        has_persistent_state: bool | None = None, has_dead_l1_slice: bool | None = None,
        quant_mode: str | None = None,
    ) -> PiecewiseSchedulePlan:
        """Solve every regime slice and assemble a SIL-compatible PiecewiseSchedulePlan."""
        axes_obj = _coerce_topology_axes(topology_axes)
        resolved_axes = _resolve_solver_axes(
            explicit_axes=axes_obj, uses_cube=uses_cube, has_persistent_state=has_persistent_state,
            has_dead_l1_slice=has_dead_l1_slice, has_empty_group=False,
            has_indirect_gather=has_indirect_gather, has_recurrence=has_recurrence,
            quant_mode=quant_mode, fallback_spec=regimes[0] if regimes else {},
        )
        results, policies = {}, {}
        for reg in regimes:
            res = self.solve_regime(
                reg, op_name=op_name, topology_axes=axes_obj, roofline=roofline, problem=problem,
                input_dtypes=input_dtypes, has_nan_inf_cases=has_nan_inf_cases,
                has_indirect_gather=has_indirect_gather, has_recurrence=has_recurrence,
                uses_cube=uses_cube, has_persistent_state=has_persistent_state,
                has_dead_l1_slice=has_dead_l1_slice, quant_mode=quant_mode,
            )
            results[res.regime_tag], policies[res.regime_tag] = res, res.to_dict()
        return PiecewiseSchedulePlan(op_name=op_name, topology_axes=resolved_axes, policies=policies, results=results)

    def recompute_regime_for_live_buffers(
        self, regime: Mapping[str, object] | object, live_buffers_or_ledger: object, *,
        op_name: str = "", topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        problem: Problem | None = None, input_dtypes: tuple[str, ...] = ("f16",),
        has_nan_inf_cases: bool = False, has_indirect_gather: bool = False,
        has_recurrence: bool = False, uses_cube: bool | None = None,
        has_persistent_state: bool | None = None, has_dead_l1_slice: bool | None = None,
        quant_mode: str | None = None, unreserved_ubuf: bool = False,
    ) -> RegimePlanResult:
        """Re-solve tile sizes and stages for live buffer footprints."""
        return recompute_regime_for_live_buffers(
            self, regime, live_buffers_or_ledger, op_name=op_name, topology_axes=topology_axes,
            roofline=roofline, problem=problem, input_dtypes=input_dtypes,
            has_nan_inf_cases=has_nan_inf_cases, has_indirect_gather=has_indirect_gather,
            has_recurrence=has_recurrence, uses_cube=uses_cube,
            has_persistent_state=has_persistent_state, has_dead_l1_slice=has_dead_l1_slice,
            quant_mode=quant_mode, unreserved_ubuf=unreserved_ubuf,
        )


def _wl_bytes(wl: object | None) -> int:
    if wl is None:
        return 0
    return int(getattr(wl, "peak_allocated_bytes", 0) or (getattr(wl, "explicit_bytes", 0) + getattr(wl, "implicit_bloat_bytes", 0)))


def _sum_scope_entry(entry: object) -> int:
    if entry is None:
        return 0
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return max(0, int(entry))
    if hasattr(entry, "peak_allocated_bytes"):
        return max(0, int(getattr(entry, "peak_allocated_bytes", 0) or 0))
    if isinstance(entry, Mapping):
        return sum(max(0, int(v)) for v in entry.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    return 0


def _extract_live_buffer_footprints(live_buffers_or_ledger: object) -> tuple[int, int]:
    if live_buffers_or_ledger is None:
        return 0, 0
    waterlines = getattr(live_buffers_or_ledger, "waterlines", None)
    if isinstance(waterlines, Mapping):
        return max(0, _wl_bytes(waterlines.get("UB"))), max(0, _wl_bytes(waterlines.get("L1")))
    if not isinstance(live_buffers_or_ledger, Mapping):
        raise TypeError(
            "live_buffers_or_ledger must be an ObjectivePhysicalLedger or Mapping, "
            f"got {type(live_buffers_or_ledger).__name__}"
        )
    ub_total, l1_total, matched_scoped = 0, 0, False
    for key, val in live_buffers_or_ledger.items():
        k_low = str(key).strip().lower()
        if k_low in ("ub", "ub_buffers", "ub_bytes", "live_ub_bytes"):
            ub_total += _sum_scope_entry(val)
            matched_scoped = True
        elif k_low in ("l1", "l1_buffers", "l1_bytes", "live_l1_bytes"):
            l1_total += _sum_scope_entry(val)
            matched_scoped = True
    if not matched_scoped:
        for key, val in live_buffers_or_ledger.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                sz = max(0, int(val))
                if any(tok in str(key).lower() for tok in ("l1", "cbuf")):
                    l1_total += sz
                else:
                    ub_total += sz
    return ub_total, l1_total


def recompute_regime_for_live_buffers(
    solver: RegimePiecewiseSolver | None, regime: Mapping[str, object] | object,
    live_buffers_or_ledger: object, *, op_name: str = "",
    topology_axes: TopologyAxes | Mapping[str, object] | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
    problem: Problem | None = None, input_dtypes: tuple[str, ...] = ("f16",),
    has_nan_inf_cases: bool = False, has_indirect_gather: bool = False,
    has_recurrence: bool = False, uses_cube: bool | None = None,
    has_persistent_state: bool | None = None, has_dead_l1_slice: bool | None = None,
    quant_mode: str | None = None, unreserved_ubuf: bool = False,
) -> RegimePlanResult:
    """Apply monotone waterline caliper contraction or headroom expansion for live buffers."""
    active_solver = solver if solver is not None else RegimePiecewiseSolver()
    base_plan = regime if isinstance(regime, RegimePlanResult) else active_solver.solve_regime(
        regime, op_name=op_name, topology_axes=topology_axes, roofline=roofline, problem=problem,
        input_dtypes=input_dtypes, has_nan_inf_cases=has_nan_inf_cases,
        has_indirect_gather=has_indirect_gather, has_recurrence=has_recurrence, uses_cube=uses_cube,
        has_persistent_state=has_persistent_state, has_dead_l1_slice=has_dead_l1_slice,
        quant_mode=quant_mode, unreserved_ubuf=unreserved_ubuf,
    )
    dims = dict(base_plan.representative_dims)
    live_ub, live_l1 = _extract_live_buffer_footprints(live_buffers_or_ledger)
    if live_ub == 0 and live_l1 == 0:
        return base_plan

    policy, facts = dict(base_plan.policy), active_solver.facts
    raw_mnk = policy.get("tile_mnk", [128, 128, 64])
    tm0, tn0, tk0 = max(1, int(raw_mnk[0])), max(1, int(raw_mnk[1])), max(1, int(raw_mnk[2]))
    tm, tn, tk = tm0, tn0, tk0
    rp0 = row_pack = max(1, int(policy.get("row_pack_factor", 1) or 1))
    s0 = l1_stages = max(1, int(policy.get("l1_stages", 1) or 1))
    l1_half_split = bool(policy.get("l1_half_bank_split", False))
    l0c_buffers = max(1, int(policy.get("l0c_buffers", 1) or 1))
    vector_mode, ub_256kb = str(policy.get("vector_mode", "simdvf")), bool(policy.get("ub_256kb_unlock", False))
    raw_axes = policy.get("topology_axes")
    axes = TopologyAxes.from_dict(raw_axes) if isinstance(raw_axes, Mapping) else None
    is_coupled_simt = bool(axes is not None and axes.compute_engine == "coupled_1c2v" and axes.iteration_regularity == "irregular_simt")
    l1_total_limit, l1_half_limit, l0c_limit = int(facts.l1_bytes), int(facts.l1_bytes // 2), int(facts.l0c_bytes)
    base_ub, base_l1 = int(policy.get("ub_estimated_bytes", 0) or 0), int(policy.get("l1_estimated_bytes", 0) or 0)
    is_vector_only = str(policy.get("cv_datapath", "none")) == "none"

    def _ub_limit(unlocked_256: bool) -> int:
        if vector_mode in ("simtvf", "hybrid") or is_coupled_simt:
            return int(facts.ub_simtvf_usable_bytes)
        return int(facts.ub_nominal_bytes if unlocked_256 else facts.ub_simdvf_usable_bytes)

    def _est_ub(m_v: int, n_v: int, rp_v: int) -> int:
        if live_ub <= 0:
            return base_ub
        num = m_v * n_v * (rp_v if is_vector_only else 1)
        den = tm0 * tn0 * (rp0 if is_vector_only else 1)
        return int(math.ceil(float(live_ub) * (max(1, num) / float(max(1, den)))))

    def _est_l1(m_v: int, n_v: int, k_v: int, st_v: int) -> int:
        if live_l1 <= 0:
            return base_l1
        scale = (max(1, m_v * k_v + k_v * n_v) / float(max(1, tm0 * tk0 + tk0 * tn0))) * (float(st_v) / float(max(1, s0)))
        return int(math.ceil(float(live_l1) * scale))

    def _half_ok(l1_b: int, st_v: int, half_split: bool) -> bool:
        return not half_split or st_v < 2 or l1_b <= 0 or (float(l1_b) * math.ceil(st_v / 2.0)) / float(st_v) <= float(l1_half_limit)

    def _halve16(v: int) -> int:
        return max(16, (v // 2 // 16) * 16) if v >= 32 else 16

    ub_limit = _ub_limit(ub_256kb)
    cur_ub, cur_l1 = _est_ub(tm, tn, row_pack), _est_l1(tm, tn, tk, l1_stages)
    if live_ub > 0 and cur_ub > ub_limit and vector_mode == "simdvf" and not is_coupled_simt and not ub_256kb and cur_ub <= int(facts.ub_nominal_bytes):
        ub_256kb, ub_limit = True, _ub_limit(True)

    shrank, caliper_steps, caliper_action = False, 0, "none"
    for _ in range(_MAX_CALIPER_STEPS):
        ub_over = live_ub > 0 and cur_ub > ub_limit
        l1_over = live_l1 > 0 and (cur_l1 > l1_total_limit or not _half_ok(cur_l1, l1_stages, l1_half_split))
        if not ub_over and not l1_over:
            break
        progressed = False
        if ub_over:
            if row_pack > 1:
                new_rp = max(1, row_pack // 2)
                if is_vector_only and tm > 1:
                    tm = max(1, (tm * new_rp) // max(1, row_pack))
                row_pack, progressed = new_rp, True
            elif is_vector_only:
                if tn >= tm and tn > 32:
                    tn, progressed = max(16, (tn // 2 // 16) * 16), True
                elif tm > 1:
                    tm, progressed = max(1, tm // 2), True
                elif tn > 16:
                    tn, progressed = max(16, tn // 2), True
            else:
                if tm >= tn and tm > 16:
                    tm, progressed = _halve16(tm), True
                elif tn > 32:
                    tn, progressed = _halve16(tn), True
                elif tm > 16:
                    tm, progressed = _halve16(tm), True
                elif tn > 16:
                    tn, progressed = 16, True
            if progressed:
                shrank, caliper_steps, caliper_action = True, caliper_steps + 1, "ub_halving"

        if l1_over and not progressed:
            if l1_stages > 2:
                l1_stages, progressed = 2, True
            elif l1_stages > 1:
                l1_stages, l1_half_split, progressed = 1, False, True
            elif tk > 16:
                tk, progressed = _halve16(tk), True
            elif tm >= tn and tm > 32:
                tm, progressed = _halve16(tm), True
            elif tn > 32:
                tn, progressed = _halve16(tn), True
            elif tm > 16:
                tm, progressed = 16, True
            if progressed:
                shrank, caliper_steps, caliper_action = True, caliper_steps + 1, "l1_stage_backoff"

        if not progressed:
            break
        cur_ub, cur_l1 = _est_ub(tm, tn, row_pack), _est_l1(tm, tn, tk, l1_stages)

    expanded = False
    if not shrank:
        max_m_cap = max(tm0, min(128, int(dims.get("M", dims.get("S_q", dims.get("rows", 128))) or 128)))
        max_n_cap = max(tn0, min(256, int(dims.get("N", dims.get("D", dims.get("cols", tn0))) or tn0)))
        max_k_cap = max(tk0, min(128, int(dims.get("K", dims.get("D_k", dims.get("D", tk0))) or tk0)))
        ub_headroom = live_ub > 0 and cur_ub < ub_limit and (base_ub == 0 or live_ub < base_ub)
        l1_headroom = live_l1 > 0 and cur_l1 < l1_total_limit and (base_l1 == 0 or live_l1 < base_l1)

        for _ in range(8):
            grew = False
            if l1_headroom and l1_stages == 1:
                cand_l1 = _est_l1(tm, tn, tk, 2)
                if cand_l1 <= l1_total_limit and _half_ok(cand_l1, 2, True):
                    l1_stages, l1_half_split, expanded, grew = 2, True, True, True
            if not grew and l1_headroom and not is_vector_only and tk + 16 <= max_k_cap:
                cand_l1 = _est_l1(tm, tn, tk + 16, l1_stages)
                if cand_l1 <= l1_total_limit and _half_ok(cand_l1, l1_stages, l1_half_split):
                    tk, expanded, grew = tk + 16, True, True
            if not grew and ub_headroom:
                if is_vector_only and row_pack < 16:
                    cand_rp = min(16, row_pack * 2)
                    cand_tm = min(max_m_cap, max(tm, (tm * cand_rp) // max(1, row_pack)))
                    if _est_ub(cand_tm, tn, cand_rp) <= ub_limit:
                        row_pack, tm, expanded, grew = cand_rp, cand_tm, True, True
                elif not is_vector_only:
                    for pair in (((tm + 16, tn) if tm + 16 <= max_m_cap else None), ((tm, tn + 16) if tn + 16 <= max_n_cap else None)):
                        if pair is None:
                            continue
                        cm, cn = pair
                        cl1 = _est_l1(cm, cn, tk, l1_stages)
                        if _est_ub(cm, cn, row_pack) <= ub_limit and cl1 <= l1_total_limit and _half_ok(cl1, l1_stages, l1_half_split) and cm * cn * 4 * l0c_buffers <= l0c_limit:
                            tm, tn, expanded, grew = cm, cn, True, True
                            break
            if not grew:
                break
            cur_ub, cur_l1 = _est_ub(tm, tn, row_pack), _est_l1(tm, tn, tk, l1_stages)

    policy.update({
        "tile_mnk": [tm, tn, tk], "row_pack_factor": row_pack,
        "l1_stages": l1_stages, "l1_half_bank_split": l1_half_split, "ub_256kb_unlock": ub_256kb,
    })
    if live_ub > 0:
        policy["ub_estimated_bytes"] = cur_ub
    if live_l1 > 0:
        policy["l1_estimated_bytes"] = cur_l1
    if tm * tn * 4 * 2 <= l0c_limit:
        policy["l0c_buffers"] = max(l0c_buffers, 2) if not is_vector_only else l0c_buffers
    elif tm * tn * 4 > l0c_limit:
        policy["l0c_buffers"] = 1

    action_tag = "shrunk" if shrank else ("expanded" if expanded else "unchanged")
    note = (
        f"Fixed-point re-solved ({action_tag}, steps={caliper_steps}, action={caliper_action}) for live buffers: "
        f"UB={cur_ub}B, L1={cur_l1}B, tile_mnk={[tm, tn, tk]}, row_pack={row_pack}, l1_stages={l1_stages}"
    )
    return RegimePlanResult(
        regime_tag=base_plan.regime_tag, op_name=base_plan.op_name, topology_axes=base_plan.topology_axes,
        solver_mode=base_plan.solver_mode, policy=policy, representative_dims=dict(base_plan.representative_dims),
        ii=base_plan.ii, buffer_versions=dict(base_plan.buffer_versions), task_stages=dict(base_plan.task_stages),
        fallback_used=base_plan.fallback_used, fallback_reason=base_plan.fallback_reason,
        persistent_state_space=base_plan.persistent_state_space, persistent_state_bytes=base_plan.persistent_state_bytes,
        notes=(*base_plan.notes, note), caliper_steps=caliper_steps, caliper_action=caliper_action,
    )


__all__ = [
    "SOLVER_MODE_ALGEBRAIC",
    "SOLVER_MODE_Z3",
    "PiecewiseSchedulePlan",
    "RegimePiecewiseSolver",
    "RegimePlanResult",
    "RooflineSummaryLike",
    "compute_critical_cube_tile_m",
    "normalize_solver_mode",
    "recompute_regime_for_live_buffers",
]
