"""Hybrid hierarchical piecewise regime solver for Ascend 950 (dav-3510)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tilefoundry.vendor.tilelang_z3 import Problem
from tilefoundry_ascend.ascend.schedule.pipeline_solver import solve_pipeline_with_monotone_backoff
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


def normalize_solver_mode(mode: str) -> str:
    """Normalize legacy solver mode names onto canonical solver identifiers."""
    normalized = str(mode).strip()
    if normalized in _LEGACY_ALGEBRAIC_MODES:
        return SOLVER_MODE_ALGEBRAIC
    return normalized


@runtime_checkable
class RooflineSummaryLike(Protocol):
    """Structural protocol satisfied by HirRooflineSummary without import cycles."""

    anchor_dims: dict[str, int]
    min_gm_bytes: int
    dominant_bound: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the Roofline summary into a JSON-compatible dictionary."""
        ...


def _coerce_roofline_mapping(
    raw: RooflineSummaryLike | Mapping[str, object] | None,
) -> dict[str, object]:
    """Convert a HirRooflineSummary or mapping into a string-keyed dictionary."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, RooflineSummaryLike):
        return raw.to_dict()
    raise TypeError(
        f"roofline must be HirRooflineSummary, Mapping, or None, got {type(raw).__name__}"
    )

_DTYPE_BYTES: dict[str, int] = {
    "f4e2m1": 1,
    "f4e1m2": 1,
    "float4_e2m1": 1,
    "float4_e1m2": 1,
    "i4": 1,
    "u4": 1,
    "q4_0": 1,
    "i8": 1,
    "u8": 1,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "fp8e4m3": 1,
    "fp8e5m2": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e8m0fnu": 1,
    "hif8": 1,
    "f8e8m0": 1,
    "f16": 2,
    "bf16": 2,
    "i16": 2,
    "u16": 2,
    "int16": 2,
    "f32": 4,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "i32": 4,
    "u32": 4,
    "int32": 4,
    "i64": 8,
    "u64": 8,
    "int64": 8,
}


def _dtype_bytes(dtype_name: str) -> int:
    return _DTYPE_BYTES.get(dtype_name.strip().lower(), 2)


def _coerce_topology_axes(raw: object) -> TopologyAxes | None:
    if raw is None:
        return None
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping):
        fields = {str(key): str(value) for key, value in raw.items()}
        return TopologyAxes.from_dict(fields)
    raise TypeError(f"topology_axes must be TopologyAxes or Mapping, got {type(raw).__name__}")


def _synthesize_axes_from_flags(
    *,
    uses_cube: bool,
    has_persistent_state: bool,
    has_dead_l1_slice: bool,
    has_empty_group: bool,
    has_indirect_gather: bool,
    has_recurrence: bool,
    quant_mode: str,
) -> TopologyAxes:
    """Construct orthogonal TopologyAxes purely from structural flags (zero op_name reliance)."""
    has_quant = bool(quant_mode and quant_mode not in ("none", "false"))
    if uses_cube:
        if has_persistent_state or has_empty_group or has_indirect_gather or has_recurrence or has_dead_l1_slice or has_quant:
            compute_engine = "coupled_1c2v"
        else:
            compute_engine = "aic_only"
    else:
        compute_engine = "aiv_only"

    if has_persistent_state:
        memory_ring = "persistent_onchip_state"
    elif uses_cube and (has_recurrence or has_dead_l1_slice):
        memory_ring = "bidirectional_cv_ring"
    elif uses_cube and has_empty_group:
        memory_ring = "l0c_ub_epilogue"
    elif uses_cube:
        memory_ring = "gm_l1_l0_fixpipe"
    elif has_indirect_gather:
        memory_ring = "ub_random_atomic"
    else:
        memory_ring = "gm_ub_stream"

    if has_empty_group:
        iteration_regularity = "ragged_group"
    elif has_indirect_gather:
        iteration_regularity = "irregular_simt"
    else:
        iteration_regularity = "affine_simd"

    return TopologyAxes(
        compute_engine=compute_engine,
        memory_ring=memory_ring,
        iteration_regularity=iteration_regularity,
    )


def _as_int_dims(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    dims: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f"regime dim {key!r} must be an int or decimal string")
        dims[str(key)] = int(value)
    return dims


def _as_str_object_mapping(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def _as_case_ratios(raw: object) -> tuple[CasePhysicsRatios, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("case_ratios must be a sequence of CasePhysicsRatios or mappings")
    ratios: list[CasePhysicsRatios] = []
    for item in raw:
        if isinstance(item, CasePhysicsRatios):
            ratios.append(item)
        elif isinstance(item, Mapping):
            ratios.append(
                CasePhysicsRatios.from_dict({str(key): value for key, value in item.items()})
            )
        else:
            raise ValueError("case_ratios entries must be CasePhysicsRatios or mappings")
    return tuple(ratios)


def _select_representative_ratio(
    ratios: tuple[CasePhysicsRatios, ...],
    representative_case_id: int | None,
) -> CasePhysicsRatios | None:
    if not ratios:
        return None
    if representative_case_id is None:
        return ratios[len(ratios) // 2]
    for ratio in ratios:
        if ratio.case_id == representative_case_id:
            return ratio
    return None


def _extract_regime_fields(
    regime: Mapping[str, object] | object,
) -> tuple[
    str,
    dict[str, int],
    bool,
    bool,
    dict[str, object],
    TopologyAxes | None,
    CasePhysicsRatios | None,
]:
    if isinstance(regime, Mapping):
        tag = str(regime.get("regime_tag", "default"))
        dims = _as_int_dims(regime.get("representative_dims") or regime.get("dims"))
        has_tail = bool(regime.get("has_unaligned_tail", False))
        has_empty = bool(regime.get("has_empty_group", False))
        raw_rf = regime.get("roofline")
        roofline = (
            _coerce_roofline_mapping(raw_rf)
            if isinstance(raw_rf, (Mapping, RooflineSummaryLike))
            else {}
        )
        reg_axes = _coerce_topology_axes(regime.get("topology_axes"))
        raw_case_id = regime.get("representative_case_id")
        representative_case_id = (
            int(raw_case_id)
            if isinstance(raw_case_id, int) and not isinstance(raw_case_id, bool)
            else None
        )
        ratios = _as_case_ratios(regime.get("case_ratios"))
        return (
            tag,
            dims,
            has_tail,
            has_empty,
            roofline,
            reg_axes,
            _select_representative_ratio(ratios, representative_case_id),
        )

    tag = str(getattr(regime, "regime_tag", "default"))
    dims = _as_int_dims(getattr(regime, "representative_dims", None))
    has_tail = bool(getattr(regime, "has_unaligned_tail", False))
    has_empty = bool(getattr(regime, "has_empty_group", False))
    raw_obj_rf = getattr(regime, "roofline", None)
    roofline = (
        _coerce_roofline_mapping(raw_obj_rf)
        if isinstance(raw_obj_rf, (Mapping, RooflineSummaryLike))
        else {}
    )
    reg_axes = _coerce_topology_axes(getattr(regime, "topology_axes", None))
    ratios = _as_case_ratios(getattr(regime, "case_ratios", None))
    representative_case_id = getattr(regime, "representative_case_id", None)
    if representative_case_id is not None and (
        isinstance(representative_case_id, bool)
        or not isinstance(representative_case_id, int)
    ):
        raise ValueError("representative_case_id must be an int")
    return (
        tag,
        dims,
        has_tail,
        has_empty,
        roofline,
        reg_axes,
        _select_representative_ratio(ratios, representative_case_id),
    )


def _mapping_float(payload: Mapping[str, object], key: str, default: float = 0.0) -> float:
    raw = payload.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"roofline field {key!r} must be numeric, got {type(raw).__name__}")
    return float(raw)


def _is_cube_bound(
    roofline: Mapping[str, object] | None,
    dims: Mapping[str, int],
    *,
    uses_cube: bool | None = None,
    axes: TopologyAxes | None = None,
) -> bool:
    if roofline:
        bound = str(roofline.get("dominant_bound", "")).lower()
        if any(k in bound for k in ("cube", "compute", "aic")):
            return True
        cube_us = _mapping_float(roofline, "cube_floor_us")
        mem_us = _mapping_float(roofline, "memory_floor_us")
        vec_us = _mapping_float(roofline, "vec_floor_us")
        if cube_us > 0.0 and cube_us >= max(mem_us, vec_us):
            return True
        if _mapping_float(roofline, "arithmetic_intensity") >= 128.0:
            return True
    m_val = dims.get("M", dims.get("S_q", dims.get("S", dims.get("m", 0))))
    if uses_cube is True or (axes is not None and axes.uses_cube):
        if axes is not None and axes.memory_ring in (
            "gm_l1_l0_fixpipe",
            "l0c_ub_epilogue",
            "bidirectional_cv_ring",
        ):
            return m_val == 0 or m_val >= 64
        if uses_cube is True:
            return m_val == 0 or m_val >= 64
    return False


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
            raise ValueError(
                f"caliper_action must be one of {sorted(_VALID_CALIPER_ACTIONS)}, got {action!r}"
            )
        object.__setattr__(self, "caliper_action", action)

    def to_dict(self) -> dict[str, object]:
        """Return a SIL SchedulePolicy-compatible dictionary."""
        return dict(self.policy)

    def to_summary_dict(self) -> dict[str, object]:
        """Return full regime solve metadata alongside the SIL policy dict."""
        return {
            "regime_tag": self.regime_tag,
            "op_name": self.op_name,
            "topology_axes": self.topology_axes.to_dict(),
            "solver_mode": self.solver_mode,
            "policy": dict(self.policy),
            "representative_dims": dict(self.representative_dims),
            "ii": self.ii,
            "buffer_versions": dict(self.buffer_versions),
            "task_stages": dict(self.task_stages),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "persistent_state_space": self.persistent_state_space,
            "persistent_state_bytes": self.persistent_state_bytes,
            "notes": list(self.notes),
            "caliper_steps": self.caliper_steps,
            "caliper_action": self.caliper_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimePlanResult:
        """Reconstruct a RegimePlanResult from serialized summary metadata."""
        axes = _coerce_topology_axes(payload.get("topology_axes"))
        if axes is None:
            axes = resolve_canonical_topology_axes({})
        raw_policy = payload.get("policy")
        policy = (
            {str(k): v for k, v in raw_policy.items()}
            if isinstance(raw_policy, Mapping)
            else {}
        )
        raw_ii = payload.get("ii")
        ii_val = int(raw_ii) if isinstance(raw_ii, int) and not isinstance(raw_ii, bool) else None
        raw_bv = payload.get("buffer_versions")
        buffer_versions = (
            {str(k): int(v) for k, v in raw_bv.items()}
            if isinstance(raw_bv, Mapping)
            else {}
        )
        raw_ts = payload.get("task_stages")
        task_stages = (
            {str(k): int(v) for k, v in raw_ts.items()}
            if isinstance(raw_ts, Mapping)
            else {}
        )
        raw_notes = payload.get("notes")
        notes = (
            tuple(str(item) for item in raw_notes)
            if isinstance(raw_notes, (list, tuple))
            else ()
        )
        raw_steps = payload.get("caliper_steps", 0)
        caliper_steps = (
            int(raw_steps) if isinstance(raw_steps, int) and not isinstance(raw_steps, bool) else 0
        )
        raw_state_bytes = payload.get("persistent_state_bytes", 0)
        state_bytes = (
            int(raw_state_bytes)
            if isinstance(raw_state_bytes, int) and not isinstance(raw_state_bytes, bool)
            else 0
        )
        return cls(
            regime_tag=str(payload.get("regime_tag", "default")),
            op_name=str(payload.get("op_name", "")),
            topology_axes=axes,
            solver_mode=normalize_solver_mode(
                str(payload.get("solver_mode", SOLVER_MODE_ALGEBRAIC))
            ),
            policy=policy,
            representative_dims=_as_int_dims(payload.get("representative_dims")),
            ii=ii_val,
            buffer_versions=buffer_versions,
            task_stages=task_stages,
            fallback_used=bool(payload.get("fallback_used", False)),
            fallback_reason=str(payload.get("fallback_reason", "")),
            persistent_state_space=str(payload.get("persistent_state_space", "none")),
            persistent_state_bytes=state_bytes,
            notes=notes,
            caliper_steps=caliper_steps,
            caliper_action=str(payload.get("caliper_action", "none")),
        )

    def to_sil_policy(self) -> object:
        """Convert to ``ascend.sil.SchedulePolicy`` if available, else return dict."""
        try:
            from tilefoundry_ascend.ascend.sil import SchedulePolicy  # noqa: PLC0415

            return SchedulePolicy.from_dict(self.policy)
        except ImportError:
            return dict(self.policy)


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
            "op_name": self.op_name,
            "topology_axes": self.topology_axes.to_dict(),
            "policies": {tag: dict(pol) for tag, pol in self.policies.items()},
            "selected_rules": list(self.selected_rules),
        }

    def to_sil_policy_envelope(self) -> object:
        """Convert to ``ascend.sil.PolicyEnvelope`` if available, else return dict."""
        try:
            from tilefoundry_ascend.ascend.sil import PolicyEnvelope  # noqa: PLC0415

            return PolicyEnvelope.from_dict(self.to_dict())
        except ImportError:
            return self.to_dict()


class RegimePiecewiseSolver:
    """Hierarchical solver dispatching static loops to Z3 and structural topologies analytically."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self.target = target or AscendTarget("ascend.950pr")
        facts = self.target.get_facts(AscendSpaceFacts)
        if not isinstance(facts, AscendSpaceFacts):
            raise RuntimeError(
                f"target {self.target.identity!r} did not provide AscendSpaceFacts"
            )
        self.facts = facts

    def _is_ragged_or_recurrent(
        self,
        *,
        axes: TopologyAxes,
        has_empty_group: bool,
        has_indirect_gather: bool,
        has_recurrence: bool,
        has_persistent_state: bool | None,
        problem: Problem | None,
    ) -> bool:
        if problem is None:
            return True
        if has_empty_group or has_indirect_gather or has_recurrence or bool(has_persistent_state):
            return True
        if axes.iteration_regularity in ("irregular_simt", "ragged_group", "dynamic_opcode_stream"):
            return True
        if axes.memory_ring in ("ub_random_atomic", "l0c_ub_epilogue", "bidirectional_cv_ring", "persistent_onchip_state"):
            return True
        return False

    def _base_policy(
        self,
        *,
        regime_tag: str,
        topology_axes: TopologyAxes,
        cube_dtype: str,
    ) -> dict[str, object]:
        return {
            "regime_tag": regime_tag,
            "topology_axes": topology_axes.to_dict(),
            "backend_tier": "tilelang_dsl",
            "tile_mnk": [128, 128, 64],
            "row_pack_factor": 1,
            "num_kv_splits": 1,
            "l1_stages": 2,
            "l1_half_bank_split": True,
            "l1_slot_aliasing": {},
            "l0c_buffers": 1,
            "ub_256kb_unlock": False,
            "ub_estimated_bytes": 0,
            "l1_estimated_bytes": 0,
            "ub_bank_pad_datablocks": 0,
            "ub_rw_shift_datablocks": 0,
            "ub_row_stride_bytes": 0,
            "cv_datapath": "none",
            "sync_mode": "mode4_flag",
            "preload_depth": 2,
            "reverse_ack_period": 14,
            "ensure_empty_subblock_sync": True,
            "swizzle_window": 4,
            "tail_wave_split": True,
            "streamk_mode": "none",
            "l2_bypass_operands": [],
            "vector_mode": "simdvf",
            "direct_no_fusion_vf": False,
            "multi_row_unroll": 1,
            "reduce_accumulators": 1,
            "simt_threads": 512,
            "hf32_allowed": False,
            "fast_unshifted_exp": False,
            "w4_mode": "none",
            "dma_paths_used": [],
            "cube_operand_dtype": cube_dtype,
        }

    def _plan_analytical_regime(
        self,
        *,
        regime_tag: str,
        op_name: str,
        axes: TopologyAxes,
        explicit_axes: bool,
        dims: dict[str, int],
        roofline: RooflineSummaryLike | Mapping[str, object] | None,
        has_unaligned_tail: bool,
        has_empty_group: bool,
        input_dtypes: tuple[str, ...],
        has_nan_inf_cases: bool,
        uses_cube: bool | None = None,
        has_persistent_state: bool | None = None,
        has_dead_l1_slice: bool | None = None,
        has_indirect_gather: bool = False,
        quant_mode: str | None = None,
        representative_ratios: CasePhysicsRatios | None = None,
        solver_mode: str = SOLVER_MODE_ALGEBRAIC,
        fallback_used: bool = False,
        fallback_reason: str = "",
    ) -> RegimePlanResult:
        roofline_map = _coerce_roofline_mapping(roofline)
        rf_anchor_dims = _as_int_dims(roofline_map.get("anchor_dims"))
        merged_dims = {**rf_anchor_dims, **dims}
        elem_dtype = input_dtypes[0] if input_dtypes else "f16"
        elem_bytes = _dtype_bytes(elem_dtype)

        eff_uses_cube = bool(uses_cube) if uses_cube is not None else axes.uses_cube

        eff_persistent_state = (
            bool(has_persistent_state)
            if has_persistent_state is not None
            else (axes.memory_ring == "persistent_onchip_state")
        )
        eff_dead_l1_slice = bool(has_dead_l1_slice) if has_dead_l1_slice is not None else False
        eff_indirect = (
            bool(has_indirect_gather)
            or axes.iteration_regularity == "irregular_simt"
            or axes.memory_ring == "ub_random_atomic"
        )
        eff_ragged = axes.iteration_regularity == "ragged_group" or bool(has_empty_group)
        eff_opcode_stream = axes.iteration_regularity == "dynamic_opcode_stream"

        raw_qmode = (quant_mode or "").strip().lower() or "none"

        _cube_dtype_aliases = {
            "f4e2m1": "f4e2m1",
            "float4_e2m1": "f4e2m1",
            "f4e1m2": "f4e1m2",
            "float4_e1m2": "f4e1m2",
            "f16": "f16",
            "float16": "float16",
            "bf16": "bf16",
            "bfloat16": "bfloat16",
            "f32": "f32",
            "float32": "float32",
            "hf32": "hf32",
            "fp8e4m3": "fp8e4m3",
            "float8_e4m3fn": "fp8e4m3",
            "fp8e5m2": "fp8e5m2",
            "float8_e5m2": "fp8e5m2",
            "hif8": "hif8",
            "i8": "i8",
            "int8": "int8",
            "u8": "u8",
            "uint8": "u8",
        }
        if not eff_uses_cube:
            cube_dtype = ""
        elif raw_qmode in ("q4_0", "w4a16", "int4", "i4", "u4", "aiv_unpack_f16"):
            cube_dtype = "q4_0"
        else:
            cube_dtype = _cube_dtype_aliases.get(elem_dtype.strip().lower(), "f16")

        policy = self._base_policy(
            regime_tag=regime_tag, topology_axes=axes, cube_dtype=cube_dtype
        )

        cube_bound = _is_cube_bound(
            roofline_map,
            merged_dims,
            uses_cube=eff_uses_cube,
            axes=axes,
        )
        notes: list[str] = []
        state_space = "none"
        state_bytes = 0

        physics_wave: WaveBand | None = None
        if representative_ratios is not None:
            engine_phase, layout_phase, residency_phase, alignment_phase, physics_wave = (
                representative_ratios.phase_signature(axes)
            )
            notes.append(
                "Physics phase "
                f"{engine_phase}/{layout_phase}/{residency_phase}/"
                f"{alignment_phase}/{physics_wave}"
            )

        l1_half = self.facts.l1_bytes // 2
        ub_simdvf = self.facts.ub_simdvf_usable_bytes
        ub_nom = self.facts.ub_nominal_bytes
        ub_simtvf = self.facts.ub_simtvf_usable_bytes
        ub_coupled_cap = (
            ub_simtvf
            if (axes.compute_engine == "coupled_1c2v" and axes.iteration_regularity == "irregular_simt")
            else ub_simdvf
        )

        m_dim = dims.get("M", dims.get("S_q", dims.get("S", dims.get("B_total", 128))))
        n_dim = dims.get("N", dims.get("D_v", dims.get("D", 128)))
        k_dim = dims.get("K", dims.get("D_k", dims.get("D", 128)))

        crit_cube_m = compute_critical_cube_tile_m(self.facts, elem_dtype)
        aligned_crit_m = max(16, ((crit_cube_m + 15) // 16) * 16)

        if eff_uses_cube and (
            axes.memory_ring in ("bidirectional_cv_ring", "persistent_onchip_state", "l0c_ub_epilogue")
            or eff_persistent_state
            or eff_ragged
        ):
            if axes.memory_ring == "bidirectional_cv_ring":
                if cube_bound or m_dim >= crit_cube_m:
                    if crit_cube_m >= 82:
                        tile_m = (
                            aligned_crit_m
                            if (m_dim < 128 or eff_indirect or eff_dead_l1_slice)
                            else 128
                        )
                    elif m_dim >= 128 and not eff_indirect and not eff_dead_l1_slice:
                        tile_m = 128
                    else:
                        m_aligned = max(aligned_crit_m, ((max(1, m_dim) + 15) // 16) * 16)
                        tile_m = min(96, m_aligned)
                    tile_n = 112 if tile_m == 96 else 128
                    tile_k = min(64, max(32, k_dim))
                    policy["tile_mnk"] = [max(aligned_crit_m, tile_m), tile_n, tile_k]
                    policy["num_kv_splits"] = 1
                else:
                    tile_m = max(16, min(64, ((max(1, m_dim) + 15) // 16) * 16))
                    policy["tile_mnk"] = [tile_m, 112, min(64, max(32, k_dim))]
                    policy["num_kv_splits"] = max(
                        1, min(self.facts.num_aic_cores, dims.get("S_kv", 1024) // 128)
                    )
                    policy["l2_bypass_operands"] = ["K", "V"]

                tm, tn, tk = policy["tile_mnk"]
                policy["l1_stages"] = 2
                policy["l1_half_bank_split"] = True
                stage_l1 = (tm * tk + tk * tn) * elem_bytes
                if stage_l1 > l1_half:
                    tk = max(32, (l1_half // max(1, (tm + tn) * elem_bytes)) // 16 * 16)
                    policy["tile_mnk"] = [tm, tn, tk]
                    stage_l1 = (tm * tk + tk * tn) * elem_bytes
                policy["l1_estimated_bytes"] = stage_l1 * 2
                l0c_slot = tm * tn * 4
                policy["l0c_buffers"] = 2 if l0c_slot * 2 <= self.facts.l0c_bytes else 1
                policy["cv_datapath"] = "l0c_to_ub_split_m+ub_to_l1_nd2nz"
                policy["ub_bank_pad_datablocks"] = 1
                policy["ub_rw_shift_datablocks"] = 8
                policy["ub_row_stride_bytes"] = tn * 4 + 32
                ub_est = ((tm // 2) * (tn * 4 + 32)) * 2
                policy["ub_estimated_bytes"] = min(ub_est, ub_coupled_cap)
                policy["sync_mode"] = "mode4_flag"
                policy["reverse_ack_period"] = 14
                policy["dma_paths_used"] = [
                    "GM->L1",
                    "L1->L0A",
                    "L1->L0B",
                    "L0C->UB",
                    "UB->L1",
                    "UB->GM",
                ]
                if eff_persistent_state:
                    d_k = max(32, min(128, k_dim))
                    d_v = max(32, min(128, n_dim))
                    state_bytes = d_k * d_v * 4
                    state_space = "L1" if state_bytes <= l1_half else "UB"
                    notes.append("Persistent on-chip FP32 state paired with bidirectional CV ring")

            elif eff_persistent_state:
                if cube_bound:
                    if crit_cube_m >= 82:
                        chunk_m = aligned_crit_m
                    else:
                        chunk_m = max(
                            aligned_crit_m, min(64, ((max(1, m_dim) + 15) // 16) * 16)
                        )
                else:
                    chunk_m = (
                        64
                        if crit_cube_m >= 82
                        else max(aligned_crit_m, min(64, ((max(1, m_dim) + 15) // 16) * 16))
                    )
                if eff_ragged and cube_bound and chunk_m < crit_cube_m:
                    chunk_m = aligned_crit_m
                d_k = max(32, min(128, dims.get("D_k", dims.get("K", dims.get("D", 128)))))
                d_v = max(32, min(128, dims.get("D_v", dims.get("N", dims.get("D", 128)))))
                state_bytes = d_k * d_v * 4
                policy["tile_mnk"] = [chunk_m, d_v, d_k]
                policy["l1_stages"] = 2
                policy["l1_half_bank_split"] = True
                policy["l0c_buffers"] = 2 if chunk_m * d_v * 4 * 2 <= self.facts.l0c_bytes else 1
                policy["cv_datapath"] = "l0c_to_ub_split_m+ub_to_l1_nd2nz"
                policy["ub_bank_pad_datablocks"] = 1
                policy["ub_rw_shift_datablocks"] = 8
                policy["ub_row_stride_bytes"] = d_v * 4 + 32
                policy["ub_estimated_bytes"] = min(ub_coupled_cap, (chunk_m // 2) * (d_v * 4 + 32) * 2)
                policy["l1_estimated_bytes"] = min(
                    self.facts.l1_bytes,
                    state_bytes + (chunk_m * d_k) * elem_bytes * 2,
                )
                policy["dma_paths_used"] = [
                    "GM->L1",
                    "L1->L0A",
                    "L1->L0B",
                    "L0C->UB",
                    "UB->L1",
                    "UB->GM",
                ]
                state_space = "L1" if state_bytes <= l1_half else "UB"
                notes.append("Persistent core ownership + FP32 state residency without GM roundtrips")

            else:
                if cube_bound:
                    if m_dim >= 128:
                        tile_m = 128
                    elif crit_cube_m >= 82:
                        tile_m = aligned_crit_m
                    else:
                        tile_m = max(
                            aligned_crit_m, min(96, ((max(1, m_dim) + 15) // 16) * 16)
                        )
                else:
                    tile_m = (
                        64
                        if crit_cube_m >= 82
                        else max(aligned_crit_m, min(64, ((max(1, m_dim) + 15) // 16) * 16))
                    )
                if cube_bound and tile_m < crit_cube_m:
                    tile_m = aligned_crit_m
                tile_n = 112 if tile_m == 96 else 128
                tile_k = 64
                policy["tile_mnk"] = [tile_m, tile_n, tile_k]
                policy["l1_stages"] = 2
                policy["l1_half_bank_split"] = True
                stage_l1 = (tile_m * tile_k + tile_k * tile_n) * elem_bytes
                policy["l1_estimated_bytes"] = min(self.facts.l1_bytes, stage_l1 * 2)
                policy["l0c_buffers"] = 2 if tile_m * tile_n * 4 * 2 <= self.facts.l0c_bytes else 1
                policy["cv_datapath"] = "l0c_to_ub_split_m"
                policy["ub_bank_pad_datablocks"] = 1
                policy["ub_row_stride_bytes"] = tile_n * 4 + 32
                policy["ub_estimated_bytes"] = min(ub_coupled_cap, ((tile_m // 2) * (tile_n * 4 + 32)) * 2)
                policy["dma_paths_used"] = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->GM"]

            if eff_dead_l1_slice:
                policy["l1_slot_aliasing"] = {"reused_stage": "dead_l1_slice"}
                notes.append("In-place L1 slot aliasing on dead L1 slice")

            if eff_indirect:
                policy["simt_threads"] = 512
                notes.append(f"AIV-ahead indirect/sparse index resolution + Cube M>={crit_cube_m} ring")

            if eff_ragged:
                policy["ensure_empty_subblock_sync"] = True
                policy["swizzle_window"] = 4
                policy["tail_wave_split"] = True
                if has_empty_group:
                    notes.append("Empty group fast skip (m==0) with symmetric subblock sync parity")
                notes.append("Rolling cross-group core assignment + per-group BaseMBalance")

        elif eff_uses_cube:
            if m_dim >= 128 or m_dim == 0:
                tile_m = max(aligned_crit_m, 128)
            elif m_dim >= crit_cube_m:
                if crit_cube_m >= 82:
                    tile_m = aligned_crit_m
                else:
                    tile_m = max(aligned_crit_m, min(96, ((max(1, m_dim) + 15) // 16) * 16))
            else:
                tile_m = 64 if crit_cube_m >= 82 else aligned_crit_m
            if cube_bound and tile_m < crit_cube_m:
                tile_m = aligned_crit_m
            tile_n = 112 if tile_m == 96 else 128
            tile_k = 64
            policy["tile_mnk"] = [tile_m, tile_n, tile_k]
            policy["l1_stages"] = 2
            policy["l1_half_bank_split"] = True
            policy["l1_estimated_bytes"] = min(
                self.facts.l1_bytes,
                (tile_m * tile_k + tile_k * tile_n) * elem_bytes * 2,
            )
            policy["l0c_buffers"] = 2 if tile_m * tile_n * 4 * 2 <= self.facts.l0c_bytes else 1
            policy["swizzle_window"] = 4
            policy["tail_wave_split"] = True
            if raw_qmode in ("q4_0", "w4a16", "int4", "i4", "u4", "aiv_unpack_f16"):
                policy["w4_mode"] = "aiv_unpack_f16"
                policy["cv_datapath"] = "l0c_to_ub_split_m"
                policy["dma_paths_used"] = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->GM"]
            elif raw_qmode in ("mxa8w4", "mxa8w4_bitshift_x64"):
                policy["w4_mode"] = "mxa8w4_bitshift_x64"
                policy["cv_datapath"] = "l0c_to_ub_split_m"
                policy["dma_paths_used"] = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->GM"]
            elif raw_qmode not in ("none", "false") or axes.compute_engine == "coupled_1c2v":
                policy["cv_datapath"] = "l0c_to_ub_split_m"
                policy["dma_paths_used"] = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->GM"]
            else:
                policy["cv_datapath"] = "fixpipe_to_gm"
                policy["dma_paths_used"] = ["GM->L1", "L1->L0A", "L1->L0B", "L0C->GM"]
            if eff_dead_l1_slice:
                policy["l1_slot_aliasing"] = {"reused_stage": "dead_l1_slice"}
                notes.append("In-place L1 slot aliasing on dead L1 slice")

        elif eff_opcode_stream:
            num_regs = max(4, min(64, dims.get("R", dims.get("num_regs", 16))))
            width = max(64, dims.get("W", dims.get("N", dims.get("cols", 1024))))
            w_slice = max(32, math.ceil(width / max(1, self.facts.num_aiv_cores)))
            w_slice = min(2048, ((w_slice + 15) // 16) * 16)
            while w_slice > 32 and num_regs * (w_slice * 4 + 32) > ub_nom:
                w_slice = max(32, (w_slice // 2 // 16) * 16)
            row_bytes = w_slice * 4 + 32
            state_bytes = num_regs * row_bytes
            policy["tile_mnk"] = [num_regs, w_slice, 1]
            policy["l1_stages"] = 1
            policy["l1_half_bank_split"] = False
            policy["cv_datapath"] = "none"
            policy["sync_mode"] = "none"
            policy["vector_mode"] = "simdvf"
            policy["direct_no_fusion_vf"] = True
            policy["ub_bank_pad_datablocks"] = 1
            policy["ub_row_stride_bytes"] = row_bytes
            policy["ub_256kb_unlock"] = state_bytes > ub_simdvf and state_bytes <= ub_nom
            policy["ub_estimated_bytes"] = min(
                ub_nom if policy["ub_256kb_unlock"] else ub_simdvf, state_bytes
            )
            policy["dma_paths_used"] = ["GM->UB", "UB->GM"]
            state_space = "UB"
            notes.append("Horizontal W-slice striping across AIV cores; 100% UB-resident VM registers")

        elif eff_persistent_state:
            rows = min(max(1, m_dim), 64)
            cols = max(16, min(2048, dims.get("N", dims.get("D", dims.get("cols", 64)))))
            padded_row_bytes = cols * 4 + 32
            while rows > 1 and rows * padded_row_bytes * 2 > ub_nom:
                rows //= 2
            state_bytes = rows * padded_row_bytes
            policy["tile_mnk"] = [rows, cols, 1]
            policy["l1_stages"] = 1
            policy["l1_half_bank_split"] = False
            policy["cv_datapath"] = "none"
            policy["vector_mode"] = "simdvf"
            policy["ub_bank_pad_datablocks"] = 1
            policy["ub_rw_shift_datablocks"] = 8
            policy["ub_row_stride_bytes"] = padded_row_bytes
            policy["ub_256kb_unlock"] = state_bytes > ub_simdvf and state_bytes <= ub_nom
            policy["ub_estimated_bytes"] = min(
                ub_nom if policy["ub_256kb_unlock"] else ub_simdvf, state_bytes * 2
            )
            policy["fast_unshifted_exp"] = not has_nan_inf_cases
            policy["dma_paths_used"] = ["GM->UB", "UB->GM"]
            state_space = "UB"
            notes.append("Persistent FP32 UB state residency across iterative loop")

        elif axes.memory_ring == "ub_random_atomic" or eff_indirect or axes.iteration_regularity == "irregular_simt":
            tile_m = min(64, max(16, m_dim))
            tile_n = min(128, max(32, n_dim))
            policy["tile_mnk"] = [tile_m, tile_n, 1]
            policy["l1_stages"] = 1
            policy["l1_half_bank_split"] = False
            policy["vector_mode"] = "simtvf"
            policy["simt_threads"] = 512
            policy["ub_256kb_unlock"] = False
            policy["ub_bank_pad_datablocks"] = 1
            policy["ub_rw_shift_datablocks"] = 8
            row_bytes = tile_n * elem_bytes + 32
            policy["ub_row_stride_bytes"] = row_bytes
            policy["ub_estimated_bytes"] = min(ub_simtvf, tile_m * row_bytes * 2)
            policy["dma_paths_used"] = ["GM->UB", "UB->GM"]
            notes.append("SIMT __launch_bounds__(512) + warp-privatized +8-padded UB accumulators")

        else:
            raw_tail_n = merged_dims.get(
                "N",
                merged_dims.get("D", merged_dims.get("cols", 1024)),
            )
            rf_bound = str(roofline_map.get("dominant_bound", "")).strip().lower()
            rf_min_gm = int(_mapping_float(roofline_map, "min_gm_bytes"))
            ub_half_bytes = ub_simdvf // 2
            work_elem_bytes = max(4, elem_bytes)
            twopass_switch_elems = max(
                16, ((ub_half_bytes // work_elem_bytes) // 16) * 16
            )
            raw_ub_tile = min(
                (ub_half_bytes - 32) // max(1, 2 * work_elem_bytes),
                (ub_simdvf // 4 - 32) // max(1, elem_bytes),
                self.facts.max_mmad_mnk,
            )
            if raw_ub_tile >= 16:
                ub_tile_elems = 1 << (raw_ub_tile.bit_length() - 1)
                ub_tile_elems = max(16, min(ub_tile_elems, raw_ub_tile))
            else:
                ub_tile_elems = 16
            row_work_bytes = max(1, raw_tail_n) * work_elem_bytes
            row_raw_bytes = max(1, raw_tail_n) * elem_bytes
            is_memory_bound = rf_bound in ("memory", "memory_bandwidth")
            exceeds_ub_half = (
                raw_tail_n > twopass_switch_elems or row_raw_bytes > ub_half_bytes
            )
            if is_memory_bound and (
                exceeds_ub_half
                or (rf_min_gm > ub_simdvf and row_work_bytes > ub_half_bytes)
            ):
                tile_n = max(16, min(raw_tail_n, ub_tile_elems))
                row_bytes = max(32, tile_n * elem_bytes)
                pack = 1
                policy["ub_tile_elems"] = tile_n
                policy["twopass_switch_elems"] = twopass_switch_elems
                policy["requires_twopass_reduction"] = True
                notes.append(
                    f"Algebraic UB twopass threshold={twopass_switch_elems} elems, "
                    f"ub_tile_elems={tile_n}"
                )
            else:
                tile_n = max(1, min(2048, raw_tail_n))
                row_bytes = max(32, tile_n * elem_bytes)
                pack = min(16, max(1, (16384 + row_bytes - 1) // row_bytes))
                while pack > 1 and pack * (row_bytes + 32) * 4 > ub_simdvf:
                    pack //= 2
                if is_memory_bound:
                    policy["ub_tile_elems"] = tile_n
                    policy["twopass_switch_elems"] = twopass_switch_elems
                    policy["requires_twopass_reduction"] = False
            policy["tile_mnk"] = [pack, tile_n, 1]
            policy["row_pack_factor"] = pack
            policy["multi_row_unroll"] = 2 if row_bytes <= 16384 else 1
            policy["l1_stages"] = 1
            policy["l1_half_bank_split"] = False
            policy["vector_mode"] = "simdvf"
            policy["ub_bank_pad_datablocks"] = 1
            policy["ub_row_stride_bytes"] = row_bytes + 32
            est = pack * (row_bytes + 32) * 4
            policy["ub_256kb_unlock"] = est > ub_simdvf and est <= ub_nom
            policy["ub_estimated_bytes"] = min(
                ub_nom if policy["ub_256kb_unlock"] else ub_simdvf, est
            )
            policy["fast_unshifted_exp"] = not has_nan_inf_cases
            policy["dma_paths_used"] = ["GM->UB", "UB->GM"]

        if has_unaligned_tail:
            notes.append("Predicated / DataCopyPad unaligned tail handling enabled")

        if physics_wave is not None:
            policy["tail_wave_split"] = (
                physics_wave != "balanced_waves" or has_unaligned_tail
            )

        return RegimePlanResult(
            regime_tag=regime_tag,
            op_name=op_name,
            topology_axes=axes,
            solver_mode=solver_mode,
            policy=policy,
            representative_dims=dict(dims),
            ii=None,
            buffer_versions={},
            task_stages={},
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            persistent_state_space=state_space,
            persistent_state_bytes=state_bytes,
            notes=tuple(notes),
        )

    def solve_regime(
        self,
        regime: Mapping[str, object] | object,
        *,
        op_name: str = "",
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        problem: Problem | None = None,
        input_dtypes: tuple[str, ...] = ("f16",),
        has_nan_inf_cases: bool = False,
        has_indirect_gather: bool = False,
        has_recurrence: bool = False,
        uses_cube: bool | None = None,
        has_persistent_state: bool | None = None,
        has_dead_l1_slice: bool | None = None,
        quant_mode: str | None = None,
        unreserved_ubuf: bool = False,
    ) -> RegimePlanResult:
        """Solve one shape regime via Z3 projection or algebraic piecewise planning."""
        tag, dims, has_tail, has_empty, reg_roofline, reg_axes, representative_ratios = (
            _extract_regime_fields(regime)
        )
        explicit_axes_obj = _coerce_topology_axes(topology_axes) or reg_axes
        has_explicit_flags = any(
            v is not None for v in (uses_cube, has_persistent_state, has_dead_l1_slice, quant_mode)
        ) or bool(has_indirect_gather)

        if explicit_axes_obj is not None:
            resolved_axes = explicit_axes_obj
        elif has_explicit_flags:
            resolved_axes = _synthesize_axes_from_flags(
                uses_cube=bool(uses_cube) if uses_cube is not None else False,
                has_persistent_state=bool(has_persistent_state),
                has_dead_l1_slice=bool(has_dead_l1_slice),
                has_empty_group=has_empty,
                has_indirect_gather=has_indirect_gather,
                has_recurrence=has_recurrence,
                quant_mode=quant_mode or "none",
            )
        else:
            resolved_axes = resolve_canonical_topology_axes(
                regime if isinstance(regime, Mapping) else {}
            )

        resolved_roofline = (
            _coerce_roofline_mapping(roofline) if roofline is not None else reg_roofline
        )

        if not self._is_ragged_or_recurrent(
            axes=resolved_axes,
            has_empty_group=has_empty,
            has_indirect_gather=has_indirect_gather,
            has_recurrence=has_recurrence,
            has_persistent_state=has_persistent_state,
            problem=problem,
        ):
            assert problem is not None
            del unreserved_ubuf
            z3_res = solve_pipeline_with_monotone_backoff(problem)
            base_plan = self._plan_analytical_regime(
                regime_tag=tag,
                op_name=op_name,
                axes=resolved_axes,
                explicit_axes=(explicit_axes_obj is not None),
                dims=dims,
                roofline=resolved_roofline,
                has_unaligned_tail=has_tail,
                has_empty_group=has_empty,
                input_dtypes=input_dtypes,
                has_nan_inf_cases=has_nan_inf_cases,
                uses_cube=uses_cube,
                has_persistent_state=has_persistent_state,
                has_dead_l1_slice=has_dead_l1_slice,
                has_indirect_gather=has_indirect_gather,
                quant_mode=quant_mode,
                representative_ratios=representative_ratios,
                solver_mode="z3_projected",
            )
            policy = dict(base_plan.policy)
            ub_bytes = sum(
                b.slot_bytes * z3_res.buffer_versions.get(b.name, 1)
                for b in z3_res.problem.buffers
                if b.space == "UB"
            )
            l1_bytes = sum(
                b.slot_bytes * z3_res.buffer_versions.get(b.name, 1)
                for b in z3_res.problem.buffers
                if b.space == "L1"
            )
            l1_vers = [
                z3_res.buffer_versions.get(b.name, 1)
                for b in z3_res.problem.buffers
                if b.space == "L1"
            ]
            l0c_vers = [
                z3_res.buffer_versions.get(b.name, 1)
                for b in z3_res.problem.buffers
                if b.space == "L0C"
            ]
            if l1_vers:
                policy["l1_stages"] = max(l1_vers)
                policy["l1_half_bank_split"] = max(l1_vers) >= 2
            if l0c_vers:
                policy["l0c_buffers"] = max(l0c_vers)
            if ub_bytes > 0:
                policy["ub_estimated_bytes"] = ub_bytes
                policy["ub_256kb_unlock"] = ub_bytes > self.facts.ub_simdvf_usable_bytes
            if l1_bytes > 0:
                policy["l1_estimated_bytes"] = l1_bytes

            return RegimePlanResult(
                regime_tag=tag,
                op_name=op_name,
                topology_axes=resolved_axes,
                solver_mode="z3_projected",
                policy=policy,
                representative_dims=dims,
                ii=z3_res.assignment.ii,
                buffer_versions=dict(z3_res.buffer_versions),
                task_stages=dict(z3_res.task_stages),
                fallback_used=False,
                fallback_reason="",
                persistent_state_space=base_plan.persistent_state_space,
                persistent_state_bytes=base_plan.persistent_state_bytes,
                notes=("Z3 modulo schedule projected at regime representative dims",),
            )

        return self._plan_analytical_regime(
            regime_tag=tag,
            op_name=op_name,
            axes=resolved_axes,
            explicit_axes=(explicit_axes_obj is not None),
            dims=dims,
            roofline=resolved_roofline,
            has_unaligned_tail=has_tail,
            has_empty_group=has_empty,
            input_dtypes=input_dtypes,
            has_nan_inf_cases=has_nan_inf_cases,
            uses_cube=uses_cube,
            has_persistent_state=has_persistent_state,
            has_dead_l1_slice=has_dead_l1_slice,
            has_indirect_gather=has_indirect_gather,
            quant_mode=quant_mode,
            representative_ratios=representative_ratios,
            solver_mode=SOLVER_MODE_ALGEBRAIC,
        )

    def solve_all_regimes(
        self,
        regimes: Sequence[Mapping[str, object] | object],
        *,
        op_name: str = "",
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        problem: Problem | None = None,
        input_dtypes: tuple[str, ...] = ("f16",),
        has_nan_inf_cases: bool = False,
        has_indirect_gather: bool = False,
        has_recurrence: bool = False,
        uses_cube: bool | None = None,
        has_persistent_state: bool | None = None,
        has_dead_l1_slice: bool | None = None,
        quant_mode: str | None = None,
    ) -> PiecewiseSchedulePlan:
        """Solve every regime slice and assemble a SIL-compatible PiecewiseSchedulePlan."""
        axes_obj = _coerce_topology_axes(topology_axes)
        if axes_obj is not None:
            resolved_axes = axes_obj
        elif any(
            v is not None for v in (uses_cube, has_persistent_state, has_dead_l1_slice, quant_mode)
        ) or has_indirect_gather:
            resolved_axes = _synthesize_axes_from_flags(
                uses_cube=bool(uses_cube) if uses_cube is not None else False,
                has_persistent_state=bool(has_persistent_state),
                has_dead_l1_slice=bool(has_dead_l1_slice),
                has_empty_group=False,
                has_indirect_gather=has_indirect_gather,
                has_recurrence=has_recurrence,
                quant_mode=quant_mode or "none",
            )
        else:
            first_reg = regimes[0] if regimes else {}
            if isinstance(first_reg, Mapping):
                resolved_axes = resolve_canonical_topology_axes(first_reg)
            else:
                reg_axes = getattr(first_reg, "topology_axes", None)
                resolved_axes = (
                    _coerce_topology_axes(reg_axes) or resolve_canonical_topology_axes({})
                )

        results: dict[str, RegimePlanResult] = {}
        policies: dict[str, dict[str, object]] = {}
        for reg in regimes:
            res = self.solve_regime(
                reg,
                op_name=op_name,
                topology_axes=axes_obj,
                roofline=roofline,
                problem=problem,
                input_dtypes=input_dtypes,
                has_nan_inf_cases=has_nan_inf_cases,
                has_indirect_gather=has_indirect_gather,
                has_recurrence=has_recurrence,
                uses_cube=uses_cube,
                has_persistent_state=has_persistent_state,
                has_dead_l1_slice=has_dead_l1_slice,
                quant_mode=quant_mode,
            )
            results[res.regime_tag] = res
            policies[res.regime_tag] = res.to_dict()
        return PiecewiseSchedulePlan(
            op_name=op_name,
            topology_axes=resolved_axes,
            policies=policies,
            results=results,
        )

    def recompute_regime_for_live_buffers(
        self,
        regime: Mapping[str, object] | object,
        live_buffers_or_ledger: object,
        *,
        op_name: str = "",
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        problem: Problem | None = None,
        input_dtypes: tuple[str, ...] = ("f16",),
        has_nan_inf_cases: bool = False,
        has_indirect_gather: bool = False,
        has_recurrence: bool = False,
        uses_cube: bool | None = None,
        has_persistent_state: bool | None = None,
        has_dead_l1_slice: bool | None = None,
        quant_mode: str | None = None,
        unreserved_ubuf: bool = False,
    ) -> RegimePlanResult:
        """Re-solve tile sizes and stages for live buffer footprints."""
        return recompute_regime_for_live_buffers(
            self,
            regime,
            live_buffers_or_ledger,
            op_name=op_name,
            topology_axes=topology_axes,
            roofline=roofline,
            problem=problem,
            input_dtypes=input_dtypes,
            has_nan_inf_cases=has_nan_inf_cases,
            has_indirect_gather=has_indirect_gather,
            has_recurrence=has_recurrence,
            uses_cube=uses_cube,
            has_persistent_state=has_persistent_state,
            has_dead_l1_slice=has_dead_l1_slice,
            quant_mode=quant_mode,
            unreserved_ubuf=unreserved_ubuf,
        )


def _extract_live_buffer_footprints(
    live_buffers_or_ledger: object,
) -> tuple[int, int]:
    """Return (live_ub_bytes, live_l1_bytes) from ObjectivePhysicalLedger or buffer Mapping."""
    if live_buffers_or_ledger is None:
        return 0, 0

    waterlines = getattr(live_buffers_or_ledger, "waterlines", None)
    if isinstance(waterlines, Mapping):
        ub_wl = waterlines.get("UB")
        l1_wl = waterlines.get("L1")
        ub_bytes = (
            int(getattr(ub_wl, "peak_allocated_bytes", 0) or (getattr(ub_wl, "explicit_bytes", 0) + getattr(ub_wl, "implicit_bloat_bytes", 0)))
            if ub_wl is not None
            else 0
        )
        l1_bytes = (
            int(getattr(l1_wl, "peak_allocated_bytes", 0) or (getattr(l1_wl, "explicit_bytes", 0) + getattr(l1_wl, "implicit_bloat_bytes", 0)))
            if l1_wl is not None
            else 0
        )
        return max(0, ub_bytes), max(0, l1_bytes)

    if not isinstance(live_buffers_or_ledger, Mapping):
        raise TypeError(
            "live_buffers_or_ledger must be an ObjectivePhysicalLedger or Mapping, "
            f"got {type(live_buffers_or_ledger).__name__}"
        )

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

    ub_total = 0
    l1_total = 0
    matched_scoped_keys = False
    for key, val in live_buffers_or_ledger.items():
        k_low = str(key).strip().lower()
        if k_low in ("ub", "ub_buffers", "ub_bytes", "live_ub_bytes"):
            ub_total += _sum_scope_entry(val)
            matched_scoped_keys = True
        elif k_low in ("l1", "l1_buffers", "l1_bytes", "live_l1_bytes"):
            l1_total += _sum_scope_entry(val)
            matched_scoped_keys = True

    if not matched_scoped_keys:
        for key, val in live_buffers_or_ledger.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            k_low = str(key).strip().lower()
            sz = max(0, int(val))
            if any(tok in k_low for tok in ("l1", "cbuf")):
                l1_total += sz
            else:
                ub_total += sz
    return ub_total, l1_total


def recompute_regime_for_live_buffers(
    solver: RegimePiecewiseSolver | None,
    regime: Mapping[str, object] | object,
    live_buffers_or_ledger: object,
    *,
    op_name: str = "",
    topology_axes: TopologyAxes | Mapping[str, object] | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
    problem: Problem | None = None,
    input_dtypes: tuple[str, ...] = ("f16",),
    has_nan_inf_cases: bool = False,
    has_indirect_gather: bool = False,
    has_recurrence: bool = False,
    uses_cube: bool | None = None,
    has_persistent_state: bool | None = None,
    has_dead_l1_slice: bool | None = None,
    quant_mode: str | None = None,
    unreserved_ubuf: bool = False,
) -> RegimePlanResult:
    """Apply monotone waterline caliper contraction or headroom expansion for live buffers."""
    active_solver = solver if solver is not None else RegimePiecewiseSolver()
    if isinstance(regime, RegimePlanResult):
        base_plan = regime
        dims = dict(regime.representative_dims)
    else:
        base_plan = active_solver.solve_regime(
            regime,
            op_name=op_name,
            topology_axes=topology_axes,
            roofline=roofline,
            problem=problem,
            input_dtypes=input_dtypes,
            has_nan_inf_cases=has_nan_inf_cases,
            has_indirect_gather=has_indirect_gather,
            has_recurrence=has_recurrence,
            uses_cube=uses_cube,
            has_persistent_state=has_persistent_state,
            has_dead_l1_slice=has_dead_l1_slice,
            quant_mode=quant_mode,
            unreserved_ubuf=unreserved_ubuf,
        )
        dims = dict(base_plan.representative_dims)

    live_ub, live_l1 = _extract_live_buffer_footprints(live_buffers_or_ledger)
    if live_ub == 0 and live_l1 == 0:
        return base_plan

    policy = dict(base_plan.policy)
    facts = active_solver.facts
    raw_mnk = policy.get("tile_mnk", [128, 128, 64])
    tm0, tn0, tk0 = max(1, int(raw_mnk[0])), max(1, int(raw_mnk[1])), max(1, int(raw_mnk[2]))
    tm, tn, tk = tm0, tn0, tk0
    rp0 = max(1, int(policy.get("row_pack_factor", 1) or 1))
    row_pack = rp0
    s0 = max(1, int(policy.get("l1_stages", 1) or 1))
    l1_stages = s0
    l1_half_split = bool(policy.get("l1_half_bank_split", False))
    l0c_buffers = max(1, int(policy.get("l0c_buffers", 1) or 1))
    vector_mode = str(policy.get("vector_mode", "simdvf"))
    ub_256kb = bool(policy.get("ub_256kb_unlock", False))

    raw_axes = policy.get("topology_axes")
    axes = TopologyAxes.from_dict(raw_axes) if isinstance(raw_axes, Mapping) else None
    is_coupled_simt = bool(
        axes is not None
        and axes.compute_engine == "coupled_1c2v"
        and axes.iteration_regularity == "irregular_simt"
    )

    def _ub_limit(unlocked_256: bool) -> int:
        if vector_mode in ("simtvf", "hybrid") or is_coupled_simt:
            return int(facts.ub_simtvf_usable_bytes)
        if unlocked_256:
            return int(facts.ub_nominal_bytes)
        return int(facts.ub_simdvf_usable_bytes)

    l1_total_limit = int(facts.l1_bytes)
    l1_half_limit = int(facts.l1_bytes // 2)
    l0c_limit = int(facts.l0c_bytes)

    base_ub = int(policy.get("ub_estimated_bytes", 0) or 0)
    base_l1 = int(policy.get("l1_estimated_bytes", 0) or 0)
    is_vector_only = str(policy.get("cv_datapath", "none")) == "none"

    def _estimate_ub(m_val: int, n_val: int, rp_val: int) -> int:
        if live_ub <= 0:
            return int(policy.get("ub_estimated_bytes", 0) or 0)
        if is_vector_only:
            scale = (max(1, m_val * n_val * rp_val)) / float(max(1, tm0 * tn0 * rp0))
        else:
            scale = (max(1, m_val * n_val)) / float(max(1, tm0 * tn0))
        return int(math.ceil(float(live_ub) * scale))

    def _estimate_l1(m_val: int, n_val: int, k_val: int, st_val: int) -> int:
        if live_l1 <= 0:
            return int(policy.get("l1_estimated_bytes", 0) or 0)
        base_tile_l1 = max(1, tm0 * tk0 + tk0 * tn0)
        cur_tile_l1 = max(1, m_val * k_val + k_val * n_val)
        scale = (float(cur_tile_l1) / float(base_tile_l1)) * (float(st_val) / float(max(1, s0)))
        return int(math.ceil(float(live_l1) * scale))

    def _half_bank_ok(l1_bytes: int, st_val: int, half_split: bool) -> bool:
        if not half_split or st_val < 2 or l1_bytes <= 0:
            return True
        per_bank = (float(l1_bytes) * math.ceil(st_val / 2.0)) / float(st_val)
        return per_bank <= float(l1_half_limit)

    ub_limit = _ub_limit(ub_256kb)
    cur_ub = _estimate_ub(tm, tn, row_pack)
    cur_l1 = _estimate_l1(tm, tn, tk, l1_stages)
    if (
        live_ub > 0
        and cur_ub > ub_limit
        and vector_mode == "simdvf"
        and not is_coupled_simt
        and not ub_256kb
        and cur_ub <= int(facts.ub_nominal_bytes)
    ):
        ub_256kb = True
        ub_limit = _ub_limit(True)

    shrank = False
    caliper_steps = 0
    caliper_action = "none"

    for _ in range(_MAX_CALIPER_STEPS):
        ub_over = live_ub > 0 and cur_ub > ub_limit
        l1_over = live_l1 > 0 and (
            cur_l1 > l1_total_limit or not _half_bank_ok(cur_l1, l1_stages, l1_half_split)
        )
        if not ub_over and not l1_over:
            break

        progressed = False
        if ub_over:
            if row_pack > 1:
                new_rp = max(1, row_pack // 2)
                if is_vector_only and tm > 1:
                    tm = max(1, (tm * new_rp) // max(1, row_pack))
                row_pack = new_rp
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "ub_halving"
            elif is_vector_only:
                if tn >= tm and tn > 32:
                    tn = max(16, (tn // 2 // 16) * 16)
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"
                elif tm > 1:
                    tm = max(1, tm // 2)
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"
                elif tn > 16:
                    tn = max(16, tn // 2)
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"
            else:
                if tm >= tn and tm > 16:
                    tm = max(16, (tm // 2 // 16) * 16) if tm >= 32 else 16
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"
                elif tn > 32:
                    tn = max(16, (tn // 2 // 16) * 16)
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"
                elif tm > 16:
                    tm = max(16, (tm // 2 // 16) * 16) if tm >= 32 else 16
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"
                elif tn > 16:
                    tn = 16
                    shrank = True
                    progressed = True
                    caliper_steps += 1
                    caliper_action = "ub_halving"

        if l1_over and not progressed:
            if l1_stages > 2:
                l1_stages = 2
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"
            elif l1_stages > 1:
                l1_stages = 1
                l1_half_split = False
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"
            elif tk > 32:
                tk = max(16, (tk // 2 // 16) * 16)
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"
            elif tk > 16:
                tk = 16
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"
            elif tm >= tn and tm > 32:
                tm = max(16, (tm // 2 // 16) * 16)
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"
            elif tn > 32:
                tn = max(16, (tn // 2 // 16) * 16)
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"
            elif tm > 16:
                tm = 16
                shrank = True
                progressed = True
                caliper_steps += 1
                caliper_action = "l1_stage_backoff"

        if not progressed:
            break
        cur_ub = _estimate_ub(tm, tn, row_pack)
        cur_l1 = _estimate_l1(tm, tn, tk, l1_stages)

    expanded = False
    if not shrank:
        max_m_cap = max(tm0, min(128, int(dims.get("M", dims.get("S_q", dims.get("rows", 128))) or 128)))
        max_n_cap = max(tn0, min(256, int(dims.get("N", dims.get("D", dims.get("cols", tn0))) or tn0)))
        max_k_cap = max(tk0, min(128, int(dims.get("K", dims.get("D_k", dims.get("D", tk0))) or tk0)))

        ub_has_headroom = live_ub > 0 and cur_ub < ub_limit and (base_ub == 0 or live_ub < base_ub)
        l1_has_headroom = live_l1 > 0 and cur_l1 < l1_total_limit and (base_l1 == 0 or live_l1 < base_l1)

        for _ in range(8):
            grew = False
            if l1_has_headroom and l1_stages == 1:
                cand_l1 = _estimate_l1(tm, tn, tk, 2)
                if cand_l1 <= l1_total_limit and _half_bank_ok(cand_l1, 2, True):
                    l1_stages = 2
                    l1_half_split = True
                    expanded = True
                    grew = True

            if not grew and l1_has_headroom and not is_vector_only and tk + 16 <= max_k_cap:
                cand_tk = tk + 16
                cand_l1 = _estimate_l1(tm, tn, cand_tk, l1_stages)
                if cand_l1 <= l1_total_limit and _half_bank_ok(cand_l1, l1_stages, l1_half_split):
                    tk = cand_tk
                    expanded = True
                    grew = True

            if not grew and ub_has_headroom:
                if is_vector_only and row_pack < 16:
                    cand_rp = min(16, row_pack * 2)
                    cand_tm = min(max_m_cap, max(tm, (tm * cand_rp) // max(1, row_pack)))
                    cand_ub = _estimate_ub(cand_tm, tn, cand_rp)
                    if cand_ub <= ub_limit:
                        row_pack = cand_rp
                        tm = cand_tm
                        expanded = True
                        grew = True
                elif not is_vector_only and tm + 16 <= max_m_cap:
                    cand_tm = tm + 16
                    cand_ub = _estimate_ub(cand_tm, tn, row_pack)
                    cand_l1 = _estimate_l1(cand_tm, tn, tk, l1_stages)
                    l0c_ok = (cand_tm * tn * 4 * l0c_buffers) <= l0c_limit
                    if (
                        cand_ub <= ub_limit
                        and cand_l1 <= l1_total_limit
                        and _half_bank_ok(cand_l1, l1_stages, l1_half_split)
                        and l0c_ok
                    ):
                        tm = cand_tm
                        expanded = True
                        grew = True
                elif not is_vector_only and tn + 16 <= max_n_cap:
                    cand_tn = tn + 16
                    cand_ub = _estimate_ub(tm, cand_tn, row_pack)
                    cand_l1 = _estimate_l1(tm, cand_tn, tk, l1_stages)
                    l0c_ok = (tm * cand_tn * 4 * l0c_buffers) <= l0c_limit
                    if (
                        cand_ub <= ub_limit
                        and cand_l1 <= l1_total_limit
                        and _half_bank_ok(cand_l1, l1_stages, l1_half_split)
                        and l0c_ok
                    ):
                        tn = cand_tn
                        expanded = True
                        grew = True

            if not grew:
                break
            cur_ub = _estimate_ub(tm, tn, row_pack)
            cur_l1 = _estimate_l1(tm, tn, tk, l1_stages)

    policy["tile_mnk"] = [tm, tn, tk]
    policy["row_pack_factor"] = row_pack
    policy["l1_stages"] = l1_stages
    policy["l1_half_bank_split"] = l1_half_split
    policy["ub_256kb_unlock"] = ub_256kb
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
        f"Fixed-point re-solved ({action_tag}, steps={caliper_steps}, "
        f"action={caliper_action}) for live buffers: "
        f"UB={cur_ub}B, L1={cur_l1}B, tile_mnk={[tm, tn, tk]}, "
        f"row_pack={row_pack}, l1_stages={l1_stages}"
    )
    return RegimePlanResult(
        regime_tag=base_plan.regime_tag,
        op_name=base_plan.op_name,
        topology_axes=base_plan.topology_axes,
        solver_mode=base_plan.solver_mode,
        policy=policy,
        representative_dims=dict(base_plan.representative_dims),
        ii=base_plan.ii,
        buffer_versions=dict(base_plan.buffer_versions),
        task_stages=dict(base_plan.task_stages),
        fallback_used=base_plan.fallback_used,
        fallback_reason=base_plan.fallback_reason,
        persistent_state_space=base_plan.persistent_state_space,
        persistent_state_bytes=base_plan.persistent_state_bytes,
        notes=(*base_plan.notes, note),
        caliper_steps=caliper_steps,
        caliper_action=caliper_action,
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
