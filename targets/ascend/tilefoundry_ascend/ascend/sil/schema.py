"""Scheduling-Interface-Layer (SIL) typed contract schemas for Ascend 950 (dav-3510)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

VALID_BACKEND_TIERS: tuple[str, ...] = (
    "tilelang_dsl",
    "ascendc_template",
)

VALID_VECTOR_MODES: tuple[str, ...] = (
    "simdvf",
    "simtvf",
    "hybrid",
)

VALID_COMPUTE_ENGINES: tuple[str, ...] = (
    "aiv_only",
    "aic_only",
    "coupled_1c2v",
)

VALID_MEMORY_RINGS: tuple[str, ...] = (
    "gm_ub_stream",
    "ub_random_atomic",
    "gm_l1_l0_fixpipe",
    "l0c_ub_epilogue",
    "bidirectional_cv_ring",
    "persistent_onchip_state",
)

VALID_ITERATION_REGULARITIES: tuple[str, ...] = (
    "affine_simd",
    "irregular_simt",
    "ragged_group",
    "dynamic_opcode_stream",
)

VALID_FREEDOM_FACETS: tuple[str, ...] = (
    "alg_compute_and_monoid",
    "tile_and_stage_knobs",
    "exec_mode_and_spr",
    "sram_layout_and_aliasing",
    "dma_and_cache_hierarchy",
    "pipeline_and_sync_graph",
    "grid_and_workspace_topology",
)


def _json_clone(value: Any) -> Any:
    """Recursively convert tuples/mappings into plain JSON-compatible Python types."""
    if isinstance(value, Mapping):
        return {str(k): _json_clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clone(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_str_tuple(raw: Any, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of strings, got {type(raw).__name__}")
    return tuple(str(item) for item in raw)


def _as_int_tuple(raw: Any, field_name: str) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of ints, got {type(raw).__name__}")
    return tuple(int(item) for item in raw)


def _as_int_dict(raw: Any, field_name: str) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(raw).__name__}")
    return {str(k): int(v) for k, v in raw.items()}


def _as_float_dict(raw: Any, field_name: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(raw).__name__}")
    return {str(k): float(v) for k, v in raw.items()}


def _as_str_dict(raw: Any, field_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(raw).__name__}")
    return {str(k): str(v) for k, v in raw.items()}


@dataclass(frozen=True)
class TopologyAxes:
    """3-axis orthogonal hardware topology specification for Ascend 950 (dav-3510)."""

    compute_engine: str
    memory_ring: str
    iteration_regularity: str

    def __post_init__(self) -> None:
        if self.compute_engine not in VALID_COMPUTE_ENGINES:
            raise ValueError(
                f"TopologyAxes.compute_engine {self.compute_engine!r} must be one of {VALID_COMPUTE_ENGINES}"
            )
        if self.memory_ring not in VALID_MEMORY_RINGS:
            raise ValueError(
                f"TopologyAxes.memory_ring {self.memory_ring!r} must be one of {VALID_MEMORY_RINGS}"
            )
        if self.iteration_regularity not in VALID_ITERATION_REGULARITIES:
            raise ValueError(
                f"TopologyAxes.iteration_regularity {self.iteration_regularity!r} must be one of {VALID_ITERATION_REGULARITIES}"
            )

    @property
    def uses_cube(self) -> bool:
        """Return whether this topology uses AIC Cube compute."""
        return self.compute_engine in ("aic_only", "coupled_1c2v")

    def axis_keys(self) -> tuple[str, str, str]:
        """Return canonical factorized axis keys (ce:*, mr:*, ir:*)."""
        return (
            f"ce:{self.compute_engine}",
            f"mr:{self.memory_ring}",
            f"ir:{self.iteration_regularity}",
        )

    def key(self) -> str:
        """Return compact canonical 3-axis coordinate key."""
        return f"{self.compute_engine}/{self.memory_ring}/{self.iteration_regularity}"

    def to_dict(self) -> dict[str, str]:
        return {
            "compute_engine": self.compute_engine,
            "memory_ring": self.memory_ring,
            "iteration_regularity": self.iteration_regularity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TopologyAxes:
        return cls(
            compute_engine=str(payload["compute_engine"]),
            memory_ring=str(payload["memory_ring"]),
            iteration_regularity=str(payload["iteration_regularity"]),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> TopologyAxes:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ConstraintRelaxationMask:
    """Provably complete MECE 7-facet target-AST partition relaxation lattice."""

    alg_compute_and_monoid: bool = True
    tile_and_stage_knobs: bool = True
    exec_mode_and_spr: bool = False
    sram_layout_and_aliasing: bool = False
    dma_and_cache_hierarchy: bool = False
    pipeline_and_sync_graph: bool = False
    grid_and_workspace_topology: bool = False

    def __post_init__(self) -> None:
        for facet in VALID_FREEDOM_FACETS:
            object.__setattr__(self, facet, bool(getattr(self, facet)))

    @classmethod
    def bottom(cls) -> ConstraintRelaxationMask:
        """Return the bottom element (T, T, F, F, F, F, F) of the 7-facet lattice."""
        return cls()

    @classmethod
    def top(cls) -> ConstraintRelaxationMask:
        """Return the top unrestricted element (T, T, T, T, T, T, T) of the 7-facet lattice."""
        return cls(
            alg_compute_and_monoid=True,
            tile_and_stage_knobs=True,
            exec_mode_and_spr=True,
            sram_layout_and_aliasing=True,
            dma_and_cache_hierarchy=True,
            pipeline_and_sync_graph=True,
            grid_and_workspace_topology=True,
        )

    @property
    def is_unrestricted_top(self) -> bool:
        """Return True iff all 7 AST partition facets are unlocked (zero shell syntax locks)."""
        return all(bool(getattr(self, facet)) for facet in VALID_FREEDOM_FACETS)

    def is_unlocked(self, facet: str) -> bool:
        """Return whether *facet* is currently unlocked."""
        if facet not in VALID_FREEDOM_FACETS:
            raise ValueError(
                f"Unknown freedom facet {facet!r}; expected one of {VALID_FREEDOM_FACETS}"
            )
        return bool(getattr(self, facet))

    def unlocked_facets(self) -> tuple[str, ...]:
        """Return ordered tuple of currently unlocked facet names."""
        return tuple(f for f in VALID_FREEDOM_FACETS if getattr(self, f))

    def locked_facets(self) -> tuple[str, ...]:
        """Return ordered tuple of currently locked facet names."""
        return tuple(f for f in VALID_FREEDOM_FACETS if not getattr(self, f))

    def unlock(self, *facets: str) -> ConstraintRelaxationMask:
        """Return a new mask with *facets* unlocked (monotonically)."""
        updates = {f: getattr(self, f) for f in VALID_FREEDOM_FACETS}
        for facet in facets:
            if facet not in VALID_FREEDOM_FACETS:
                raise ValueError(
                    f"Unknown freedom facet {facet!r}; expected one of {VALID_FREEDOM_FACETS}"
                )
            updates[facet] = True
        return ConstraintRelaxationMask(**updates)

    def join(self, other: ConstraintRelaxationMask) -> ConstraintRelaxationMask:
        """Return lattice join (elementwise OR) of self and other."""
        return ConstraintRelaxationMask(
            **{f: bool(getattr(self, f) or getattr(other, f)) for f in VALID_FREEDOM_FACETS}
        )

    def meet(self, other: ConstraintRelaxationMask) -> ConstraintRelaxationMask:
        """Return lattice meet (elementwise AND) of self and other."""
        return ConstraintRelaxationMask(
            **{f: bool(getattr(self, f) and getattr(other, f)) for f in VALID_FREEDOM_FACETS}
        )

    def to_dict(self) -> dict[str, bool]:
        """Serialize mask to plain bool dict."""
        return {f: bool(getattr(self, f)) for f in VALID_FREEDOM_FACETS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConstraintRelaxationMask:
        """Reconstruct ConstraintRelaxationMask from a mapping."""
        return cls(
            alg_compute_and_monoid=bool(payload.get("alg_compute_and_monoid", True)),
            tile_and_stage_knobs=bool(payload.get("tile_and_stage_knobs", True)),
            exec_mode_and_spr=bool(payload.get("exec_mode_and_spr", False)),
            sram_layout_and_aliasing=bool(payload.get("sram_layout_and_aliasing", False)),
            dma_and_cache_hierarchy=bool(payload.get("dma_and_cache_hierarchy", False)),
            pipeline_and_sync_graph=bool(payload.get("pipeline_and_sync_graph", False)),
            grid_and_workspace_topology=bool(payload.get("grid_and_workspace_topology", False)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> ConstraintRelaxationMask:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class RegimeSlice:
    """Shape and performance regime slice for one operator."""

    regime_tag: str
    case_nums: tuple[int, ...]
    representative_dims: dict[str, int]
    has_unaligned_tail: bool
    has_empty_group: bool = False
    ref_perf_us: float = 0.0
    t_hw_us: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.regime_tag, str) or not self.regime_tag.strip():
            raise ValueError("RegimeSlice.regime_tag must be a non-empty string")
        object.__setattr__(self, "case_nums", _as_int_tuple(self.case_nums, "case_nums"))
        object.__setattr__(
            self,
            "representative_dims",
            _as_int_dict(self.representative_dims, "representative_dims"),
        )
        object.__setattr__(self, "has_unaligned_tail", bool(self.has_unaligned_tail))
        object.__setattr__(self, "has_empty_group", bool(self.has_empty_group))
        ref_perf = float(self.ref_perf_us)
        t_hw = float(self.t_hw_us)
        if ref_perf < 0.0 or t_hw < 0.0:
            raise ValueError("RegimeSlice timings ref_perf_us and t_hw_us must be non-negative")
        object.__setattr__(self, "ref_perf_us", ref_perf)
        object.__setattr__(self, "t_hw_us", t_hw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_tag": self.regime_tag,
            "case_nums": list(self.case_nums),
            "representative_dims": dict(self.representative_dims),
            "has_unaligned_tail": self.has_unaligned_tail,
            "has_empty_group": self.has_empty_group,
            "ref_perf_us": self.ref_perf_us,
            "t_hw_us": self.t_hw_us,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RegimeSlice:
        return cls(
            regime_tag=str(payload["regime_tag"]),
            case_nums=tuple(int(x) for x in payload.get("case_nums", ())),
            representative_dims={
                str(k): int(v) for k, v in (payload.get("representative_dims") or {}).items()
            },
            has_unaligned_tail=bool(payload["has_unaligned_tail"]),
            has_empty_group=bool(payload.get("has_empty_group", False)),
            ref_perf_us=float(payload.get("ref_perf_us", 0.0)),
            t_hw_us=float(payload.get("t_hw_us", 0.0)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> RegimeSlice:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class HardwareLegalityBounds:
    """Ascend 950 (dav-3510) hardware capacity and architectural redlines."""

    l1_bytes: int = 524288
    l1_half_bank_bytes: int = 262144
    l0a_bytes: int = 65536
    l0b_bytes: int = 65536
    l0c_bytes: int = 262144
    ub_simdvf_bytes: int = 253952
    ub_unreserved_bytes: int = 262144
    ub_simtvf_bytes: int = 221184
    ssbuf_bytes: int = 3072
    max_mmad_mnk: int = 4095
    max_crosscore_depth: int = 15
    hardware_mutex_count: int = 28
    ub_banks: int = 16
    ub_datablock_bytes: int = 32
    forbidden_dma_paths: tuple[str, ...] = ("L1->GM", "GM->L0A", "GM->L0B")
    forbid_cube_int4: bool = True

    def __post_init__(self) -> None:
        int_fields = (
            "l1_bytes",
            "l1_half_bank_bytes",
            "l0a_bytes",
            "l0b_bytes",
            "l0c_bytes",
            "ub_simdvf_bytes",
            "ub_unreserved_bytes",
            "ub_simtvf_bytes",
            "ssbuf_bytes",
            "max_mmad_mnk",
            "max_crosscore_depth",
            "hardware_mutex_count",
            "ub_banks",
            "ub_datablock_bytes",
        )
        for name in int_fields:
            val = int(getattr(self, name))
            if val <= 0:
                raise ValueError(f"HardwareLegalityBounds.{name} must be > 0, got {val}")
            object.__setattr__(self, name, val)
        if self.l1_half_bank_bytes > self.l1_bytes:
            raise ValueError("l1_half_bank_bytes cannot exceed l1_bytes")
        if self.ub_simtvf_bytes > self.ub_simdvf_bytes:
            raise ValueError("ub_simtvf_bytes cannot exceed ub_simdvf_bytes")
        if self.ub_simdvf_bytes > self.ub_unreserved_bytes:
            raise ValueError("ub_simdvf_bytes cannot exceed ub_unreserved_bytes")
        object.__setattr__(
            self,
            "forbidden_dma_paths",
            _as_str_tuple(self.forbidden_dma_paths, "forbidden_dma_paths"),
        )
        object.__setattr__(self, "forbid_cube_int4", bool(self.forbid_cube_int4))

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1_bytes": self.l1_bytes,
            "l1_half_bank_bytes": self.l1_half_bank_bytes,
            "l0a_bytes": self.l0a_bytes,
            "l0b_bytes": self.l0b_bytes,
            "l0c_bytes": self.l0c_bytes,
            "ub_simdvf_bytes": self.ub_simdvf_bytes,
            "ub_unreserved_bytes": self.ub_unreserved_bytes,
            "ub_simtvf_bytes": self.ub_simtvf_bytes,
            "ssbuf_bytes": self.ssbuf_bytes,
            "max_mmad_mnk": self.max_mmad_mnk,
            "max_crosscore_depth": self.max_crosscore_depth,
            "hardware_mutex_count": self.hardware_mutex_count,
            "ub_banks": self.ub_banks,
            "ub_datablock_bytes": self.ub_datablock_bytes,
            "forbidden_dma_paths": list(self.forbidden_dma_paths),
            "forbid_cube_int4": self.forbid_cube_int4,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HardwareLegalityBounds:
        return cls(
            l1_bytes=int(payload.get("l1_bytes", 524288)),
            l1_half_bank_bytes=int(payload.get("l1_half_bank_bytes", 262144)),
            l0a_bytes=int(payload.get("l0a_bytes", 65536)),
            l0b_bytes=int(payload.get("l0b_bytes", 65536)),
            l0c_bytes=int(payload.get("l0c_bytes", 262144)),
            ub_simdvf_bytes=int(payload.get("ub_simdvf_bytes", 253952)),
            ub_unreserved_bytes=int(payload.get("ub_unreserved_bytes", 262144)),
            ub_simtvf_bytes=int(payload.get("ub_simtvf_bytes", 221184)),
            ssbuf_bytes=int(payload.get("ssbuf_bytes", 3072)),
            max_mmad_mnk=int(payload.get("max_mmad_mnk", 4095)),
            max_crosscore_depth=int(payload.get("max_crosscore_depth", 15)),
            hardware_mutex_count=int(payload.get("hardware_mutex_count", 28)),
            ub_banks=int(payload.get("ub_banks", 16)),
            ub_datablock_bytes=int(payload.get("ub_datablock_bytes", 32)),
            forbidden_dma_paths=tuple(
                str(x)
                for x in payload.get(
                    "forbidden_dma_paths", ("L1->GM", "GM->L0A", "GM->L0B")
                )
            ),
            forbid_cube_int4=bool(payload.get("forbid_cube_int4", True)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> HardwareLegalityBounds:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class AdvisoryEnvelope:
    """Compiler & static analysis advisory envelope for one operator."""

    op_name: str
    cannbot_task: str
    level: int
    topology_axes: TopologyAxes
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    roofline: dict[str, object]
    structural_features: dict[str, object]
    regimes: tuple[RegimeSlice, ...]
    legality_bounds: HardwareLegalityBounds = field(default_factory=HardwareLegalityBounds)
    has_nan_inf_cases: bool = False
    has_inplace_aliasing: bool = False
    precision_thresholds: dict[str, float] = field(default_factory=dict)
    hir_blueprint: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.op_name, str) or not self.op_name.strip():
            raise ValueError("AdvisoryEnvelope.op_name must be a non-empty string")
        if not isinstance(self.cannbot_task, str) or not self.cannbot_task.strip():
            raise ValueError("AdvisoryEnvelope.cannbot_task must be a non-empty string")
        level = int(self.level)
        if not (1 <= level <= 5):
            raise ValueError(f"AdvisoryEnvelope.level must be in 1..5, got {level}")
        object.__setattr__(self, "level", level)
        if isinstance(self.topology_axes, Mapping):
            object.__setattr__(self, "topology_axes", TopologyAxes.from_dict(self.topology_axes))
        elif not isinstance(self.topology_axes, TopologyAxes):
            raise TypeError(
                f"AdvisoryEnvelope.topology_axes must be TopologyAxes or Mapping, got {type(self.topology_axes).__name__}"
            )
        object.__setattr__(self, "input_dtypes", _as_str_tuple(self.input_dtypes, "input_dtypes"))
        object.__setattr__(self, "output_dtypes", _as_str_tuple(self.output_dtypes, "output_dtypes"))
        if not isinstance(self.roofline, Mapping):
            raise ValueError("AdvisoryEnvelope.roofline must be a mapping")
        object.__setattr__(self, "roofline", _json_clone(self.roofline))
        if not isinstance(self.structural_features, Mapping):
            raise ValueError("AdvisoryEnvelope.structural_features must be a mapping")
        object.__setattr__(self, "structural_features", _json_clone(self.structural_features))
        regimes_tuple = tuple(self.regimes)
        if not regimes_tuple:
            raise ValueError("AdvisoryEnvelope.regimes must be non-empty")
        for item in regimes_tuple:
            if not isinstance(item, RegimeSlice):
                raise TypeError(f"Expected RegimeSlice in regimes, got {type(item).__name__}")
        object.__setattr__(self, "regimes", regimes_tuple)
        if not isinstance(self.legality_bounds, HardwareLegalityBounds):
            raise TypeError("legality_bounds must be HardwareLegalityBounds")
        object.__setattr__(self, "has_nan_inf_cases", bool(self.has_nan_inf_cases))
        object.__setattr__(self, "has_inplace_aliasing", bool(self.has_inplace_aliasing))
        object.__setattr__(
            self,
            "precision_thresholds",
            _as_float_dict(self.precision_thresholds, "precision_thresholds"),
        )
        if self.hir_blueprint is not None:
            if not isinstance(self.hir_blueprint, Mapping):
                raise ValueError("hir_blueprint must be a mapping or None")
            object.__setattr__(
                self, "hir_blueprint", _json_clone(self.hir_blueprint)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "cannbot_task": self.cannbot_task,
            "level": self.level,
            "topology_axes": self.topology_axes.to_dict(),
            "input_dtypes": list(self.input_dtypes),
            "output_dtypes": list(self.output_dtypes),
            "roofline": _json_clone(self.roofline),
            "structural_features": _json_clone(self.structural_features),
            "regimes": [r.to_dict() for r in self.regimes],
            "legality_bounds": self.legality_bounds.to_dict(),
            "has_nan_inf_cases": self.has_nan_inf_cases,
            "has_inplace_aliasing": self.has_inplace_aliasing,
            "precision_thresholds": dict(self.precision_thresholds),
            "hir_blueprint": (
                _json_clone(self.hir_blueprint)
                if self.hir_blueprint is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdvisoryEnvelope:
        raw_bounds = payload.get("legality_bounds")
        bounds = (
            HardwareLegalityBounds.from_dict(raw_bounds)
            if isinstance(raw_bounds, Mapping)
            else HardwareLegalityBounds()
        )
        raw_packet = payload.get("hir_blueprint")
        raw_axes = payload["topology_axes"]
        axes = (
            raw_axes
            if isinstance(raw_axes, TopologyAxes)
            else TopologyAxes.from_dict(raw_axes)
        )
        return cls(
            op_name=str(payload["op_name"]),
            cannbot_task=str(payload["cannbot_task"]),
            level=int(payload["level"]),
            topology_axes=axes,
            input_dtypes=tuple(str(x) for x in payload.get("input_dtypes", ())),
            output_dtypes=tuple(str(x) for x in payload.get("output_dtypes", ())),
            roofline=dict(payload.get("roofline") or {}),
            structural_features=dict(payload.get("structural_features") or {}),
            regimes=tuple(RegimeSlice.from_dict(r) for r in (payload.get("regimes") or ())),
            legality_bounds=bounds,
            has_nan_inf_cases=bool(payload.get("has_nan_inf_cases", False)),
            has_inplace_aliasing=bool(payload.get("has_inplace_aliasing", False)),
            precision_thresholds={
                str(k): float(v)
                for k, v in (payload.get("precision_thresholds") or {}).items()
            },
            hir_blueprint=dict(raw_packet) if isinstance(raw_packet, Mapping) else None,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> AdvisoryEnvelope:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class SchedulePolicy:
    """Per-regime tactical schedule policy bridging solver, templates, and verifier."""

    regime_tag: str
    topology_axes: TopologyAxes
    backend_tier: str = "tilelang_dsl"
    tile_mnk: tuple[int, int, int] = (128, 128, 64)
    row_pack_factor: int = 1
    num_kv_splits: int = 1
    l1_stages: int = 2
    l1_half_bank_split: bool = True
    l1_slot_aliasing: dict[str, str] = field(default_factory=dict)
    l0c_buffers: int = 1
    ub_256kb_unlock: bool = False
    ub_estimated_bytes: int = 0
    l1_estimated_bytes: int = 0
    ub_bank_pad_datablocks: int = 0
    ub_rw_shift_datablocks: int = 0
    ub_row_stride_bytes: int = 0
    cv_datapath: str = "none"
    sync_mode: str = "mode4_flag"
    preload_depth: int = 2
    reverse_ack_period: int = 14
    ensure_empty_subblock_sync: bool = True
    swizzle_window: int = 4
    tail_wave_split: bool = True
    streamk_mode: str = "none"
    l2_bypass_operands: tuple[str, ...] = ()
    vector_mode: str = "simdvf"
    direct_no_fusion_vf: bool = False
    multi_row_unroll: int = 1
    reduce_accumulators: int = 1
    simt_threads: int = 512
    hf32_allowed: bool = False
    fast_unshifted_exp: bool = False
    w4_mode: str = "none"
    dma_paths_used: tuple[str, ...] = ()
    cube_operand_dtype: str = ""
    freedom_mask: ConstraintRelaxationMask = field(default_factory=ConstraintRelaxationMask.bottom)

    def __post_init__(self) -> None:
        if not isinstance(self.regime_tag, str) or not self.regime_tag.strip():
            raise ValueError("SchedulePolicy.regime_tag must be a non-empty string")
        if isinstance(self.topology_axes, Mapping):
            object.__setattr__(self, "topology_axes", TopologyAxes.from_dict(self.topology_axes))
        elif not isinstance(self.topology_axes, TopologyAxes):
            raise TypeError(
                f"SchedulePolicy.topology_axes must be TopologyAxes or Mapping, got {type(self.topology_axes).__name__}"
            )
        if self.backend_tier not in VALID_BACKEND_TIERS:
            raise ValueError(
                f"SchedulePolicy.backend_tier {self.backend_tier!r} must be one of {VALID_BACKEND_TIERS}"
            )
        if self.vector_mode not in VALID_VECTOR_MODES:
            raise ValueError(
                f"SchedulePolicy.vector_mode {self.vector_mode!r} must be one of {VALID_VECTOR_MODES}"
            )
        tile = _as_int_tuple(self.tile_mnk, "tile_mnk")
        if len(tile) != 3 or any(d <= 0 for d in tile):
            raise ValueError(f"SchedulePolicy.tile_mnk must be 3 positive ints, got {tile}")
        object.__setattr__(self, "tile_mnk", tile)

        for name in (
            "row_pack_factor",
            "num_kv_splits",
            "l1_stages",
            "l0c_buffers",
            "swizzle_window",
            "multi_row_unroll",
            "reduce_accumulators",
        ):
            val = int(getattr(self, name))
            if val < 1:
                raise ValueError(f"SchedulePolicy.{name} must be >= 1, got {val}")
            object.__setattr__(self, name, val)

        for name in (
            "ub_estimated_bytes",
            "l1_estimated_bytes",
            "ub_bank_pad_datablocks",
            "ub_rw_shift_datablocks",
            "ub_row_stride_bytes",
            "preload_depth",
            "reverse_ack_period",
            "simt_threads",
        ):
            val = int(getattr(self, name))
            if val < 0:
                raise ValueError(f"SchedulePolicy.{name} must be >= 0, got {val}")
            object.__setattr__(self, name, val)

        object.__setattr__(self, "l1_half_bank_split", bool(self.l1_half_bank_split))
        object.__setattr__(
            self, "l1_slot_aliasing", _as_str_dict(self.l1_slot_aliasing, "l1_slot_aliasing")
        )
        object.__setattr__(self, "ub_256kb_unlock", bool(self.ub_256kb_unlock))
        object.__setattr__(self, "cv_datapath", str(self.cv_datapath))
        object.__setattr__(self, "sync_mode", str(self.sync_mode))
        object.__setattr__(
            self, "ensure_empty_subblock_sync", bool(self.ensure_empty_subblock_sync)
        )
        object.__setattr__(self, "tail_wave_split", bool(self.tail_wave_split))
        object.__setattr__(self, "streamk_mode", str(self.streamk_mode))
        object.__setattr__(
            self,
            "l2_bypass_operands",
            _as_str_tuple(self.l2_bypass_operands, "l2_bypass_operands"),
        )
        object.__setattr__(self, "direct_no_fusion_vf", bool(self.direct_no_fusion_vf))
        object.__setattr__(self, "hf32_allowed", bool(self.hf32_allowed))
        object.__setattr__(self, "fast_unshifted_exp", bool(self.fast_unshifted_exp))
        object.__setattr__(self, "w4_mode", str(self.w4_mode))
        object.__setattr__(
            self, "dma_paths_used", _as_str_tuple(self.dma_paths_used, "dma_paths_used")
        )
        object.__setattr__(self, "cube_operand_dtype", str(self.cube_operand_dtype))
        if isinstance(self.freedom_mask, Mapping):
            object.__setattr__(
                self, "freedom_mask", ConstraintRelaxationMask.from_dict(self.freedom_mask)
            )
        elif not isinstance(self.freedom_mask, ConstraintRelaxationMask):
            raise TypeError(
                f"SchedulePolicy.freedom_mask must be ConstraintRelaxationMask or Mapping, got {type(self.freedom_mask).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_tag": self.regime_tag,
            "topology_axes": self.topology_axes.to_dict(),
            "backend_tier": self.backend_tier,
            "tile_mnk": list(self.tile_mnk),
            "row_pack_factor": self.row_pack_factor,
            "num_kv_splits": self.num_kv_splits,
            "l1_stages": self.l1_stages,
            "l1_half_bank_split": self.l1_half_bank_split,
            "l1_slot_aliasing": dict(self.l1_slot_aliasing),
            "l0c_buffers": self.l0c_buffers,
            "ub_256kb_unlock": self.ub_256kb_unlock,
            "ub_estimated_bytes": self.ub_estimated_bytes,
            "l1_estimated_bytes": self.l1_estimated_bytes,
            "ub_bank_pad_datablocks": self.ub_bank_pad_datablocks,
            "ub_rw_shift_datablocks": self.ub_rw_shift_datablocks,
            "ub_row_stride_bytes": self.ub_row_stride_bytes,
            "cv_datapath": self.cv_datapath,
            "sync_mode": self.sync_mode,
            "preload_depth": self.preload_depth,
            "reverse_ack_period": self.reverse_ack_period,
            "ensure_empty_subblock_sync": self.ensure_empty_subblock_sync,
            "swizzle_window": self.swizzle_window,
            "tail_wave_split": self.tail_wave_split,
            "streamk_mode": self.streamk_mode,
            "l2_bypass_operands": list(self.l2_bypass_operands),
            "vector_mode": self.vector_mode,
            "direct_no_fusion_vf": self.direct_no_fusion_vf,
            "multi_row_unroll": self.multi_row_unroll,
            "reduce_accumulators": self.reduce_accumulators,
            "simt_threads": self.simt_threads,
            "hf32_allowed": self.hf32_allowed,
            "fast_unshifted_exp": self.fast_unshifted_exp,
            "w4_mode": self.w4_mode,
            "dma_paths_used": list(self.dma_paths_used),
            "cube_operand_dtype": self.cube_operand_dtype,
            "freedom_mask": self.freedom_mask.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SchedulePolicy:
        raw_tile = payload.get("tile_mnk", (128, 128, 64))
        tile_mnk = (int(raw_tile[0]), int(raw_tile[1]), int(raw_tile[2]))
        raw_axes = payload["topology_axes"]
        axes = (
            raw_axes
            if isinstance(raw_axes, TopologyAxes)
            else TopologyAxes.from_dict(raw_axes)
        )
        raw_mask = payload.get("freedom_mask")
        mask = (
            raw_mask
            if isinstance(raw_mask, ConstraintRelaxationMask)
            else (
                ConstraintRelaxationMask.from_dict(raw_mask)
                if isinstance(raw_mask, Mapping)
                else ConstraintRelaxationMask.bottom()
            )
        )
        return cls(
            regime_tag=str(payload["regime_tag"]),
            topology_axes=axes,
            backend_tier=str(payload.get("backend_tier", "tilelang_dsl")),
            tile_mnk=tile_mnk,
            row_pack_factor=int(payload.get("row_pack_factor", 1)),
            num_kv_splits=int(payload.get("num_kv_splits", 1)),
            l1_stages=int(payload.get("l1_stages", 2)),
            l1_half_bank_split=bool(payload.get("l1_half_bank_split", True)),
            l1_slot_aliasing={
                str(k): str(v) for k, v in (payload.get("l1_slot_aliasing") or {}).items()
            },
            l0c_buffers=int(payload.get("l0c_buffers", 1)),
            ub_256kb_unlock=bool(payload.get("ub_256kb_unlock", False)),
            ub_estimated_bytes=int(payload.get("ub_estimated_bytes", 0)),
            l1_estimated_bytes=int(payload.get("l1_estimated_bytes", 0)),
            ub_bank_pad_datablocks=int(payload.get("ub_bank_pad_datablocks", 0)),
            ub_rw_shift_datablocks=int(payload.get("ub_rw_shift_datablocks", 0)),
            ub_row_stride_bytes=int(payload.get("ub_row_stride_bytes", 0)),
            cv_datapath=str(payload.get("cv_datapath", "none")),
            sync_mode=str(payload.get("sync_mode", "mode4_flag")),
            preload_depth=int(payload.get("preload_depth", 2)),
            reverse_ack_period=int(payload.get("reverse_ack_period", 14)),
            ensure_empty_subblock_sync=bool(payload.get("ensure_empty_subblock_sync", True)),
            swizzle_window=int(payload.get("swizzle_window", 4)),
            tail_wave_split=bool(payload.get("tail_wave_split", True)),
            streamk_mode=str(payload.get("streamk_mode", "none")),
            l2_bypass_operands=tuple(
                str(x) for x in payload.get("l2_bypass_operands", ())
            ),
            vector_mode=str(payload.get("vector_mode", "simdvf")),
            direct_no_fusion_vf=bool(payload.get("direct_no_fusion_vf", False)),
            multi_row_unroll=int(payload.get("multi_row_unroll", 1)),
            reduce_accumulators=int(payload.get("reduce_accumulators", 1)),
            simt_threads=int(payload.get("simt_threads", 512)),
            hf32_allowed=bool(payload.get("hf32_allowed", False)),
            fast_unshifted_exp=bool(payload.get("fast_unshifted_exp", False)),
            w4_mode=str(payload.get("w4_mode", "none")),
            dma_paths_used=tuple(str(x) for x in payload.get("dma_paths_used", ())),
            cube_operand_dtype=str(payload.get("cube_operand_dtype", "")),
            freedom_mask=mask,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> SchedulePolicy:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class PolicyEnvelope:
    """Operator-wide collection of per-regime schedule policies."""

    op_name: str
    policies: dict[str, SchedulePolicy]
    selected_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.op_name, str) or not self.op_name.strip():
            raise ValueError("PolicyEnvelope.op_name must be a non-empty string")
        if not isinstance(self.policies, Mapping) or not self.policies:
            raise ValueError("PolicyEnvelope.policies must be a non-empty mapping")
        normalized: dict[str, SchedulePolicy] = {}
        for key, policy in self.policies.items():
            if not isinstance(policy, SchedulePolicy):
                raise TypeError(
                    f"PolicyEnvelope.policies[{key!r}] must be SchedulePolicy, got {type(policy).__name__}"
                )
            tag = str(key)
            if tag != policy.regime_tag:
                raise ValueError(
                    f"Policy key {tag!r} does not match policy.regime_tag {policy.regime_tag!r}"
                )
            normalized[tag] = policy
        object.__setattr__(self, "policies", normalized)
        object.__setattr__(
            self, "selected_rules", _as_str_tuple(self.selected_rules, "selected_rules")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
            "selected_rules": list(self.selected_rules),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PolicyEnvelope:
        raw_policies = payload.get("policies") or {}
        policies = {
            str(k): SchedulePolicy.from_dict(v) for k, v in raw_policies.items()
        }
        return cls(
            op_name=str(payload["op_name"]),
            policies=policies,
            selected_rules=tuple(str(x) for x in payload.get("selected_rules", ())),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> PolicyEnvelope:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class VerificationReport:
    """Result of deterministic SIL hardware legality verification."""

    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    suggested_fixes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "errors", _as_str_tuple(self.errors, "errors"))
        object.__setattr__(self, "warnings", _as_str_tuple(self.warnings, "warnings"))
        if not isinstance(self.suggested_fixes, Mapping):
            raise ValueError("VerificationReport.suggested_fixes must be a mapping")
        object.__setattr__(self, "suggested_fixes", _json_clone(self.suggested_fixes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "suggested_fixes": _json_clone(self.suggested_fixes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationReport:
        return cls(
            passed=bool(payload["passed"]),
            errors=tuple(str(x) for x in payload.get("errors", ())),
            warnings=tuple(str(x) for x in payload.get("warnings", ())),
            suggested_fixes=dict(payload.get("suggested_fixes") or {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> VerificationReport:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class TelemetryFeedback:
    """Closed-loop compile/profile telemetry from Stage A/B/C and CANNBench."""

    stage_a_passed: bool
    stage_b_passed: bool
    stage_c_passed: bool = False
    spill_bytes: int = 0
    vloopv2_hit: bool = True
    measured_latency_us: float = 0.0
    eq3_score: float = 0.0
    hap_reward: float = 0.0
    pipe_utilization: dict[str, float] = field(default_factory=dict)
    bank_conflict_ratio: float = 0.0
    l2_hit_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_a_passed", bool(self.stage_a_passed))
        object.__setattr__(self, "stage_b_passed", bool(self.stage_b_passed))
        object.__setattr__(self, "stage_c_passed", bool(self.stage_c_passed))
        spill = int(self.spill_bytes)
        if spill < 0:
            raise ValueError(f"TelemetryFeedback.spill_bytes must be >= 0, got {spill}")
        object.__setattr__(self, "spill_bytes", spill)
        object.__setattr__(self, "vloopv2_hit", bool(self.vloopv2_hit))
        object.__setattr__(self, "measured_latency_us", float(self.measured_latency_us))
        object.__setattr__(self, "eq3_score", float(self.eq3_score))
        object.__setattr__(self, "hap_reward", float(self.hap_reward))
        object.__setattr__(
            self,
            "pipe_utilization",
            _as_float_dict(self.pipe_utilization, "pipe_utilization"),
        )
        object.__setattr__(self, "bank_conflict_ratio", float(self.bank_conflict_ratio))
        object.__setattr__(self, "l2_hit_rate", float(self.l2_hit_rate))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_a_passed": self.stage_a_passed,
            "stage_b_passed": self.stage_b_passed,
            "stage_c_passed": self.stage_c_passed,
            "spill_bytes": self.spill_bytes,
            "vloopv2_hit": self.vloopv2_hit,
            "measured_latency_us": self.measured_latency_us,
            "eq3_score": self.eq3_score,
            "hap_reward": self.hap_reward,
            "pipe_utilization": dict(self.pipe_utilization),
            "bank_conflict_ratio": self.bank_conflict_ratio,
            "l2_hit_rate": self.l2_hit_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TelemetryFeedback:
        return cls(
            stage_a_passed=bool(payload["stage_a_passed"]),
            stage_b_passed=bool(payload["stage_b_passed"]),
            stage_c_passed=bool(payload.get("stage_c_passed", False)),
            spill_bytes=int(payload.get("spill_bytes", 0)),
            vloopv2_hit=bool(payload.get("vloopv2_hit", True)),
            measured_latency_us=float(payload.get("measured_latency_us", 0.0)),
            eq3_score=float(payload.get("eq3_score", 0.0)),
            hap_reward=float(payload.get("hap_reward", 0.0)),
            pipe_utilization={
                str(k): float(v)
                for k, v in (payload.get("pipe_utilization") or {}).items()
            },
            bank_conflict_ratio=float(payload.get("bank_conflict_ratio", 0.0)),
            l2_hit_rate=float(payload.get("l2_hit_rate", 0.0)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> TelemetryFeedback:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class SchedulingEnvelope:
    """Top-level SIL artifact bundling advisory, policy, verification, and telemetry."""

    advisory: AdvisoryEnvelope
    policy: PolicyEnvelope
    verification: VerificationReport | None = None
    telemetry: TelemetryFeedback | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.advisory, AdvisoryEnvelope):
            raise TypeError("advisory must be AdvisoryEnvelope")
        if not isinstance(self.policy, PolicyEnvelope):
            raise TypeError("policy must be PolicyEnvelope")
        if self.verification is not None and not isinstance(
            self.verification, VerificationReport
        ):
            raise TypeError("verification must be VerificationReport or None")
        if self.telemetry is not None and not isinstance(self.telemetry, TelemetryFeedback):
            raise TypeError("telemetry must be TelemetryFeedback or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "advisory": self.advisory.to_dict(),
            "policy": self.policy.to_dict(),
            "verification": (
                self.verification.to_dict() if self.verification is not None else None
            ),
            "telemetry": self.telemetry.to_dict() if self.telemetry is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SchedulingEnvelope:
        raw_verif = payload.get("verification")
        raw_telem = payload.get("telemetry")
        return cls(
            advisory=AdvisoryEnvelope.from_dict(payload["advisory"]),
            policy=PolicyEnvelope.from_dict(payload["policy"]),
            verification=(
                VerificationReport.from_dict(raw_verif)
                if isinstance(raw_verif, Mapping)
                else None
            ),
            telemetry=(
                TelemetryFeedback.from_dict(raw_telem)
                if isinstance(raw_telem, Mapping)
                else None
            ),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> SchedulingEnvelope:
        return cls.from_dict(json.loads(raw))

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> SchedulingEnvelope:
        target = Path(path)
        return cls.from_json(target.read_text(encoding="utf-8"))
