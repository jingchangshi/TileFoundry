"""Scheduling-Interface-Layer (SIL) typed contract schemas for Ascend 950 (dav-3510)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tilefoundry_ascend.ascend.target.facts import AscendSpaceFacts, cube_macs_per_cycle

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


CasePhysicsKind = Literal["reduction", "matmul", "attention", "elementwise"]
LayoutPhase = Literal["contiguous", "strided_or_gather"]
ResidencyBand = Literal["vreg_resident", "sram_1pass", "gm_multipass"]
AlignmentBand = Literal["sub_atom_simt", "aligned_vector", "unaligned_predicated"]
WaveBand = Literal["under_occupied_split", "balanced_waves", "ragged_groups"]
EnginePhase = Literal["aiv_vector_or_gemv", "aic_cube", "coupled_1c2v"]
PhaseSignature = tuple[EnginePhase, LayoutPhase, ResidencyBand, AlignmentBand, WaveBand]

VREG_CAPACITY_BYTES: int = 4096

DTYPE_BYTES: dict[str, float] = {
    "bool": 1.0,
    "uint8": 1.0,
    "u8": 1.0,
    "int8": 1.0,
    "i8": 1.0,
    "float8_e4m3fn": 1.0,
    "float8_e5m2": 1.0,
    "float8_e8m0fnu": 1.0,
    "fp8": 1.0,
    "fp8e4m3": 1.0,
    "fp8e5m2": 1.0,
    "f8e4m3": 1.0,
    "f8e5m2": 1.0,
    "hif8": 1.0,
    "float4_e2m1": 0.5,
    "float4_e1m2": 0.5,
    "float4_e2m1fn": 0.5,
    "float4_e2m1fnx2": 0.5,
    "fp4": 0.5,
    "f4e2m1": 0.5,
    "f4e1m2": 0.5,
    "int16": 2.0,
    "float16": 2.0,
    "half": 2.0,
    "f16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int32": 4.0,
    "float32": 4.0,
    "float": 4.0,
    "f32": 4.0,
    "int64": 8.0,
    "float64": 8.0,
}

_CUBE_DTYPE_CANON_AND_BYTES: dict[str, tuple[str, float]] = {
    "f4e2m1": ("f4e2m1", 0.5),
    "float4_e2m1": ("f4e2m1", 0.5),
    "float4_e2m1fn": ("f4e2m1", 0.5),
    "float4_e2m1fnx2": ("f4e2m1", 0.5),
    "fp4e2m1": ("f4e2m1", 0.5),
    "fp4": ("f4e2m1", 0.5),
    "f4": ("f4e2m1", 0.5),
    "f4e1m2": ("f4e1m2", 0.5),
    "float4_e1m2": ("f4e1m2", 0.5),
    "fp4e1m2": ("f4e1m2", 0.5),
    "fp8e4m3": ("fp8e4m3", 1.0),
    "float8_e4m3fn": ("fp8e4m3", 1.0),
    "fp8_e4m3": ("fp8e4m3", 1.0),
    "f8e4m3": ("fp8e4m3", 1.0),
    "fp8": ("fp8e4m3", 1.0),
    "fp8e5m2": ("fp8e5m2", 1.0),
    "float8_e5m2": ("fp8e5m2", 1.0),
    "fp8_e5m2": ("fp8e5m2", 1.0),
    "f8e5m2": ("fp8e5m2", 1.0),
    "float8_e8m0fnu": ("fp8e4m3", 1.0),
    "f8e8m0": ("fp8e4m3", 1.0),
    "hif8": ("hif8", 1.0),
    "i8": ("i8", 1.0),
    "int8": ("i8", 1.0),
    "u8": ("u8", 1.0),
    "uint8": ("u8", 1.0),
    "f16": ("f16", 2.0),
    "float16": ("f16", 2.0),
    "half": ("f16", 2.0),
    "bf16": ("bf16", 2.0),
    "bfloat16": ("bf16", 2.0),
    "hf32": ("hf32", 4.0),
    "f32": ("f32", 4.0),
    "float32": ("f32", 4.0),
    "float": ("f32", 4.0),
}


def dtype_bytes_for(dtype_name: str) -> float:
    """Return physical storage bytes per element for a known dtype name."""
    key = dtype_name.strip().lower()
    if key not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype_name!r} for physics ratio computation")
    return DTYPE_BYTES[key]


def shape_numel(shape: Sequence[int]) -> int:
    """Return the positive element count of a shape, treating non-positive axes as one."""
    numel = 1
    for dim in shape:
        numel *= max(int(dim), 1)
    return numel


def _is_shape_dim(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def flatten_int_shapes(raw: object) -> tuple[tuple[int, ...], ...]:
    """Normalize a cann-bench input_shape payload into concrete integer shape tuples."""
    shapes: list[tuple[int, ...]] = []
    if not isinstance(raw, (list, tuple)):
        return ()
    for entry in raw:
        if not isinstance(entry, (list, tuple)):
            continue
        if entry and isinstance(entry[0], (list, tuple)):
            for sub in entry:
                if isinstance(sub, (list, tuple)) and all(_is_shape_dim(dim) for dim in sub):
                    shapes.append(tuple(int(dim) for dim in sub))
        elif all(_is_shape_dim(dim) for dim in entry):
            shapes.append(tuple(int(dim) for dim in entry))
    return tuple(shapes)


def flatten_str_values(raw: object) -> tuple[str, ...]:
    """Normalize a nested dtype or tag payload into a flat tuple of strings."""
    values: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            values.extend(flatten_str_values(item))
    elif raw is not None:
        values.append(str(raw))
    return tuple(values)


def has_unaligned_tail_for_case(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str],
    uses_cube: bool,
) -> bool:
    """Return whether any tensor tail or cube operand violates 32-byte or 16-element alignment."""
    if not shapes:
        return False
    first_dtype = dtypes[0] if dtypes else "float16"
    elem_bytes = max(dtype_bytes_for(first_dtype), 1.0)
    align_elems = max(1, int(32 // elem_bytes))
    cube_align = 16 if uses_cube else align_elems
    for shape in shapes:
        if not shape:
            continue
        last = shape[-1]
        if last > 1 and last % align_elems != 0:
            return True
        if uses_cube:
            for dim in shape[-2:]:
                if dim > 1 and dim % cube_align != 0:
                    return True
        if len(shape) >= 2 and shape[0] > 15 and shape[0] % 2 == 1:
            return True
    return False


def compute_critical_cube_tile_m(
    facts: AscendSpaceFacts | None,
    elem_dtype: str,
    *,
    l2_bytes_per_cycle_per_aic: float = 100.0,
    dual_operand_miss: bool = False,
) -> int:
    """Compute the analytical Cube tile-M waterline from MAC rate, dtype width and L2 bandwidth."""
    bandwidth = float(l2_bytes_per_cycle_per_aic)
    if bandwidth <= 0.0:
        raise ValueError(
            f"l2_bytes_per_cycle_per_aic must be > 0, got {l2_bytes_per_cycle_per_aic}"
        )
    key = elem_dtype.strip().lower()
    canon_dtype, dtype_bytes = _CUBE_DTYPE_CANON_AND_BYTES.get(key, ("f16", 2.0))
    macs = (
        facts.cube_macs_per_cycle(canon_dtype)
        if facts is not None
        else cube_macs_per_cycle(canon_dtype)
    )
    miss_factor = 2.0 if dual_operand_miss else 1.0
    return max(1, int(math.ceil(float(macs) * dtype_bytes * miss_factor / bandwidth)))


def _payload_float(payload: Mapping[str, object], key: str) -> float:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"CasePhysicsRatios.{key} must be numeric")
    return float(raw)


def _payload_int(payload: Mapping[str, object], key: str) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"CasePhysicsRatios.{key} must be an int")
    return raw


def _payload_bool(payload: Mapping[str, object], key: str) -> bool:
    raw = payload.get(key)
    if not isinstance(raw, bool):
        raise ValueError(f"CasePhysicsRatios.{key} must be a bool")
    return raw


@dataclass(frozen=True, kw_only=True)
class CasePhysicsRatios:
    """Dimensionless physical ratios and derived phase bands for one operator case."""

    case_id: int
    stride_ratio: float
    contig_atom_ratio: float
    is_atom_aligned: bool
    working_set_sram_ratio: float
    working_set_vreg_ratio: float
    wave_occupancy_ratio: float
    oi_ridge_ratio: float
    inner_contiguous_bytes: int
    has_empty_group: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int):
            raise ValueError("CasePhysicsRatios.case_id must be an int")
        if self.case_id < 1:
            raise ValueError("CasePhysicsRatios.case_id must be positive")
        for name, raw_value in (
            ("stride_ratio", self.stride_ratio),
            ("contig_atom_ratio", self.contig_atom_ratio),
            ("working_set_sram_ratio", self.working_set_sram_ratio),
            ("working_set_vreg_ratio", self.working_set_vreg_ratio),
            ("wave_occupancy_ratio", self.wave_occupancy_ratio),
            ("oi_ridge_ratio", self.oi_ridge_ratio),
        ):
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"CasePhysicsRatios.{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.stride_ratio < 1.0:
            raise ValueError("CasePhysicsRatios.stride_ratio must be >= 1.0")
        if self.contig_atom_ratio <= 0.0:
            raise ValueError("CasePhysicsRatios.contig_atom_ratio must be > 0.0")
        if isinstance(self.inner_contiguous_bytes, bool) or not isinstance(
            self.inner_contiguous_bytes, int
        ):
            raise ValueError("CasePhysicsRatios.inner_contiguous_bytes must be an int")
        if self.inner_contiguous_bytes < 1:
            raise ValueError("CasePhysicsRatios.inner_contiguous_bytes must be positive")
        object.__setattr__(self, "is_atom_aligned", bool(self.is_atom_aligned))
        object.__setattr__(self, "has_empty_group", bool(self.has_empty_group))

    def phase_signature(
        self,
        base_axes: TopologyAxes,
    ) -> tuple[EnginePhase, LayoutPhase, ResidencyBand, AlignmentBand, WaveBand]:
        """Return the (engine, layout, residency, alignment, wave) phase coordinate."""
        layout: LayoutPhase = "contiguous" if self.stride_ratio <= 1.0 else "strided_or_gather"
        residency: ResidencyBand
        if self.working_set_vreg_ratio <= 1.0:
            residency = "vreg_resident"
        elif self.working_set_sram_ratio <= 1.0:
            residency = "sram_1pass"
        else:
            residency = "gm_multipass"
        alignment: AlignmentBand
        if self.contig_atom_ratio < 1.0:
            alignment = "sub_atom_simt"
        elif self.is_atom_aligned:
            alignment = "aligned_vector"
        else:
            alignment = "unaligned_predicated"
        wave: WaveBand
        if self.has_empty_group:
            wave = "ragged_groups"
        elif self.wave_occupancy_ratio < 0.5:
            wave = "under_occupied_split"
        else:
            wave = "balanced_waves"
        engine: EnginePhase
        if base_axes.uses_cube:
            if self.oi_ridge_ratio < 0.25:
                engine = "aiv_vector_or_gemv"
            elif base_axes.compute_engine == "coupled_1c2v":
                engine = "coupled_1c2v"
            else:
                engine = "aic_cube"
        else:
            engine = "aiv_vector_or_gemv"
        return engine, layout, residency, alignment, wave

    def effective_axes(self, base_axes: TopologyAxes) -> TopologyAxes:
        """Return the post-phase-transition topology axes implied by these ratios."""
        engine, _, residency, _, wave = self.phase_signature(base_axes)
        compute_engine = base_axes.compute_engine
        memory_ring = base_axes.memory_ring
        iteration_regularity = base_axes.iteration_regularity
        if base_axes.uses_cube and engine == "aiv_vector_or_gemv":
            compute_engine = "aiv_only"
            memory_ring = "gm_ub_stream"
        if (
            residency == "gm_multipass"
            and base_axes.memory_ring == "persistent_onchip_state"
        ):
            memory_ring = (
                "l0c_ub_epilogue"
                if base_axes.compute_engine == "coupled_1c2v"
                else "gm_ub_stream"
            )
        if wave == "ragged_groups":
            iteration_regularity = "ragged_group"
        if self.contig_atom_ratio < 1.0 and compute_engine == "aiv_only":
            iteration_regularity = "irregular_simt"
        return TopologyAxes(
            compute_engine=compute_engine,
            memory_ring=memory_ring,
            iteration_regularity=iteration_regularity,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "stride_ratio": self.stride_ratio,
            "contig_atom_ratio": self.contig_atom_ratio,
            "is_atom_aligned": self.is_atom_aligned,
            "working_set_sram_ratio": self.working_set_sram_ratio,
            "working_set_vreg_ratio": self.working_set_vreg_ratio,
            "wave_occupancy_ratio": self.wave_occupancy_ratio,
            "oi_ridge_ratio": self.oi_ridge_ratio,
            "has_empty_group": self.has_empty_group,
            "inner_contiguous_bytes": self.inner_contiguous_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CasePhysicsRatios:
        raw_empty_group = payload.get("has_empty_group", False)
        if not isinstance(raw_empty_group, bool):
            raise ValueError("CasePhysicsRatios.has_empty_group must be a bool")
        return cls(
            case_id=_payload_int(payload, "case_id"),
            stride_ratio=_payload_float(payload, "stride_ratio"),
            contig_atom_ratio=_payload_float(payload, "contig_atom_ratio"),
            is_atom_aligned=_payload_bool(payload, "is_atom_aligned"),
            working_set_sram_ratio=_payload_float(payload, "working_set_sram_ratio"),
            working_set_vreg_ratio=_payload_float(payload, "working_set_vreg_ratio"),
            wave_occupancy_ratio=_payload_float(payload, "wave_occupancy_ratio"),
            oi_ridge_ratio=_payload_float(payload, "oi_ridge_ratio"),
            inner_contiguous_bytes=_payload_int(payload, "inner_contiguous_bytes"),
            has_empty_group=raw_empty_group,
        )


def classify_case_regime(
    ratios: CasePhysicsRatios,
    *,
    kind: CasePhysicsKind,
    axes: TopologyAxes,
    dims: Mapping[str, int],
) -> str:
    """Derive a stable regime tag from physics phase bands rather than operator names."""
    if ratios.has_empty_group:
        return "empty_group"
    engine, layout, residency, _, wave = ratios.phase_signature(axes)
    parts: list[str] = []
    if kind == "attention":
        is_decode = dims.get("S_q", dims.get("M", 0)) <= 2
        parts.append("decode" if is_decode else "prefill")
        if wave == "under_occupied_split":
            parts.append("split")
    elif kind == "matmul":
        if engine == "aiv_vector_or_gemv":
            parts.append("gemv")
        elif engine == "coupled_1c2v":
            parts.append("mixcore")
        else:
            parts.append("cube")
        if residency == "gm_multipass":
            parts.append("multipass")
        if wave == "under_occupied_split":
            parts.append("small")
    elif kind == "reduction":
        parts.append("reduce")
        parts.append("strided" if layout == "strided_or_gather" else "tail")
        if residency == "gm_multipass":
            parts.append("twopass")
        if ratios.contig_atom_ratio < 1.0:
            parts.append("subatom")
    else:
        parts.append("vector")
        if layout == "strided_or_gather":
            parts.append("strided")
        if residency == "gm_multipass":
            parts.append("twopass")
        if ratios.contig_atom_ratio < 1.0:
            parts.append("subatom")
    return "_".join(parts)


def group_case_regime_tags(
    ratios_by_case: Mapping[int, CasePhysicsRatios],
    dims_by_case: Mapping[int, Mapping[str, int]],
    *,
    kind: CasePhysicsKind,
    axes: TopologyAxes,
) -> dict[str, tuple[int, ...]]:
    """Partition per-case ratios into shared stable phase tags.

    The extractor and the campaign use the same tags.
    """
    grouped: dict[str, list[int]] = {}
    for case_id in sorted(ratios_by_case):
        dims = dims_by_case.get(case_id)
        if dims is None:
            raise ValueError(f"missing structural dims for case {case_id}")
        tag = classify_case_regime(
            ratios_by_case[case_id],
            kind=kind,
            axes=axes,
            dims=dims,
        )
        grouped.setdefault(tag, []).append(case_id)
    return {tag: tuple(case_ids) for tag, case_ids in grouped.items()}


_PHASE_AXIS_INDEX: dict[str, int] = {
    "engine": 0,
    "layout": 1,
    "residency": 2,
    "alignment": 3,
    "wave": 4,
}

_PHASE_BAND_ALIASES: dict[str, dict[str, str]] = {
    "engine": {"aiv_vector_or_gemv": "vec", "aic_cube": "cube", "coupled_1c2v": "cv"},
    "layout": {"contiguous": "contig", "strided_or_gather": "strided"},
    "residency": {"vreg_resident": "vreg", "sram_1pass": "sram", "gm_multipass": "gmpass"},
    "alignment": {
        "sub_atom_simt": "subatom",
        "aligned_vector": "aligned",
        "unaligned_predicated": "unaligned",
    },
    "wave": {
        "ragged_groups": "ragged",
        "under_occupied_split": "underfill",
        "balanced_waves": "filled",
    },
}

_PHASE_SPLIT_AXES: tuple[str, ...] = ("wave", "residency", "alignment", "layout", "engine")


def _phase_band_alias(signature: PhaseSignature, axis_name: str) -> str:
    return _PHASE_BAND_ALIASES[axis_name][signature[_PHASE_AXIS_INDEX[axis_name]]]


def case_workload_bytes(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str],
) -> int:
    """Return dtype-aware total input bytes for one official case."""
    default_dtype = dtypes[0] if dtypes else "float16"
    total = 0.0
    for index, shape in enumerate(shapes):
        dtype_name = dtypes[index] if index < len(dtypes) else default_dtype
        total += shape_numel(shape) * dtype_bytes_for(dtype_name)
    return int(total)


def case_primary_dtype_bytes(case_entry: Mapping[str, object]) -> float:
    """Return the declared element width of one official case's first dtype."""
    dtypes = flatten_str_values(case_entry.get("dtype"))
    if not dtypes:
        raise ValueError("case entry requires at least one declared dtype")
    return dtype_bytes_for(dtypes[0])


def case_workload_scale(workload_bytes: int, dims: Mapping[str, int]) -> tuple[int, int, int]:
    """Return the shared representative key: bytes, element count, largest dimension."""
    if isinstance(workload_bytes, bool) or not isinstance(workload_bytes, int):
        raise ValueError("workload_bytes must be an int")
    if workload_bytes < 0:
        raise ValueError("workload_bytes must be non-negative")
    total_elements = max(
        0,
        int(dims.get("total_elements", dims.get("total_elems", 0))),
    )
    largest_dim = max((int(value) for value in dims.values() if value > 0), default=0)
    return workload_bytes, total_elements, largest_dim


@dataclass(frozen=True, kw_only=True)
class CasePhysicsRegimeCase:
    """One official case inside a physical regime with its raw 5D phase coordinate."""

    case_id: int
    dims: dict[str, int]
    phase_signature: PhaseSignature
    workload_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int):
            raise ValueError("CasePhysicsRegimeCase.case_id must be an int")
        if self.case_id < 1:
            raise ValueError("CasePhysicsRegimeCase.case_id must be positive")
        object.__setattr__(self, "dims", _as_int_dict(self.dims, "dims"))
        if isinstance(self.workload_bytes, bool) or not isinstance(self.workload_bytes, int):
            raise ValueError("CasePhysicsRegimeCase.workload_bytes must be an int")
        if self.workload_bytes < 0:
            raise ValueError("CasePhysicsRegimeCase.workload_bytes must be non-negative")


def select_case_regime_representative(
    cases: Sequence[CasePhysicsRegimeCase],
) -> CasePhysicsRegimeCase:
    """Return the highest-workload case, breaking ties by case id."""
    if not cases:
        raise ValueError("select_case_regime_representative needs at least one case")
    return max(
        cases,
        key=lambda case: (
            *case_workload_scale(case.workload_bytes, case.dims),
            case.case_id,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class CasePhysicsRegimeGroup:
    """One regime of the public 2-4 physical partition mapping."""

    regime_tag: str
    cases: tuple[CasePhysicsRegimeCase, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.regime_tag, str) or not self.regime_tag.strip():
            raise ValueError("CasePhysicsRegimeGroup.regime_tag must be a non-empty string")
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("CasePhysicsRegimeGroup.cases must not be empty")
        for case in cases:
            if not isinstance(case, CasePhysicsRegimeCase):
                raise ValueError(
                    "CasePhysicsRegimeGroup.cases must contain CasePhysicsRegimeCase values"
                )
        unique_ids = {case.case_id for case in cases}
        if len(unique_ids) != len(cases):
            raise ValueError("CasePhysicsRegimeGroup.cases must have unique case ids")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "rationale", str(self.rationale))

    @property
    def case_ids(self) -> tuple[int, ...]:
        """Return the covered official case ids in ascending order."""
        return tuple(sorted(case.case_id for case in self.cases))

    @property
    def phase_coordinates(self) -> tuple[str, ...]:
        """Return every raw 5D phase coordinate covered by this regime."""
        return tuple(sorted({"/".join(case.phase_signature) for case in self.cases}))

    @property
    def representative_case(self) -> CasePhysicsRegimeCase:
        """Return the highest-workload case that supplies Stage-0 dimensions."""
        return select_case_regime_representative(self.cases)

    @property
    def description(self) -> str:
        """Return an explainable description that preserves every raw coordinate."""
        coordinates = "; ".join(self.phase_coordinates)
        rationale = self.rationale.strip()
        reason = f"{rationale} " if rationale else ""
        return (
            f"Physical regime {self.regime_tag} covering official cases {list(self.case_ids)} "
            f"at 5D coordinates: {coordinates}. {reason}Representative case "
            f"{self.representative_case.case_id} supplies the Stage-0 dimensions."
        )


def _split_case_regime_group(
    group: CasePhysicsRegimeGroup,
    *,
    max_new_groups: int,
) -> tuple[CasePhysicsRegimeGroup, ...]:
    ordered = sorted(group.cases, key=lambda case: case.case_id)
    for axis_name in _PHASE_SPLIT_AXES:
        buckets: dict[str, list[CasePhysicsRegimeCase]] = {}
        for case in ordered:
            alias = _phase_band_alias(case.phase_signature, axis_name)
            buckets.setdefault(alias, []).append(case)
        if 1 < len(buckets) <= max_new_groups:
            reason = (
                f"{group.rationale.strip()} Cases sharing regime {group.regime_tag} split by "
                f"{axis_name} band to meet the partition floor. "
            ).strip()
            return tuple(
                CasePhysicsRegimeGroup(
                    regime_tag=f"{group.regime_tag}_{axis_name}_{alias}",
                    cases=tuple(entries),
                    rationale=reason,
                )
                for alias, entries in buckets.items()
            )
    workload_order = sorted(
        ordered,
        key=lambda case: (
            *case_workload_scale(case.workload_bytes, case.dims),
            case.case_id,
        ),
    )
    midpoint = len(workload_order) // 2
    if midpoint == 0:
        raise ValueError(f"regime {group.regime_tag!r} has too few cases to split")
    reason = (
        f"{group.rationale.strip()} Cases sharing every phase coordinate under regime "
        f"{group.regime_tag} split by dtype-aware workload scale to meet the partition floor."
    ).strip()
    return (
        CasePhysicsRegimeGroup(
            regime_tag=f"{group.regime_tag}_latency",
            cases=tuple(workload_order[:midpoint]),
            rationale=reason,
        ),
        CasePhysicsRegimeGroup(
            regime_tag=f"{group.regime_tag}_throughput",
            cases=tuple(workload_order[midpoint:]),
            rationale=reason,
        ),
    )


def _phase_distance(left_sig: PhaseSignature, right_sig: PhaseSignature) -> int:
    """Compute weighted lexicographic 5D phase distance."""
    distance = 0
    if left_sig[0] != right_sig[0]:
        distance += 100
    if left_sig[1] != right_sig[1]:
        distance += 30
    if left_sig[2] != right_sig[2]:
        distance += 2
    if left_sig[3] != right_sig[3]:
        distance += 20 if "sub_atom_simt" in (left_sig[3], right_sig[3]) else 3
    if left_sig[4] != right_sig[4]:
        distance += 1
    return distance


def _merge_nearest_regimes(
    entries_by_tag: dict[str, list[CasePhysicsRegimeCase]],
) -> tuple[str, str, int]:
    tags = sorted(entries_by_tag)
    best: tuple[int, str, str] | None = None
    for left_index, left in enumerate(tags):
        for right in tags[left_index + 1 :]:
            distance = min(
                _phase_distance(left_case.phase_signature, right_case.phase_signature)
                for left_case in entries_by_tag[left]
                for right_case in entries_by_tag[right]
            )
            candidate = (distance, left, right)
            if best is None or candidate < best:
                best = candidate
    assert best is not None, "nearest-regime merge needs at least two regimes"
    return best[1], best[2], best[0]


def partition_case_regimes(
    ratios_by_case: Mapping[int, CasePhysicsRatios],
    dims_by_case: Mapping[int, Mapping[str, int]],
    workload_bytes_by_case: Mapping[int, int],
    *,
    kind: CasePhysicsKind,
    axes: TopologyAxes,
    min_partitions: int = 2,
    max_partitions: int = 4,
) -> tuple[CasePhysicsRegimeGroup, ...]:
    """Partition official cases into 2-4 physical regimes while preserving raw phase evidence.

    Cases group by ``classify_case_regime``. Above the cap, wave-occupancy ``_split``
    variants fold into their base mechanism and remaining regimes merge at the nearest 5D
    phase distance. A single regime splits on the first differing phase band, or by
    dtype-aware workload scale when every coordinate coincides. No operator name or case id
    selects a partition.
    """
    if not ratios_by_case:
        raise ValueError("partition_case_regimes needs at least one official case")
    if not 1 <= min_partitions <= max_partitions:
        raise ValueError("partition bounds must satisfy 1 <= min_partitions <= max_partitions")
    case_ids = set(ratios_by_case)
    for name, mapping in (
        ("dims", dims_by_case),
        ("workload bytes", workload_bytes_by_case),
    ):
        if set(mapping) != case_ids:
            missing = sorted(case_ids - set(mapping))
            extra = sorted(set(mapping) - case_ids)
            raise ValueError(f"case {name} mismatch: missing={missing}, extra={extra}")
    entries_by_tag: dict[str, list[CasePhysicsRegimeCase]] = {}
    for case_id in sorted(case_ids):
        ratios = ratios_by_case[case_id]
        if ratios.case_id != case_id:
            raise ValueError(f"case {case_id} ratios carry case_id {ratios.case_id}")
        dims = {str(key): int(value) for key, value in dims_by_case[case_id].items()}
        workload_bytes = workload_bytes_by_case[case_id]
        if isinstance(workload_bytes, bool) or not isinstance(workload_bytes, int):
            raise ValueError(f"case {case_id} workload_bytes must be an int")
        if workload_bytes < 0:
            raise ValueError(f"case {case_id} workload_bytes must be non-negative")
        tag = classify_case_regime(ratios, kind=kind, axes=axes, dims=dims)
        entries_by_tag.setdefault(tag, []).append(
            CasePhysicsRegimeCase(
                case_id=case_id,
                dims=dims,
                phase_signature=ratios.phase_signature(axes),
                workload_bytes=workload_bytes,
            )
        )
    rationale_parts: list[str] = []
    if len(entries_by_tag) > max_partitions:
        folded: dict[str, list[CasePhysicsRegimeCase]] = {}
        for tag, entries in entries_by_tag.items():
            folded.setdefault(tag.removesuffix("_split"), []).extend(entries)
        if set(folded) != set(entries_by_tag):
            rationale_parts.append(
                "Wave-occupancy _split variants folded into their base mechanism. "
            )
        entries_by_tag = folded
    while len(entries_by_tag) > max_partitions:
        left, right, distance = _merge_nearest_regimes(entries_by_tag)
        left_entries = entries_by_tag.pop(left)
        right_entries = entries_by_tag.pop(right)
        if len(left_entries) != len(right_entries):
            merged_tag = left if len(left_entries) > len(right_entries) else right
        else:
            merged_tag = min(left, right)
        entries_by_tag[merged_tag] = sorted(
            (*left_entries, *right_entries),
            key=lambda case: case.case_id,
        )
        rationale_parts.append(
            f"Regimes {left} and {right} merged at 5D phase distance {distance} to stay "
            f"within the {max_partitions}-partition cap. "
        )
    groups = [
        CasePhysicsRegimeGroup(
            regime_tag=tag,
            cases=tuple(entries),
            rationale="".join(rationale_parts),
        )
        for tag, entries in entries_by_tag.items()
    ]
    while len(groups) < min_partitions:
        candidates = [group for group in groups if len(group.cases) >= 2]
        if not candidates:
            break
        chosen = sorted(candidates, key=lambda group: (-len(group.cases), group.regime_tag))[0]
        position = groups.index(chosen)
        replacement = _split_case_regime_group(
            chosen,
            max_new_groups=max_partitions - len(groups) + 1,
        )
        groups = [*groups[:position], *replacement, *groups[position + 1 :]]
    groups.sort(key=lambda group: group.case_ids[0])
    return tuple(groups)


@dataclass(frozen=True, kw_only=True)
class CasePhysicsInput:
    """Raw physical quantities for one case before ratio normalization."""

    case_id: int
    kind: CasePhysicsKind
    stride_ratio: float
    atom_bytes: int
    sram_capacity_bytes: int
    inner_contiguous_bytes: int
    working_set_bytes: float
    vreg_extent_elements: int
    outer_tiles: int
    physical_cores: int
    oi_ridge_ratio: float
    has_empty_group: bool = False

    def __post_init__(self) -> None:
        if self.case_id < 1 or self.outer_tiles < 1 or self.physical_cores < 1:
            raise ValueError("CasePhysicsInput requires positive case_id, tiles and cores")
        if self.atom_bytes < 1 or self.sram_capacity_bytes < 1:
            raise ValueError("CasePhysicsInput requires positive atom and SRAM capacities")
        if self.inner_contiguous_bytes < 1 or self.vreg_extent_elements < 1:
            raise ValueError("CasePhysicsInput requires positive contiguous and VREG extents")
        if self.stride_ratio < 1.0 or self.oi_ridge_ratio <= 0.0:
            raise ValueError("CasePhysicsInput requires stride_ratio >= 1 and oi_ridge_ratio > 0")
        if not math.isfinite(self.working_set_bytes) or self.working_set_bytes < 0.0:
            raise ValueError("CasePhysicsInput.working_set_bytes must be finite and non-negative")

    def to_ratios(self) -> CasePhysicsRatios:
        """Normalize raw quantities into the dimensionless public ratio contract."""
        return compute_case_physics_ratios(self)


@dataclass(frozen=True, kw_only=True)
class CasePhysicsContext:
    """Physical environment and semantic class used to derive one case's ratios."""

    kind: CasePhysicsKind
    dtype_bytes: float
    facts: AscendSpaceFacts
    sram_capacity_bytes: int
    physical_cores: int
    atom_bytes: int
    reduction_axis: int = -1
    strided_gather: bool = False
    has_empty_group: bool = False


def build_case_physics_context(
    kind: CasePhysicsKind,
    dtype_bytes: float,
    facts: AscendSpaceFacts,
    *,
    reduction_axis: int = -1,
    strided_gather: bool = False,
    has_empty_group: bool = False,
) -> CasePhysicsContext:
    """Select SRAM capacity, core count and atom width for one semantic case class."""
    if dtype_bytes <= 0.0:
        raise ValueError(f"dtype_bytes must be > 0, got {dtype_bytes}")
    if kind in ("matmul", "attention"):
        sram_capacity_bytes = facts.l1_bytes
        physical_cores = facts.num_aic_cores
    else:
        sram_capacity_bytes = facts.ub_simdvf_usable_bytes
        physical_cores = facts.num_aiv_cores
    atom_bytes = (
        max(1, int(16.0 * dtype_bytes)) if kind == "matmul" else facts.ub_datablock_bytes
    )
    return CasePhysicsContext(
        kind=kind,
        dtype_bytes=dtype_bytes,
        facts=facts,
        sram_capacity_bytes=sram_capacity_bytes,
        physical_cores=physical_cores,
        atom_bytes=atom_bytes,
        reduction_axis=reduction_axis,
        strided_gather=strided_gather,
        has_empty_group=has_empty_group,
    )


def compute_case_physics_ratios(raw: CasePhysicsInput) -> CasePhysicsRatios:
    """Normalize raw case quantities into the six public dimensionless physics ratios."""
    return CasePhysicsRatios(
        case_id=raw.case_id,
        stride_ratio=raw.stride_ratio,
        contig_atom_ratio=raw.inner_contiguous_bytes / raw.atom_bytes,
        is_atom_aligned=raw.inner_contiguous_bytes % raw.atom_bytes == 0,
        working_set_sram_ratio=raw.working_set_bytes / raw.sram_capacity_bytes,
        working_set_vreg_ratio=raw.vreg_extent_elements * 4.0 / VREG_CAPACITY_BYTES,
        wave_occupancy_ratio=raw.outer_tiles / raw.physical_cores,
        oi_ridge_ratio=raw.oi_ridge_ratio,
        has_empty_group=raw.has_empty_group,
        inner_contiguous_bytes=raw.inner_contiguous_bytes,
    )


def _int_attr(attrs: Mapping[str, object], key: str) -> int | None:
    raw = attrs.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"case attr {key!r} must be an int, got {type(raw).__name__}")
    return raw


def _int_list_attr(attrs: Mapping[str, object], key: str) -> tuple[int, ...]:
    raw = attrs.get(key)
    if not isinstance(raw, (list, tuple)):
        return ()
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"case attr {key!r} must contain only ints")
        values.append(item)
    return tuple(values)


def has_empty_group_attrs(attrs: Mapping[str, object] | None) -> bool:
    """Return whether cumulative group offsets contain a zero-width group."""
    if attrs is None:
        return False
    group_list = attrs.get("group_list")
    if not isinstance(group_list, (list, tuple)) or not group_list:
        return False
    previous = 0
    for raw in group_list:
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        if raw <= previous:
            return True
        previous = raw
    return False


def _case_layout(attrs: Mapping[str, object]) -> str:
    raw = attrs.get("inputLayout", attrs.get("input_layout", attrs.get("layout", "BSND")))
    return str(raw).strip().upper()


def _sequence_length(shape: Sequence[int], layout: str) -> int | None:
    if not shape:
        return None
    if len(shape) == 4:
        index = 2 if layout == "BNSD" else 1
    else:
        index = 0
    value = int(shape[index])
    return value if value > 0 else None


def _attention_kv_length(shapes: Sequence[Sequence[int]], layout: str) -> int | None:
    lengths = [
        length
        for shape in shapes[1:3]
        if (length := _sequence_length(shape, layout)) is not None
    ]
    return max(lengths) if lengths else None


def infer_case_dims_structural(
    shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, object],
    *,
    axes: TopologyAxes,
    uses_cube: bool,
) -> dict[str, int]:
    """Extract canonical M/N/K/E and sequence dims structurally without operator names."""
    if not shapes or not shapes[0]:
        return {"M": 1, "N": 1, "K": 1}
    primary = [int(dim) for dim in shapes[0]]
    dims: dict[str, int] = {}

    if (axes.iteration_regularity == "ragged_group" or "group_list" in attrs) and len(shapes) >= 2:
        weight = shapes[1]
        dims["M"] = primary[0]
        dims["K"] = primary[-1]
        dims["E"] = int(weight[0]) if len(weight) >= 3 else 1
        dims["N"] = int(weight[-1]) if len(weight) >= 2 else 1
        return dims

    in_features = _int_attr(attrs, "in_features")
    out_features = _int_attr(attrs, "out_features")
    if (in_features is not None or out_features is not None) and len(shapes) >= 2:
        dims["M"] = 1 if len(primary) == 1 else shape_numel(primary[:-1])
        dims["K"] = in_features if in_features is not None else primary[-1]
        final_shape = shapes[-1]
        fallback_n = int(final_shape[0]) if final_shape else 1
        dims["N"] = out_features if out_features is not None else fallback_n
        return dims

    if axes.memory_ring == "bidirectional_cv_ring" and len(shapes) >= 2:
        layout = _case_layout(attrs)
        if len(primary) == 4:
            dims["B"] = primary[0]
            dims["S_q"] = primary[2] if layout == "BNSD" else primary[1]
            dims["H_q"] = primary[1] if layout == "BNSD" else primary[2]
            dims["D"] = primary[3]
            dims["M"] = dims["B"] * dims["S_q"]
        elif len(primary) == 3:
            dims["T"] = primary[0]
            dims["H_q"] = primary[1]
            dims["D"] = primary[2]
            query_lengths = _int_list_attr(attrs, "actual_seq_qlen")
            dims["S_q"] = max(query_lengths) if query_lengths else primary[0]
            dims["M"] = dims["S_q"]
        else:
            dims["M"] = shape_numel(primary[:-1])
            dims["K"] = primary[-1]
            secondary = [int(dim) for dim in shapes[1]]
            if (
                len(secondary) == 2
                and secondary[-1] == dims["K"]
                and secondary[0] != dims["K"]
            ):
                dims["N"] = secondary[0]
            else:
                dims["N"] = secondary[-1] if secondary else primary[-1]
        kv_length = _attention_kv_length(shapes, layout)
        dims["S_kv"] = kv_length if kv_length is not None else dims.get("S_q", dims["M"])
        if len(primary) != 2:
            dims["K"] = primary[-1]
            dims["N"] = int(shapes[2][-1]) if len(shapes) >= 3 and shapes[2] else primary[-1]
        return dims

    if uses_cube and len(shapes) >= 2 and shapes[1] and len(primary) <= 3 and len(shapes[1]) <= 3:
        dims["M"] = shape_numel(primary[:-1])
        dims["K"] = primary[-1]
        rhs = shapes[1]
        if len(rhs) == 2 and rhs[-1] == dims["K"] and rhs[0] != dims["K"]:
            dims["N"] = int(rhs[0])
        else:
            dims["N"] = int(rhs[-1]) if rhs else 1
        return dims

    if len(primary) >= 2:
        dims["rows"] = shape_numel(primary[:-1])
        dims["cols"] = primary[-1]
        dims["M"] = dims["rows"]
        dims["N"] = dims["cols"]
    else:
        dims["rows"] = 1
        dims["cols"] = primary[0]
        dims["M"] = 1
        dims["N"] = primary[0]
    if uses_cube:
        dims["K"] = primary[-1]
        if len(shapes) >= 2 and shapes[1]:
            dims["N"] = int(shapes[1][-1])
    dims["total_elems"] = shape_numel(primary)
    return dims


def infer_case_physics_kind(axes: TopologyAxes, *, has_reduction: bool) -> CasePhysicsKind:
    """Classify a case from topology axes and an explicit reduction signal."""
    if axes.memory_ring == "bidirectional_cv_ring" and axes.uses_cube:
        return "attention"
    if axes.uses_cube:
        return "matmul"
    return "reduction" if has_reduction else "elementwise"


def _optional_dim(dims: Mapping[str, int], names: Sequence[str]) -> int | None:
    for name in names:
        value = dims.get(name)
        if value is not None and value > 0:
            return value
    return None


def _require_dim(dims: Mapping[str, int], names: Sequence[str], what: str) -> int:
    value = _optional_dim(dims, names)
    if value is None:
        raise ValueError(f"{what} requires one of dims {tuple(names)} to be positive")
    return value


def _reduction_axis(attrs: Mapping[str, object], default_axis: int) -> int:
    raw = attrs.get("dim", attrs.get("axis", default_axis))
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"reduction axis must be an int, got {type(raw).__name__}")
    return raw


def _cube_dtype_name(dtype_bytes: float) -> str:
    if dtype_bytes <= 0.5:
        return "f4"
    if dtype_bytes <= 1.0:
        return "fp8e4m3"
    if dtype_bytes <= 2.0:
        return "f16"
    return "f32"


def _derive_reduction_input(
    case_id: int,
    primary: Sequence[int],
    attrs: Mapping[str, object],
    context: CasePhysicsContext,
) -> CasePhysicsInput:
    axis = _reduction_axis(attrs, context.reduction_axis)
    if axis < 0:
        axis += len(primary)
    if axis < 0 or axis >= len(primary):
        raise ValueError(f"reduction axis {axis} out of range for shape {tuple(primary)}")
    active = primary[axis]
    if active < 1:
        raise ValueError(f"reduction axis {axis} has non-positive extent {active}")
    inner = shape_numel(primary[axis + 1 :])
    rows = shape_numel(primary[:axis])
    contiguous_elements = inner if inner > 1 else active
    return CasePhysicsInput(
        case_id=case_id,
        kind="reduction",
        stride_ratio=float(max(1, inner)),
        atom_bytes=context.atom_bytes,
        sram_capacity_bytes=context.sram_capacity_bytes,
        inner_contiguous_bytes=max(1, int(round(contiguous_elements * context.dtype_bytes))),
        working_set_bytes=active * min(inner, 64) * max(2.0 * context.dtype_bytes, 4.0),
        vreg_extent_elements=active,
        outer_tiles=rows,
        physical_cores=context.physical_cores,
        oi_ridge_ratio=1.0,
        has_empty_group=context.has_empty_group,
    )


def _derive_elementwise_input(
    case_id: int,
    primary: Sequence[int],
    context: CasePhysicsContext,
) -> CasePhysicsInput:
    width = primary[-1]
    rows = shape_numel(primary[:-1])
    if width < 1:
        raise ValueError(f"elementwise input shape {tuple(primary)} has a non-positive width")
    return CasePhysicsInput(
        case_id=case_id,
        kind="elementwise",
        stride_ratio=1.0,
        atom_bytes=context.atom_bytes,
        sram_capacity_bytes=context.sram_capacity_bytes,
        inner_contiguous_bytes=max(1, int(round(width * context.dtype_bytes))),
        working_set_bytes=min(rows, 64) * width * max(2.0 * context.dtype_bytes, 4.0),
        vreg_extent_elements=width,
        outer_tiles=rows,
        physical_cores=context.physical_cores,
        oi_ridge_ratio=1.0,
        has_empty_group=context.has_empty_group,
    )


def _derive_matmul_input(
    case_id: int,
    dims: Mapping[str, int],
    context: CasePhysicsContext,
) -> CasePhysicsInput:
    m_dim = _require_dim(dims, ("M", "m_rows", "rows"), "matmul")
    k_dim = _require_dim(dims, ("K", "in_features", "k_dim"), "matmul")
    n_dim = _require_dim(dims, ("N", "out_features", "n_dim", "cols"), "matmul")
    batch = _optional_dim(dims, ("B", "b_dim")) or 1
    critical_m = compute_critical_cube_tile_m(
        context.facts,
        _cube_dtype_name(context.dtype_bytes),
    )
    working_set = (
        min(m_dim, 128) * min(k_dim, 64) + min(k_dim, 64) * min(n_dim, 128)
    ) * context.dtype_bytes
    return CasePhysicsInput(
        case_id=case_id,
        kind="matmul",
        stride_ratio=1.0,
        atom_bytes=context.atom_bytes,
        sram_capacity_bytes=context.sram_capacity_bytes,
        inner_contiguous_bytes=max(1, int(round(k_dim * context.dtype_bytes))),
        working_set_bytes=working_set,
        vreg_extent_elements=k_dim,
        outer_tiles=m_dim * batch,
        physical_cores=context.physical_cores,
        oi_ridge_ratio=m_dim / float(critical_m),
        has_empty_group=context.has_empty_group,
    )


def _derive_attention_input(
    case_id: int,
    shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, object],
    dims: Mapping[str, int],
    context: CasePhysicsContext,
) -> CasePhysicsInput:
    layout = _case_layout(attrs)
    s_q = _require_dim(dims, ("S_q", "S", "M", "m_rows"), "attention")
    s_kv = _optional_dim(dims, ("S_kv", "s_kv"))
    if s_kv is None:
        s_kv = _attention_kv_length(shapes, layout)
    if s_kv is None:
        s_kv = s_q
    d_nope = _optional_dim(dims, ("D_nope", "d_nope"))
    d_rope = _optional_dim(dims, ("D_rope", "d_rope"))
    d_qk = _optional_dim(dims, ("D_qk", "D", "d_dim", "dk"))
    if d_qk is None and d_nope is not None and d_rope is not None:
        d_qk = d_nope + d_rope
    if d_qk is None:
        d_qk = int(shapes[0][-1]) if shapes[0] else 0
    d_v = _optional_dim(dims, ("D_v", "dv")) or d_nope or d_qk
    segments = [value for value in (d_nope, d_rope, d_qk, d_v) if value is not None and value > 0]
    if not segments:
        raise ValueError("attention requires a positive head dimension")
    segment = min(segments)
    stride_ratio = 4.0 if context.strided_gather else (2.0 if layout == "BNSD" else 1.0)
    batch = _optional_dim(dims, ("B", "b_dim")) or 1
    critical_m = compute_critical_cube_tile_m(
        context.facts,
        _cube_dtype_name(context.dtype_bytes),
    )
    return CasePhysicsInput(
        case_id=case_id,
        kind="attention",
        stride_ratio=stride_ratio,
        atom_bytes=context.atom_bytes,
        sram_capacity_bytes=context.sram_capacity_bytes,
        inner_contiguous_bytes=max(1, int(round(segment * context.dtype_bytes))),
        working_set_bytes=min(s_q, 64) * s_kv * 4.0,
        vreg_extent_elements=s_kv,
        outer_tiles=batch * max(1, math.ceil(s_q / 16)),
        physical_cores=context.physical_cores,
        oi_ridge_ratio=s_q / float(critical_m),
        has_empty_group=context.has_empty_group,
    )


def derive_case_physics_input(
    case_id: int,
    shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, object],
    dims: Mapping[str, int],
    context: CasePhysicsContext,
) -> CasePhysicsInput:
    """Derive raw physical quantities for one case from structural dims and input shapes."""
    if not shapes or not shapes[0]:
        raise ValueError("case physics require at least one non-empty input shape")
    primary = [int(dim) for dim in shapes[0]]
    if context.kind == "attention":
        return _derive_attention_input(case_id, shapes, attrs, dims, context)
    if context.kind == "matmul":
        return _derive_matmul_input(case_id, dims, context)
    if context.kind == "reduction":
        return _derive_reduction_input(case_id, primary, attrs, context)
    return _derive_elementwise_input(case_id, primary, context)


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

    def unlocked_facet_names(self) -> tuple[str, ...]:
        """Return ordered tuple of currently unlocked facet names."""
        return self.unlocked_facets()

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
    case_ratios: tuple[CasePhysicsRatios, ...] = ()
    representative_case_id: int | None = None
    phase_coordinates: tuple[str, ...] = ()
    rationale: str = ""

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
        ratios = tuple(self.case_ratios)
        for ratio in ratios:
            if not isinstance(ratio, CasePhysicsRatios):
                raise ValueError("RegimeSlice.case_ratios must contain CasePhysicsRatios values")
        object.__setattr__(self, "case_ratios", ratios)
        if self.representative_case_id is not None:
            if isinstance(self.representative_case_id, bool) or not isinstance(
                self.representative_case_id, int
            ):
                raise ValueError("RegimeSlice.representative_case_id must be an int")
            if self.representative_case_id not in self.case_nums:
                raise ValueError(
                    "RegimeSlice.representative_case_id must belong to case_nums"
                )
        object.__setattr__(
            self,
            "phase_coordinates",
            _as_str_tuple(self.phase_coordinates, "phase_coordinates"),
        )
        object.__setattr__(self, "rationale", str(self.rationale))

    def representative_ratios(self) -> CasePhysicsRatios | None:
        """Return the physics ratios of the representative case."""
        if not self.case_ratios:
            return None
        if self.representative_case_id is not None:
            for ratio in self.case_ratios:
                if ratio.case_id == self.representative_case_id:
                    return ratio
            return None
        return self.case_ratios[len(self.case_ratios) // 2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_tag": self.regime_tag,
            "case_nums": list(self.case_nums),
            "representative_dims": dict(self.representative_dims),
            "has_unaligned_tail": self.has_unaligned_tail,
            "has_empty_group": self.has_empty_group,
            "ref_perf_us": self.ref_perf_us,
            "t_hw_us": self.t_hw_us,
            "case_ratios": [ratio.to_dict() for ratio in self.case_ratios],
            "representative_case_id": self.representative_case_id,
            "phase_coordinates": list(self.phase_coordinates),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RegimeSlice:
        raw_ratios = payload.get("case_ratios") or ()
        if isinstance(raw_ratios, (str, bytes)) or not isinstance(raw_ratios, Sequence):
            raise ValueError("RegimeSlice.case_ratios must be a sequence")
        raw_representative = payload.get("representative_case_id")
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
            case_ratios=tuple(
                CasePhysicsRatios.from_dict(item)
                for item in raw_ratios
                if isinstance(item, Mapping)
            ),
            representative_case_id=(
                int(raw_representative) if raw_representative is not None else None
            ),
            phase_coordinates=tuple(
                str(item) for item in (payload.get("phase_coordinates") or ())
            ),
            rationale=str(payload.get("rationale", "")),
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
