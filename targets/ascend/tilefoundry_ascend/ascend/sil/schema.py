"""Scheduling-Interface-Layer (SIL) typed contract schemas for Ascend 950 (dav-3510)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

_T = TypeVar("_T")

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text, json_clone as _json_clone
from tilefoundry_ascend.ascend.target.facts import AscendSpaceFacts, cube_macs_per_cycle

VALID_BACKEND_TIERS: tuple[str, ...] = ("tilelang_dsl", "ascendc_template")
VALID_VECTOR_MODES: tuple[str, ...] = ("simdvf", "simtvf", "hybrid")
VALID_COMPUTE_ENGINES: tuple[str, ...] = ("aiv_only", "aic_only", "coupled_1c2v")
VALID_MEMORY_RINGS: tuple[str, ...] = (
    "gm_ub_stream", "ub_random_atomic", "gm_l1_l0_fixpipe",
    "l0c_ub_epilogue", "bidirectional_cv_ring", "persistent_onchip_state",
)
VALID_ITERATION_REGULARITIES: tuple[str, ...] = ("affine_simd", "irregular_simt", "ragged_group", "dynamic_opcode_stream")
VALID_FREEDOM_FACETS: tuple[str, ...] = (
    "alg_compute_and_monoid", "tile_and_stage_knobs", "exec_mode_and_spr",
    "sram_layout_and_aliasing", "dma_and_cache_hierarchy", "pipeline_and_sync_graph", "grid_and_workspace_topology",
)


def _as_seq(raw: object, field_name: str, conv: Callable[[object], _T], label: str) -> tuple[_T, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of {label}, got {type(raw).__name__}")
    return tuple(conv(item) for item in raw)


def _as_dict(raw: object, field_name: str, conv: Callable[[object], _T]) -> dict[str, _T]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(raw).__name__}")
    return {str(k): conv(v) for k, v in raw.items()}


def _as_str_tuple(raw: object, field_name: str) -> tuple[str, ...]: return _as_seq(raw, field_name, str, "strings")
def _as_int_tuple(raw: object, field_name: str) -> tuple[int, ...]: return _as_seq(raw, field_name, int, "ints")
def _as_int_dict(raw: object, field_name: str) -> dict[str, int]: return _as_dict(raw, field_name, int)
def _as_float_dict(raw: object, field_name: str) -> dict[str, float]: return _as_dict(raw, field_name, float)
def _as_str_dict(raw: object, field_name: str) -> dict[str, str]: return _as_dict(raw, field_name, str)


def _require_nonempty_str(val: object, label: str) -> None:
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_int(val: object, label: str, *, min_val: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{label} must be an int")
    if min_val is not None and val < min_val:
        raise ValueError(f"{label} must be positive" if min_val == 1 else f"{label} must be >= {min_val}")
    return val


def _coerce_topology_axes_field(raw: object, label: str) -> TopologyAxes:
    if isinstance(raw, Mapping):
        return TopologyAxes.from_dict(raw)
    if not isinstance(raw, TopologyAxes):
        raise TypeError(f"{label} must be TopologyAxes or Mapping, got {type(raw).__name__}")
    return raw


@dataclass(frozen=True)
class TopologyAxes(JsonRecordMixin):
    """3-axis orthogonal hardware topology specification for Ascend 950 (dav-3510)."""

    compute_engine: str
    memory_ring: str
    iteration_regularity: str

    def __post_init__(self) -> None:
        for attr, valid in (
            ("compute_engine", VALID_COMPUTE_ENGINES),
            ("memory_ring", VALID_MEMORY_RINGS),
            ("iteration_regularity", VALID_ITERATION_REGULARITIES),
        ):
            if (val := getattr(self, attr)) not in valid:
                raise ValueError(f"TopologyAxes.{attr} {val!r} must be one of {valid}")

    @property
    def uses_cube(self) -> bool:
        """Return whether this topology uses AIC Cube compute."""
        return self.compute_engine in ("aic_only", "coupled_1c2v")

    def axis_keys(self) -> tuple[str, str, str]:
        """Return canonical factorized axis keys (ce:*, mr:*, ir:*)."""
        return (f"ce:{self.compute_engine}", f"mr:{self.memory_ring}", f"ir:{self.iteration_regularity}")

    def key(self) -> str:
        """Return compact canonical 3-axis coordinate key."""
        return f"{self.compute_engine}/{self.memory_ring}/{self.iteration_regularity}"


CasePhysicsKind = Literal["reduction", "matmul", "attention", "elementwise"]
LayoutPhase = Literal["contiguous", "strided_or_gather"]
ResidencyBand = Literal["vreg_resident", "sram_1pass", "gm_multipass"]
AlignmentBand = Literal["sub_atom_simt", "aligned_vector", "unaligned_predicated"]
WaveBand = Literal["under_occupied_split", "balanced_waves", "ragged_groups"]
EnginePhase = Literal["aiv_vector_or_gemv", "aic_cube", "coupled_1c2v"]
PhaseSignature = tuple[EnginePhase, LayoutPhase, ResidencyBand, AlignmentBand, WaveBand]
VREG_CAPACITY_BYTES: int = 4096

DTYPE_BYTES: dict[str, float] = {
    **{k: 0.5 for k in ("float4_e2m1", "float4_e1m2", "float4_e2m1fn", "float4_e2m1fnx2", "fp4", "f4e2m1", "f4e1m2")},
    **{k: 1.0 for k in ("bool", "uint8", "u8", "int8", "i8", "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "fp8", "fp8e4m3", "fp8e5m2", "f8e4m3", "f8e5m2", "hif8")},
    **{k: 2.0 for k in ("int16", "float16", "half", "f16", "bfloat16", "bf16")},
    **{k: 4.0 for k in ("int32", "float32", "float", "f32")},
    "int64": 8.0, "float64": 8.0,
}

_CUBE_DTYPE_CANON_AND_BYTES: dict[str, tuple[str, float]] = {
    k: (canon, b)
    for canon, b, aliases in (
        ("f4e2m1", 0.5, ("f4e2m1", "float4_e2m1", "float4_e2m1fn", "float4_e2m1fnx2", "fp4e2m1", "fp4", "f4")),
        ("f4e1m2", 0.5, ("f4e1m2", "float4_e1m2", "fp4e1m2")),
        ("fp8e4m3", 1.0, ("fp8e4m3", "float8_e4m3fn", "fp8_e4m3", "f8e4m3", "fp8", "float8_e8m0fnu", "f8e8m0")),
        ("fp8e5m2", 1.0, ("fp8e5m2", "float8_e5m2", "fp8_e5m2", "f8e5m2")),
        ("hif8", 1.0, ("hif8",)), ("i8", 1.0, ("i8", "int8")), ("u8", 1.0, ("u8", "uint8")),
        ("f16", 2.0, ("f16", "float16", "half")), ("bf16", 2.0, ("bf16", "bfloat16")),
        ("hf32", 4.0, ("hf32",)), ("f32", 4.0, ("f32", "float32", "float")),
    )
    for k in aliases
}


def dtype_bytes_for(dtype_name: str) -> float:
    """Return physical storage bytes per element for a known dtype name."""
    if (key := dtype_name.strip().lower()) not in DTYPE_BYTES:
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
    if not isinstance(raw, (list, tuple)):
        return ()
    shapes: list[tuple[int, ...]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)):
            subs = entry if (entry and isinstance(entry[0], (list, tuple))) else (entry,)
            for sub in subs:
                if isinstance(sub, (list, tuple)) and all(_is_shape_dim(d) for d in sub):
                    shapes.append(tuple(int(d) for d in sub))
    return tuple(shapes)


def flatten_str_values(raw: object) -> tuple[str, ...]:
    """Normalize a nested dtype or tag payload into a flat tuple of strings."""
    if not isinstance(raw, (list, tuple)):
        return () if raw is None else (str(raw),)
    return tuple(v for item in raw for v in flatten_str_values(item))


def has_unaligned_tail_for_case(shapes: Sequence[Sequence[int]], dtypes: Sequence[str], uses_cube: bool) -> bool:
    """Return whether any tensor tail or cube operand violates 32-byte or 16-element alignment."""
    if not shapes:
        return False
    align_elems = max(1, int(32 // max(dtype_bytes_for(dtypes[0] if dtypes else "float16"), 1.0)))
    cube_align = 16 if uses_cube else align_elems
    for shape in shapes:
        if not shape:
            continue
        if (shape[-1] > 1 and shape[-1] % align_elems != 0) or (
            uses_cube and any(d > 1 and d % cube_align != 0 for d in shape[-2:])
        ) or (len(shape) >= 2 and shape[0] > 15 and shape[0] % 2 == 1):
            return True
    return False


def compute_critical_cube_tile_m(
    facts: AscendSpaceFacts | None, elem_dtype: str, *, l2_bytes_per_cycle_per_aic: float = 100.0, dual_operand_miss: bool = False
) -> int:
    """Compute the analytical Cube tile-M waterline from MAC rate, dtype width and L2 bandwidth."""
    if (bandwidth := float(l2_bytes_per_cycle_per_aic)) <= 0.0:
        raise ValueError(f"l2_bytes_per_cycle_per_aic must be > 0, got {l2_bytes_per_cycle_per_aic}")
    canon_dtype, dtype_bytes = _CUBE_DTYPE_CANON_AND_BYTES.get(elem_dtype.strip().lower(), ("f16", 2.0))
    macs = facts.cube_macs_per_cycle(canon_dtype) if facts is not None else cube_macs_per_cycle(canon_dtype)
    return max(1, int(math.ceil(float(macs) * dtype_bytes * (2.0 if dual_operand_miss else 1.0) / bandwidth)))


def _payload_float(payload: Mapping[str, object], key: str) -> float:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"CasePhysicsRatios.{key} must be numeric")
    return float(raw)


def _payload_int(payload: Mapping[str, object], key: str) -> int:
    return _require_int(payload.get(key), f"CasePhysicsRatios.{key}")


def _payload_bool(payload: Mapping[str, object], key: str) -> bool:
    raw = payload.get(key)
    if not isinstance(raw, bool):
        raise ValueError(f"CasePhysicsRatios.{key} must be a bool")
    return raw


@dataclass(frozen=True, kw_only=True)
class CasePhysicsRatios(JsonRecordMixin):
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
        _require_int(self.case_id, "CasePhysicsRatios.case_id", min_val=1)
        for name in (
            "stride_ratio", "contig_atom_ratio", "working_set_sram_ratio",
            "working_set_vreg_ratio", "wave_occupancy_ratio", "oi_ridge_ratio",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"CasePhysicsRatios.{name} must be finite and non-negative")
            object.__setattr__(self, name, val)
        if self.stride_ratio < 1.0:
            raise ValueError("CasePhysicsRatios.stride_ratio must be >= 1.0")
        if self.contig_atom_ratio <= 0.0:
            raise ValueError("CasePhysicsRatios.contig_atom_ratio must be > 0.0")
        _require_int(self.inner_contiguous_bytes, "CasePhysicsRatios.inner_contiguous_bytes", min_val=1)
        object.__setattr__(self, "is_atom_aligned", bool(self.is_atom_aligned))
        object.__setattr__(self, "has_empty_group", bool(self.has_empty_group))

    def phase_signature(self, base_axes: TopologyAxes) -> tuple[EnginePhase, LayoutPhase, ResidencyBand, AlignmentBand, WaveBand]:
        """Return the (engine, layout, residency, alignment, wave) phase coordinate."""
        layout: LayoutPhase = "contiguous" if self.stride_ratio <= 1.0 else "strided_or_gather"
        residency: ResidencyBand = "vreg_resident" if self.working_set_vreg_ratio <= 1.0 else "sram_1pass" if self.working_set_sram_ratio <= 1.0 else "gm_multipass"
        alignment: AlignmentBand = "sub_atom_simt" if self.contig_atom_ratio < 1.0 else "aligned_vector" if self.is_atom_aligned else "unaligned_predicated"
        wave: WaveBand = "ragged_groups" if self.has_empty_group else "under_occupied_split" if self.wave_occupancy_ratio < 0.5 else "balanced_waves"
        engine: EnginePhase = "aiv_vector_or_gemv" if (not base_axes.uses_cube or self.oi_ridge_ratio < 0.25) else "coupled_1c2v" if base_axes.compute_engine == "coupled_1c2v" else "aic_cube"
        return engine, layout, residency, alignment, wave

    def effective_axes(self, base_axes: TopologyAxes) -> TopologyAxes:
        """Return the post-phase-transition topology axes implied by these ratios."""
        engine, _, residency, _, wave = self.phase_signature(base_axes)
        ce, mr, ir = base_axes.compute_engine, base_axes.memory_ring, base_axes.iteration_regularity
        if base_axes.uses_cube and engine == "aiv_vector_or_gemv":
            ce, mr = "aiv_only", "gm_ub_stream"
        if residency == "gm_multipass" and base_axes.memory_ring == "persistent_onchip_state":
            mr = "l0c_ub_epilogue" if base_axes.compute_engine == "coupled_1c2v" else "gm_ub_stream"
        if wave == "ragged_groups":
            ir = "ragged_group"
        if self.contig_atom_ratio < 1.0 and ce == "aiv_only":
            ir = "irregular_simt"
        return TopologyAxes(compute_engine=ce, memory_ring=mr, iteration_regularity=ir)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CasePhysicsRatios:
        if not isinstance(raw_empty := payload.get("has_empty_group", False), bool):
            raise ValueError("CasePhysicsRatios.has_empty_group must be a bool")
        return cls(
            case_id=_payload_int(payload, "case_id"), stride_ratio=_payload_float(payload, "stride_ratio"),
            contig_atom_ratio=_payload_float(payload, "contig_atom_ratio"), is_atom_aligned=_payload_bool(payload, "is_atom_aligned"),
            working_set_sram_ratio=_payload_float(payload, "working_set_sram_ratio"), working_set_vreg_ratio=_payload_float(payload, "working_set_vreg_ratio"),
            wave_occupancy_ratio=_payload_float(payload, "wave_occupancy_ratio"), oi_ridge_ratio=_payload_float(payload, "oi_ridge_ratio"),
            inner_contiguous_bytes=_payload_int(payload, "inner_contiguous_bytes"), has_empty_group=raw_empty,
        )


def classify_case_regime(ratios: CasePhysicsRatios, *, kind: CasePhysicsKind, axes: TopologyAxes, dims: Mapping[str, int]) -> str:
    """Derive a stable regime tag from physics phase bands rather than operator names."""
    if ratios.has_empty_group:
        return "empty_group"
    engine, layout, residency, _, wave = ratios.phase_signature(axes)
    parts: list[str] = []
    if kind == "attention":
        parts.append("decode" if dims.get("S_q", dims.get("M", 0)) <= 2 else "prefill")
        if wave == "under_occupied_split":
            parts.append("split")
    elif kind == "matmul":
        parts.append("gemv" if engine == "aiv_vector_or_gemv" else "mixcore" if engine == "coupled_1c2v" else "cube")
        if residency == "gm_multipass":
            parts.append("multipass")
        if wave == "under_occupied_split":
            parts.append("small")
    else:
        parts.extend(["reduce", "strided" if layout == "strided_or_gather" else "tail"] if kind == "reduction" else (["vector", "strided"] if layout == "strided_or_gather" else ["vector"]))
        if residency == "gm_multipass":
            parts.append("twopass")
        if ratios.contig_atom_ratio < 1.0:
            parts.append("subatom")
    return "_".join(parts)


def group_case_regime_tags(
    ratios_by_case: Mapping[int, CasePhysicsRatios], dims_by_case: Mapping[int, Mapping[str, int]], *, kind: CasePhysicsKind, axes: TopologyAxes
) -> dict[str, tuple[int, ...]]:
    """Partition per-case ratios into shared stable phase tags.

    The extractor and the campaign use the same tags.
    """
    grouped: dict[str, list[int]] = {}
    for case_id in sorted(ratios_by_case):
        if (dims := dims_by_case.get(case_id)) is None:
            raise ValueError(f"missing structural dims for case {case_id}")
        grouped.setdefault(classify_case_regime(ratios_by_case[case_id], kind=kind, axes=axes, dims=dims), []).append(case_id)
    return {tag: tuple(cids) for tag, cids in grouped.items()}


_PHASE_AXIS_INDEX: dict[str, int] = {"engine": 0, "layout": 1, "residency": 2, "alignment": 3, "wave": 4}
_PHASE_BAND_ALIASES: dict[str, dict[str, str]] = {
    "engine": {"aiv_vector_or_gemv": "vec", "aic_cube": "cube", "coupled_1c2v": "cv"},
    "layout": {"contiguous": "contig", "strided_or_gather": "strided"},
    "residency": {"vreg_resident": "vreg", "sram_1pass": "sram", "gm_multipass": "gmpass"},
    "alignment": {"sub_atom_simt": "subatom", "aligned_vector": "aligned", "unaligned_predicated": "unaligned"},
    "wave": {"ragged_groups": "ragged", "under_occupied_split": "underfill", "balanced_waves": "filled"},
}
_PHASE_SPLIT_AXES: tuple[str, ...] = ("wave", "residency", "alignment", "layout", "engine")


def _phase_band_alias(signature: PhaseSignature, axis_name: str) -> str:
    return _PHASE_BAND_ALIASES[axis_name][signature[_PHASE_AXIS_INDEX[axis_name]]]


def case_workload_bytes(shapes: Sequence[Sequence[int]], dtypes: Sequence[str]) -> int:
    """Return dtype-aware total input bytes for one official case."""
    default_dtype = dtypes[0] if dtypes else "float16"
    return int(sum(shape_numel(s) * dtype_bytes_for(dtypes[i] if i < len(dtypes) else default_dtype) for i, s in enumerate(shapes)))


def case_primary_dtype_bytes(case_entry: Mapping[str, object]) -> float:
    """Return the declared element width of one official case's first dtype."""
    if not (dtypes := flatten_str_values(case_entry.get("dtype"))):
        raise ValueError("case entry requires at least one declared dtype")
    return dtype_bytes_for(dtypes[0])


def case_workload_scale(workload_bytes: int, dims: Mapping[str, int]) -> tuple[int, int, int]:
    """Return the shared representative key: bytes, element count, largest dimension."""
    _require_int(workload_bytes, "workload_bytes", min_val=0)
    return workload_bytes, max(0, int(dims.get("total_elements", dims.get("total_elems", 0)))), max((int(v) for v in dims.values() if v > 0), default=0)


@dataclass(frozen=True, kw_only=True)
class CasePhysicsRegimeCase:
    """One official case inside a physical regime with its raw 5D phase coordinate."""

    case_id: int
    dims: dict[str, int]
    phase_signature: PhaseSignature
    workload_bytes: int

    def __post_init__(self) -> None:
        _require_int(self.case_id, "CasePhysicsRegimeCase.case_id", min_val=1)
        object.__setattr__(self, "dims", _as_int_dict(self.dims, "dims"))
        _require_int(self.workload_bytes, "CasePhysicsRegimeCase.workload_bytes", min_val=0)


def select_case_regime_representative(cases: Sequence[CasePhysicsRegimeCase]) -> CasePhysicsRegimeCase:
    """Return the highest-workload case, breaking ties by case id."""
    if not cases:
        raise ValueError("select_case_regime_representative needs at least one case")
    return max(cases, key=lambda c: (*case_workload_scale(c.workload_bytes, c.dims), c.case_id))


@dataclass(frozen=True, kw_only=True)
class CasePhysicsRegimeGroup:
    """One regime of the public 2-4 physical partition mapping."""

    regime_tag: str
    cases: tuple[CasePhysicsRegimeCase, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_str(self.regime_tag, "CasePhysicsRegimeGroup.regime_tag")
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("CasePhysicsRegimeGroup.cases must not be empty")
        if any(not isinstance(c, CasePhysicsRegimeCase) for c in cases):
            raise ValueError("CasePhysicsRegimeGroup.cases must contain CasePhysicsRegimeCase values")
        if len({c.case_id for c in cases}) != len(cases):
            raise ValueError("CasePhysicsRegimeGroup.cases must have unique case ids")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "rationale", str(self.rationale))

    @property
    def case_ids(self) -> tuple[int, ...]:
        """Return the covered official case ids in ascending order."""
        return tuple(sorted(c.case_id for c in self.cases))

    @property
    def phase_coordinates(self) -> tuple[str, ...]:
        """Return every raw 5D phase coordinate covered by this regime."""
        return tuple(sorted({"/".join(c.phase_signature) for c in self.cases}))

    @property
    def representative_case(self) -> CasePhysicsRegimeCase:
        """Return the highest-workload case that supplies Stage-0 dimensions."""
        return select_case_regime_representative(self.cases)

    @property
    def description(self) -> str:
        """Return an explainable description that preserves every raw coordinate."""
        reason = f"{self.rationale.strip()} " if self.rationale.strip() else ""
        return (
            f"Physical regime {self.regime_tag} covering official cases {list(self.case_ids)} "
            f"at 5D coordinates: {'; '.join(self.phase_coordinates)}. {reason}Representative case "
            f"{self.representative_case.case_id} supplies the Stage-0 dimensions."
        )


def _split_case_regime_group(group: CasePhysicsRegimeGroup, *, max_new_groups: int) -> tuple[CasePhysicsRegimeGroup, ...]:
    ordered = sorted(group.cases, key=lambda c: c.case_id)
    for axis_name in _PHASE_SPLIT_AXES:
        buckets: dict[str, list[CasePhysicsRegimeCase]] = {}
        for case in ordered:
            buckets.setdefault(_phase_band_alias(case.phase_signature, axis_name), []).append(case)
        if 1 < len(buckets) <= max_new_groups:
            reason = f"{group.rationale.strip()} Cases sharing regime {group.regime_tag} split by {axis_name} band to meet the partition floor. ".strip()
            return tuple(CasePhysicsRegimeGroup(regime_tag=f"{group.regime_tag}_{axis_name}_{al}", cases=tuple(es), rationale=reason) for al, es in buckets.items())
    workload_order = sorted(ordered, key=lambda c: (*case_workload_scale(c.workload_bytes, c.dims), c.case_id))
    if (midpoint := len(workload_order) // 2) == 0:
        raise ValueError(f"regime {group.regime_tag!r} has too few cases to split")
    reason = f"{group.rationale.strip()} Cases sharing every phase coordinate under regime {group.regime_tag} split by dtype-aware workload scale to meet the partition floor.".strip()
    return (
        CasePhysicsRegimeGroup(regime_tag=f"{group.regime_tag}_latency", cases=tuple(workload_order[:midpoint]), rationale=reason),
        CasePhysicsRegimeGroup(regime_tag=f"{group.regime_tag}_throughput", cases=tuple(workload_order[midpoint:]), rationale=reason),
    )


def _phase_distance(left_sig: PhaseSignature, right_sig: PhaseSignature) -> int:
    """Compute weighted lexicographic 5D phase distance."""
    d = (100 if left_sig[0] != right_sig[0] else 0) + (30 if left_sig[1] != right_sig[1] else 0) + (2 if left_sig[2] != right_sig[2] else 0)
    if left_sig[3] != right_sig[3]:
        d += 20 if "sub_atom_simt" in (left_sig[3], right_sig[3]) else 3
    return d + (1 if left_sig[4] != right_sig[4] else 0)


def _merge_nearest_regimes(entries_by_tag: dict[str, list[CasePhysicsRegimeCase]]) -> tuple[str, str, int]:
    tags = sorted(entries_by_tag)
    best: tuple[int, str, str] | None = None
    for idx, left in enumerate(tags):
        for right in tags[idx + 1 :]:
            dist = min(_phase_distance(lc.phase_signature, rc.phase_signature) for lc in entries_by_tag[left] for rc in entries_by_tag[right])
            if best is None or (dist, left, right) < best:
                best = (dist, left, right)
    assert best is not None, "nearest-regime merge needs at least two regimes"
    return best[1], best[2], best[0]


def partition_case_regimes(
    ratios_by_case: Mapping[int, CasePhysicsRatios], dims_by_case: Mapping[int, Mapping[str, int]],
    workload_bytes_by_case: Mapping[int, int], *, kind: CasePhysicsKind, axes: TopologyAxes,
    min_partitions: int = 2, max_partitions: int = 4,
) -> tuple[CasePhysicsRegimeGroup, ...]:
    """Partition official cases into 2-4 physical regimes while preserving raw phase evidence."""
    if not ratios_by_case:
        raise ValueError("partition_case_regimes needs at least one official case")
    if not 1 <= min_partitions <= max_partitions:
        raise ValueError("partition bounds must satisfy 1 <= min_partitions <= max_partitions")
    case_ids = set(ratios_by_case)
    for name, mapping in (("dims", dims_by_case), ("workload bytes", workload_bytes_by_case)):
        if set(mapping) != case_ids:
            raise ValueError(f"case {name} mismatch: missing={sorted(case_ids - set(mapping))}, extra={sorted(set(mapping) - case_ids)}")
    entries_by_tag: dict[str, list[CasePhysicsRegimeCase]] = {}
    for cid in sorted(case_ids):
        if (ratios := ratios_by_case[cid]).case_id != cid:
            raise ValueError(f"case {cid} ratios carry case_id {ratios.case_id}")
        dims = {str(k): int(v) for k, v in dims_by_case[cid].items()}
        wbytes = _require_int(workload_bytes_by_case[cid], f"case {cid} workload_bytes", min_val=0)
        entries_by_tag.setdefault(classify_case_regime(ratios, kind=kind, axes=axes, dims=dims), []).append(
            CasePhysicsRegimeCase(case_id=cid, dims=dims, phase_signature=ratios.phase_signature(axes), workload_bytes=wbytes)
        )
    rationale_parts: list[str] = []
    if len(entries_by_tag) > max_partitions:
        folded: dict[str, list[CasePhysicsRegimeCase]] = {}
        for tag, entries in entries_by_tag.items():
            folded.setdefault(tag.removesuffix("_split"), []).extend(entries)
        if set(folded) != set(entries_by_tag):
            rationale_parts.append("Wave-occupancy _split variants folded into their base mechanism. ")
        entries_by_tag = folded
    while len(entries_by_tag) > max_partitions:
        left, right, dist = _merge_nearest_regimes(entries_by_tag)
        le, re_ = entries_by_tag.pop(left), entries_by_tag.pop(right)
        merged_tag = (left if len(le) > len(re_) else right) if len(le) != len(re_) else min(left, right)
        entries_by_tag[merged_tag] = sorted((*le, *re_), key=lambda c: c.case_id)
        rationale_parts.append(f"Regimes {left} and {right} merged at 5D phase distance {dist} to stay within the {max_partitions}-partition cap. ")
    groups = [CasePhysicsRegimeGroup(regime_tag=t, cases=tuple(es), rationale="".join(rationale_parts)) for t, es in entries_by_tag.items()]
    while len(groups) < min_partitions:
        if not (candidates := [g for g in groups if len(g.cases) >= 2]):
            break
        chosen = sorted(candidates, key=lambda g: (-len(g.cases), g.regime_tag))[0]
        pos = groups.index(chosen)
        groups = [*groups[:pos], *_split_case_regime_group(chosen, max_new_groups=max_partitions - len(groups) + 1), *groups[pos + 1 :]]
    groups.sort(key=lambda g: g.case_ids[0])
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
    kind: CasePhysicsKind, dtype_bytes: float, facts: AscendSpaceFacts,
    *, reduction_axis: int = -1, strided_gather: bool = False, has_empty_group: bool = False,
) -> CasePhysicsContext:
    """Select SRAM capacity, core count and atom width for one semantic case class."""
    if dtype_bytes <= 0.0:
        raise ValueError(f"dtype_bytes must be > 0, got {dtype_bytes}")
    is_cube = kind in ("matmul", "attention")
    return CasePhysicsContext(
        kind=kind, dtype_bytes=dtype_bytes, facts=facts,
        sram_capacity_bytes=facts.l1_bytes if is_cube else facts.ub_simdvf_usable_bytes,
        physical_cores=facts.num_aic_cores if is_cube else facts.num_aiv_cores,
        atom_bytes=max(1, int(16.0 * dtype_bytes)) if kind == "matmul" else facts.ub_datablock_bytes,
        reduction_axis=reduction_axis, strided_gather=strided_gather, has_empty_group=has_empty_group,
    )


def compute_case_physics_ratios(raw: CasePhysicsInput) -> CasePhysicsRatios:
    """Normalize raw case quantities into the six public dimensionless physics ratios."""
    return CasePhysicsRatios(
        case_id=raw.case_id, stride_ratio=raw.stride_ratio, contig_atom_ratio=raw.inner_contiguous_bytes / raw.atom_bytes,
        is_atom_aligned=raw.inner_contiguous_bytes % raw.atom_bytes == 0,
        working_set_sram_ratio=raw.working_set_bytes / raw.sram_capacity_bytes,
        working_set_vreg_ratio=raw.vreg_extent_elements * 4.0 / VREG_CAPACITY_BYTES,
        wave_occupancy_ratio=raw.outer_tiles / raw.physical_cores, oi_ridge_ratio=raw.oi_ridge_ratio,
        has_empty_group=raw.has_empty_group, inner_contiguous_bytes=raw.inner_contiguous_bytes,
    )


def _int_attr(attrs: Mapping[str, object], key: str) -> int | None:
    if (raw := attrs.get(key)) is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"case attr {key!r} must be an int, got {type(raw).__name__}")
    return raw


def _int_list_attr(attrs: Mapping[str, object], key: str) -> tuple[int, ...]:
    if not isinstance(raw := attrs.get(key), (list, tuple)):
        return ()
    if any(isinstance(i, bool) or not isinstance(i, int) for i in raw):
        raise ValueError(f"case attr {key!r} must contain only ints")
    return tuple(raw)


def has_empty_group_attrs(attrs: Mapping[str, object] | None) -> bool:
    """Return whether cumulative group offsets contain a zero-width group."""
    if not isinstance(group_list := (attrs.get("group_list") if attrs is not None else None), (list, tuple)) or not group_list:
        return False
    prev = 0
    for raw in group_list:
        if not isinstance(raw, bool) and isinstance(raw, int):
            if raw <= prev:
                return True
            prev = raw
    return False


def _case_layout(attrs: Mapping[str, object]) -> str:
    return str(attrs.get("inputLayout", attrs.get("input_layout", attrs.get("layout", "BSND")))).strip().upper()


def _sequence_length(shape: Sequence[int], layout: str) -> int | None:
    if not shape:
        return None
    val = int(shape[(2 if layout == "BNSD" else 1) if len(shape) == 4 else 0])
    return val if val > 0 else None


def _attention_kv_length(shapes: Sequence[Sequence[int]], layout: str) -> int | None:
    lengths = [ln for s in shapes[1:3] if (ln := _sequence_length(s, layout)) is not None]
    return max(lengths) if lengths else None


def infer_case_dims_structural(
    shapes: Sequence[Sequence[int]], attrs: Mapping[str, object], *, axes: TopologyAxes, uses_cube: bool
) -> dict[str, int]:
    """Extract canonical M/N/K/E and sequence dims structurally without operator names."""
    if not shapes or not shapes[0]:
        return {"M": 1, "N": 1, "K": 1}
    primary = [int(d) for d in shapes[0]]
    dims: dict[str, int] = {}
    if (axes.iteration_regularity == "ragged_group" or "group_list" in attrs) and len(shapes) >= 2:
        w = shapes[1]
        return {"M": primary[0], "K": primary[-1], "E": int(w[0]) if len(w) >= 3 else 1, "N": int(w[-1]) if len(w) >= 2 else 1}
    in_f, out_f = _int_attr(attrs, "in_features"), _int_attr(attrs, "out_features")
    if (in_f is not None or out_f is not None) and len(shapes) >= 2:
        return {"M": 1 if len(primary) == 1 else shape_numel(primary[:-1]), "K": in_f if in_f is not None else primary[-1], "N": out_f if out_f is not None else (int(shapes[-1][0]) if shapes[-1] else 1)}
    if axes.memory_ring == "bidirectional_cv_ring" and len(shapes) >= 2:
        layout = _case_layout(attrs)
        if len(primary) == 4:
            sq_i, hq_i = (2, 1) if layout == "BNSD" else (1, 2)
            dims.update({"B": primary[0], "S_q": primary[sq_i], "H_q": primary[hq_i], "D": primary[3], "M": primary[0] * primary[sq_i]})
        elif len(primary) == 3:
            sq = max(qlen) if (qlen := _int_list_attr(attrs, "actual_seq_qlen")) else primary[0]
            dims.update({"T": primary[0], "H_q": primary[1], "D": primary[2], "S_q": sq, "M": sq})
        else:
            sec, k_val = [int(d) for d in shapes[1]], primary[-1]
            n_val = sec[0] if (len(sec) == 2 and sec[-1] == k_val and sec[0] != k_val) else (sec[-1] if sec else k_val)
            dims.update({"M": shape_numel(primary[:-1]), "K": k_val, "N": n_val})
        kv_len = _attention_kv_length(shapes, layout)
        dims["S_kv"] = kv_len if kv_len is not None else dims.get("S_q", dims["M"])
        if len(primary) != 2:
            dims["K"] = primary[-1]
            dims["N"] = int(shapes[2][-1]) if len(shapes) >= 3 and shapes[2] else primary[-1]
        return dims
    if uses_cube and len(shapes) >= 2 and shapes[1] and len(primary) <= 3 and len(shapes[1]) <= 3:
        rhs, k_val = shapes[1], primary[-1]
        n_val = int(rhs[0]) if (len(rhs) == 2 and rhs[-1] == k_val and rhs[0] != k_val) else (int(rhs[-1]) if rhs else 1)
        return {"M": shape_numel(primary[:-1]), "K": k_val, "N": n_val}
    rows, cols = (shape_numel(primary[:-1]) if len(primary) >= 2 else 1), (primary[-1] if len(primary) >= 2 else primary[0])
    dims = {"rows": rows, "cols": cols, "M": rows, "N": cols}
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
    return "matmul" if axes.uses_cube else ("reduction" if has_reduction else "elementwise")


def _optional_dim(dims: Mapping[str, int], names: Sequence[str]) -> int | None:
    return next((v for n in names if (v := dims.get(n)) is not None and v > 0), None)


def _require_dim(dims: Mapping[str, int], names: Sequence[str], what: str) -> int:
    if (value := _optional_dim(dims, names)) is None:
        raise ValueError(f"{what} requires one of dims {tuple(names)} to be positive")
    return value


def _reduction_axis(attrs: Mapping[str, object], default_axis: int) -> int:
    return _require_int(attrs.get("dim", attrs.get("axis", default_axis)), "reduction axis")


def _cube_dtype_name(dtype_bytes: float) -> str:
    return "f4" if dtype_bytes <= 0.5 else "fp8e4m3" if dtype_bytes <= 1.0 else "f16" if dtype_bytes <= 2.0 else "f32"


def _derive_reduction_input(case_id: int, primary: Sequence[int], attrs: Mapping[str, object], ctx: CasePhysicsContext) -> CasePhysicsInput:
    axis = _reduction_axis(attrs, ctx.reduction_axis)
    axis = axis + len(primary) if axis < 0 else axis
    if not 0 <= axis < len(primary):
        raise ValueError(f"reduction axis {axis} out of range for shape {tuple(primary)}")
    if (active := primary[axis]) < 1:
        raise ValueError(f"reduction axis {axis} has non-positive extent {active}")
    inner, rows = shape_numel(primary[axis + 1 :]), shape_numel(primary[:axis])
    return CasePhysicsInput(
        case_id=case_id, kind="reduction", stride_ratio=float(max(1, inner)), atom_bytes=ctx.atom_bytes,
        sram_capacity_bytes=ctx.sram_capacity_bytes, inner_contiguous_bytes=max(1, int(round((inner if inner > 1 else active) * ctx.dtype_bytes))),
        working_set_bytes=active * min(inner, 64) * max(2.0 * ctx.dtype_bytes, 4.0), vreg_extent_elements=active,
        outer_tiles=rows, physical_cores=ctx.physical_cores, oi_ridge_ratio=1.0, has_empty_group=ctx.has_empty_group,
    )


def _derive_elementwise_input(case_id: int, primary: Sequence[int], ctx: CasePhysicsContext) -> CasePhysicsInput:
    width, rows = primary[-1], shape_numel(primary[:-1])
    if width < 1:
        raise ValueError(f"elementwise input shape {tuple(primary)} has a non-positive width")
    return CasePhysicsInput(
        case_id=case_id, kind="elementwise", stride_ratio=1.0, atom_bytes=ctx.atom_bytes,
        sram_capacity_bytes=ctx.sram_capacity_bytes, inner_contiguous_bytes=max(1, int(round(width * ctx.dtype_bytes))),
        working_set_bytes=min(rows, 64) * width * max(2.0 * ctx.dtype_bytes, 4.0), vreg_extent_elements=width,
        outer_tiles=rows, physical_cores=ctx.physical_cores, oi_ridge_ratio=1.0, has_empty_group=ctx.has_empty_group,
    )


def _derive_matmul_input(case_id: int, dims: Mapping[str, int], ctx: CasePhysicsContext) -> CasePhysicsInput:
    m_dim, k_dim, n_dim = _require_dim(dims, ("M", "m_rows", "rows"), "matmul"), _require_dim(dims, ("K", "in_features", "k_dim"), "matmul"), _require_dim(dims, ("N", "out_features", "n_dim", "cols"), "matmul")
    batch = _optional_dim(dims, ("B", "b_dim")) or 1
    critical_m = compute_critical_cube_tile_m(ctx.facts, _cube_dtype_name(ctx.dtype_bytes))
    ws = (min(m_dim, 128) * min(k_dim, 64) + min(k_dim, 64) * min(n_dim, 128)) * ctx.dtype_bytes
    return CasePhysicsInput(
        case_id=case_id, kind="matmul", stride_ratio=1.0, atom_bytes=ctx.atom_bytes,
        sram_capacity_bytes=ctx.sram_capacity_bytes, inner_contiguous_bytes=max(1, int(round(k_dim * ctx.dtype_bytes))),
        working_set_bytes=ws, vreg_extent_elements=k_dim, outer_tiles=m_dim * batch,
        physical_cores=ctx.physical_cores, oi_ridge_ratio=m_dim / float(critical_m), has_empty_group=ctx.has_empty_group,
    )


def _derive_attention_input(
    case_id: int, shapes: Sequence[Sequence[int]], attrs: Mapping[str, object], dims: Mapping[str, int], ctx: CasePhysicsContext
) -> CasePhysicsInput:
    layout = _case_layout(attrs)
    s_q = _require_dim(dims, ("S_q", "S", "M", "m_rows"), "attention")
    s_kv = _optional_dim(dims, ("S_kv", "s_kv")) or _attention_kv_length(shapes, layout) or s_q
    d_nope, d_rope = _optional_dim(dims, ("D_nope", "d_nope")), _optional_dim(dims, ("D_rope", "d_rope"))
    d_qk = _optional_dim(dims, ("D_qk", "D", "d_dim", "dk"))
    if d_qk is None:
        d_qk = (d_nope + d_rope) if (d_nope is not None and d_rope is not None) else (int(shapes[0][-1]) if shapes[0] else 0)
    d_v = _optional_dim(dims, ("D_v", "dv")) or d_nope or d_qk
    if not (segments := [v for v in (d_nope, d_rope, d_qk, d_v) if v is not None and v > 0]):
        raise ValueError("attention requires a positive head dimension")
    batch = _optional_dim(dims, ("B", "b_dim")) or 1
    critical_m = compute_critical_cube_tile_m(ctx.facts, _cube_dtype_name(ctx.dtype_bytes))
    return CasePhysicsInput(
        case_id=case_id, kind="attention", stride_ratio=4.0 if ctx.strided_gather else (2.0 if layout == "BNSD" else 1.0),
        atom_bytes=ctx.atom_bytes, sram_capacity_bytes=ctx.sram_capacity_bytes,
        inner_contiguous_bytes=max(1, int(round(min(segments) * ctx.dtype_bytes))), working_set_bytes=min(s_q, 64) * s_kv * 4.0,
        vreg_extent_elements=s_kv, outer_tiles=batch * max(1, math.ceil(s_q / 16)), physical_cores=ctx.physical_cores,
        oi_ridge_ratio=s_q / float(critical_m), has_empty_group=ctx.has_empty_group,
    )


def derive_case_physics_input(
    case_id: int, shapes: Sequence[Sequence[int]], attrs: Mapping[str, object], dims: Mapping[str, int], context: CasePhysicsContext
) -> CasePhysicsInput:
    """Derive raw physical quantities for one case from structural dims and input shapes."""
    if not shapes or not shapes[0]:
        raise ValueError("case physics require at least one non-empty input shape")
    primary = [int(d) for d in shapes[0]]
    if context.kind == "attention":
        return _derive_attention_input(case_id, shapes, attrs, dims, context)
    if context.kind == "matmul":
        return _derive_matmul_input(case_id, dims, context)
    if context.kind == "reduction":
        return _derive_reduction_input(case_id, primary, attrs, context)
    return _derive_elementwise_input(case_id, primary, context)


@dataclass(frozen=True)
class ConstraintRelaxationMask(JsonRecordMixin):
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
        return cls(**dict.fromkeys(VALID_FREEDOM_FACETS, True))

    @property
    def is_unrestricted_top(self) -> bool:
        """Return True iff all 7 AST partition facets are unlocked (zero shell syntax locks)."""
        return all(bool(getattr(self, f)) for f in VALID_FREEDOM_FACETS)

    def _validate_facet(self, facet: str) -> None:
        if facet not in VALID_FREEDOM_FACETS:
            raise ValueError(f"Unknown freedom facet {facet!r}; expected one of {VALID_FREEDOM_FACETS}")

    def is_unlocked(self, facet: str) -> bool:
        """Return whether *facet* is currently unlocked."""
        self._validate_facet(facet)
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
            self._validate_facet(facet)
            updates[facet] = True
        return ConstraintRelaxationMask(**updates)

    def join(self, other: ConstraintRelaxationMask) -> ConstraintRelaxationMask:
        """Return lattice join (elementwise OR) of self and other."""
        return ConstraintRelaxationMask(**{f: bool(getattr(self, f) or getattr(other, f)) for f in VALID_FREEDOM_FACETS})

    def meet(self, other: ConstraintRelaxationMask) -> ConstraintRelaxationMask:
        """Return lattice meet (elementwise AND) of self and other."""
        return ConstraintRelaxationMask(**{f: bool(getattr(self, f) and getattr(other, f)) for f in VALID_FREEDOM_FACETS})


@dataclass(frozen=True)
class RegimeSlice(JsonRecordMixin):
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
        _require_nonempty_str(self.regime_tag, "RegimeSlice.regime_tag")
        object.__setattr__(self, "case_nums", _as_int_tuple(self.case_nums, "case_nums"))
        object.__setattr__(self, "representative_dims", _as_int_dict(self.representative_dims, "representative_dims"))
        object.__setattr__(self, "has_unaligned_tail", bool(self.has_unaligned_tail))
        object.__setattr__(self, "has_empty_group", bool(self.has_empty_group))
        ref_perf, t_hw = float(self.ref_perf_us), float(self.t_hw_us)
        if ref_perf < 0.0 or t_hw < 0.0:
            raise ValueError("RegimeSlice timings ref_perf_us and t_hw_us must be non-negative")
        object.__setattr__(self, "ref_perf_us", ref_perf)
        object.__setattr__(self, "t_hw_us", t_hw)
        ratios = tuple(self.case_ratios)
        if any(not isinstance(r, CasePhysicsRatios) for r in ratios):
            raise ValueError("RegimeSlice.case_ratios must contain CasePhysicsRatios values")
        object.__setattr__(self, "case_ratios", ratios)
        if self.representative_case_id is not None and _require_int(self.representative_case_id, "RegimeSlice.representative_case_id") not in self.case_nums:
            raise ValueError("RegimeSlice.representative_case_id must belong to case_nums")
        object.__setattr__(self, "phase_coordinates", _as_str_tuple(self.phase_coordinates, "phase_coordinates"))
        object.__setattr__(self, "rationale", str(self.rationale))

    def representative_ratios(self) -> CasePhysicsRatios | None:
        """Return the physics ratios of the representative case."""
        if not self.case_ratios:
            return None
        if self.representative_case_id is not None:
            return next((r for r in self.case_ratios if r.case_id == self.representative_case_id), None)
        return self.case_ratios[len(self.case_ratios) // 2]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimeSlice:
        raw_ratios = payload.get("case_ratios") or ()
        if isinstance(raw_ratios, (str, bytes)) or not isinstance(raw_ratios, Sequence):
            raise ValueError("RegimeSlice.case_ratios must be a sequence")
        norm = dict(payload)
        norm["case_ratios"] = tuple(CasePhysicsRatios.from_dict(item) for item in raw_ratios if isinstance(item, Mapping))
        return super().from_dict(norm)


@dataclass(frozen=True)
class HardwareLegalityBounds(JsonRecordMixin):
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
        for name in (
            "l1_bytes", "l1_half_bank_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes",
            "ub_simdvf_bytes", "ub_unreserved_bytes", "ub_simtvf_bytes", "ssbuf_bytes",
            "max_mmad_mnk", "max_crosscore_depth", "hardware_mutex_count", "ub_banks", "ub_datablock_bytes",
        ):
            if (val := int(getattr(self, name))) <= 0:
                raise ValueError(f"HardwareLegalityBounds.{name} must be > 0, got {val}")
            object.__setattr__(self, name, val)
        if self.l1_half_bank_bytes > self.l1_bytes:
            raise ValueError("l1_half_bank_bytes cannot exceed l1_bytes")
        if self.ub_simtvf_bytes > self.ub_simdvf_bytes:
            raise ValueError("ub_simtvf_bytes cannot exceed ub_simdvf_bytes")
        if self.ub_simdvf_bytes > self.ub_unreserved_bytes:
            raise ValueError("ub_simtvf_bytes cannot exceed ub_unreserved_bytes")
        object.__setattr__(self, "forbidden_dma_paths", _as_str_tuple(self.forbidden_dma_paths, "forbidden_dma_paths"))
        object.__setattr__(self, "forbid_cube_int4", bool(self.forbid_cube_int4))


@dataclass(frozen=True)
class AdvisoryEnvelope(JsonRecordMixin):
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
        _require_nonempty_str(self.op_name, "AdvisoryEnvelope.op_name")
        _require_nonempty_str(self.cannbot_task, "AdvisoryEnvelope.cannbot_task")
        if not 1 <= (level := int(self.level)) <= 5:
            raise ValueError(f"AdvisoryEnvelope.level must be in 1..5, got {level}")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "topology_axes", _coerce_topology_axes_field(self.topology_axes, "AdvisoryEnvelope.topology_axes"))
        object.__setattr__(self, "input_dtypes", _as_str_tuple(self.input_dtypes, "input_dtypes"))
        object.__setattr__(self, "output_dtypes", _as_str_tuple(self.output_dtypes, "output_dtypes"))
        for mname in ("roofline", "structural_features"):
            if not isinstance(mval := getattr(self, mname), Mapping):
                raise ValueError(f"AdvisoryEnvelope.{mname} must be a mapping")
            object.__setattr__(self, mname, _json_clone(mval))
        if not (regimes_tuple := tuple(self.regimes)):
            raise ValueError("AdvisoryEnvelope.regimes must be non-empty")
        for item in regimes_tuple:
            if not isinstance(item, RegimeSlice):
                raise TypeError(f"Expected RegimeSlice in regimes, got {type(item).__name__}")
        object.__setattr__(self, "regimes", regimes_tuple)
        if not isinstance(self.legality_bounds, HardwareLegalityBounds):
            raise TypeError("legality_bounds must be HardwareLegalityBounds")
        object.__setattr__(self, "has_nan_inf_cases", bool(self.has_nan_inf_cases))
        object.__setattr__(self, "has_inplace_aliasing", bool(self.has_inplace_aliasing))
        object.__setattr__(self, "precision_thresholds", _as_float_dict(self.precision_thresholds, "precision_thresholds"))
        if self.hir_blueprint is not None:
            if not isinstance(self.hir_blueprint, Mapping):
                raise ValueError("hir_blueprint must be a mapping or None")
            object.__setattr__(self, "hir_blueprint", _json_clone(self.hir_blueprint))


@dataclass(frozen=True)
class SchedulePolicy(JsonRecordMixin):
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
        _require_nonempty_str(self.regime_tag, "SchedulePolicy.regime_tag")
        object.__setattr__(self, "topology_axes", _coerce_topology_axes_field(self.topology_axes, "SchedulePolicy.topology_axes"))
        if self.backend_tier not in VALID_BACKEND_TIERS:
            raise ValueError(f"SchedulePolicy.backend_tier {self.backend_tier!r} must be one of {VALID_BACKEND_TIERS}")
        if self.vector_mode not in VALID_VECTOR_MODES:
            raise ValueError(f"SchedulePolicy.vector_mode {self.vector_mode!r} must be one of {VALID_VECTOR_MODES}")
        tile = _as_int_tuple(self.tile_mnk, "tile_mnk")
        if len(tile) != 3 or any(d <= 0 for d in tile):
            raise ValueError(f"SchedulePolicy.tile_mnk must be 3 positive ints, got {tile}")
        object.__setattr__(self, "tile_mnk", tile)
        for name in ("row_pack_factor", "num_kv_splits", "l1_stages", "l0c_buffers", "swizzle_window", "multi_row_unroll", "reduce_accumulators"):
            if (val := int(getattr(self, name))) < 1:
                raise ValueError(f"SchedulePolicy.{name} must be >= 1, got {val}")
            object.__setattr__(self, name, val)
        for name in ("ub_estimated_bytes", "l1_estimated_bytes", "ub_bank_pad_datablocks", "ub_rw_shift_datablocks", "ub_row_stride_bytes", "preload_depth", "reverse_ack_period", "simt_threads"):
            if (val := int(getattr(self, name))) < 0:
                raise ValueError(f"SchedulePolicy.{name} must be >= 0, got {val}")
            object.__setattr__(self, name, val)
        for name in ("l1_half_bank_split", "ub_256kb_unlock", "ensure_empty_subblock_sync", "tail_wave_split", "direct_no_fusion_vf", "hf32_allowed", "fast_unshifted_exp"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("cv_datapath", "sync_mode", "streamk_mode", "w4_mode", "cube_operand_dtype"):
            object.__setattr__(self, name, str(getattr(self, name)))
        object.__setattr__(self, "l1_slot_aliasing", _as_str_dict(self.l1_slot_aliasing, "l1_slot_aliasing"))
        object.__setattr__(self, "l2_bypass_operands", _as_str_tuple(self.l2_bypass_operands, "l2_bypass_operands"))
        object.__setattr__(self, "dma_paths_used", _as_str_tuple(self.dma_paths_used, "dma_paths_used"))
        if isinstance(self.freedom_mask, Mapping):
            object.__setattr__(self, "freedom_mask", ConstraintRelaxationMask.from_dict(self.freedom_mask))
        elif not isinstance(self.freedom_mask, ConstraintRelaxationMask):
            raise TypeError(f"SchedulePolicy.freedom_mask must be ConstraintRelaxationMask or Mapping, got {type(self.freedom_mask).__name__}")


@dataclass(frozen=True)
class PolicyEnvelope(JsonRecordMixin):
    """Operator-wide collection of per-regime schedule policies."""

    op_name: str
    policies: dict[str, SchedulePolicy]
    selected_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_str(self.op_name, "PolicyEnvelope.op_name")
        if not isinstance(self.policies, Mapping) or not self.policies:
            raise ValueError("PolicyEnvelope.policies must be a non-empty mapping")
        normalized: dict[str, SchedulePolicy] = {}
        for key, policy in self.policies.items():
            if not isinstance(policy, SchedulePolicy):
                raise TypeError(f"PolicyEnvelope.policies[{key!r}] must be SchedulePolicy, got {type(policy).__name__}")
            if (tag := str(key)) != policy.regime_tag:
                raise ValueError(f"Policy key {tag!r} does not match policy.regime_tag {policy.regime_tag!r}")
            normalized[tag] = policy
        object.__setattr__(self, "policies", normalized)
        object.__setattr__(self, "selected_rules", _as_str_tuple(self.selected_rules, "selected_rules"))


@dataclass(frozen=True)
class VerificationReport(JsonRecordMixin):
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


@dataclass(frozen=True)
class TelemetryFeedback(JsonRecordMixin):
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
        for name in ("stage_a_passed", "stage_b_passed", "stage_c_passed", "vloopv2_hit"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if (spill := int(self.spill_bytes)) < 0:
            raise ValueError(f"TelemetryFeedback.spill_bytes must be >= 0, got {spill}")
        object.__setattr__(self, "spill_bytes", spill)
        for name in ("measured_latency_us", "eq3_score", "hap_reward", "bank_conflict_ratio", "l2_hit_rate"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "pipe_utilization", _as_float_dict(self.pipe_utilization, "pipe_utilization"))


@dataclass(frozen=True)
class SchedulingEnvelope(JsonRecordMixin):
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
        if self.verification is not None and not isinstance(self.verification, VerificationReport):
            raise TypeError("verification must be VerificationReport or None")
        if self.telemetry is not None and not isinstance(self.telemetry, TelemetryFeedback):
            raise TypeError("telemetry must be TelemetryFeedback or None")

    def save_json(self, path: str | Path) -> Path:
        atomic_write_text(target := Path(path), self.to_json(indent=2) + "\n")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> SchedulingEnvelope:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
