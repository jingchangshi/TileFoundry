"""Pure-Python analytical Roofline and blueprint packets for Ascend targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class HirRooflineSummary:
    """Analytical compute and HBM Roofline summary derived from PyTorch semantics."""

    anchor_dims: dict[str, int]
    compute_flops: int
    min_gm_bytes: int
    arithmetic_intensity: float
    compute_floor_us: float
    memory_floor_us: float
    ideal_roofline_us: float
    dominant_bound: str
    requested_anchor_dims: dict[str, int] = field(default_factory=dict)
    resolved_hir_dims: dict[str, int] = field(default_factory=dict)
    shape_exact_match: bool = True
    fallback_reason: str | None = None
    cube_flops: int = 0
    vector_flops: int = 0
    special_ops: int = 0
    cube_floor_us: float = 0.0
    vec_floor_us: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "anchor_dims": {str(k): int(v) for k, v in self.anchor_dims.items()},
            "requested_anchor_dims": {
                str(k): int(v) for k, v in self.requested_anchor_dims.items()
            },
            "resolved_hir_dims": {str(k): int(v) for k, v in self.resolved_hir_dims.items()},
            "shape_exact_match": bool(self.shape_exact_match),
            "fallback_reason": self.fallback_reason,
            "compute_flops": int(self.compute_flops),
            "cube_flops": int(self.cube_flops),
            "vector_flops": int(self.vector_flops),
            "special_ops": int(self.special_ops),
            "min_gm_bytes": int(self.min_gm_bytes),
            "arithmetic_intensity": float(self.arithmetic_intensity),
            "compute_floor_us": float(self.compute_floor_us),
            "cube_floor_us": float(self.cube_floor_us),
            "vec_floor_us": float(self.vec_floor_us),
            "memory_floor_us": float(self.memory_floor_us),
            "ideal_roofline_us": float(self.ideal_roofline_us),
            "dominant_bound": str(self.dominant_bound),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HirRooflineSummary:
        """Reconstruct a HirRooflineSummary from a serialized dictionary."""
        raw_dims = payload.get("anchor_dims") or {}
        anchor_dims = {str(k): int(v) for k, v in dict(raw_dims).items()}
        raw_req = payload.get("requested_anchor_dims")
        raw_res = payload.get("resolved_hir_dims")
        requested_dims = (
            {str(k): int(v) for k, v in dict(raw_req).items()}
            if raw_req is not None
            else dict(anchor_dims)
        )
        resolved_dims = (
            {str(k): int(v) for k, v in dict(raw_res).items()}
            if raw_res is not None
            else dict(anchor_dims)
        )
        raw_fallback = payload.get("fallback_reason")
        return cls(
            anchor_dims=anchor_dims,
            compute_flops=int(payload["compute_flops"]),
            min_gm_bytes=int(payload["min_gm_bytes"]),
            arithmetic_intensity=float(payload["arithmetic_intensity"]),
            compute_floor_us=float(payload["compute_floor_us"]),
            memory_floor_us=float(payload["memory_floor_us"]),
            ideal_roofline_us=float(payload["ideal_roofline_us"]),
            dominant_bound=str(payload["dominant_bound"]),
            requested_anchor_dims=requested_dims,
            resolved_hir_dims=resolved_dims,
            shape_exact_match=bool(payload.get("shape_exact_match", True)),
            fallback_reason=str(raw_fallback) if raw_fallback is not None else None,
            cube_flops=int(payload.get("cube_flops", 0)),
            vector_flops=int(payload.get("vector_flops", 0)),
            special_ops=int(payload.get("special_ops", 0)),
            cube_floor_us=float(payload.get("cube_floor_us", 0.0)),
            vec_floor_us=float(payload.get("vec_floor_us", 0.0)),
        )

    def format_for_prompt(self) -> str:
        """Format the Roofline summary as compact prompt text."""
        display_dims = self.resolved_hir_dims or self.anchor_dims
        dims = ", ".join(f"{k}={v}" for k, v in sorted(display_dims.items())) or "default"
        mismatch = ""
        if not self.shape_exact_match:
            mismatch = f", shape_exact_match=False, requested={self.requested_anchor_dims}"
        if self.fallback_reason:
            mismatch += f", fallback={self.fallback_reason}"
        return (
            f"Roofline[{dims}]: flops={self.compute_flops} "
            f"(cube={self.cube_flops}, vec={self.vector_flops}, special={self.special_ops}), "
            f"min_gm_bytes={self.min_gm_bytes}, AI={self.arithmetic_intensity:.4f}, "
            f"compute_floor_us={self.compute_floor_us!r}, "
            f"memory_floor_us={self.memory_floor_us!r}, "
            f"ideal_roofline_us={self.ideal_roofline_us!r}, "
            f"dominant_bound={self.dominant_bound}{mismatch}"
        )


def _parse_semantic_source(raw: object) -> Literal["pytorch_golden"]:
    value = str(raw)
    if value == "pytorch_golden":
        return "pytorch_golden"
    raise ValueError(f"semantic_source must be 'pytorch_golden', got {value!r}")


def _parse_traffic_basis(raw: object) -> Literal["declared_io_exact", "estimated_shape_dtype"]:
    value = str(raw)
    if value in ("declared_io_exact", "provable_lower_bound"):
        return "declared_io_exact"
    if value in ("estimated_shape_dtype", "estimated"):
        return "estimated_shape_dtype"
    raise ValueError(
        "traffic_basis must be 'declared_io_exact' or 'estimated_shape_dtype', "
        f"got {value!r}"
    )


def _parse_coordinate_scope(raw: object) -> Literal["representative_anchor"]:
    value = str(raw)
    if value == "representative_anchor":
        return "representative_anchor"
    raise ValueError(f"coordinate_scope must be 'representative_anchor', got {value!r}")


@dataclass(frozen=True)
class BlueprintProvenance:
    """Provenance metadata for unified semantic-first operator blueprints."""

    semantic_source: Literal["pytorch_golden"] = "pytorch_golden"
    traffic_basis: Literal["declared_io_exact", "estimated_shape_dtype"] = "declared_io_exact"
    coordinate_scope: Literal["representative_anchor"] = "representative_anchor"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "semantic_source", _parse_semantic_source(self.semantic_source)
        )
        object.__setattr__(self, "traffic_basis", _parse_traffic_basis(self.traffic_basis))
        object.__setattr__(
            self, "coordinate_scope", _parse_coordinate_scope(self.coordinate_scope)
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize provenance metadata to a JSON-compatible mapping."""
        return {
            "semantic_source": self.semantic_source,
            "traffic_basis": self.traffic_basis,
            "coordinate_scope": self.coordinate_scope,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BlueprintProvenance:
        """Reconstruct a BlueprintProvenance from a serialized dictionary."""
        return cls(
            semantic_source=_parse_semantic_source(
                payload.get("semantic_source", "pytorch_golden")
            ),
            traffic_basis=_parse_traffic_basis(
                payload.get("traffic_basis", "declared_io_exact")
            ),
            coordinate_scope=_parse_coordinate_scope(
                payload.get("coordinate_scope", "representative_anchor")
            ),
        )


@dataclass(frozen=True)
class HirBlueprintPacket:
    """Dialect-neutral semantic Roofline and blueprint packet."""

    entry_name: str
    roofline: HirRooflineSummary
    lowering_diagnostics: tuple[str, ...] = ()
    requested_anchor_dims: dict[str, int] = field(default_factory=dict)
    resolved_hir_dims: dict[str, int] = field(default_factory=dict)
    shape_exact_match: bool = True
    fallback_reason: str | None = None
    provenance: BlueprintProvenance | None = None

    def __post_init__(self) -> None:
        req_dims = {str(k): int(v) for k, v in self.requested_anchor_dims.items()}
        res_dims = {str(k): int(v) for k, v in self.resolved_hir_dims.items()}
        object.__setattr__(self, "entry_name", str(self.entry_name))
        object.__setattr__(
            self,
            "lowering_diagnostics",
            tuple(str(item) for item in self.lowering_diagnostics),
        )
        object.__setattr__(self, "requested_anchor_dims", req_dims)
        object.__setattr__(self, "resolved_hir_dims", res_dims)
        object.__setattr__(self, "shape_exact_match", bool(self.shape_exact_match))
        object.__setattr__(
            self,
            "fallback_reason",
            str(self.fallback_reason) if self.fallback_reason is not None else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "entry_name": self.entry_name,
            "roofline": self.roofline.to_dict(),
            "lowering_diagnostics": list(self.lowering_diagnostics),
            "requested_anchor_dims": {
                str(k): int(v) for k, v in self.requested_anchor_dims.items()
            },
            "resolved_hir_dims": {str(k): int(v) for k, v in self.resolved_hir_dims.items()},
            "shape_exact_match": bool(self.shape_exact_match),
            "fallback_reason": self.fallback_reason,
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HirBlueprintPacket:
        """Reconstruct a HirBlueprintPacket from a serialized dictionary."""
        raw_roofline = payload["roofline"]
        if isinstance(raw_roofline, HirRooflineSummary):
            roofline = raw_roofline
        elif isinstance(raw_roofline, Mapping):
            roofline = HirRooflineSummary.from_dict(raw_roofline)
        else:
            raise TypeError(
                f"roofline must be HirRooflineSummary or Mapping, got {type(raw_roofline).__name__}"
            )
        raw_diags = payload.get("lowering_diagnostics") or ()
        raw_req = payload.get("requested_anchor_dims")
        raw_res = payload.get("resolved_hir_dims")
        raw_prov = payload.get("provenance")
        requested_dims = (
            {str(k): int(v) for k, v in dict(raw_req).items()}
            if isinstance(raw_req, Mapping)
            else dict(roofline.requested_anchor_dims)
        )
        resolved_dims = (
            {str(k): int(v) for k, v in dict(raw_res).items()}
            if isinstance(raw_res, Mapping)
            else dict(roofline.resolved_hir_dims)
        )
        shape_exact = (
            bool(payload["shape_exact_match"])
            if "shape_exact_match" in payload
            else roofline.shape_exact_match
        )
        if "fallback_reason" in payload:
            raw_fb = payload.get("fallback_reason")
            fallback_reason = str(raw_fb) if raw_fb is not None else None
        else:
            fallback_reason = roofline.fallback_reason
        if raw_prov is None or isinstance(raw_prov, BlueprintProvenance):
            provenance = raw_prov
        elif isinstance(raw_prov, Mapping):
            provenance = BlueprintProvenance.from_dict(raw_prov)
        else:
            raise TypeError(
                f"provenance must be BlueprintProvenance, Mapping, or None, "
                f"got {type(raw_prov).__name__}"
            )
        return cls(
            entry_name=str(payload["entry_name"]),
            roofline=roofline,
            lowering_diagnostics=tuple(str(item) for item in raw_diags),
            requested_anchor_dims=requested_dims,
            resolved_hir_dims=resolved_dims,
            shape_exact_match=shape_exact,
            fallback_reason=fallback_reason,
            provenance=provenance,
        )

    def format_for_prompt(self) -> str:
        """Render dialect-neutral Roofline and physical schedule metrics."""
        lines = [
            f"Entry: {self.entry_name}",
            self.roofline.format_for_prompt(),
        ]
        if self.provenance is not None:
            lines.append(
                f"Provenance: semantic_source={self.provenance.semantic_source}, "
                f"traffic_basis={self.provenance.traffic_basis}"
            )
        if not self.shape_exact_match or self.fallback_reason is not None:
            lines.append(
                f"Shape Provenance: shape_exact_match={self.shape_exact_match}, "
                f"resolved_hir_dims={self.resolved_hir_dims}, "
                f"requested_anchor_dims={self.requested_anchor_dims}"
            )
            if self.fallback_reason is not None:
                lines.append(f"Fallback Reason: {self.fallback_reason}")
        if self.lowering_diagnostics:
            lines.append("Lowering Diagnostics:")
            lines.extend(f"- {diag}" for diag in self.lowering_diagnostics)
        return "\n".join(lines)


__all__ = [
    "BlueprintProvenance",
    "HirBlueprintPacket",
    "HirRooflineSummary",
]
