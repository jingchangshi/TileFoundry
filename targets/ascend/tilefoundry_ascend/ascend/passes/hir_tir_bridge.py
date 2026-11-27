"""Pure-Python analytical Roofline and blueprint packets for Ascend targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin


@dataclass(frozen=True)
class HirRooflineSummary(JsonRecordMixin):
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HirRooflineSummary:
        """Reconstruct a HirRooflineSummary from a serialized dictionary."""
        norm = dict(payload)
        anchor_dims = {str(k): int(v) for k, v in dict(norm.get("anchor_dims") or {}).items()}
        norm["anchor_dims"] = anchor_dims
        if norm.get("requested_anchor_dims") is None:
            norm["requested_anchor_dims"] = dict(anchor_dims)
        if norm.get("resolved_hir_dims") is None:
            norm["resolved_hir_dims"] = dict(anchor_dims)
        return super().from_dict(norm)

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
class BlueprintProvenance(JsonRecordMixin):
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


@dataclass(frozen=True)
class HirBlueprintPacket(JsonRecordMixin):
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
        norm = dict(payload)
        norm["roofline"] = roofline
        if not isinstance(norm.get("requested_anchor_dims"), Mapping):
            norm["requested_anchor_dims"] = dict(roofline.requested_anchor_dims)
        if not isinstance(norm.get("resolved_hir_dims"), Mapping):
            norm["resolved_hir_dims"] = dict(roofline.resolved_hir_dims)
        if "shape_exact_match" not in norm:
            norm["shape_exact_match"] = roofline.shape_exact_match
        if "fallback_reason" not in norm:
            norm["fallback_reason"] = roofline.fallback_reason
        return super().from_dict(norm)

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
