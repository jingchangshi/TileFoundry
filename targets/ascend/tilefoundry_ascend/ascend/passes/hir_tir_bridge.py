"""Dialect-neutral HIR Roofline and Z3 blueprint compilation for Ascend targets."""

from __future__ import annotations

import ast
import inspect
import math
import runpy
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tilefoundry.analysis import AnalysisError, ComputeCostMetadata, analyze, check_program
from tilefoundry.analysis.check import program_dim_vars
from tilefoundry.analysis.iteration_scope import ScopeBuilder
from tilefoundry.analysis.visitor import AnalyzeContext
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, VerifyError, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.nn.conv2d import Conv2D
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.pipeline import (
    PipelineError,
    normalize_pipeline_references,
    rebind_cloned_pipeline_references,
)
from tilefoundry.ir.hir.schedule import ScheduleOp, ScheduleRefusal
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.types.utils import static_dim_value, tensor_types
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError
from tilefoundry.schedule.finalize import finalize
from tilefoundry.visitor_registry.contexts import ValuationContext
from tilefoundry.visitor_registry.valuation import ValuationVisitor
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_ascend.ascend.ir.tir.instructions import AscendMmad
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tir_envelope
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_SPECIAL_VECTOR_OP_EQUIV = 8


@dataclass(frozen=True)
class HirRooflineSummary:
    """Analytical compute and HBM Roofline summary derived from authored HIR."""

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


_Z3_LOOP_SUMMARY_KEYS = frozenset({"ii", "stages", "phases", "versions"})


def _filter_z3_pipeline_summary(raw: Mapping[str, object] | None) -> dict[str, object]:
    """Retain only genuine Z3 per-loop stage summary mappings."""
    if not isinstance(raw, Mapping):
        return {}
    cleaned: dict[str, object] = {}
    for key, val in raw.items():
        if isinstance(val, Mapping) and bool(_Z3_LOOP_SUMMARY_KEYS & set(val.keys())):
            cleaned[str(key)] = dict(val)
    return cleaned


def _parse_semantic_source(raw: object) -> Literal["pytorch_golden", "pytorch_golden_plus_hir"]:
    value = str(raw)
    if value == "pytorch_golden":
        return "pytorch_golden"
    if value == "pytorch_golden_plus_hir":
        return "pytorch_golden_plus_hir"
    raise ValueError(
        "semantic_source must be 'pytorch_golden' or 'pytorch_golden_plus_hir', "
        f"got {value!r}"
    )


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

    semantic_source: Literal["pytorch_golden", "pytorch_golden_plus_hir"]
    traffic_basis: Literal["declared_io_exact", "estimated_shape_dtype"]
    coordinate_scope: Literal["representative_anchor"] = "representative_anchor"
    has_authored_hir: bool = False
    has_scheduled_hir: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "semantic_source", _parse_semantic_source(self.semantic_source)
        )
        object.__setattr__(self, "traffic_basis", _parse_traffic_basis(self.traffic_basis))
        object.__setattr__(
            self, "coordinate_scope", _parse_coordinate_scope(self.coordinate_scope)
        )
        object.__setattr__(self, "has_authored_hir", bool(self.has_authored_hir))
        object.__setattr__(self, "has_scheduled_hir", bool(self.has_scheduled_hir))

    def to_dict(self) -> dict[str, object]:
        """Serialize provenance metadata to a JSON-compatible mapping."""
        return {
            "semantic_source": self.semantic_source,
            "traffic_basis": self.traffic_basis,
            "coordinate_scope": self.coordinate_scope,
            "has_authored_hir": self.has_authored_hir,
            "has_scheduled_hir": self.has_scheduled_hir,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BlueprintProvenance:
        """Reconstruct a BlueprintProvenance from a serialized dictionary."""
        return cls(
            semantic_source=_parse_semantic_source(payload["semantic_source"]),
            traffic_basis=_parse_traffic_basis(payload["traffic_basis"]),
            coordinate_scope=_parse_coordinate_scope(
                payload.get("coordinate_scope", "representative_anchor")
            ),
            has_authored_hir=bool(payload.get("has_authored_hir", False)),
            has_scheduled_hir=bool(payload.get("has_scheduled_hir", False)),
        )


@dataclass(frozen=True)
class BlueprintScheduleDelta:
    """Optional schedule enrichment extracted when authored or scheduled HIR exists."""

    unscheduled_ledger: ObjectivePhysicalLedger | None = None
    scheduled_ledger: ObjectivePhysicalLedger | None = None
    z3_stage_summary: dict[str, object] = field(default_factory=dict)
    recommended_facets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "z3_stage_summary",
            _filter_z3_pipeline_summary(self.z3_stage_summary),
        )
        object.__setattr__(
            self,
            "recommended_facets",
            tuple(str(item) for item in self.recommended_facets),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize optional HIR schedule enrichment to a JSON-compatible mapping."""
        return {
            "unscheduled_ledger": (
                self.unscheduled_ledger.to_dict()
                if self.unscheduled_ledger is not None
                else None
            ),
            "scheduled_ledger": (
                self.scheduled_ledger.to_dict()
                if self.scheduled_ledger is not None
                else None
            ),
            "z3_stage_summary": dict(self.z3_stage_summary),
            "recommended_facets": list(self.recommended_facets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BlueprintScheduleDelta:
        """Reconstruct a BlueprintScheduleDelta from a serialized dictionary."""
        raw_unsched = payload.get("unscheduled_ledger")
        raw_sched = payload.get("scheduled_ledger")
        raw_z3 = payload.get("z3_stage_summary")
        raw_facets = payload.get("recommended_facets")
        if raw_unsched is None or isinstance(raw_unsched, ObjectivePhysicalLedger):
            unsched = raw_unsched
        elif isinstance(raw_unsched, Mapping):
            unsched = ObjectivePhysicalLedger.from_dict(raw_unsched)
        else:
            raise TypeError(
                f"unscheduled_ledger must be Mapping or None, got {type(raw_unsched).__name__}"
            )
        if raw_sched is None or isinstance(raw_sched, ObjectivePhysicalLedger):
            sched = raw_sched
        elif isinstance(raw_sched, Mapping):
            sched = ObjectivePhysicalLedger.from_dict(raw_sched)
        else:
            raise TypeError(
                f"scheduled_ledger must be Mapping or None, got {type(raw_sched).__name__}"
            )
        z3_map = _filter_z3_pipeline_summary(raw_z3 if isinstance(raw_z3, Mapping) else None)
        facets = (
            tuple(str(item) for item in raw_facets)
            if isinstance(raw_facets, (list, tuple))
            else ()
        )
        return cls(
            unscheduled_ledger=unsched,
            scheduled_ledger=sched,
            z3_stage_summary=z3_map,
            recommended_facets=facets,
        )


@dataclass(frozen=True, init=False)
class HirBlueprintPacket:
    """Dialect-neutral semantic Roofline, optional HIR schedule delta, and physical ledger."""

    entry_name: str
    roofline: HirRooflineSummary
    reference_ledger: ObjectivePhysicalLedger | None = None
    lowering_diagnostics: tuple[str, ...] = ()
    requested_anchor_dims: dict[str, int] = field(default_factory=dict)
    resolved_hir_dims: dict[str, int] = field(default_factory=dict)
    shape_exact_match: bool = True
    fallback_reason: str | None = None
    provenance: BlueprintProvenance | None = None
    schedule_delta: BlueprintScheduleDelta | None = None
    _z3_pipeline_summary: dict[str, object] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        entry_name: str,
        roofline: HirRooflineSummary,
        z3_pipeline_summary: Mapping[str, object] | None = None,
        reference_ledger: ObjectivePhysicalLedger | None = None,
        lowering_diagnostics: tuple[str, ...] = (),
        requested_anchor_dims: Mapping[str, int] | None = None,
        resolved_hir_dims: Mapping[str, int] | None = None,
        shape_exact_match: bool = True,
        fallback_reason: str | None = None,
        provenance: BlueprintProvenance | None = None,
        schedule_delta: BlueprintScheduleDelta | None = None,
        *,
        _z3_pipeline_summary: Mapping[str, object] | None = None,
    ) -> None:
        raw_z3 = (
            _z3_pipeline_summary
            if _z3_pipeline_summary is not None
            else z3_pipeline_summary
        )
        stored_z3 = (
            dict(schedule_delta.z3_stage_summary)
            if schedule_delta is not None
            else _filter_z3_pipeline_summary(raw_z3)
        )
        req_dims = (
            {str(k): int(v) for k, v in requested_anchor_dims.items()}
            if requested_anchor_dims is not None
            else {}
        )
        res_dims = (
            {str(k): int(v) for k, v in resolved_hir_dims.items()}
            if resolved_hir_dims is not None
            else {}
        )
        object.__setattr__(self, "entry_name", str(entry_name))
        object.__setattr__(self, "roofline", roofline)
        object.__setattr__(self, "reference_ledger", reference_ledger)
        object.__setattr__(
            self,
            "lowering_diagnostics",
            tuple(str(item) for item in lowering_diagnostics),
        )
        object.__setattr__(self, "requested_anchor_dims", req_dims)
        object.__setattr__(self, "resolved_hir_dims", res_dims)
        object.__setattr__(self, "shape_exact_match", bool(shape_exact_match))
        object.__setattr__(
            self,
            "fallback_reason",
            str(fallback_reason) if fallback_reason is not None else None,
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "schedule_delta", schedule_delta)
        object.__setattr__(self, "_z3_pipeline_summary", stored_z3)

    @property
    def z3_pipeline_summary(self) -> dict[str, object]:
        """Return genuine Z3 stage summaries without non-Z3 metadata keys."""
        if self.schedule_delta is not None:
            return dict(self.schedule_delta.z3_stage_summary)
        return dict(self._z3_pipeline_summary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "entry_name": self.entry_name,
            "roofline": self.roofline.to_dict(),
            "z3_pipeline_summary": dict(self.z3_pipeline_summary),
            "reference_ledger": (
                self.reference_ledger.to_dict() if self.reference_ledger is not None else None
            ),
            "lowering_diagnostics": list(self.lowering_diagnostics),
            "requested_anchor_dims": {
                str(k): int(v) for k, v in self.requested_anchor_dims.items()
            },
            "resolved_hir_dims": {str(k): int(v) for k, v in self.resolved_hir_dims.items()},
            "shape_exact_match": bool(self.shape_exact_match),
            "fallback_reason": self.fallback_reason,
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
            "schedule_delta": (
                self.schedule_delta.to_dict() if self.schedule_delta is not None else None
            ),
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
        raw_ledger = payload.get("reference_ledger")
        raw_z3 = payload.get("z3_pipeline_summary")
        raw_diags = payload.get("lowering_diagnostics") or ()
        raw_req = payload.get("requested_anchor_dims")
        raw_res = payload.get("resolved_hir_dims")
        raw_prov = payload.get("provenance")
        raw_delta = payload.get("schedule_delta")
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
        if raw_delta is None or isinstance(raw_delta, BlueprintScheduleDelta):
            schedule_delta = raw_delta
        elif isinstance(raw_delta, Mapping):
            schedule_delta = BlueprintScheduleDelta.from_dict(raw_delta)
        else:
            raise TypeError(
                f"schedule_delta must be BlueprintScheduleDelta, Mapping, or None, "
                f"got {type(raw_delta).__name__}"
            )
        if raw_ledger is None or isinstance(raw_ledger, ObjectivePhysicalLedger):
            ref_ledger = raw_ledger
        elif isinstance(raw_ledger, Mapping):
            ref_ledger = ObjectivePhysicalLedger.from_dict(raw_ledger)
        else:
            raise TypeError(
                f"reference_ledger must be ObjectivePhysicalLedger, Mapping, or None, "
                f"got {type(raw_ledger).__name__}"
            )
        z3_map = _filter_z3_pipeline_summary(raw_z3 if isinstance(raw_z3, Mapping) else None)
        return cls(
            entry_name=str(payload["entry_name"]),
            roofline=roofline,
            z3_pipeline_summary=z3_map,
            reference_ledger=ref_ledger,
            lowering_diagnostics=tuple(str(item) for item in raw_diags),
            requested_anchor_dims=requested_dims,
            resolved_hir_dims=resolved_dims,
            shape_exact_match=shape_exact,
            fallback_reason=fallback_reason,
            provenance=provenance,
            schedule_delta=schedule_delta,
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
                f"traffic_basis={self.provenance.traffic_basis}, "
                f"has_authored_hir={self.provenance.has_authored_hir}, "
                f"has_scheduled_hir={self.provenance.has_scheduled_hir}"
            )
        if not self.shape_exact_match or self.fallback_reason is not None:
            lines.append(
                f"Shape Provenance: shape_exact_match={self.shape_exact_match}, "
                f"resolved_hir_dims={self.resolved_hir_dims}, "
                f"requested_anchor_dims={self.requested_anchor_dims}"
            )
            if self.fallback_reason is not None:
                lines.append(f"Fallback Reason: {self.fallback_reason}")
        lines.append(f"Z3 Pipelines: {self.z3_pipeline_summary}")
        if self.reference_ledger is not None:
            lines.append(
                f"Reference Ledger: mode={self.reference_ledger.analysis_mode}, "
                f"legal={self.reference_ledger.is_legal}, "
                f"dominant_pipe={self.reference_ledger.dominant_pipe}, "
                f"overflow={self.reference_ledger.max_overflow_bytes}"
            )
        if self.lowering_diagnostics:
            lines.append("Lowering Diagnostics:")
            lines.extend(f"- {diag}" for diag in self.lowering_diagnostics)
        return "\n".join(lines)


@dataclass(frozen=True)
class _LoadedModuleInfo:
    """Loaded HIR module plus parameterized-build provenance."""

    module: Module
    parameterized_applied: bool
    fallback_reason: str | None
    build_param_names: frozenset[str]
    source: str
    path: Path


def _load_module_from_source(
    source: str,
    tmpdir: Path,
    filename: str,
    *,
    anchor_dims: Mapping[str, int] | None,
) -> _LoadedModuleInfo:
    """Write HIR source to *tmpdir* and load via build() factory or load_authored_ir."""
    path = tmpdir / filename
    path.write_text(source, encoding="utf-8")
    fallback_reason: str | None = None
    build_param_names: set[str] = set()
    try:
        ns = runpy.run_path(str(path))
    except (VerifyError, ValueError, AssertionError, TypeError):
        raise

    build_fn = ns.get("build")
    if callable(build_fn):
        sig = inspect.signature(build_fn)
        build_param_names = {
            str(k)
            for k, param in sig.parameters.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        if anchor_dims:
            matched = {
                str(k): int(v)
                for k, v in anchor_dims.items()
                if str(k) in build_param_names
            }
            if matched:
                try:
                    built = build_fn(**matched)
                    if isinstance(built, Module):
                        return _LoadedModuleInfo(
                            module=built,
                            parameterized_applied=True,
                            fallback_reason=None,
                            build_param_names=frozenset(build_param_names),
                            source=source,
                            path=path,
                        )
                except (VerifyError, ValueError, AssertionError, TypeError):
                    raise

    return _LoadedModuleInfo(
        module=load_authored_ir(str(path)),
        parameterized_applied=False,
        fallback_reason=fallback_reason,
        build_param_names=frozenset(build_param_names),
        source=source,
        path=path,
    )


def _tensor_shape_expr_names(annotation: ast.expr | None) -> tuple[str | None, ...]:
    """Return symbolic axis names from a Tensor[(d0, d1, ...), dtype] AST annotation."""
    if not isinstance(annotation, ast.Subscript):
        return ()
    slc = annotation.slice
    if isinstance(slc, ast.Tuple) and slc.elts:
        shape_node = slc.elts[0]
    else:
        return ()
    elts = shape_node.elts if isinstance(shape_node, ast.Tuple) else [shape_node]
    return tuple(elt.id if isinstance(elt, ast.Name) else None for elt in elts)


def _extract_ast_param_shape_names(source: str, fn_name: str) -> dict[str, tuple[str | None, ...]]:
    """Map parameter names of *fn_name* in *source* to their symbolic tensor axis names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    mapping: dict[str, tuple[str | None, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != fn_name:
            continue
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            names = _tensor_shape_expr_names(arg.annotation)
            if names:
                mapping.setdefault(arg.arg, names)
    return mapping


def _extract_resolved_hir_dims(
    source: str,
    authored_fn,
    checked_fn,
    build_param_names: frozenset[str],
) -> dict[str, int]:
    """Extract resolved static dimension bindings directly from *checked_fn* parameters."""
    resolved: dict[str, int] = {}
    for p_before, p_after in zip(authored_fn.params, checked_fn.params, strict=False):
        if not isinstance(p_before.type, TensorType) or not isinstance(p_after.type, TensorType):
            continue
        for d_before, d_after in zip(p_before.type.shape, p_after.type.shape, strict=False):
            dim_name = getattr(d_before, "name", None)
            val = static_dim_value(d_after)
            if dim_name and val is not None:
                resolved.setdefault(str(dim_name), int(val))

    ast_shapes = _extract_ast_param_shape_names(source, str(checked_fn.name))
    for param in checked_fn.params:
        if not isinstance(param.type, TensorType):
            continue
        axis_names = ast_shapes.get(str(param.name), ())
        for axis_name, dim_val in zip(axis_names, param.type.shape, strict=False):
            if axis_name is None:
                continue
            if build_param_names and axis_name not in build_param_names:
                continue
            val = static_dim_value(dim_val)
            if val is not None:
                resolved.setdefault(axis_name, int(val))
    return resolved


def _evaluate_shape_match(
    requested_dims: Mapping[str, int],
    resolved_dims: Mapping[str, int],
    *,
    fallback_reason: str | None,
    parameterized_applied: bool,
    declared_dimvars: set[str],
    build_param_names: frozenset[str],
) -> bool:
    """Return True only when parameterized resolution succeeded and all resolvable dims match."""
    if fallback_reason is not None:
        return False
    if not requested_dims:
        return True
    resolvable = set(resolved_dims.keys()) | declared_dimvars | set(build_param_names)
    dimvar_bound = any(str(k) in declared_dimvars for k in requested_dims)
    if not (parameterized_applied or dimvar_bound):
        return False
    for key, val in requested_dims.items():
        if key in resolvable and resolved_dims.get(key) != int(val):
            return False
    return True


def _check_with_filtered_dims(
    mod: Module,
    *,
    anchor_dims: Mapping[str, int] | None,
):
    """Specialize declared DimVars and return checked module, fn, authored fn, and DimVars."""
    normalize_pipeline_references(mod)
    authored_fn = mod.entry_function()
    declared = set(program_dim_vars(mod, authored_fn).keys()) | set(
        dim_vars_by_name(mod.effective_topologies())
    )
    filtered = (
        {str(k): int(v) for k, v in anchor_dims.items() if str(k) in declared}
        if anchor_dims
        else {}
    )
    checked_mod, checked_fn = check_program(
        mod, authored_fn, dims=filtered if filtered else None
    )
    rebind_cloned_pipeline_references(checked_mod)
    return checked_mod, checked_fn, authored_fn, declared


def _tensor_static_bytes(ttype: TensorType) -> int:
    """Return static byte footprint for a TensorType, or 0 if dynamic."""
    dims: list[int] = []
    for dim in ttype.shape:
        val = static_dim_value(dim)
        if val is None or val < 0:
            return 0
        dims.append(val)
    elems = math.prod(dims) if dims else 1
    return (elems * ttype.dtype.bit_width + 7) // 8


def _is_cube_matrix_op(op: object) -> bool:
    """Return True if *op* executes on AIC Cube matrix units rather than AIV vector units."""
    if isinstance(op, (MatMul, Conv2D)):
        return True
    if isinstance(op, ScheduleOp):
        return isinstance(op.op, AscendMmad)
    return False


class _WorkSplitVisitor(ValuationVisitor[None]):
    """Accumulate loop-multiplied Cube FLOPs, Vector FLOPs, and non-FLOP service ops."""

    def __init__(
        self,
        callee_totals: Mapping[int, tuple[dict[str, int], dict[str, int], dict[str, int]]],
    ) -> None:
        super().__init__()
        self.callee_totals = callee_totals
        self.cube_flops: dict[str, int] = {}
        self.vec_flops: dict[str, int] = {}
        self.other_ops: dict[str, int] = {}

    def default_visit_leaf(self, expr, _operands, ctx) -> None:
        if not isinstance(expr, Call):
            return
        repeats = ctx.scope.repeats_of(expr, ctx.topology_level)
        trips = int(repeats.loop_trips)
        callee = ctx.callee
        if callee is not None and id(callee) in self.callee_totals:
            c_cube, c_vec, c_other = self.callee_totals[id(callee)]
            for k, v in c_cube.items():
                self.cube_flops[k] = self.cube_flops.get(k, 0) + v * trips
            for k, v in c_vec.items():
                self.vec_flops[k] = self.vec_flops.get(k, 0) + v * trips
            for k, v in c_other.items():
                self.other_ops[k] = self.other_ops.get(k, 0) + v * trips
            return

        record = get_metadata(expr, ComputeCostMetadata)
        if record is None:
            return
        dest_flops = self.cube_flops if _is_cube_matrix_op(expr.target) else self.vec_flops
        for dtype_name, spread in record.flops.kinds:
            dest_flops[dtype_name] = dest_flops.get(dtype_name, 0) + int(spread.total) * trips
        for kind, spread in record.other_ops.kinds:
            self.other_ops[kind] = self.other_ops.get(kind, 0) + v_total(spread) * trips


def v_total(spread) -> int:
    """Return integer total from a ComputeCost spread entry."""
    return int(spread.total)


def _split_cube_and_vector_work(
    checked_mod: Module,
    checked_fn,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Split function compute cost into Cube FLOPs, Vector FLOPs, and service ops."""
    topologies = checked_mod.effective_topologies()
    topology_level = topologies[0].name if topologies else None
    definitions: dict[int, object] = {}
    contexts: list[AnalyzeContext] = []
    frames = tuple(
        ValuationVisitor.definitions(
            ValuationContext(checked_mod, checked_fn), postorder=True
        )
    )
    for frame in frames:
        context = AnalyzeContext(
            module=frame.module,
            function=frame.function,
            target=frame.target,
            topologies=frame.topologies,
            owner=frame.function,
            topology_level=topology_level,
            entry=checked_fn,
            definitions=definitions,
        )
        context.scope = ScopeBuilder(
            frame.module,
            frame.function,
            ctx=context,
            is_root=frame.function is checked_fn,
        ).build()
        definitions[id(frame.function)] = context.scope
        contexts.append(context)

    per_fn: dict[int, tuple[dict[str, int], dict[str, int], dict[str, int]]] = {}
    for context in contexts:
        visitor = _WorkSplitVisitor(per_fn)
        visitor.visit(context.function.body, context)
        per_fn[id(context.function)] = (visitor.cube_flops, visitor.vec_flops, visitor.other_ops)

    if id(checked_fn) in per_fn:
        return per_fn[id(checked_fn)]
    cc = get_metadata(checked_fn, ComputeCostMetadata)
    if cc is None:
        return {}, {}, {}
    return (
        {},
        {name: int(spread.total) for name, spread in cc.flops.kinds},
        {kind: int(spread.total) for kind, spread in cc.other_ops.kinds},
    )


_CUBE_DTYPE_ALIASES: dict[str, str] = {
    "fp16": "f16",
    "float16": "f16",
    "bfloat16": "bf16",
    "fp32": "f32",
    "float32": "f32",
    "fp8": "fp8e4m3",
    "float8_e4m3fn": "fp8e4m3",
    "float8_e5m2": "fp8e5m2",
    "hifp8": "hif8",
    "int8": "i8",
    "uint8": "u8",
    "fp4": "f4e2m1",
    "mxfp4": "f4e2m1",
    "float4_e2m1fn": "f4e2m1",
}


def _resolve_aic_rate(
    dtype_name: str,
    aic_rates: Mapping[DType, int],
    *,
    spaces: AscendSpaceFacts,
    aic_bf16_rate: int | float,
    fallback_rate: int | float,
) -> float:
    """Resolve AIC FLOP/s for *dtype_name* with dav-3510 precision scaling vs BF16."""
    dt = getattr(DType, dtype_name, None)
    direct = aic_rates.get(dt) if dt is not None else None
    if direct is not None and direct > 0:
        return float(direct)
    norm_key = _CUBE_DTYPE_ALIASES.get(dtype_name.strip().lower(), dtype_name.strip().lower())
    if aic_bf16_rate and aic_bf16_rate > 0:
        try:
            bf16_macs = float(spaces.cube_macs_per_cycle("bf16"))
            macs = float(spaces.cube_macs_per_cycle(norm_key))
            if bf16_macs > 0.0 and macs > 0.0:
                return float(aic_bf16_rate) * (macs / bf16_macs)
        except ValueError:
            pass
    return float(fallback_rate)


def _compute_roofline_summary(
    checked_mod: Module,
    checked_fn,
    *,
    requested_anchor_dims: Mapping[str, int],
    resolved_hir_dims: Mapping[str, int],
    shape_exact_match: bool,
    fallback_reason: str | None,
    target: AscendTarget,
) -> HirRooflineSummary:
    """Compute HirRooflineSummary separating AIC Cube and AIV Vector/transcendental work."""
    analyze(checked_mod, checked_fn, analysis=("compute-cost",))
    cube_flops_by_dtype, vec_flops_by_dtype, other_ops_by_kind = _split_cube_and_vector_work(
        checked_mod, checked_fn
    )
    cube_flops = sum(cube_flops_by_dtype.values())
    special_ops = int(other_ops_by_kind.get("special", 0))
    non_special_ops = sum(
        int(count) for kind, count in other_ops_by_kind.items() if kind != "special" and count > 0
    )
    vector_flops = sum(vec_flops_by_dtype.values()) + special_ops * _SPECIAL_VECTOR_OP_EQUIV
    compute_flops = cube_flops + vector_flops

    io_tensors: list[TensorType] = []
    for param in checked_fn.params:
        io_tensors.extend(tensor_types(param.type))
    ret_type = checked_fn.return_type or checked_fn.body.type
    if ret_type is not None:
        io_tensors.extend(tensor_types(ret_type))
    min_gm_bytes = sum(_tensor_static_bytes(ttype) for ttype in io_tensors)
    arithmetic_intensity = (
        round(compute_flops / float(min_gm_bytes), 4) if min_gm_bytes > 0 else 0.0
    )

    dev = target.device
    unit_flops = dev.unit_flops_per_second
    aic_rates = unit_flops.get("aic", {}) if getattr(dev, "aic_count", 0) > 0 else {}
    aiv_rates = unit_flops.get("aiv", {}) if getattr(dev, "aiv_count", 0) > 0 else {}
    aic_fallback = max(aic_rates.values(), default=0)
    aiv_vec_rate = (
        aiv_rates.get(DType.f32)
        or max(aiv_rates.values(), default=0)
        or int(getattr(dev, "aiv_lane_events_per_second", 0))
    )
    aiv_service_rate = int(getattr(dev, "aiv_lane_events_per_second", 0)) or aiv_vec_rate

    spaces = target.get_facts(AscendSpaceFacts)
    aic_bf16_rate = (
        aic_rates.get(DType.bf16)
        or aic_rates.get(DType.f16)
        or aic_fallback
    )

    cube_floor_s = 0.0
    for dtype_name, flop_count in cube_flops_by_dtype.items():
        if flop_count <= 0:
            continue
        rate = _resolve_aic_rate(
            dtype_name,
            aic_rates,
            spaces=spaces,
            aic_bf16_rate=aic_bf16_rate,
            fallback_rate=aic_fallback or aiv_vec_rate,
        )
        if rate > 0.0:
            cube_floor_s += flop_count / rate

    vec_floor_s = 0.0
    for dtype_name, flop_count in vec_flops_by_dtype.items():
        if flop_count <= 0:
            continue
        dt = getattr(DType, dtype_name, None)
        rate = aiv_rates.get(dt) if dt is not None else None
        if rate is None or rate <= 0:
            rate = aiv_vec_rate
        if rate and rate > 0:
            vec_floor_s += flop_count / float(rate)
    if special_ops > 0 and aiv_vec_rate > 0:
        vec_floor_s += (special_ops * _SPECIAL_VECTOR_OP_EQUIV) / float(aiv_vec_rate)
    if non_special_ops > 0 and aiv_service_rate > 0:
        vec_floor_s += non_special_ops / float(aiv_service_rate)

    cube_floor_us = cube_floor_s * 1e6
    vec_floor_us = vec_floor_s * 1e6
    compute_floor_us = vec_floor_us if cube_flops == 0 else max(cube_floor_us, vec_floor_us)

    hbm_bw = float(getattr(dev, "hbm_bandwidth_bytes_per_second", 0))
    memory_floor_us = (min_gm_bytes / hbm_bw) * 1e6 if hbm_bw > 0 else 0.0
    ideal_roofline_us = max(compute_floor_us, memory_floor_us)

    if ideal_roofline_us <= 0.0:
        dominant_bound = "none"
    elif abs(compute_floor_us - memory_floor_us) <= 1e-12 * max(1.0, ideal_roofline_us):
        dominant_bound = "balanced"
    elif memory_floor_us > compute_floor_us:
        dominant_bound = "memory"
    else:
        dominant_bound = "compute"

    resolved_copy = dict(resolved_hir_dims)
    return HirRooflineSummary(
        anchor_dims=dict(resolved_copy),
        compute_flops=compute_flops,
        min_gm_bytes=min_gm_bytes,
        arithmetic_intensity=arithmetic_intensity,
        compute_floor_us=compute_floor_us,
        memory_floor_us=memory_floor_us,
        ideal_roofline_us=ideal_roofline_us,
        dominant_bound=dominant_bound,
        requested_anchor_dims={str(k): int(v) for k, v in requested_anchor_dims.items()},
        resolved_hir_dims=resolved_copy,
        shape_exact_match=bool(shape_exact_match),
        fallback_reason=fallback_reason,
        cube_flops=cube_flops,
        vector_flops=vector_flops,
        special_ops=special_ops,
        cube_floor_us=cube_floor_us,
        vec_floor_us=vec_floor_us,
    )


def _format_z3_pipeline_summary(raw_pipelines: object) -> dict[str, object]:
    """Extract normalized ii, stages, phases, and versions from ascend.pipelines metadata."""
    if not isinstance(raw_pipelines, Mapping):
        return {}
    summary: dict[str, object] = {}
    for loop_key, loop_info in raw_pipelines.items():
        if not isinstance(loop_info, Mapping):
            continue
        summary[str(loop_key)] = {
            "ii": int(loop_info["ii"]) if loop_info.get("ii") is not None else None,
            "stages": {
                str(k): int(v) for k, v in dict(loop_info.get("stages") or {}).items()
            },
            "phases": {
                str(k): int(v) for k, v in dict(loop_info.get("phases") or {}).items()
            },
            "versions": {
                str(k): int(v) for k, v in dict(loop_info.get("versions") or {}).items()
            },
        }
    return summary


_LOWERING_ERRORS = (
    ScheduleRefusal,
    LoweringError,
    AnalysisError,
    PipelineError,
    VerifyError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
)


def _lower_scheduled_module(
    sched_info: _LoadedModuleInfo,
    *,
    anchor_dims: Mapping[str, int] | None,
    resolved_target: AscendTarget,
) -> tuple[
    ObjectivePhysicalLedger | None,
    dict[str, object],
    str | None,
    dict[str, int],
    set[str],
]:
    """Check, solve, verify, and inspect one scheduled HIR module without TIR script dumps."""
    sched_checked_mod, sched_checked_fn, sched_authored_fn, sched_declared = (
        _check_with_filtered_dims(sched_info.module, anchor_dims=anchor_dims)
    )
    sched_resolved_dims = _extract_resolved_hir_dims(
        sched_info.source,
        sched_authored_fn,
        sched_checked_fn,
        sched_info.build_param_names,
    )
    analyzed = analyze(sched_checked_mod, sched_checked_fn, analysis="memory")
    lowered = finalize(analyzed, authored=sched_authored_fn)
    lowered_entry = lowered.entry_function()
    verify_prim_function(lowered_entry, module_fns=lowered)
    ref_ledger = inspect_tir_envelope(lowered_entry, target=resolved_target).ledger
    z3_summary = _format_z3_pipeline_summary(lowered.metadata.get("ascend.pipelines", {}))
    return (
        ref_ledger,
        z3_summary,
        str(lowered_entry.name),
        sched_resolved_dims,
        sched_declared,
    )


def _derive_recommended_facets(
    ledger: ObjectivePhysicalLedger | None,
    z3_stage_summary: Mapping[str, object],
) -> tuple[str, ...]:
    """Derive 7-facet relaxation hints directly from HIR schedule telemetry."""
    facets: list[str] = []

    def _add(facet: str) -> None:
        if facet not in facets:
            facets.append(facet)

    if z3_stage_summary:
        _add("sync_skeleton")
        _add("loop_nesting")
    if ledger is not None:
        if ledger.gm_roundtrip_bytes > 0:
            _add("memory_space")
            _add("sync_skeleton")
        if ledger.unaligned_dma_padding_bytes > 0 or 0 < ledger.dominant_contig_bytes < 16384:
            _add("tile_shape")
            _add("layout_cast")
        if ledger.in_loop_pipe_all_barriers > 0:
            _add("sync_skeleton")
            _add("loop_nesting")
        if ledger.has_online_rescale_loop:
            _add("loop_nesting")
        if ledger.ub_bank_conflict_risk > 0.08 or ledger.l1_bank_conflict_risk > 0.08:
            _add("layout_cast")
    return tuple(facets)


def compile_hir_schedule_delta(
    authored_hir_source: str,
    *,
    scheduled_hir_source: str | None = None,
    anchor_dims: Mapping[str, int] | None = None,
    target: AscendTarget | None = None,
) -> tuple[BlueprintScheduleDelta, ObjectivePhysicalLedger | None, tuple[str, ...]]:
    """Extract optional HIR ledgers and Z3 stage summaries without computing Roofline."""
    unscheduled_ledger: ObjectivePhysicalLedger | None = None
    scheduled_ledger: ObjectivePhysicalLedger | None = None
    z3_stage_summary: dict[str, object] = {}
    diagnostics: list[str] = []

    def _load_resilient(
        source: str,
        tmpdir: Path,
        filename: str,
    ) -> tuple[_LoadedModuleInfo, Mapping[str, int] | None]:
        try:
            return (
                _load_module_from_source(source, tmpdir, filename, anchor_dims=anchor_dims),
                anchor_dims,
            )
        except (VerifyError, ValueError, AssertionError, TypeError) as exc:
            diagnostics.append(f"hir_default_dims_fallback({filename}): {type(exc).__name__}: {exc}")
            return (
                _load_module_from_source(source, tmpdir, filename, anchor_dims=None),
                None,
            )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        authored_info, authored_dims = _load_resilient(
            authored_hir_source, tmpdir, "authored_hir.py"
        )
        resolved_target = target or authored_info.module.resolve_target()
        if not isinstance(resolved_target, AscendTarget):
            resolved_target = AscendTarget("ascend.950pr")

        try:
            unsched_led, unsched_z3, _name, _dims, _decl = _lower_scheduled_module(
                authored_info, anchor_dims=authored_dims, resolved_target=resolved_target
            )
            unscheduled_ledger = unsched_led
            if scheduled_hir_source is None:
                z3_stage_summary = unsched_z3
        except _LOWERING_ERRORS as exc:
            if scheduled_hir_source is None:
                diagnostics.append(f"{type(exc).__name__}: {exc}")

        if scheduled_hir_source is not None and scheduled_hir_source.strip():
            sched_info, sched_dims = _load_resilient(
                scheduled_hir_source, tmpdir, "scheduled_hir.py"
            )
            try:
                sched_led, sched_z3, _name, _dims, _decl = _lower_scheduled_module(
                    sched_info, anchor_dims=sched_dims, resolved_target=resolved_target
                )
                scheduled_ledger = sched_led
                z3_stage_summary = sched_z3
            except _LOWERING_ERRORS as exc:
                diagnostics.append(f"scheduled_hir_lowering: {type(exc).__name__}: {exc}")

    primary_ledger = scheduled_ledger if scheduled_ledger is not None else unscheduled_ledger
    recommended_facets = _derive_recommended_facets(primary_ledger, z3_stage_summary)
    delta = BlueprintScheduleDelta(
        unscheduled_ledger=unscheduled_ledger,
        scheduled_ledger=scheduled_ledger,
        z3_stage_summary=z3_stage_summary,
        recommended_facets=recommended_facets,
    )
    return delta, primary_ledger, tuple(diagnostics)


def compile_hir_blueprint(
    authored_hir_source: str,
    *,
    scheduled_hir_source: str | None = None,
    anchor_dims: Mapping[str, int] | None = None,
    target: AscendTarget | None = None,
) -> HirBlueprintPacket:
    """Compile authored and optional scheduled HIR into a dialect-neutral blueprint."""
    requested_dims = {str(k): int(v) for k, v in (anchor_dims or {}).items()}
    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        authored_info = _load_module_from_source(
            authored_hir_source, tmpdir, "authored_hir.py", anchor_dims=anchor_dims
        )
        resolved_target = target or authored_info.module.resolve_target()
        if not isinstance(resolved_target, AscendTarget):
            resolved_target = AscendTarget("ascend.950pr")

        checked_mod, checked_fn, authored_fn, declared_dimvars = _check_with_filtered_dims(
            authored_info.module, anchor_dims=anchor_dims
        )
        entry_name = str(checked_fn.name)
        authored_resolved_dims = _extract_resolved_hir_dims(
            authored_hir_source,
            authored_fn,
            checked_fn,
            authored_info.build_param_names,
        )
        authored_exact = _evaluate_shape_match(
            requested_dims,
            authored_resolved_dims,
            fallback_reason=authored_info.fallback_reason,
            parameterized_applied=authored_info.parameterized_applied,
            declared_dimvars=declared_dimvars,
            build_param_names=authored_info.build_param_names,
        )

        sched_info = (
            _load_module_from_source(
                scheduled_hir_source, tmpdir, "scheduled_hir.py", anchor_dims=anchor_dims
            )
            if scheduled_hir_source is not None
            else authored_info
        )

        ref_ledger: ObjectivePhysicalLedger | None = None
        z3_summary: dict[str, object] = {}
        diagnostics: list[str] = []
        sched_fallback_reason = sched_info.fallback_reason
        sched_param_applied = sched_info.parameterized_applied
        sched_resolved_dims: dict[str, int] = dict(authored_resolved_dims)
        sched_declared: set[str] = set(declared_dimvars)

        try:
            (
                ref_ledger,
                z3_summary,
                lowered_name,
                sched_resolved_dims,
                sched_declared,
            ) = _lower_scheduled_module(
                sched_info, anchor_dims=anchor_dims, resolved_target=resolved_target
            )
            if lowered_name is not None:
                entry_name = lowered_name
        except _LOWERING_ERRORS as exc:
            if scheduled_hir_source is not None:
                raise
            diagnostics.append(f"{type(exc).__name__}: {exc}")

    combined_fallback = authored_info.fallback_reason or sched_fallback_reason
    if authored_info.fallback_reason and sched_fallback_reason and (
        authored_info.fallback_reason != sched_fallback_reason
    ):
        combined_fallback = (
            f"authored={authored_info.fallback_reason}; scheduled={sched_fallback_reason}"
        )

    packet_resolved_dims = (
        sched_resolved_dims if scheduled_hir_source is not None else authored_resolved_dims
    )
    sched_exact = _evaluate_shape_match(
        requested_dims,
        packet_resolved_dims,
        fallback_reason=sched_fallback_reason,
        parameterized_applied=sched_param_applied,
        declared_dimvars=sched_declared,
        build_param_names=sched_info.build_param_names,
    )
    packet_exact = bool(authored_exact and sched_exact and combined_fallback is None)

    roofline = _compute_roofline_summary(
        checked_mod,
        checked_fn,
        requested_anchor_dims=requested_dims,
        resolved_hir_dims=authored_resolved_dims,
        shape_exact_match=authored_exact,
        fallback_reason=combined_fallback,
        target=resolved_target,
    )
    has_scheduled = scheduled_hir_source is not None
    schedule_delta = BlueprintScheduleDelta(
        unscheduled_ledger=None if has_scheduled else ref_ledger,
        scheduled_ledger=ref_ledger if has_scheduled else None,
        z3_stage_summary=z3_summary,
        recommended_facets=_derive_recommended_facets(ref_ledger, z3_summary),
    )
    provenance = BlueprintProvenance(
        semantic_source="pytorch_golden_plus_hir",
        traffic_basis="declared_io_exact",
        coordinate_scope="representative_anchor",
        has_authored_hir=True,
        has_scheduled_hir=has_scheduled,
    )

    return HirBlueprintPacket(
        entry_name=entry_name,
        roofline=roofline,
        reference_ledger=ref_ledger,
        lowering_diagnostics=tuple(diagnostics),
        requested_anchor_dims=requested_dims,
        resolved_hir_dims=dict(packet_resolved_dims),
        shape_exact_match=packet_exact,
        fallback_reason=combined_fallback,
        provenance=provenance,
        schedule_delta=schedule_delta,
    )


__all__ = [
    "BlueprintProvenance",
    "BlueprintScheduleDelta",
    "HirBlueprintPacket",
    "HirRooflineSummary",
    "compile_hir_blueprint",
    "compile_hir_schedule_delta",
]
