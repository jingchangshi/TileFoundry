"""HIR-to-TIR navigation packet compilation for Ascend targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
import math
from pathlib import Path
import runpy
import tempfile

from tilefoundry.analysis import AnalysisError, ComputeCostMetadata, analyze, check_program
from tilefoundry.analysis.check import program_dim_vars
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import VerifyError, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.pipeline import (
    PipelineError,
    normalize_pipeline_references,
    rebind_cloned_pipeline_references,
)
from tilefoundry.ir.hir.schedule import ScheduleRefusal
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.types.utils import static_dim_value, tensor_types
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError
from tilefoundry.schedule.candidates import candidates as report_candidates
from tilefoundry.schedule.finalize import finalize
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tir_envelope
from tilefoundry_ascend.ascend.target import AscendTarget


@dataclass(frozen=True)
class HirRooflineSummary:
    """Analytical compute and HBM roofline summary derived from authored HIR."""

    anchor_dims: dict[str, int]
    compute_flops: int
    min_gm_bytes: int
    arithmetic_intensity: float
    compute_floor_us: float
    memory_floor_us: float
    ideal_roofline_us: float
    dominant_bound: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "anchor_dims": {str(k): int(v) for k, v in self.anchor_dims.items()},
            "compute_flops": int(self.compute_flops),
            "min_gm_bytes": int(self.min_gm_bytes),
            "arithmetic_intensity": float(self.arithmetic_intensity),
            "compute_floor_us": float(self.compute_floor_us),
            "memory_floor_us": float(self.memory_floor_us),
            "ideal_roofline_us": float(self.ideal_roofline_us),
            "dominant_bound": str(self.dominant_bound),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HirRooflineSummary:
        """Reconstruct a HirRooflineSummary from a serialized dictionary."""
        raw_dims = payload.get("anchor_dims") or {}
        return cls(
            anchor_dims={str(k): int(v) for k, v in dict(raw_dims).items()},
            compute_flops=int(payload["compute_flops"]),
            min_gm_bytes=int(payload["min_gm_bytes"]),
            arithmetic_intensity=float(payload["arithmetic_intensity"]),
            compute_floor_us=float(payload["compute_floor_us"]),
            memory_floor_us=float(payload["memory_floor_us"]),
            ideal_roofline_us=float(payload["ideal_roofline_us"]),
            dominant_bound=str(payload["dominant_bound"]),
        )

    def format_for_prompt(self) -> str:
        """Format the roofline summary as compact prompt text."""
        dims = ", ".join(f"{k}={v}" for k, v in sorted(self.anchor_dims.items())) or "default"
        return (
            f"Roofline[{dims}]: flops={self.compute_flops}, "
            f"min_gm_bytes={self.min_gm_bytes}, AI={self.arithmetic_intensity:.4f}, "
            f"compute_floor_us={self.compute_floor_us:.4f}, "
            f"memory_floor_us={self.memory_floor_us:.4f}, "
            f"ideal_roofline_us={self.ideal_roofline_us:.4f}, "
            f"dominant_bound={self.dominant_bound}"
        )


@dataclass(frozen=True)
class HirCandidateOpEntry:
    """Legal target instruction candidates for one authored HIR call site."""

    hir_op_name: str
    operand_shapes: tuple[str, ...]
    legal_ascend_ops: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "hir_op_name": self.hir_op_name,
            "operand_shapes": list(self.operand_shapes),
            "legal_ascend_ops": list(self.legal_ascend_ops),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HirCandidateOpEntry:
        """Reconstruct a HirCandidateOpEntry from a serialized dictionary."""
        return cls(
            hir_op_name=str(payload["hir_op_name"]),
            operand_shapes=tuple(str(item) for item in (payload.get("operand_shapes") or ())),
            legal_ascend_ops=tuple(str(item) for item in (payload.get("legal_ascend_ops") or ())),
        )

    def format_for_prompt(self) -> str:
        """Format this candidate op entry as one concise prompt line."""
        shapes = ", ".join(self.operand_shapes)
        ops = ", ".join(self.legal_ascend_ops) or "none"
        return f"- {self.hir_op_name} ({shapes}) -> [{ops}]"


@dataclass(frozen=True)
class HirTirNavigationPacket:
    """Combined HIR roofline, legal ops, Z3 pipeline schedule, and TIR ledger."""

    entry_name: str
    roofline: HirRooflineSummary
    candidate_ops: tuple[HirCandidateOpEntry, ...]
    tir_script: str | None
    z3_pipeline_summary: dict[str, object]
    tir_ledger: ObjectivePhysicalLedger | None
    lowering_diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "entry_name": self.entry_name,
            "roofline": self.roofline.to_dict(),
            "candidate_ops": [entry.to_dict() for entry in self.candidate_ops],
            "tir_script": self.tir_script,
            "z3_pipeline_summary": dict(self.z3_pipeline_summary),
            "tir_ledger": self.tir_ledger.to_dict() if self.tir_ledger is not None else None,
            "lowering_diagnostics": list(self.lowering_diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HirTirNavigationPacket:
        """Reconstruct a HirTirNavigationPacket from a serialized dictionary."""
        raw_cands = payload.get("candidate_ops") or ()
        raw_ledger = payload.get("tir_ledger")
        raw_z3 = payload.get("z3_pipeline_summary") or {}
        raw_diags = payload.get("lowering_diagnostics") or ()
        return cls(
            entry_name=str(payload["entry_name"]),
            roofline=HirRooflineSummary.from_dict(payload["roofline"]),
            candidate_ops=tuple(
                item
                if isinstance(item, HirCandidateOpEntry)
                else HirCandidateOpEntry.from_dict(item)
                for item in raw_cands
            ),
            tir_script=str(payload["tir_script"]) if payload.get("tir_script") is not None else None,
            z3_pipeline_summary=dict(raw_z3),
            tir_ledger=(
                raw_ledger
                if isinstance(raw_ledger, ObjectivePhysicalLedger) or raw_ledger is None
                else ObjectivePhysicalLedger.from_dict(raw_ledger)
            ),
            lowering_diagnostics=tuple(str(item) for item in raw_diags),
        )

    def format_for_prompt(self) -> str:
        """Render the navigation packet as structured prompt guidance."""
        lines = [
            f"Entry: {self.entry_name}",
            self.roofline.format_for_prompt(),
            f"Candidate Ops ({len(self.candidate_ops)}):",
            *(entry.format_for_prompt() for entry in self.candidate_ops),
            f"Z3 Pipelines: {self.z3_pipeline_summary}",
        ]
        if self.tir_ledger is not None:
            lines.append(
                f"TIR Ledger: mode={self.tir_ledger.analysis_mode}, "
                f"legal={self.tir_ledger.is_legal}, "
                f"dominant_pipe={self.tir_ledger.dominant_pipe}, "
                f"overflow={self.tir_ledger.max_overflow_bytes}"
            )
        if self.lowering_diagnostics:
            lines.append("Lowering Diagnostics:")
            lines.extend(f"- {diag}" for diag in self.lowering_diagnostics)
        if self.tir_script is not None:
            lines.append("TIR Script:")
            lines.append(self.tir_script)
        return "\n".join(lines)


def _load_module_from_source(
    source: str,
    tmpdir: Path,
    filename: str,
    *,
    anchor_dims: Mapping[str, int] | None,
) -> Module:
    """Write HIR source to *tmpdir* and load via build() factory or load_authored_ir."""
    path = tmpdir / filename
    path.write_text(source, encoding="utf-8")
    if anchor_dims:
        try:
            ns = runpy.run_path(str(path))
            build_fn = ns.get("build")
            if callable(build_fn):
                sig = inspect.signature(build_fn)
                matched = {
                    str(k): int(v)
                    for k, v in anchor_dims.items()
                    if str(k) in sig.parameters
                    and sig.parameters[str(k)].kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                if matched:
                    built = build_fn(**matched)
                    if isinstance(built, Module):
                        return built
        except (ValueError, AssertionError, TypeError):
            pass
    return load_authored_ir(str(path))


def _check_with_filtered_dims(
    mod: Module,
    *,
    anchor_dims: Mapping[str, int] | None,
):
    """Specialize only declared DimVars and return (checked_mod, checked_fn, authored_fn)."""
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
    return checked_mod, checked_fn, authored_fn


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


def _compute_roofline_summary(
    checked_mod: Module,
    checked_fn,
    *,
    anchor_dims: Mapping[str, int] | None,
    target: AscendTarget,
) -> HirRooflineSummary:
    """Compute HirRooflineSummary using compute-cost analysis and device hardware ceilings."""
    analyze(checked_mod, checked_fn, analysis=("compute-cost",))
    cc = get_metadata(checked_fn, ComputeCostMetadata)
    flops_by_dtype = (
        {name: int(spread.total) for name, spread in cc.flops.kinds}
        if cc is not None
        else {}
    )
    compute_flops = sum(flops_by_dtype.values())

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
    merged_rates: dict[DType, int] = dict(aic_rates)
    for dt, rate in aiv_rates.items():
        merged_rates.setdefault(dt, rate)
    fallback_rate = aiv_rates.get(DType.f32) or max(merged_rates.values(), default=0)

    compute_floor_s = 0.0
    for dtype_name, flop_count in flops_by_dtype.items():
        if flop_count <= 0:
            continue
        dt = getattr(DType, dtype_name, None)
        rate = merged_rates.get(dt) if dt is not None else None
        if rate is None or rate <= 0:
            rate = fallback_rate
        if rate and rate > 0:
            compute_floor_s += flop_count / float(rate)
    compute_floor_us = compute_floor_s * 1e6

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

    return HirRooflineSummary(
        anchor_dims={str(k): int(v) for k, v in (anchor_dims or {}).items()},
        compute_flops=compute_flops,
        min_gm_bytes=min_gm_bytes,
        arithmetic_intensity=arithmetic_intensity,
        compute_floor_us=compute_floor_us,
        memory_floor_us=memory_floor_us,
        ideal_roofline_us=ideal_roofline_us,
        dominant_bound=dominant_bound,
    )


def _extract_candidate_ops(
    checked_mod: Module,
    checked_fn,
) -> tuple[HirCandidateOpEntry, ...]:
    """Extract per-call candidate instruction entries for unscheduled HIR calls."""
    try:
        cand_report = report_candidates(checked_mod, checked_fn)
    except (ValueError, ScheduleRefusal, AnalysisError):
        return ()

    entries: list[HirCandidateOpEntry] = []
    for row in cand_report.get("lines", ()):
        cands = [str(item["id"]) for item in row.get("candidates", ())]
        if not cands:
            cands = [str(item["id"]) for item in row.get("refused", ())]
        legal_ops = tuple(dict.fromkeys(cands))
        entries.append(
            HirCandidateOpEntry(
                hir_op_name=str(row["op"]),
                operand_shapes=tuple(str(opnd) for opnd in row.get("operands", ())),
                legal_ascend_ops=legal_ops,
            )
        )
    return tuple(entries)


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


def compile_hir_navigation_packet(
    authored_hir_source: str,
    *,
    scheduled_hir_source: str | None = None,
    anchor_dims: Mapping[str, int] | None = None,
    target: AscendTarget | None = None,
) -> HirTirNavigationPacket:
    """Compile authored and optional scheduled HIR into a navigation packet for agents."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        authored_mod = _load_module_from_source(
            authored_hir_source, tmpdir, "authored_hir.py", anchor_dims=anchor_dims
        )
        resolved_target = target or authored_mod.resolve_target()
        if not isinstance(resolved_target, AscendTarget):
            resolved_target = AscendTarget("ascend.950pr")

        checked_mod, checked_fn, _authored_fn = _check_with_filtered_dims(
            authored_mod, anchor_dims=anchor_dims
        )
        entry_name = str(checked_fn.name)
        roofline = _compute_roofline_summary(
            checked_mod, checked_fn, anchor_dims=anchor_dims, target=resolved_target
        )
        candidate_ops = _extract_candidate_ops(checked_mod, checked_fn)

        sched_mod = (
            _load_module_from_source(
                scheduled_hir_source, tmpdir, "scheduled_hir.py", anchor_dims=anchor_dims
            )
            if scheduled_hir_source is not None
            else authored_mod
        )

    tir_script: str | None = None
    tir_ledger: ObjectivePhysicalLedger | None = None
    z3_summary: dict[str, object] = {}
    diagnostics: list[str] = []

    try:
        sched_checked_mod, sched_checked_fn, sched_authored_fn = _check_with_filtered_dims(
            sched_mod, anchor_dims=anchor_dims
        )
        analyzed = analyze(sched_checked_mod, sched_checked_fn, analysis="memory")
        lowered = finalize(analyzed, authored=sched_authored_fn)
        lowered_entry = lowered.entry_function()
        verify_prim_function(lowered_entry, module_fns=lowered)
        tir_script = as_script(lowered)
        tir_ledger = inspect_tir_envelope(lowered_entry, target=resolved_target).ledger
        z3_summary = _format_z3_pipeline_summary(lowered.metadata.get("ascend.pipelines", {}))
        entry_name = str(lowered_entry.name)
    except (
        ScheduleRefusal,
        LoweringError,
        AnalysisError,
        PipelineError,
        VerifyError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
    ) as exc:
        diagnostics.append(f"{type(exc).__name__}: {exc}")

    return HirTirNavigationPacket(
        entry_name=entry_name,
        roofline=roofline,
        candidate_ops=candidate_ops,
        tir_script=tir_script,
        z3_pipeline_summary=z3_summary,
        tir_ledger=tir_ledger,
        lowering_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "HirCandidateOpEntry",
    "HirRooflineSummary",
    "HirTirNavigationPacket",
    "compile_hir_navigation_packet",
]
