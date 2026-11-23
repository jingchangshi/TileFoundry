"""Pure 3-axis TopologyAxes CapabilityGate router and Track escalation evaluator."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from tilefoundry.ir.core.expr import Call, Constant, Expr
from tilefoundry.ir.core.expr import Tuple as IrTuple
from tilefoundry.ir.hir.execution_region import ExecutionRegion
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    ExecutionScope,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    Stmt,
    While,
)
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.mesh import flattened_levels
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry_ascend.agent.router.isa_gap_scanner import PipeGapReport
from tilefoundry_ascend.ascend.ir.execution import SIMDVF_KIND, SIMTVF_KIND
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target import AscendTarget

VALID_QUANT_GRANULARITIES = (
    "none",
    "per_tensor",
    "per_token_or_block",
    "mixed_fp8_fp4",
)

VALID_RECURRENCES = (
    "none",
    "associative_online_rescale",
    "non_associative_scan",
)

VALID_TRACKS = ("developer", "simt_vf", "manual_ssbuf")

VALID_BACKEND_TIERS = ("tilelang_dsl", "ascendc_template")

VALID_ESCALATION_TIERS = ("tilelang_dsl", "inline_ascendc_vf", "ascendc_template")

_ROUTER_TIER_ALIASES: dict[str, str] = {
    "tilelang_dsl": "tilelang_dsl",
    "cce_cpp20": "ascendc_template",
    "cce_trackb": "ascendc_template",
    "hybrid_trackb": "ascendc_template",
    "tl_templates_cpp20": "ascendc_template",
    "ascendc_trackb": "ascendc_template",
    "ascendc_template": "ascendc_template",
    "inline_cce_vf": "inline_ascendc_vf",
    "inline_ascendc_vf": "inline_ascendc_vf",
}


def normalize_router_tier(value: str) -> str:
    """Normalize a canonical dialect id or legacy alias to the router's internal tier id."""
    normalized = _ROUTER_TIER_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(
            f"unsupported router tier {value!r}; expected one of {tuple(_ROUTER_TIER_ALIASES)}"
        )
    return normalized

_SUBBYTE_DTYPE_TOKENS: tuple[str, ...] = (
    "float4",
    "fp4",
    "mxfp4",
    "f4e2m1",
    "f4e1m2",
    "e2m1",
    "e1m2",
    "int4",
    "uint4",
    "q4_0",
)

def _primitives_for_axes(axes: TopologyAxes) -> tuple[str, ...]:
    if axes.iteration_regularity == "dynamic_opcode_stream":
        return ("T.SimdVF", "T.SimtVF", "T.Persistent")
    if axes.memory_ring == "persistent_onchip_state":
        return ("T.Persistent", "T.Stage", "T.dual_copy", "T.SimdVF")
    if axes.memory_ring == "bidirectional_cv_ring":
        return ("T.Scope", "T.alloc_ss", "T.dual_copy", "T.SimdVF")
    if axes.memory_ring == "l0c_ub_epilogue":
        return ("T.Scope", "T.alloc_ss", "T.dual_copy", "T.Persistent")
    if axes.memory_ring == "gm_l1_l0_fixpipe":
        return ("T.Pipelined", "T.gemm", "T.dual_copy", "T.Persistent")
    if axes.memory_ring == "ub_random_atomic" or axes.iteration_regularity == "irregular_simt":
        return ("T.SimtVF", "T.get_thread_binding", "T.alloc_ss", "T.Persistent")
    return ("T.Pipelined", "T.Stage", "T.SimdVF", "T.dual_copy")


_BLOCKING_GAP_BY_RING: dict[str, tuple[str, ...]] = {
    "gm_ub_stream": ("dma_dual_copy_and_nd2nz", "simd_vld_vst"),
    "gm_l1_l0_fixpipe": ("dma_dual_copy_and_nd2nz",),
    "bidirectional_cv_ring": ("dma_dual_copy_and_nd2nz", "crosscore_flag_set_wait"),
    "persistent_onchip_state": ("dma_dual_copy_and_nd2nz",),
    "l0c_ub_epilogue": ("fixpipe_l0c_to_ub_split_m",),
    "ub_random_atomic": (
        "simt_gm_atomic_cas",
        "hardware_mutex_28",
        "simd_vgather_vscatter",
    ),
}

_BLOCKING_GAP_BY_REGULARITY: dict[str, tuple[str, ...]] = {
    "irregular_simt": ("simt_gm_atomic_cas", "hardware_mutex_28", "simd_vgather_vscatter"),
    "ragged_group": ("fixpipe_l0c_to_ub_split_m",),
    "dynamic_opcode_stream": ("scalar_set_ctrl_spr", "simt_gm_atomic_cas"),
}

_RING_SCORE_DEDUCTION: dict[str, float] = {
    "bidirectional_cv_ring": 0.15,
    "persistent_onchip_state": 0.20,
    "l0c_ub_epilogue": 0.25,
    "ub_random_atomic": 0.35,
}

_REGULARITY_SCORE_DEDUCTION: dict[str, float] = {
    "dynamic_opcode_stream": 0.35,
    "ragged_group": 0.25,
    "irregular_simt": 0.25,
}


def _coerce_axes(raw: TopologyAxes | Mapping[str, object]) -> TopologyAxes:
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping):
        return TopologyAxes.from_dict(raw)
    raise TypeError(f"Expected TopologyAxes or Mapping, got {type(raw).__name__}")


@dataclass(frozen=True)
class RouteDecision:
    """Selected DSL lowering track, 3-axis TopologyAxes, and backend tier."""

    track: str
    topology_axes: TopologyAxes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    rationale: tuple[str, ...] = ()
    recommended_primitives: tuple[str, ...] = ()
    recommended_backend_tier: str = "tilelang_dsl"
    case_ratios: CasePhysicsRatios | None = None
    pipe_gaps: PipeGapReport | None = None
    suitability_score: float = 1.0
    features: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.track not in VALID_TRACKS:
            raise ValueError(f"unsupported route track {self.track!r}")
        if isinstance(self.topology_axes, Mapping):
            object.__setattr__(
                self, "topology_axes", TopologyAxes.from_dict(self.topology_axes)
            )
        elif not isinstance(self.topology_axes, TopologyAxes):
            raise TypeError(
                "RouteDecision.topology_axes must be TopologyAxes, "
                f"got {type(self.topology_axes).__name__}"
            )
        normalized_tier = normalize_router_tier(self.recommended_backend_tier)
        if normalized_tier not in VALID_BACKEND_TIERS:
            raise ValueError(
                f"unsupported backend tier {self.recommended_backend_tier!r}"
            )
        object.__setattr__(self, "recommended_backend_tier", normalized_tier)
        score = float(self.suitability_score)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(
                f"RouteDecision.suitability_score must be in [0.0, 1.0], got {score!r}"
            )
        object.__setattr__(self, "suitability_score", score)
        if self.features is not None:
            if not isinstance(self.features, Mapping):
                raise TypeError(
                    "RouteDecision.features must be Mapping or None, "
                    f"got {type(self.features).__name__}"
                )
            object.__setattr__(
                self,
                "features",
                {str(k): v for k, v in self.features.items()},
            )


class CapabilityGate:
    """Pure 3-axis TopologyAxes router and ISA suitability scorer."""

    @staticmethod
    def suitability_score(
        topology_axes: TopologyAxes,
        pipe_gaps: PipeGapReport | None = None,
        case_ratios: CasePhysicsRatios | None = None,
    ) -> float:
        """Compute deterministic [0.0, 1.0] DSL suitability score from 3 axes, gaps, and ratios."""
        base_axes = _coerce_axes(topology_axes)
        axes = case_ratios.effective_axes(base_axes) if case_ratios is not None else base_axes

        blocking_prims: set[str] = set(_BLOCKING_GAP_BY_RING.get(axes.memory_ring, ()))
        blocking_prims.update(_BLOCKING_GAP_BY_REGULARITY.get(axes.iteration_regularity, ()))
        if axes.uses_cube and axes.memory_ring == "persistent_onchip_state":
            blocking_prims.add("crosscore_flag_set_wait")

        if pipe_gaps is not None:
            for prim in blocking_prims:
                if pipe_gaps.has_gap(prim):
                    return 0.0

        score = 1.0
        score -= _RING_SCORE_DEDUCTION.get(axes.memory_ring, 0.0)
        score -= _REGULARITY_SCORE_DEDUCTION.get(axes.iteration_regularity, 0.0)
        if case_ratios is not None and case_ratios.contig_atom_ratio < 1.0:
            score -= 0.20

        if pipe_gaps is not None and pipe_gaps.missing_primitives:
            non_blocking = sum(
                1 for prim in pipe_gaps.missing_primitives if prim not in blocking_prims
            )
            if non_blocking > 0:
                score -= min(0.25, 0.015 * float(non_blocking))

        return round(max(0.0, min(1.0, score)), 6)

    @staticmethod
    def route(
        topology_axes: TopologyAxes,
        *,
        pipe_gaps: PipeGapReport | None = None,
        case_ratios: CasePhysicsRatios | None = None,
        router_enabled: bool = True,
        semantic_features: Mapping[str, object] | None = None,
    ) -> RouteDecision:
        """Route a 3-axis TopologyAxes coordinate to execution track and backend tier."""
        base_axes = _coerce_axes(topology_axes)
        if not router_enabled:
            return RouteDecision(
                track="developer",
                topology_axes=base_axes,
                rationale=("track router disabled (ablation); neutral developer route",),
                recommended_primitives=_primitives_for_axes(base_axes),
                recommended_backend_tier="tilelang_dsl",
                case_ratios=case_ratios,
                pipe_gaps=pipe_gaps,
                suitability_score=1.0,
                features=None,
            )

        axes = case_ratios.effective_axes(base_axes) if case_ratios is not None else base_axes
        score = CapabilityGate.suitability_score(
            axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios
        )

        feat_dict: dict[str, object] = (
            {str(k): v for k, v in semantic_features.items()}
            if semantic_features is not None
            else {}
        )
        feat_dict["compute_engine"] = axes.compute_engine
        feat_dict["memory_ring"] = axes.memory_ring
        feat_dict["iteration_regularity"] = axes.iteration_regularity
        feat_dict["uses_cube"] = axes.uses_cube
        feat_dict["suitability_score"] = score
        if case_ratios is not None:
            feat_dict["inner_contiguous_bytes"] = case_ratios.inner_contiguous_bytes
            feat_dict["persistent_wave_occupancy"] = case_ratios.wave_occupancy_ratio
            feat_dict["wave_occupancy_ratio"] = case_ratios.wave_occupancy_ratio
            feat_dict["contig_atom_ratio"] = case_ratios.contig_atom_ratio
            feat_dict["oi_ridge_ratio"] = case_ratios.oi_ridge_ratio
            feat_dict["has_empty_group"] = case_ratios.has_empty_group

        has_handoff = bool(feat_dict.get("dynamic_cross_core_handoff", False))
        has_indirect = bool(feat_dict.get("indirect_sparse_gather", False))
        simt_dom = bool(feat_dict.get("simt_dominant", False))
        is_non_assoc = str(feat_dict.get("cross_iter_recurrence", "none")) == "non_associative_scan"
        inner_bytes = int(feat_dict.get("inner_contiguous_bytes", 0) or 0)
        sub32_width = (0 < inner_bytes < 32) or (
            case_ratios is not None and case_ratios.contig_atom_ratio < 1.0
        )

        reasons: list[str] = [f"3-axis route -> {axes.key()} (score={score:.3f})"]
        if axes.memory_ring in ("bidirectional_cv_ring", "l0c_ub_epilogue") or has_handoff or (
            axes.uses_cube
            and (has_indirect or axes.iteration_regularity in ("irregular_simt", "ragged_group"))
        ):
            track = "manual_ssbuf"
            if has_indirect:
                reasons.append("indirect sparse/page gather detected")
            if has_handoff:
                reasons.append("dynamic cross-core metadata/scale handoff detected")
        elif (
            axes.memory_ring in ("ub_random_atomic", "persistent_onchip_state")
            or axes.iteration_regularity in ("irregular_simt", "dynamic_opcode_stream")
            or is_non_assoc
            or simt_dom
            or (sub32_width and not axes.uses_cube)
        ):
            track = "simt_vf"
            if simt_dom:
                reasons.append("fine-grained thread-level vector reduction flagged")
            if is_non_assoc:
                reasons.append("non-associative cross-iteration recurrence")
            if sub32_width and not axes.uses_cube:
                reasons.append("sub-32B innermost contiguous vector width")
        else:
            track = "developer"

        quant = str(feat_dict.get("quant_granularity", "none"))
        if (
            score == 0.0
            or axes.memory_ring in ("ub_random_atomic", "l0c_ub_epilogue", "persistent_onchip_state")
            or axes.iteration_regularity in ("irregular_simt", "ragged_group", "dynamic_opcode_stream")
            or (axes.memory_ring == "gm_l1_l0_fixpipe" and quant != "none")
            or bool(feat_dict.get("requires_subbyte_dtype", False))
            or bool(feat_dict.get("requires_inplace_gm_cas", False))
        ):
            backend_tier = "ascendc_template"
            if score == 0.0:
                reasons.append("blocking TileLang ISA gap detected; escalating to ascendc_template")
        else:
            backend_tier = "tilelang_dsl"

        if case_ratios is not None:
            reasons.append(
                f"case ratios contig_bytes={case_ratios.inner_contiguous_bytes}B "
                f"wave_occ={case_ratios.wave_occupancy_ratio:.3f} -> {axes.key()}"
            )

        return RouteDecision(
            track=track,
            topology_axes=axes,
            rationale=tuple(reasons),
            recommended_primitives=_primitives_for_axes(axes),
            recommended_backend_tier=backend_tier,
            case_ratios=case_ratios,
            pipe_gaps=pipe_gaps,
            suitability_score=score,
            features=feat_dict,
        )


@dataclass(frozen=True)
class EscalationDecision:
    """Closed-loop t>0 track escalation decision with MECE 7-facet relaxation mask."""

    should_escalate: bool
    from_tier: str
    to_tier: str
    trigger_reasons: tuple[str, ...] = ()
    freedom_mask: ConstraintRelaxationMask | None = None
    suggested_primitives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_from = normalize_router_tier(self.from_tier)
        normalized_to = normalize_router_tier(self.to_tier)
        if normalized_from not in VALID_ESCALATION_TIERS:
            raise ValueError(f"unsupported from_tier {self.from_tier!r}")
        if normalized_to not in VALID_ESCALATION_TIERS:
            raise ValueError(f"unsupported to_tier {self.to_tier!r}")
        object.__setattr__(self, "from_tier", normalized_from)
        object.__setattr__(self, "to_tier", normalized_to)
        if isinstance(self.freedom_mask, Mapping):
            object.__setattr__(
                self, "freedom_mask", ConstraintRelaxationMask.from_dict(self.freedom_mask)
            )


def _ratio_from_ledger(ledger: ObjectivePhysicalLedger | None) -> float:
    if ledger is None or not ledger.pipe_cycles:
        return 0.0
    aic_cycles = sum(
        cycles
        for key, cycles in ledger.pipe_cycles.items()
        if key.startswith("AIC.") and key != "AIC.PIPE_S"
    )
    aiv_cycles = sum(
        cycles
        for key, cycles in ledger.pipe_cycles.items()
        if key.startswith(("AIV0.", "AIV1.")) and not key.endswith("PIPE_S")
    )
    if aic_cycles == 0:
        return 0.0
    if aiv_cycles == 0:
        return float(aic_cycles)
    return float(aic_cycles) / float(aiv_cycles)


def _classify_dtype_flags(dtype_str: str) -> tuple[bool, bool, bool]:
    """Return (is_fp8, is_fp4_or_4bit, is_i8) for a dtype string."""
    s = dtype_str.lower()
    is_fp4 = any(tok in s for tok in ("float4", "fp4", "f4e", "e2m1", "e1m2", "int4", "uint4", "q4_0")) or s in ("i4", "u4")
    is_fp8 = (not is_fp4) and any(tok in s for tok in ("float8", "fp8", "f8e", "e4m3", "e5m2", "e8m0", "hif8", "hifloat8"))
    is_i8 = (not is_fp4) and (not is_fp8) and (
        any(tok in s for tok in ("int8", "uint8")) or s in ("i8", "u8")
    )
    return is_fp8, is_fp4, is_i8


def _resolve_quant_granularity(
    *,
    has_fp8: bool,
    has_fp4: bool,
    has_i8: bool,
    has_quant_op: bool,
    per_tensor: bool = False,
) -> str:
    if has_fp4 and has_fp8:
        return "mixed_fp8_fp4"
    if per_tensor and not (has_fp8 or has_fp4):
        return "per_tensor"
    if has_fp8 or has_fp4 or has_i8 or has_quant_op:
        return "per_token_or_block"
    return "none"


def _scan_tilelang_features(func) -> dict[str, object]:
    """Scan a PrimFunc for static bounds, quant, gather, recurrence, and sync."""
    from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    static_bounds = [True]
    indirect_gather = [False]
    dynamic_handoff = [False]
    has_fp8 = [False]
    has_fp4 = [False]
    has_i8 = [False]
    has_ssbuf = [False]
    has_cube = [False]
    has_vec = [False]
    has_online_exp = [False]
    has_non_assoc_scan = [False]
    has_simt_vf = [False]
    min_inner_bytes: list[int] = []
    grid_tiles: list[int] = []
    indirect_vars: list[object] = []

    def _is_indirect_var(v) -> bool:
        return any(hasattr(v, "same_as") and v.same_as(iv) for iv in indirect_vars)

    def _check_dtype(dt_obj) -> None:
        is_fp8, is_fp4, is_i8 = _classify_dtype_flags(str(dt_obj))
        if is_fp4:
            has_fp4[0] = True
        elif is_fp8:
            has_fp8[0] = True
        elif is_i8:
            has_i8[0] = True

    def _has_load(expr) -> bool:
        hit = [False]

        def _f(node) -> None:
            if isinstance(node, tirx.BufferLoad):
                hit[0] = True

        tirx.stmt_functor.post_order_visit(expr, _f)
        return hit[0]

    def _has_indirect_expr(expr) -> bool:
        hit = [False]

        def _f(node) -> None:
            if isinstance(node, tirx.BufferLoad):
                hit[0] = True
            elif isinstance(node, tirx.Var) and _is_indirect_var(node):
                hit[0] = True

        tirx.stmt_functor.post_order_visit(expr, _f)
        return hit[0]

    def _analyze_loop_recurrence(loop_node, *, is_while: bool) -> None:
        """Structurally detect online rescale (exp+max) or loop-carried buffer RAW scan."""
        loaded_bufs: set[object] = set()
        stored_bufs: set[object] = set()
        gemm_dst_bufs: set[object] = set()
        loop_has_exp = [False]
        loop_has_max_or_online = [False]
        loop_has_non_add_raw_op = [False]

        def _scan_body(node) -> None:
            if isinstance(node, tirx.BufferLoad):
                loaded_bufs.add(node.buffer)
            elif isinstance(node, tirx.BufferStore):
                stored_bufs.add(node.buffer)
                rhs_ops = [False]

                def _rhs_check(rnode) -> None:
                    if isinstance(rnode, (tirx.Mul, tirx.Sub, tirx.Div, tirx.FloorDiv, tirx.Mod, tirx.Select, tirx.Call)):
                        rhs_ops[0] = True

                tirx.stmt_functor.post_order_visit(node.value, _rhs_check)
                if rhs_ops[0]:
                    loop_has_non_add_raw_op[0] = True
            elif isinstance(node, tirx.Max):
                loop_has_max_or_online[0] = True
            elif isinstance(node, tirx.Call):
                op_name = str(getattr(node.op, "name", node.op)).lower()
                if "exp" in op_name:
                    loop_has_exp[0] = True
                if "max" in op_name or "expdif" in op_name or "online" in op_name:
                    loop_has_max_or_online[0] = True
                if ("gemm" in op_name or "mmad" in op_name) and len(node.args) >= 3:
                    c_arg = node.args[2]
                    if isinstance(c_arg, tirx.BufferLoad):
                        gemm_dst_bufs.add(c_arg.buffer)
                    elif isinstance(c_arg, tirx.Call) and c_arg.args and isinstance(c_arg.args[0], tirx.BufferLoad):
                        gemm_dst_bufs.add(c_arg.args[0].buffer)

        tirx.stmt_functor.post_order_visit(loop_node.body, _scan_body)
        raw_bufs = (loaded_bufs & stored_bufs) - gemm_dst_bufs
        if loop_has_exp[0] and (loop_has_max_or_online[0] or bool(raw_bufs)):
            has_online_exp[0] = True
        elif is_while and (bool(stored_bufs) or bool(raw_bufs)):
            has_non_assoc_scan[0] = True
        elif raw_bufs and (loop_has_non_add_raw_op[0] or len(raw_bufs) >= 1):
            ext = getattr(loop_node, "extent", None)
            if not (hasattr(ext, "value") and isinstance(ext.value, int) and ext.value <= 1):
                has_non_assoc_scan[0] = True

    for param in func.params:
        buf = func.buffer_map.get(param)
        if buf is not None:
            _check_dtype(buf.dtype)
            if buf.shape:
                last = buf.shape[-1]
                if hasattr(last, "value") and isinstance(last.value, int):
                    eb = int((TLDataType(str(buf.dtype)).bits + 7) // 8)
                    min_inner_bytes.append(int(last.value) * eb)

    def _visit(node) -> None:
        if isinstance(node, tirx.While):
            static_bounds[0] = False
            _analyze_loop_recurrence(node, is_while=True)
        elif isinstance(node, tirx.Bind):
            if _has_indirect_expr(node.value):
                indirect_vars.append(node.var)
        elif isinstance(node, tirx.For):
            ext = node.extent
            tb = getattr(node, "thread_binding", None)
            tb_tag = str(getattr(tb, "thread_tag", "")) if tb is not None else ""
            if tb is not None and ("block" in tb_tag or tb_tag == ""):
                if hasattr(ext, "value") and isinstance(ext.value, int):
                    grid_tiles.append(int(ext.value))
            else:
                if not (hasattr(ext, "value") and isinstance(ext.value, int)):
                    static_bounds[0] = False
                if tb is None:
                    _analyze_loop_recurrence(node, is_while=False)
        elif isinstance(node, tirx.IfThenElse):
            if _has_load(node.condition):
                static_bounds[0] = False
        elif isinstance(node, tirx.AttrStmt):
            key = str(node.attr_key)
            if key == "thread_extent":
                if hasattr(node.value, "value") and isinstance(node.value.value, int):
                    iv = getattr(node.node, "var", None)
                    iv_name = getattr(iv, "name_hint", "")
                    tag = str(getattr(node.node, "thread_tag", ""))
                    if "blockIdx" in tag or "blockIdx" in iv_name:
                        grid_tiles.append(int(node.value.value))
                    elif "cthread" not in tag and "cthread" not in iv_name:
                        has_simt_vf[0] = True
            elif "simt" in key.lower():
                has_simt_vf[0] = True
        elif isinstance(node, tirx.AllocBuffer):
            _check_dtype(node.buffer.dtype)
            if node.buffer.scope() == "shared.ss":
                has_ssbuf[0] = True
                dynamic_handoff[0] = True
        elif isinstance(node, tirx.SBlock):
            for iv in getattr(node, "iter_vars", ()):
                tag = str(getattr(iv, "thread_tag", ""))
                dom_ext = getattr(getattr(iv, "dom", None), "extent", None)
                if "blockIdx" in tag and hasattr(dom_ext, "value") and isinstance(dom_ext.value, int):
                    grid_tiles.append(int(dom_ext.value))
            for buf in node.alloc_buffers:
                _check_dtype(buf.dtype)
                if buf.scope() == "shared.ss":
                    has_ssbuf[0] = True
                    dynamic_handoff[0] = True
        elif isinstance(node, tirx.BufferLoad):
            if any(_has_indirect_expr(idx) for idx in node.indices):
                indirect_gather[0] = True
                has_vec[0] = True
        elif isinstance(node, tirx.Call):
            op_name = str(getattr(node.op, "name", node.op)).lower()
            if "gather" in op_name or "scatter" in op_name:
                indirect_gather[0] = True
                has_vec[0] = True
            if "cross_core" in op_name or "ssbuf" in op_name:
                dynamic_handoff[0] = True
            if "gemm" in op_name or "mmad" in op_name:
                has_cube[0] = True
            if "simd" in op_name or "exp" in op_name:
                has_vec[0] = True
            if "expdif" in op_name or "online" in op_name:
                has_online_exp[0] = True
            if "region" in op_name and len(node.args) >= 3 and isinstance(node.args[0], tirx.BufferLoad):
                bl = node.args[0]
                if any(_has_indirect_expr(idx) for idx in bl.indices):
                    indirect_gather[0] = True
                last = node.args[-1]
                if hasattr(last, "value") and isinstance(last.value, int):
                    eb = max(1, (TLDataType(str(bl.buffer.dtype)).bits + 7) // 8)
                    min_inner_bytes.append(int(last.value) * eb)

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    quant = _resolve_quant_granularity(
        has_fp8=has_fp8[0],
        has_fp4=has_fp4[0],
        has_i8=has_i8[0],
        has_quant_op=False,
    )
    if has_online_exp[0]:
        recurrence = "associative_online_rescale"
    elif has_non_assoc_scan[0]:
        recurrence = "non_associative_scan"
    else:
        recurrence = "none"
    if has_ssbuf[0]:
        dynamic_handoff[0] = True
    return {
        "static_bounds": static_bounds[0],
        "indirect_gather": indirect_gather[0],
        "dynamic_handoff": dynamic_handoff[0],
        "quant": quant,
        "recurrence": recurrence,
        "has_simt_vf": has_simt_vf[0],
        "has_cube": has_cube[0],
        "has_vec": has_vec[0],
        "inner_bytes": min(min_inner_bytes) if min_inner_bytes else 0,
        "total_tiles": grid_tiles[0] if grid_tiles else 0,
    }


def _scan_tilefoundry_ir(fn_or_stmt) -> dict[str, object]:
    """Scan a TileFoundry HIR/TIR function or statement for all 8 structural features."""
    static_bounds = [True]
    has_simt = [False]
    has_cube = [False]
    has_vec = [False]
    has_fp8 = [False]
    has_fp4 = [False]
    has_i8 = [False]
    has_quant_op = [False]
    quant_per_tensor = [False]
    has_exp_in_loop = [False]
    has_max_in_loop = [False]
    has_softmax_in_loop = [False]
    inner_bytes: list[int] = []
    grid_tiles: list[int] = []
    recurrence = ["none"]
    indirect = [False]
    handoff = [False]

    def _record_type(ttype) -> None:
        if isinstance(ttype, TupleType):
            for field in ttype.fields:
                _record_type(field)
            return
        if isinstance(ttype, TensorType):
            is_fp8, is_fp4, is_i8 = _classify_dtype_flags(
                getattr(ttype.dtype, "name", str(ttype.dtype))
            )
            if is_fp4:
                has_fp4[0] = True
            elif is_fp8:
                has_fp8[0] = True
            elif is_i8:
                has_i8[0] = True
            storage_name = str(getattr(ttype, "storage", "")).lower()
            if "ssbuf" in storage_name or "shared.ss" in storage_name:
                handoff[0] = True
            if ttype.shape:
                dim = static_dim_value(ttype.shape[-1])
                if dim is not None and dim > 0:
                    bw = int(getattr(ttype.dtype, "bit_width", 16))
                    inner_bytes.append(max(1, (dim * bw + 7) // 8))

    def _record_mesh(mesh) -> None:
        if mesh is None:
            return
        try:
            lvl_info = flattened_levels(mesh)
            for topology, (shape, _strides) in zip(mesh.topologies, lvl_info):
                tname = str(getattr(topology, "name", topology)).lower()
                if any(tok in tname for tok in ("block", "grid", "aic", "core")):
                    prod = 1
                    has_int = False
                    for dim in shape:
                        if isinstance(dim, int) and dim > 0:
                            prod *= dim
                            has_int = True
                    if has_int:
                        grid_tiles.append(prod)
        except Exception:
            return

    def _inspect_op(op, *, in_loop: bool) -> None:
        if op is None:
            return
        cls_name = type(op).__name__
        schema_name = str(getattr(getattr(op, "_op_schema", None), "name", ""))
        op_str = f"{cls_name} {schema_name}".lower()

        if cls_name in ("MatMul", "Conv2D", "AscendMmad", "Dot") or any(
            tok in op_str for tok in ("matmul", "conv2d", "mmad", "gemm")
        ):
            has_cube[0] = True

        if cls_name == "Quant" or any(
            tok in op_str for tok in ("quant", "bit_unpack", "mx_quant")
        ):
            has_quant_op[0] = True
            has_vec[0] = True
            scheme = str(getattr(op, "scheme", "")).lower()
            if "per_tensor" in scheme:
                quant_per_tensor[0] = True
            td = getattr(op, "target_dtype", None)
            if td is not None:
                is_fp8, is_fp4, is_i8 = _classify_dtype_flags(
                    getattr(td, "name", str(td))
                )
                has_fp8[0] = has_fp8[0] or is_fp8
                has_fp4[0] = has_fp4[0] or is_fp4
                has_i8[0] = has_i8[0] or is_i8

        if cls_name in (
            "IndexSelect",
            "IndexAdd",
            "IndexCopy",
            "AscendVgather",
            "AscendUbToGmAtomic",
            "AscendL0cToGmAtomic",
        ) or any(
            tok in op_str
            for tok in ("index_select", "index_add", "index_copy", "vgather", "gather", "scatter", "topk", "sort")
        ):
            indirect[0] = True
            has_vec[0] = True

        if cls_name in (
            "AscendDualCopy",
            "AscendDualCopyL0cToUb",
            "AscendDualCopyUbToL1",
            "AscendCrossCoreSync",
        ) or any(tok in op_str for tok in ("dual_copy", "cross_core", "ssbuf")):
            handoff[0] = True

        if cls_name == "SoftMax" or "softmax" in op_str:
            has_vec[0] = True
            if in_loop:
                has_softmax_in_loop[0] = True

        if "vexp" in op_str or (
            cls_name in ("Unary", "AscendVunary") and "exp" in str(getattr(op, "kind", "")).lower()
        ):
            has_vec[0] = True
            if in_loop:
                has_exp_in_loop[0] = True

        if "vmax" in op_str or (
            cls_name in ("Reduce", "AscendVreduce", "Binary", "AscendVbinary") and "max" in str(getattr(op, "kind", "")).lower()
        ):
            has_vec[0] = True
            if in_loop:
                has_max_in_loop[0] = True

        if cls_name.startswith("AscendV") or cls_name in (
            "Binary",
            "Unary",
            "Reduce",
            "Clamp",
            "Where",
            "Cast",
            "RMSNorm",
            "LayerNorm",
            "RoPE",
            "Gelu",
            "Silu",
            "ReLU",
            "Tanh",
            "Sigmoid",
            "Softplus",
            "TopK",
            "ArgMax",
            "AscendUbToReg",
            "AscendRegToUb",
        ):
            has_vec[0] = True

    def _is_static_bound(val) -> bool:
        if val is None:
            return False
        if isinstance(val, int) and not isinstance(val, bool):
            return True
        if isinstance(val, Constant) and isinstance(val.value, int) and not isinstance(val.value, bool):
            return True
        return static_dim_value(val) is not None

    def _walk_expr(expr: Expr | None, *, in_loop: bool = False) -> None:
        if expr is None:
            return
        _record_type(getattr(expr, "type", None))
        if isinstance(expr, Call):
            _inspect_op(expr.target, in_loop=in_loop)
            for arg in expr.args:
                _walk_expr(arg, in_loop=in_loop)
            return
        if isinstance(expr, IrTuple):
            for el in expr.elements:
                _walk_expr(el, in_loop=in_loop)
            return
        if isinstance(expr, ExecutionRegion):
            if expr.kind == SIMTVF_KIND.name or "simtvf" in expr.kind.lower():
                has_simt[0] = True
                indirect[0] = True
                has_vec[0] = True
            elif expr.kind == SIMDVF_KIND.name or "simdvf" in expr.kind.lower():
                has_vec[0] = True
            for param in expr.params:
                _record_type(getattr(param, "type", None))
            for arg in expr.args:
                _walk_expr(arg, in_loop=in_loop)
            _walk_expr(expr.body, in_loop=in_loop)
            return
        if isinstance(expr, MeshRegion):
            _record_mesh(expr.mesh)
            for param in expr.params:
                _record_type(getattr(param, "type", None))
            for arg in expr.args:
                _walk_expr(arg, in_loop=in_loop)
            _walk_expr(expr.body, in_loop=in_loop)
            return
        if isinstance(expr, LoopRegion):
            if static_dim_value(expr.extent) is None:
                static_bounds[0] = False
            for param in expr.params:
                _record_type(getattr(param, "type", None))
            for arg in expr.args:
                _walk_expr(arg, in_loop=in_loop)
            prev_exp, prev_max, prev_sm = (
                has_exp_in_loop[0],
                has_max_in_loop[0],
                has_softmax_in_loop[0],
            )
            has_exp_in_loop[0] = False
            has_max_in_loop[0] = False
            has_softmax_in_loop[0] = False
            _walk_expr(expr.body, in_loop=True)
            for yv in expr.yield_values:
                _walk_expr(yv, in_loop=True)
            if len(expr.yield_values) > 0:
                if (has_exp_in_loop[0] and has_max_in_loop[0]) or has_softmax_in_loop[0]:
                    recurrence[0] = "associative_online_rescale"
                elif recurrence[0] == "none":
                    recurrence[0] = "non_associative_scan"
            elif (has_exp_in_loop[0] and has_max_in_loop[0]) or has_softmax_in_loop[0]:
                recurrence[0] = "associative_online_rescale"
            has_exp_in_loop[0] = prev_exp or has_exp_in_loop[0]
            has_max_in_loop[0] = prev_max or has_max_in_loop[0]
            has_softmax_in_loop[0] = prev_sm or has_softmax_in_loop[0]
            return
        if isinstance(expr, HirFunction):
            for param in expr.params:
                _record_type(getattr(param, "type", None))
            _walk_expr(expr.body, in_loop=in_loop)

    def _walk_stmt(stmt: Stmt | None, *, in_loop: bool = False) -> None:
        if stmt is None:
            return
        if isinstance(stmt, Sequential):
            for child in stmt.body:
                _walk_stmt(child, in_loop=in_loop)
            return
        if isinstance(stmt, LetStmt):
            _record_type(getattr(stmt.var, "type", None))
            _walk_expr(stmt.value, in_loop=in_loop)
            return
        if isinstance(stmt, While):
            static_bounds[0] = False
            _walk_expr(stmt.cond, in_loop=True)
            _walk_stmt(stmt.body, in_loop=True)
            if recurrence[0] == "none":
                if (has_exp_in_loop[0] and has_max_in_loop[0]) or has_softmax_in_loop[0]:
                    recurrence[0] = "associative_online_rescale"
                else:
                    recurrence[0] = "non_associative_scan"
            return
        if isinstance(stmt, For):
            ext = getattr(stmt, "extent", None)
            if ext is not None:
                if not _is_static_bound(ext):
                    static_bounds[0] = False
            else:
                if not (_is_static_bound(stmt.start) and _is_static_bound(stmt.stop)):
                    static_bounds[0] = False
            _walk_stmt(stmt.body, in_loop=True)
            if (has_exp_in_loop[0] and has_max_in_loop[0]) or has_softmax_in_loop[0]:
                recurrence[0] = "associative_online_rescale"
            return
        if isinstance(stmt, If):
            _walk_expr(stmt.cond, in_loop=in_loop)
            _walk_stmt(stmt.then_body, in_loop=in_loop)
            _walk_stmt(stmt.else_body, in_loop=in_loop)
            return
        if isinstance(stmt, ExecutionScope):
            kind_str = str(getattr(stmt.kind, "name", stmt.kind))
            if kind_str == SIMTVF_KIND.name or "simtvf" in kind_str.lower():
                has_simt[0] = True
                indirect[0] = True
                has_vec[0] = True
            elif kind_str == SIMDVF_KIND.name or "simdvf" in kind_str.lower():
                has_vec[0] = True
            _walk_stmt(stmt.body, in_loop=in_loop)
            return
        if isinstance(stmt, MeshScope):
            _record_mesh(stmt.mesh)
            _walk_stmt(stmt.body, in_loop=in_loop)
            return
        if isinstance(stmt, Evaluate):
            _inspect_op(stmt.callable, in_loop=in_loop)
            for arg in stmt.args:
                _walk_expr(arg, in_loop=in_loop)
            return
        if hasattr(stmt, "args"):
            for arg in stmt.args:
                _record_type(getattr(arg, "type", None))

    if isinstance(fn_or_stmt, PrimFunction):
        for param in fn_or_stmt.params:
            _record_type(getattr(param, "type", None))
        _walk_stmt(fn_or_stmt.body)
    elif isinstance(fn_or_stmt, HirFunction):
        _walk_expr(fn_or_stmt)
    elif isinstance(fn_or_stmt, Stmt):
        _walk_stmt(fn_or_stmt)
    elif isinstance(fn_or_stmt, Expr):
        _walk_expr(fn_or_stmt)

    quant = _resolve_quant_granularity(
        has_fp8=has_fp8[0],
        has_fp4=has_fp4[0],
        has_i8=has_i8[0],
        has_quant_op=has_quant_op[0],
        per_tensor=quant_per_tensor[0],
    )
    return {
        "static_bounds": static_bounds[0],
        "indirect_gather": indirect[0],
        "dynamic_handoff": handoff[0],
        "quant": quant,
        "recurrence": recurrence[0],
        "has_simt_vf": has_simt[0],
        "has_cube": has_cube[0],
        "has_vec": has_vec[0],
        "inner_bytes": min(inner_bytes) if inner_bytes else 0,
        "total_tiles": max(grid_tiles) if grid_tiles else 0,
    }


def _infer_axes_from_semantic_dict(
    semantic: Mapping[str, object],
    *,
    simt_dominant: bool = False,
) -> TopologyAxes:
    if all(
        k in semantic for k in ("compute_engine", "memory_ring", "iteration_regularity")
    ):
        return TopologyAxes.from_dict(semantic)
    static_bounds = bool(semantic.get("static_affine_bounds", True))
    cube_ratio = float(semantic.get("cube_to_vector_ratio", 0.0) or 0.0)
    uses_cube = bool(semantic.get("uses_cube", False)) or cube_ratio > 0.0
    quant = str(semantic.get("quant_granularity", "none"))
    indirect = bool(semantic.get("indirect_sparse_gather", False))
    recurrence = str(semantic.get("cross_iter_recurrence", "none"))
    handoff = bool(semantic.get("dynamic_cross_core_handoff", False))
    inner_bytes = int(semantic.get("inner_contiguous_bytes", 0) or 0)
    is_non_assoc = recurrence == "non_associative_scan"
    is_cv_ring = (
        uses_cube
        and (recurrence == "associative_online_rescale" or handoff or indirect)
        and (static_bounds or indirect)
    )
    is_grouped = uses_cube and not is_cv_ring and not static_bounds
    is_simt = (
        not uses_cube
        and not is_non_assoc
        and (
            simt_dominant
            or indirect
            or handoff
            or (0 < inner_bytes < 32)
            or not static_bounds
        )
    )
    if is_non_assoc:
        return TopologyAxes(
            compute_engine="coupled_1c2v" if uses_cube else "aiv_only",
            memory_ring="persistent_onchip_state",
            iteration_regularity=(
                "dynamic_opcode_stream" if (not static_bounds and not uses_cube) else "affine_simd"
            ),
        )
    if is_cv_ring:
        return TopologyAxes(
            compute_engine="coupled_1c2v",
            memory_ring="bidirectional_cv_ring",
            iteration_regularity="irregular_simt" if indirect else "affine_simd",
        )
    if is_grouped:
        return TopologyAxes(
            compute_engine="coupled_1c2v",
            memory_ring="l0c_ub_epilogue",
            iteration_regularity="ragged_group",
        )
    if uses_cube:
        return TopologyAxes(
            compute_engine=(
                "coupled_1c2v" if (quant != "none" or handoff) else "aic_only"
            ),
            memory_ring="gm_l1_l0_fixpipe",
            iteration_regularity="affine_simd",
        )
    if is_simt:
        return TopologyAxes(
            compute_engine="aiv_only",
            memory_ring="ub_random_atomic",
            iteration_regularity="irregular_simt",
        )
    return TopologyAxes(
        compute_engine="aiv_only",
        memory_ring="gm_ub_stream",
        iteration_regularity="affine_simd",
    )


def _analyze_ir_and_ledger(
    ir_obj: object | None = None,
    *,
    ledger: ObjectivePhysicalLedger | None = None,
    target: AscendTarget | None = None,
    simt_dominant: bool = False,
    case_ratios: CasePhysicsRatios | None = None,
    semantic_features: Mapping[str, object] | None = None,
) -> tuple[TopologyAxes, dict[str, object]]:
    """Derive 3-axis TopologyAxes and plain semantic features from IR and ledger."""
    resolved_target = target or AscendTarget("ascend.950pr")
    aic_count = max(1, resolved_target.device.aic_count)

    info: dict[str, object] = {
        "static_bounds": True,
        "indirect_gather": False,
        "dynamic_handoff": False,
        "quant": "none",
        "recurrence": "none",
        "has_simt_vf": False,
        "has_cube": False,
        "has_vec": False,
        "inner_bytes": 0,
        "total_tiles": 0,
    }
    if ir_obj is not None:
        if isinstance(ir_obj, (PrimFunction, HirFunction, Stmt, LoopRegion, Expr)):
            info = _scan_tilefoundry_ir(ir_obj)
        else:
            from tvm import IRModule, tirx  # noqa: PLC0415

            target_fn = None
            if isinstance(ir_obj, IRModule):
                fns = [f for f in ir_obj.functions.values() if isinstance(f, tirx.PrimFunc)]
                target_fn = fns[0] if fns else None
            elif isinstance(ir_obj, tirx.PrimFunc):
                target_fn = ir_obj
            if target_fn is not None:
                info = _scan_tilelang_features(target_fn)

    static_bounds = bool(info["static_bounds"])
    has_simt_flag = bool(info["has_simt_vf"])
    if ledger is not None:
        dynamic_tags = {"while_loop", "data_dependent_branch"}
        if any(
            tag in dynamic_tags or tag.startswith("dynamic_for:")
            for tag in ledger.opaque_dynamic_regions
        ):
            static_bounds = False
        if "simt_vf" in ledger.opaque_dynamic_regions:
            has_simt_flag = True

    ratio = _ratio_from_ledger(ledger)
    if ratio == 0.0:
        if info["has_cube"] and info["has_vec"]:
            ratio = 1.0
        elif info["has_cube"]:
            ratio = 2.0

    tiles = int(info["total_tiles"])
    occupancy = float(tiles) / float(aic_count) if tiles > 0 else 0.0
    inner_b = int(info["inner_bytes"])
    if case_ratios is not None:
        inner_b = case_ratios.inner_contiguous_bytes
        occupancy = case_ratios.wave_occupancy_ratio

    merged: dict[str, object] = {
        "static_affine_bounds": static_bounds,
        "cube_to_vector_ratio": ratio,
        "quant_granularity": str(info["quant"]),
        "indirect_sparse_gather": bool(info["indirect_gather"]),
        "cross_iter_recurrence": str(info["recurrence"]),
        "dynamic_cross_core_handoff": bool(info["dynamic_handoff"]),
        "inner_contiguous_bytes": inner_b,
        "persistent_wave_occupancy": occupancy,
        "uses_cube": ratio > 0.0,
        "simt_dominant": bool(simt_dominant or has_simt_flag),
    }
    if semantic_features is not None:
        for k, v in semantic_features.items():
            merged[str(k)] = v

    axes = _infer_axes_from_semantic_dict(
        merged,
        simt_dominant=bool(merged.get("simt_dominant", False)),
    )
    if case_ratios is not None:
        axes = case_ratios.effective_axes(axes)
    return axes, merged


def route_execution_track(
    target_or_ir: object | None = None,
    *,
    ledger: ObjectivePhysicalLedger | None = None,
    target: AscendTarget | None = None,
    pipe_gaps: PipeGapReport | None = None,
    case_ratios: CasePhysicsRatios | None = None,
    router_enabled: bool = True,
    simt_dominant: bool = False,
    semantic_features: Mapping[str, object] | None = None,
) -> RouteDecision:
    """Route TopologyAxes, semantic dict, or IR+ledger via CapabilityGate."""
    if isinstance(target_or_ir, TopologyAxes):
        merged = (
            {str(k): v for k, v in semantic_features.items()}
            if semantic_features is not None
            else {}
        )
        if simt_dominant:
            merged["simt_dominant"] = True
        return CapabilityGate.route(
            target_or_ir,
            pipe_gaps=pipe_gaps,
            case_ratios=case_ratios,
            router_enabled=router_enabled,
            semantic_features=merged or None,
        )

    if isinstance(target_or_ir, Mapping):
        merged = {str(k): v for k, v in target_or_ir.items()}
        if semantic_features is not None:
            for k, v in semantic_features.items():
                merged[str(k)] = v
        if simt_dominant:
            merged["simt_dominant"] = True
        axes = _infer_axes_from_semantic_dict(
            merged,
            simt_dominant=bool(merged.get("simt_dominant", False)),
        )
        return CapabilityGate.route(
            axes,
            pipe_gaps=pipe_gaps,
            case_ratios=case_ratios,
            router_enabled=router_enabled,
            semantic_features=merged,
        )

    axes, merged = _analyze_ir_and_ledger(
        target_or_ir,
        ledger=ledger,
        target=target,
        simt_dominant=simt_dominant,
        case_ratios=case_ratios,
        semantic_features=semantic_features,
    )
    return CapabilityGate.route(
        axes,
        pipe_gaps=pipe_gaps,
        case_ratios=case_ratios,
        router_enabled=router_enabled,
        semantic_features=merged,
    )


def _to_snake_case(name: str) -> str:
    text = str(name).strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = text.replace("-", "_").replace(" ", "_").lower()
    return re.sub(r"_+", "_", text).strip("_")


def _extract_op_name_and_dtypes(proto_or_advisory: Mapping[str, object]) -> tuple[str, list[str], str]:
    op_block = proto_or_advisory.get("operator")
    source: Mapping[str, object] = (
        op_block if isinstance(op_block, Mapping) else proto_or_advisory
    )
    raw_name = (
        proto_or_advisory.get("cannbot_task")
        or proto_or_advisory.get("op_name")
        or proto_or_advisory.get("id")
        or source.get("schema")
        or source.get("name")
        or ""
    )
    op_name = _to_snake_case(str(raw_name))
    dtypes: list[str] = []
    for key in ("input_dtypes", "output_dtypes"):
        vals = proto_or_advisory.get(key)
        if isinstance(vals, (list, tuple)):
            dtypes.extend(str(v) for v in vals)
    for section in ("inputs", "outputs"):
        entries = source.get(section)
        if isinstance(entries, (list, tuple)):
            for entry in entries:
                if isinstance(entry, Mapping):
                    dt = entry.get("dtype")
                    if isinstance(dt, (list, tuple)):
                        dtypes.extend(str(d) for d in dt)
                    elif isinstance(dt, str):
                        dtypes.append(dt)
    parts: list[str] = [
        str(src.get(k, ""))
        for src in (proto_or_advisory, source)
        for k in ("category", "formula", "description", "semantic_contract", "quant_mode", "op_kind")
    ]
    attrs = source.get("attrs")
    if isinstance(attrs, (list, tuple)):
        for a in attrs:
            parts.append(str(a.get("name", "")) if isinstance(a, Mapping) else str(a))
    elif isinstance(attrs, Mapping):
        parts.extend(str(k) for k in attrs.keys())
    desc_text = " ".join(parts).lower()
    return op_name, dtypes, desc_text


def route_from_op_spec(
    proto_or_advisory: Mapping[str, object],
    *,
    case_ratios: CasePhysicsRatios | None = None,
    pipe_gaps: PipeGapReport | None = None,
    router_enabled: bool = True,
) -> RouteDecision:
    """Route a cann-bench proto.yaml or AdvisoryEnvelope dict to 3-axis TopologyAxes & 3 tracks."""
    _op_name, dtypes, desc_text = _extract_op_name_and_dtypes(proto_or_advisory)
    raw_axes = proto_or_advisory.get("topology_axes")
    explicit_axes: TopologyAxes | None = None
    if isinstance(raw_axes, TopologyAxes):
        explicit_axes = raw_axes
    elif isinstance(raw_axes, Mapping) and all(
        k in raw_axes for k in ("compute_engine", "memory_ring", "iteration_regularity")
    ):
        explicit_axes = TopologyAxes.from_dict(raw_axes)

    has_fp8 = False
    has_fp4 = False
    has_i8 = False
    for dt in dtypes:
        is_fp8, is_fp4, is_i8 = _classify_dtype_flags(dt)
        has_fp8 = has_fp8 or is_fp8
        has_fp4 = has_fp4 or is_fp4
        has_i8 = has_i8 or is_i8
    has_quant_structural = any(
        tok in desc_text
        for tok in ("quant", "antiquant", "scaleout", "pertokenquant", "qweight", "mxscale")
    )
    quant = _resolve_quant_granularity(
        has_fp8=has_fp8,
        has_fp4=has_fp4,
        has_i8=has_i8,
        has_quant_op=has_quant_structural,
    )

    axes = (
        explicit_axes
        if explicit_axes is not None
        else resolve_canonical_topology_axes(proto_or_advisory)
    )
    semantic_features: dict[str, object] = {
        "static_affine_bounds": axes.iteration_regularity == "affine_simd",
        "cube_to_vector_ratio": 1.0 if axes.uses_cube else 0.0,
        "quant_granularity": quant,
        "indirect_sparse_gather": axes.iteration_regularity == "irregular_simt",
        "cross_iter_recurrence": (
            "non_associative_scan"
            if axes.memory_ring == "persistent_onchip_state"
            else (
                "associative_online_rescale"
                if axes.memory_ring == "bidirectional_cv_ring"
                else "none"
            )
        ),
        "dynamic_cross_core_handoff": axes.memory_ring
        in ("l0c_ub_epilogue", "bidirectional_cv_ring"),
        "inner_contiguous_bytes": int(proto_or_advisory.get("inner_contiguous_bytes", 0) or 0),
        "persistent_wave_occupancy": float(proto_or_advisory.get("wave_occupancy", 0.0) or 0.0),
    }
    return CapabilityGate.route(
        axes,
        pipe_gaps=pipe_gaps,
        case_ratios=case_ratios,
        router_enabled=router_enabled,
        semantic_features=semantic_features,
    )


def _is_subbyte_dtype(dtype_str: str) -> bool:
    s = dtype_str.strip().lower()
    if s in ("i4", "u4"):
        return True
    return any(tok in s for tok in _SUBBYTE_DTYPE_TOKENS)


def route_initial_by_input_contract(
    spec_or_axes: TopologyAxes | Mapping[str, object] | object,
    *,
    case_ratios: CasePhysicsRatios | None = None,
    pipe_gaps: PipeGapReport | None = None,
) -> RouteDecision:
    """Causal t=0 router: escalate to ascendc_template ONLY for DSL-unrepresentable contracts.

    Starts every unseen/standard operator on ``tilelang_dsl`` without hindsight bias unless the
    input/output semantic contract itself requires:
      1. Sub-byte packed formats (``f4e2m1``, ``f4e1m2``, ``i4``, ``u4``, ``q4_0``, ``mxfp4``)
      2. In-place GM compare-and-swap (CAS) atomics
      3. Dynamic bytecode / opcode stream execution
    """
    subbyte_reasons: list[str] = []
    gm_cas_reasons: list[str] = []
    opcode_reasons: list[str] = []

    if isinstance(spec_or_axes, TopologyAxes):
        axes = (
            case_ratios.effective_axes(spec_or_axes)
            if case_ratios is not None
            else spec_or_axes
        )
        if axes.iteration_regularity == "dynamic_opcode_stream":
            opcode_reasons.append("input_contract: dynamic bytecode/opcode stream required")
        routed = CapabilityGate.route(axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios)
        features = dict(routed.features) if routed.features is not None else {}
        track = routed.track
        prims = routed.recommended_primitives
        score = routed.suitability_score
    else:
        if isinstance(spec_or_axes, Mapping):
            spec_map: Mapping[str, object] = spec_or_axes
        elif hasattr(spec_or_axes, "to_dict") and callable(getattr(spec_or_axes, "to_dict")):
            raw_dict = spec_or_axes.to_dict()
            if not isinstance(raw_dict, Mapping):
                raise TypeError("to_dict() must return a Mapping")
            spec_map = raw_dict
        else:
            raise TypeError(
                f"Expected TopologyAxes, Mapping, or SIL object, got {type(spec_or_axes).__name__}"
            )

        _op_name, dtypes, _desc_text = _extract_op_name_and_dtypes(spec_map)
        op_block = spec_map.get("operator")
        source: Mapping[str, object] = op_block if isinstance(op_block, Mapping) else spec_map
        contract_text = " ".join(
            str(src.get(k, ""))
            for src in (spec_map, source)
            for k in ("semantic_contract", "formula", "description", "atomics", "op_kind")
        ).lower()

        subbyte_hits = [dt for dt in dtypes if _is_subbyte_dtype(dt)]
        qmode = str(spec_map.get("quant_mode", source.get("quant_mode", ""))).strip().lower()
        if subbyte_hits:
            subbyte_reasons.append(f"input_contract: sub-byte packed dtypes {tuple(subbyte_hits)}")
        elif _is_subbyte_dtype(qmode):
            subbyte_reasons.append(f"input_contract: sub-byte quant_mode={qmode!r}")

        cas_flags = any(
            bool(src.get(k, False))
            for src in (spec_map, source)
            for k in (
                "requires_inplace_gm_cas",
                "inplace_gm_cas",
                "has_gm_cas_atomic",
                "gm_cas_atomic",
                "atomic_cas",
            )
        )
        if cas_flags or any(
            tok in contract_text
            for tok in (
                "gm_atomic_cas",
                "asc_atomic_cas",
                "in-place gm cas",
                "inplace gm cas",
                "gm cas",
                "compare_and_swap",
            )
        ):
            gm_cas_reasons.append("input_contract: in-place GM CAS atomic required")

        raw_axes = spec_map.get("topology_axes")
        axis_reg = ""
        if isinstance(raw_axes, Mapping):
            axis_reg = str(raw_axes.get("iteration_regularity", ""))
        elif hasattr(raw_axes, "iteration_regularity"):
            axis_reg = str(getattr(raw_axes, "iteration_regularity"))
        reg_val = str(spec_map.get("iteration_regularity", axis_reg)).strip().lower()

        opcode_flags = any(
            bool(src.get(k, False))
            for src in (spec_map, source)
            for k in (
                "dynamic_bytecode",
                "dynamic_opcode_stream",
                "has_opcode_stream",
                "opcode_stream",
                "bytecode_interpreter",
            )
        )
        if (
            reg_val == "dynamic_opcode_stream"
            or opcode_flags
            or any(tok in contract_text for tok in ("bytecode", "opcode_stream", "opcode stream"))
        ):
            opcode_reasons.append("input_contract: dynamic bytecode/opcode stream required")

        has_fp8 = False
        has_fp4 = bool(subbyte_hits)
        has_i8 = False
        for dt in dtypes:
            fp8_f, fp4_f, i8_f = _classify_dtype_flags(dt)
            has_fp8 = has_fp8 or fp8_f
            has_fp4 = has_fp4 or fp4_f
            has_i8 = has_i8 or i8_f
        quant = _resolve_quant_granularity(
            has_fp8=has_fp8,
            has_fp4=has_fp4,
            has_i8=has_i8,
            has_quant_op=any(tok in contract_text for tok in ("quant", "antiquant")),
        )
        if quant == "mixed_fp8_fp4" and not subbyte_reasons:
            subbyte_reasons.append("input_contract: sub-byte FP4 mixed_fp8_fp4 dtype")

        if isinstance(raw_axes, TopologyAxes):
            axes = raw_axes
        elif isinstance(raw_axes, Mapping) and all(
            k in raw_axes for k in ("compute_engine", "memory_ring", "iteration_regularity")
        ):
            axes = TopologyAxes.from_dict(raw_axes)
        else:
            axes = resolve_canonical_topology_axes(spec_map)
        if case_ratios is not None:
            axes = case_ratios.effective_axes(axes)
        if opcode_reasons and axes.iteration_regularity != "dynamic_opcode_stream":
            axes = TopologyAxes(
                compute_engine=axes.compute_engine,
                memory_ring="persistent_onchip_state",
                iteration_regularity="dynamic_opcode_stream",
            )

        semantic_features: dict[str, object] = {
            "static_affine_bounds": reg_val in ("", "affine_simd"),
            "cube_to_vector_ratio": 1.0 if axes.uses_cube else 0.0,
            "quant_granularity": quant,
            "indirect_sparse_gather": bool(
                spec_map.get("has_indirect_gather", reg_val == "irregular_simt")
            ),
            "cross_iter_recurrence": (
                "non_associative_scan"
                if bool(spec_map.get("has_persistent_state", False))
                else "none"
            ),
            "dynamic_cross_core_handoff": bool(spec_map.get("has_dead_l1_slice", False)),
            "inner_contiguous_bytes": int(spec_map.get("inner_contiguous_bytes", 0) or 0),
            "persistent_wave_occupancy": float(spec_map.get("wave_occupancy", 0.0) or 0.0),
            "requires_subbyte_dtype": bool(subbyte_reasons),
            "requires_inplace_gm_cas": bool(gm_cas_reasons),
        }
        routed = CapabilityGate.route(
            axes,
            pipe_gaps=pipe_gaps,
            case_ratios=case_ratios,
            semantic_features=semantic_features,
        )
        features = dict(routed.features) if routed.features is not None else semantic_features
        track = routed.track
        prims = routed.recommended_primitives
        score = routed.suitability_score

    hard_walls = [*subbyte_reasons, *gm_cas_reasons, *opcode_reasons]
    if hard_walls:
        return RouteDecision(
            track=track,
            topology_axes=axes,
            rationale=tuple(hard_walls),
            recommended_primitives=prims,
            recommended_backend_tier="ascendc_template",
            case_ratios=case_ratios,
            pipe_gaps=pipe_gaps,
            suitability_score=score,
            features=features,
        )

    return RouteDecision(
        track=track,
        topology_axes=axes,
        rationale=(
            "t=0 input contract representable in TileLang DSL; "
            "starting on tilelang_dsl without hindsight bias",
        ),
        recommended_primitives=prims,
        recommended_backend_tier="tilelang_dsl",
        case_ratios=case_ratios,
        pipe_gaps=pipe_gaps,
        suitability_score=score,
        features=features,
    )


def apply_case_ratios(
    decision: RouteDecision,
    ratios: CasePhysicsRatios,
) -> RouteDecision:
    """Project real per-case physics ratios onto a RouteDecision."""
    routed = CapabilityGate.route(
        decision.topology_axes,
        pipe_gaps=decision.pipe_gaps,
        case_ratios=ratios,
        semantic_features=decision.features,
    )
    backend_tier = (
        "ascendc_template"
        if decision.recommended_backend_tier == "ascendc_template"
        else routed.recommended_backend_tier
    )
    return RouteDecision(
        track=routed.track,
        topology_axes=routed.topology_axes,
        rationale=(
            *decision.rationale,
            f"case ratios contig_bytes={ratios.inner_contiguous_bytes}B "
            f"wave_occ={ratios.wave_occupancy_ratio:.3f} -> {routed.topology_axes.key()}",
        ),
        recommended_primitives=routed.recommended_primitives,
        recommended_backend_tier=backend_tier,
        case_ratios=ratios,
        pipe_gaps=decision.pipe_gaps,
        suitability_score=routed.suitability_score,
        features=routed.features,
    )


def evaluate_track_escalation(
    *,
    current_tier: str,
    roofline_eta: float,
    target_eta: float,
    stagnation_rounds: int,
    dominant_bound: str = "",
    compiler_summary: object | None = None,
    structural_features: Mapping[str, object] | None = None,
) -> EscalationDecision:
    """Closed-loop t>0 escalation from tilelang_dsl -> inline_ascendc_vf or ascendc_template."""
    tier = normalize_router_tier(current_tier)
    if tier not in VALID_ESCALATION_TIERS:
        raise ValueError(
            f"Unsupported current_tier {current_tier!r}; expected one of {VALID_ESCALATION_TIERS}"
        )

    if roofline_eta >= target_eta:
        return EscalationDecision(
            should_escalate=False,
            from_tier=tier,
            to_tier=tier,
            trigger_reasons=("roofline_target_met",),
        )

    sf_map: dict[str, object] = (
        {str(k): v for k, v in structural_features.items()}
        if isinstance(structural_features, Mapping)
        else (
            {str(k): v for k, v in structural_features.to_dict().items()}
            if hasattr(structural_features, "to_dict")
            else {}
        )
    )

    if tier == "ascendc_template":
        from tilefoundry_ascend.ascend.templates.registry import (  # noqa: PLC0415
            relax_facets_for_bottleneck,
        )

        raw_existing_mask = sf_map.get("freedom_mask")
        base_mask = (
            ConstraintRelaxationMask.from_dict(raw_existing_mask)
            if isinstance(raw_existing_mask, Mapping)
            else (
                raw_existing_mask
                if isinstance(raw_existing_mask, ConstraintRelaxationMask)
                else ConstraintRelaxationMask.bottom()
            )
        )
        relaxed_mask = relax_facets_for_bottleneck(
            base_mask,
            dominant_bound=dominant_bound,
            compiler_summary=compiler_summary,
            l1_overflow=bool(sf_map.get("l1_overflow", False)),
            ub_overflow=bool(sf_map.get("ub_overflow", False)),
        )
        if bool(
            sf_map.get("has_dead_l1_slice", False)
            or sf_map.get("requires_custom_ring_or_aliasing", False)
        ):
            relaxed_mask = relaxed_mask.unlock("sram_layout_and_aliasing", "pipeline_and_sync_graph")
        if bool(sf_map.get("dynamic_cross_core_handoff", False)):
            relaxed_mask = relaxed_mask.unlock("pipeline_and_sync_graph")
        if stagnation_rounds >= 4 and roofline_eta < min(0.65, float(target_eta)):
            relaxed_mask = ConstraintRelaxationMask.top()
        return EscalationDecision(
            should_escalate=False,
            from_tier=tier,
            to_tier=tier,
            trigger_reasons=("already_at_ascendc_template",),
            freedom_mask=relaxed_mask,
        )

    if stagnation_rounds < 2:
        return EscalationDecision(
            should_escalate=False,
            from_tier=tier,
            to_tier=tier,
            trigger_reasons=(f"stagnation_rounds={stagnation_rounds}<2; keep iterating on {tier}",),
        )

    if compiler_summary is None:
        comp: dict[str, object] = {}
    elif isinstance(compiler_summary, Mapping):
        comp = {str(k): v for k, v in compiler_summary.items()}
    elif hasattr(compiler_summary, "to_dict") and callable(getattr(compiler_summary, "to_dict")):
        comp = {str(k): v for k, v in compiler_summary.to_dict().items()}
    else:
        comp = {
            "spill_bytes": getattr(compiler_summary, "spill_bytes", 0),
            "vloopv2_loops": getattr(compiler_summary, "vloopv2_loops", 0),
            "software_loops": getattr(compiler_summary, "software_loops", 0),
            "has_local_membar_in_vf_call": getattr(
                compiler_summary, "has_local_membar_in_vf_call", False
            ),
            "vf_fusion_breaks": getattr(compiler_summary, "vf_fusion_breaks", 0),
            "alias_serialized_buffers": getattr(
                compiler_summary, "alias_serialized_buffers", 0
            ),
            "dual_issue_ipc_collapse": getattr(compiler_summary, "dual_issue_ipc_collapse", 0.0),
            "dcache_dcci_flushes": getattr(compiler_summary, "dcache_dcci_flushes", 0),
        }

    vf_breaks = int(comp.get("vf_fusion_breaks", 0) or 0)
    alias_ser = int(comp.get("alias_serialized_buffers", 0) or 0)
    local_membar = bool(comp.get("has_local_membar_in_vf_call", False))
    sw_loops = int(comp.get("software_loops", 0) or 0)
    vloopv2 = int(comp.get("vloopv2_loops", 0) or 0)
    ipc_collapse = float(comp.get("dual_issue_ipc_collapse", 0.0) or 0.0)
    dcci_flushes = int(comp.get("dcache_dcci_flushes", 0) or 0)
    spill_bytes = int(comp.get("spill_bytes", 0) or 0)

    cube_ratio = float(sf_map.get("cube_to_vector_ratio", 0.0) or 0.0)
    uses_cube = bool(sf_map.get("uses_cube", False)) or cube_ratio > 0.0
    cross_core = bool(
        sf_map.get("dynamic_cross_core_handoff", False)
        or sf_map.get("cross_core_mode4_blocked", False)
    )
    dead_l1 = bool(
        sf_map.get("has_dead_l1_slice", False) or sf_map.get("dead_l1_reuse_blocked", False)
    )
    indirect = bool(sf_map.get("indirect_sparse_gather", False))
    dist_wall = bool(
        sf_map.get("requires_dist_unpack", False)
        or sf_map.get("ub_load_store_bottleneck", False)
    )
    dom = dominant_bound.strip().lower()

    reasons: list[str] = []
    template_wall = False
    vf_wall = False

    if cross_core or dead_l1:
        template_wall = True
        reasons.append("cross_core_mode4_or_dead_l1_reuse_blocked_by_dsl")
    if alias_ser >= 2 or (alias_ser >= 1 and uses_cube):
        template_wall = True
        reasons.append(f"alias_serialized_buffers={alias_ser}_blocks_ping_pong_ring")
    if dcci_flushes > 0 and indirect:
        template_wall = True
        reasons.append(f"dcache_dcci_flushes={dcci_flushes}_with_indirect_gather")
    if tier == "inline_ascendc_vf":
        template_wall = True
        reasons.append("inline_ascendc_vf_stagnated")

    if vf_breaks > 0:
        vf_wall = True
        reasons.append(f"vf_fusion_breaks={vf_breaks}_ub_roundtrip")
    if local_membar:
        vf_wall = True
        reasons.append("local_membar_in_vf_call")
    if sw_loops > 0 and vloopv2 == 0:
        vf_wall = True
        reasons.append(f"vloopv2_missed_software_loops={sw_loops}")
    if ipc_collapse >= 0.4:
        vf_wall = True
        reasons.append(f"dual_issue_ipc_collapse={ipc_collapse:.2f}")
    if alias_ser > 0 and not uses_cube:
        vf_wall = True
        reasons.append(f"vector_alias_serialized_buffers={alias_ser}")
    if dcci_flushes > 0 and not indirect:
        vf_wall = True
        reasons.append(f"dcache_dcci_flushes={dcci_flushes}")
    if spill_bytes > 0:
        vf_wall = True
        reasons.append(f"compiler_spill_bytes={spill_bytes}")
    if dist_wall or any(tok in dom for tok in ("ub", "vector", "regbase", "spill")):
        vf_wall = True
        reasons.append(f"ub_vector_distribution_bottleneck({dom or 'dist'})")

    if template_wall:
        from tilefoundry_ascend.ascend.templates.registry import (  # noqa: PLC0415
            relax_facets_for_bottleneck,
        )

        relaxed_mask = relax_facets_for_bottleneck(
            ConstraintRelaxationMask.bottom(),
            dominant_bound=dominant_bound,
            compiler_summary=comp,
            l1_overflow=bool(sf_map.get("l1_overflow", False)),
            ub_overflow=bool(sf_map.get("ub_overflow", False)),
        )
        if dead_l1 or alias_ser > 0:
            relaxed_mask = relaxed_mask.unlock("sram_layout_and_aliasing")
        if cross_core or alias_ser > 0 or dcci_flushes > 0:
            relaxed_mask = relaxed_mask.unlock("pipeline_and_sync_graph")
        if dcci_flushes > 0:
            relaxed_mask = relaxed_mask.unlock("dma_and_cache_hierarchy")
        if stagnation_rounds >= 4 and roofline_eta < min(0.65, float(target_eta)):
            relaxed_mask = ConstraintRelaxationMask.top()
        return EscalationDecision(
            should_escalate=True,
            from_tier=tier,
            to_tier="ascendc_template",
            trigger_reasons=tuple(reasons),
            freedom_mask=relaxed_mask,
            suggested_primitives=(
                "CrossCoreSetFlag<mode=4>",
                "L1_slot_aliasing",
                "HalfL1Isolation",
            ),
        )

    if vf_wall and tier == "tilelang_dsl":
        return EscalationDecision(
            should_escalate=True,
            from_tier=tier,
            to_tier="inline_ascendc_vf",
            trigger_reasons=tuple(reasons),
            freedom_mask=ConstraintRelaxationMask.bottom(),
            suggested_primitives=(
                "LoadDist::DIST_UNPACK_B16",
                "StoreDist::DIST_PACK_B32",
                "Reg::ReduceDataBlock",
                "__no_simd_vf_fusion__ __simd_vf__",
            ),
        )

    return EscalationDecision(
        should_escalate=False,
        from_tier=tier,
        to_tier=tier,
        trigger_reasons=("no_track_a_lowering_wall_detected",),
    )


__all__ = [
    "CapabilityGate",
    "EscalationDecision",
    "RouteDecision",
    "VALID_BACKEND_TIERS",
    "VALID_ESCALATION_TIERS",
    "VALID_QUANT_GRANULARITIES",
    "VALID_RECURRENCES",
    "VALID_TRACKS",
    "apply_case_ratios",
    "evaluate_track_escalation",
    "normalize_router_tier",
    "resolve_canonical_topology_axes",
    "route_execution_track",
    "route_from_op_spec",
    "route_initial_by_input_contract",
]
