"""Pure 3-axis TopologyAxes CapabilityGate router and Track escalation evaluator."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tilefoundry_ascend.agent.router.isa_gap_scanner import PipeGapReport
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target import AscendTarget

VALID_QUANT_GRANULARITIES = ("none", "per_tensor", "per_token_or_block", "mixed_fp8_fp4")
VALID_RECURRENCES = ("none", "associative_online_rescale", "non_associative_scan")
VALID_TRACKS = ("developer", "simt_vf", "manual_ssbuf")
VALID_BACKEND_TIERS = ("tilelang_dsl", "ascendc_template")
VALID_ESCALATION_TIERS = ("tilelang_dsl", "inline_ascendc_vf", "ascendc_template")

_ROUTER_TIER_ALIASES: dict[str, str] = {
    "tilelang_dsl": "tilelang_dsl", "cce_cpp20": "ascendc_template", "cce_trackb": "ascendc_template",
    "hybrid_trackb": "ascendc_template", "tl_templates_cpp20": "ascendc_template",
    "ascendc_trackb": "ascendc_template", "ascendc_template": "ascendc_template",
    "inline_cce_vf": "inline_ascendc_vf", "inline_ascendc_vf": "inline_ascendc_vf",
}
_SUBBYTE_DTYPE_TOKENS = (
    "float4", "fp4", "mxfp4", "f4e2m1", "f4e1m2", "e2m1", "e1m2", "int4", "uint4", "q4_0",
)
_PRIMITIVES_BY_RING: dict[str, tuple[str, ...]] = {
    "persistent_onchip_state": ("T.Persistent", "T.Stage", "T.dual_copy", "T.SimdVF"),
    "bidirectional_cv_ring": ("T.Scope", "T.alloc_ss", "T.dual_copy", "T.SimdVF"),
    "l0c_ub_epilogue": ("T.Scope", "T.alloc_ss", "T.dual_copy", "T.Persistent"),
    "gm_l1_l0_fixpipe": ("T.Pipelined", "T.gemm", "T.dual_copy", "T.Persistent"),
    "ub_random_atomic": ("T.SimtVF", "T.get_thread_binding", "T.alloc_ss", "T.Persistent"),
}
_BLOCKING_GAP_BY_RING: dict[str, tuple[str, ...]] = {
    "gm_ub_stream": ("dma_dual_copy_and_nd2nz", "simd_vld_vst"),
    "gm_l1_l0_fixpipe": ("dma_dual_copy_and_nd2nz",),
    "bidirectional_cv_ring": ("dma_dual_copy_and_nd2nz", "crosscore_flag_set_wait"),
    "persistent_onchip_state": ("dma_dual_copy_and_nd2nz",),
    "l0c_ub_epilogue": ("fixpipe_l0c_to_ub_split_m",),
    "ub_random_atomic": ("simt_gm_atomic_cas", "hardware_mutex_28", "simd_vgather_vscatter"),
}
_BLOCKING_GAP_BY_REGULARITY: dict[str, tuple[str, ...]] = {
    "irregular_simt": ("simt_gm_atomic_cas", "hardware_mutex_28", "simd_vgather_vscatter"),
    "ragged_group": ("fixpipe_l0c_to_ub_split_m",),
    "dynamic_opcode_stream": ("scalar_set_ctrl_spr", "simt_gm_atomic_cas"),
}
_RING_SCORE_DEDUCTION = {
    "bidirectional_cv_ring": 0.15, "persistent_onchip_state": 0.20,
    "l0c_ub_epilogue": 0.25, "ub_random_atomic": 0.35,
}
_REGULARITY_SCORE_DEDUCTION = {"dynamic_opcode_stream": 0.35, "ragged_group": 0.25, "irregular_simt": 0.25}


def normalize_router_tier(value: str) -> str:
    """Normalize a canonical dialect id or legacy alias to the router's internal tier id."""
    normalized = _ROUTER_TIER_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported router tier {value!r}; expected one of {tuple(_ROUTER_TIER_ALIASES)}")
    return normalized


def _primitives_for_axes(axes: TopologyAxes) -> tuple[str, ...]:
    if axes.iteration_regularity == "dynamic_opcode_stream":
        return ("T.SimdVF", "T.SimtVF", "T.Persistent")
    if axes.memory_ring in _PRIMITIVES_BY_RING:
        return _PRIMITIVES_BY_RING[axes.memory_ring]
    if axes.iteration_regularity == "irregular_simt":
        return _PRIMITIVES_BY_RING["ub_random_atomic"]
    return ("T.Pipelined", "T.Stage", "T.SimdVF", "T.dual_copy")


def _coerce_axes(raw: TopologyAxes | Mapping[str, object]) -> TopologyAxes:
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping):
        return TopologyAxes.from_dict(raw)
    raise TypeError(f"Expected TopologyAxes or Mapping, got {type(raw).__name__}")


def _as_dict(obj: object | None) -> dict[str, object]:
    if isinstance(obj, Mapping):
        return {str(k): v for k, v in obj.items()}
    if obj is not None and callable(getattr(obj, "to_dict", None)) and isinstance(res := obj.to_dict(), Mapping):
        return {str(k): v for k, v in res.items()}
    return {}


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
            object.__setattr__(self, "topology_axes", TopologyAxes.from_dict(self.topology_axes))
        elif not isinstance(self.topology_axes, TopologyAxes):
            raise TypeError(f"RouteDecision.topology_axes must be TopologyAxes, got {type(self.topology_axes).__name__}")
        tier = normalize_router_tier(self.recommended_backend_tier)
        if tier not in VALID_BACKEND_TIERS:
            raise ValueError(f"unsupported backend tier {self.recommended_backend_tier!r}")
        object.__setattr__(self, "recommended_backend_tier", tier)
        score = float(self.suitability_score)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(f"RouteDecision.suitability_score must be in [0.0, 1.0], got {score!r}")
        object.__setattr__(self, "suitability_score", score)
        if self.features is not None:
            if not isinstance(self.features, Mapping):
                raise TypeError(f"RouteDecision.features must be Mapping or None, got {type(self.features).__name__}")
            object.__setattr__(self, "features", {str(k): v for k, v in self.features.items()})


class CapabilityGate:
    """Pure 3-axis TopologyAxes router and ISA suitability scorer."""

    @staticmethod
    def suitability_score(
        topology_axes: TopologyAxes, pipe_gaps: PipeGapReport | None = None,
        case_ratios: CasePhysicsRatios | None = None,
    ) -> float:
        """Compute deterministic [0.0, 1.0] DSL suitability score from 3 axes, gaps, and ratios."""
        base_axes = _coerce_axes(topology_axes)
        axes = case_ratios.effective_axes(base_axes) if case_ratios is not None else base_axes
        blocking = set(_BLOCKING_GAP_BY_RING.get(axes.memory_ring, ())) | set(
            _BLOCKING_GAP_BY_REGULARITY.get(axes.iteration_regularity, ())
        )
        if axes.uses_cube and axes.memory_ring == "persistent_onchip_state":
            blocking.add("crosscore_flag_set_wait")
        if pipe_gaps is not None and any(pipe_gaps.has_gap(p) for p in blocking):
            return 0.0

        score = (
            1.0 - _RING_SCORE_DEDUCTION.get(axes.memory_ring, 0.0)
            - _REGULARITY_SCORE_DEDUCTION.get(axes.iteration_regularity, 0.0)
            - (0.20 if case_ratios is not None and case_ratios.contig_atom_ratio < 1.0 else 0.0)
        )
        if pipe_gaps is not None and pipe_gaps.missing_primitives:
            if (non_blocking := sum(1 for p in pipe_gaps.missing_primitives if p not in blocking)) > 0:
                score -= min(0.25, 0.015 * float(non_blocking))
        return round(max(0.0, min(1.0, score)), 6)

    @staticmethod
    def route(
        topology_axes: TopologyAxes, *, pipe_gaps: PipeGapReport | None = None,
        case_ratios: CasePhysicsRatios | None = None, router_enabled: bool = True,
        semantic_features: Mapping[str, object] | None = None,
    ) -> RouteDecision:
        """Route a 3-axis TopologyAxes coordinate to execution track and backend tier."""
        base_axes = _coerce_axes(topology_axes)
        if not router_enabled:
            return RouteDecision(
                track="developer", topology_axes=base_axes,
                rationale=("track router disabled (ablation); neutral developer route",),
                recommended_primitives=_primitives_for_axes(base_axes),
                recommended_backend_tier="tilelang_dsl", case_ratios=case_ratios, pipe_gaps=pipe_gaps,
            )

        axes = case_ratios.effective_axes(base_axes) if case_ratios is not None else base_axes
        score = CapabilityGate.suitability_score(axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios)
        feat_dict = _as_dict(semantic_features)
        feat_dict.update({
            "compute_engine": axes.compute_engine, "memory_ring": axes.memory_ring,
            "iteration_regularity": axes.iteration_regularity, "uses_cube": axes.uses_cube,
            "suitability_score": score,
        })
        if case_ratios is not None:
            feat_dict.update({
                "inner_contiguous_bytes": case_ratios.inner_contiguous_bytes,
                "persistent_wave_occupancy": case_ratios.wave_occupancy_ratio,
                "wave_occupancy_ratio": case_ratios.wave_occupancy_ratio,
                "contig_atom_ratio": case_ratios.contig_atom_ratio,
                "oi_ridge_ratio": case_ratios.oi_ridge_ratio, "has_empty_group": case_ratios.has_empty_group,
            })

        has_handoff = bool(feat_dict.get("dynamic_cross_core_handoff", False))
        has_indirect = bool(feat_dict.get("indirect_sparse_gather", False))
        simt_dom = bool(feat_dict.get("simt_dominant", False))
        is_non_assoc = str(feat_dict.get("cross_iter_recurrence", "none")) == "non_associative_scan"
        inner_bytes = int(feat_dict.get("inner_contiguous_bytes", 0) or 0)
        sub32 = (0 < inner_bytes < 32) or (case_ratios is not None and case_ratios.contig_atom_ratio < 1.0)

        reasons = [f"3-axis route -> {axes.key()} (score={score:.3f})"]
        if axes.memory_ring in ("bidirectional_cv_ring", "l0c_ub_epilogue") or has_handoff or (
            axes.uses_cube and (has_indirect or axes.iteration_regularity in ("irregular_simt", "ragged_group"))
        ):
            track = "manual_ssbuf"
            if has_indirect:
                reasons.append("indirect sparse/page gather detected")
            if has_handoff:
                reasons.append("dynamic cross-core metadata/scale handoff detected")
        elif (
            axes.memory_ring in ("ub_random_atomic", "persistent_onchip_state")
            or axes.iteration_regularity in ("irregular_simt", "dynamic_opcode_stream")
            or is_non_assoc or simt_dom or (sub32 and not axes.uses_cube)
        ):
            track = "simt_vf"
            if simt_dom:
                reasons.append("fine-grained thread-level vector reduction flagged")
            if is_non_assoc:
                reasons.append("non-associative cross-iteration recurrence")
            if sub32 and not axes.uses_cube:
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
            reasons.append(f"case ratios contig_bytes={case_ratios.inner_contiguous_bytes}B wave_occ={case_ratios.wave_occupancy_ratio:.3f} -> {axes.key()}")

        return RouteDecision(
            track=track, topology_axes=axes, rationale=tuple(reasons),
            recommended_primitives=_primitives_for_axes(axes), recommended_backend_tier=backend_tier,
            case_ratios=case_ratios, pipe_gaps=pipe_gaps, suitability_score=score, features=feat_dict,
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
        n_from, n_to = normalize_router_tier(self.from_tier), normalize_router_tier(self.to_tier)
        if n_from not in VALID_ESCALATION_TIERS:
            raise ValueError(f"unsupported from_tier {self.from_tier!r}")
        if n_to not in VALID_ESCALATION_TIERS:
            raise ValueError(f"unsupported to_tier {self.to_tier!r}")
        object.__setattr__(self, "from_tier", n_from)
        object.__setattr__(self, "to_tier", n_to)
        if isinstance(self.freedom_mask, Mapping):
            object.__setattr__(self, "freedom_mask", ConstraintRelaxationMask.from_dict(self.freedom_mask))


def _ratio_from_ledger(ledger: ObjectivePhysicalLedger | None) -> float:
    if ledger is None or not ledger.pipe_cycles:
        return 0.0
    aic = sum(c for k, c in ledger.pipe_cycles.items() if k.startswith("AIC.") and k != "AIC.PIPE_S")
    aiv = sum(
        c for k, c in ledger.pipe_cycles.items()
        if k.startswith(("AIV0.", "AIV1.")) and not k.endswith("PIPE_S")
    )
    return 0.0 if aic == 0 else (float(aic) if aiv == 0 else float(aic) / float(aiv))


def _classify_dtype_flags(dtype_str: str) -> tuple[bool, bool, bool]:
    s = dtype_str.lower()
    is_fp4 = any(t in s for t in _SUBBYTE_DTYPE_TOKENS) or s in ("i4", "u4")
    is_fp8 = not is_fp4 and any(t in s for t in ("float8", "fp8", "f8e", "e4m3", "e5m2", "e8m0", "hif8", "hifloat8"))
    is_i8 = not is_fp4 and not is_fp8 and (any(t in s for t in ("int8", "uint8")) or s in ("i8", "u8"))
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
    return "per_token_or_block" if (has_fp8 or has_fp4 or has_i8 or has_quant_op) else "none"


def _scan_dtypes_quant(dtypes: Iterable[str], *, has_quant_op: bool = False) -> tuple[bool, bool, bool, str]:
    has_fp8 = has_fp4 = has_i8 = False
    for dt in dtypes:
        fp8, fp4, i8 = _classify_dtype_flags(dt)
        has_fp8, has_fp4, has_i8 = has_fp8 or fp8, has_fp4 or fp4, has_i8 or i8
    quant = _resolve_quant_granularity(
        has_fp8=has_fp8, has_fp4=has_fp4, has_i8=has_i8, has_quant_op=has_quant_op
    )
    return has_fp8, has_fp4, has_i8, quant


def _scan_tilelang_features(func) -> dict[str, object]:
    """Scan a PrimFunc for static bounds, quant, gather, recurrence, and sync."""
    from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    static_bounds, indirect_gather, dynamic_handoff = True, False, False
    has_fp8 = has_fp4 = has_i8 = has_ssbuf = has_cube = has_vec = has_online_exp = has_non_assoc_scan = has_simt_vf = False
    min_inner_bytes: list[int] = []
    grid_tiles: list[int] = []
    indirect_vars: list[object] = []

    def _is_indirect_var(v) -> bool:
        return any(hasattr(v, "same_as") and v.same_as(iv) for iv in indirect_vars)

    def _check_dtype(dt_obj) -> None:
        nonlocal has_fp8, has_fp4, has_i8
        fp8, fp4, i8 = _classify_dtype_flags(str(dt_obj))
        has_fp4, has_fp8, has_i8 = has_fp4 or fp4, has_fp8 or (not fp4 and fp8), has_i8 or (not fp4 and not fp8 and i8)

    def _expr_matches(expr, pred) -> bool:
        hit = False
        def _f(node) -> None:
            nonlocal hit
            if not hit and pred(node):
                hit = True
        tirx.stmt_functor.post_order_visit(expr, _f)
        return hit

    def _has_load(expr) -> bool:
        return _expr_matches(expr, lambda n: isinstance(n, tirx.BufferLoad))

    def _has_indirect_expr(expr) -> bool:
        return _expr_matches(expr, lambda n: isinstance(n, tirx.BufferLoad) or (isinstance(n, tirx.Var) and _is_indirect_var(n)))

    def _analyze_loop_recurrence(loop_node, *, is_while: bool) -> None:
        nonlocal has_online_exp, has_non_assoc_scan
        loaded, stored, gemm_dst = set(), set(), set()
        loop_exp = loop_online = loop_non_add = False
        non_add = (tirx.Mul, tirx.Sub, tirx.Div, tirx.FloorDiv, tirx.Mod, tirx.Select, tirx.Call)

        def _scan_body(node) -> None:
            nonlocal loop_exp, loop_online, loop_non_add
            if isinstance(node, tirx.BufferLoad):
                loaded.add(node.buffer)
            elif isinstance(node, tirx.BufferStore):
                stored.add(node.buffer)
                if _expr_matches(node.value, lambda r: isinstance(r, non_add)):
                    loop_non_add = True
            elif isinstance(node, tirx.Max):
                loop_online = True
            elif isinstance(node, tirx.Call):
                op_name = str(getattr(node.op, "name", node.op)).lower()
                if "exp" in op_name:
                    loop_exp = True
                if any(t in op_name for t in ("max", "expdif", "online")):
                    loop_online = True
                if ("gemm" in op_name or "mmad" in op_name) and len(node.args) >= 3:
                    c_arg = node.args[2]
                    if isinstance(c_arg, tirx.BufferLoad):
                        gemm_dst.add(c_arg.buffer)
                    elif isinstance(c_arg, tirx.Call) and c_arg.args and isinstance(c_arg.args[0], tirx.BufferLoad):
                        gemm_dst.add(c_arg.args[0].buffer)

        tirx.stmt_functor.post_order_visit(loop_node.body, _scan_body)
        raw_bufs = (loaded & stored) - gemm_dst
        if loop_exp and (loop_online or raw_bufs):
            has_online_exp = True
        elif (is_while and (stored or raw_bufs)) or (
            raw_bufs and (loop_non_add or len(raw_bufs) >= 1)
            and not (hasattr(ext := getattr(loop_node, "extent", None), "value") and isinstance(ext.value, int) and ext.value <= 1)
        ):
            has_non_assoc_scan = True

    for param in func.params:
        if (buf := func.buffer_map.get(param)) is not None:
            _check_dtype(buf.dtype)
            if buf.shape and hasattr(last := buf.shape[-1], "value") and isinstance(last.value, int):
                min_inner_bytes.append(int(last.value) * ((TLDataType(str(buf.dtype)).bits + 7) // 8))

    def _check_alloc_buf(buf) -> None:
        nonlocal has_ssbuf, dynamic_handoff
        _check_dtype(buf.dtype)
        if buf.scope() == "shared.ss":
            has_ssbuf = dynamic_handoff = True

    def _visit(node) -> None:
        nonlocal static_bounds, indirect_gather, dynamic_handoff, has_cube, has_vec, has_online_exp, has_simt_vf
        if isinstance(node, tirx.While):
            static_bounds = False
            _analyze_loop_recurrence(node, is_while=True)
        elif isinstance(node, tirx.Bind):
            if _has_indirect_expr(node.value):
                indirect_vars.append(node.var)
        elif isinstance(node, tirx.For):
            ext, tb = node.extent, getattr(node, "thread_binding", None)
            tb_tag = str(getattr(tb, "thread_tag", "")) if tb is not None else ""
            if tb is not None and ("block" in tb_tag or tb_tag == ""):
                if hasattr(ext, "value") and isinstance(ext.value, int):
                    grid_tiles.append(int(ext.value))
            else:
                if not (hasattr(ext, "value") and isinstance(ext.value, int)):
                    static_bounds = False
                if tb is None:
                    _analyze_loop_recurrence(node, is_while=False)
        elif isinstance(node, tirx.IfThenElse) and _has_load(node.condition):
            static_bounds = False
        elif isinstance(node, tirx.AttrStmt):
            key = str(node.attr_key)
            if key == "thread_extent" and hasattr(node.value, "value") and isinstance(node.value.value, int):
                iv_name = getattr(getattr(node.node, "var", None), "name_hint", "")
                tag = str(getattr(node.node, "thread_tag", ""))
                if "blockIdx" in tag or "blockIdx" in iv_name:
                    grid_tiles.append(int(node.value.value))
                elif "cthread" not in tag and "cthread" not in iv_name:
                    has_simt_vf = True
            elif "simt" in key.lower():
                has_simt_vf = True
        elif isinstance(node, tirx.AllocBuffer):
            _check_alloc_buf(node.buffer)
        elif isinstance(node, tirx.SBlock):
            for iv in getattr(node, "iter_vars", ()):
                dom_ext = getattr(getattr(iv, "dom", None), "extent", None)
                if "blockIdx" in str(getattr(iv, "thread_tag", "")) and hasattr(dom_ext, "value") and isinstance(dom_ext.value, int):
                    grid_tiles.append(int(dom_ext.value))
            for buf in node.alloc_buffers:
                _check_alloc_buf(buf)
        elif isinstance(node, tirx.BufferLoad) and any(_has_indirect_expr(i) for i in node.indices):
            indirect_gather = has_vec = True
        elif isinstance(node, tirx.Call):
            op_name = str(getattr(node.op, "name", node.op)).lower()
            if "gather" in op_name or "scatter" in op_name:
                indirect_gather = has_vec = True
            if "cross_core" in op_name or "ssbuf" in op_name:
                dynamic_handoff = True
            if "gemm" in op_name or "mmad" in op_name:
                has_cube = True
            if "simd" in op_name or "exp" in op_name:
                has_vec = True
            if "expdif" in op_name or "online" in op_name:
                has_online_exp = True
            if "region" in op_name and len(node.args) >= 3 and isinstance(node.args[0], tirx.BufferLoad):
                bl, last = node.args[0], node.args[-1]
                if any(_has_indirect_expr(i) for i in bl.indices):
                    indirect_gather = True
                if hasattr(last, "value") and isinstance(last.value, int):
                    min_inner_bytes.append(int(last.value) * max(1, (TLDataType(str(bl.buffer.dtype)).bits + 7) // 8))

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    quant = _resolve_quant_granularity(has_fp8=has_fp8, has_fp4=has_fp4, has_i8=has_i8, has_quant_op=False)
    recurrence = "associative_online_rescale" if has_online_exp else ("non_associative_scan" if has_non_assoc_scan else "none")
    return {
        "static_bounds": static_bounds, "indirect_gather": indirect_gather,
        "dynamic_handoff": dynamic_handoff or has_ssbuf, "quant": quant, "recurrence": recurrence,
        "has_simt_vf": has_simt_vf, "has_cube": has_cube, "has_vec": has_vec,
        "inner_bytes": min(min_inner_bytes) if min_inner_bytes else 0,
        "total_tiles": grid_tiles[0] if grid_tiles else 0,
    }


def _infer_axes_from_semantic_dict(semantic: Mapping[str, object], *, simt_dominant: bool = False) -> TopologyAxes:
    if all(k in semantic for k in ("compute_engine", "memory_ring", "iteration_regularity")):
        return TopologyAxes.from_dict(semantic)
    static_bounds = bool(semantic.get("static_affine_bounds", True))
    uses_cube = bool(semantic.get("uses_cube", False)) or float(semantic.get("cube_to_vector_ratio", 0.0) or 0.0) > 0.0
    quant = str(semantic.get("quant_granularity", "none"))
    indirect = bool(semantic.get("indirect_sparse_gather", False))
    recurrence = str(semantic.get("cross_iter_recurrence", "none"))
    handoff = bool(semantic.get("dynamic_cross_core_handoff", False))
    inner_bytes = int(semantic.get("inner_contiguous_bytes", 0) or 0)

    if recurrence == "non_associative_scan":
        return TopologyAxes(
            "coupled_1c2v" if uses_cube else "aiv_only", "persistent_onchip_state",
            "dynamic_opcode_stream" if (not static_bounds and not uses_cube) else "affine_simd",
        )
    if uses_cube and (recurrence == "associative_online_rescale" or handoff or indirect) and (static_bounds or indirect):
        return TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "irregular_simt" if indirect else "affine_simd")
    if uses_cube and not static_bounds:
        return TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
    if uses_cube:
        return TopologyAxes("coupled_1c2v" if (quant != "none" or handoff) else "aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    if simt_dominant or indirect or handoff or (0 < inner_bytes < 32) or not static_bounds:
        return TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")
    return TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")


def _analyze_ir_and_ledger(
    ir_obj: object | None = None, *, ledger: ObjectivePhysicalLedger | None = None,
    target: AscendTarget | None = None, simt_dominant: bool = False,
    case_ratios: CasePhysicsRatios | None = None, semantic_features: Mapping[str, object] | None = None,
) -> tuple[TopologyAxes, dict[str, object]]:
    aic_count = max(1, (target or AscendTarget("ascend.950pr")).device.aic_count)
    info: dict[str, object] = {
        "static_bounds": True, "indirect_gather": False, "dynamic_handoff": False, "quant": "none",
        "recurrence": "none", "has_simt_vf": False, "has_cube": False, "has_vec": False, "inner_bytes": 0, "total_tiles": 0,
    }
    if ir_obj is not None:
        from tvm import IRModule, tirx  # noqa: PLC0415

        target_fn = next((f for f in ir_obj.functions.values() if isinstance(f, tirx.PrimFunc)), None) if isinstance(ir_obj, IRModule) else (ir_obj if isinstance(ir_obj, tirx.PrimFunc) else None)
        if target_fn is not None:
            info = _scan_tilelang_features(target_fn)

    static_bounds, has_simt_flag = bool(info["static_bounds"]), bool(info["has_simt_vf"])
    if ledger is not None:
        if any(t in {"while_loop", "data_dependent_branch"} or t.startswith("dynamic_for:") for t in ledger.opaque_dynamic_regions):
            static_bounds = False
        if "simt_vf" in ledger.opaque_dynamic_regions:
            has_simt_flag = True

    ratio = _ratio_from_ledger(ledger) or (1.0 if info["has_cube"] and info["has_vec"] else (2.0 if info["has_cube"] else 0.0))
    tiles = int(info["total_tiles"])
    merged: dict[str, object] = {
        "static_affine_bounds": static_bounds, "cube_to_vector_ratio": ratio,
        "quant_granularity": str(info["quant"]), "indirect_sparse_gather": bool(info["indirect_gather"]),
        "cross_iter_recurrence": str(info["recurrence"]), "dynamic_cross_core_handoff": bool(info["dynamic_handoff"]),
        "inner_contiguous_bytes": case_ratios.inner_contiguous_bytes if case_ratios else int(info["inner_bytes"]),
        "persistent_wave_occupancy": case_ratios.wave_occupancy_ratio if case_ratios else (float(tiles) / float(aic_count) if tiles > 0 else 0.0),
        "uses_cube": ratio > 0.0, "simt_dominant": bool(simt_dominant or has_simt_flag),
        **_as_dict(semantic_features),
    }
    axes = _infer_axes_from_semantic_dict(merged, simt_dominant=bool(merged.get("simt_dominant", False)))
    return (case_ratios.effective_axes(axes) if case_ratios is not None else axes), merged


def route_execution_track(
    target_or_ir: object | None = None, *, ledger: ObjectivePhysicalLedger | None = None,
    target: AscendTarget | None = None, pipe_gaps: PipeGapReport | None = None,
    case_ratios: CasePhysicsRatios | None = None, router_enabled: bool = True,
    simt_dominant: bool = False, semantic_features: Mapping[str, object] | None = None,
) -> RouteDecision:
    """Route TopologyAxes, semantic dict, or IR+ledger via CapabilityGate."""
    if isinstance(target_or_ir, TopologyAxes):
        merged = _as_dict(semantic_features)
        if simt_dominant:
            merged["simt_dominant"] = True
        return CapabilityGate.route(target_or_ir, pipe_gaps=pipe_gaps, case_ratios=case_ratios, router_enabled=router_enabled, semantic_features=merged or None)
    if isinstance(target_or_ir, Mapping):
        merged = {**_as_dict(target_or_ir), **_as_dict(semantic_features)}
        if simt_dominant:
            merged["simt_dominant"] = True
        axes = _infer_axes_from_semantic_dict(merged, simt_dominant=bool(merged.get("simt_dominant", False)))
    else:
        axes, merged = _analyze_ir_and_ledger(
            target_or_ir, ledger=ledger, target=target, simt_dominant=simt_dominant,
            case_ratios=case_ratios, semantic_features=semantic_features,
        )
    return CapabilityGate.route(axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios, router_enabled=router_enabled, semantic_features=merged)


def _to_snake_case(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip().split("(", 1)[0].strip())
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    return re.sub(r"_+", "_", text.replace("-", "_").replace(" ", "_").lower()).strip("_")


def _extract_op_name_and_dtypes(proto_or_advisory: Mapping[str, object]) -> tuple[str, list[str], str]:
    op_block = proto_or_advisory.get("operator")
    source: Mapping[str, object] = op_block if isinstance(op_block, Mapping) else proto_or_advisory
    raw_name = (
        proto_or_advisory.get("cannbot_task") or proto_or_advisory.get("op_name")
        or proto_or_advisory.get("id") or source.get("schema") or source.get("name") or ""
    )
    dtypes: list[str] = []
    for key in ("input_dtypes", "output_dtypes"):
        if isinstance(vals := proto_or_advisory.get(key), (list, tuple)):
            dtypes.extend(str(v) for v in vals)
    for section in ("inputs", "outputs"):
        if isinstance(entries := source.get(section), (list, tuple)):
            for entry in entries:
                if isinstance(entry, Mapping):
                    dt = entry.get("dtype")
                    dtypes.extend(str(d) for d in dt) if isinstance(dt, (list, tuple)) else (dtypes.append(dt) if isinstance(dt, str) else None)
    parts = [str(src.get(k, "")) for src in (proto_or_advisory, source) for k in ("category", "formula", "description", "semantic_contract", "quant_mode", "op_kind")]
    attrs = source.get("attrs")
    if isinstance(attrs, (list, tuple)):
        parts.extend(str(a.get("name", "")) if isinstance(a, Mapping) else str(a) for a in attrs)
    elif isinstance(attrs, Mapping):
        parts.extend(str(k) for k in attrs)
    return _to_snake_case(str(raw_name)), dtypes, " ".join(parts).lower()


def _explicit_axes_from_spec(spec: Mapping[str, object]) -> TopologyAxes | None:
    raw = spec.get("topology_axes")
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping) and all(k in raw for k in ("compute_engine", "memory_ring", "iteration_regularity")):
        return TopologyAxes.from_dict(raw)
    return None


def route_from_op_spec(
    proto_or_advisory: Mapping[str, object], *, case_ratios: CasePhysicsRatios | None = None,
    pipe_gaps: PipeGapReport | None = None, router_enabled: bool = True,
) -> RouteDecision:
    """Route a cann-bench proto.yaml or AdvisoryEnvelope dict to 3-axis TopologyAxes & 3 tracks."""
    _op_name, dtypes, desc_text = _extract_op_name_and_dtypes(proto_or_advisory)
    _, _, _, quant = _scan_dtypes_quant(dtypes, has_quant_op=any(t in desc_text for t in ("quant", "antiquant", "scaleout", "pertokenquant", "qweight", "mxscale")))
    axes = _explicit_axes_from_spec(proto_or_advisory) or resolve_canonical_topology_axes(proto_or_advisory)
    semantic_features: dict[str, object] = {
        "static_affine_bounds": axes.iteration_regularity == "affine_simd",
        "cube_to_vector_ratio": 1.0 if axes.uses_cube else 0.0, "quant_granularity": quant,
        "indirect_sparse_gather": axes.iteration_regularity == "irregular_simt",
        "cross_iter_recurrence": "non_associative_scan" if axes.memory_ring == "persistent_onchip_state" else ("associative_online_rescale" if axes.memory_ring == "bidirectional_cv_ring" else "none"),
        "dynamic_cross_core_handoff": axes.memory_ring in ("l0c_ub_epilogue", "bidirectional_cv_ring"),
        "inner_contiguous_bytes": int(proto_or_advisory.get("inner_contiguous_bytes", 0) or 0),
        "persistent_wave_occupancy": float(proto_or_advisory.get("wave_occupancy", 0.0) or 0.0),
    }
    return CapabilityGate.route(axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios, router_enabled=router_enabled, semantic_features=semantic_features)


def _is_subbyte_dtype(dtype_str: str) -> bool:
    s = dtype_str.strip().lower()
    return s in ("i4", "u4") or any(tok in s for tok in _SUBBYTE_DTYPE_TOKENS)


def route_initial_by_input_contract(
    spec_or_axes: TopologyAxes | Mapping[str, object] | object, *,
    case_ratios: CasePhysicsRatios | None = None, pipe_gaps: PipeGapReport | None = None,
) -> RouteDecision:
    """Causal t=0 router: escalate to ascendc_template ONLY for DSL-unrepresentable contracts."""
    subbyte_reasons: list[str] = []
    gm_cas_reasons: list[str] = []
    opcode_reasons: list[str] = []

    if isinstance(spec_or_axes, TopologyAxes):
        axes = case_ratios.effective_axes(spec_or_axes) if case_ratios is not None else spec_or_axes
        if axes.iteration_regularity == "dynamic_opcode_stream":
            opcode_reasons.append("input_contract: dynamic bytecode/opcode stream required")
        routed = CapabilityGate.route(axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios)
        features = dict(routed.features) if routed.features is not None else {}
    else:
        if isinstance(spec_or_axes, Mapping):
            spec_map: Mapping[str, object] = spec_or_axes
        elif callable(getattr(spec_or_axes, "to_dict", None)):
            if not isinstance(raw_dict := spec_or_axes.to_dict(), Mapping):
                raise TypeError("to_dict() must return a Mapping")
            spec_map = raw_dict
        else:
            raise TypeError(f"Expected TopologyAxes, Mapping, or SIL object, got {type(spec_or_axes).__name__}")

        _op_name, dtypes, _ = _extract_op_name_and_dtypes(spec_map)
        op_block = spec_map.get("operator")
        source: Mapping[str, object] = op_block if isinstance(op_block, Mapping) else spec_map
        contract_text = " ".join(str(src.get(k, "")) for src in (spec_map, source) for k in ("semantic_contract", "formula", "description", "atomics", "op_kind")).lower()

        subbyte_hits = [dt for dt in dtypes if _is_subbyte_dtype(dt)]
        qmode = str(spec_map.get("quant_mode", source.get("quant_mode", ""))).strip().lower()
        if subbyte_hits:
            subbyte_reasons.append(f"input_contract: sub-byte packed dtypes {tuple(subbyte_hits)}")
        elif _is_subbyte_dtype(qmode):
            subbyte_reasons.append(f"input_contract: sub-byte quant_mode={qmode!r}")

        cas_keys = ("requires_inplace_gm_cas", "inplace_gm_cas", "has_gm_cas_atomic", "gm_cas_atomic", "atomic_cas")
        if any(bool(src.get(k, False)) for src in (spec_map, source) for k in cas_keys) or any(
            t in contract_text for t in ("gm_atomic_cas", "asc_atomic_cas", "in-place gm cas", "inplace gm cas", "gm cas", "compare_and_swap")
        ):
            gm_cas_reasons.append("input_contract: in-place GM CAS atomic required")

        raw_axes = spec_map.get("topology_axes")
        axis_reg = str(raw_axes.get("iteration_regularity", "")) if isinstance(raw_axes, Mapping) else str(getattr(raw_axes, "iteration_regularity", ""))
        reg_val = str(spec_map.get("iteration_regularity", axis_reg)).strip().lower()
        op_keys = ("dynamic_bytecode", "dynamic_opcode_stream", "has_opcode_stream", "opcode_stream", "bytecode_interpreter")
        if reg_val == "dynamic_opcode_stream" or any(bool(src.get(k, False)) for src in (spec_map, source) for k in op_keys) or any(t in contract_text for t in ("bytecode", "opcode_stream", "opcode stream")):
            opcode_reasons.append("input_contract: dynamic bytecode/opcode stream required")

        _, _, _, quant = _scan_dtypes_quant(dtypes, has_quant_op=any(t in contract_text for t in ("quant", "antiquant")))
        if quant == "mixed_fp8_fp4" and not subbyte_reasons:
            subbyte_reasons.append("input_contract: sub-byte FP4 mixed_fp8_fp4 dtype")

        axes = _explicit_axes_from_spec(spec_map) or resolve_canonical_topology_axes(spec_map)
        if case_ratios is not None:
            axes = case_ratios.effective_axes(axes)
        if opcode_reasons and axes.iteration_regularity != "dynamic_opcode_stream":
            axes = TopologyAxes(axes.compute_engine, "persistent_onchip_state", "dynamic_opcode_stream")

        sem_feats: dict[str, object] = {
            "static_affine_bounds": reg_val in ("", "affine_simd"), "cube_to_vector_ratio": 1.0 if axes.uses_cube else 0.0,
            "quant_granularity": quant, "indirect_sparse_gather": bool(spec_map.get("has_indirect_gather", reg_val == "irregular_simt")),
            "cross_iter_recurrence": "non_associative_scan" if bool(spec_map.get("has_persistent_state", False)) else "none",
            "dynamic_cross_core_handoff": bool(spec_map.get("has_dead_l1_slice", False)),
            "inner_contiguous_bytes": int(spec_map.get("inner_contiguous_bytes", 0) or 0),
            "persistent_wave_occupancy": float(spec_map.get("wave_occupancy", 0.0) or 0.0),
            "requires_subbyte_dtype": bool(subbyte_reasons), "requires_inplace_gm_cas": bool(gm_cas_reasons),
        }
        routed = CapabilityGate.route(axes, pipe_gaps=pipe_gaps, case_ratios=case_ratios, semantic_features=sem_feats)
        features = dict(routed.features) if routed.features is not None else sem_feats

    hard_walls = (*subbyte_reasons, *gm_cas_reasons, *opcode_reasons)
    return RouteDecision(
        track=routed.track, topology_axes=axes,
        rationale=hard_walls if hard_walls else ("t=0 input contract representable in TileLang DSL; starting on tilelang_dsl without hindsight bias",),
        recommended_primitives=routed.recommended_primitives,
        recommended_backend_tier="ascendc_template" if hard_walls else "tilelang_dsl",
        case_ratios=case_ratios, pipe_gaps=pipe_gaps, suitability_score=routed.suitability_score, features=features,
    )


def apply_case_ratios(decision: RouteDecision, ratios: CasePhysicsRatios) -> RouteDecision:
    """Project real per-case physics ratios onto a RouteDecision."""
    routed = CapabilityGate.route(decision.topology_axes, pipe_gaps=decision.pipe_gaps, case_ratios=ratios, semantic_features=decision.features)
    return RouteDecision(
        track=routed.track, topology_axes=routed.topology_axes,
        rationale=(*decision.rationale, f"case ratios contig_bytes={ratios.inner_contiguous_bytes}B wave_occ={ratios.wave_occupancy_ratio:.3f} -> {routed.topology_axes.key()}"),
        recommended_primitives=routed.recommended_primitives,
        recommended_backend_tier="ascendc_template" if decision.recommended_backend_tier == "ascendc_template" else routed.recommended_backend_tier,
        case_ratios=ratios, pipe_gaps=decision.pipe_gaps, suitability_score=routed.suitability_score, features=routed.features,
    )


def _build_escalated_mask(
    base_mask: ConstraintRelaxationMask, *, dominant_bound: str, compiler_summary: object | None,
    sf_map: Mapping[str, object], stagnation_rounds: int, roofline_eta: float, target_eta: float,
    unlock_sram: bool = False, unlock_sync: bool = False, unlock_dma: bool = False,
) -> ConstraintRelaxationMask:
    from tilefoundry_ascend.ascend.templates.registry import (  # noqa: PLC0415
        relax_facets_for_bottleneck,
    )

    mask = relax_facets_for_bottleneck(
        base_mask, dominant_bound=dominant_bound, compiler_summary=compiler_summary,
        l1_overflow=bool(sf_map.get("l1_overflow", False)), ub_overflow=bool(sf_map.get("ub_overflow", False)),
    )
    extra = [f for cond, f in ((unlock_sram, "sram_layout_and_aliasing"), (unlock_sync, "pipeline_and_sync_graph"), (unlock_dma, "dma_and_cache_hierarchy")) if cond]
    if extra:
        mask = mask.unlock(*extra)
    return ConstraintRelaxationMask.top() if (stagnation_rounds >= 4 and roofline_eta < min(0.65, float(target_eta))) else mask


def evaluate_track_escalation(
    *, current_tier: str, roofline_eta: float, target_eta: float, stagnation_rounds: int,
    dominant_bound: str = "", compiler_summary: object | None = None,
    structural_features: Mapping[str, object] | None = None,
) -> EscalationDecision:
    """Closed-loop t>0 escalation from tilelang_dsl to inline_ascendc_vf or ascendc_template."""
    tier = normalize_router_tier(current_tier)
    if tier not in VALID_ESCALATION_TIERS:
        raise ValueError(f"Unsupported current_tier {current_tier!r}; expected one of {VALID_ESCALATION_TIERS}")
    if roofline_eta >= target_eta:
        return EscalationDecision(False, tier, tier, trigger_reasons=("roofline_target_met",))

    sf_map = _as_dict(structural_features)
    if tier == "ascendc_template":
        raw_mask = sf_map.get("freedom_mask")
        base_mask = ConstraintRelaxationMask.from_dict(raw_mask) if isinstance(raw_mask, Mapping) else (raw_mask if isinstance(raw_mask, ConstraintRelaxationMask) else ConstraintRelaxationMask.bottom())
        custom = bool(sf_map.get("has_dead_l1_slice", False) or sf_map.get("requires_custom_ring_or_aliasing", False))
        handoff = bool(sf_map.get("dynamic_cross_core_handoff", False))
        return EscalationDecision(
            False, tier, tier, trigger_reasons=("already_at_ascendc_template",),
            freedom_mask=_build_escalated_mask(
                base_mask, dominant_bound=dominant_bound, compiler_summary=compiler_summary, sf_map=sf_map,
                stagnation_rounds=stagnation_rounds, roofline_eta=roofline_eta, target_eta=target_eta,
                unlock_sram=custom, unlock_sync=custom or handoff,
            ),
        )

    if stagnation_rounds < 2:
        return EscalationDecision(False, tier, tier, trigger_reasons=(f"stagnation_rounds={stagnation_rounds}<2; keep iterating on {tier}",))

    comp = _as_dict(compiler_summary)
    if compiler_summary is not None and not comp:
        comp = {
            k: getattr(compiler_summary, k, dflt) for k, dflt in (
                ("spill_bytes", 0), ("vloopv2_loops", 0), ("software_loops", 0),
                ("has_local_membar_in_vf_call", False), ("vf_fusion_breaks", 0),
                ("alias_serialized_buffers", 0), ("dual_issue_ipc_collapse", 0.0), ("dcache_dcci_flushes", 0),
            )
        }

    vf_breaks, alias_ser = int(comp.get("vf_fusion_breaks", 0) or 0), int(comp.get("alias_serialized_buffers", 0) or 0)
    local_membar, sw_loops = bool(comp.get("has_local_membar_in_vf_call", False)), int(comp.get("software_loops", 0) or 0)
    vloopv2, ipc_collapse = int(comp.get("vloopv2_loops", 0) or 0), float(comp.get("dual_issue_ipc_collapse", 0.0) or 0.0)
    dcci_flushes, spill_bytes = int(comp.get("dcache_dcci_flushes", 0) or 0), int(comp.get("spill_bytes", 0) or 0)

    uses_cube = bool(sf_map.get("uses_cube", False)) or float(sf_map.get("cube_to_vector_ratio", 0.0) or 0.0) > 0.0
    cross_core = bool(sf_map.get("dynamic_cross_core_handoff", False) or sf_map.get("cross_core_mode4_blocked", False))
    dead_l1 = bool(sf_map.get("has_dead_l1_slice", False) or sf_map.get("dead_l1_reuse_blocked", False))
    indirect = bool(sf_map.get("indirect_sparse_gather", False))
    dist_wall = bool(sf_map.get("requires_dist_unpack", False) or sf_map.get("ub_load_store_bottleneck", False))
    dom = dominant_bound.strip().lower()

    reasons: list[str] = []
    template_wall = vf_wall = False
    for cond, msg in (
        (cross_core or dead_l1, "cross_core_mode4_or_dead_l1_reuse_blocked_by_dsl"),
        (alias_ser >= 2 or (alias_ser >= 1 and uses_cube), f"alias_serialized_buffers={alias_ser}_blocks_ping_pong_ring"),
        (dcci_flushes > 0 and indirect, f"dcache_dcci_flushes={dcci_flushes}_with_indirect_gather"),
        (tier == "inline_ascendc_vf", "inline_ascendc_vf_stagnated"),
    ):
        if cond:
            template_wall = True
            reasons.append(msg)

    for cond, msg in (
        (vf_breaks > 0, f"vf_fusion_breaks={vf_breaks}_ub_roundtrip"),
        (local_membar, "local_membar_in_vf_call"),
        (sw_loops > 0 and vloopv2 == 0, f"vloopv2_missed_software_loops={sw_loops}"),
        (ipc_collapse >= 0.4, f"dual_issue_ipc_collapse={ipc_collapse:.2f}"),
        (alias_ser > 0 and not uses_cube, f"vector_alias_serialized_buffers={alias_ser}"),
        (dcci_flushes > 0 and not indirect, f"dcache_dcci_flushes={dcci_flushes}"),
        (spill_bytes > 0, f"compiler_spill_bytes={spill_bytes}"),
        (dist_wall or any(t in dom for t in ("ub", "vector", "regbase", "spill")), f"ub_vector_distribution_bottleneck({dom or 'dist'})"),
    ):
        if cond:
            vf_wall = True
            reasons.append(msg)

    if template_wall:
        return EscalationDecision(
            True, tier, "ascendc_template", trigger_reasons=tuple(reasons),
            freedom_mask=_build_escalated_mask(
                ConstraintRelaxationMask.bottom(), dominant_bound=dominant_bound, compiler_summary=comp,
                sf_map=sf_map, stagnation_rounds=stagnation_rounds, roofline_eta=roofline_eta, target_eta=target_eta,
                unlock_sram=dead_l1 or alias_ser > 0, unlock_sync=cross_core or alias_ser > 0 or dcci_flushes > 0,
                unlock_dma=dcci_flushes > 0,
            ),
            suggested_primitives=("CrossCoreSetFlag<mode=4>", "L1_slot_aliasing", "HalfL1Isolation"),
        )
    if vf_wall and tier == "tilelang_dsl":
        return EscalationDecision(
            True, tier, "inline_ascendc_vf", trigger_reasons=tuple(reasons),
            freedom_mask=ConstraintRelaxationMask.bottom(),
            suggested_primitives=("LoadDist::DIST_UNPACK_B16", "StoreDist::DIST_PACK_B32", "Reg::ReduceDataBlock", "__no_simd_vf_fusion__ __simd_vf__"),
        )
    return EscalationDecision(False, tier, tier, trigger_reasons=("no_track_a_lowering_wall_detected",))


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
