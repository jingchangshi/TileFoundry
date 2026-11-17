"""Eight-dimensional structural feature extractor and track router for Ascend kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    ExecutionScope,
    For,
    If,
    MeshScope,
    Sequential,
    Stmt,
    While,
)
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry_ascend.ascend.ir.execution import SIMTVF_KIND
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
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


@dataclass(frozen=True)
class StructuralFeatures:
    """Normalized 8-dimensional structural kernel feature vector phi_1..phi_8."""

    phi_1_static_affine_bounds: bool
    phi_2_cube_to_vector_ratio: float
    phi_3_quant_granularity: str
    phi_4_indirect_sparse_gather: bool
    phi_5_cross_iter_recurrence: str
    phi_6_dynamic_cross_core_handoff: bool
    phi_7_inner_contiguous_bytes: int
    phi_8_persistent_wave_occupancy: float

    def __post_init__(self) -> None:
        if self.phi_3_quant_granularity not in VALID_QUANT_GRANULARITIES:
            raise ValueError(
                f"invalid quant granularity {self.phi_3_quant_granularity!r}"
            )
        if self.phi_5_cross_iter_recurrence not in VALID_RECURRENCES:
            raise ValueError(
                f"invalid recurrence kind {self.phi_5_cross_iter_recurrence!r}"
            )

    def as_tuple(self) -> tuple[bool, float, str, bool, str, bool, int, float]:
        """Return ordered tuple (phi_1, ..., phi_8)."""
        return (
            self.phi_1_static_affine_bounds,
            self.phi_2_cube_to_vector_ratio,
            self.phi_3_quant_granularity,
            self.phi_4_indirect_sparse_gather,
            self.phi_5_cross_iter_recurrence,
            self.phi_6_dynamic_cross_core_handoff,
            self.phi_7_inner_contiguous_bytes,
            self.phi_8_persistent_wave_occupancy,
        )

    def to_dict(self) -> dict[str, object]:
        """Return feature values keyed by phi_1..phi_8 and symbolic names."""
        return {
            "phi_1": self.phi_1_static_affine_bounds,
            "phi_2": self.phi_2_cube_to_vector_ratio,
            "phi_3": self.phi_3_quant_granularity,
            "phi_4": self.phi_4_indirect_sparse_gather,
            "phi_5": self.phi_5_cross_iter_recurrence,
            "phi_6": self.phi_6_dynamic_cross_core_handoff,
            "phi_7": self.phi_7_inner_contiguous_bytes,
            "phi_8": self.phi_8_persistent_wave_occupancy,
            "static_affine_bounds": self.phi_1_static_affine_bounds,
            "cube_to_vector_ratio": self.phi_2_cube_to_vector_ratio,
            "quant_granularity": self.phi_3_quant_granularity,
            "indirect_sparse_gather": self.phi_4_indirect_sparse_gather,
            "cross_iter_recurrence": self.phi_5_cross_iter_recurrence,
            "dynamic_cross_core_handoff": self.phi_6_dynamic_cross_core_handoff,
            "inner_contiguous_bytes": self.phi_7_inner_contiguous_bytes,
            "persistent_wave_occupancy": self.phi_8_persistent_wave_occupancy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StructuralFeatures:
        """Reconstruct StructuralFeatures from phi_1..phi_8 or symbolic dictionary keys."""

        def _pick(short_key: str, long_key: str, default: object) -> object:
            if short_key in payload:
                return payload[short_key]
            if long_key in payload:
                return payload[long_key]
            full_key = f"{short_key}_{long_key}"
            if full_key in payload:
                return payload[full_key]
            return default

        return cls(
            phi_1_static_affine_bounds=bool(_pick("phi_1", "static_affine_bounds", True)),
            phi_2_cube_to_vector_ratio=float(_pick("phi_2", "cube_to_vector_ratio", 0.0)),
            phi_3_quant_granularity=str(_pick("phi_3", "quant_granularity", "none")),
            phi_4_indirect_sparse_gather=bool(_pick("phi_4", "indirect_sparse_gather", False)),
            phi_5_cross_iter_recurrence=str(_pick("phi_5", "cross_iter_recurrence", "none")),
            phi_6_dynamic_cross_core_handoff=bool(
                _pick("phi_6", "dynamic_cross_core_handoff", False)
            ),
            phi_7_inner_contiguous_bytes=int(_pick("phi_7", "inner_contiguous_bytes", 0)),
            phi_8_persistent_wave_occupancy=float(
                _pick("phi_8", "persistent_wave_occupancy", 0.0)
            ),
        )


@dataclass(frozen=True)
class RouteDecision:
    """Selected DSL lowering track alongside rationale and recommended primitives."""

    track: str
    features: StructuralFeatures
    rationale: tuple[str, ...]
    recommended_primitives: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.track not in VALID_TRACKS:
            raise ValueError(f"unsupported route track {self.track!r}")


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


def _scan_tilelang_features(
    func,
) -> dict[str, object]:
    """Scan a TVM/TileLang PrimFunc for static bounds, quant, indirect gather, and sync."""
    from tilelang.utils.language import DataType as TLDataType
    from tvm import tirx

    static_bounds = [True]
    indirect_gather = [False]
    dynamic_handoff = [False]
    has_fp8 = [False]
    has_fp4 = [False]
    has_ssbuf = [False]
    has_cube = [False]
    has_vec = [False]
    has_online_exp = [False]
    has_simt_vf = [False]
    min_inner_bytes: list[int] = []
    grid_tiles: list[int] = []
    indirect_vars: list[object] = []

    def _is_indirect_var(v) -> bool:
        return any(hasattr(v, "same_as") and v.same_as(iv) for iv in indirect_vars)

    def _check_dtype(dt_obj) -> None:
        s = str(dt_obj).lower()
        if "float4" in s or "fp4" in s or "e2m1" in s:
            has_fp4[0] = True
        elif "float8" in s or "fp8" in s or "e4m3" in s or "e5m2" in s:
            has_fp8[0] = True

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
            elif not (hasattr(ext, "value") and isinstance(ext.value, int)):
                static_bounds[0] = False
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
            if "gather" in op_name:
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
    if has_fp4[0] and has_fp8[0]:
        quant = "mixed_fp8_fp4"
    elif has_fp8[0] or has_fp4[0]:
        quant = "per_token_or_block"
    else:
        quant = "none"
    recurrence = "associative_online_rescale" if has_online_exp[0] else "none"
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
    """Scan a TileFoundry HIR/TIR function for static bounds, quant, and inner width."""
    static_bounds = [True]
    has_simt = [False]
    inner_bytes: list[int] = []
    recurrence = ["none"]
    indirect = [False]
    handoff = [False]

    def _record_type(ttype) -> None:
        if isinstance(ttype, TensorType) and ttype.shape:
            dim = static_dim_value(ttype.shape[-1])
            if dim is not None and dim > 0:
                eb = max(1, (ttype.dtype.bit_width + 7) // 8)
                inner_bytes.append(dim * eb)

    if isinstance(fn_or_stmt, (PrimFunction, HirFunction)):
        for param in fn_or_stmt.params:
            _record_type(getattr(param, "type", None))
        body = fn_or_stmt.body
    else:
        body = fn_or_stmt

    def _walk_stmt(stmt: Stmt | None) -> None:
        if stmt is None:
            return
        if isinstance(stmt, Sequential):
            for child in stmt.body:
                _walk_stmt(child)
            return
        if isinstance(stmt, While):
            static_bounds[0] = False
            _walk_stmt(stmt.body)
            return
        if isinstance(stmt, For):
            if static_dim_value(stmt.extent) is None:
                static_bounds[0] = False
            _walk_stmt(stmt.body)
            return
        if isinstance(stmt, If):
            _walk_stmt(stmt.then_body)
            _walk_stmt(stmt.else_body)
            return
        if isinstance(stmt, ExecutionScope):
            if getattr(stmt.kind, "name", stmt.kind) == SIMTVF_KIND.name:
                has_simt[0] = True
            _walk_stmt(stmt.body)
            return
        if isinstance(stmt, MeshScope):
            _walk_stmt(stmt.body)
            return
        if hasattr(stmt, "args"):
            for arg in stmt.args:
                _record_type(getattr(arg, "type", None))

    if isinstance(body, Stmt):
        _walk_stmt(body)
    elif isinstance(body, LoopRegion):
        if static_dim_value(body.extent) is None:
            static_bounds[0] = False

    return {
        "static_bounds": static_bounds[0],
        "indirect_gather": indirect[0],
        "dynamic_handoff": handoff[0],
        "quant": "none",
        "recurrence": recurrence[0],
        "has_simt_vf": has_simt[0],
        "has_cube": False,
        "has_vec": True,
        "inner_bytes": min(inner_bytes) if inner_bytes else 0,
        "total_tiles": 0,
    }


def extract_structural_features(
    ir_obj: object | None = None,
    *,
    ledger: ObjectivePhysicalLedger | None = None,
    target: AscendTarget | None = None,
    total_tiles: int | None = None,
    quant_granularity: str | None = None,
    cross_iter_recurrence: str | None = None,
    indirect_sparse_gather: bool | None = None,
    dynamic_cross_core_handoff: bool | None = None,
    inner_contiguous_bytes: int | None = None,
) -> StructuralFeatures:
    """Compute the 8-dimensional structural vector phi_1..phi_8 from IR and ledger."""
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
        if isinstance(ir_obj, (PrimFunction, HirFunction, Stmt, LoopRegion)):
            info = _scan_tilefoundry_ir(ir_obj)
        else:
            from tvm import IRModule, tirx

            target_fn = None
            if isinstance(ir_obj, IRModule):
                fns = [f for f in ir_obj.functions.values() if isinstance(f, tirx.PrimFunc)]
                target_fn = fns[0] if fns else None
            elif isinstance(ir_obj, tirx.PrimFunc):
                target_fn = ir_obj
            if target_fn is not None:
                info = _scan_tilelang_features(target_fn)

    static_bounds = bool(info["static_bounds"])
    if ledger is not None:
        dynamic_tags = {"while_loop", "data_dependent_branch"}
        if any(
            tag in dynamic_tags or tag.startswith("dynamic_for:")
            for tag in ledger.opaque_dynamic_regions
        ):
            static_bounds = False

    ratio = _ratio_from_ledger(ledger)
    if ratio == 0.0:
        if info["has_cube"] and info["has_vec"]:
            ratio = 1.0
        elif info["has_cube"]:
            ratio = 2.0

    quant = quant_granularity or str(info["quant"])
    indirect = (
        indirect_sparse_gather
        if indirect_sparse_gather is not None
        else bool(info["indirect_gather"])
    )
    recurrence = cross_iter_recurrence or str(info["recurrence"])
    handoff = (
        dynamic_cross_core_handoff
        if dynamic_cross_core_handoff is not None
        else bool(info["dynamic_handoff"])
    )
    inner_b = (
        inner_contiguous_bytes
        if inner_contiguous_bytes is not None
        else int(info["inner_bytes"])
    )
    tiles = total_tiles if total_tiles is not None else int(info["total_tiles"])
    occupancy = float(tiles) / float(aic_count) if tiles > 0 else 0.0

    return StructuralFeatures(
        phi_1_static_affine_bounds=static_bounds,
        phi_2_cube_to_vector_ratio=ratio,
        phi_3_quant_granularity=quant,
        phi_4_indirect_sparse_gather=indirect,
        phi_5_cross_iter_recurrence=recurrence,
        phi_6_dynamic_cross_core_handoff=handoff,
        phi_7_inner_contiguous_bytes=inner_b,
        phi_8_persistent_wave_occupancy=occupancy,
    )


def route_execution_track(
    features: StructuralFeatures,
    *,
    simt_dominant: bool = False,
) -> RouteDecision:
    """Select Ascend execution track (developer, simt_vf, or manual_ssbuf)."""
    reasons: list[str] = []
    if features.phi_4_indirect_sparse_gather or features.phi_6_dynamic_cross_core_handoff:
        if features.phi_4_indirect_sparse_gather:
            reasons.append("phi_4 indirect sparse/page gather detected")
        if features.phi_6_dynamic_cross_core_handoff:
            reasons.append("phi_6 dynamic cross-core metadata/scale handoff detected")
        return RouteDecision(
            track="manual_ssbuf",
            features=features,
            rationale=tuple(reasons),
            recommended_primitives=("T.Scope", "T.alloc_ss", "T.dual_copy", "T.Persistent"),
        )

    if not features.phi_1_static_affine_bounds and features.phi_2_cube_to_vector_ratio > 0.0:
        reasons.append("dynamic bounds paired with coupled Cube/Vector execution")
        return RouteDecision(
            track="manual_ssbuf",
            features=features,
            rationale=tuple(reasons),
            recommended_primitives=("T.Scope", "T.alloc_ss", "T.dual_copy"),
        )

    is_non_assoc = features.phi_5_cross_iter_recurrence == "non_associative_scan"
    unaligned_vec = (
        0 < features.phi_7_inner_contiguous_bytes < 32
        and features.phi_2_cube_to_vector_ratio == 0.0
    )
    if simt_dominant or is_non_assoc or unaligned_vec or not features.phi_1_static_affine_bounds:
        if simt_dominant:
            reasons.append("fine-grained thread-level vector reduction flagged")
        if is_non_assoc:
            reasons.append("phi_5 non-associative cross-iteration recurrence")
        if unaligned_vec:
            reasons.append("phi_7 sub-32B innermost contiguous vector width")
        if not features.phi_1_static_affine_bounds:
            reasons.append("phi_1 non-affine bounds on pure vector loop")
        return RouteDecision(
            track="simt_vf",
            features=features,
            rationale=tuple(reasons),
            recommended_primitives=("T.SimtVF", "T.get_thread_binding", "T.Persistent"),
        )

    reasons.append("phi_1 affine static bounds with no indirect gather or dynamic handoff")
    return RouteDecision(
        track="developer",
        features=features,
        rationale=tuple(reasons),
        recommended_primitives=("T.Pipelined", "T.Stage", "T.SimdVF", "T.dual_copy"),
    )


__all__ = [
    "RouteDecision",
    "StructuralFeatures",
    "VALID_QUANT_GRANULARITIES",
    "VALID_RECURRENCES",
    "VALID_TRACKS",
    "extract_structural_features",
    "route_execution_track",
]
